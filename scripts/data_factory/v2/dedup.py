"""Exact deduplication, DataTrove MinHash orchestration, and decontamination."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import ParquetWriter
from datatrove.utils.word_tokenizers import WordTokenizer

from scripts.data_factory.v2.benchmarks import read_benchmark_rows
from scripts.data_factory.v2.config import DataFactoryConfig
from scripts.data_factory.v2.runtime import DistributionStats, TaskOutputGuard, build_executor

PROFILE_NAMES = ("zh", "en", "code")


def minhash_dimensions(profile: dict[str, Any]) -> tuple[int, int]:
    buckets = int(profile["num_buckets"])
    threshold = float(profile["threshold"])
    if buckets < 2 or not 0 < threshold < 1:
        raise ValueError("MinHash requires num_buckets >= 2 and 0 < threshold < 1")
    # DataTrove LSH knee: (1 / buckets) ** (1 / hashes_per_bucket).
    hashes = max(1, round(math.log(1 / buckets) / math.log(threshold)))
    return buckets, hashes


def _candidate_dir(config: DataFactoryConfig, plan: dict[str, Any]) -> Path:
    return config.run_root / "candidates" / str(plan["plan_sha256"])[:16]


def _parquet_files(root: Path) -> list[Path]:
    files = sorted(path for path in root.rglob("*.parquet") if path.is_file()) if root.is_dir() else []
    if not files:
        raise FileNotFoundError(f"no Parquet files under {root}")
    return files


def _normalized_hash_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def content_sha256(text: str) -> str:
    return hashlib.sha256(_normalized_hash_text(text).encode("utf-8")).hexdigest()


def _minhash_profile(metadata: dict[str, Any]) -> str:
    domain = str(metadata.get("domain") or "")
    language = str(metadata.get("language") or "")
    if domain in {"code", "structured"}:
        return "code"
    if language in {"zh", "zh_en_mixed"}:
        return "zh"
    return "en"


def _document_schema(extra_fields: list[tuple[str, Any]] | None = None):
    import pyarrow as pa

    fields = [
        ("id", pa.string()),
        ("text", pa.large_string()),
        ("parent_doc_id", pa.string()),
        ("source", pa.string()),
        ("subset", pa.string()),
        ("source_path", pa.string()),
        ("revision", pa.string()),
        ("license", pa.string()),
        ("url", pa.string()),
        ("language", pa.string()),
        ("domain", pa.string()),
        ("char_count", pa.int64()),
        ("hanzi_count", pa.int64()),
        ("latin_count", pa.int64()),
        ("digit_count", pa.int64()),
        ("quality_prior", pa.float64()),
        ("tags", pa.list_(pa.string())),
        ("metadata_json", pa.large_string()),
        ("candidate_bucket", pa.string()),
        ("estimated_tokens", pa.int64()),
        ("sample_key", pa.uint64()),
        ("plan_sha256", pa.string()),
    ]
    return pa.schema(fields + (extra_fields or []))


def _dedup_document_schema():
    import pyarrow as pa

    return _document_schema(
        [("content_sha256", pa.string()), ("minhash_profile", pa.string())]
    )


def _executor_kwargs(options: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in options.items() if key != "tasks"}


class ParquetDocumentReader(PipelineStep):
    name = "Parquet document reader"
    type = "Reader"

    def __init__(self, files: list[Path]) -> None:
        super().__init__()
        self.files = tuple(str(path) for path in files)

    def run(
        self,
        data: DocumentsPipeline = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[Document]:
        del data
        import pyarrow.parquet as pq

        for path_value in self.files[rank::world_size]:
            self.stat_update("input_files")
            parquet = pq.ParquetFile(path_value)
            for batch in parquet.iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    metadata = {key: value for key, value in row.items() if key not in {"id", "text"}}
                    self.stat_update("documents")
                    yield Document(id=str(row["id"]), text=str(row["text"]), metadata=metadata)


class ExactSignature(PipelineStep):
    name = "SHA-256 signature"
    type = "Dedup"

    def __init__(self, source_priority: dict[str, int]) -> None:
        super().__init__()
        self.source_priority = source_priority

    def run(
        self,
        data: DocumentsPipeline,
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[Document]:
        del rank, world_size
        for document in data:
            digest = content_sha256(document.text)
            quality = document.metadata.get("quality_prior")
            document.metadata.update(
                {
                    "content_sha256": digest,
                    "hash_prefix": digest[:2],
                    "source_priority": self.source_priority.get(
                        str(document.metadata.get("source")), 1_000_000
                    ),
                    "dedup_quality": float(quality) if isinstance(quality, (int, float)) else -1.0e30,
                }
            )
            self.stat_update("documents")
            yield document


def _signature_adapter(_writer, document: Document) -> dict[str, Any]:
    return {
        "content_sha256": document.metadata["content_sha256"],
        "doc_id": document.id,
        "source_priority": document.metadata["source_priority"],
        "dedup_quality": document.metadata["dedup_quality"],
        "char_count": document.metadata["char_count"],
    }


def _signature_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("content_sha256", pa.string()),
            ("doc_id", pa.string()),
            ("source_priority", pa.int64()),
            ("dedup_quality", pa.float64()),
            ("char_count", pa.int64()),
        ]
    )


class ExactCluster(PipelineStep):
    name = "SHA-256 representative selection"
    type = "Dedup"

    def __init__(self, signatures_dir: Path, removals_dir: Path) -> None:
        super().__init__()
        self.signatures_dir = signatures_dir
        self.removals_dir = removals_dir

    def run(
        self,
        data: DocumentsPipeline = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[Document]:
        del data
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.removals_dir.mkdir(parents=True, exist_ok=True)
        removal_schema = pa.schema(
            [
                ("doc_id", pa.string()),
                ("content_sha256", pa.string()),
                ("kept_doc_id", pa.string()),
            ]
        )
        for prefix_value in range(rank, 256, world_size):
            prefix = f"{prefix_value:02x}"
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for path in sorted((self.signatures_dir / prefix).glob("*.parquet")):
                table = pq.read_table(path)
                for row in table.to_pylist():
                    grouped[str(row["content_sha256"])].append(row)
            removed: list[dict[str, str]] = []
            for digest, rows in grouped.items():
                if len(rows) < 2:
                    continue
                representative = min(
                    rows,
                    key=lambda row: (
                        int(row["source_priority"]),
                        -float(row["dedup_quality"]),
                        -int(row["char_count"]),
                        str(row["doc_id"]),
                    ),
                )
                for row in rows:
                    if row["doc_id"] != representative["doc_id"]:
                        removed.append(
                            {
                                "doc_id": str(row["doc_id"]),
                                "content_sha256": digest,
                                "kept_doc_id": str(representative["doc_id"]),
                            }
                        )
            pq.write_table(
                pa.Table.from_pylist(removed, schema=removal_schema),
                self.removals_dir / f"{prefix}.parquet",
                compression="zstd",
            )
            self.stat_update("hash_prefixes")
            self.stat_update("removed_documents", value=len(removed), unit="prefix")
        return
        yield  # pragma: no cover


class ExactRemoval(PipelineStep):
    name = "SHA-256 duplicate filter"
    type = "Dedup"

    def __init__(self, removals_dir: Path) -> None:
        super().__init__()
        self.removals_dir = removals_dir
        self._cache: dict[str, set[str]] = {}

    def _removed_ids(self, prefix: str) -> set[str]:
        if prefix not in self._cache:
            import pyarrow.parquet as pq

            path = self.removals_dir / f"{prefix}.parquet"
            self._cache[prefix] = (
                set(pq.read_table(path, columns=["doc_id"])["doc_id"].to_pylist())
                if path.is_file()
                else set()
            )
        return self._cache[prefix]

    def run(
        self,
        data: DocumentsPipeline,
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[Document]:
        del rank, world_size
        for document in data:
            digest = content_sha256(document.text)
            if document.id in self._removed_ids(digest[:2]):
                self.stat_update("removed")
                continue
            document.metadata["content_sha256"] = digest
            document.metadata["minhash_profile"] = _minhash_profile(document.metadata)
            self.stat_update("forwarded")
            yield document


def _source_priority(config: DataFactoryConfig) -> dict[str, int]:
    quality_order = {
        "curated_zh": 0,
        "curated_en": 0,
        "classical": 1,
        "scientific": 1,
        "math": 1,
        "code": 1,
        "web_zh": 2,
        "web_multilingual": 2,
        "broad_zh": 3,
    }
    return {
        name: quality_order.get(source.quality_profile, 100)
        for name, source in config.source_registry.sources.items()
    }


def run_exact_dedup(
    config: DataFactoryConfig,
    plan: dict[str, Any],
    **executor_options: Any,
) -> dict[str, Any]:
    hashes = [*plan.get("previous_plan_hashes", []), plan["plan_sha256"]]
    input_files = sorted({
        path for digest in hashes
        for path in _parquet_files(config.run_root / "candidates" / digest[:16])
    })
    root = config.run_root / "dedup" / "exact" / str(plan["plan_sha256"])[:16]
    signatures = root / "signatures"
    removals = root / "remove_ids"
    documents = root / "documents"
    tasks = min(executor_options.get("tasks") or len(input_files), len(input_files))

    signature_pipeline = [
        ParquetDocumentReader(input_files),
        ExactSignature(_source_priority(config)),
        TaskOutputGuard(signatures),
        ParquetWriter(
            output_folder=str(signatures),
            output_filename="${hash_prefix}/${rank}.parquet",
            compression="zstd",
            adapter=_signature_adapter,
            schema=_signature_schema(),
            batch_size=4_000,
            max_file_size=256 * 2**20,
        ),
    ]
    signature_executor = build_executor(
        pipeline=signature_pipeline,
        logging_dir=root / "logs" / "signatures",
        job_name=f"cc_exact_sig_{config.phase}",
        tasks=tasks,
        **_executor_kwargs(executor_options),
    )
    cluster_executor = build_executor(
        pipeline=[ExactCluster(signatures, removals)],
        logging_dir=root / "logs" / "cluster",
        job_name=f"cc_exact_cluster_{config.phase}",
        tasks=min(256, max(1, executor_options["workers"])),
        depends=signature_executor if executor_options["executor"] == "slurm" else None,
        **_executor_kwargs(executor_options),
    )
    filter_pipeline = [
        ParquetDocumentReader(input_files),
        ExactRemoval(removals),
        DistributionStats(),
        TaskOutputGuard(documents),
        ParquetWriter(
            output_folder=str(documents),
            output_filename="${minhash_profile}/${rank}.parquet",
            compression="zstd",
            expand_metadata=True,
            schema=_dedup_document_schema(),

            batch_size=2_000,
            max_file_size=512 * 2**20,
        ),
    ]
    filter_executor = build_executor(
        pipeline=filter_pipeline,
        logging_dir=root / "logs" / "filter",
        job_name=f"cc_exact_filter_{config.phase}",
        tasks=tasks,
        depends=cluster_executor if executor_options["executor"] == "slurm" else None,
        **_executor_kwargs(executor_options),
    )
    if executor_options["executor"] == "local":
        signature_executor.run()
        cluster_executor.run()
    filter_executor.run()
    return {"stage": "exact_dedup", "root": str(root), "tasks": tasks}


class CharacterTokenizer(WordTokenizer):
    def word_tokenize(self, text: str) -> list[str]:
        return [char for char in text if not char.isspace()]

    def sent_tokenize(self, text: str) -> list[str]:
        return [text]

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        return [(0, len(text))]


class CodeTokenizer(CharacterTokenizer):
    TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\s]")

    def word_tokenize(self, text: str) -> list[str]:
        return self.TOKEN_RE.findall(text)


def run_minhash(
    config: DataFactoryConfig,
    plan: dict[str, Any],
    **executor_options: Any,
) -> dict[str, Any]:
    if not config.dedup.minhash_enabled:
        return {"stage": "minhash", "skipped": True, "reason": "exact_only profile"}
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import (
        MinhashConfig,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.utils.hashing import HashConfig
    from datatrove.utils.typeshelper import Languages

    exact_root = config.run_root / "dedup" / "exact" / str(plan["plan_sha256"])[:16] / "documents"
    root = config.run_root / "dedup" / "minhash" / str(plan["plan_sha256"])[:16]
    results: dict[str, Any] = {}
    for profile in PROFILE_NAMES:
        input_dir = exact_root / profile
        files = sorted(input_dir.glob("*.parquet")) if input_dir.is_dir() else []
        if not files:
            results[profile] = {"skipped": True, "reason": "empty profile"}
            continue
        profile_raw = config.dedup.minhash_profiles[profile]
        num_buckets, hashes_per_bucket = minhash_dimensions(profile_raw)
        mh_config = MinhashConfig(
            n_grams=int(profile_raw["ngram"]),
            num_buckets=num_buckets,
            hashes_per_bucket=hashes_per_bucket,
            seed=config.seed,
            hash_config=HashConfig(precision=64),
        )
        tokenizer = CharacterTokenizer() if profile == "zh" else CodeTokenizer() if profile == "code" else Languages.english
        tasks = len(files)
        signatures = root / profile / "signatures"
        buckets = root / profile / "buckets"
        removals = root / profile / "remove_ids"
        output = root / "documents" / profile
        reader = lambda: ParquetReader(data_folder=str(input_dir), glob_pattern="*.parquet")
        stage1 = build_executor(
            pipeline=[reader(), MinhashDedupSignature(str(signatures), config=mh_config, language=tokenizer)],
            logging_dir=root / profile / "logs" / "signatures",
            job_name=f"cc_mh1_{config.phase}_{profile}",
            tasks=tasks,
            **_executor_kwargs(executor_options),
        )
        bucket_tasks = mh_config.num_buckets
        stage2 = build_executor(
            pipeline=[MinhashDedupBuckets(str(signatures), str(buckets), config=mh_config, only_dedup_in_index=False)],
            logging_dir=root / profile / "logs" / "buckets",
            job_name=f"cc_mh2_{config.phase}_{profile}",
            tasks=bucket_tasks,
            depends=stage1 if executor_options["executor"] == "slurm" else None,
            **_executor_kwargs(executor_options),
        )
        stage3 = build_executor(
            pipeline=[MinhashDedupCluster(str(buckets), str(removals), config=mh_config)],
            logging_dir=root / profile / "logs" / "cluster",
            job_name=f"cc_mh3_{config.phase}_{profile}",
            tasks=1,
            depends=stage2 if executor_options["executor"] == "slurm" else None,
            **_executor_kwargs(executor_options),
        )
        stage4 = build_executor(
            pipeline=[
                reader(),
                MinhashDedupFilter(str(removals)),
                DistributionStats(),
                TaskOutputGuard(output),
                ParquetWriter(
                    output_folder=str(output),
                    output_filename="${rank}.parquet",
                    compression="zstd",
                    expand_metadata=True,
                    schema=_dedup_document_schema(),

                    max_file_size=512 * 2**20,
                ),
            ],
            logging_dir=root / profile / "logs" / "filter",
            job_name=f"cc_mh4_{config.phase}_{profile}",
            tasks=tasks,
            depends=stage3 if executor_options["executor"] == "slurm" else None,
            **_executor_kwargs(executor_options),
        )
        if executor_options["executor"] == "local":
            stage1.run()
            stage2.run()
            stage3.run()
        stage4.run()
        results[profile] = {
            "tasks": tasks, "output": str(output),
            "requested_lsh_threshold": profile_raw["threshold"],
            "effective_lsh_threshold": (1 / num_buckets) ** (1 / hashes_per_bucket),
            "num_buckets": num_buckets, "hashes_per_bucket": hashes_per_bucket,
        }
    return {"stage": "minhash", "root": str(root), "profiles": results}


def _flatten_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _flatten_text(item)]
    if isinstance(value, list):
        return [text for item in value for text in _flatten_text(item)]
    return []


def _benchmark_index(registry_path: Path) -> tuple[set[str], set[str], set[tuple[str, ...]]]:
    import glob as glob_module
    import yaml

    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    exact: set[str] = set()
    zh_ngrams: set[str] = set()
    word_ngrams: set[tuple[str, ...]] = set()
    missing: list[str] = []
    for benchmark in raw.get("benchmarks", []):
        benchmark_name = str(benchmark.get("name", "unnamed"))
        fields = tuple(benchmark.get("text_fields", []))
        matched = 0
        for pattern in benchmark.get("paths", []):
            expanded = os.path.expandvars(pattern)
            for path_value in glob_module.glob(expanded, recursive=True):
                path = Path(path_value)
                if not path.is_file():
                    continue
                matched += 1
                for row in read_benchmark_rows(
                    path,
                    benchmark_name=benchmark_name,
                    text_fields=fields,
                ):
                    texts = [text for field in fields for text in _flatten_text(row.get(field))]
                    text = _normalized_hash_text("\n".join(texts))
                    if not text:
                        continue
                    exact.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
                    compact = re.sub(r"\s+", "", text)
                    zh_ngrams.update(compact[index : index + 32] for index in range(max(0, len(compact) - 31)))
                    words = tuple(re.findall(r"\w+", text.lower()))
                    word_ngrams.update(words[index : index + 13] for index in range(max(0, len(words) - 12)))
        if benchmark.get("required", True) and matched == 0:
            missing.append(benchmark_name)
    if missing:
        raise FileNotFoundError(
            f"required benchmark paths are unresolved or empty: {sorted(missing)}"
        )
    if not exact:
        raise RuntimeError(f"benchmark registry produced no contamination entries: {registry_path}")
    return exact, zh_ngrams, word_ngrams


class DecontaminationFilter(PipelineStep):
    name = "Benchmark decontamination"
    type = "Filter"

    def __init__(self, exact: set[str], zh_ngrams: set[str], word_ngrams: set[tuple[str, ...]]) -> None:
        super().__init__()
        self.exact = exact
        self.zh_ngrams = zh_ngrams
        self.word_ngrams = word_ngrams

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> Iterator[Document]:
        del rank, world_size
        for document in data:
            text = _normalized_hash_text(document.text)
            if hashlib.sha256(text.encode("utf-8")).hexdigest() in self.exact:
                self.stat_update("removed_exact")
                continue
            if str(document.metadata.get("language")) in {"zh", "zh_en_mixed"}:
                compact = re.sub(r"\s+", "", text)
                hits = sum(compact[index : index + 32] in self.zh_ngrams for index in range(max(0, len(compact) - 31)))
                contaminated = hits >= 2
            else:
                words = tuple(re.findall(r"\w+", text.lower()))
                contaminated = any(
                    words[index : index + 13] in self.word_ngrams
                    for index in range(max(0, len(words) - 12))
                )
            if contaminated:
                self.stat_update("removed_ngram")
                continue
            self.stat_update("forwarded")
            yield document


def run_decontamination(
    config: DataFactoryConfig,
    plan: dict[str, Any],
    **executor_options: Any,
) -> dict[str, Any]:
    if config.dedup.minhash_enabled:
        input_root = config.run_root / "dedup" / "minhash" / str(plan["plan_sha256"])[:16] / "documents"
    else:
        input_root = config.run_root / "dedup" / "exact" / str(plan["plan_sha256"])[:16] / "documents"
    files = _parquet_files(input_root)
    exact, zh_ngrams, word_ngrams = _benchmark_index(config.dedup.decontamination_registry)
    output = config.run_root / "decontaminated" / str(plan["plan_sha256"])[:16]
    tasks = min(executor_options.get("tasks") or len(files), len(files))
    pipeline = [
        ParquetDocumentReader(files),
        DecontaminationFilter(exact, zh_ngrams, word_ngrams),
        DistributionStats(),
        TaskOutputGuard(output),
        ParquetWriter(
            output_folder=str(output),
            output_filename="${rank}.parquet",
            compression="zstd",
            expand_metadata=True,
            schema=_dedup_document_schema(),

            max_file_size=512 * 2**20,
        ),
    ]
    executor = build_executor(
        pipeline=pipeline,
        logging_dir=config.run_root / "logs" / "decontamination" / str(plan["plan_sha256"])[:16],
        job_name=f"cc_decontam_{config.phase}",
        tasks=tasks,
        **_executor_kwargs(executor_options),
    )
    executor.run()
    return {"stage": "decontaminate", "output": str(output), "tasks": tasks}
