"""Fast Phase 1 candidate sampling, exact tokenization, quota selection and packing.

Pipeline invariants:
1. corpus-wide passes never invoke the tokenizer;
2. exact dedup/decontamination happens before tokenization;
3. every retained candidate document is encoded exactly once;
4. quota selection and sequence packing operate on stored token ids.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data_factory.candidate_features import (
    ALIGNMENT_POOLS,
    AlignmentFeatureMatcher,
    CandidateClassifier,
    CandidateSkip,
)
from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.fast_common import (
    PartitionedParquetWriter,
    TokenEstimator,
    json_dumps,
    json_loads,
    parquet_files,
    short_content_hash,
    stable_fraction,
    stable_key,
    write_fast_report,
)
from scripts.data_factory.io_utils import JsonlShardWriter, TokenJsonlShardWriter, expand_paths, iter_jsonl
from scripts.data_factory.selection import ALIGNMENT_VALIDATION_GROUPS, NATURAL_VALIDATION_WEIGHTS, scale_quotas
from scripts.data_factory.text import content_hash, normalize_text


CANDIDATE_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("text", pa.large_string()),
        ("source", pa.string()),
        ("intent", pa.string()),
        ("pool", pa.string()),
        ("quota_group", pa.string()),
        ("license", pa.string()),
        ("license_status", pa.string()),
        ("estimated_tokens", pa.int64()),
        ("sample_key", pa.uint64()),
        ("bridge_hits", pa.large_string()),
        ("new_hanzi_hits", pa.large_string()),
        ("meta_json", pa.large_string()),
    ]
)

TOKENIZED_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("text", pa.large_string()),
        ("source", pa.string()),
        ("intent", pa.string()),
        ("pool", pa.string()),
        ("quota_group", pa.string()),
        ("license", pa.string()),
        ("license_status", pa.string()),
        ("token_count", pa.int64()),
        ("sample_key", pa.uint64()),
        ("bridge_hits", pa.large_string()),
        ("new_hanzi_hits", pa.large_string()),
        ("meta_json", pa.large_string()),
        ("input_ids", pa.large_list(pa.int32())),
    ]
)

PACKED_SCHEMA = pa.schema(
    [
        ("input_ids", pa.large_list(pa.int32())),
        ("token_count", pa.int32()),
    ]
)

CHINESE_INTENTS = frozenset(
    {"chinese_natural", "mixed_zh_en", "multi_hanzi_bridge", "new_hanzi_coverage"}
)
FEATURE_INTENTS = frozenset({"multi_hanzi_bridge", "new_hanzi_coverage"})


def _cache_row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    record = {
        "doc_id": row["doc_id"],
        "text": row["text"],
        "source": row["source"],
        "category": row["category"],
        "quota_group": row.get("quota_group"),
        "license": row.get("license"),
        "license_status": row.get("license_status"),
        "path": row.get("path"),
        "subset": row.get("subset"),
    }
    record.update(json_loads(row.get("meta_json"), {}))
    return record


def _base_intent(pool: str, quota_group: str | None) -> str:
    if pool == "specialized":
        if quota_group not in {"code", "math_science", "structured"}:
            raise ValueError(f"invalid specialized quota group: {quota_group!r}")
        return f"specialized:{quota_group}"
    if pool not in {"chinese_natural", "mixed_zh_en", "non_chinese"}:
        raise ValueError(f"unsupported candidate pool: {pool!r}")
    return pool


def _train_targets(config: PipelineConfig) -> dict[str, int]:
    return {
        "chinese_natural": config.quotas["chinese_natural"],
        "multi_hanzi_bridge": config.quotas["multi_hanzi_bridge"],
        "new_hanzi_coverage": config.quotas["new_hanzi_coverage"],
        "non_chinese": config.quotas["non_chinese"],
        "mixed_zh_en": config.quotas["mixed_zh_en"],
        "specialized:code": config.specialized_quotas["code"],
        "specialized:math_science": config.specialized_quotas["math_science"],
        "specialized:structured": config.specialized_quotas["structured"],
    }


def _validation_requests(config: PipelineConfig) -> list[dict[str, Any]]:
    natural_targets = scale_quotas(NATURAL_VALIDATION_WEIGHTS, config.validation.natural_tokens)
    natural_mapping = {
        "chinese_natural": "chinese_natural",
        "non_chinese": "non_chinese",
        "mixed_zh_en": "mixed_zh_en",
        "code": "specialized:code",
        "math_science": "specialized:math_science",
        "structured": "specialized:structured",
    }
    requests = [
        {
            "split": "validation_natural",
            "category": category,
            "intent": natural_mapping[category],
            "target_tokens": target,
        }
        for category, target in natural_targets.items()
    ]
    alignment_targets = scale_quotas(
        {name: 1 for name in ALIGNMENT_VALIDATION_GROUPS},
        config.validation.alignment_tokens,
    )
    alignment_mapping = {
        "new_hanzi_coverage": "new_hanzi_coverage",
        "multi_hanzi_bridge": "multi_hanzi_bridge",
        "original_hanzi": "chinese_natural",
        "non_chinese": "non_chinese",
    }
    requests.extend(
        {
            "split": "validation_alignment",
            "category": category,
            "intent": alignment_mapping[category],
            "target_tokens": target,
        }
        for category, target in alignment_targets.items()
    )
    return requests


def _candidate_targets(config: PipelineConfig) -> dict[str, int]:
    targets = Counter(_train_targets(config))
    for request in _validation_requests(config):
        targets[str(request["intent"])] += int(request["target_tokens"])
    return dict(targets)


def _estimate_row(estimator: TokenEstimator, row: dict[str, Any]) -> int:
    return estimator.estimate(
        source=str(row["source"]),
        char_count=int(row["char_count"]),
        hanzi_count=int(row["hanzi_count"]),
        latin_count=int(row["latin_count"]),
        digit_count=int(row["digit_count"]),
    )


def _allocate_source_targets(
    available: dict[str, int],
    target: int,
    cap_ratio: float | None,
) -> dict[str, int]:
    """Proportionally allocate a target while optionally capping one source's share."""
    positive = {source: max(0, int(tokens)) for source, tokens in available.items() if tokens > 0}
    if not positive or target <= 0:
        return {}
    target = min(int(target), sum(positive.values()))
    cap: int | None = None
    if cap_ratio is not None and len(positive) >= math.ceil(1 / cap_ratio):
        cap = max(1, int(target * cap_ratio))

    remaining_sources = set(positive)
    result = {source: 0 for source in positive}
    remaining_target = target
    while remaining_sources and remaining_target > 0:
        available_total = sum(positive[source] for source in remaining_sources)
        if available_total <= 0:
            break
        changed = False
        for source in list(remaining_sources):
            share = remaining_target * positive[source] / available_total
            limit = min(positive[source], cap) if cap is not None else positive[source]
            if share >= limit:
                result[source] = int(limit)
                remaining_target -= int(limit)
                remaining_sources.remove(source)
                changed = True
        if not changed:
            floors: dict[str, int] = {}
            remainders: list[tuple[float, str]] = []
            for source in remaining_sources:
                raw = remaining_target * positive[source] / available_total
                value = min(positive[source], int(math.floor(raw)))
                floors[source] = value
                remainders.append((raw - value, source))
            for source, value in floors.items():
                result[source] = value
            missing = remaining_target - sum(floors.values())
            for _fraction, source in sorted(remainders, reverse=True):
                if missing <= 0:
                    break
                if result[source] < positive[source]:
                    result[source] += 1
                    missing -= 1
            remaining_target = 0
    return {source: tokens for source, tokens in result.items() if tokens > 0}


def _load_exclusion_hashes(config: PipelineConfig) -> set[str]:
    hashes: set[str] = set()
    paths = expand_paths(config.dedup.get("decontamination_paths", []), config.repo_root)
    for row in iter_jsonl(paths):
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            hashes.add(content_hash(text))
    return hashes


def _iter_cache_rows(config: PipelineConfig) -> Iterator[dict[str, Any]]:
    columns = [
        "doc_id",
        "text",
        "source",
        "category",
        "quota_group",
        "license",
        "license_status",
        "path",
        "subset",
        "char_count",
        "hanzi_count",
        "latin_count",
        "digit_count",
        "meta_json",
    ]
    for path in parquet_files(config.fast_cache_dir):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            yield from batch.to_pylist()


def prescan_and_sample_candidates(
    config: PipelineConfig,
    estimator: TokenEstimator,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    fast = config.fast_pipeline
    if overwrite:
        shutil.rmtree(config.fast_candidate_dir, ignore_errors=True)
    elif parquet_files(config.fast_candidate_dir):
        raise FileExistsError(
            f"candidate Parquet already exists under {config.fast_candidate_dir}; use --overwrite"
        )

    classifier = CandidateClassifier(config.quality)
    matcher = AlignmentFeatureMatcher(config)
    available: dict[str, Counter[str]] = defaultdict(Counter)
    bridge_occurrences: Counter[int] = Counter()
    bridge_documents: Counter[int] = Counter()
    new_hanzi_documents: Counter[str] = Counter()
    scanned = 0
    skipped: Counter[str] = Counter()

    # Pass 1: only cheap statistics and alignment string matching.
    for cached in _iter_cache_rows(config):
        scanned += 1
        record = _cache_row_to_record(cached)
        text = str(cached["text"])
        try:
            pool, quota_group, _stats = classifier.classify(record, text)
        except CandidateSkip as error:
            skipped[error.reason] += 1
            continue
        estimated = _estimate_row(estimator, cached)
        source = str(cached["source"])
        base = _base_intent(pool, quota_group)
        available[base][source] += estimated
        alignment_eligible = pool in ALIGNMENT_POOLS or (
            pool == "specialized" and quota_group == "math_science"
        )
        if alignment_eligible:
            bridge_hits = matcher.bridge_hits(text)
            new_hits = matcher.new_hanzi_hits(text)
            if bridge_hits:
                available["multi_hanzi_bridge"][source] += estimated
                bridge_occurrences.update(bridge_hits)
                bridge_documents.update(bridge_hits.keys())
            if new_hits:
                available["new_hanzi_coverage"][source] += estimated
                new_hanzi_documents.update(new_hits.keys())
        if scanned % 1_000_000 == 0:
            print(f"fast prescan: {scanned:,} cached documents")

    top_bridge = {
        token_id for token_id, _count in bridge_occurrences.most_common(config.vocab_alignment.bridge_top_token_count)
    }
    targets = _candidate_targets(config)
    source_targets: dict[str, dict[str, int]] = {}
    rates: dict[str, dict[str, float]] = {}
    for intent, target in targets.items():
        cap = fast.source_cap_ratio if intent in CHINESE_INTENTS else None
        allocation = _allocate_source_targets(dict(available.get(intent, {})), target, cap)
        source_targets[intent] = allocation
        buffer = fast.feature_oversample_ratio if intent in FEATURE_INTENTS else fast.oversample_ratio
        rates[intent] = {
            source: min(1.0, buffer * allocated / max(1, available[intent][source]))
            for source, allocated in allocation.items()
        }

    exclusions = _load_exclusion_hashes(config)
    seen_hashes: set[bytes] = set()
    candidate_tokens: Counter[str] = Counter()
    candidate_records: Counter[str] = Counter()
    removed: Counter[str] = Counter()

    with PartitionedParquetWriter(
        config.fast_candidate_dir,
        CANDIDATE_SCHEMA,
        prefix="candidate",
        max_rows=fast.candidate_rows_per_shard,
        compression=fast.parquet_compression,
    ) as writer:
        for cached in _iter_cache_rows(config):
            record = _cache_row_to_record(cached)
            text = str(cached["text"])
            try:
                pool, quota_group, _stats = classifier.classify(record, text)
            except CandidateSkip as error:
                removed[error.reason] += 1
                continue
            source = str(cached["source"])
            base = _base_intent(pool, quota_group)
            estimated = _estimate_row(estimator, cached)
            alignment_eligible = pool in ALIGNMENT_POOLS or (
                pool == "specialized" and quota_group == "math_science"
            )
            bridge_hits = matcher.bridge_hits(text) if alignment_eligible else Counter()
            if top_bridge:
                bridge_hits = Counter(
                    {token_id: count for token_id, count in bridge_hits.items() if token_id in top_bridge}
                )
            new_hits = matcher.new_hanzi_hits(text) if alignment_eligible else Counter()

            selected: list[str] = []
            if new_hits:
                rate = rates.get("new_hanzi_coverage", {}).get(source, 0.0)
                if stable_fraction(config.seed, "candidate:new_hanzi_coverage", str(cached["doc_id"])) < rate:
                    selected.append("new_hanzi_coverage")
            if bridge_hits:
                rate = rates.get("multi_hanzi_bridge", {}).get(source, 0.0)
                if stable_fraction(config.seed, "candidate:multi_hanzi_bridge", str(cached["doc_id"])) < rate:
                    selected.append("multi_hanzi_bridge")
            base_rate = rates.get(base, {}).get(source, 0.0)
            if stable_fraction(config.seed, f"candidate:{base}", str(cached["doc_id"])) < base_rate:
                selected.append(base)
            if not selected:
                continue

            # Alignment intents take priority so the same parent document never pays
            # exact-tokenization cost twice or leaks across quota categories.
            if "new_hanzi_coverage" in selected:
                intent = "new_hanzi_coverage"
            elif "multi_hanzi_bridge" in selected:
                intent = "multi_hanzi_bridge"
            else:
                intent = base

            if content_hash(text) in exclusions:
                removed["contamination_exact"] += 1
                continue
            digest = short_content_hash(text)
            if digest in seen_hashes:
                removed["exact_duplicate"] += 1
                continue
            seen_hashes.add(digest)

            meta = {
                "source_category": cached.get("category"),
                "source_quota_group": cached.get("quota_group"),
                "path": cached.get("path"),
                "subset": cached.get("subset"),
                **json_loads(cached.get("meta_json"), {}),
            }
            writer.write(
                intent,
                {
                    "doc_id": str(cached["doc_id"]),
                    "text": text,
                    "source": source,
                    "intent": intent,
                    "pool": pool,
                    "quota_group": quota_group,
                    "license": str(cached.get("license") or "unknown"),
                    "license_status": str(cached.get("license_status") or "unknown"),
                    "estimated_tokens": estimated,
                    "sample_key": stable_key(config.seed, f"candidate-output:{intent}", str(cached["doc_id"])),
                    "bridge_hits": json_dumps({str(key): value for key, value in bridge_hits.items()}),
                    "new_hanzi_hits": json_dumps(dict(new_hits)),
                    "meta_json": json_dumps(meta),
                },
            )
            candidate_tokens[intent] += estimated
            candidate_records[intent] += 1

    return write_fast_report(
        config.reports_dir / "phase1_fast_prescan_report.json",
        {
            "stage": "fast_prescan",
            "scanned_documents": scanned,
            "skipped_by_reason": dict(sorted(skipped.items())),
            "candidate_removed_by_reason": dict(sorted(removed.items())),
            "available_estimated_tokens": {
                intent: dict(sorted(values.items())) for intent, values in sorted(available.items())
            },
            "targets": targets,
            "source_targets": source_targets,
            "sampling_rates": rates,
            "top_bridge_tokens": [
                {"old_token_id": token_id, "occurrences": bridge_occurrences[token_id], "documents": bridge_documents[token_id]}
                for token_id, _count in bridge_occurrences.most_common(config.vocab_alignment.bridge_top_token_count)
            ],
            "new_hanzi_document_frequency": dict(new_hanzi_documents.most_common()),
            "candidate_records": dict(sorted(candidate_records.items())),
            "candidate_estimated_tokens": dict(sorted(candidate_tokens.items())),
            "exact_dedup_keys": len(seen_hashes),
        },
    )


def _load_tokenizer(config: PipelineConfig, rayon_threads: int):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(max(1, rayon_threads))
    from tokenizers import Tokenizer

    path = config.tokenizer_path / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return Tokenizer.from_file(str(path))


def tokenize_candidates(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    fast = config.fast_pipeline
    candidate_files = parquet_files(config.fast_candidate_dir)
    if not candidate_files:
        raise FileNotFoundError("no fast candidates; run fast prescan first")
    if overwrite:
        shutil.rmtree(config.fast_tokenized_dir, ignore_errors=True)
    elif parquet_files(config.fast_tokenized_dir):
        raise FileExistsError(
            f"tokenized candidates exist under {config.fast_tokenized_dir}; use --overwrite"
        )
    tokenizer = _load_tokenizer(config, workers or fast.tokenizer_rayon_threads)

    records = Counter()
    tokens = Counter()
    skipped = Counter()
    batch: list[dict[str, Any]] = []
    batch_chars = 0

    with PartitionedParquetWriter(
        config.fast_tokenized_dir,
        TOKENIZED_SCHEMA,
        prefix="tokenized",
        max_rows=fast.tokenized_rows_per_shard,
        compression=fast.parquet_compression,
    ) as writer:

        def consume() -> None:
            nonlocal batch, batch_chars
            if not batch:
                return
            texts = [str(row["text"]) for row in batch]
            try:
                encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
                pairs = list(zip(batch, encodings, strict=True))
            except Exception:
                pairs = []
                for row, text in zip(batch, texts, strict=True):
                    try:
                        pairs.append((row, tokenizer.encode(text, add_special_tokens=False)))
                    except Exception:
                        skipped["tokenization_error"] += 1
            for row, encoding in pairs:
                ids = list(encoding.ids)
                if not ids:
                    skipped["zero_tokens"] += 1
                    continue
                intent = str(row["intent"])
                writer.write(
                    intent,
                    {
                        "doc_id": row["doc_id"],
                        "text": row["text"] if fast.keep_candidate_text else "",
                        "source": row["source"],
                        "intent": intent,
                        "pool": row["pool"],
                        "quota_group": row.get("quota_group"),
                        "license": row["license"],
                        "license_status": row["license_status"],
                        "token_count": len(ids),
                        "sample_key": int(row["sample_key"]),
                        "bridge_hits": row["bridge_hits"],
                        "new_hanzi_hits": row["new_hanzi_hits"],
                        "meta_json": row["meta_json"],
                        "input_ids": ids,
                    },
                )
                records[intent] += 1
                tokens[intent] += len(ids)
            batch = []
            batch_chars = 0

        for path in candidate_files:
            parquet = pq.ParquetFile(path)
            for arrow_batch in parquet.iter_batches(batch_size=4096):
                for row in arrow_batch.to_pylist():
                    text_len = len(str(row["text"]))
                    if batch and (
                        len(batch) >= fast.exact_batch_size
                        or batch_chars + text_len > fast.exact_batch_chars
                    ):
                        consume()
                    batch.append(row)
                    batch_chars += text_len
        consume()

    return write_fast_report(
        config.reports_dir / "phase1_fast_tokenization_report.json",
        {
            "stage": "exact_tokenization",
            "records_by_intent": dict(sorted(records.items())),
            "tokens_by_intent": dict(sorted(tokens.items())),
            "total_records": sum(records.values()),
            "total_tokens": sum(tokens.values()),
            "skipped_by_reason": dict(sorted(skipped.items())),
            "tokenizer_path": str(config.tokenizer_path),
            "rayon_threads": workers or fast.tokenizer_rayon_threads,
        },
    )


def _read_hanzi_resource(path: Path) -> set[str]:
    chars: set[str] = set()
    if not path.is_file():
        return chars
    with path.open("rt", encoding="utf-8") as file:
        for line in file:
            value = line.lstrip("\ufeff").strip()
            if not value or value.startswith("#"):
                continue
            field = value.split("\t", 1)[0].strip()
            if field and field.lower() != "char":
                chars.add(field[0])
    return chars


def _priority_hanzi(config: PipelineConfig) -> set[str]:
    chars: set[str] = set()
    for path in config.vocab_alignment.priority_hanzi_paths:
        chars.update(_read_hanzi_resource(path))
    return chars


def _intent_files(config: PipelineConfig, intent: str) -> list[Path]:
    safe = PartitionedParquetWriter._safe_partition(intent)
    root = config.fast_tokenized_dir / safe
    return sorted(root.glob("*.parquet")) if root.is_dir() else []


def _iter_tokenized(config: PipelineConfig, intent: str) -> Iterator[dict[str, Any]]:
    for path in _intent_files(config, intent):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048):
            yield from batch.to_pylist()


class PackedWriter:
    def __init__(self, root: Path, name: str, *, sequence_length: int, shard_tokens: int, eos_id: int) -> None:
        self.root = root
        self.name = name
        self.sequence_length = sequence_length
        self.shard_tokens = shard_tokens
        self.eos_id = eos_id
        self.buffer: list[int] = []
        self.rows: list[dict[str, Any]] = []
        self.shard_index = 0
        self.tokens_in_shard = 0
        self.total_tokens = 0
        self.total_sequences = 0
        self.files: list[dict[str, Any]] = []
        self.root.mkdir(parents=True, exist_ok=True)

    def _write_rows(self) -> None:
        if not self.rows:
            return
        path = self.root / f"{self.name}-{self.shard_index:05d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(self.rows, schema=PACKED_SCHEMA),
            path,
            compression="zstd",
            use_dictionary=False,
        )
        self.files.append(
            {"path": str(path), "sequences": len(self.rows), "tokens": self.tokens_in_shard}
        )
        self.rows = []
        self.shard_index += 1
        self.tokens_in_shard = 0

    def _emit(self) -> None:
        if not self.buffer:
            return
        row = {"input_ids": list(self.buffer), "token_count": len(self.buffer)}
        self.rows.append(row)
        self.tokens_in_shard += len(self.buffer)
        self.total_tokens += len(self.buffer)
        self.total_sequences += 1
        self.buffer.clear()
        if self.tokens_in_shard >= self.shard_tokens:
            self._write_rows()

    def add_document(self, input_ids: list[int]) -> None:
        offset = 0
        while offset < len(input_ids):
            remaining = self.sequence_length - len(self.buffer)
            take = min(remaining, len(input_ids) - offset)
            self.buffer.extend(input_ids[offset : offset + take])
            offset += take
            if len(self.buffer) == self.sequence_length:
                self._emit()
        if self.buffer:
            if len(self.buffer) < self.sequence_length:
                self.buffer.append(self.eos_id)
            if len(self.buffer) == self.sequence_length:
                self._emit()

    def close(self) -> None:
        self._emit()
        self._write_rows()


def _eos_token_id(config: PipelineConfig) -> int:
    tokenizer_config = config.tokenizer_path / "tokenizer_config.json"
    raw = json.loads(tokenizer_config.read_text(encoding="utf-8")) if tokenizer_config.is_file() else {}
    eos = raw.get("eos_token")
    # Qwen tokenizer_config often stores a token string while generation_config stores the id.
    generation = config.tokenizer_path / "generation_config.json"
    if generation.is_file():
        value = json.loads(generation.read_text(encoding="utf-8")).get("eos_token_id")
        if isinstance(value, int):
            return value
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(config.tokenizer_path / "tokenizer.json"))
    if isinstance(eos, str):
        token_id = tokenizer.token_to_id(eos)
        if token_id is not None:
            return int(token_id)
    token_id = tokenizer.token_to_id("<|endoftext|>")
    if token_id is None:
        raise RuntimeError("cannot resolve eos_token_id for packed Phase 1 data")
    return int(token_id)


@dataclass
class SelectionSink:
    config: PipelineConfig
    train_writer: TokenJsonlShardWriter
    validation_natural: JsonlShardWriter
    validation_alignment: JsonlShardWriter
    train_packer: PackedWriter
    natural_packer: PackedWriter
    alignment_packer: PackedWriter

    def write(self, row: dict[str, Any], *, split: str, category: str) -> None:
        token_count = int(row["token_count"])
        meta = json_loads(row.get("meta_json"), {})
        output = {
            "doc_id": row["doc_id"],
            "parent_doc_id": row["doc_id"],
            "text": row.get("text", ""),
            "source": row["source"],
            "category": category if not category.startswith("specialized:") else "specialized",
            "quota_group": (
                category.split(":", 1)[1]
                if category.startswith("specialized:")
                else row.get("quota_group")
            ),
            "license": row["license"],
            "license_status": row["license_status"],
            "token_count": token_count,
            **{key: value for key, value in meta.items() if key not in {"text", "category", "quota_group"}},
        }
        ids = [int(value) for value in row["input_ids"]]
        if split == "train":
            self.train_writer.write(output)
            self.train_packer.add_document(ids)
        elif split == "validation_natural":
            self.validation_natural.write(output)
            self.natural_packer.add_document(ids)
        elif split == "validation_alignment":
            self.validation_alignment.write(output)
            self.alignment_packer.add_document(ids)
        else:
            raise ValueError(split)


def _load_new_hanzi(config: PipelineConfig) -> set[str]:
    raw = json.loads(config.vocab_alignment.new_hanzi_token_ids_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("new_hanzi_token_ids.json must contain an object")
    return {
        char for char in raw.values()
        if isinstance(char, str) and len(char) == 1
    }


def _feature_preselection(
    config: PipelineConfig,
    sink: SelectionSink,
    selected: set[int],
    train_tokens: Counter[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    # ---- New-Hanzi coverage -------------------------------------------------
    new_chars = _load_new_hanzi(config)
    priority = _priority_hanzi(config) & new_chars
    observed: set[str] = set()
    available_documents: Counter[str] = Counter()
    for row in _iter_tokenized(config, "new_hanzi_coverage"):
        hits = {str(char) for char in json_loads(row.get("new_hanzi_hits"), {})}
        observed.update(hits)
        available_documents.update(hits)

    char_counts: Counter[str] = Counter()
    intent = "new_hanzi_coverage"
    target = _train_targets(config)[intent]

    def take_new(predicate) -> None:
        for row in _iter_tokenized(config, intent):
            if train_tokens[intent] >= target:
                break
            hits = {str(char) for char in json_loads(row.get("new_hanzi_hits"), {})}
            if not hits or not predicate(hits):
                continue
            key = stable_key(config.seed, "selected", str(row["doc_id"]))
            if key in selected:
                continue
            selected.add(key)
            sink.write(row, split="train", category=intent)
            train_tokens[intent] += int(row["token_count"])
            char_counts.update(hits)

    # First ensure every observed new Hanzi appears in the selected train pool.
    take_new(lambda hits: any(char_counts[char] == 0 for char in hits))
    minimum = config.vocab_alignment.priority_hanzi_min_documents
    if train_tokens[intent] < target:
        take_new(
            lambda hits: any(
                char in priority and char_counts[char] < min(minimum, available_documents[char])
                for char in hits
            )
        )

    uncovered_observed = sorted(observed - set(char_counts), key=ord)
    priority_met = sum(
        char_counts[char] >= min(minimum, available_documents[char])
        for char in priority
    )
    priority_ratio = priority_met / len(priority) if priority else 1.0
    new_passed = not uncovered_observed and priority_ratio >= config.vocab_alignment.priority_hanzi_coverage
    report[intent] = {
        "passed": new_passed,
        "target_chars": len(new_chars),
        "observed_candidate_chars": len(observed),
        "covered_observed_chars": len(observed) - len(uncovered_observed),
        "uncovered_observed_chars": uncovered_observed,
        "priority_characters": len(priority),
        "priority_characters_met": priority_met,
        "priority_coverage": priority_ratio,
        "required_priority_coverage": config.vocab_alignment.priority_hanzi_coverage,
        "min_documents": minimum,
        "tokens_preselected": train_tokens[intent],
    }

    # ---- Removed multi-Hanzi bridge coverage -------------------------------
    prescan = json.loads((config.reports_dir / "phase1_fast_prescan_report.json").read_text(encoding="utf-8"))
    top_bridge = [int(item["old_token_id"]) for item in prescan.get("top_bridge_tokens", [])]
    top_set = set(top_bridge)
    available_bridge_documents: Counter[int] = Counter()
    for row in _iter_tokenized(config, "multi_hanzi_bridge"):
        hits = {int(key) for key in json_loads(row.get("bridge_hits"), {}) if int(key) in top_set}
        available_bridge_documents.update(hits)

    bridge_counts: Counter[int] = Counter()
    intent = "multi_hanzi_bridge"
    target = _train_targets(config)[intent]
    minimum = config.vocab_alignment.bridge_min_contexts
    for row in _iter_tokenized(config, intent):
        if train_tokens[intent] >= target:
            break
        hits = {
            int(key) for key in json_loads(row.get("bridge_hits"), {})
            if int(key) in top_set
        }
        if not any(
            bridge_counts[token_id] < min(minimum, available_bridge_documents[token_id])
            for token_id in hits
        ):
            continue
        key = stable_key(config.seed, "selected", str(row["doc_id"]))
        if key in selected:
            continue
        selected.add(key)
        sink.write(row, split="train", category=intent)
        train_tokens[intent] += int(row["token_count"])
        bridge_counts.update(hits)

    unmet = [
        token_id for token_id in top_bridge
        if bridge_counts[token_id] < min(minimum, available_bridge_documents[token_id])
    ]
    bridge_passed = not unmet
    report[intent] = {
        "passed": bridge_passed,
        "top_bridge_tokens": len(top_bridge),
        "covered_tokens": len(top_bridge) - len(unmet),
        "unmet_count": len(unmet),
        "unmet_token_ids": unmet[:100],
        "min_contexts": minimum,
        "tokens_preselected": train_tokens[intent],
    }
    return report

def _select_request(
    config: PipelineConfig,
    sink: SelectionSink,
    *,
    intent: str,
    split: str,
    category: str,
    target_tokens: int,
    selected: set[int],
    current_tokens: int = 0,
) -> int:
    if current_tokens >= target_tokens:
        return current_tokens
    tokenization_report = json.loads(
        (config.reports_dir / "phase1_fast_tokenization_report.json").read_text(encoding="utf-8")
    )
    total_available = int(tokenization_report.get("tokens_by_intent", {}).get(intent, 0))
    remaining = max(0, target_tokens - current_tokens)
    probability = min(1.0, remaining / max(1, total_available) * 1.10)

    # First pass is fully hash-randomized.  A second deterministic refill pass is
    # only used for the small residual caused by sampling variance.
    for pass_index, threshold in enumerate((probability, 1.0)):
        namespace = f"final:{split}:{category}:pass{pass_index}"
        for row in _iter_tokenized(config, intent):
            if current_tokens >= target_tokens:
                break
            selected_key = stable_key(config.seed, "selected", str(row["doc_id"]))
            if selected_key in selected:
                continue
            if stable_fraction(config.seed, namespace, str(row["doc_id"])) >= threshold:
                continue
            selected.add(selected_key)
            sink.write(row, split=split, category=category)
            current_tokens += int(row["token_count"])
        if current_tokens >= target_tokens:
            break
    return current_tokens


def finalize_fast_phase1(config: PipelineConfig, *, overwrite: bool = False) -> dict[str, Any]:
    fast = config.fast_pipeline
    if overwrite:
        shutil.rmtree(config.fast_final_dir, ignore_errors=True)
    elif config.fast_final_dir.exists() and any(config.fast_final_dir.rglob("*")):
        raise FileExistsError(f"fast final output exists under {config.fast_final_dir}; use --overwrite")
    config.fast_final_dir.mkdir(parents=True, exist_ok=True)

    eos_id = _eos_token_id(config)
    train_dir = config.fast_final_dir / "train"
    validation_dir = config.fast_final_dir / "validation"
    packed_dir = config.fast_final_dir / "packed"
    train_writer = TokenJsonlShardWriter(train_dir, "train", config.final_shard_tokens)
    natural_writer = JsonlShardWriter(validation_dir, "validation_natural", max_records=100_000)
    alignment_writer = JsonlShardWriter(validation_dir, "validation_alignment", max_records=100_000)
    train_packer = PackedWriter(
        packed_dir / "train", "train", sequence_length=fast.pack_sequence_length,
        shard_tokens=fast.pack_shard_tokens, eos_id=eos_id,
    )
    natural_packer = PackedWriter(
        packed_dir / "validation_natural", "validation_natural", sequence_length=fast.pack_sequence_length,
        shard_tokens=max(fast.pack_sequence_length, config.validation.natural_tokens * 2), eos_id=eos_id,
    )
    alignment_packer = PackedWriter(
        packed_dir / "validation_alignment", "validation_alignment", sequence_length=fast.pack_sequence_length,
        shard_tokens=max(fast.pack_sequence_length, config.validation.alignment_tokens * 2), eos_id=eos_id,
    )
    sink = SelectionSink(
        config, train_writer, natural_writer, alignment_writer,
        train_packer, natural_packer, alignment_packer,
    )

    selected: set[int] = set()
    train_tokens: Counter[str] = Counter()
    coverage = _feature_preselection(config, sink, selected, train_tokens)

    validation_results: list[dict[str, Any]] = []
    for request in _validation_requests(config):
        actual = _select_request(
            config,
            sink,
            intent=str(request["intent"]),
            split=str(request["split"]),
            category=str(request["category"]),
            target_tokens=int(request["target_tokens"]),
            selected=selected,
        )
        validation_results.append({**request, "actual_tokens": actual})

    train_targets = _train_targets(config)
    for intent, target in train_targets.items():
        train_tokens[intent] = _select_request(
            config,
            sink,
            intent=intent,
            split="train",
            category=intent,
            target_tokens=target,
            selected=selected,
            current_tokens=train_tokens[intent],
        )

    train_writer.close()
    natural_writer.close()
    alignment_writer.close()
    train_packer.close()
    natural_packer.close()
    alignment_packer.close()

    train_checks = {
        intent: {
            "target_tokens": target,
            "actual_tokens": train_tokens[intent],
            "relative_error": abs(train_tokens[intent] - target) / target,
            "passed": abs(train_tokens[intent] - target) / target <= config.tolerance,
        }
        for intent, target in train_targets.items()
    }
    validation_checks = [
        {
            **row,
            "relative_error": abs(int(row["actual_tokens"]) - int(row["target_tokens"])) / int(row["target_tokens"]),
            "passed": abs(int(row["actual_tokens"]) - int(row["target_tokens"])) / int(row["target_tokens"]) <= config.tolerance,
        }
        for row in validation_results
    ]
    coverage_passed = all(bool(item.get("passed", False)) for item in coverage.values())
    passed = (
        all(item["passed"] for item in train_checks.values())
        and all(item["passed"] for item in validation_checks)
        and coverage_passed
    )

    return write_fast_report(
        config.reports_dir / "phase1_fast_final_report.json",
        {
            "stage": "finalize",
            "passed": passed,
            "train": {
                "target_tokens": sum(train_targets.values()),
                "actual_tokens": sum(train_tokens.values()),
                "checks": train_checks,
                "files": train_writer.files,
            },
            "validation": validation_checks,
            "coverage_preselection": coverage,
            "selected_parent_documents": len(selected),
            "packed": {
                "eos_token_id": eos_id,
                "sequence_length": fast.pack_sequence_length,
                "train": {
                    "tokens_including_eos": train_packer.total_tokens,
                    "sequences": train_packer.total_sequences,
                    "files": train_packer.files,
                },
                "validation_natural": {
                    "tokens_including_eos": natural_packer.total_tokens,
                    "sequences": natural_packer.total_sequences,
                    "files": natural_packer.files,
                },
                "validation_alignment": {
                    "tokens_including_eos": alignment_packer.total_tokens,
                    "sequences": alignment_packer.total_sequences,
                    "files": alignment_packer.files,
                },
            },
        },
    )


def build_fast_phase1(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    calibration = json.loads(config.fast_calibration_path.read_text(encoding="utf-8"))
    estimator = TokenEstimator.from_report(calibration)
    prescan = prescan_and_sample_candidates(config, estimator, overwrite=overwrite)
    tokenization = tokenize_candidates(config, overwrite=overwrite, workers=workers)
    final = finalize_fast_phase1(config, overwrite=overwrite)
    return {
        "passed": bool(final.get("passed", False)),
        "target_tokens": sum(config.quotas.values()),
        "actual_tokens": int(final.get("train", {}).get("actual_tokens", 0)),
        "prescan_report": str(config.reports_dir / "phase1_fast_prescan_report.json"),
        "tokenization_report": str(config.reports_dir / "phase1_fast_tokenization_report.json"),
        "final_report": str(config.reports_dir / "phase1_fast_final_report.json"),
        "prescan": prescan,
        "tokenization": tokenization,
        "final": final,
    }
