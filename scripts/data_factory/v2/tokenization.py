"""Exact tokenization, final quota selection, ms-swift export, and packing."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from scripts.data_factory.v2.config import DataFactoryConfig
from scripts.data_factory.v2.mixture import ShardedParquetWriter, _input_schema
from scripts.data_factory.v2.sampling import stable_fraction, stable_key
from scripts.data_factory.v2.sampling import load_new_characters
from scripts.data_factory.v2.quality import MixtureMetrics


def _files(root: Path) -> list[Path]:
    files = sorted(path for path in root.rglob("*.parquet") if path.is_file()) if root.is_dir() else []
    if not files:
        raise FileNotFoundError(f"no Parquet files under {root}")
    return files


def _iter_rows(files: list[Path]) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024):
            yield from batch.to_pylist()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tokenize_selected(
    config: DataFactoryConfig,
    plan: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_root = config.run_root / "selected" / str(plan["plan_sha256"])[:16]
    input_files = _files(input_root)
    mixture_report = json.loads((config.run_root / "reports" / "mixture_report.json").read_text())
    if not mixture_report.get("passed") or mixture_report.get("plan_sha256") != plan["plan_sha256"]:
        raise RuntimeError("tokenize requires a passed mixture for the current plan")
    output_root = config.run_root / "tokenized" / str(plan["plan_sha256"])[:16]
    if output_root.exists() and any(output_root.rglob("*.parquet")):
        if not overwrite:
            raise FileExistsError(f"tokenized output already exists: {output_root}")
        shutil.rmtree(output_root)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(config.tokenizer_path / "tokenizer.json"))
    import pyarrow as pa

    schema = _input_schema(input_files).append(pa.field("input_ids", pa.list_(pa.int64())))
    schema = schema.append(pa.field("token_count", pa.int64())).append(pa.field("tokenizer_sha256", pa.string()))
    writer = ShardedParquetWriter(output_root, schema=schema, max_rows=25_000)
    records: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    batch: list[dict[str, Any]] = []
    selected_ids = {
        str(row["id"]): (str(row["candidate_bucket"]), hashlib.sha256(str(row["text"]).encode()).hexdigest())
        for row in _iter_rows(input_files)
    }
    reused_ids: set[str] = set()
    for digest in reversed(plan.get("previous_plan_hashes", [])):
        previous = config.run_root / "tokenized" / digest[:16]
        if not previous.is_dir():
            continue
        for row in _iter_rows(sorted(previous.rglob("*.parquet"))):
            key = str(row["id"])
            if key not in selected_ids or key in reused_ids or row.get("tokenizer_sha256") != config.hashes.tokenizer:
                continue
            bucket, text_hash = selected_ids[key]
            if hashlib.sha256(str(row["text"]).encode()).hexdigest() != text_hash:
                continue
            writer.write(bucket, {**row, "candidate_bucket": bucket})
            reused_ids.add(key)
            records[bucket] += 1
            tokens[bucket] += int(row["token_count"])

    def consume() -> None:
        nonlocal batch
        if not batch:
            return
        encodings = tokenizer.encode_batch(
            [str(row["text"]) for row in batch],
            add_special_tokens=False,
        )
        for row, encoding in zip(batch, encodings, strict=True):
            bucket = str(row["candidate_bucket"])
            output = dict(row)
            output["input_ids"] = [int(token_id) for token_id in encoding.ids]
            output["token_count"] = len(encoding.ids)
            output["tokenizer_sha256"] = config.hashes.tokenizer
            writer.write(bucket, output)
            records[bucket] += 1
            tokens[bucket] += len(encoding.ids)
        batch = []

    for row in _iter_rows(input_files):
        if str(row["id"]) in reused_ids:
            continue
        batch.append(row)
        if len(batch) >= config.calibration.batch_size:
            consume()
    consume()
    writer.close()
    report = {
        "stage": "tokenize",
        "run_id": config.run_id,
        "plan_sha256": plan["plan_sha256"],
        "tokenizer_sha256": config.hashes.tokenizer,
        "records": dict(records),
        "tokens": dict(tokens),
        "total_tokens": sum(tokens.values()),
        "reused_documents": len(reused_ids),
        "output": str(output_root),
    }
    _write_json(config.run_root / "reports" / "tokenization_report.json", report)
    return report


class SwiftJsonlWriter:
    def __init__(self, root: Path, prefix: str, max_rows: int = 100_000) -> None:
        self.root = root
        self.prefix = prefix
        self.max_rows = max_rows
        self.index = 0
        self.rows = 0
        self.stream = None
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, text: str) -> None:
        if self.stream is None or self.rows >= self.max_rows:
            if self.stream is not None:
                self.stream.close()
            path = self.root / f"{self.prefix}-{self.index:05d}.jsonl"
            self.stream = path.open("wt", encoding="utf-8")
            self.index += 1
            self.rows = 0
        value = {"messages": [{"role": "assistant", "content": text}]}
        self.stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.rows += 1

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()


class PackedWriter:
    def __init__(self, root: Path, sequence_length: int, max_rows: int = 10_000) -> None:
        self.root = root
        self.sequence_length = sequence_length
        self.max_rows = max_rows
        self.buffer: list[int] = []
        self.document_ids: list[str] = []
        self.rows: list[dict[str, Any]] = []
        self.index = 0
        self.total_tokens = 0
        self.root.mkdir(parents=True, exist_ok=True)

    def _emit(self) -> None:
        if not self.buffer:
            return
        self.rows.append(
            {
                "input_ids": list(self.buffer),
                "token_count": len(self.buffer),
                "sequence_length": self.sequence_length,
                "document_ids": list(dict.fromkeys(self.document_ids)),
            }
        )
        self.total_tokens += len(self.buffer)
        self.buffer.clear()
        self.document_ids.clear()
        if len(self.rows) >= self.max_rows:
            self._flush()

    def add(self, doc_id: str, input_ids: list[int], eos_id: int) -> None:
        values = [*input_ids, eos_id]
        offset = 0
        while offset < len(values):
            take = min(self.sequence_length - len(self.buffer), len(values) - offset)
            self.buffer.extend(values[offset : offset + take])
            self.document_ids.append(doc_id)
            offset += take
            if len(self.buffer) == self.sequence_length:
                self._emit()

    def _flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(
            pa.Table.from_pylist(self.rows),
            self.root / f"part-{self.index:05d}.parquet",
            compression="zstd",
        )
        self.index += 1
        self.rows.clear()

    def close(self) -> None:
        self._emit()
        self._flush()


def _eos_id(config: DataFactoryConfig) -> int:
    for name in ("generation_config.json", "config.json"):
        path = config.tokenizer_path / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id")
            if isinstance(value, int):
                return value
    raise RuntimeError(f"cannot resolve eos_token_id from {config.tokenizer_path}")


def _length_for_document(config: DataFactoryConfig, doc_id: str) -> int:
    value = stable_fraction(config.seed, "sequence-length", doc_id)
    cumulative = 0.0
    for length, weight in sorted(config.sequence.lengths.items()):
        cumulative += weight
        if value < cumulative:
            return length
    return max(config.sequence.lengths)


def finalize_dataset(
    config: DataFactoryConfig,
    plan: dict[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    tokenized_root = config.run_root / "tokenized" / str(plan["plan_sha256"])[:16]
    tokenized_files = _files(tokenized_root)
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(config.tokenizer_path / "tokenizer.json"))
    final_root = config.run_root / "final"
    if final_root.exists() and any(final_root.rglob("*")):
        if not overwrite:
            raise FileExistsError(f"final output already exists: {final_root}")
        shutil.rmtree(final_root)

    available: Counter[str] = Counter()
    for row in _iter_rows(tokenized_files):
        available[str(row["candidate_bucket"])] += int(row["token_count"])

    validation_targets = {
        bucket.name: int(round(config.validation_tokens * bucket.fraction))
        for bucket in config.buckets
    }
    validation_candidates: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for row in _iter_rows(tokenized_files):
        bucket = str(row["candidate_bucket"])
        probability = min(1.0, validation_targets[bucket] * 1.5 / max(1, available[bucket]))
        doc_id = str(row.get("parent_doc_id") or row["id"])
        if stable_fraction(config.seed, f"validation:{bucket}", doc_id) < probability:
            validation_candidates[bucket].append(
                (stable_key(config.seed, f"validation-order:{bucket}", doc_id), doc_id, int(row["token_count"]))
            )

    validation_ids: set[str] = set()
    for bucket, candidates in validation_candidates.items():
        total = 0
        for _key, doc_id, tokens in sorted(candidates):
            if total >= validation_targets[bucket]:
                break
            validation_ids.add(doc_id)
            total += tokens

    schema = _input_schema(tokenized_files)
    train_writer = ShardedParquetWriter(final_root / "train", schema=schema)
    val_writer = ShardedParquetWriter(final_root / "validation", schema=schema)
    swift_train = SwiftJsonlWriter(final_root / "ms_swift" / "train", "train")
    swift_val = SwiftJsonlWriter(final_root / "ms_swift" / "validation", "validation")
    eos_id = _eos_id(config)
    packers = {
        length: PackedWriter(final_root / "packed" / str(length), length)
        for length in config.sequence.lengths
    }
    train_tokens: Counter[str] = Counter()
    validation_tokens: Counter[str] = Counter()
    train_documents: set[str] = set()
    validation_documents: set[str] = set()
    metrics = MixtureMetrics(config)
    new_characters = load_new_characters(config.enhancement.token_ids_path)
    character_df: Counter[str] = Counter()

    def clip(row: dict[str, Any], remaining: int) -> dict[str, Any]:
        if int(row["token_count"]) <= remaining:
            return row
        output = dict(row)
        output["input_ids"] = [int(value) for value in row["input_ids"][:remaining]]
        output["token_count"] = remaining
        output["text"] = tokenizer.decode(output["input_ids"], skip_special_tokens=False)
        return output

    for row in _iter_rows(tokenized_files):
        bucket = str(row["candidate_bucket"])
        doc_id = str(row.get("parent_doc_id") or row["id"])
        if doc_id in train_documents or doc_id in validation_documents:
            continue
        tokens = int(row["token_count"])
        if doc_id in validation_ids:
            if validation_tokens[bucket] >= validation_targets[bucket]:
                continue
            row = clip(row, validation_targets[bucket] - validation_tokens[bucket])
            tokens = int(row["token_count"])
            val_writer.write(bucket, row)
            swift_val.write(str(row["text"]))
            validation_tokens[bucket] += tokens
            validation_documents.add(doc_id)
            continue
        if train_tokens[bucket] >= config.bucket_tokens[bucket]:
            continue
        bucket_spec = next(b for b in config.buckets if b.name == bucket)
        source = str(row["source"])
        source_remaining = int(round(config.bucket_tokens[bucket] * bucket_spec.source_weights[source])) - metrics.sources[bucket][source]
        if source_remaining <= 0:
            continue
        row = clip(row, min(config.bucket_tokens[bucket] - train_tokens[bucket], source_remaining))
        tokens = int(row["token_count"])
        train_writer.write(bucket, row)
        swift_train.write(str(row["text"]))
        train_tokens[bucket] += tokens
        train_documents.add(doc_id)
        metrics.add(row, tokens)
        character_df.update(set(str(row["text"])) & new_characters)
        length = _length_for_document(config, doc_id)
        packers[length].add(doc_id, [int(value) for value in row["input_ids"]], eos_id)

    train_writer.close()
    val_writer.close()
    swift_train.close()
    swift_val.close()
    for packer in packers.values():
        packer.close()

    train_checks = {
        bucket.name: {
            "target_tokens": config.bucket_tokens[bucket.name],
            "actual_tokens": train_tokens[bucket.name],
            "relative_error": abs(train_tokens[bucket.name] - config.bucket_tokens[bucket.name])
            / config.bucket_tokens[bucket.name],
        }
        for bucket in config.buckets
    }
    validation_checks = {
        bucket.name: {
            "target_tokens": validation_targets[bucket.name],
            "actual_tokens": validation_tokens[bucket.name],
        }
        for bucket in config.buckets
    }
    shortfalls = {
        bucket: max(0, value["target_tokens"] - value["actual_tokens"])
        for bucket, value in train_checks.items()
        if value["actual_tokens"] < value["target_tokens"]
    }
    passed = (
        all(value["relative_error"] <= 0.01 for value in train_checks.values())
        and not (train_documents & validation_documents)
        and all(validation_tokens[name] >= target * 0.95 for name, target in validation_targets.items())
        and metrics.report()["passed"]
    )
    report = {
        "stage": "finalize",
        "phase": config.phase,
        "run_id": config.run_id,
        "passed": passed,
        "plan_sha256": plan["plan_sha256"],
        "distribution": metrics.report(),
        "coverage_is_diagnostic": True,
        "training_character_coverage": {
            str(goal): {
                "actual_characters": sum(character_df[c] >= goal for c in new_characters),
                "total_characters": len(new_characters),
                "target_fraction": fraction,
            }
            for goal, fraction in config.enhancement.coverage_targets.items()
        },
        "source_shortfalls": {
            b.name: {
                source: max(0, int(round(config.bucket_tokens[b.name] * weight)) - metrics.sources[b.name][source])
                for source, weight in b.source_weights.items()
            } for b in config.buckets
        },
        "tokenizer_sha256": config.hashes.tokenizer,
        "train": train_checks,
        "validation": validation_checks,
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents),
        "document_overlap": len(train_documents & validation_documents),
        "exact_shortfalls": shortfalls,
        "packing": {
            str(length): {"tokens_including_eos": packer.total_tokens}
            for length, packer in packers.items()
        },
        "ms_swift": {
            "format": "assistant-only messages JSONL",
            "train": str(final_root / "ms_swift" / "train"),
            "validation": str(final_root / "ms_swift" / "validation"),
            "requires_swift_cached_export": True,
        },
        "output": str(final_root),
    }
    _write_json(config.run_root / "reports" / "finalization_report.json", report)
    _write_json(config.run_root / "plans" / "exact_shortfalls.json", {"shortfalls": shortfalls})
    return report
