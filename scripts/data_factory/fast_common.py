"""Shared helpers for the fast Phase 1 data pipeline.

The fast pipeline deliberately keeps expensive tokenization out of corpus-wide
filtering.  Raw sources are normalized once into Parquet, token statistics are
calibrated on a small deterministic sample, and only the oversampled candidate
pool is tokenized exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data_factory.io_utils import utc_now_iso


FAST_PIPELINE_VERSION = "phase1_fast_v1"


def stable_key(seed: int, namespace: str, doc_id: str) -> int:
    payload = f"{seed}:{namespace}:{doc_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def stable_fraction(seed: int, namespace: str, doc_id: str) -> float:
    return stable_key(seed, namespace, doc_id) / float(1 << 64)


def short_content_hash(text: str) -> bytes:
    """Return a compact exact-dedup key for already-normalized text."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def parquet_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def iter_parquet_batches(
    paths: Iterable[Path],
    *,
    columns: list[str] | None = None,
    batch_size: int = 4096,
) -> Iterator[pa.RecordBatch]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        yield from parquet.iter_batches(batch_size=batch_size, columns=columns)


class PartitionedParquetWriter:
    """Small append-only Parquet shard writer partitioned by a string key."""

    def __init__(
        self,
        root: Path,
        schema: pa.Schema,
        *,
        prefix: str,
        max_rows: int,
        compression: str = "zstd",
    ) -> None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self.root = root
        self.schema = schema
        self.prefix = prefix
        self.max_rows = max_rows
        self.compression = compression
        self.root.mkdir(parents=True, exist_ok=True)
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._indices: dict[str, int] = {}
        self.files: list[dict[str, Any]] = []
        self.total_rows = 0

    @staticmethod
    def _safe_partition(value: str) -> str:
        result = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
        return result.strip("_") or "unknown"

    def write(self, partition: str, row: dict[str, Any]) -> None:
        key = self._safe_partition(partition)
        buffer = self._buffers.setdefault(key, [])
        buffer.append(row)
        self.total_rows += 1
        if len(buffer) >= self.max_rows:
            self.flush(key)

    def flush(self, partition: str) -> None:
        key = self._safe_partition(partition)
        rows = self._buffers.get(key)
        if not rows:
            return
        index = self._indices.get(key, 0)
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.prefix}-{index:05d}.parquet"
        table = pa.Table.from_pylist(rows, schema=self.schema)
        pq.write_table(
            table,
            path,
            compression=self.compression,
            use_dictionary=True,
            write_statistics=True,
        )
        self.files.append(
            {
                "path": str(path),
                "partition": key,
                "rows": len(rows),
                "bytes": path.stat().st_size,
            }
        )
        self._indices[key] = index + 1
        rows.clear()

    def close(self) -> None:
        for key in list(self._buffers):
            self.flush(key)

    def __enter__(self) -> "PartitionedParquetWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class TokenEstimator:
    """Per-source linear estimator fitted by ``calibrate_tokens``."""

    source_coefficients: dict[str, tuple[float, float, float, float, float]]
    source_ratios: dict[str, float]
    global_ratio: float

    @classmethod
    def from_report(cls, report: dict[str, Any]) -> "TokenEstimator":
        sources = report.get("sources", {})
        coefficients: dict[str, tuple[float, float, float, float, float]] = {}
        ratios: dict[str, float] = {}
        for source, stats in sources.items():
            raw_coeffs = stats.get("linear_coefficients")
            if isinstance(raw_coeffs, list) and len(raw_coeffs) == 5:
                coefficients[str(source)] = tuple(float(value) for value in raw_coeffs)  # type: ignore[assignment]
            ratio = stats.get("tokens_per_char")
            if ratio is not None:
                ratios[str(source)] = float(ratio)
        global_ratio = float(report.get("global", {}).get("tokens_per_char", 1.0))
        return cls(coefficients, ratios, global_ratio)

    def estimate(
        self,
        *,
        source: str,
        char_count: int,
        hanzi_count: int,
        latin_count: int,
        digit_count: int,
    ) -> int:
        char_count = max(0, int(char_count))
        hanzi_count = max(0, int(hanzi_count))
        latin_count = max(0, int(latin_count))
        digit_count = max(0, int(digit_count))
        other_count = max(0, char_count - hanzi_count - latin_count - digit_count)
        coeffs = self.source_coefficients.get(source)
        estimate: float
        if coeffs is not None:
            values = (hanzi_count, latin_count, digit_count, other_count, 1)
            estimate = sum(weight * value for weight, value in zip(coeffs, values, strict=True))
            # A badly conditioned fit must never poison sampling rates.
            if not math.isfinite(estimate) or estimate <= 0:
                coeffs = None
        if coeffs is None:
            ratio = self.source_ratios.get(source, self.global_ratio)
            estimate = max(1.0, char_count * ratio)
        return max(1, int(round(estimate)))


def write_fast_report(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    from scripts.data_factory.io_utils import write_json

    value = {
        "pipeline_version": FAST_PIPELINE_VERSION,
        "generated_at": utc_now_iso(),
        **payload,
    }
    write_json(path, value)
    return value
