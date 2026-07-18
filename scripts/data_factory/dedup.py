"""Disk-backed exact and MinHash near-duplicate removal."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.io_utils import JsonlShardWriter, expand_paths, iter_jsonl, utc_now_iso, write_json
from scripts.data_factory.text import code_shingles, content_hash, dedup_kind, natural_shingles


class DedupIndex:
    def __init__(self, path: Path, config: PipelineConfig) -> None:
        try:
            from datasketch import MinHash
        except ImportError as error:
            raise RuntimeError("datasketch is required for near-duplicate removal") from error

        self.minhash_class = MinHash
        self.config = config
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL UNIQUE,
                signature BLOB,
                num_perm INTEGER NOT NULL,
                kind TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT NOT NULL,
                phase TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lsh (
                kind TEXT NOT NULL,
                band INTEGER NOT NULL,
                bucket BLOB NOT NULL,
                doc_id TEXT NOT NULL,
                PRIMARY KEY (kind, band, bucket, doc_id)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS lsh_lookup ON lsh(kind, band, bucket);
            """
        )
        self.connection.commit()
        self.pending = 0

    def _settings(self, kind: str) -> tuple[int, int, float]:
        if kind == "code":
            return (
                int(self.config.dedup.get("code_num_perm", 256)),
                int(self.config.dedup.get("code_band_size", 4)),
                float(self.config.dedup.get("code_threshold", 0.85)),
            )
        return (
            int(self.config.dedup.get("natural_num_perm", 128)),
            int(self.config.dedup.get("natural_band_size", 4)),
            float(self.config.dedup.get("natural_threshold", 0.80)),
        )

    def _signature(self, text: str, kind: str, num_perm: int) -> np.ndarray | None:
        shingles: Iterable[bytes] = code_shingles(text) if kind == "code" else natural_shingles(text)
        minhash = self.minhash_class(num_perm=num_perm, seed=int(self.config.seed & 0xFFFFFFFF))
        count = 0
        for shingle in shingles:
            minhash.update(shingle)
            count += 1
        return minhash.hashvalues if count else None

    @staticmethod
    def _buckets(signature: np.ndarray, band_size: int) -> Iterable[tuple[int, bytes]]:
        if len(signature) % band_size:
            raise ValueError("MinHash permutation count must be divisible by band_size")
        for band, start in enumerate(range(0, len(signature), band_size)):
            payload = signature[start : start + band_size].tobytes()
            yield band, hashlib.blake2b(payload, digest_size=16).digest()

    def find_duplicate(
        self, row: dict[str, Any]
    ) -> tuple[str | None, str | None, str, str, np.ndarray | None, int, float | None]:
        digest = content_hash(str(row["text"]))
        doc_id = str(row["doc_id"])
        existing_doc_id = self.connection.execute(
            "SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if existing_doc_id is not None:
            reason = "exact" if str(existing_doc_id[0]) == digest else "doc_id_conflict"
            similarity = 1.0 if reason == "exact" else None
            return reason, doc_id, digest, dedup_kind(row), None, 0, similarity
        exact = self.connection.execute(
            "SELECT doc_id, phase FROM documents WHERE content_hash = ?", (digest,)
        ).fetchone()
        if exact is not None:
            reason = "contamination_exact" if exact[1] == "evaluation" else "exact"
            return reason, str(exact[0]), digest, dedup_kind(row), None, 0, 1.0

        kind = dedup_kind(row)
        num_perm, band_size, threshold = self._settings(kind)
        if len(str(row["text"])) < int(self.config.dedup.get("near_min_chars", 50)):
            return None, None, digest, kind, None, 0, None
        signature = self._signature(str(row["text"]), kind, num_perm)
        if signature is None:
            return None, None, digest, kind, None, 0, None

        candidate_limit = int(self.config.dedup.get("candidate_limit_per_bucket", 2000))
        candidate_ids: set[str] = set()
        for band, bucket in self._buckets(signature, band_size):
            rows = self.connection.execute(
                "SELECT doc_id FROM lsh WHERE kind = ? AND band = ? AND bucket = ? LIMIT ?",
                (kind, band, bucket, candidate_limit),
            )
            candidate_ids.update(str(value[0]) for value in rows)

        for candidate_id in sorted(candidate_ids):
            stored = self.connection.execute(
                "SELECT signature, num_perm, phase FROM documents WHERE doc_id = ?", (candidate_id,)
            ).fetchone()
            if stored is None or stored[0] is None or int(stored[1]) != num_perm:
                continue
            candidate_signature = np.frombuffer(stored[0], dtype=np.uint64)
            similarity = float(np.count_nonzero(signature == candidate_signature) / num_perm)
            if similarity >= threshold:
                reason = "contamination_near" if stored[2] == "evaluation" else "near"
                return reason, candidate_id, digest, kind, signature, num_perm, similarity
        return None, None, digest, kind, signature, num_perm, None

    def insert(
        self,
        row: dict[str, Any],
        digest: str,
        kind: str,
        signature: np.ndarray | None,
        num_perm: int,
        phase: str = "phase1",
    ) -> None:
        doc_id = str(row["doc_id"])
        self.connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                digest,
                None if signature is None else signature.tobytes(),
                num_perm,
                kind,
                str(row["category"]),
                str(row["source"]),
                phase,
            ),
        )
        if signature is not None:
            _, band_size, _ = self._settings(kind)
            self.connection.executemany(
                "INSERT INTO lsh(kind, band, bucket, doc_id) VALUES (?, ?, ?, ?)",
                ((kind, band, bucket, doc_id) for band, bucket in self._buckets(signature, band_size)),
            )
        self.pending += 1
        if self.pending >= 1000:
            self.connection.commit()
            self.pending = 0

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def preload_exclusions(index: DedupIndex, paths: list[Path]) -> int:
    inserted = 0
    for row in iter_jsonl(paths):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        digest = content_hash(text)
        candidate = {
            "text": text,
            "doc_id": f"evaluation-{digest}",
            "category": "evaluation",
            "source": str(row.get("source", "evaluation")),
        }
        reason, _, digest, kind, signature, num_perm, _ = index.find_duplicate(candidate)
        if reason is None:
            index.insert(candidate, digest, kind, signature, num_perm, phase="evaluation")
            inserted += 1
    return inserted


def export_registry(database_path: Path, output_path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required to export the dedup registry") from error

    schema = pa.schema(
        [
            ("doc_id", pa.string()),
            ("content_hash", pa.string()),
            ("signature", pa.binary()),
            ("num_perm", pa.int32()),
            ("kind", pa.string()),
            ("category", pa.string()),
            ("source", pa.string()),
            ("phase", pa.string()),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    cursor = connection.execute(
        "SELECT doc_id, content_hash, signature, num_perm, kind, category, source, phase FROM documents ORDER BY doc_id"
    )
    writer = parquet.ParquetWriter(output_path, schema, compression="zstd")
    try:
        while True:
            rows = cursor.fetchmany(10_000)
            if not rows:
                break
            writer.write_table(pa.Table.from_pylist([dict(zip(schema.names, row, strict=True)) for row in rows], schema=schema))
    finally:
        writer.close()
        connection.close()


def deduplicate(config: PipelineConfig, overwrite: bool = False) -> dict[str, Any]:
    paths = sorted(config.normalized_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no normalized JSONL files found under {config.normalized_dir}")
    config.deduplicated_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    database_path = config.deduplicated_dir / "dedup_registry.sqlite"
    existing_outputs = list(config.deduplicated_dir.glob("part-*.jsonl"))
    if (database_path.exists() or existing_outputs) and not overwrite:
        raise FileExistsError(f"deduplicated outputs already exist under {config.deduplicated_dir}; pass --overwrite")
    if overwrite:
        database_path.unlink(missing_ok=True)
        database_path.with_name(database_path.name + "-wal").unlink(missing_ok=True)
        database_path.with_name(database_path.name + "-shm").unlink(missing_ok=True)
        for path in existing_outputs:
            path.unlink()

    removed_path = config.reports_dir / "dedup_removed.jsonl"
    removed_file = removed_path.open("wt", encoding="utf-8", newline="\n")
    writer = JsonlShardWriter(config.deduplicated_dir, "part", config.deduplicated_shard_records)
    index = DedupIndex(database_path, config)
    exclusion_paths = expand_paths(config.dedup.get("decontamination_paths", []), config.repo_root)
    exclusion_records = preload_exclusions(index, exclusion_paths)
    reasons: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    input_count = 0
    try:
        for row in iter_jsonl(paths):
            input_count += 1
            source = str(row.get("source", "unknown"))
            source_stats = by_source.setdefault(source, Counter())
            reason, duplicate_of, digest, kind, signature, num_perm, similarity = index.find_duplicate(row)
            if reason is not None:
                reasons[reason] += 1
                source_stats[f"removed_{reason}"] += 1
                removed_file.write(
                    json.dumps(
                        {
                            "doc_id": row.get("doc_id"),
                            "duplicate_of": duplicate_of,
                            "reason": reason,
                            "similarity": similarity,
                            "source": source,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                continue
            row["content_hash"] = digest
            index.insert(row, digest, kind, signature, num_perm)
            writer.write(row)
            source_stats["kept"] += 1
    finally:
        index.close()
        writer.close()
        removed_file.close()

    registry_path = config.deduplicated_dir / "dedup_registry.parquet"
    export_registry(database_path, registry_path)
    report = {
        "generated_at": utc_now_iso(),
        "input_records": input_count,
        "kept_records": writer.total_records,
        "removed_records": input_count - writer.total_records,
        "removed_by_reason": dict(sorted(reasons.items())),
        "decontamination_files": [str(path) for path in exclusion_paths],
        "decontamination_records": exclusion_records,
        "by_source": {source: dict(sorted(stats.items())) for source, stats in sorted(by_source.items())},
        "output_files": writer.files,
        "removed_log": str(removed_path),
        "registry_database": str(database_path),
        "registry_parquet": str(registry_path),
    }
    write_json(config.reports_dir / "dedup_report.json", report)
    return report

