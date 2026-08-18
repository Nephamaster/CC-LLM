"""Deterministic exact and MinHash deduplication with parallel feature computation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.io_utils import expand_paths, iter_jsonl, utc_now_iso, write_json
from scripts.data_factory.text import code_shingles, content_hash, dedup_kind, natural_shingles


DEDUP_STATE_VERSION = 2
_WORKER_DEDUP_CONFIG: dict[str, Any] | None = None
_WORKER_SEED = 0


def _settings(dedup_config: dict[str, Any], kind: str) -> tuple[int, int, float]:
    if kind == "code":
        return (
            int(dedup_config.get("code_num_perm", 256)),
            int(dedup_config.get("code_band_size", 4)),
            float(dedup_config.get("code_threshold", 0.85)),
        )
    return (
        int(dedup_config.get("natural_num_perm", 128)),
        int(dedup_config.get("natural_band_size", 4)),
        float(dedup_config.get("natural_threshold", 0.80)),
    )


def _signature(text: str, kind: str, num_perm: int, seed: int) -> bytes | None:
    try:
        from datasketch import MinHash
    except ImportError as error:
        raise RuntimeError("datasketch is required for near-duplicate removal") from error

    shingles: Iterable[bytes] = code_shingles(text) if kind == "code" else natural_shingles(text)
    minhash = MinHash(num_perm=num_perm, seed=int(seed & 0xFFFFFFFF))
    def update(values: list[bytes]) -> None:
        update_batch = getattr(minhash, "update_batch", None)
        if callable(update_batch):
            update_batch(values)
        else:
            for value in values:
                minhash.update(value)

    buffer: list[bytes] = []
    count = 0
    for shingle in shingles:
        buffer.append(shingle)
        count += 1
        if len(buffer) >= 4096:
            update(buffer)
            buffer.clear()
    if buffer:
        update(buffer)
    return minhash.hashvalues.tobytes() if count else None


def _feature(row: dict[str, Any], dedup_config: dict[str, Any], seed: int) -> dict[str, Any]:
    text = str(row["text"])
    kind = dedup_kind(row)
    digest = content_hash(text)
    num_perm, _, _ = _settings(dedup_config, kind)
    signature = None
    if len(text) >= int(dedup_config.get("near_min_chars", 50)):
        signature = _signature(text, kind, num_perm, seed)
    return {
        "digest": digest,
        "kind": kind,
        "signature": signature,
        "num_perm": num_perm if signature is not None else 0,
    }


def _init_feature_worker(dedup_config: dict[str, Any], seed: int) -> None:
    global _WORKER_DEDUP_CONFIG, _WORKER_SEED
    _WORKER_DEDUP_CONFIG = dedup_config
    _WORKER_SEED = seed


def _compute_feature_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if _WORKER_DEDUP_CONFIG is None:
        raise RuntimeError("dedup feature worker is not initialized")
    return [_feature(row, _WORKER_DEDUP_CONFIG, _WORKER_SEED) for row in rows]


class DedupIndex:
    def __init__(self, path: Path, config: PipelineConfig) -> None:
        self.config = config
        self.connection = sqlite3.connect(path)
        cache_mb = max(64, int(config.dedup.get("sqlite_cache_mb", 1024)))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute(f"PRAGMA cache_size=-{cache_mb * 1024}")
        self.connection.execute(f"PRAGMA mmap_size={cache_mb * 1024 * 1024}")
        self.connection.execute("PRAGMA busy_timeout=60000")
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
                phase TEXT NOT NULL,
                input_shard TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lsh (
                kind TEXT NOT NULL,
                band INTEGER NOT NULL,
                bucket BLOB NOT NULL,
                doc_id TEXT NOT NULL,
                PRIMARY KEY (kind, band, bucket, doc_id)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS lsh_lookup ON lsh(kind, band, bucket);
            CREATE INDEX IF NOT EXISTS documents_input_shard ON documents(input_shard);
            """
        )
        self.connection.commit()
        self.pending = 0
        self._candidate_query_cache: dict[int, str] = {}
        self.commit_interval = max(1000, int(config.dedup.get("commit_interval", 5000)))

    def _settings(self, kind: str) -> tuple[int, int, float]:
        return _settings(self.config.dedup, kind)

    @staticmethod
    def _buckets(signature: np.ndarray, band_size: int) -> Iterator[tuple[int, bytes]]:
        if len(signature) % band_size:
            raise ValueError("MinHash permutation count must be divisible by band_size")
        for band, start in enumerate(range(0, len(signature), band_size)):
            payload = signature[start : start + band_size].tobytes()
            yield band, hashlib.blake2b(payload, digest_size=16).digest()

    def _candidate_ids(
        self,
        kind: str,
        signature: np.ndarray,
        band_size: int,
        candidate_limit: int,
    ) -> set[str]:
        probes = list(self._buckets(signature, band_size))
        query = self._candidate_query_cache.get(len(probes))
        if query is None:
            part = (
                "SELECT doc_id FROM ("
                "SELECT doc_id FROM lsh WHERE kind = ? AND band = ? AND bucket = ? LIMIT ?"
                ")"
            )
            query = "SELECT DISTINCT doc_id FROM (" + " UNION ALL ".join(
                part for _ in probes
            ) + ")"
            self._candidate_query_cache[len(probes)] = query
        parameters: list[Any] = []
        for band, bucket in probes:
            parameters.extend((kind, band, bucket, candidate_limit))
        return {str(row[0]) for row in self.connection.execute(query, parameters)}

    def _candidate_signatures(self, candidate_ids: set[str]) -> dict[str, tuple[bytes | None, int, str]]:
        values: dict[str, tuple[bytes | None, int, str]] = {}
        ordered = sorted(candidate_ids)
        for start in range(0, len(ordered), 800):
            chunk = ordered[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT doc_id, signature, num_perm, phase FROM documents WHERE doc_id IN ({placeholders})",
                chunk,
            )
            for doc_id, signature, num_perm, phase in rows:
                values[str(doc_id)] = (signature, int(num_perm), str(phase))
        return values

    def find_duplicate(
        self,
        row: dict[str, Any],
        feature: dict[str, Any],
    ) -> tuple[str | None, str | None, float | None]:
        digest = str(feature["digest"])
        doc_id = str(row["doc_id"])
        existing_doc_id = self.connection.execute(
            "SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if existing_doc_id is not None:
            reason = "exact" if str(existing_doc_id[0]) == digest else "doc_id_conflict"
            return reason, doc_id, 1.0 if reason == "exact" else None

        exact = self.connection.execute(
            "SELECT doc_id, phase FROM documents WHERE content_hash = ?", (digest,)
        ).fetchone()
        if exact is not None:
            reason = "contamination_exact" if exact[1] == "evaluation" else "exact"
            return reason, str(exact[0]), 1.0

        signature_bytes = feature.get("signature")
        if signature_bytes is None:
            return None, None, None
        kind = str(feature["kind"])
        num_perm = int(feature["num_perm"])
        signature = np.frombuffer(signature_bytes, dtype=np.uint64)
        _, band_size, threshold = self._settings(kind)
        candidate_limit = int(self.config.dedup.get("candidate_limit_per_bucket", 2000))
        candidate_ids = self._candidate_ids(
            kind,
            signature,
            band_size,
            candidate_limit,
        )

        candidates = self._candidate_signatures(candidate_ids)
        for candidate_id in sorted(candidate_ids):
            stored = candidates.get(candidate_id)
            if stored is None or stored[0] is None or stored[1] != num_perm:
                continue
            candidate_signature = np.frombuffer(stored[0], dtype=np.uint64)
            similarity = float(np.count_nonzero(signature == candidate_signature) / num_perm)
            if similarity >= threshold:
                reason = "contamination_near" if stored[2] == "evaluation" else "near"
                return reason, candidate_id, similarity
        return None, None, None

    def insert(
        self,
        row: dict[str, Any],
        feature: dict[str, Any],
        *,
        phase: str,
        input_shard: str,
    ) -> None:
        signature_bytes = feature.get("signature")
        num_perm = int(feature["num_perm"])
        kind = str(feature["kind"])
        doc_id = str(row["doc_id"])
        self.connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                str(feature["digest"]),
                signature_bytes,
                num_perm,
                kind,
                str(row["category"]),
                str(row["source"]),
                phase,
                input_shard,
            ),
        )
        if signature_bytes is not None:
            signature = np.frombuffer(signature_bytes, dtype=np.uint64)
            _, band_size, _ = self._settings(kind)
            self.connection.executemany(
                "INSERT INTO lsh(kind, band, bucket, doc_id) VALUES (?, ?, ?, ?)",
                ((kind, band, bucket, doc_id) for band, bucket in self._buckets(signature, band_size)),
            )
        self.pending += 1
        if self.pending >= self.commit_interval:
            self.commit()

    def input_shard_count(self, input_shard: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM documents WHERE input_shard = ?",
            (input_shard,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def remove_input_shard(self, input_shard: str) -> int:
        doc_ids = [
            str(row[0])
            for row in self.connection.execute(
                "SELECT doc_id FROM documents WHERE input_shard = ?", (input_shard,)
            )
        ]
        for start in range(0, len(doc_ids), 800):
            chunk = doc_ids[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            self.connection.execute(f"DELETE FROM lsh WHERE doc_id IN ({placeholders})", chunk)
            self.connection.execute(f"DELETE FROM documents WHERE doc_id IN ({placeholders})", chunk)
        self.commit()
        return len(doc_ids)

    def commit(self) -> None:
        self.connection.commit()
        self.pending = 0

    def close(self) -> None:
        self.commit()
        self.connection.close()


class JsonlFileWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.file = path.open("wb")
        self.digest = hashlib.sha256()
        self.records = 0

    def write(self, row: dict[str, Any]) -> None:
        payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.file.write(payload)
        self.digest.update(payload)
        self.records += 1

    def close(self) -> dict[str, Any]:
        if not self.file.closed:
            self.file.close()
        return {
            "path": str(self.path),
            "records": self.records,
            "bytes": self.path.stat().st_size,
            "sha256": self.digest.hexdigest(),
        }


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
        feature = _feature(candidate, index.config.dedup, index.config.seed)
        reason, _, _ = index.find_duplicate(candidate, feature)
        if reason is None:
            index.insert(candidate, feature, phase="evaluation", input_shard="__evaluation__")
            inserted += 1
    index.commit()
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
        "SELECT doc_id, content_hash, signature, num_perm, kind, category, source, phase "
        "FROM documents ORDER BY doc_id"
    )
    writer = parquet.ParquetWriter(output_path, schema, compression="zstd")
    try:
        while True:
            rows = cursor.fetchmany(10_000)
            if not rows:
                break
            writer.write_table(
                pa.Table.from_pylist(
                    [dict(zip(schema.names, row, strict=True)) for row in rows],
                    schema=schema,
                )
            )
    finally:
        writer.close()
        connection.close()


def _input_metadata(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]


def _run_fingerprint(config: PipelineConfig, paths: list[Path], exclusion_paths: list[Path]) -> str:
    performance_keys = {
        "batch_size",
        "batch_chars",
        "sqlite_cache_mb",
        "commit_interval",
        "export_registry_parquet",
    }
    payload = {
        "version": DEDUP_STATE_VERSION,
        "seed": config.seed,
        "dedup": {key: value for key, value in config.dedup.items() if key not in performance_keys},
        "inputs": _input_metadata(paths),
        "exclusions": _input_metadata(exclusion_paths),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iter_batches(path: Path, max_records: int, max_chars: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    characters = 0
    for row in iter_jsonl([path]):
        row_chars = len(str(row.get("text", "")))
        if batch and (len(batch) >= max_records or characters + row_chars > max_chars):
            yield batch
            batch = []
            characters = 0
        batch.append(row)
        characters += row_chars
    if batch:
        yield batch


def _apply_feature_batch(
    index: DedupIndex,
    input_shard: str,
    rows: list[dict[str, Any]],
    features: list[dict[str, Any]],
    output_writer: JsonlFileWriter,
    removed_writer: JsonlFileWriter,
    reasons: Counter[str],
    by_source: dict[str, Counter[str]],
) -> None:
    if len(rows) != len(features):
        raise RuntimeError("dedup worker returned a mismatched feature batch")
    for row, feature in zip(rows, features, strict=True):
        source = str(row.get("source", "unknown"))
        source_stats = by_source.setdefault(source, Counter())
        reason, duplicate_of, similarity = index.find_duplicate(row, feature)
        if reason is not None:
            reasons[reason] += 1
            source_stats[f"removed_{reason}"] += 1
            removed_writer.write(
                {
                    "doc_id": row.get("doc_id"),
                    "duplicate_of": duplicate_of,
                    "reason": reason,
                    "similarity": similarity,
                    "source": source,
                }
            )
            continue
        row["content_hash"] = feature["digest"]
        index.insert(row, feature, phase="phase1", input_shard=input_shard)
        output_writer.write(row)
        source_stats["kept"] += 1


def _process_shard(
    index: DedupIndex,
    executor: ProcessPoolExecutor | None,
    input_path: Path,
    output_path: Path,
    removed_path: Path,
    config: PipelineConfig,
    workers: int,
) -> dict[str, Any]:
    input_shard = str(input_path.resolve())
    index.remove_input_shard(input_shard)
    output_temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    removed_temporary = removed_path.with_suffix(removed_path.suffix + ".tmp")
    output_temporary.unlink(missing_ok=True)
    removed_temporary.unlink(missing_ok=True)
    output_writer = JsonlFileWriter(output_temporary)
    removed_writer = JsonlFileWriter(removed_temporary)
    reasons: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    input_count = 0
    max_records = max(1, int(config.dedup.get("batch_size", 128)))
    max_chars = max(1, int(config.dedup.get("batch_chars", 1_000_000)))

    def consume(rows: list[dict[str, Any]], features: list[dict[str, Any]]) -> None:
        nonlocal input_count
        input_count += len(rows)
        _apply_feature_batch(
            index,
            input_shard,
            rows,
            features,
            output_writer,
            removed_writer,
            reasons,
            by_source,
        )

    try:
        if executor is None:
            for rows in _iter_batches(input_path, max_records, max_chars):
                consume(rows, [_feature(row, config.dedup, config.seed) for row in rows])
        else:
            pending: deque[tuple[Future[list[dict[str, Any]]], list[dict[str, Any]]]] = deque()
            max_in_flight = max(2, workers * 2)
            for rows in _iter_batches(input_path, max_records, max_chars):
                pending.append((executor.submit(_compute_feature_batch, rows), rows))
                if len(pending) >= max_in_flight:
                    future, submitted_rows = pending.popleft()
                    consume(submitted_rows, future.result())
            while pending:
                future, submitted_rows = pending.popleft()
                consume(submitted_rows, future.result())
        index.commit()
        output_file = output_writer.close()
        removed_file = removed_writer.close()
        os.replace(output_temporary, output_path)
        os.replace(removed_temporary, removed_path)
        output_file.update({"path": str(output_path), "bytes": output_path.stat().st_size})
        removed_file.update({"path": str(removed_path), "bytes": removed_path.stat().st_size})
    except BaseException:
        output_writer.close()
        removed_writer.close()
        raise

    return {
        "input_path": input_shard,
        "input_size": input_path.stat().st_size,
        "input_mtime_ns": input_path.stat().st_mtime_ns,
        "input": input_count,
        "kept": output_file["records"],
        "removed": input_count - int(output_file["records"]),
        "reasons": dict(sorted(reasons.items())),
        "by_source": {
            source: dict(sorted(stats.items())) for source, stats in sorted(by_source.items())
        },
        "output_file": output_file,
        "removed_file": removed_file,
    }


def _valid_completed_result(result: dict[str, Any], input_path: Path) -> bool:
    if result.get("input_path") != str(input_path.resolve()):
        return False
    if int(result.get("input_size", -1)) != input_path.stat().st_size:
        return False
    if int(result.get("input_mtime_ns", -1)) != input_path.stat().st_mtime_ns:
        return False
    for key in ("output_file", "removed_file"):
        item = result.get(key, {})
        path = Path(str(item.get("path", "")))
        if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
            return False
    return True


def _valid_artifact(item: dict[str, Any] | None) -> bool:
    if not isinstance(item, dict):
        return False
    path = Path(str(item.get("path", "")))
    return path.is_file() and path.stat().st_size == int(item.get("bytes", -1))


def _merge_removed_logs(results: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as target:
        for result in results:
            with Path(result["removed_file"]["path"]).open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
                    digest.update(chunk)
    os.replace(temporary, output_path)
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def deduplicate(
    config: PipelineConfig,
    overwrite: bool = False,
    *,
    resume: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    if overwrite and resume:
        raise ValueError("overwrite and resume cannot be used together")
    paths = sorted(config.normalized_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no normalized JSONL files found under {config.normalized_dir}")
    config.deduplicated_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    database_path = config.deduplicated_dir / "dedup_registry.sqlite"
    state_path = config.deduplicated_dir / "dedup_state.json"
    registry_path = config.deduplicated_dir / "dedup_registry.parquet"
    removed_root = config.reports_dir / ".dedup_removed"
    removed_path = config.reports_dir / "dedup_removed.jsonl"
    existing_outputs = sorted(config.deduplicated_dir.glob("part-*.jsonl"))
    exclusion_paths = expand_paths(config.dedup.get("decontamination_paths", []), config.repo_root)
    run_fingerprint = _run_fingerprint(config, paths, exclusion_paths)

    if overwrite:
        for path in existing_outputs:
            path.unlink()
        for path in (database_path, state_path, registry_path, removed_path):
            path.unlink(missing_ok=True)
        database_path.with_name(database_path.name + "-wal").unlink(missing_ok=True)
        database_path.with_name(database_path.name + "-shm").unlink(missing_ok=True)
        shutil.rmtree(removed_root, ignore_errors=True)
    elif resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"dedup resume state is missing: {state_path}; use --overwrite")
    elif database_path.exists() or state_path.exists() or existing_outputs:
        raise FileExistsError(
            f"deduplicated outputs already exist under {config.deduplicated_dir}; "
            "pass --overwrite or --resume"
        )

    if resume:
        with state_path.open("rt", encoding="utf-8") as file:
            state = json.load(file)
        if state.get("version") != DEDUP_STATE_VERSION:
            raise RuntimeError("dedup state schema changed; restart with --overwrite")
        if state.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("dedup inputs or semantic configuration changed; restart with --overwrite")
    else:
        state = {
            "version": DEDUP_STATE_VERSION,
            "run_fingerprint": run_fingerprint,
            "generated_at": utc_now_iso(),
            "registry_initialized": False,
            "exclusion_records": 0,
            "completed": [],
        }
        write_json(state_path, state)

    completed: list[dict[str, Any]] = list(state.get("completed", []))
    if len(completed) > len(paths):
        raise RuntimeError("dedup resume state contains more shards than the current input")
    for index, result in enumerate(completed):
        if not _valid_completed_result(result, paths[index]):
            raise RuntimeError(
                f"completed dedup shard {index} is missing or changed; restart with --overwrite"
            )

    if not state.get("registry_initialized"):
        for path in (
            database_path,
            database_path.with_name(database_path.name + "-wal"),
            database_path.with_name(database_path.name + "-shm"),
        ):
            path.unlink(missing_ok=True)
        index = DedupIndex(database_path, config)
        exclusion_records = preload_exclusions(index, exclusion_paths)
        state["registry_initialized"] = True
        state["exclusion_records"] = exclusion_records
        write_json(state_path, state)
    else:
        if not database_path.is_file():
            raise RuntimeError("dedup registry is missing; restart with --overwrite")
        index = DedupIndex(database_path, config)
        for result in completed:
            input_shard = str(result["input_path"])
            if index.input_shard_count(input_shard) != int(result["kept"]):
                index.close()
                raise RuntimeError(
                    f"dedup registry is inconsistent for {input_shard}; restart with --overwrite"
                )

    worker_count = workers or config.dedup_workers
    if worker_count <= 0:
        index.close()
        raise ValueError("workers must be positive")
    removed_root.mkdir(parents=True, exist_ok=True)
    executor = None
    if worker_count > 1:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_feature_worker,
            initargs=(config.dedup, config.seed),
        )

    try:
        for shard_index in range(len(completed), len(paths)):
            input_path = paths[shard_index]
            output_path = config.deduplicated_dir / f"part-{shard_index:05d}.jsonl"
            shard_removed_path = removed_root / f"part-{shard_index:05d}.jsonl"
            result = _process_shard(
                index,
                executor,
                input_path,
                output_path,
                shard_removed_path,
                config,
                worker_count,
            )
            completed.append(result)
            state["completed"] = completed
            state.pop("removed_log", None)
            state.pop("registry_parquet", None)
            state["updated_at"] = utc_now_iso()
            write_json(state_path, state)
            print(
                f"deduplicated {shard_index + 1}/{len(paths)}: {input_path.name} "
                f"(kept {result['kept']:,}, removed {result['removed']:,})",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        index.close()

    removed_artifact = state.get("removed_log")
    if not _valid_artifact(removed_artifact):
        removed_artifact = _merge_removed_logs(completed, removed_path)
        state["removed_log"] = removed_artifact

    registry_output: str | None = None
    if bool(config.dedup.get("export_registry_parquet", True)):
        registry_artifact = state.get("registry_parquet")
        if not _valid_artifact(registry_artifact):
            export_registry(database_path, registry_path)
            registry_artifact = {
                "path": str(registry_path),
                "bytes": registry_path.stat().st_size,
            }
            state["registry_parquet"] = registry_artifact
        registry_output = str(registry_artifact["path"])
    state["updated_at"] = utc_now_iso()
    write_json(state_path, state)

    reasons: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    for result in completed:
        reasons.update(result.get("reasons", {}))
        for source, stats in result.get("by_source", {}).items():
            by_source.setdefault(source, Counter()).update(stats)
    input_count = sum(int(result["input"]) for result in completed)
    kept_count = sum(int(result["kept"]) for result in completed)
    report = {
        "generated_at": utc_now_iso(),
        "input_records": input_count,
        "kept_records": kept_count,
        "removed_records": input_count - kept_count,
        "removed_by_reason": dict(sorted(reasons.items())),
        "decontamination_files": [str(path) for path in exclusion_paths],
        "decontamination_records": int(state["exclusion_records"]),
        "workers": worker_count,
        "by_source": {
            source: dict(sorted(stats.items())) for source, stats in sorted(by_source.items())
        },
        "output_files": [result["output_file"] for result in completed],
        "removed_log": str(removed_path),
        "removed_shard_logs": [result["removed_file"] for result in completed],
        "registry_database": str(database_path),
        "registry_parquet": registry_output,
        "resume_state": str(state_path),
    }
    write_json(config.reports_dir / "dedup_report.json", report)
    return report
