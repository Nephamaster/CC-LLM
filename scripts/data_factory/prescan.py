"""Lightweight Phase 1 corpus scan and document preselection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from scripts.data_factory.candidate_features import (
    ALIGNMENT_POOLS,
    AlignmentFeatureMatcher,
    CandidateClassifier,
    CandidateSkip,
)
from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.io_utils import file_sha256, utc_now_iso
from scripts.data_factory.selection import (
    ALIGNMENT_VALIDATION_GROUPS,
    NATURAL_VALIDATION_WEIGHTS,
    scale_quotas,
)


PRESCAN_VERSION = "phase1_prescan_v1"
BRIDGE_CONTEXT_BUFFER_FACTOR = 2.0
BRIDGE_FILL_BUFFER_FACTOR = 2.0
_SCAN_WORKER: "LightweightScanner | None" = None
_BRIDGE_WORKER: "TopBridgeScanner | None" = None


def _stable_key(seed: int, namespace: str, doc_id: str) -> int:
    payload = f"{seed}:{namespace}:{doc_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & (
        (1 << 63) - 1
    )


def _stable_fraction(seed: int, namespace: str, doc_id: str) -> float:
    return _stable_key(seed, namespace, doc_id) / float(1 << 63)


def _alignment_eligible(pool: str, quota_group: str | None) -> bool:
    return pool in ALIGNMENT_POOLS or (
        pool == "specialized" and quota_group == "math_science"
    )


def _estimate_window_tokens(text: str, stats: dict[str, Any], target_tokens: int) -> int:
    hanzi = int(stats.get("hanzi", 0))
    non_hanzi = max(0, len(text) - hanzi)
    rough_tokens = hanzi + math.ceil(non_hanzi / 4)
    return max(1, min(target_tokens, rough_tokens))


def _open_worker_database(path: Path, schema: str) -> sqlite3.Connection:
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(schema)
    return connection


class LightweightScanner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.classifier = CandidateClassifier(config.quality)
        self.matcher = AlignmentFeatureMatcher(config)

    def scan(self, input_path: Path, output_path: Path) -> dict[str, Any]:
        connection = _open_worker_database(
            output_path,
            """
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                input_path TEXT NOT NULL,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                source TEXT NOT NULL,
                pool TEXT NOT NULL,
                quota_group TEXT,
                estimated_tokens INTEGER NOT NULL,
                sample_key INTEGER NOT NULL,
                output_key INTEGER NOT NULL,
                new_hanzi_hits TEXT NOT NULL
            );
            CREATE TABLE bridge_frequency (
                token_id INTEGER PRIMARY KEY,
                occurrences INTEGER NOT NULL,
                documents INTEGER NOT NULL
            );
            """,
        )
        bridge_occurrences: Counter[int] = Counter()
        bridge_documents: Counter[int] = Counter()
        skipped: Counter[str] = Counter()
        rows: list[tuple[Any, ...]] = []
        processed = 0
        accepted = 0

        try:
            with input_path.open("rb") as file:
                while True:
                    offset = file.tell()
                    raw_line = file.readline()
                    if not raw_line:
                        break
                    processed += 1
                    if not raw_line.strip():
                        skipped["empty_line"] += 1
                        continue
                    try:
                        row = json.loads(raw_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        skipped["invalid_json"] += 1
                        continue
                    if not isinstance(row, dict):
                        skipped["invalid_record"] += 1
                        continue
                    missing = [
                        key
                        for key in ("doc_id", "source", "category", "text")
                        if key not in row
                    ]
                    if missing:
                        skipped["invalid_record"] += 1
                        continue
                    if row.get("source") == "clue_benchmark":
                        skipped["excluded_clue"] += 1
                        continue
                    if row.get("synthetic"):
                        skipped["synthetic_alignment_text"] += 1
                        continue

                    text = str(row["text"])
                    try:
                        pool, quota_group, stats = self.classifier.classify(row, text)
                    except CandidateSkip as error:
                        skipped[error.reason] += 1
                        continue
                    except Exception:
                        skipped["classification_error"] += 1
                        continue

                    bridge_hits: Counter[int] = Counter()
                    new_hanzi_hits: Counter[str] = Counter()
                    if _alignment_eligible(pool, quota_group):
                        bridge_hits = self.matcher.bridge_hits(text)
                        new_hanzi_hits = self.matcher.new_hanzi_hits(text)
                        bridge_occurrences.update(bridge_hits)
                        bridge_documents.update(bridge_hits.keys())

                    doc_id = str(row["doc_id"])
                    rows.append(
                        (
                            doc_id,
                            str(input_path.resolve()),
                            offset,
                            len(raw_line),
                            str(row["source"]),
                            pool,
                            quota_group,
                            _estimate_window_tokens(
                                text,
                                stats,
                                self.config.windowing.target_tokens,
                            ),
                            _stable_key(self.config.seed, "prescan-sample", doc_id),
                            _stable_key(self.config.seed, "prescan-output", doc_id),
                            json.dumps(
                                dict(sorted(new_hanzi_hits.items())),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        )
                    )
                    accepted += 1
                    if len(rows) >= 2_000:
                        self._insert_documents(connection, rows)
                        rows.clear()

            if rows:
                self._insert_documents(connection, rows)
            connection.executemany(
                """
                INSERT INTO bridge_frequency(token_id, occurrences, documents)
                VALUES (?, ?, ?)
                """,
                [
                    (token_id, occurrences, bridge_documents[token_id])
                    for token_id, occurrences in bridge_occurrences.items()
                ],
            )
            connection.commit()
        finally:
            connection.close()

        return {
            "path": str(input_path),
            "processed_records": processed,
            "accepted_records": accepted,
            "skipped_records": sum(skipped.values()),
            "skipped_by_reason": dict(sorted(skipped.items())),
            "bridge_token_types": len(bridge_occurrences),
        }

    @staticmethod
    def _insert_documents(
        connection: sqlite3.Connection,
        rows: list[tuple[Any, ...]],
    ) -> None:
        connection.executemany(
            """
            INSERT OR IGNORE INTO documents (
                doc_id, input_path, byte_offset, byte_length, source, pool,
                quota_group, estimated_tokens, sample_key, output_key,
                new_hanzi_hits
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


class TopBridgeScanner:
    def __init__(
        self,
        config: PipelineConfig,
        token_document_frequency: dict[int, int],
        fill_probability: float,
    ) -> None:
        try:
            import ahocorasick
        except ImportError as error:
            raise RuntimeError("pyahocorasick is required for bridge scanning") from error

        with config.vocab_alignment.removed_multi_hanzi_tokens_path.open(
            "rt", encoding="utf-8"
        ) as file:
            removed = json.load(file)
        top_ids = set(token_document_frequency)
        automaton = ahocorasick.Automaton(ahocorasick.STORE_INTS)
        for row in removed:
            token_id = row.get("old_token_id") if isinstance(row, dict) else None
            token = row.get("token") if isinstance(row, dict) else None
            if token_id in top_ids and isinstance(token, str):
                automaton.add_word(token, int(token_id))
        automaton.make_automaton()

        self.config = config
        self.automaton = automaton
        self.token_document_frequency = token_document_frequency
        self.fill_probability = fill_probability
        self.context_target = math.ceil(
            config.vocab_alignment.bridge_min_contexts * BRIDGE_CONTEXT_BUFFER_FACTOR
        )

    def scan(self, input_path: Path, output_path: Path) -> dict[str, Any]:
        connection = _open_worker_database(
            output_path,
            """
            CREATE TABLE bridge_documents (
                doc_id TEXT PRIMARY KEY,
                bridge_hits TEXT NOT NULL
            );
            """,
        )
        rows: list[tuple[str, str]] = []
        processed = 0
        matched = 0
        kept = 0
        try:
            with input_path.open("rb") as file:
                for raw_line in file:
                    if not raw_line.strip():
                        continue
                    processed += 1
                    try:
                        row = json.loads(raw_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if not isinstance(row, dict) or row.get("source") == "clue_benchmark":
                        continue
                    if row.get("synthetic") or "doc_id" not in row or "text" not in row:
                        continue
                    text = str(row["text"])
                    hits = Counter(
                        int(token_id) for _end, token_id in self.automaton.iter(text)
                    )
                    if not hits:
                        continue
                    matched += 1
                    doc_id = str(row["doc_id"])
                    keep = (
                        _stable_fraction(self.config.seed, "bridge-fill", doc_id)
                        < self.fill_probability
                    )
                    if not keep:
                        keep = any(
                            _stable_fraction(
                                self.config.seed,
                                f"bridge-context:{token_id}",
                                doc_id,
                            )
                            < min(
                                1.0,
                                self.context_target
                                / max(1, self.token_document_frequency[token_id]),
                            )
                            for token_id in hits
                        )
                    if not keep:
                        continue
                    kept += 1
                    rows.append(
                        (
                            doc_id,
                            json.dumps(
                                {str(key): value for key, value in sorted(hits.items())},
                                separators=(",", ":"),
                            ),
                        )
                    )
                    if len(rows) >= 2_000:
                        connection.executemany(
                            "INSERT OR IGNORE INTO bridge_documents VALUES (?, ?)",
                            rows,
                        )
                        connection.commit()
                        rows.clear()
            if rows:
                connection.executemany(
                    "INSERT OR IGNORE INTO bridge_documents VALUES (?, ?)",
                    rows,
                )
                connection.commit()
        finally:
            connection.close()
        return {
            "path": str(input_path),
            "processed_records": processed,
            "matched_records": matched,
            "kept_records": kept,
        }


def _init_scan_worker(config: PipelineConfig) -> None:
    global _SCAN_WORKER
    _SCAN_WORKER = LightweightScanner(config)


def _scan_worker(task: tuple[Path, Path]) -> tuple[Path, Path, dict[str, Any]]:
    if _SCAN_WORKER is None:
        raise RuntimeError("prescan worker is not initialized")
    input_path, output_path = task
    return input_path, output_path, _SCAN_WORKER.scan(input_path, output_path)


def _init_bridge_worker(
    config: PipelineConfig,
    token_document_frequency: dict[int, int],
    fill_probability: float,
) -> None:
    global _BRIDGE_WORKER
    _BRIDGE_WORKER = TopBridgeScanner(
        config,
        token_document_frequency,
        fill_probability,
    )


def _bridge_worker(task: tuple[Path, Path]) -> tuple[Path, Path, dict[str, Any]]:
    if _BRIDGE_WORKER is None:
        raise RuntimeError("bridge worker is not initialized")
    input_path, output_path = task
    return input_path, output_path, _BRIDGE_WORKER.scan(input_path, output_path)


class DocumentIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                input_path TEXT NOT NULL,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                source TEXT NOT NULL,
                pool TEXT NOT NULL,
                quota_group TEXT,
                estimated_tokens INTEGER NOT NULL,
                sample_key INTEGER NOT NULL,
                output_key INTEGER NOT NULL,
                new_hanzi_hits TEXT NOT NULL,
                selected_intent TEXT,
                materialized INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS bridge_frequency (
                token_id INTEGER PRIMARY KEY,
                occurrences INTEGER NOT NULL,
                documents INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bridge_documents (
                doc_id TEXT PRIMARY KEY,
                bridge_hits TEXT NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            );
            CREATE TABLE IF NOT EXISTS scan_progress (
                stage TEXT NOT NULL,
                path TEXT NOT NULL,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                report_json TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY(stage, path)
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def ensure_fingerprint(self, fingerprint: str) -> None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'fingerprint'"
        ).fetchone()
        if row is None:
            records = int(
                self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            )
            if records:
                raise RuntimeError(
                    "document index has no fingerprint; rerun sample with --overwrite"
                )
            self.set_metadata("fingerprint", fingerprint)
        elif str(row[0]) != fingerprint:
            raise RuntimeError(
                "prescan configuration or vocabulary metadata changed; "
                "rerun sample with --overwrite"
            )

    def set_metadata(self, key: str, value: Any) -> None:
        serialized = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, serialized),
        )
        self.connection.commit()

    def get_metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row[0])

    def input_is_complete(self, stage: str, path: Path) -> bool:
        resolved = str(path.resolve())
        row = self.connection.execute(
            "SELECT size, mtime_ns FROM scan_progress WHERE stage = ? AND path = ?",
            (stage, resolved),
        ).fetchone()
        if row is None:
            return False
        stat = path.stat()
        if (int(row[0]), int(row[1])) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f"completed input changed; rerun with --overwrite: {path}")
        return True

    def _mark_complete(
        self,
        stage: str,
        path: Path,
        report: dict[str, Any],
    ) -> None:
        stat = path.stat()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO scan_progress
                (stage, path, size, mtime_ns, report_json, completed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                stage,
                str(path.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
                json.dumps(report, ensure_ascii=False, separators=(",", ":")),
                utc_now_iso(),
            ),
        )

    def merge_scan_shard(
        self,
        input_path: Path,
        shard_path: Path,
        report: dict[str, Any],
    ) -> None:
        self.connection.execute("ATTACH DATABASE ? AS scan_shard", (str(shard_path),))
        try:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO documents (
                    doc_id, input_path, byte_offset, byte_length, source, pool,
                    quota_group, estimated_tokens, sample_key, output_key,
                    new_hanzi_hits
                )
                SELECT
                    doc_id, input_path, byte_offset, byte_length, source, pool,
                    quota_group, estimated_tokens, sample_key, output_key,
                    new_hanzi_hits
                FROM scan_shard.documents
                """
            )
            frequencies = list(
                self.connection.execute(
                    """
                    SELECT token_id, occurrences, documents
                    FROM scan_shard.bridge_frequency
                    """
                )
            )
            self.connection.executemany(
                """
                INSERT INTO bridge_frequency(token_id, occurrences, documents)
                VALUES (?, ?, ?)
                ON CONFLICT(token_id) DO UPDATE SET
                    occurrences = bridge_frequency.occurrences + excluded.occurrences,
                    documents = bridge_frequency.documents + excluded.documents
                """,
                frequencies,
            )
            self._mark_complete("features", input_path, report)
            self.connection.commit()
        finally:
            self.connection.execute("DETACH DATABASE scan_shard")

    def merge_bridge_shard(
        self,
        input_path: Path,
        shard_path: Path,
        report: dict[str, Any],
    ) -> None:
        self.connection.execute("ATTACH DATABASE ? AS bridge_shard", (str(shard_path),))
        try:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO bridge_documents(doc_id, bridge_hits)
                SELECT b.doc_id, b.bridge_hits
                FROM bridge_shard.bridge_documents b
                JOIN documents d ON d.doc_id = b.doc_id
                WHERE d.pool IN ('chinese_natural', 'mixed_zh_en')
                   OR (d.pool = 'specialized' AND d.quota_group = 'math_science')
                """
            )
            self._mark_complete("bridge", input_path, report)
            self.connection.commit()
        finally:
            self.connection.execute("DETACH DATABASE bridge_shard")

    def create_indexes(self) -> None:
        self.connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS document_pool_sample
                ON documents(pool, quota_group, selected_intent, sample_key);
            CREATE INDEX IF NOT EXISTS document_new_hanzi
                ON documents(selected_intent, sample_key, doc_id)
                WHERE new_hanzi_hits != '{}';
            CREATE INDEX IF NOT EXISTS document_input_offset
                ON documents(input_path, byte_offset);
            CREATE INDEX IF NOT EXISTS document_materialized
                ON documents(materialized, input_path, byte_offset);
            """
        )
        self.connection.commit()

    def top_bridge_tokens(self, limit: int) -> list[dict[str, int]]:
        rows = self.connection.execute(
            """
            SELECT token_id, occurrences, documents
            FROM bridge_frequency
            ORDER BY occurrences DESC, token_id
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "old_token_id": int(token_id),
                "occurrences": int(occurrences),
                "documents": int(documents),
            }
            for token_id, occurrences, documents in rows
        ]

    def scan_reports(self, stage: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT report_json FROM scan_progress WHERE stage = ? ORDER BY path",
            (stage,),
        )
        return [json.loads(row[0]) for row in rows]

    def reset_preselection(self) -> None:
        self.connection.execute(
            "UPDATE documents SET selected_intent = NULL, materialized = 0"
        )
        self.connection.execute("DELETE FROM metadata WHERE key = 'preselection_report'")
        self.connection.commit()

    def selected_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM documents WHERE selected_intent IS NOT NULL"
            ).fetchone()[0]
        )

    def materialized_count(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM documents WHERE materialized = 1"
            ).fetchone()[0]
        )

    def reset_materialized(self) -> None:
        self.connection.execute("UPDATE documents SET materialized = 0")
        self.connection.commit()

    def mark_materialized(self, doc_ids: list[str]) -> None:
        self.connection.executemany(
            "UPDATE documents SET materialized = 1 WHERE doc_id = ?",
            [(doc_id,) for doc_id in doc_ids],
        )
        self.connection.commit()

    def inventory(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT pool, quota_group, source, COUNT(*), SUM(estimated_tokens)
            FROM documents
            GROUP BY pool, quota_group, source
            ORDER BY pool, quota_group, source
            """
        )
        return [
            {
                "pool": str(pool),
                "quota_group": quota_group,
                "source": str(source),
                "records": int(records),
                "estimated_tokens": int(tokens),
            }
            for pool, quota_group, source, records, tokens in rows
        ]

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def _prescan_fingerprint(config: PipelineConfig) -> str:
    payload = {
        "version": PRESCAN_VERSION,
        "seed": config.seed,
        "quality": config.quality,
        "window_target_tokens": config.windowing.target_tokens,
        "preselection_buffer_ratio": config.preselection_buffer_ratio,
        "bridge_top_token_count": config.vocab_alignment.bridge_top_token_count,
        "bridge_min_contexts": config.vocab_alignment.bridge_min_contexts,
        "removed_multi_hanzi_tokens_sha256": file_sha256(
            config.vocab_alignment.removed_multi_hanzi_tokens_path
        ),
        "new_hanzi_token_ids_sha256": file_sha256(
            config.vocab_alignment.new_hanzi_token_ids_path
        ),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _temporary_shard_path(cache_dir: Path, stage: str, path: Path) -> Path:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{stage}-{digest}.sqlite"


def _run_scan_jobs(
    *,
    tasks: list[tuple[Path, Path]],
    workers: int,
    initializer: Any,
    initargs: tuple[Any, ...],
    worker_function: Any,
    merge: Any,
    label: str,
) -> None:
    if not tasks:
        return
    if workers == 1:
        initializer(*initargs)
        for index, task in enumerate(tasks, start=1):
            input_path, shard_path, report = worker_function(task)
            merge(input_path, shard_path, report)
            shard_path.unlink(missing_ok=True)
            print(f"{label} {index}/{len(tasks)}: {input_path.name}")
        return

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        futures = {executor.submit(worker_function, task): task[0] for task in tasks}
        completed = 0
        for future in as_completed(futures):
            input_path, shard_path, report = future.result()
            merge(input_path, shard_path, report)
            shard_path.unlink(missing_ok=True)
            completed += 1
            print(f"{label} {completed}/{len(tasks)}: {input_path.name}")


def _summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for report in reports:
        for key in (
            "processed_records",
            "accepted_records",
            "skipped_records",
            "matched_records",
            "kept_records",
        ):
            totals[key] += int(report.get(key, 0))
        skipped.update(report.get("skipped_by_reason", {}))
    return {
        **totals,
        "skipped_by_reason": dict(sorted(skipped.items())),
    }


def build_document_index(
    config: PipelineConfig,
    database_path: Path,
    *,
    resume: bool,
    workers: int,
) -> tuple[DocumentIndex, dict[str, Any]]:
    paths = sorted(config.deduplicated_dir.glob("part-*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"no deduplicated JSONL files found under {config.deduplicated_dir}"
        )
    cache_dir = config.final_dir / ".prescan"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index = DocumentIndex(database_path)
    index.ensure_fingerprint(_prescan_fingerprint(config))

    feature_tasks = [
        (path, _temporary_shard_path(cache_dir, "features", path))
        for path in paths
        if not (resume and index.input_is_complete("features", path))
    ]
    _run_scan_jobs(
        tasks=feature_tasks,
        workers=workers,
        initializer=_init_scan_worker,
        initargs=(config,),
        worker_function=_scan_worker,
        merge=index.merge_scan_shard,
        label="prescan features",
    )
    index.create_indexes()

    top_bridge = index.top_bridge_tokens(
        config.vocab_alignment.bridge_top_token_count
    )
    token_document_frequency = {
        row["old_token_id"]: row["documents"] for row in top_bridge
    }
    eligible_documents = int(
        index.connection.execute(
            """
            SELECT COUNT(*) FROM documents
            WHERE pool IN ('chinese_natural', 'mixed_zh_en')
               OR (pool = 'specialized' AND quota_group = 'math_science')
            """
        ).fetchone()[0]
    )
    alignment_target = scale_quotas(
        {name: 1 for name in ALIGNMENT_VALIDATION_GROUPS},
        config.validation.alignment_tokens,
    )["multi_hanzi_bridge"]
    desired_bridge_documents = math.ceil(
        (
            config.quotas["multi_hanzi_bridge"] + alignment_target
        )
        * config.preselection_buffer_ratio
        / config.windowing.target_tokens
    )
    fill_probability = min(
        1.0,
        desired_bridge_documents
        * BRIDGE_FILL_BUFFER_FACTOR
        / max(1, eligible_documents),
    )

    bridge_tasks = [
        (path, _temporary_shard_path(cache_dir, "bridge", path))
        for path in paths
        if not (resume and index.input_is_complete("bridge", path))
    ]
    _run_scan_jobs(
        tasks=bridge_tasks,
        workers=workers,
        initializer=_init_bridge_worker,
        initargs=(config, token_document_frequency, fill_probability),
        worker_function=_bridge_worker,
        merge=index.merge_bridge_shard,
        label="prescan bridge",
    )

    report = {
        "version": PRESCAN_VERSION,
        "input_files": len(paths),
        "workers": workers,
        "feature_shards_processed_this_run": len(feature_tasks),
        "feature_shards_resumed": len(paths) - len(feature_tasks),
        "features": _summarize_reports(index.scan_reports("features")),
        "bridge_shards_processed_this_run": len(bridge_tasks),
        "bridge_shards_resumed": len(paths) - len(bridge_tasks),
        "bridge": _summarize_reports(index.scan_reports("bridge")),
        "top_bridge_tokens": top_bridge,
        "bridge_fill_probability": fill_probability,
        "inventory": index.inventory(),
    }
    index.set_metadata("scan_report", report)
    return index, report


def _read_hanzi_resource(path: Path) -> set[str]:
    chars: set[str] = set()
    with path.open("rt", encoding="utf-8") as file:
        for line in file:
            value = line.lstrip("\ufeff").strip()
            if not value or value.startswith("#"):
                continue
            field = value.split("	", 1)[0].strip()
            if field and field.lower() != "char":
                chars.add(field[0])
    return chars


def _load_new_hanzi(config: PipelineConfig) -> set[str]:
    with config.vocab_alignment.new_hanzi_token_ids_path.open(
        "rt", encoding="utf-8"
    ) as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("new_hanzi_token_ids.json must contain an object")
    return {
        char
        for char in value.values()
        if isinstance(char, str) and len(char) == 1
    }


def _preselection_targets(config: PipelineConfig) -> dict[str, int]:
    natural = scale_quotas(
        NATURAL_VALIDATION_WEIGHTS,
        config.validation.natural_tokens,
    )
    alignment = scale_quotas(
        {name: 1 for name in ALIGNMENT_VALIDATION_GROUPS},
        config.validation.alignment_tokens,
    )
    base_targets = {
        "new_hanzi_coverage": (
            config.quotas["new_hanzi_coverage"] + alignment["new_hanzi_coverage"]
        ),
        "multi_hanzi_bridge": (
            config.quotas["multi_hanzi_bridge"] + alignment["multi_hanzi_bridge"]
        ),
        "chinese_natural": (
            config.quotas["chinese_natural"]
            + natural["chinese_natural"]
            + alignment["original_hanzi"]
        ),
        "non_chinese": (
            config.quotas["non_chinese"]
            + natural["non_chinese"]
            + alignment["non_chinese"]
        ),
        "mixed_zh_en": config.quotas["mixed_zh_en"] + natural["mixed_zh_en"],
        "specialized:code": config.specialized_quotas["code"] + natural["code"],
        "specialized:math_science": (
            config.specialized_quotas["math_science"] + natural["math_science"]
        ),
        "specialized:structured": (
            config.specialized_quotas["structured"] + natural["structured"]
        ),
    }
    return {
        name: math.ceil(tokens * config.preselection_buffer_ratio)
        for name, tokens in base_targets.items()
    }


def _preselection_fingerprint(
    config: PipelineConfig,
    targets: dict[str, int],
) -> str:
    payload = {
        "version": PRESCAN_VERSION,
        "targets": targets,
        "priority_hanzi_min_documents": (
            config.vocab_alignment.priority_hanzi_min_documents
        ),
        "priority_hanzi_coverage": config.vocab_alignment.priority_hanzi_coverage,
        "priority_hanzi_sha256": {
            str(path): file_sha256(path)
            for path in config.vocab_alignment.priority_hanzi_paths
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _readonly_query(
    index: DocumentIndex,
    sql: str,
    values: tuple[Any, ...] = (),
) -> Iterator[tuple[Any, ...]]:
    connection = sqlite3.connect(f"file:{index.path}?mode=ro", uri=True)
    try:
        cursor = connection.execute(sql, values)
        try:
            yield from cursor
        finally:
            cursor.close()
    finally:
        connection.close()


def _flush_assignments(
    index: DocumentIndex,
    intent: str,
    doc_ids: list[str],
) -> None:
    if not doc_ids:
        return
    index.connection.executemany(
        """
        UPDATE documents SET selected_intent = ?
        WHERE doc_id = ? AND selected_intent IS NULL
        """,
        [(intent, doc_id) for doc_id in doc_ids],
    )
    index.connection.commit()
    doc_ids.clear()


def _select_simple_documents(
    index: DocumentIndex,
    *,
    intent: str,
    target_tokens: int,
    pool: str,
    quota_group: str | None = None,
    source_tokens: Counter[str] | None = None,
    source_token_cap: int | None = None,
) -> dict[str, Any]:
    clauses = ["pool = ?", "selected_intent IS NULL"]
    values: list[Any] = [pool]
    if quota_group is not None:
        clauses.append("quota_group = ?")
        values.append(quota_group)
    sql = (
        "SELECT doc_id, source, estimated_tokens FROM documents "
        f"WHERE {' AND '.join(clauses)} ORDER BY sample_key, doc_id"
    )
    selected: list[str] = []
    records = 0
    tokens = 0
    for doc_id, source, estimated_tokens in _readonly_query(index, sql, tuple(values)):
        estimated_tokens = int(estimated_tokens)
        source = str(source)
        if (
            source_tokens is not None
            and source_token_cap is not None
            and source_tokens[source] + estimated_tokens > source_token_cap
        ):
            continue
        selected.append(str(doc_id))
        records += 1
        tokens += estimated_tokens
        if source_tokens is not None:
            source_tokens[source] += estimated_tokens
        if len(selected) >= 5_000:
            _flush_assignments(index, intent, selected)
        if tokens >= target_tokens:
            break
    _flush_assignments(index, intent, selected)
    return {
        "target_estimated_tokens": target_tokens,
        "records": records,
        "estimated_tokens": tokens,
    }

def _select_new_hanzi_documents(
    index: DocumentIndex,
    config: PipelineConfig,
    target_tokens: int,
    source_tokens: Counter[str],
    source_token_cap: int,
) -> dict[str, Any]:
    intent = "new_hanzi_coverage"
    new_chars = _load_new_hanzi(config)
    priority_chars: set[str] = set()
    for path in config.vocab_alignment.priority_hanzi_paths:
        if not path.is_file():
            raise FileNotFoundError(f"priority Hanzi resource is missing: {path}")
        priority_chars.update(_read_hanzi_resource(path))
    priority_chars &= new_chars

    observed: set[str] = set()
    for (hits_json,) in _readonly_query(
        index,
        "SELECT new_hanzi_hits FROM documents WHERE new_hanzi_hits != '{}'",
    ):
        observed.update(json.loads(hits_json))

    document_counts: Counter[str] = Counter()
    records = 0
    tokens = 0
    pending: list[str] = []

    def run_pass(predicate: Any) -> None:
        nonlocal records, tokens
        for doc_id, source, estimated_tokens, hits_json in _readonly_query(
            index,
            """
            SELECT doc_id, source, estimated_tokens, new_hanzi_hits
            FROM documents
            WHERE new_hanzi_hits != '{}' AND selected_intent IS NULL
            ORDER BY sample_key, doc_id
            """,
        ):
            hits = set(json.loads(hits_json))
            estimated_tokens = int(estimated_tokens)
            source = str(source)
            if not predicate(hits):
                continue
            if source_tokens[source] + estimated_tokens > source_token_cap:
                continue
            pending.append(str(doc_id))
            document_counts.update(hits)
            source_tokens[source] += estimated_tokens
            records += 1
            tokens += estimated_tokens
            if len(pending) >= 5_000:
                _flush_assignments(index, intent, pending)
            if tokens >= target_tokens:
                break
        _flush_assignments(index, intent, pending)

    run_pass(lambda hits: any(document_counts[char] == 0 for char in hits))
    context_target = math.ceil(
        config.vocab_alignment.priority_hanzi_min_documents
        * config.preselection_buffer_ratio
    )
    if tokens < target_tokens:
        run_pass(
            lambda hits: any(
                char in priority_chars and document_counts[char] < context_target
                for char in hits
            )
        )
    if tokens < target_tokens:
        run_pass(lambda _hits: True)

    priority_met = sum(
        document_counts[char] >= context_target for char in priority_chars
    )
    return {
        "target_estimated_tokens": target_tokens,
        "records": records,
        "estimated_tokens": tokens,
        "observed_chars": len(observed),
        "covered_observed_chars": len(observed & document_counts.keys()),
        "priority_chars": len(priority_chars),
        "priority_context_target": context_target,
        "priority_chars_met": priority_met,
    }

def _select_bridge_documents(
    index: DocumentIndex,
    config: PipelineConfig,
    target_tokens: int,
    top_bridge: list[dict[str, int]],
    source_tokens: Counter[str],
    source_token_cap: int,
) -> dict[str, Any]:
    intent = "multi_hanzi_bridge"
    top_ids = {row["old_token_id"] for row in top_bridge}
    context_target = math.ceil(
        config.vocab_alignment.bridge_min_contexts
        * config.preselection_buffer_ratio
    )
    document_counts: Counter[int] = Counter()
    records = 0
    tokens = 0
    pending: list[str] = []

    def run_pass(predicate: Any) -> None:
        nonlocal records, tokens
        for doc_id, source, estimated_tokens, hits_json in _readonly_query(
            index,
            """
            SELECT d.doc_id, d.source, d.estimated_tokens, b.bridge_hits
            FROM bridge_documents b
            JOIN documents d ON d.doc_id = b.doc_id
            WHERE d.selected_intent IS NULL
            ORDER BY d.sample_key, d.doc_id
            """,
        ):
            hits = {int(token_id) for token_id in json.loads(hits_json)}
            estimated_tokens = int(estimated_tokens)
            source = str(source)
            if not predicate(hits):
                continue
            if source_tokens[source] + estimated_tokens > source_token_cap:
                continue
            pending.append(str(doc_id))
            document_counts.update(hits & top_ids)
            source_tokens[source] += estimated_tokens
            records += 1
            tokens += estimated_tokens
            if len(pending) >= 5_000:
                _flush_assignments(index, intent, pending)
            if tokens >= target_tokens:
                break
        _flush_assignments(index, intent, pending)

    run_pass(
        lambda hits: any(
            token_id in top_ids and document_counts[token_id] < context_target
            for token_id in hits
        )
    )
    if tokens < target_tokens:
        run_pass(lambda _hits: True)

    required = {
        row["old_token_id"]: min(context_target, row["documents"])
        for row in top_bridge
    }
    unmet = [
        token_id
        for token_id, minimum in required.items()
        if document_counts[token_id] < minimum
    ]
    return {
        "target_estimated_tokens": target_tokens,
        "records": records,
        "estimated_tokens": tokens,
        "tracked_top_tokens": len(top_ids),
        "context_target": context_target,
        "unmet_context_tokens": len(unmet),
        "unmet_token_ids": unmet,
    }

def preselect_documents(
    index: DocumentIndex,
    config: PipelineConfig,
    top_bridge: list[dict[str, int]],
) -> dict[str, Any]:
    targets = _preselection_targets(config)
    fingerprint = _preselection_fingerprint(config, targets)
    stored_fingerprint = index.get_metadata("preselection_fingerprint")
    stored_report = index.get_metadata("preselection_report")
    if (
        stored_fingerprint == fingerprint
        and stored_report is not None
        and index.selected_count() > 0
    ):
        return json.loads(stored_report)
    if index.materialized_count() > 0:
        raise RuntimeError(
            "preselection configuration changed after materialization; "
            "rerun sample with --overwrite"
        )

    index.reset_preselection()
    chinese_intents = (
        "new_hanzi_coverage",
        "multi_hanzi_bridge",
        "mixed_zh_en",
        "chinese_natural",
    )
    chinese_source_tokens: Counter[str] = Counter()
    chinese_source_cap = math.ceil(
        sum(targets[name] for name in chinese_intents) * 0.40
    )

    details: dict[str, Any] = {}
    details["new_hanzi_coverage"] = _select_new_hanzi_documents(
        index,
        config,
        targets["new_hanzi_coverage"],
        chinese_source_tokens,
        chinese_source_cap,
    )
    details["multi_hanzi_bridge"] = _select_bridge_documents(
        index,
        config,
        targets["multi_hanzi_bridge"],
        top_bridge,
        chinese_source_tokens,
        chinese_source_cap,
    )
    details["mixed_zh_en"] = _select_simple_documents(
        index,
        intent="mixed_zh_en",
        target_tokens=targets["mixed_zh_en"],
        pool="mixed_zh_en",
        source_tokens=chinese_source_tokens,
        source_token_cap=chinese_source_cap,
    )
    for group in ("code", "math_science", "structured"):
        intent = f"specialized:{group}"
        details[intent] = _select_simple_documents(
            index,
            intent=intent,
            target_tokens=targets[intent],
            pool="specialized",
            quota_group=group,
        )
    details["chinese_natural"] = _select_simple_documents(
        index,
        intent="chinese_natural",
        target_tokens=targets["chinese_natural"],
        pool="chinese_natural",
        source_tokens=chinese_source_tokens,
        source_token_cap=chinese_source_cap,
    )
    details["non_chinese"] = _select_simple_documents(
        index,
        intent="non_chinese",
        target_tokens=targets["non_chinese"],
        pool="non_chinese",
    )

    shortfalls = {
        name: {
            "target_estimated_tokens": targets[name],
            "actual_estimated_tokens": int(details[name]["estimated_tokens"]),
        }
        for name in targets
        if int(details[name]["estimated_tokens"]) < targets[name]
    }

    report = {
        "buffer_ratio": config.preselection_buffer_ratio,
        "targets": targets,
        "selection": details,
        "target_reached": not shortfalls,
        "shortfalls": shortfalls,
        "selected_documents": index.selected_count(),
        "chinese_source_budget": {
            "estimated_token_cap": chinese_source_cap,
            "estimated_tokens_by_source": dict(
                sorted(chinese_source_tokens.items())
            ),
        },
    }
    index.set_metadata("preselection_fingerprint", fingerprint)
    index.set_metadata("preselection_report", report)
    return report

def iter_selected_document_batches(
    index: DocumentIndex,
    *,
    max_records: int,
    max_chars: int,
) -> Iterator[list[tuple[dict[str, Any], str]]]:
    rows = _readonly_query(
        index,
        """
        SELECT
            doc_id, input_path, byte_offset, byte_length, selected_intent
        FROM documents
        WHERE selected_intent IS NOT NULL AND materialized = 0
        ORDER BY input_path, byte_offset
        """,
    )
    current_path: str | None = None
    current_file = None
    batch: list[tuple[dict[str, Any], str]] = []
    batch_chars = 0
    try:
        for doc_id, input_path, offset, byte_length, intent in rows:
            if input_path != current_path:
                if current_file is not None:
                    current_file.close()
                current_path = str(input_path)
                current_file = Path(current_path).open("rb")
            assert current_file is not None
            current_file.seek(int(offset))
            raw_line = current_file.read(int(byte_length))
            row = json.loads(raw_line.decode("utf-8"))
            if str(row.get("doc_id")) != str(doc_id):
                raise RuntimeError(
                    f"document locator mismatch for {doc_id} in {input_path}"
                )
            row_chars = len(str(row.get("text", "")))
            if batch and (
                len(batch) >= max_records or batch_chars + row_chars > max_chars
            ):
                yield batch
                batch = []
                batch_chars = 0
            batch.append((row, str(intent)))
            batch_chars += row_chars
        if batch:
            yield batch
    finally:
        if current_file is not None:
            current_file.close()
