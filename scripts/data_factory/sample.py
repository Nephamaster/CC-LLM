"""Token counting, quota selection, deterministic shuffle, and sharding."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.data_factory.candidate_features import CandidateSkip, Phase1CandidateBuilder
from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.io_utils import (
    TokenJsonlShardWriter,
    file_sha256,
    iter_jsonl,
    utc_now_iso,
    write_json,
)
from scripts.data_factory.prescan import (
    PRESCAN_VERSION,
    DocumentIndex,
    build_document_index,
    iter_selected_document_batches,
    preselect_documents,
)


def _stable_key(seed: int, namespace: str, doc_id: str) -> int:
    payload = f"{seed}:{namespace}:{doc_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


class CandidateIndex:
    CANDIDATE_COLUMNS = (
        "doc_id",
        "parent_doc_id",
        "candidate_role",
        "pool",
        "quota_group",
        "source",
        "token_count",
        "sample_key",
        "output_key",
        "eligible_bridge",
        "eligible_new_hanzi",
        "bridge_hit_count",
        "new_hanzi_count",
        "row_json",
        "selected_category",
    )
    INDEX_NAMES = (
        "candidate_sampling",
        "candidate_sampling_any",
        "candidate_sampling_source",
        "candidate_output",
        "candidate_parent",
        "candidate_bridge",
        "candidate_new_hanzi",
    )

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                doc_id TEXT PRIMARY KEY,
                parent_doc_id TEXT NOT NULL,
                candidate_role TEXT NOT NULL,
                pool TEXT NOT NULL,
                quota_group TEXT,
                source TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                sample_key INTEGER NOT NULL,
                output_key INTEGER NOT NULL,
                eligible_bridge INTEGER NOT NULL,
                eligible_new_hanzi INTEGER NOT NULL,
                bridge_hit_count INTEGER NOT NULL,
                new_hanzi_count INTEGER NOT NULL,
                row_json TEXT NOT NULL,
                selected_category TEXT
            );
            CREATE TABLE IF NOT EXISTS indexing_progress (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                report_json TEXT NOT NULL,
                completed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidate_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS parent_assignments (
                parent_doc_id TEXT PRIMARY KEY,
                split TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS selections (
                doc_id TEXT PRIMARY KEY,
                parent_doc_id TEXT NOT NULL,
                split TEXT NOT NULL,
                category TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                FOREIGN KEY(doc_id) REFERENCES candidates(doc_id),
                FOREIGN KEY(parent_doc_id) REFERENCES parent_assignments(parent_doc_id)
            );
            CREATE INDEX IF NOT EXISTS selection_split ON selections(split, category);
            CREATE INDEX IF NOT EXISTS selection_parent ON selections(parent_doc_id, split);
            """
        )
        columns = tuple(
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(candidates)")
        )
        if columns != self.CANDIDATE_COLUMNS:
            self.connection.close()
            raise RuntimeError(
                "candidate index schema is obsolete; rerun the sample action with --overwrite"
            )
        self.connection.commit()

    def ensure_fingerprint(self, fingerprint: str) -> None:
        row = self.connection.execute(
            "SELECT value FROM candidate_metadata WHERE key = 'fingerprint'"
        ).fetchone()
        if row is None:
            records = int(self.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
            if records:
                raise RuntimeError(
                    "candidate index has no build fingerprint; rerun with --overwrite"
                )
            self.connection.execute(
                "INSERT INTO candidate_metadata(key, value) VALUES ('fingerprint', ?)",
                (fingerprint,),
            )
            self.connection.commit()
        elif str(row[0]) != fingerprint:
            raise RuntimeError(
                "candidate index configuration or vocabulary metadata changed; "
                "rerun with --overwrite"
            )

    def set_metadata(self, key: str, value: Any) -> None:
        serialized = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO candidate_metadata(key, value) VALUES (?, ?)",
            (key, serialized),
        )
        self.connection.commit()

    def get_metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM candidate_metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row[0])

    def drop_sampling_indexes(self) -> None:
        for name in self.INDEX_NAMES:
            self.connection.execute(f"DROP INDEX IF EXISTS {name}")
        self.connection.commit()

    def create_sampling_indexes(self) -> None:
        self.connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS candidate_sampling
                ON candidates(pool, quota_group, sample_key, parent_doc_id);
            CREATE INDEX IF NOT EXISTS candidate_sampling_any
                ON candidates(pool, sample_key, parent_doc_id);
            CREATE INDEX IF NOT EXISTS candidate_sampling_source
                ON candidates(pool, source, sample_key, parent_doc_id);
            CREATE INDEX IF NOT EXISTS candidate_output ON candidates(output_key);
            CREATE INDEX IF NOT EXISTS candidate_parent ON candidates(parent_doc_id);
            CREATE INDEX IF NOT EXISTS candidate_bridge
                ON candidates(eligible_bridge, sample_key, parent_doc_id);
            CREATE INDEX IF NOT EXISTS candidate_new_hanzi
                ON candidates(eligible_new_hanzi, sample_key, parent_doc_id);
            """
        )
        self.connection.commit()

    def add_many(self, rows: list[tuple[dict[str, Any], int]], seed: int) -> int:
        values = []
        for row, token_count in rows:
            doc_id = str(row["doc_id"])
            parent_doc_id = str(row.get("parent_doc_id", doc_id))
            roles = row.get("candidate_roles", [row.get("candidate_role", "base")])
            if isinstance(roles, str):
                roles = [roles]
            candidate_role = ",".join(sorted(str(role) for role in roles))
            bridge_hits = row.get("bridge_hits", {})
            new_hanzi_hits = row.get("new_hanzi_hits", {})
            row["token_count"] = token_count
            values.append(
                (
                    doc_id,
                    parent_doc_id,
                    candidate_role,
                    str(row.get("candidate_pool", row["category"])),
                    row.get("quota_group"),
                    str(row["source"]),
                    token_count,
                    _stable_key(seed, "sample", doc_id),
                    _stable_key(seed, "output", doc_id),
                    int(bool(row.get("eligible_bridge", bridge_hits))),
                    int(bool(row.get("eligible_new_hanzi", new_hanzi_hits))),
                    sum(int(count) for count in bridge_hits.values()),
                    sum(int(count) for count in new_hanzi_hits.values()),
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                )
            )
        before = self.connection.total_changes
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO candidates (
                doc_id, parent_doc_id, candidate_role, pool, quota_group, source,
                token_count, sample_key, output_key, eligible_bridge,
                eligible_new_hanzi, bridge_hit_count, new_hanzi_count, row_json,
                selected_category
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            values,
        )
        return self.connection.total_changes - before

    def existing_doc_ids(self, doc_ids: list[str]) -> set[str]:
        return self._existing_values("doc_id", doc_ids)

    def existing_parent_doc_ids(self, doc_ids: list[str]) -> set[str]:
        return self._existing_values("parent_doc_id", doc_ids)

    def _existing_values(self, column: str, values: list[str]) -> set[str]:
        existing: set[str] = set()
        for start in range(0, len(values), 900):
            chunk = values[start : start + 900]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT DISTINCT {column} FROM candidates WHERE {column} IN ({placeholders})",
                chunk,
            )
            existing.update(str(row[0]) for row in rows)
        return existing

    def input_is_complete(self, path: Path) -> bool:
        resolved = str(path.resolve())
        row = self.connection.execute(
            "SELECT size, mtime_ns FROM indexing_progress WHERE path = ?", (resolved,)
        ).fetchone()
        if row is None:
            return False
        stat = path.stat()
        if (int(row[0]), int(row[1])) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f"completed input changed; rerun with --overwrite: {path}")
        return True

    def mark_input_complete(self, path: Path, report: dict[str, Any]) -> None:
        stat = path.stat()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO indexing_progress
                (path, size, mtime_ns, report_json, completed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(path.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
                json.dumps(report, ensure_ascii=False, separators=(",", ":")),
                utc_now_iso(),
            ),
        )
        self.connection.commit()

    def progress_reports(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT report_json FROM indexing_progress")
        return [json.loads(row[0]) for row in rows]

    def totals(self) -> tuple[int, int]:
        row = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM candidates"
        ).fetchone()
        return int(row[0]), int(row[1])

    def feature_inventory(self) -> dict[str, int]:
        row = self.connection.execute(
            """
            SELECT
                COUNT(DISTINCT parent_doc_id),
                SUM(eligible_bridge),
                COUNT(DISTINCT CASE WHEN eligible_bridge = 1 THEN parent_doc_id END),
                COALESCE(SUM(CASE WHEN eligible_bridge = 1 THEN token_count ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN eligible_bridge = 1 THEN bridge_hit_count ELSE 0 END), 0),
                SUM(eligible_new_hanzi),
                COUNT(DISTINCT CASE WHEN eligible_new_hanzi = 1 THEN parent_doc_id END),
                COALESCE(SUM(CASE WHEN eligible_new_hanzi = 1 THEN token_count ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN eligible_new_hanzi = 1 THEN new_hanzi_count ELSE 0 END), 0)
            FROM candidates
            """
        ).fetchone()
        return {
            "parent_records": int(row[0] or 0),
            "bridge_candidate_records": int(row[1] or 0),
            "bridge_parent_records": int(row[2] or 0),
            "bridge_candidate_tokens": int(row[3] or 0),
            "bridge_occurrences": int(row[4] or 0),
            "new_hanzi_candidate_records": int(row[5] or 0),
            "new_hanzi_parent_records": int(row[6] or 0),
            "new_hanzi_candidate_tokens": int(row[7] or 0),
            "new_hanzi_occurrences": int(row[8] or 0),
        }

    def inventory(self) -> list[tuple[str, str | None, str, int, int]]:
        return list(
            self.connection.execute(
                """
                SELECT pool, quota_group, source, COUNT(*), SUM(token_count)
                FROM candidates GROUP BY pool, quota_group, source ORDER BY pool, quota_group, source
                """
            )
        )

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

@dataclass
class TokenCountResult:
    rows: list[tuple[dict[str, Any], int]]
    zero_token_records: int
    skipped_by_reason: Counter[str]
    skipped_examples: list[dict[str, str]]


class BatchTokenCounter:
    def __init__(self, config: PipelineConfig, workers: int) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["RAYON_NUM_THREADS"] = str(workers)

        from transformers import AutoTokenizer

        from src.vocab.qwen3_char_tokenizer import normalize_text
        from src.vocab.unicode_ranges import CJK_RANGES

        self.auto_tokenizer_class = AutoTokenizer
        self.tokenizer_path = config.tokenizer_path
        self.thread_local = threading.local()
        if not self._get_tokenizer().is_fast:
            raise RuntimeError("Phase 1 sampling requires a fast tokenizer")
        self.normalize_text = normalize_text
        ranges = "".join(f"{chr(item.start)}-{chr(item.end)}" for item in CJK_RANGES)
        self.hanzi_pattern = re.compile(f"[{ranges}]")
        self.hanzi_token_ids = _target_hanzi_token_ids(config)
        self.supported_hanzi = set(self.hanzi_token_ids)
        self._validate_hanzi_alignment()

    def _get_tokenizer(self):
        tokenizer = getattr(self.thread_local, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self.auto_tokenizer_class.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
                use_fast=True,
            )
            self.thread_local.tokenizer = tokenizer
        return tokenizer

    def _validate_hanzi_alignment(self) -> None:
        tokenizer = self._get_tokenizer()
        chars = sorted(self.hanzi_token_ids, key=ord)
        mismatches: list[dict[str, Any]] = []
        for start in range(0, len(chars), 2048):
            chunk = chars[start : start + 2048]
            encoded = tokenizer(
                chunk,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            for char, token_ids in zip(chunk, encoded, strict=True):
                expected = self.hanzi_token_ids[char]
                if token_ids != [expected]:
                    mismatches.append({"char": char, "expected": expected, "actual": token_ids})
                    if len(mismatches) >= 10:
                        break
            if mismatches:
                break
        if mismatches:
            raise RuntimeError(f"fast tokenizer is not aligned with the character vocabulary: {mismatches}")

        all_hanzi = "".join(chars)
        expected = [self.hanzi_token_ids[char] for char in chars]
        actual = tokenizer.encode(all_hanzi, add_special_tokens=False)
        if actual != expected:
            mismatch = next(
                (index for index, (left, right) in enumerate(zip(actual, expected)) if left != right),
                min(len(actual), len(expected)),
            )
            raise RuntimeError(
                "fast tokenizer merged adjacent Hanzi: "
                f"position={mismatch}, expected_tokens={len(expected)}, actual_tokens={len(actual)}"
            )

        for text in ("\u4e2d\u56fdABC123", "def \u68c0\u67e5(x):\n    return x + 1"):
            expected = self._reference_encode(text)
            actual = tokenizer.encode(text, add_special_tokens=False)
            if actual != expected:
                raise RuntimeError(f"fast tokenizer is not aligned on mixed text: {text!r}")

    def _reference_encode(self, text: str) -> list[int]:
        input_ids: list[int] = []
        position = 0
        while position < len(text):
            char = text[position]
            if char in self.hanzi_token_ids:
                input_ids.append(self.hanzi_token_ids[char])
                position += 1
                continue
            next_position = position + 1
            while next_position < len(text) and text[next_position] not in self.hanzi_token_ids:
                next_position += 1
            input_ids.extend(
                self._get_tokenizer().encode(text[position:next_position], add_special_tokens=False)
            )
            position = next_position
        return input_ids

    def _unsupported_hanzi(self, text: str) -> str | None:
        for match in self.hanzi_pattern.finditer(text):
            char = match.group(0)
            if char not in self.supported_hanzi:
                return char
        return None

    def _encode_lengths(self, texts: list[str]) -> list[int]:
        encoded = self._get_tokenizer()(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_length=True,
        )
        lengths = encoded.get("length")
        if lengths is None:
            lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
        return [int(length) for length in lengths]

    @staticmethod
    def _add_skip(
        skipped_by_reason: Counter[str],
        skipped_examples: list[dict[str, str]],
        row: dict[str, Any],
        reason: str,
        error: str,
    ) -> None:
        skipped_by_reason[reason] += 1
        if len(skipped_examples) < 100:
            skipped_examples.append(
                {
                    "doc_id": str(row.get("doc_id", "unknown")),
                    "reason": reason,
                    "error": error,
                }
            )

    def count_rows(self, rows: list[dict[str, Any]]) -> TokenCountResult:
        prepared: list[tuple[dict[str, Any], str]] = []
        skipped_by_reason: Counter[str] = Counter()
        skipped_examples: list[dict[str, str]] = []
        for row in rows:
            missing = [key for key in ("doc_id", "category", "source", "text") if key not in row]
            if missing:
                self._add_skip(
                    skipped_by_reason,
                    skipped_examples,
                    row,
                    "invalid_record",
                    f"missing required fields: {missing}",
                )
                continue
            try:
                text = self.normalize_text(str(row["text"]))
            except Exception as error:
                self._add_skip(
                    skipped_by_reason,
                    skipped_examples,
                    row,
                    "invalid_text",
                    f"{type(error).__name__}: {error}",
                )
                continue
            unsupported = self._unsupported_hanzi(text)
            if unsupported is not None:
                self._add_skip(
                    skipped_by_reason,
                    skipped_examples,
                    row,
                    "unsupported_hanzi",
                    f"Hanzi is missing from char tokenizer vocab: {unsupported} U+{ord(unsupported):04X}",
                )
                continue
            prepared.append((row, text))

        counted: list[tuple[dict[str, Any], int]] = []
        if prepared:
            try:
                lengths = self._encode_lengths([text for _row, text in prepared])
                if len(lengths) != len(prepared):
                    raise RuntimeError("tokenizer returned an unexpected number of lengths")
                counted.extend((row, length) for (row, _text), length in zip(prepared, lengths, strict=True))
            except Exception:
                for row, text in prepared:
                    try:
                        counted.append((row, self._encode_lengths([text])[0]))
                    except Exception as error:
                        self._add_skip(
                            skipped_by_reason,
                            skipped_examples,
                            row,
                            "tokenization_error",
                            f"{type(error).__name__}: {error}",
                        )

        zero_token_records = sum(token_count == 0 for _row, token_count in counted)
        positive = [(row, token_count) for row, token_count in counted if token_count > 0]
        return TokenCountResult(positive, zero_token_records, skipped_by_reason, skipped_examples)


class Phase1CandidateProcessor:
    def __init__(self, config: PipelineConfig, token_counter: BatchTokenCounter) -> None:
        self.token_counter = token_counter
        self.builder = Phase1CandidateBuilder(
            config,
            self._count_tokens,
            count_many=self.token_counter._encode_lengths,
        )

    def _count_tokens(self, text: str) -> int:
        return self.token_counter._encode_lengths([text])[0]

    def count_rows(self, rows: list[dict[str, Any]]) -> TokenCountResult:
        counted = self.token_counter.count_rows(rows)
        candidates: list[tuple[dict[str, Any], int]] = []
        skipped_by_reason = Counter(counted.skipped_by_reason)
        skipped_examples = list(counted.skipped_examples)

        for row, full_token_count in counted.rows:
            text = self.token_counter.normalize_text(str(row["text"]))
            value = dict(row)
            value["text"] = text
            try:
                candidates.extend(self.builder.build(value, text, full_token_count))
            except CandidateSkip as error:
                BatchTokenCounter._add_skip(
                    skipped_by_reason,
                    skipped_examples,
                    row,
                    error.reason,
                    str(error),
                )
            except Exception as error:
                BatchTokenCounter._add_skip(
                    skipped_by_reason,
                    skipped_examples,
                    row,
                    "candidate_build_error",
                    f"{type(error).__name__}: {error}",
                )

        return TokenCountResult(
            rows=candidates,
            zero_token_records=counted.zero_token_records,
            skipped_by_reason=skipped_by_reason,
            skipped_examples=skipped_examples,
        )


def _candidate_index_fingerprint(config: PipelineConfig) -> str:
    payload = {
        "pipeline_version": PRESCAN_VERSION,
        "schema": CandidateIndex.CANDIDATE_COLUMNS,
        "seed": config.seed,
        "quality": config.quality,
        "windowing": {
            "min_tokens": config.windowing.min_tokens,
            "target_tokens": config.windowing.target_tokens,
            "max_tokens": config.windowing.max_tokens,
        },
        "vocab_alignment": {
            "bridge_top_token_count": config.vocab_alignment.bridge_top_token_count,
            "removed_multi_hanzi_tokens_sha256": file_sha256(
                config.vocab_alignment.removed_multi_hanzi_tokens_path
            ),
            "new_hanzi_token_ids_sha256": file_sha256(
                config.vocab_alignment.new_hanzi_token_ids_path
            ),
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def _target_hanzi_token_ids(config: PipelineConfig) -> dict[str, int]:
    path = config.tokenizer_path / "features" / "char_feature_index.jsonl"
    return {
        str(row["char"]): int(row["token_id"])
        for row in iter_jsonl([path])
        if row.get("is_hanzi")
        and isinstance(row.get("char"), str)
        and len(row["char"]) == 1
    }


def _filter_candidates_for_intent(
    rows: list[tuple[dict[str, Any], int]],
    intents: dict[str, str],
) -> list[tuple[dict[str, Any], int]]:
    simple_intents: dict[str, tuple[str, str | None]] = {
        "chinese_natural": ("chinese_natural", None),
        "mixed_zh_en": ("mixed_zh_en", None),
        "non_chinese": ("non_chinese", None),
        "specialized:code": ("specialized", "code"),
        "specialized:math_science": ("specialized", "math_science"),
        "specialized:structured": ("specialized", "structured"),
    }
    filtered: list[tuple[dict[str, Any], int]] = []
    for row, token_count in rows:
        parent_doc_id = str(row.get("parent_doc_id", row.get("doc_id", "")))
        intent = intents.get(parent_doc_id)
        roles = row.get("candidate_roles", [])
        if isinstance(roles, str):
            roles = [roles]

        value = dict(row)
        if intent == "new_hanzi_coverage":
            if not value.get("eligible_new_hanzi"):
                continue
            value["candidate_roles"] = ["new_hanzi_coverage"]
            value["eligible_bridge"] = False
            value["bridge_hits"] = {}
        elif intent == "multi_hanzi_bridge":
            if not value.get("eligible_bridge"):
                continue
            value["candidate_roles"] = ["multi_hanzi_bridge"]
            value["eligible_new_hanzi"] = False
            value["new_hanzi_hits"] = {}
        elif intent in simple_intents:
            if "base" not in roles:
                continue
            pool, quota_group = simple_intents[intent]
            value["candidate_pool"] = pool
            value["quota_group"] = quota_group
            value["candidate_roles"] = ["base"]
            value["eligible_bridge"] = False
            value["eligible_new_hanzi"] = False
            value["bridge_hits"] = {}
            value["new_hanzi_hits"] = {}
        else:
            continue
        filtered.append((value, token_count))
    return filtered


def _materialize_preselected_candidates(
    config: PipelineConfig,
    index: CandidateIndex,
    document_index: DocumentIndex,
    *,
    workers: int,
) -> dict[str, Any]:
    token_counter = BatchTokenCounter(config, workers)
    candidate_processor = Phase1CandidateProcessor(config, token_counter)
    existing_candidates, _tokens = index.totals()
    if existing_candidates == 0 and document_index.materialized_count() > 0:
        document_index.reset_materialized()

    started = time.monotonic()
    processed = 0
    inserted = 0
    reused = 0
    zero_token_records = 0
    skipped: Counter[str] = Counter()
    skipped_examples: list[dict[str, str]] = []
    max_in_flight = max(1, workers * 2)
    in_flight: dict[
        Future[TokenCountResult],
        list[tuple[dict[str, Any], str]],
    ] = {}

    def consume(completed: set[Future[TokenCountResult]]) -> None:
        nonlocal processed, inserted, reused, zero_token_records
        for future in completed:
            items = in_flight.pop(future)
            rows = [row for row, _intent in items]
            intents = {str(row["doc_id"]): intent for row, intent in items}
            try:
                result = future.result()
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                result = TokenCountResult(
                    rows=[],
                    zero_token_records=0,
                    skipped_by_reason=Counter({"worker_error": len(rows)}),
                    skipped_examples=[
                        {
                            "doc_id": str(row.get("doc_id", "unknown")),
                            "reason": "worker_error",
                            "error": message,
                        }
                        for row in rows[:100]
                    ],
                )

            candidates = _filter_candidates_for_intent(result.rows, intents)
            added = index.add_many(candidates, config.seed)
            index.connection.commit()
            document_index.mark_materialized(list(intents))
            inserted += added
            reused += len(candidates) - added
            zero_token_records += result.zero_token_records
            skipped.update(result.skipped_by_reason)
            remaining = 100 - len(skipped_examples)
            if remaining > 0:
                skipped_examples.extend(result.skipped_examples[:remaining])
            processed += len(items)
            if processed % 10_000 < len(items):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"exact materialization {processed:,}/"
                    f"{document_index.selected_count():,} documents "
                    f"({processed / elapsed:,.0f}/s)"
                )

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="phase1-tokenizer",
    ) as executor:
        for batch in iter_selected_document_batches(
            document_index,
            max_records=config.sample_batch_size,
            max_chars=config.sample_batch_chars,
        ):
            future = executor.submit(
                candidate_processor.count_rows,
                [row for row, _intent in batch],
            )
            in_flight[future] = batch
            if len(in_flight) >= max_in_flight:
                completed, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
                consume(completed)
        while in_flight:
            completed, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
            consume(completed)

    return {
        "selected_documents": document_index.selected_count(),
        "materialized_documents": document_index.materialized_count(),
        "processed_this_run": processed,
        "candidate_records_inserted": inserted,
        "candidate_records_reused": reused,
        "zero_token_records": zero_token_records,
        "skipped_records": sum(skipped.values()),
        "skipped_by_reason": dict(sorted(skipped.items())),
        "skipped_examples": skipped_examples,
        "elapsed_seconds": time.monotonic() - started,
    }


def _build_candidate_index(
    config: PipelineConfig,
    database_path: Path,
    *,
    resume: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    worker_count = config.sample_workers if workers is None else workers
    if worker_count <= 0:
        raise ValueError("workers must be positive")

    started = time.monotonic()
    document_path = database_path.with_name("document_index.sqlite")
    document_index, prescan_report = build_document_index(
        config,
        document_path,
        resume=resume,
        workers=worker_count,
    )
    try:
        top_bridge = document_index.top_bridge_tokens(
            config.vocab_alignment.bridge_top_token_count
        )
        preselection_report = preselect_documents(
            document_index,
            config,
            top_bridge,
        )
        if not preselection_report["target_reached"]:
            raise RuntimeError(
                "insufficient preselected candidates: "
                + json.dumps(
                    preselection_report["shortfalls"],
                    ensure_ascii=False,
                )
            )

        index = CandidateIndex(database_path)
        try:
            index.ensure_fingerprint(_candidate_index_fingerprint(config))
            index.set_metadata("bridge_top_tokens", top_bridge)
            index.drop_sampling_indexes()
            materialization_report = _materialize_preselected_candidates(
                config,
                index,
                document_index,
                workers=worker_count,
            )
            print("building SQLite sampling indexes")
            index.create_sampling_indexes()
            records, tokens = index.totals()
            feature_inventory = index.feature_inventory()
            inventory = [
                {
                    "pool": pool,
                    "quota_group": quota_group,
                    "source": source,
                    "records": inventory_records,
                    "tokens": inventory_tokens,
                }
                for pool, quota_group, source, inventory_records, inventory_tokens
                in index.inventory()
            ]
        finally:
            index.close()
    finally:
        document_index.close()

    return {
        "records": records,
        "parent_records": feature_inventory["parent_records"],
        "tokens": tokens,
        "candidate_records": records,
        "feature_inventory": feature_inventory,
        "input_files": prescan_report["input_files"],
        "processed_this_run": materialization_report["processed_this_run"],
        "elapsed_seconds": time.monotonic() - started,
        "batch_size": config.sample_batch_size,
        "batch_chars": config.sample_batch_chars,
        "workers": worker_count,
        "inventory": inventory,
        "prescan": prescan_report,
        "preselection": preselection_report,
        "materialization": materialization_report,
        "document_database": str(document_path),
    }

def _emit_final(
    index: CandidateIndex,
    config: PipelineConfig,
) -> tuple[TokenJsonlShardWriter, dict[str, Counter[str]]]:
    from scripts.data_factory.selection import materialize_row, split_rows

    writer = TokenJsonlShardWriter(config.final_dir, "train", config.final_shard_tokens)
    stats: dict[str, Counter[str]] = {
        "categories": Counter(),
        "specialized": Counter(),
        "sources": Counter(),
        "licenses": Counter(),
        "license_status": Counter(),
    }
    for category, token_count, row_json in split_rows(index.connection, "train"):
        row = materialize_row(str(category), int(token_count), str(row_json))
        writer.write(row)
        tokens = int(token_count)
        stats["categories"][str(category)] += tokens
        if category == "specialized":
            stats["specialized"][str(row.get("quota_group", "unknown"))] += tokens
        stats["sources"][str(row.get("source", "unknown"))] += tokens
        stats["licenses"][str(row.get("license", "unknown"))] += tokens
        stats["license_status"][str(row.get("license_status", "unknown"))] += tokens
    writer.close()
    return writer, stats


def _category_checks(
    actual: Counter[str],
    targets: dict[str, int],
    tolerance: float,
) -> tuple[dict[str, Any], bool]:
    checks: dict[str, Any] = {}
    passed = True
    for category, target in targets.items():
        value = int(actual.get(category, 0))
        relative_error = abs(value - target) / target
        category_passed = relative_error <= tolerance
        checks[category] = {
            "target_tokens": target,
            "actual_tokens": value,
            "relative_error": relative_error,
            "passed": category_passed,
        }
        passed = passed and category_passed
    return checks, passed


def sample_phase1(
    config: PipelineConfig,
    overwrite: bool = False,
    resume: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    from scripts.data_factory.build_phase1_validation import export_validation_sets
    from scripts.data_factory.selection import (
        CHINESE_TRAIN_CATEGORIES,
        Phase1Selector,
        parent_split_overlap,
        selection_inventory,
    )

    config.final_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    database_path = config.final_dir / "candidate_index.sqlite"
    document_database_path = config.final_dir / "document_index.sqlite"
    prescan_cache_dir = config.final_dir / ".prescan"
    final_files = list(config.final_dir.glob("train-*.jsonl"))
    manifest_path = config.final_dir / "manifest.json"
    validation_dir = config.phase_root / "validation"
    validation_files = [
        validation_dir / "validation_natural.jsonl",
        validation_dir / "validation_alignment.jsonl",
        config.reports_dir / "phase1_validation_report.json",
    ]

    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if (
        database_path.exists()
        or document_database_path.exists()
        or final_files
    ) and not (overwrite or resume):
        raise FileExistsError(
            f"final outputs already exist under {config.final_dir}; pass --overwrite or --resume"
        )
    if overwrite:
        for sqlite_path in (database_path, document_database_path):
            for suffix in ("", "-wal", "-shm"):
                Path(f"{sqlite_path}{suffix}").unlink(missing_ok=True)
        if prescan_cache_dir.is_dir():
            for cache_path in prescan_cache_dir.iterdir():
                if cache_path.is_file():
                    cache_path.unlink()
            prescan_cache_dir.rmdir()
    if overwrite or resume:
        for path in [*final_files, manifest_path, *validation_files]:
            path.unlink(missing_ok=True)

    index_report = _build_candidate_index(
        config,
        database_path,
        resume=resume,
        workers=workers,
    )
    index = CandidateIndex(database_path)
    try:
        selector = Phase1Selector(index.connection, config)
        selector.reset()
        validation_selection = selector.reserve_validation()
        selection, alignment_coverage = selector.select_training()
        overlap = parent_split_overlap(index.connection)
        selected_inventory = selection_inventory(index.connection, "train")
        writer, stats = _emit_final(index, config)
    finally:
        index.close()

    validation_report = export_validation_sets(
        config,
        database_path,
        overwrite=True,
        reservation_report=validation_selection,
    )

    category_checks, categories_passed = _category_checks(
        stats["categories"],
        config.quotas,
        config.tolerance,
    )
    specialized_checks, specialized_passed = _category_checks(
        stats["specialized"],
        config.specialized_quotas,
        config.tolerance,
    )

    chinese_source_tokens: Counter[str] = Counter()
    for row in selected_inventory:
        if row["category"] in CHINESE_TRAIN_CATEGORIES:
            chinese_source_tokens[row["source"]] += int(row["tokens"])
    chinese_target = sum(config.quotas[name] for name in CHINESE_TRAIN_CATEGORIES)
    source_cap = int(chinese_target * 0.40)
    source_concentration = {
        source: {
            "tokens": tokens,
            "ratio": tokens / chinese_target,
            "passed": tokens <= source_cap,
        }
        for source, tokens in sorted(chinese_source_tokens.items())
    }
    source_concentration_passed = all(
        item["passed"] for item in source_concentration.values()
    )

    unknown_license_tokens = int(stats["licenses"].get("unknown", 0))
    conflicting_status_tokens = sum(
        int(stats["license_status"].get(status, 0))
        for status in ("unknown", "conflict", "unverified")
    )
    licenses_passed = unknown_license_tokens == 0 and conflicting_status_tokens == 0
    coverage_passed = all(
        bool(report["passed"]) for report in alignment_coverage.values()
    )
    passed = all(
        (
            categories_passed,
            specialized_passed,
            source_concentration_passed,
            licenses_passed,
            coverage_passed,
            validation_report["passed"],
            overlap == 0,
        )
    )

    generated_at = utc_now_iso()
    coverage_report = {
        "generated_at": generated_at,
        "passed": coverage_passed,
        **alignment_coverage,
    }
    write_json(
        config.reports_dir / "phase1_alignment_coverage_report.json",
        coverage_report,
    )

    report = {
        "generated_at": generated_at,
        "passed": passed,
        "target_tokens": sum(config.quotas.values()),
        "actual_tokens": writer.total_tokens,
        "tolerance": config.tolerance,
        "category_checks": category_checks,
        "specialized_checks": specialized_checks,
        "selection": selection,
        "validation_selection": validation_selection,
        "validation_passed": validation_report["passed"],
        "parent_split_overlap_records": overlap,
        "alignment_coverage": {
            name: {
                key: value
                for key, value in coverage.items()
                if key != "character_stats" and key != "token_stats"
            }
            for name, coverage in alignment_coverage.items()
        },
        "candidate_index": index_report,
        "selected_inventory": selected_inventory,
        "tokens_by_source": dict(sorted(stats["sources"].items())),
        "tokens_by_license": dict(sorted(stats["licenses"].items())),
        "tokens_by_license_status": dict(sorted(stats["license_status"].items())),
        "license_checks": {
            "unknown_license_tokens": unknown_license_tokens,
            "conflicting_or_unverified_status_tokens": conflicting_status_tokens,
            "passed": licenses_passed,
        },
        "source_concentration": {
            "chinese_target_tokens": chinese_target,
            "single_source_token_cap": source_cap,
            "passed": source_concentration_passed,
            "sources": source_concentration,
        },
        "files": writer.files,
        "candidate_database": str(database_path),
        "validation_report": str(
            config.reports_dir / "phase1_validation_report.json"
        ),
        "alignment_coverage_report": str(
            config.reports_dir / "phase1_alignment_coverage_report.json"
        ),
    }
    write_json(config.reports_dir / "phase1_build_report.json", report)

    validation_outputs = [
        {
            "path": validation_report["natural"]["output_path"],
            "records": validation_report["natural"]["records"],
            "tokens": validation_report["natural"]["tokens"],
            "sha256": validation_report["natural"]["output_sha256"],
        },
        {
            "path": validation_report["alignment"]["output_path"],
            "records": validation_report["alignment"]["records"],
            "tokens": validation_report["alignment"]["tokens"],
            "sha256": validation_report["alignment"]["output_sha256"],
        },
    ]
    write_json(
        manifest_path,
        {
            "generated_at": generated_at,
            "train": {
                "records": writer.total_records,
                "tokens": writer.total_tokens,
                "files": writer.files,
            },
            "validation": validation_outputs,
        },
    )
    return report
