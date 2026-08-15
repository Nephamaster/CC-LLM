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

from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.io_utils import TokenJsonlShardWriter, iter_jsonl, utc_now_iso, write_json


def _stable_key(seed: int, namespace: str, doc_id: str) -> int:
    payload = f"{seed}:{namespace}:{doc_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


class CandidateIndex:
    INDEX_NAMES = (
        "candidate_sampling",
        "candidate_sampling_any",
        "candidate_sampling_source",
        "candidate_output",
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
                pool TEXT NOT NULL,
                quota_group TEXT,
                source TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                sample_key INTEGER NOT NULL,
                output_key INTEGER NOT NULL,
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
            """
        )
        self.connection.commit()

    def drop_sampling_indexes(self) -> None:
        for name in self.INDEX_NAMES:
            self.connection.execute(f"DROP INDEX IF EXISTS {name}")
        self.connection.commit()

    def create_sampling_indexes(self) -> None:
        self.connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS candidate_sampling
                ON candidates(pool, quota_group, selected_category, sample_key);
            CREATE INDEX IF NOT EXISTS candidate_sampling_any
                ON candidates(pool, selected_category, sample_key);
            CREATE INDEX IF NOT EXISTS candidate_sampling_source
                ON candidates(pool, source, selected_category, sample_key);
            CREATE INDEX IF NOT EXISTS candidate_output ON candidates(output_key);
            """
        )
        self.connection.commit()

    def add_many(self, rows: list[tuple[dict[str, Any], int]], seed: int) -> int:
        values = []
        for row, token_count in rows:
            doc_id = str(row["doc_id"])
            row["token_count"] = token_count
            values.append(
                (
                    doc_id,
                    str(row["category"]),
                    row.get("quota_group"),
                    str(row["source"]),
                    token_count,
                    _stable_key(seed, "sample", doc_id),
                    _stable_key(seed, "output", doc_id),
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                )
            )
        before = self.connection.total_changes
        self.connection.executemany(
            "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            values,
        )
        return self.connection.total_changes - before

    def existing_doc_ids(self, doc_ids: list[str]) -> set[str]:
        existing: set[str] = set()
        for start in range(0, len(doc_ids), 900):
            chunk = doc_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT doc_id FROM candidates WHERE doc_id IN ({placeholders})",
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

    def reset_selection(self) -> None:
        self.connection.execute("UPDATE candidates SET selected_category = NULL")
        self.connection.commit()

    def available_tokens(self, pool: str, quota_group: str | None = None, source: str | None = None) -> int:
        clauses = ["pool = ?", "selected_category IS NULL"]
        values: list[Any] = [pool]
        if quota_group is not None:
            clauses.append("quota_group = ?")
            values.append(quota_group)
        if source is not None:
            clauses.append("source = ?")
            values.append(source)
        row = self.connection.execute(
            f"SELECT COALESCE(SUM(token_count), 0) FROM candidates WHERE {' AND '.join(clauses)}", values
        ).fetchone()
        return int(row[0])

    def select(
        self,
        pool: str,
        category: str,
        target_tokens: int,
        quota_group: str | None = None,
        source: str | None = None,
    ) -> dict[str, int]:
        if target_tokens <= 0:
            return {"records": 0, "tokens": 0}
        clauses = ["pool = ?", "selected_category IS NULL"]
        values: list[Any] = [pool]
        if quota_group is not None:
            clauses.append("quota_group = ?")
            values.append(quota_group)
        if source is not None:
            clauses.append("source = ?")
            values.append(source)
        self.connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS current_selection "
            "(doc_id TEXT PRIMARY KEY, token_count INTEGER NOT NULL)"
        )
        self.connection.execute("DELETE FROM current_selection")
        self.connection.execute(
            f"""
            INSERT INTO current_selection(doc_id, token_count)
            SELECT doc_id, token_count FROM (
                SELECT
                    doc_id,
                    token_count,
                    SUM(token_count) OVER (
                        ORDER BY sample_key, doc_id ROWS UNBOUNDED PRECEDING
                    ) AS cumulative_tokens
                FROM candidates
                WHERE {' AND '.join(clauses)}
            )
            WHERE cumulative_tokens - token_count < ?
            """,
            [*values, target_tokens],
        )
        records, tokens = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(token_count), 0) FROM current_selection"
        ).fetchone()
        self.connection.execute(
            """
            UPDATE candidates SET selected_category = ?
            WHERE doc_id IN (SELECT doc_id FROM current_selection)
            """,
            (category,),
        )
        self.connection.commit()
        return {"records": int(records), "tokens": int(tokens)}

    def inventory(self) -> list[tuple[str, str | None, str, int, int]]:
        return list(
            self.connection.execute(
                """
                SELECT pool, quota_group, source, COUNT(*), SUM(token_count)
                FROM candidates GROUP BY pool, quota_group, source ORDER BY pool, quota_group, source
                """
            )
        )

    def selected_inventory(self) -> list[tuple[str, str, str | None, int, int]]:
        return list(
            self.connection.execute(
                """
                SELECT selected_category, source, quota_group, COUNT(*), SUM(token_count)
                FROM candidates WHERE selected_category IS NOT NULL
                GROUP BY selected_category, source, quota_group
                ORDER BY selected_category, source, quota_group
                """
            )
        )

    def selected_rows(self):
        return self.connection.execute(
            """
            SELECT selected_category, token_count, row_json
            FROM candidates WHERE selected_category IS NOT NULL ORDER BY output_key
            """
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


def _target_hanzi_token_ids(config: PipelineConfig) -> dict[str, int]:
    path = config.tokenizer_path / "features" / "char_feature_index.jsonl"
    return {
        str(row["char"]): int(row["token_id"])
        for row in iter_jsonl([path])
        if row.get("is_hanzi")
        and isinstance(row.get("char"), str)
        and len(row["char"]) == 1
    }


def _iter_row_batches(path: Path, max_records: int, max_chars: int):
    batch: list[dict[str, Any]] = []
    char_count = 0
    for row in iter_jsonl([path]):
        row_chars = len(str(row.get("text", "")))
        if batch and (len(batch) >= max_records or char_count + row_chars > max_chars):
            yield batch
            batch = []
            char_count = 0
        batch.append(row)
        char_count += row_chars
    if batch:
        yield batch


def _summarize_indexing_progress(reports: list[dict[str, Any]]) -> dict[str, Any]:
    skipped_by_reason: Counter[str] = Counter()
    skipped_examples: list[dict[str, str]] = []
    totals: Counter[str] = Counter()
    for report in reports:
        totals["processed_records"] += int(report.get("processed_records", 0))
        totals["zero_token_records"] += int(report.get("zero_token_records", 0))
        totals["reused_records"] += int(report.get("reused_records", 0))
        skipped_by_reason.update(report.get("skipped_by_reason", {}))
        remaining = 100 - len(skipped_examples)
        if remaining > 0:
            skipped_examples.extend(report.get("skipped_examples", [])[:remaining])
    return {
        **totals,
        "skipped_records": sum(skipped_by_reason.values()),
        "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "skipped_examples": skipped_examples,
    }


def _build_candidate_index(
    config: PipelineConfig,
    database_path: Path,
    *,
    resume: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    paths = sorted(config.deduplicated_dir.glob("part-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no deduplicated JSONL files found under {config.deduplicated_dir}")

    worker_count = config.sample_workers if workers is None else workers
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    token_counter = BatchTokenCounter(config, worker_count)
    index = CandidateIndex(database_path)
    index.drop_sampling_indexes()
    started = time.monotonic()
    processed_this_run = 0
    resumed_files = 0
    skipped_this_run: Counter[str] = Counter()
    next_progress = 100_000
    last_progress_records = 0
    last_progress_time = started
    max_in_flight = max(1, worker_count * 2)

    def advance_progress(count: int) -> None:
        nonlocal processed_this_run, next_progress, last_progress_records, last_progress_time
        processed_this_run += count
        while processed_this_run >= next_progress:
            now = time.monotonic()
            interval_seconds = max(now - last_progress_time, 1e-6)
            interval_records = processed_this_run - last_progress_records
            interval_rate = interval_records / interval_seconds
            average_rate = processed_this_run / max(now - started, 1e-6)
            print(
                f"processed {processed_this_run:,} records "
                f"(current {interval_rate:,.0f}/s, average {average_rate:,.0f}/s, "
                f"skipped {sum(skipped_this_run.values()):,})"
            )
            last_progress_records = processed_this_run
            last_progress_time = now
            next_progress += 100_000

    def worker_error_result(rows: list[dict[str, Any]], error: Exception) -> TokenCountResult:
        message = f"{type(error).__name__}: {error}"
        examples = [
            {
                "doc_id": str(row.get("doc_id", "unknown")),
                "reason": "worker_error",
                "error": message,
            }
            for row in rows[:100]
        ]
        return TokenCountResult([], 0, Counter({"worker_error": len(rows)}), examples)

    def consume_futures(
        futures: set[Future[TokenCountResult]],
        in_flight: dict[Future[TokenCountResult], tuple[int, list[dict[str, Any]]]],
        shard_stats: Counter[str],
        shard_skipped: Counter[str],
        shard_examples: list[dict[str, str]],
    ) -> None:
        for future in futures:
            batch_records, pending_rows = in_flight.pop(future)
            try:
                result = future.result()
            except Exception as error:
                result = worker_error_result(pending_rows, error)

            inserted = index.add_many(result.rows, config.seed)
            shard_stats["reused_records"] += len(result.rows) - inserted
            shard_stats["zero_token_records"] += result.zero_token_records
            shard_skipped.update(result.skipped_by_reason)
            skipped_this_run.update(result.skipped_by_reason)
            remaining_examples = 100 - len(shard_examples)
            if remaining_examples > 0:
                shard_examples.extend(result.skipped_examples[:remaining_examples])
            index.connection.commit()
            advance_progress(batch_records)

    try:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="phase1-tokenizer") as executor:
            for path in paths:
                if resume and index.input_is_complete(path):
                    resumed_files += 1
                    print(f"resumed completed shard: {path.name}")
                    continue

                shard_stats: Counter[str] = Counter()
                shard_skipped: Counter[str] = Counter()
                shard_examples: list[dict[str, str]] = []
                in_flight: dict[Future[TokenCountResult], tuple[int, list[dict[str, Any]]]] = {}

                for batch in _iter_row_batches(path, config.sample_batch_size, config.sample_batch_chars):
                    shard_stats["processed_records"] += len(batch)
                    pending = batch
                    if resume:
                        doc_ids = [str(row["doc_id"]) for row in batch if "doc_id" in row]
                        existing = index.existing_doc_ids(doc_ids)
                        if existing:
                            pending = [
                                row
                                for row in batch
                                if "doc_id" not in row or str(row["doc_id"]) not in existing
                            ]
                            shard_stats["reused_records"] += len(batch) - len(pending)

                    if not pending:
                        advance_progress(len(batch))
                        continue

                    future = executor.submit(token_counter.count_rows, pending)
                    in_flight[future] = (len(batch), pending)
                    if len(in_flight) >= max_in_flight:
                        completed, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
                        consume_futures(
                            completed,
                            in_flight,
                            shard_stats,
                            shard_skipped,
                            shard_examples,
                        )

                while in_flight:
                    completed, _pending = wait(in_flight, return_when=FIRST_COMPLETED)
                    consume_futures(
                        completed,
                        in_flight,
                        shard_stats,
                        shard_skipped,
                        shard_examples,
                    )

                index.mark_input_complete(
                    path,
                    {
                        "path": str(path),
                        "processed_records": shard_stats["processed_records"],
                        "zero_token_records": shard_stats["zero_token_records"],
                        "reused_records": shard_stats["reused_records"],
                        "skipped_by_reason": dict(sorted(shard_skipped.items())),
                        "skipped_examples": shard_examples,
                    },
                )
                print(f"completed input shard: {path.name}")
        print("building SQLite sampling indexes")
        index.create_sampling_indexes()
        records, tokens = index.totals()
        progress = _summarize_indexing_progress(index.progress_reports())
        inventory = [
            {
                "pool": pool,
                "quota_group": quota_group,
                "source": source,
                "records": inventory_records,
                "tokens": inventory_tokens,
            }
            for pool, quota_group, source, inventory_records, inventory_tokens in index.inventory()
        ]
    finally:
        index.close()

    return {
        "records": records,
        "tokens": tokens,
        **progress,
        "input_files": len(paths),
        "resumed_files": resumed_files,
        "processed_this_run": processed_this_run,
        "elapsed_seconds": time.monotonic() - started,
        "batch_size": config.sample_batch_size,
        "batch_chars": config.sample_batch_chars,
        "workers": worker_count,
        "inventory": inventory,
    }


def _add_result(target: dict[str, int], value: dict[str, int]) -> None:
    target["records"] += value["records"]
    target["tokens"] += value["tokens"]


def _select_quotas(index: CandidateIndex, config: PipelineConfig) -> dict[str, Any]:
    index.reset_selection()
    details: dict[str, Any] = {}

    high_quality = {"records": 0, "tokens": 0}
    _add_result(
        high_quality,
        index.select("chinese_high_quality", "chinese_high_quality", config.quotas["chinese_high_quality"]),
    )
    details["chinese_high_quality"] = high_quality

    general = {"records": 0, "tokens": 0, "parts": {}}
    clue_available = index.available_tokens("chinese_general", source="clue_benchmark")
    clue_target = min(config.quotas["chinese_general"], clue_available)
    clue_result = index.select(
        "chinese_general", "chinese_general", clue_target, source="clue_benchmark"
    )
    general["parts"]["clue_benchmark"] = clue_result
    _add_result(general, clue_result)
    remaining = max(0, config.quotas["chinese_general"] - general["tokens"])
    other_general = index.select("chinese_general", "chinese_general", remaining)
    general["parts"]["other_general"] = other_general
    _add_result(general, other_general)
    remaining = max(0, config.quotas["chinese_general"] - general["tokens"])
    chinese_fallback = index.select("chinese_high_quality", "chinese_general", remaining)
    general["parts"]["fineweb_chinese_fallback"] = chinese_fallback
    _add_result(general, chinese_fallback)
    details["chinese_general"] = general

    non_chinese = index.select("non_chinese", "non_chinese", config.quotas["non_chinese"])
    details["non_chinese"] = non_chinese

    mixed = {"records": 0, "tokens": 0, "parts": {}}
    for quota_group, target in config.mixed_quotas.items():
        result = index.select("mixed_zh_en", "mixed_zh_en", target, quota_group=quota_group)
        mixed["parts"][quota_group] = result
        _add_result(mixed, result)
    remaining = max(0, config.quotas["mixed_zh_en"] - mixed["tokens"])
    fallback = index.select("mixed_zh_en", "mixed_zh_en", remaining)
    mixed["parts"]["fallback"] = fallback
    _add_result(mixed, fallback)
    details["mixed_zh_en"] = mixed

    supplemental = {"records": 0, "tokens": 0, "parts": {}}
    for quota_group, target in config.supplemental_quotas.items():
        result = index.select("supplemental", "supplemental", target, quota_group=quota_group)
        supplemental["parts"][quota_group] = result
        _add_result(supplemental, result)
    remaining = max(0, config.quotas["supplemental"] - supplemental["tokens"])
    fallback = index.select("supplemental", "supplemental", remaining)
    supplemental["parts"]["fallback"] = fallback
    _add_result(supplemental, fallback)
    details["supplemental"] = supplemental
    return details


def _target_hanzi(config: PipelineConfig) -> set[str]:
    return set(_target_hanzi_token_ids(config))

def _emit_final(
    index: CandidateIndex,
    config: PipelineConfig,
    target_hanzi: set[str],
) -> tuple[TokenJsonlShardWriter, dict[str, Counter[str]], set[str]]:
    writer = TokenJsonlShardWriter(config.final_dir, "train", config.final_shard_tokens)
    missing_hanzi = set(target_hanzi)
    stats: dict[str, Counter[str]] = {
        "categories": Counter(),
        "sources": Counter(),
        "licenses": Counter(),
        "license_status": Counter(),
    }
    for category, token_count, row_json in index.selected_rows():
        row = json.loads(row_json)
        row["category"] = category
        row["token_count"] = int(token_count)
        writer.write(row)
        if missing_hanzi:
            missing_hanzi.difference_update(str(row["text"]))
        stats["categories"][str(category)] += int(token_count)
        stats["sources"][str(row.get("source", "unknown"))] += int(token_count)
        stats["licenses"][str(row.get("license", "unknown"))] += int(token_count)
        stats["license_status"][str(row.get("license_status", "unknown"))] += int(token_count)
    writer.close()
    return writer, stats, missing_hanzi


def sample_phase1(
    config: PipelineConfig,
    overwrite: bool = False,
    resume: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    config.final_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    database_path = config.final_dir / "candidate_index.sqlite"
    final_files = list(config.final_dir.glob("train-*.jsonl"))
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if (database_path.exists() or final_files) and not (overwrite or resume):
        raise FileExistsError(
            f"final outputs already exist under {config.final_dir}; pass --overwrite or --resume"
        )
    if overwrite:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{database_path}{suffix}").unlink(missing_ok=True)
    if overwrite or resume:
        for path in final_files:
            path.unlink()

    index_report = _build_candidate_index(config, database_path, resume=resume, workers=workers)
    index = CandidateIndex(database_path)
    target_hanzi = _target_hanzi(config)
    try:
        selection = _select_quotas(index, config)
        selected_inventory = [
            {
                "category": category,
                "source": source,
                "quota_group": quota_group,
                "records": records,
                "tokens": tokens,
            }
            for category, source, quota_group, records, tokens in index.selected_inventory()
        ]
        writer, stats, missing_hanzi = _emit_final(index, config, target_hanzi)
    finally:
        index.close()

    category_checks: dict[str, Any] = {}
    passed = True
    for category, target in config.quotas.items():
        actual = int(stats["categories"].get(category, 0))
        relative_error = abs(actual - target) / target
        category_passed = relative_error <= config.tolerance
        passed = passed and category_passed
        category_checks[category] = {
            "target_tokens": target,
            "actual_tokens": actual,
            "relative_error": relative_error,
            "passed": category_passed,
        }

    hanzi_coverage = 1.0 - len(missing_hanzi) / len(target_hanzi) if target_hanzi else 1.0
    passed = passed and not missing_hanzi

    approved_exceptions = int(stats["license_status"].get("approved_exception", 0))
    report = {
        "generated_at": utc_now_iso(),
        "passed": passed,
        "target_tokens": sum(config.quotas.values()),
        "actual_tokens": writer.total_tokens,
        "tolerance": config.tolerance,
        "category_checks": category_checks,
        "hanzi_coverage": {
            "target_chars": len(target_hanzi),
            "covered_chars": len(target_hanzi) - len(missing_hanzi),
            "coverage": hanzi_coverage,
            "missing_count": len(missing_hanzi),
            "missing_chars": sorted(missing_hanzi)[:1000],
            "passed": not missing_hanzi,
        },
        "selection": selection,
        "candidate_index": index_report,
        "selected_inventory": selected_inventory,
        "tokens_by_source": dict(sorted(stats["sources"].items())),
        "tokens_by_license": dict(sorted(stats["licenses"].items())),
        "tokens_by_license_status": dict(sorted(stats["license_status"].items())),
        "approved_license_exception_tokens": approved_exceptions,
        "approved_license_exception_note": "CLUE benchmark license is unknown and is retained only by explicit Phase 1 approval.",
        "files": writer.files,
        "candidate_database": str(database_path),
    }
    write_json(config.reports_dir / "phase1_build_report.json", report)
    write_json(
        config.final_dir / "manifest.json",
        {
            "generated_at": report["generated_at"],
            "records": writer.total_records,
            "tokens": writer.total_tokens,
            "files": writer.files,
        },
    )
    return report

