"""Token counting, quota selection, deterministic shuffle, and sharding."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.io_utils import TokenJsonlShardWriter, iter_jsonl, utc_now_iso, write_json


def _stable_key(seed: int, namespace: str, doc_id: str) -> int:
    payload = f"{seed}:{namespace}:{doc_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


class CandidateIndex:
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

    def add(self, row: dict[str, Any], token_count: int, seed: int) -> None:
        doc_id = str(row["doc_id"])
        self.connection.execute(
            "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                doc_id,
                str(row["category"]),
                row.get("quota_group"),
                str(row["source"]),
                token_count,
                _stable_key(seed, "sample", doc_id),
                _stable_key(seed, "output", doc_id),
                json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            ),
        )

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


def _build_candidate_index(config: PipelineConfig, database_path: Path) -> dict[str, Any]:
    from src.vocab.qwen3_char_tokenizer import Qwen3CharTokenizer, Qwen3CharTokenizerConfig

    paths = sorted(config.deduplicated_dir.glob("part-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no deduplicated JSONL files found under {config.deduplicated_dir}")
    tokenizer = Qwen3CharTokenizer(
        Qwen3CharTokenizerConfig(
            tokenizer_dir=config.tokenizer_path,
            features_dir=config.tokenizer_path / "features",
        )
    )
    index = CandidateIndex(database_path)
    counts: Counter[str] = Counter()
    try:
        for row_number, row in enumerate(iter_jsonl(paths), start=1):
            token_count = len(tokenizer.encode(str(row["text"]), add_special_tokens=False)["input_ids"])
            if token_count == 0:
                counts["zero_token_records"] += 1
                continue
            row["token_count"] = token_count
            index.add(row, token_count, config.seed)
            counts["records"] += 1
            counts["tokens"] += token_count
            if row_number % 1000 == 0:
                index.connection.commit()
            if row_number % 100_000 == 0:
                print(f"indexed {row_number:,} records")
        index.connection.commit()
        inventory = [
            {
                "pool": pool,
                "quota_group": quota_group,
                "source": source,
                "records": records,
                "tokens": tokens,
            }
            for pool, quota_group, source, records, tokens in index.inventory()
        ]
    finally:
        index.close()
    return {"records": counts["records"], "tokens": counts["tokens"], "inventory": inventory}


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
    path = config.tokenizer_path / "features" / "char_feature_index.jsonl"
    return {
        str(row["char"])
        for row in iter_jsonl([path])
        if row.get("is_hanzi")
        and isinstance(row.get("char"), str)
        and len(row["char"]) == 1
    }


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


def sample_phase1(config: PipelineConfig, overwrite: bool = False) -> dict[str, Any]:
    config.final_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    database_path = config.final_dir / "candidate_index.sqlite"
    final_files = list(config.final_dir.glob("train-*.jsonl"))
    if (database_path.exists() or final_files) and not overwrite:
        raise FileExistsError(f"final outputs already exist under {config.final_dir}; pass --overwrite")
    if overwrite:
        database_path.unlink(missing_ok=True)
        for path in final_files:
            path.unlink()

    index_report = _build_candidate_index(config, database_path)
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

