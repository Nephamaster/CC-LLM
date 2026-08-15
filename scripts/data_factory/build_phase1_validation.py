"""Build a held-out Phase 1 validation set from unselected candidates."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.data_factory.config import PipelineConfig, load_config
from scripts.data_factory.io_utils import file_sha256, utc_now_iso, write_json


DEFAULT_TARGET_TOKENS = 1_000_000


def scale_quotas(quotas: dict[str, int], target_tokens: int) -> dict[str, int]:
    """Scale integer quotas to an exact target with largest-remainder rounding."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    total = sum(quotas.values())
    if total <= 0:
        raise ValueError("quota total must be positive")

    scaled: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    for name, weight in quotas.items():
        value, remainder = divmod(target_tokens * weight, total)
        scaled[name] = value
        remainders.append((remainder, name))
    for _remainder, name in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : target_tokens - sum(scaled.values())
    ]:
        scaled[name] += 1
    return scaled


class ValidationIndex:
    """Select held-out rows without changing the persistent candidate index."""

    REQUIRED_INDEXES = {
        "candidate_sampling",
        "candidate_sampling_any",
        "candidate_sampling_source",
    }

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"candidate index does not exist: {path}")
        self.connection = sqlite3.connect(path)
        index_names = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        missing_indexes = sorted(self.REQUIRED_INDEXES - index_names)
        if missing_indexes:
            self.close()
            raise RuntimeError(
                f"candidate index is missing sampling indexes {missing_indexes}; "
                "rerun or resume the Phase 1 sample action"
            )

        training_selection_present = any(
            self.connection.execute(
                """
                SELECT 1 FROM candidates INDEXED BY candidate_sampling_any
                WHERE pool = ? AND selected_category IS NOT NULL LIMIT 1
                """,
                (pool,),
            ).fetchone()
            is not None
            for pool in (
                "chinese_general",
                "chinese_high_quality",
                "mixed_zh_en",
                "non_chinese",
                "supplemental",
            )
        )
        if not training_selection_present:
            self.close()
            raise RuntimeError("candidate index has no training selection; run the Phase 1 sample action first")

        self.connection.execute(
            """
            CREATE TEMP TABLE validation_selection (
                doc_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                token_count INTEGER NOT NULL
            )
            """
        )
        self.selected_doc_ids: set[str] = set()

    @staticmethod
    def _index_name(quota_group: str | None, source: str | None) -> str:
        if source is not None:
            return "candidate_sampling_source"
        if quota_group is not None:
            return "candidate_sampling"
        return "candidate_sampling_any"

    @staticmethod
    def _filters(
        pool: str,
        quota_group: str | None = None,
        source: str | None = None,
    ) -> tuple[list[str], list[Any]]:
        clauses = ["c.pool = ?", "c.selected_category IS NULL"]
        values: list[Any] = [pool]
        if quota_group is not None:
            clauses.append("c.quota_group = ?")
            values.append(quota_group)
        if source is not None:
            clauses.append("c.source = ?")
            values.append(source)
        return clauses, values

    def select(
        self,
        pool: str,
        category: str,
        target_tokens: int,
        quota_group: str | None = None,
        source: str | None = None,
    ) -> dict[str, int]:
        if target_tokens <= 0:
            return {"target_tokens": target_tokens, "records": 0, "tokens": 0}

        clauses, values = self._filters(pool, quota_group, source)
        index_name = self._index_name(quota_group, source)
        cursor = self.connection.execute(
            f"""
            SELECT c.doc_id, c.token_count
            FROM candidates c INDEXED BY {index_name}
            WHERE {' AND '.join(clauses)}
            ORDER BY c.sample_key
            """,
            values,
        )
        selected: list[tuple[str, str, int]] = []
        selected_tokens = 0
        try:
            for doc_id, token_count in cursor:
                doc_id = str(doc_id)
                if doc_id in self.selected_doc_ids:
                    continue
                token_count = int(token_count)
                selected.append((doc_id, category, token_count))
                selected_tokens += token_count
                if selected_tokens >= target_tokens:
                    break
        finally:
            cursor.close()

        if selected:
            self.connection.executemany(
                "INSERT INTO validation_selection(doc_id, category, token_count) VALUES (?, ?, ?)",
                selected,
            )
            self.selected_doc_ids.update(doc_id for doc_id, _category, _tokens in selected)
        return {
            "target_tokens": target_tokens,
            "records": len(selected),
            "tokens": selected_tokens,
        }

    def selected_rows(self):
        return self.connection.execute(
            """
            SELECT v.category, v.token_count, c.row_json
            FROM validation_selection v
            JOIN candidates c ON c.doc_id = v.doc_id
            ORDER BY c.output_key, c.doc_id
            """
        )

    def selected_inventory(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT v.category, c.source, c.quota_group, COUNT(*), SUM(v.token_count)
            FROM validation_selection v
            JOIN candidates c ON c.doc_id = v.doc_id
            GROUP BY v.category, c.source, c.quota_group
            ORDER BY v.category, c.source, c.quota_group
            """
        )
        return [
            {
                "category": category,
                "source": source,
                "quota_group": quota_group,
                "records": int(records),
                "tokens": int(tokens),
            }
            for category, source, quota_group, records, tokens in rows
        ]

    def train_overlap(self) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM validation_selection v
            JOIN candidates c ON c.doc_id = v.doc_id
            WHERE c.selected_category IS NOT NULL
            """
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self.connection.close()


def _add_result(target: dict[str, Any], name: str, value: dict[str, int]) -> None:
    target["parts"][name] = value
    target["records"] += value["records"]
    target["tokens"] += value["tokens"]


def select_validation_quotas(
    index: ValidationIndex,
    config: PipelineConfig,
    target_tokens: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    quotas = scale_quotas(config.quotas, target_tokens)
    details: dict[str, Any] = {}

    high_quality = index.select(
        "chinese_high_quality",
        "chinese_high_quality",
        quotas["chinese_high_quality"],
    )
    details["chinese_high_quality"] = high_quality

    general: dict[str, Any] = {"records": 0, "tokens": 0, "parts": {}}

    _add_result(
        general,
        "clue_benchmark",
        index.select(
            "chinese_general",
            "chinese_general",
            quotas["chinese_general"],
            source="clue_benchmark",
        ),
    )
    remaining = max(0, quotas["chinese_general"] - general["tokens"])
    _add_result(
        general,
        "other_general",
        index.select("chinese_general", "chinese_general", remaining),
    )
    remaining = max(0, quotas["chinese_general"] - general["tokens"])
    _add_result(
        general,
        "high_quality_fallback",
        index.select("chinese_high_quality", "chinese_general", remaining),
    )
    details["chinese_general"] = general

    details["non_chinese"] = index.select(
        "non_chinese",
        "non_chinese",
        quotas["non_chinese"],
    )

    mixed: dict[str, Any] = {"records": 0, "tokens": 0, "parts": {}}
    mixed_quotas = scale_quotas(config.mixed_quotas, quotas["mixed_zh_en"])
    for quota_group, group_target in mixed_quotas.items():
        _add_result(
            mixed,
            quota_group,
            index.select(
                "mixed_zh_en",
                "mixed_zh_en",
                group_target,
                quota_group=quota_group,
            ),
        )
    remaining = max(0, quotas["mixed_zh_en"] - mixed["tokens"])
    _add_result(
        mixed,
        "fallback",
        index.select("mixed_zh_en", "mixed_zh_en", remaining),
    )
    details["mixed_zh_en"] = mixed

    supplemental: dict[str, Any] = {"records": 0, "tokens": 0, "parts": {}}
    supplemental_quotas = scale_quotas(config.supplemental_quotas, quotas["supplemental"])
    for quota_group, group_target in supplemental_quotas.items():
        _add_result(
            supplemental,
            quota_group,
            index.select(
                "supplemental",
                "supplemental",
                group_target,
                quota_group=quota_group,
            ),
        )
    remaining = max(0, quotas["supplemental"] - supplemental["tokens"])
    _add_result(
        supplemental,
        "fallback",
        index.select("supplemental", "supplemental", remaining),
    )
    details["supplemental"] = supplemental
    return quotas, details


def _write_validation(index: ValidationIndex, output_path: Path) -> tuple[int, int, dict[str, int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    records = 0
    tokens = 0
    sources: Counter[str] = Counter()
    try:
        with temporary.open("wt", encoding="utf-8", newline="\n") as file:
            for category, token_count, row_json in index.selected_rows():
                row = json.loads(row_json)
                row["category"] = category
                row["token_count"] = int(token_count)
                file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                file.write("\n")
                records += 1
                tokens += int(token_count)
                sources[str(row.get("source", "unknown"))] += int(token_count)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return records, tokens, dict(sorted(sources.items()))


def build_validation_set(
    config: PipelineConfig,
    target_tokens: int,
    output_path: Path,
    report_path: Path,
    database_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    existing = [path for path in (output_path, report_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"validation outputs already exist: {existing}; pass --overwrite")

    index = ValidationIndex(database_path)
    try:
        quotas, selection = select_validation_quotas(index, config, target_tokens)
        shortfalls = {
            category: {"target_tokens": quotas[category], "actual_tokens": int(result["tokens"])}
            for category, result in selection.items()
            if int(result["tokens"]) < quotas[category]
        }

        overlap = index.train_overlap()
        if overlap:
            raise RuntimeError(f"validation selection overlaps {overlap} training documents")

        records, actual_tokens, tokens_by_source = _write_validation(index, output_path)
        inventory = index.selected_inventory()
        training_selection_present = True
    finally:
        index.close()

    category_checks = {
        category: {
            "target_tokens": category_target,
            "actual_tokens": int(selection[category]["tokens"]),
            "overshoot_tokens": int(selection[category]["tokens"]) - category_target,
        }
        for category, category_target in quotas.items()
    }
    report = {
        "generated_at": utc_now_iso(),
        "passed": records > 0 and overlap == 0,
        "target_reached": actual_tokens >= target_tokens,
        "category_quotas_met": not shortfalls,
        "category_shortfalls": shortfalls,
        "target_tokens": target_tokens,
        "actual_tokens": actual_tokens,
        "records": records,
        "document_level_split": True,
        "training_selection_present": training_selection_present,
        "training_overlap_records": overlap,
        "config": str(config.path),
        "candidate_database": str(database_path),
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "category_checks": category_checks,
        "selection": selection,
        "selected_inventory": inventory,
        "tokens_by_source": tokens_by_source,
    }
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("scripts/data_factory/phase1_config.json"))
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--database-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    database_path = args.database_path or config.final_dir / "candidate_index.sqlite"
    output_path = args.output_path or config.phase_root / "validation" / "validation.jsonl"
    report_path = args.report_path or config.reports_dir / "phase1_validation_report.json"
    report = build_validation_set(
        config=config,
        target_tokens=args.target_tokens,
        output_path=output_path.resolve(),
        report_path=report_path.resolve(),
        database_path=database_path.resolve(),
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "records": report["records"],
                "target_tokens": report["target_tokens"],
                "actual_tokens": report["actual_tokens"],
                "output_path": report["output_path"],
                "report_path": str(report_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
