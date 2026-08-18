"""Export the Phase 1 validation splits reserved before training selection."""

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
from scripts.data_factory.selection import (
    ALIGNMENT_VALIDATION_GROUPS,
    NATURAL_VALIDATION_WEIGHTS,
    materialize_row,
    parent_split_overlap,
    scale_quotas,
    selection_inventory,
    split_rows,
)


def _expected_targets(config: PipelineConfig) -> dict[str, dict[str, int]]:
    return {
        "validation_natural": scale_quotas(
            NATURAL_VALIDATION_WEIGHTS,
            config.validation.natural_tokens,
        ),
        "validation_alignment": scale_quotas(
            {name: 1 for name in ALIGNMENT_VALIDATION_GROUPS},
            config.validation.alignment_tokens,
        ),
    }


def _write_split(
    connection: sqlite3.Connection,
    split: str,
    output_path: Path,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    records = 0
    tokens = 0
    categories: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    try:
        with temporary.open("wt", encoding="utf-8", newline="\n") as file:
            for category, token_count, row_json in split_rows(connection, split):
                row = materialize_row(str(category), int(token_count), str(row_json))
                file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                file.write("\n")
                records += 1
                tokens += int(token_count)
                categories[str(category)] += int(token_count)
                sources[str(row.get("source", "unknown"))] += int(token_count)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "records": records,
        "tokens": tokens,
        "tokens_by_category": dict(sorted(categories.items())),
        "tokens_by_source": dict(sorted(sources.items())),
        "selected_inventory": selection_inventory(connection, split),
    }


def export_validation_sets(
    config: PipelineConfig,
    database_path: Path,
    *,
    natural_output_path: Path | None = None,
    alignment_output_path: Path | None = None,
    report_path: Path | None = None,
    overwrite: bool = False,
    reservation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    natural_output_path = (
        natural_output_path
        or config.phase_root / "validation" / "validation_natural.jsonl"
    )
    alignment_output_path = (
        alignment_output_path
        or config.phase_root / "validation" / "validation_alignment.jsonl"
    )
    report_path = report_path or config.reports_dir / "phase1_validation_report.json"
    outputs = (natural_output_path, alignment_output_path, report_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"validation outputs already exist: {existing}; pass --overwrite")
    if not database_path.is_file():
        raise FileNotFoundError(f"candidate index does not exist: {database_path}")

    connection = sqlite3.connect(database_path)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if not {"parent_assignments", "selections"} <= tables:
            raise RuntimeError(
                "candidate index predates parent-exclusive selection; rerun sample with --overwrite"
            )
        selected = int(
            connection.execute(
                "SELECT COUNT(*) FROM selections WHERE split LIKE 'validation_%'"
            ).fetchone()[0]
        )
        if selected == 0:
            raise RuntimeError(
                "no reserved validation selection exists; run the Phase 1 sample action first"
            )

        overlap = parent_split_overlap(connection)
        natural = _write_split(
            connection,
            "validation_natural",
            natural_output_path,
        )
        alignment = _write_split(
            connection,
            "validation_alignment",
            alignment_output_path,
        )
    finally:
        connection.close()

    targets = _expected_targets(config)
    checks: dict[str, Any] = {}
    passed = overlap == 0
    for split, result in (
        ("validation_natural", natural),
        ("validation_alignment", alignment),
    ):
        category_checks: dict[str, Any] = {}
        split_passed = True
        for category, target in targets[split].items():
            actual = int(result["tokens_by_category"].get(category, 0))
            category_passed = actual >= target
            split_passed = split_passed and category_passed
            category_checks[category] = {
                "target_tokens": target,
                "actual_tokens": actual,
                "passed": category_passed,
            }
        checks[split] = {
            "target_tokens": sum(targets[split].values()),
            "actual_tokens": result["tokens"],
            "records": result["records"],
            "target_reached": split_passed,
            "passed": result["records"] > 0,
            "category_checks": category_checks,
        }
        passed = passed and checks[split]["passed"]

    report = {
        "generated_at": utc_now_iso(),
        "passed": passed,
        "document_level_split": True,
        "parent_overlap_records": overlap,
        "candidate_database": str(database_path),
        "reservation": reservation_report,
        "checks": checks,
        "natural": natural,
        "alignment": alignment,
    }
    write_json(report_path, report)
    return report


def build_validation_set(
    config: PipelineConfig,
    database_path: Path,
    *,
    overwrite: bool = False,
    natural_output_path: Path | None = None,
    alignment_output_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    return export_validation_sets(
        config,
        database_path,
        natural_output_path=natural_output_path,
        alignment_output_path=alignment_output_path,
        report_path=report_path,
        overwrite=overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/data_factory/phase1_config.json"),
    )
    parser.add_argument("--database-path", type=Path)
    parser.add_argument("--natural-output-path", type=Path)
    parser.add_argument("--alignment-output-path", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    database_path = (
        args.database_path or config.final_dir / "candidate_index.sqlite"
    ).resolve()
    report = build_validation_set(
        config=config,
        database_path=database_path,
        overwrite=args.overwrite,
        natural_output_path=(
            args.natural_output_path.resolve()
            if args.natural_output_path
            else None
        ),
        alignment_output_path=(
            args.alignment_output_path.resolve()
            if args.alignment_output_path
            else None
        ),
        report_path=args.report_path.resolve() if args.report_path else None,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "natural": report["natural"]["output_path"],
                "alignment": report["alignment"]["output_path"],
                "report_path": str(
                    (args.report_path or config.reports_dir / "phase1_validation_report.json").resolve()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
