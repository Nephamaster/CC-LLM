"""Command-line facade for Phase 1 data construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.data_factory.config import load_config


def _summary(action: str, report: dict[str, Any]) -> None:
    keys = (
        "passed",
        "input_records",
        "kept_records",
        "removed_records",
        "target_tokens",
        "actual_tokens",
        "totals",
    )
    values = {key: report[key] for key in keys if key in report}
    print(json.dumps({"action": action, **values}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "dedup", "sample", "all"))
    parser.add_argument("--config", type=Path, default=Path("scripts/data_factory/phase1_config.json"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.action in {"prepare", "all"}:
        from scripts.data_factory.prepare import prepare_sources

        _summary("prepare", prepare_sources(config, overwrite=args.overwrite))
    if args.action in {"dedup", "all"}:
        from scripts.data_factory.dedup import deduplicate

        _summary("dedup", deduplicate(config, overwrite=args.overwrite))
    if args.action in {"sample", "all"}:
        from scripts.data_factory.sample import sample_phase1

        _summary("sample", sample_phase1(config, overwrite=args.overwrite))


if __name__ == "__main__":
    main()

