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
    parser.add_argument("--resume", action="store_true", help="resume an interrupted sample action")
    parser.add_argument("--workers", type=int, help="concurrent tokenizer batches for sample")
    parser.add_argument(
        "--source",
        action="append",
        help="prepare only the named source; repeat to select multiple sources",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source and args.action != "prepare":
        raise SystemExit("--source can only be used with the prepare action")
    if args.resume and args.action != "sample":
        raise SystemExit("--resume can only be used with the sample action")
    if args.workers is not None and args.action != "sample":
        raise SystemExit("--workers can only be used with the sample action")
    if args.workers is not None and args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together")
    config = load_config(args.config)
    if args.action in {"prepare", "all"}:
        from scripts.data_factory.prepare import prepare_sources

        source_names = set(args.source) if args.source else None
        _summary(
            "prepare",
            prepare_sources(config, overwrite=args.overwrite, source_names=source_names),
        )
    if args.action in {"dedup", "all"}:
        from scripts.data_factory.dedup import deduplicate

        _summary("dedup", deduplicate(config, overwrite=args.overwrite))
    if args.action in {"sample", "all"}:
        from scripts.data_factory.sample import sample_phase1

        _summary(
            "sample",
            sample_phase1(
                config,
                overwrite=args.overwrite,
                resume=args.resume,
                workers=args.workers,
            ),
        )


if __name__ == "__main__":
    main()

