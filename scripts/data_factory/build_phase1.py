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
        "selected_file_count",
        "selected_bytes",
        "shortfalls",
        "totals",
    )
    values = {key: report[key] for key in keys if key in report}
    print(json.dumps({"action": action, **values}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare", "dedup", "sample", "validation", "all",
            "cache", "calibrate", "plan", "fast_sample", "fast_all",
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("scripts/data_factory/phase1_config.json"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted processing stage")
    parser.add_argument("--workers", type=int, help="override worker processes for prepare, dedup, or sample")
    parser.add_argument(
        "--candidate-tokens",
        type=int,
        help="estimated token budget sampled from normalized data (default: config value, 1.1B)",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="prepare only the named source; repeat to select multiple sources",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source and args.action not in {"prepare", "cache"}:
        raise SystemExit("--source can only be used with prepare or cache")
    if args.resume and args.action in {"validation", "calibrate", "plan", "fast_sample", "fast_all"}:
        raise SystemExit(f"--resume is not supported by the {args.action} action")
    if args.workers is not None and args.action == "validation":
        raise SystemExit("--workers is not supported by the validation action")
    if args.workers is not None and args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.candidate_tokens is not None and args.candidate_tokens <= 0:
        raise SystemExit("--candidate-tokens must be positive")
    if args.candidate_tokens is not None and args.action not in {"sample", "all"}:
        raise SystemExit("--candidate-tokens can only be used with sample or all")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite cannot be used together")
    config = load_config(args.config)
    if args.action in {"cache", "fast_all"}:
        from scripts.data_factory.source_cache import build_source_cache

        source_names = set(args.source) if args.source else None
        _summary(
            "cache",
            build_source_cache(
                config,
                overwrite=args.overwrite,
                source_names=source_names,
                workers=args.workers,
            ),
        )
    if args.action in {"calibrate", "fast_all"}:
        from scripts.data_factory.token_calibration import calibrate_tokens

        _summary(
            "calibrate",
            calibrate_tokens(
                config,
                overwrite=args.overwrite,
                workers=args.workers,
            ),
        )
    if args.action == "plan":
        from scripts.data_factory.fast_sample import _candidate_targets
        from scripts.data_factory.sampling_plan import build_sampling_plan

        _summary(
            "plan",
            build_sampling_plan(
                config,
                _candidate_targets(config),
                overwrite=args.overwrite,
            ),
        )
    if args.action in {"fast_sample", "fast_all"}:
        from scripts.data_factory.fast_sample import build_fast_phase1

        _summary(
            "fast_sample",
            build_fast_phase1(
                config,
                overwrite=args.overwrite,
                workers=args.workers,
            ),
        )
    if args.action in {"prepare", "all"}:
        from scripts.data_factory.prepare import prepare_sources

        source_names = set(args.source) if args.source else None
        _summary(
            "prepare",
            prepare_sources(
                config,
                overwrite=args.overwrite,
                source_names=source_names,
                resume=args.resume,
                workers=args.workers,
            ),
        )
    if args.action == "dedup":
        from scripts.data_factory.dedup import deduplicate

        _summary(
            "dedup",
            deduplicate(
                config,
                overwrite=args.overwrite,
                resume=args.resume,
                workers=args.workers,
            ),
        )
    if args.action in {"sample", "all"}:
        from scripts.data_factory.sample import sample_phase1

        _summary(
            "sample",
            sample_phase1(
                config,
                overwrite=args.overwrite,
                resume=args.resume,
                workers=args.workers,
                candidate_tokens=args.candidate_tokens,
            ),
        )
    if args.action == "validation":
        from scripts.data_factory.build_phase1_validation import build_validation_set

        _summary(
            "validation",
            build_validation_set(
                config,
                config.final_dir / "candidate_index.sqlite",
                overwrite=args.overwrite,
            ),
        )


if __name__ == "__main__":
    main()

