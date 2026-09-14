"""Build a deterministic file-level candidate sampling plan from calibration data."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.fast_common import json_dumps, stable_key, write_fast_report


PLAN_VERSION = "phase1_sampling_plan_v1"
CHINESE_INTENTS = frozenset(
    {"chinese_natural", "mixed_zh_en", "multi_hanzi_bridge", "new_hanzi_coverage"}
)
FEATURE_INTENTS = frozenset({"multi_hanzi_bridge", "new_hanzi_coverage"})


def allocate_source_targets(
    available: dict[str, int],
    target: int,
    cap_ratio: float | None,
) -> dict[str, int]:
    """Allocate an intent target proportionally while respecting feasible caps."""
    positive = {source: max(0, int(tokens)) for source, tokens in available.items() if tokens > 0}
    if not positive or target <= 0:
        return {}
    target = min(int(target), sum(positive.values()))
    cap: int | None = None
    if cap_ratio is not None and len(positive) >= math.ceil(1 / cap_ratio):
        cap = max(1, int(target * cap_ratio))

    remaining_sources = set(positive)
    result = {source: 0 for source in positive}
    remaining_target = target
    while remaining_sources and remaining_target > 0:
        available_total = sum(positive[source] for source in remaining_sources)
        if available_total <= 0:
            break
        saturated = False
        for source in list(remaining_sources):
            share = remaining_target * positive[source] / available_total
            limit = min(positive[source], cap) if cap is not None else positive[source]
            if share >= limit:
                result[source] = int(limit)
                remaining_target -= int(limit)
                remaining_sources.remove(source)
                saturated = True
        if saturated:
            continue

        floors: dict[str, int] = {}
        remainders: list[tuple[float, str]] = []
        for source in remaining_sources:
            raw = remaining_target * positive[source] / available_total
            value = min(positive[source], int(math.floor(raw)))
            floors[source] = value
            remainders.append((raw - value, source))
        for source, value in floors.items():
            result[source] = value
        missing = remaining_target - sum(floors.values())
        for _fraction, source in sorted(remainders, reverse=True):
            if missing <= 0:
                break
            if result[source] < positive[source]:
                result[source] += 1
                missing -= 1
        remaining_target = 0
    return {source: tokens for source, tokens in result.items() if tokens > 0}


def _cache_entries(cache_report: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for task in cache_report.get("tasks", []):
        source_group = str(task["source"])
        for item in task.get("files", []):
            path = Path(item["path"])
            if not path.is_file():
                raise FileNotFoundError(f"cached Parquet file is missing: {path}")
            entries.append(
                {
                    "source_group": source_group,
                    "task": str(task["task"]),
                    "task_fingerprint": str(task.get("fingerprint", "")),
                    "path": str(path),
                    "rows": int(item.get("rows", 0)),
                    "bytes": int(item.get("bytes", path.stat().st_size)),
                }
            )
    return entries


def _fingerprint(
    config: PipelineConfig,
    targets: dict[str, int],
    entries: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> str:
    payload = {
        "version": PLAN_VERSION,
        "seed": config.seed,
        "targets": targets,
        "oversample_ratio": config.fast_pipeline.oversample_ratio,
        "feature_oversample_ratio": config.fast_pipeline.feature_oversample_ratio,
        "source_cap_ratio": config.fast_pipeline.source_cap_ratio,
        "files": [
            (row["source_group"], row["path"], row["rows"], row["task_fingerprint"])
            for row in entries
        ],
        "calibration": calibration.get("source_groups", {}),
    }
    return hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()


def build_sampling_plan(
    config: PipelineConfig,
    targets: dict[str, int],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    cache_report_path = config.reports_dir / "phase1_fast_cache_report.json"
    if not cache_report_path.is_file():
        raise FileNotFoundError(f"cache report is missing: {cache_report_path}")
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    calibration = json.loads(config.fast_calibration_path.read_text(encoding="utf-8"))
    entries = _cache_entries(cache_report)
    if not entries:
        raise RuntimeError("cache report contains no Parquet files")

    plan_fingerprint = _fingerprint(config, targets, entries, calibration)
    if config.fast_plan_path.is_file() and not overwrite:
        previous = json.loads(config.fast_plan_path.read_text(encoding="utf-8"))
        if previous.get("plan_fingerprint") == plan_fingerprint:
            return previous

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["source_group"])].append(entry)

    calibration_groups = calibration.get("source_groups", {})
    group_capacity: dict[str, int] = {}
    intent_rates: dict[str, dict[str, float]] = defaultdict(dict)
    for source_group, files in grouped.items():
        group_stats = calibration_groups.get(source_group)
        if not isinstance(group_stats, dict) or int(group_stats.get("sampled_documents", 0)) <= 0:
            continue
        tokens_per_document = float(group_stats.get("tokens_per_document", 0.0))
        if tokens_per_document <= 0:
            continue
        group_capacity[source_group] = int(
            round(sum(int(item["rows"]) for item in files) * tokens_per_document)
        )
        for intent, stats in group_stats.get("intents", {}).items():
            rate = float(stats.get("token_rate", 0.0))
            if rate > 0:
                intent_rates[str(intent)][source_group] = rate

    available_by_intent: dict[str, dict[str, int]] = {}
    allocations: dict[str, dict[str, int]] = {}
    shortfalls: dict[str, dict[str, int]] = {}
    for intent, target in targets.items():
        available = {
            source_group: int(group_capacity[source_group] * rate)
            for source_group, rate in intent_rates.get(intent, {}).items()
            if source_group in group_capacity
        }
        available_by_intent[intent] = available
        cap = config.fast_pipeline.source_cap_ratio if intent in CHINESE_INTENTS else None
        allocation = allocate_source_targets(available, target, cap)
        allocations[intent] = allocation
        allocated = sum(allocation.values())
        if allocated < target:
            shortfalls[intent] = {
                "target_tokens": int(target),
                "estimated_available_tokens": sum(available.values()),
                "allocated_tokens": allocated,
                "missing_tokens": int(target) - allocated,
            }

    base_required: dict[str, float] = defaultdict(float)
    feature_required: dict[str, float] = defaultdict(float)
    for intent, allocation in allocations.items():
        buffer = (
            config.fast_pipeline.feature_oversample_ratio
            if intent in FEATURE_INTENTS
            else config.fast_pipeline.oversample_ratio
        )
        for source_group, target in allocation.items():
            rate = intent_rates[intent][source_group]
            required = target * buffer / rate
            if intent in FEATURE_INTENTS:
                feature_required[source_group] = max(feature_required[source_group], required)
            else:
                base_required[source_group] += required

    selected_files: list[dict[str, Any]] = []
    selected_capacity: dict[str, int] = {}
    for source_group, capacity in sorted(group_capacity.items()):
        required = min(capacity, max(base_required[source_group], feature_required[source_group]))
        if required <= 0:
            continue
        group_stats = calibration_groups[source_group]
        tokens_per_document = float(group_stats["tokens_per_document"])
        ordered = sorted(
            grouped[source_group],
            key=lambda row: stable_key(
                config.seed,
                f"sampling-plan-file:{source_group}",
                str(row["path"]),
            ),
        )
        accumulated = 0
        for entry in ordered:
            estimated_tokens = int(round(int(entry["rows"]) * tokens_per_document))
            selected_files.append({**entry, "estimated_tokens": estimated_tokens})
            accumulated += estimated_tokens
            if accumulated >= required:
                break
        selected_capacity[source_group] = accumulated

    sampling_rates: dict[str, dict[str, float]] = {}
    for intent, allocation in allocations.items():
        buffer = (
            config.fast_pipeline.feature_oversample_ratio
            if intent in FEATURE_INTENTS
            else config.fast_pipeline.oversample_ratio
        )
        sampling_rates[intent] = {
            source_group: min(
                1.0,
                buffer
                * target
                / max(1.0, selected_capacity.get(source_group, 0) * intent_rates[intent][source_group]),
            )
            for source_group, target in allocation.items()
            if selected_capacity.get(source_group, 0) > 0
        }

    selected_files.sort(key=lambda row: (str(row["source_group"]), str(row["path"])))
    config.fast_plan_dir.mkdir(parents=True, exist_ok=True)
    return write_fast_report(
        config.fast_plan_path,
        {
            "stage": "sampling_plan",
            "plan_version": PLAN_VERSION,
            "plan_fingerprint": plan_fingerprint,
            "passed": not shortfalls,
            "targets": targets,
            "available_estimated_tokens": available_by_intent,
            "source_allocations": allocations,
            "sampling_rates": sampling_rates,
            "source_capacity_tokens": group_capacity,
            "selected_capacity_tokens": selected_capacity,
            "selected_files": selected_files,
            "selected_file_count": len(selected_files),
            "selected_bytes": sum(int(row["bytes"]) for row in selected_files),
            "shortfalls": shortfalls,
        },
    )
