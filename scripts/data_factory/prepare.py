"""Normalize and filter configured Phase 1 sources."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.io_utils import JsonlShardWriter, file_sha256, utc_now_iso, write_json
from scripts.data_factory.sources import source_iterators
from scripts.data_factory.text import clean_record


def _output_paths(directory: Path, source_names: set[str] | None) -> list[Path]:
    if not directory.exists():
        return []
    if source_names is None:
        return sorted(directory.glob("*.jsonl"))
    return sorted(path for name in source_names for path in directory.glob(f"{name}-*.jsonl"))


def _load_report(path: Path | None, config: PipelineConfig) -> dict[str, Any]:
    if path is not None and path.is_file():
        with path.open("rt", encoding="utf-8") as file:
            value = json.load(file)
        if isinstance(value, dict) and isinstance(value.get("sources"), dict):
            return value
    return {
        "generated_at": utc_now_iso(),
        "config": str(config.path),
        "config_sha256": file_sha256(config.path),
        "sources": {},
        "totals": {"input": 0, "kept": 0, "rejected": 0},
    }


def _update_totals(report: dict[str, Any]) -> None:
    report["totals"] = {
        key: sum(int(source.get(key, 0)) for source in report["sources"].values())
        for key in ("input", "kept", "rejected")
    }


def prepare_sources(
    config: PipelineConfig,
    overwrite: bool = False,
    source_names: set[str] | None = None,
) -> dict[str, Any]:
    config.raw_manifest_dir.mkdir(parents=True, exist_ok=True)
    config.normalized_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    iterators = source_iterators(config)
    available = {name for name, _ in iterators}
    if source_names is not None:
        unknown = source_names - available
        if unknown:
            raise ValueError(f"unknown Phase 1 sources: {sorted(unknown)}; available: {sorted(available)}")
        iterators = [(name, rows) for name, rows in iterators if name in source_names]

    existing = _output_paths(config.normalized_dir, source_names)
    if existing and not overwrite:
        scope = "selected sources" if source_names is not None else str(config.normalized_dir)
        raise FileExistsError(f"normalized data already exists for {scope}; pass --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()

    shutil.copyfile(config.path, config.raw_manifest_dir / "phase1_config.snapshot.json")
    suffix = "" if source_names is None else "." + "_".join(sorted(source_names))
    rejection_path = config.reports_dir / f"prepare_rejections{suffix}.jsonl"
    rejection_file = rejection_path.open("wt", encoding="utf-8", newline="\n")
    report_path = config.reports_dir / "source_inventory.json"
    report = _load_report(report_path, config) if source_names is not None else _load_report(None, config)
    report.update(
        {
            "generated_at": utc_now_iso(),
            "config": str(config.path),
            "config_sha256": file_sha256(config.path),
        }
    )

    try:
        for source_name, rows in iterators:
            reasons: Counter[str] = Counter()
            input_count = 0
            kept_count = 0
            writer = JsonlShardWriter(config.normalized_dir, source_name, config.normalized_shard_records)
            for row in rows:
                input_count += 1
                cleaned, reason = clean_record(row, config.quality)
                if cleaned is None:
                    reasons[str(reason)] += 1
                    if sum(reasons.values()) <= 100:
                        rejection_file.write(
                            json.dumps(
                                {"source": source_name, "doc_id": row.get("doc_id"), "reason": reason},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                    continue
                writer.write(cleaned)
                kept_count += 1
            writer.close()
            report["sources"][source_name] = {
                "input": input_count,
                "kept": kept_count,
                "rejected": input_count - kept_count,
                "reasons": dict(sorted(reasons.items())),
                "files": writer.files,
            }
    finally:
        rejection_file.close()

    _update_totals(report)
    if source_names is None:
        report["rejection_log"] = str(rejection_path)
    else:
        logs = dict(report.get("source_rejection_logs", {}))
        for source_name in source_names:
            logs[source_name] = str(rejection_path)
        report["source_rejection_logs"] = logs
    write_json(report_path, report)
    write_json(
        config.raw_manifest_dir / "source_manifest.json",
        {
            "generated_at": report["generated_at"],
            "config_sha256": report["config_sha256"],
            "sources": config.sources,
        },
    )
    return report