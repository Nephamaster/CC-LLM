"""Normalize and filter all configured Phase 1 sources."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from typing import Any

from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.io_utils import JsonlShardWriter, file_sha256, utc_now_iso, write_json
from scripts.data_factory.sources import source_iterators
from scripts.data_factory.text import clean_record


def _clear_jsonl(directory) -> None:
    if directory.exists():
        for path in directory.glob("*.jsonl"):
            path.unlink()


def prepare_sources(config: PipelineConfig, overwrite: bool = False) -> dict[str, Any]:
    config.raw_manifest_dir.mkdir(parents=True, exist_ok=True)
    config.normalized_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    existing = list(config.normalized_dir.glob("*.jsonl"))
    if existing and not overwrite:
        raise FileExistsError(f"normalized data already exists under {config.normalized_dir}; pass --overwrite")
    if overwrite:
        _clear_jsonl(config.normalized_dir)

    shutil.copyfile(config.path, config.raw_manifest_dir / "phase1_config.snapshot.json")
    rejection_path = config.reports_dir / "prepare_rejections.jsonl"
    rejection_file = rejection_path.open("wt", encoding="utf-8", newline="\n")
    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "config": str(config.path),
        "config_sha256": file_sha256(config.path),
        "sources": {},
        "totals": {"input": 0, "kept": 0, "rejected": 0},
    }

    try:
        for source_name, rows in source_iterators(config):
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
            rejected_count = input_count - kept_count
            report["sources"][source_name] = {
                "input": input_count,
                "kept": kept_count,
                "rejected": rejected_count,
                "reasons": dict(sorted(reasons.items())),
                "files": writer.files,
            }
            report["totals"]["input"] += input_count
            report["totals"]["kept"] += kept_count
            report["totals"]["rejected"] += rejected_count
    finally:
        rejection_file.close()

    report["rejection_log"] = str(rejection_path)
    write_json(config.reports_dir / "source_inventory.json", report)
    write_json(
        config.raw_manifest_dir / "source_manifest.json",
        {
            "generated_at": report["generated_at"],
            "config_sha256": report["config_sha256"],
            "sources": config.sources,
        },
    )
    return report

