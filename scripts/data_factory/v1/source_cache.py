"""Build a reusable normalized Parquet cache from configured Phase 1 sources."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.fast_common import FAST_PIPELINE_VERSION, json_dumps, write_fast_report
from scripts.data_factory.prepare import PrepareTask, _build_tasks, _iter_task_rows
from scripts.data_factory.text import clean_record, mixed_language_stats


CACHE_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("text", pa.large_string()),
        ("source", pa.string()),
        ("category", pa.string()),
        ("quota_group", pa.string()),
        ("license", pa.string()),
        ("license_status", pa.string()),
        ("path", pa.string()),
        ("subset", pa.string()),
        ("char_count", pa.int64()),
        ("hanzi_count", pa.int64()),
        ("latin_count", pa.int64()),
        ("digit_count", pa.int64()),
        ("meta_json", pa.large_string()),
    ]
)


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_")


def _cache_row(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row["text"])
    stats = mixed_language_stats(text)
    hot = {
        "doc_id",
        "text",
        "source",
        "category",
        "quota_group",
        "license",
        "license_status",
        "path",
        "subset",
    }
    meta = {key: value for key, value in row.items() if key not in hot}
    return {
        "doc_id": str(row["doc_id"]),
        "text": text,
        "source": str(row.get("source", "unknown")),
        "category": str(row.get("category", "unknown")),
        "quota_group": None if row.get("quota_group") is None else str(row.get("quota_group")),
        "license": str(row.get("license", "unknown")),
        "license_status": str(row.get("license_status", "unknown")),
        "path": None if row.get("path") is None else str(row.get("path")),
        "subset": None if row.get("subset") is None else str(row.get("subset")),
        "char_count": len(text),
        "hanzi_count": int(stats["hanzi"]),
        "latin_count": int(stats["latin_chars"]),
        "digit_count": sum(char.isdigit() for char in text),
        "meta_json": json_dumps(meta),
    }


def _write_cache_task(
    config: PipelineConfig,
    task: PrepareTask,
    cache_root: Path,
    max_rows: int,
    compression: str,
) -> dict[str, Any]:
    source_dir = cache_root / _safe(task.source_name)
    source_dir.mkdir(parents=True, exist_ok=True)
    prefix = task.output_prefix
    for existing in source_dir.glob(f"{prefix}-*.parquet"):
        existing.unlink()

    writer: pq.ParquetWriter | None = None
    output_path: Path | None = None
    shard_index = 0
    rows_in_shard = 0
    buffer: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    input_count = 0
    kept = 0
    reasons: Counter[str] = Counter()

    def flush(force_close: bool = False) -> None:
        nonlocal writer, output_path, shard_index, rows_in_shard, buffer
        if buffer:
            if writer is None:
                output_path = source_dir / f"{prefix}-{shard_index:05d}.parquet"
                writer = pq.ParquetWriter(
                    output_path,
                    CACHE_SCHEMA,
                    compression=compression,
                    use_dictionary=True,
                    write_statistics=True,
                )
            table = pa.Table.from_pylist(buffer, schema=CACHE_SCHEMA)
            writer.write_table(table)
            rows_in_shard += len(buffer)
            buffer = []
        if writer is not None and (force_close or rows_in_shard >= max_rows):
            assert output_path is not None
            writer.close()
            files.append(
                {
                    "path": str(output_path),
                    "rows": rows_in_shard,
                    "bytes": output_path.stat().st_size,
                }
            )
            writer = None
            output_path = None
            rows_in_shard = 0
            shard_index += 1

    try:
        for raw in _iter_task_rows(config, task):
            input_count += 1
            cleaned, reason = clean_record(raw, config.quality)
            if cleaned is None:
                reasons[str(reason)] += 1
                continue
            buffer.append(_cache_row(cleaned))
            kept += 1
            if len(buffer) >= min(4096, max_rows):
                flush()
            if rows_in_shard >= max_rows:
                flush(force_close=True)
    finally:
        flush(force_close=True)

    return {
        "task": task.key,
        "source": task.source_name,
        "fingerprint": task.fingerprint,
        "input_records": input_count,
        "kept_records": kept,
        "removed_records": input_count - kept,
        "removed_by_reason": dict(sorted(reasons.items())),
        "files": files,
    }


def build_source_cache(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    source_names: set[str] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    fast = config.fast_pipeline
    cache_root = config.fast_cache_dir
    report_path = config.reports_dir / "phase1_fast_cache_report.json"
    worker_count = config.prepare_workers if workers is None else workers
    if worker_count <= 0:
        raise ValueError("workers must be positive")

    tasks, selected_sources = _build_tasks(config, source_names)
    if overwrite and source_names is None:
        shutil.rmtree(cache_root, ignore_errors=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Resume is deliberately artifact based: a task is reused only when every
    # file listed in the previous report still exists and its fingerprint is unchanged.
    previous: dict[str, Any] = {}
    if report_path.is_file() and not overwrite:
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
    previous_tasks = {
        str(row.get("task")): row
        for row in previous.get("tasks", [])
        if isinstance(row, dict)
    }

    # Preserve already-completed tasks from other sources when the user builds
    # the cache incrementally with repeated --source flags.
    selected_keys = {task.key for task in tasks}
    results: list[dict[str, Any]] = [
        row
        for key, row in previous_tasks.items()
        if key not in selected_keys
        and all(Path(item["path"]).is_file() for item in row.get("files", []))
    ]
    pending: list[PrepareTask] = []
    for task in tasks:
        old = previous_tasks.get(task.key)
        if (
            old
            and old.get("fingerprint") == task.fingerprint
            and all(Path(item["path"]).is_file() for item in old.get("files", []))
        ):
            results.append(old)
        else:
            pending.append(task)

    if pending:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _write_cache_task,
                    config,
                    task,
                    cache_root,
                    fast.cache_rows_per_shard,
                    fast.parquet_compression,
                ): task
                for task in pending
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"cache {result['task']}: kept={result['kept_records']:,}/"
                    f"{result['input_records']:,}"
                )

    results.sort(key=lambda row: str(row["task"]))
    totals = {
        "input_records": sum(int(row["input_records"]) for row in results),
        "kept_records": sum(int(row["kept_records"]) for row in results),
        "removed_records": sum(int(row["removed_records"]) for row in results),
        "parquet_files": sum(len(row.get("files", [])) for row in results),
        "bytes": sum(
            int(item.get("bytes", 0))
            for row in results
            for item in row.get("files", [])
        ),
    }
    return write_fast_report(
        report_path,
        {
            "stage": "source_cache",
            "cache_schema_version": FAST_PIPELINE_VERSION,
            "cache_root": str(cache_root),
            "sources": selected_sources,
            "workers": worker_count,
            "tasks": results,
            "totals": totals,
        },
    )
