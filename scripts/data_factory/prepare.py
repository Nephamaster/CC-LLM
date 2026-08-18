"""Normalize configured Phase 1 sources with file-level parallelism and resume."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.io_utils import JsonlShardWriter, expand_paths, file_sha256, utc_now_iso, write_json
from scripts.data_factory.sources import (
    iter_cci3_hq,
    iter_clue,
    iter_external,
    iter_fineweb_chinese,
    iter_fineweb_english,
    iter_wanjuan,
)
from scripts.data_factory.text import clean_record


PREPARE_STATE_VERSION = 2
_FILE_SPLIT_SOURCES = frozenset({"fineweb_chinese", "cci3_hq", "wanjuan", "fineweb_english"})


@dataclass(frozen=True)
class PrepareTask:
    key: str
    source_name: str
    kind: str
    ordinal: int
    source_config: dict[str, Any]
    input_paths: tuple[str, ...]
    output_prefix: str
    fingerprint: str


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _source_specs(config: PipelineConfig) -> list[tuple[str, str, dict[str, Any]]]:
    sources = config.sources
    specs: list[tuple[str, str, dict[str, Any]]] = []
    for source_name in ("clue", "fineweb_chinese", "cci3_hq", "wanjuan", "fineweb_english"):
        source_config = sources.get(source_name)
        if source_config and source_config.get("enabled", True):
            specs.append((source_name, source_name, dict(source_config)))
    for source_config in sources.get("external", []):
        if source_config.get("enabled", True):
            specs.append((str(source_config["name"]), "external", dict(source_config)))
    return specs


def _path_metadata(paths: tuple[str, ...]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        stat = path.stat()
        metadata.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return metadata


def _task_fingerprint(
    config: PipelineConfig,
    source_name: str,
    kind: str,
    source_config: dict[str, Any],
    paths: tuple[str, ...],
) -> str:
    payload = {
        "version": PREPARE_STATE_VERSION,
        "source_name": source_name,
        "kind": kind,
        "source_config": source_config,
        "quality": config.quality,
        "normalized_shard_records": config.normalized_shard_records,
        "inputs": _path_metadata(paths),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_tasks(
    config: PipelineConfig,
    source_names: set[str] | None,
) -> tuple[list[PrepareTask], list[str]]:
    specs = _source_specs(config)
    available = {source_name for source_name, _, _ in specs}
    if source_names is not None:
        unknown = source_names - available
        if unknown:
            raise ValueError(f"unknown Phase 1 sources: {sorted(unknown)}; available: {sorted(available)}")
        specs = [spec for spec in specs if spec[0] in source_names]

    tasks: list[PrepareTask] = []
    selected_sources: list[str] = []
    for source_name, kind, source_config in specs:
        selected_sources.append(source_name)
        if kind == "clue":
            root = Path(source_config["path"])
            if not root.is_absolute():
                root = config.repo_root / root
            input_paths = tuple(str(path.resolve()) for path in sorted(root.glob("*/train-*.parquet")))
            if not input_paths:
                raise FileNotFoundError(f"no CLUE train parquet files found under {root}")
            task_paths = [input_paths]
        else:
            paths = tuple(str(path) for path in expand_paths(source_config.get("paths", []), config.repo_root))
            if not paths and not (kind == "external" and source_config.get("optional", True)):
                raise FileNotFoundError(f"no input files match source {source_name!r}")
            task_paths = [(path,) for path in paths] if kind in _FILE_SPLIT_SOURCES else [paths]

        for ordinal, paths in enumerate(task_paths):
            prefix = f"{_safe_component(source_name)}-t{ordinal:05d}"
            fingerprint = _task_fingerprint(config, source_name, kind, source_config, paths)
            tasks.append(
                PrepareTask(
                    key=prefix,
                    source_name=source_name,
                    kind=kind,
                    ordinal=ordinal,
                    source_config=source_config,
                    input_paths=paths,
                    output_prefix=prefix,
                    fingerprint=fingerprint,
                )
            )
    return tasks, selected_sources


def _iter_task_rows(config: PipelineConfig, task: PrepareTask) -> Iterator[dict[str, Any]]:
    source_config = dict(task.source_config)
    if task.kind in _FILE_SPLIT_SOURCES:
        source_config["paths"] = list(task.input_paths)
    if task.kind == "clue":
        yield from iter_clue(config, source_config)
    elif task.kind == "fineweb_chinese":
        yield from iter_fineweb_chinese(config, source_config)
    elif task.kind == "cci3_hq":
        yield from iter_cci3_hq(config, source_config)
    elif task.kind == "wanjuan":
        yield from iter_wanjuan(config, source_config)
    elif task.kind == "fineweb_english":
        yield from iter_fineweb_english(config, source_config)
    elif task.kind == "external":
        yield from iter_external(config, source_config)
    else:
        raise ValueError(f"unsupported prepare task kind: {task.kind}")


def _process_task(config: PipelineConfig, task: PrepareTask, staging_root: Path) -> dict[str, Any]:
    staging_dir = staging_root / task.key
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    writer = JsonlShardWriter(staging_dir, task.output_prefix, config.normalized_shard_records)
    reasons: Counter[str] = Counter()
    rejection_samples: list[dict[str, Any]] = []
    input_count = 0
    try:
        for row in _iter_task_rows(config, task):
            input_count += 1
            cleaned, reason = clean_record(row, config.quality)
            if cleaned is None:
                reasons[str(reason)] += 1
                if len(rejection_samples) < 100:
                    rejection_samples.append(
                        {"source": task.source_name, "doc_id": row.get("doc_id"), "reason": reason}
                    )
                continue
            writer.write(cleaned)
    finally:
        writer.close()

    return {
        "key": task.key,
        "source": task.source_name,
        "fingerprint": task.fingerprint,
        "input": input_count,
        "kept": writer.total_records,
        "rejected": input_count - writer.total_records,
        "reasons": dict(sorted(reasons.items())),
        "rejection_samples": rejection_samples,
        "files": writer.files,
    }


def _state_suffix(source_names: set[str] | None) -> str:
    if source_names is None:
        return ""
    return "." + "_".join(_safe_component(value) for value in sorted(source_names))


def _run_fingerprint(tasks: list[PrepareTask]) -> str:
    payload = [(task.key, task.fingerprint) for task in tasks]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("rt", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _outputs_exist(result: dict[str, Any]) -> bool:
    for item in result.get("files", []):
        path = Path(str(item["path"]))
        if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
            return False
    return True


def _remove_task_outputs(directory: Path, task: PrepareTask) -> None:
    for path in directory.glob(f"{task.output_prefix}-*.jsonl"):
        path.unlink()


def _commit_task_result(
    directory: Path,
    staging_root: Path,
    task: PrepareTask,
    result: dict[str, Any],
) -> dict[str, Any]:
    _remove_task_outputs(directory, task)
    committed_files: list[dict[str, Any]] = []
    for item in result["files"]:
        source_path = Path(str(item["path"]))
        target_path = directory / source_path.name
        os.replace(source_path, target_path)
        committed = dict(item)
        committed["path"] = str(target_path)
        committed["bytes"] = target_path.stat().st_size
        committed_files.append(committed)
    shutil.rmtree(staging_root / task.key, ignore_errors=True)
    committed_result = dict(result)
    committed_result["files"] = committed_files
    return committed_result


def _existing_outputs(directory: Path, source_names: list[str]) -> list[Path]:
    return sorted(
        path
        for source_name in source_names
        for path in directory.glob(f"{_safe_component(source_name)}-*.jsonl")
    )


def _load_inventory(path: Path | None, config: PipelineConfig) -> dict[str, Any]:
    if path is not None and path.is_file():
        value = _load_json(path)
        if isinstance(value.get("sources"), dict):
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
    *,
    resume: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    if overwrite and resume:
        raise ValueError("overwrite and resume cannot be used together")
    config.raw_manifest_dir.mkdir(parents=True, exist_ok=True)
    config.normalized_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)

    tasks, selected_sources = _build_tasks(config, source_names)
    suffix = _state_suffix(source_names)
    state_root = config.normalized_dir / ".prepare"
    staging_root = state_root / "staging"
    state_path = state_root / f"state{suffix}.json"
    state_root.mkdir(parents=True, exist_ok=True)
    run_fingerprint = _run_fingerprint(tasks)
    existing = (
        sorted(config.normalized_dir.glob("*.jsonl"))
        if source_names is None
        else _existing_outputs(config.normalized_dir, selected_sources)
    )

    if overwrite:
        for path in existing:
            path.unlink()
        state_path.unlink(missing_ok=True)
        if source_names is None:
            shutil.rmtree(staging_root, ignore_errors=True)
        else:
            for task in tasks:
                shutil.rmtree(staging_root / task.key, ignore_errors=True)
    elif resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"prepare resume state is missing: {state_path}; use --overwrite")
    elif existing:
        scope = "selected sources" if source_names is not None else str(config.normalized_dir)
        raise FileExistsError(f"normalized data already exists for {scope}; pass --overwrite or --resume")

    if resume:
        state = _load_json(state_path)
        if state.get("version") != PREPARE_STATE_VERSION:
            raise RuntimeError("prepare state schema changed; restart with --overwrite")
    else:
        state = {
            "version": PREPARE_STATE_VERSION,
            "run_fingerprint": run_fingerprint,
            "generated_at": utc_now_iso(),
            "tasks": {},
        }
        write_json(state_path, state)

    completed: dict[str, dict[str, Any]] = dict(state.get("tasks", {}))
    current_keys = {task.key for task in tasks}
    for obsolete_key in set(completed) - current_keys:
        result = completed.pop(obsolete_key)
        for item in result.get("files", []):
            Path(str(item["path"])).unlink(missing_ok=True)

    state["run_fingerprint"] = run_fingerprint
    state["tasks"] = completed
    state["updated_at"] = utc_now_iso()
    write_json(state_path, state)

    pending: list[PrepareTask] = []
    for task in tasks:
        result = completed.get(task.key)
        if result and result.get("fingerprint") == task.fingerprint and _outputs_exist(result):
            continue
        completed.pop(task.key, None)
        _remove_task_outputs(config.normalized_dir, task)
        shutil.rmtree(staging_root / task.key, ignore_errors=True)
        pending.append(task)

    worker_count = workers or int(getattr(config, "prepare_workers", 8))
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    finished_count = len(tasks) - len(pending)

    def commit(task: PrepareTask, result: dict[str, Any]) -> None:
        nonlocal finished_count
        result = _commit_task_result(config.normalized_dir, staging_root, task, result)
        completed[task.key] = result
        state["tasks"] = completed
        state["updated_at"] = utc_now_iso()
        write_json(state_path, state)
        finished_count += 1
        print(
            f"prepared {finished_count}/{len(tasks)}: {task.input_paths[0] if task.input_paths else task.source_name} "
            f"(kept {result['kept']:,}, rejected {result['rejected']:,})",
            flush=True,
        )

    if worker_count == 1:
        for task in pending:
            commit(task, _process_task(config, task, staging_root))
    elif pending:
        with ProcessPoolExecutor(max_workers=min(worker_count, len(pending))) as executor:
            futures = {
                executor.submit(_process_task, config, task, staging_root): task
                for task in pending
            }
            for future in as_completed(futures):
                task = futures[future]
                commit(task, future.result())

    task_results = [completed[task.key] for task in tasks]
    rejection_path = config.reports_dir / f"prepare_rejections{suffix}.jsonl"
    with rejection_path.open("wt", encoding="utf-8", newline="\n") as rejection_file:
        source_samples: Counter[str] = Counter()
        for result in task_results:
            for sample in result.get("rejection_samples", []):
                source = str(sample["source"])
                if source_samples[source] >= 100:
                    continue
                rejection_file.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                source_samples[source] += 1

    report_path = config.reports_dir / "source_inventory.json"
    report = _load_inventory(report_path if source_names is not None else None, config)
    report.update(
        {
            "generated_at": utc_now_iso(),
            "config": str(config.path),
            "config_sha256": file_sha256(config.path),
            "prepare_workers": worker_count,
            "prepare_state": str(state_path),
        }
    )
    for source_name in selected_sources:
        source_results = [result for result in task_results if result["source"] == source_name]
        reasons: Counter[str] = Counter()
        for result in source_results:
            reasons.update(result.get("reasons", {}))
        report["sources"][source_name] = {
            "input": sum(int(result["input"]) for result in source_results),
            "kept": sum(int(result["kept"]) for result in source_results),
            "rejected": sum(int(result["rejected"]) for result in source_results),
            "reasons": dict(sorted(reasons.items())),
            "files": [item for result in source_results for item in result["files"]],
        }
    _update_totals(report)
    if source_names is None:
        report["rejection_log"] = str(rejection_path)
    else:
        logs = dict(report.get("source_rejection_logs", {}))
        for source_name in selected_sources:
            logs[source_name] = str(rejection_path)
        report["source_rejection_logs"] = logs
    write_json(report_path, report)

    shutil.copyfile(config.path, config.raw_manifest_dir / "phase1_config.snapshot.json")
    write_json(
        config.raw_manifest_dir / "source_manifest.json",
        {
            "generated_at": report["generated_at"],
            "config_sha256": report["config_sha256"],
            "sources": config.sources,
        },
    )
    return report
