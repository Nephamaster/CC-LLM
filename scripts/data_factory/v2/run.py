"""Command-line entrypoint for Data Factory V2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.data_factory.v2.config import DataFactoryConfig, load_data_factory_config
from scripts.data_factory.v2.documents import (
    expand_source_paths,
    inspect_source,
    source_contract_hash,
)


def _active_sources(config: DataFactoryConfig) -> list[str]:
    return sorted({source for bucket in config.buckets for source in bucket.source_weights})


def _selected_sources(config: DataFactoryConfig, requested: list[str] | None) -> list[str]:
    active = set(_active_sources(config))
    if not requested:
        return sorted(active)
    unknown = set(requested) - set(config.source_registry.sources)
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    disallowed = {
        name
        for name in requested
        if config.phase not in config.source_registry.sources[name].phases
    }
    if disallowed:
        raise ValueError(f"sources do not allow {config.phase}: {sorted(disallowed)}")
    return sorted(set(requested))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def inspect_sources(
    config: DataFactoryConfig,
    source_names: list[str],
    *,
    max_files: int,
    max_rows: int,
) -> dict[str, Any]:
    report_dir = config.corpus_root / "metadata" / "inspection"
    reports: dict[str, Any] = {}
    for source_name in source_names:
        report = inspect_source(
            config.source_registry.sources[source_name],
            max_files=max_files,
            max_rows=max_rows,
        )
        _write_json(report_dir / f"{source_name}.json", report)
        reports[source_name] = report
    return {
        "stage": "inspect",
        "passed": all(report["passed"] for report in reports.values()),
        "sources": reports,
    }


def _file_manifest_rows(config: DataFactoryConfig, source_name: str) -> list[dict[str, Any]]:
    source = config.source_registry.sources[source_name]
    rows: list[dict[str, Any]] = []
    for path in expand_source_paths(source):
        stat = path.stat()
        row: dict[str, Any] = {
            "source": source.name,
            "path": str(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "reader": source.reader,
            "adapter": source.adapter,
            "source_contract_sha256": source_contract_hash(source),
            "homepage": source.homepage,
            "rows": None,
            "row_groups": None,
        }
        if source.reader == "parquet":
            import pyarrow.parquet as pq

            metadata = pq.ParquetFile(path).metadata
            row["rows"] = metadata.num_rows
            row["row_groups"] = metadata.num_row_groups
        payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        row["fingerprint"] = hashlib.sha256(payload).hexdigest()
        rows.append(row)
    return rows


def build_manifests(config: DataFactoryConfig, source_names: list[str]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir = config.corpus_root / "registry" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Any] = {}
    combined: list[dict[str, Any]] = []
    for source_name in source_names:
        rows = _file_manifest_rows(config, source_name)
        path = output_dir / f"{source_name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        digest = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        sources[source_name] = {
            "path": str(path),
            "sha256": digest,
            "files": len(rows),
            "bytes": sum(int(row["size_bytes"]) for row in rows),
        }
        combined.extend(rows)
    combined_hash = hashlib.sha256(
        json.dumps(combined, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "stage": "manifest",
        "source_manifest_sha256": combined_hash,
        "sources": sources,
    }
    _write_json(output_dir / f"{config.phase}.manifest.json", report)
    return report


def _load_phase_manifest(config: DataFactoryConfig, source_names: list[str]) -> dict[str, Any]:
    path = config.corpus_root / "registry" / "manifests" / f"{config.phase}.manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"phase manifest is missing: {path}; run the manifest stage")
    report = json.loads(path.read_text(encoding="utf-8"))
    missing = set(source_names) - set(report.get("sources", {}))
    if missing:
        raise RuntimeError(
            f"phase manifest is incomplete for {config.phase}: {sorted(missing)}; "
            "rerun manifest without --source"
        )
    return report


def _require_passed_inspection(config: DataFactoryConfig, source_names: list[str]) -> None:
    report_dir = config.corpus_root / "metadata" / "inspection"
    failures: list[str] = []
    for source_name in source_names:
        path = report_dir / f"{source_name}.json"
        if not path.is_file():
            failures.append(f"{source_name}: missing inspection report")
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        source = config.source_registry.sources[source_name]
        if not report.get("passed", False):
            failures.append(f"{source_name}: inspection failed")
        elif report.get("source_contract_sha256") != source_contract_hash(source):
            failures.append(f"{source_name}: inspection is stale")
    if failures:
        raise RuntimeError("cache requires passed inspection: " + "; ".join(failures))


def _load_run_config(args: argparse.Namespace, source_names: list[str]) -> tuple[DataFactoryConfig, dict[str, Any]]:
    base = load_data_factory_config(args.config, profile=args.profile)
    manifest = _load_phase_manifest(base, source_names)
    config = load_data_factory_config(
        args.config,
        profile=args.profile,
        source_manifest_sha256=manifest["source_manifest_sha256"],
        require_tokenizer=True,
    )
    config.write_snapshot(config.run_root / "config.snapshot.yaml")
    return config, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("inspect", "manifest", "cache", "calibrate", "plan", "candidate", "exact_dedup", "minhash", "decontaminate", "mixture", "tokenize", "finalize"),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=("default", "exact_only"), default="default")
    parser.add_argument("--source", action="append", help="limit inspect/manifest/cache to one source; repeatable")
    parser.add_argument("--max-files", type=int, default=3, help="files inspected per source")
    parser.add_argument("--max-rows", type=int, default=20, help="rows inspected per source")
    parser.add_argument("--executor", choices=("local", "slurm"), default="local")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--slurm-partition")
    parser.add_argument("--slurm-time", default="12:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem-per-cpu-gb", type=int, default=4)
    parser.add_argument("--venv-path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--round", type=int, default=0, dest="round_index")
    return parser.parse_args()


def _executor_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "executor": args.executor,
        "workers": args.workers,
        "tasks": args.tasks,
        "slurm_partition": args.slurm_partition,
        "slurm_time": args.slurm_time,
        "cpus_per_task": args.cpus_per_task,
        "mem_per_cpu_gb": args.mem_per_cpu_gb,
        "venv_path": args.venv_path,
    }


def main() -> None:
    args = parse_args()
    if args.max_files <= 0 or args.max_rows <= 0:
        raise SystemExit("--max-files and --max-rows must be positive")
    if args.workers <= 0 or (args.tasks is not None and args.tasks <= 0):
        raise SystemExit("--workers and --tasks must be positive")
    base_config = load_data_factory_config(args.config, profile=args.profile)
    source_names = _selected_sources(base_config, args.source)
    if args.source and args.stage not in {"inspect", "manifest", "cache"}:
        raise SystemExit(f"--source is not supported by the {args.stage} stage")

    if args.stage == "inspect":
        result = inspect_sources(
            base_config,
            source_names,
            max_files=args.max_files,
            max_rows=args.max_rows,
        )
    elif args.stage in {"manifest", "cache"}:
        manifest = build_manifests(base_config, source_names)
        config = load_data_factory_config(
            args.config,
            profile=args.profile,
            source_manifest_sha256=manifest["source_manifest_sha256"],
        )
        if args.stage == "manifest":
            result = {**manifest, "run_id": config.run_id}
        else:
            from scripts.data_factory.v2.cache import run_cache

            _require_passed_inspection(config, source_names)
            config.write_snapshot(config.run_root / "config.snapshot.yaml")
            runs = run_cache(
                config,
                source_names,
                source_manifests=manifest["sources"],
                **_executor_options(args),
            )
            result = {
                "stage": "cache",
                "run_id": config.run_id,
                "source_manifest_sha256": manifest["source_manifest_sha256"],
                "sources": runs,
            }
    else:
        config, manifest = _load_run_config(args, source_names)
        from scripts.data_factory.v2.sampling import (
            build_plan,
            calibrate,
            calibration_path,
            plan_path,
        )

        if args.stage == "calibrate":
            result = calibrate(
                config,
                manifest["sources"],
                overwrite=args.overwrite,
            )
        else:
            calibration_file = calibration_path(config)
            if not calibration_file.is_file():
                raise FileNotFoundError(
                    f"calibration is missing: {calibration_file}; run calibrate first"
                )
            calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
            if args.stage == "plan":
                result = build_plan(
                    config,
                    calibration,
                    round_index=args.round_index,
                    overwrite=args.overwrite,
                )
            else:
                candidate_plan = plan_path(config, args.round_index)
                if not candidate_plan.is_file():
                    raise FileNotFoundError(
                        f"candidate plan is missing: {candidate_plan}; run plan first"
                    )
                plan = json.loads(candidate_plan.read_text(encoding="utf-8"))
                if args.stage == "candidate":
                    from scripts.data_factory.v2.candidate import run_candidate

                    result = run_candidate(
                        config,
                        plan,
                        calibration,
                        **_executor_options(args),
                    )
                elif args.stage in {"exact_dedup", "minhash", "decontaminate"}:
                    from scripts.data_factory.v2.dedup import (
                        run_decontamination,
                        run_exact_dedup,
                        run_minhash,
                    )

                    actions = {
                        "exact_dedup": run_exact_dedup,
                        "minhash": run_minhash,
                        "decontaminate": run_decontamination,
                    }
                    result = actions[args.stage](
                        config,
                        plan,
                        **_executor_options(args),
                    )
                elif args.stage == "mixture":
                    from scripts.data_factory.v2.mixture import build_mixture

                    result = build_mixture(
                        config,
                        plan,
                        overwrite=args.overwrite,
                    )
                else:
                    from scripts.data_factory.v2.tokenization import (
                        finalize_dataset,
                        tokenize_selected,
                    )

                    action = tokenize_selected if args.stage == "tokenize" else finalize_dataset
                    result = action(
                        config,
                        plan,
                        overwrite=args.overwrite,
                    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
