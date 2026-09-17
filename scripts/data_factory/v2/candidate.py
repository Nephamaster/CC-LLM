"""DataTrove pipeline for one-pass materialization of a planned candidate pool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import ParquetWriter

from scripts.data_factory.v2.config import DataFactoryConfig
from scripts.data_factory.v2.runtime import TaskOutputGuard, build_executor
from scripts.data_factory.v2.sampling import (
    CACHE_COLUMNS,
    eligible_buckets,
    load_new_characters,
    stable_fraction,
    stable_key,
)


def candidate_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("id", pa.string()),
            ("text", pa.large_string()),
            ("parent_doc_id", pa.string()),
            ("source", pa.string()),
            ("subset", pa.string()),
            ("source_path", pa.string()),
            ("revision", pa.string()),
            ("license", pa.string()),
            ("url", pa.string()),
            ("language", pa.string()),
            ("domain", pa.string()),
            ("char_count", pa.int64()),
            ("hanzi_count", pa.int64()),
            ("latin_count", pa.int64()),
            ("digit_count", pa.int64()),
            ("quality_prior", pa.float64()),
            ("tags", pa.list_(pa.string())),
            ("metadata_json", pa.large_string()),
            ("candidate_bucket", pa.string()),
            ("estimated_tokens", pa.int64()),
            ("sample_key", pa.uint64()),
            ("plan_sha256", pa.string()),
        ]
    )


class PlannedCacheReader(PipelineStep):
    name = "Planned cache reader"
    type = "Reader"

    def __init__(self, selected_files: list[dict[str, Any]]) -> None:
        super().__init__()
        self.selected_files = tuple(
            (str(item["source"]), str(item["path"]))
            for item in selected_files
        )

    def run(
        self,
        data: DocumentsPipeline = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[Document]:
        del data
        import pyarrow.parquet as pq

        for planned_source, path_value in self.selected_files[rank::world_size]:
            path = Path(path_value)
            self.stat_update("input_files")
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=4096, columns=CACHE_COLUMNS):
                for row in batch.to_pylist():
                    source = str(row["source"])
                    if source != planned_source:
                        raise RuntimeError(
                            f"planned source {planned_source} does not match cache row source {source}"
                        )
                    metadata = {key: value for key, value in row.items() if key not in {"id", "text"}}
                    self.stat_update("documents")
                    yield Document(id=str(row["id"]), text=str(row["text"]), metadata=metadata)


class CandidateSelector(PipelineStep):
    name = "Candidate selector"
    type = "Filter"

    def __init__(
        self,
        config: DataFactoryConfig,
        plan: dict[str, Any],
        calibration: dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        self.plan_sha256 = str(plan["plan_sha256"])
        self.sampling_rates = {
            str(bucket): {str(source): float(rate) for source, rate in values.items()}
            for bucket, values in plan["sampling_rates"].items()
        }
        self.tokens_per_character = {
            str(source): float(stats["tokens_per_character"])
            for source, stats in calibration["sources"].items()
        }
        self.bucket_densities = {
            str(source): {
                str(bucket): float(stats["tokens_per_character"])
                for bucket, stats in source_stats.get("buckets", {}).items()
                if stats.get("tokens_per_character", 0) > 0
            }
            for source, source_stats in calibration["sources"].items()
        }
        self.new_characters = load_new_characters(config.enhancement.token_ids_path)
        self.seed = config.seed

    def run(
        self,
        data: DocumentsPipeline,
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[Document]:
        del rank, world_size
        for document in data:
            self.stat_update("documents")
            eligible = eligible_buckets(
                self.config,
                document.text,
                document.metadata,
                self.new_characters,
            )
            if not eligible:
                self.stat_update("dropped_no_bucket")
                continue
            source = str(document.metadata["source"])
            if not any(
                stable_fraction(self.seed, f"candidate:{name}:{source}", document.id)
                < self.sampling_rates.get(name, {}).get(source, 0.0)
                for name in eligible
            ):
                self.stat_update("dropped_sampling")
                continue
            bucket = eligible[0]
            document.metadata.update(
                {
                    "candidate_bucket": bucket,
                    "estimated_tokens": max(
                        1,
                        int(
                            round(
                                int(document.metadata["char_count"])
                                * self.bucket_densities.get(source, {}).get(
                                    bucket, self.tokens_per_character[source]
                                )
                            )
                        ),
                    ),
                    "sample_key": stable_key(self.seed, f"candidate-output:{bucket}", document.id),
                    "plan_sha256": self.plan_sha256,
                }
            )
            self.stat_update("forwarded")
            self.stat_update(f"forwarded_{bucket}")
            self.stat_update(f"tokens_{bucket}_{source}", value=document.metadata["estimated_tokens"])
            yield document


def build_candidate_executor(
    config: DataFactoryConfig,
    plan: dict[str, Any],
    calibration: dict[str, Any],
    *,
    executor: str,
    workers: int,
    tasks: int | None = None,
    slurm_partition: str | None = None,
    slurm_time: str = "12:00:00",
    cpus_per_task: int = 1,
    mem_per_cpu_gb: int = 4,
    venv_path: str | None = None,
):
    if not plan.get("passed", False):
        raise RuntimeError(f"candidate plan has shortfalls: {plan.get('shortfalls', [])}")
    selected_files = list(plan.get("selected_files", []))
    if not selected_files:
        raise RuntimeError("candidate plan selected no cache files")
    task_count = min(tasks or len(selected_files), len(selected_files))
    output_dir = config.run_root / "candidates" / str(plan["plan_sha256"])[:16]
    logging_dir = config.run_root / "logs" / "candidate" / str(plan["plan_sha256"])[:16]
    pipeline = [
        PlannedCacheReader(selected_files),
        CandidateSelector(config, plan, calibration),
        TaskOutputGuard(output_dir),
        ParquetWriter(
            output_folder=str(output_dir),
            output_filename="${rank}.parquet",
            compression="zstd",
            batch_size=2_000,
            expand_metadata=True,
            max_file_size=512 * 2**20,
            schema=candidate_schema(),
        ),
    ]
    return build_executor(
        pipeline=pipeline,
        logging_dir=logging_dir,
        job_name=f"cc_candidate_{config.phase}",
        executor=executor,
        tasks=task_count,
        workers=workers,
        slurm_partition=slurm_partition,
        slurm_time=slurm_time,
        cpus_per_task=cpus_per_task,
        mem_per_cpu_gb=mem_per_cpu_gb,
        venv_path=venv_path,
    )


def run_candidate(
    config: DataFactoryConfig,
    plan: dict[str, Any],
    calibration: dict[str, Any],
    **executor_options: Any,
) -> dict[str, Any]:
    pipeline_executor = build_candidate_executor(
        config,
        plan,
        calibration,
        **executor_options,
    )
    pipeline_executor.run()
    return {
        "stage": "candidate",
        "phase": config.phase,
        "run_id": config.run_id,
        "plan_sha256": plan["plan_sha256"],
        "selected_files": len(plan["selected_files"]),
        "executor": executor_options["executor"],
    }
