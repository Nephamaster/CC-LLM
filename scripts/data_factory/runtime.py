"""Shared DataTrove executor and task-output lifecycle helpers."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Iterator

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep


class DistributionStats(PipelineStep):
    """Record output quantities by source and bucket for replenishment audits."""

    name = "Bucket and source output"
    type = "Stats"

    def run(self, data, rank=0, world_size=1):
        for document in data:
            key = f"{document.metadata.get('candidate_bucket', 'unknown')}/{document.metadata.get('source', 'unknown')}"
            self.stat_update(f"documents/{key}")
            self.stat_update(f"tokens/{key}", value=int(document.metadata.get("estimated_tokens", 0)))
            yield document


class TaskOutputGuard(PipelineStep):
    """Remove incomplete Parquet files before an unfinished task restarts."""

    name = "Task output guard"
    type = "Writer preparation"

    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self.output_dir = output_dir

    def run(
        self,
        data: DocumentsPipeline,
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[Document]:
        del world_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rank_name = str(rank).zfill(5)
        candidates = list(self.output_dir.rglob(f"{rank_name}.parquet"))
        candidates.extend(self.output_dir.rglob(f"*_{rank_name}.parquet"))
        for path in candidates:
            if path.is_file():
                path.unlink()
        yield from data


def build_executor(
    *,
    pipeline: list[Any],
    logging_dir: Path,
    job_name: str,
    executor: str,
    tasks: int,
    workers: int,
    slurm_partition: str | None = None,
    slurm_time: str = "12:00:00",
    cpus_per_task: int = 1,
    mem_per_cpu_gb: int = 4,
    venv_path: str | None = None,
    depends: Any = None,
):
    if tasks <= 0 or workers <= 0:
        raise ValueError("tasks and workers must be positive")
    logging_dir.mkdir(parents=True, exist_ok=True)
    layout_path = logging_dir / "task_layout.json"
    layout = {"tasks": tasks}
    if layout_path.is_file() and json.loads(layout_path.read_text()) != layout:
        raise ValueError(f"task sharding changed for existing completion markers: {logging_dir}")
    layout_path.write_text(json.dumps(layout) + "\n")
    workers = min(workers, tasks)
    if executor == "local":
        from datatrove.executor import LocalPipelineExecutor

        return LocalPipelineExecutor(
            pipeline=pipeline,
            logging_dir=str(logging_dir),
            tasks=tasks,
            workers=workers,
        )
    if executor == "slurm":
        if not slurm_partition:
            raise ValueError("--slurm-partition is required for the Slurm executor")
        from datatrove.executor import SlurmPipelineExecutor

        kwargs: dict[str, Any] = {}
        if venv_path:
            kwargs["venv_path"] = venv_path
        return SlurmPipelineExecutor(
            pipeline=pipeline,
            logging_dir=str(logging_dir),
            slurm_logs_folder=str(logging_dir / "slurm_logs"),
            job_name=job_name,
            tasks=tasks,
            workers=workers,
            time=slurm_time,
            partition=slurm_partition,
            cpus_per_task=cpus_per_task,
            mem_per_cpu_gb=mem_per_cpu_gb,
            depends=depends,
            **kwargs,
        )
    raise ValueError("executor must be local or slurm")
