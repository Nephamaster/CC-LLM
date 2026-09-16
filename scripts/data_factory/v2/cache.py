"""DataTrove pipeline for building the shared canonical Parquet cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import ParquetWriter

from scripts.data_factory.v2.config import DataFactoryConfig, SourceSpec
from scripts.data_factory.v2.documents import (
    SourceRecordError,
    adapt_records,
    clean_and_tag,
    expand_source_paths,
    iter_raw_records,
    source_cache_id,
)
from scripts.data_factory.v2.runtime import TaskOutputGuard, build_executor


def cache_schema():
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
        ]
    )



class CanonicalSourceReader(PipelineStep):
    """Read one registered source and emit clean canonical DataTrove documents."""

    name = "Canonical source reader"
    type = "Reader"

    def __init__(self, source: SourceSpec, paths: list[Path]) -> None:
        super().__init__()
        self.source = source
        self.paths = tuple(str(path) for path in paths)

    def run(
        self,
        data: DocumentsPipeline = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> Iterator[Document]:
        del data
        paths = [Path(path) for path in self.paths[rank::world_size]]

        def record_read_error(error: dict[str, Any]) -> None:
            self.stat_update("read_errors")
            self.stat_update(f"read_error_{error['reason']}")

        for path in paths:
            self.stat_update("input_files")
            for raw in iter_raw_records(self.source, [path], on_error=record_read_error):
                self.stat_update("raw_documents")
                try:
                    adapted_records = adapt_records(
                        self.source,
                        raw,
                        lambda reason: self.stat_update(f"dropped_{reason}"),
                    )
                    for adapted in adapted_records:
                        try:
                            canonical = clean_and_tag(self.source, adapted)
                        except SourceRecordError as error:
                            self.stat_update("dropped_documents")
                            self.stat_update(f"dropped_{error.reason}")
                            continue
                        self.stat_update("forwarded_documents")
                        self.stat_update("forwarded_characters", value=len(canonical.text), unit="doc")
                        yield Document(
                            id=canonical.doc_id,
                            text=canonical.text,
                            metadata=canonical.metadata,
                        )
                except SourceRecordError as error:
                    self.stat_update("dropped_documents")
                    self.stat_update(f"dropped_{error.reason}")
                    continue


def build_cache_executor(
    config: DataFactoryConfig,
    source_name: str,
    source_manifest_sha256: str,
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
    source = config.source_registry.sources.get(source_name)
    if source is None:
        raise ValueError(f"unknown source: {source_name}")
    if config.phase not in source.phases:
        raise ValueError(f"source {source_name} does not allow {config.phase}")
    paths = expand_source_paths(source)
    task_count = min(tasks or len(paths), len(paths))
    if workers <= 0 or task_count <= 0:
        raise ValueError("workers and tasks must be positive")

    cache_id = source_cache_id(source, source_manifest_sha256)
    output_dir = config.corpus_root / "cache" / source.name / cache_id
    logging_dir = config.corpus_root / "metadata" / "cache_logs" / source.name / cache_id
    pipeline = [
        CanonicalSourceReader(source, paths),
        TaskOutputGuard(output_dir),
        ParquetWriter(
            output_folder=str(output_dir),
            output_filename="${rank}.parquet",
            compression="zstd",
            batch_size=2_000,
            expand_metadata=True,
            max_file_size=512 * 2**20,
            schema=cache_schema(),
        ),
    ]

    return build_executor(
        pipeline=pipeline,
        logging_dir=logging_dir,
        job_name=f"cc_cache_{source.name}",
        executor=executor,
        tasks=task_count,
        workers=workers,
        slurm_partition=slurm_partition,
        slurm_time=slurm_time,
        cpus_per_task=cpus_per_task,
        mem_per_cpu_gb=mem_per_cpu_gb,
        venv_path=venv_path,
    )


def run_cache(
    config: DataFactoryConfig,
    source_names: list[str],
    source_manifests: dict[str, dict[str, Any]],
    **executor_options: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for source_name in source_names:
        pipeline_executor = build_cache_executor(
            config,
            source_name,
            source_manifest_sha256=str(source_manifests[source_name]["sha256"]),
            **executor_options,
        )
        pipeline_executor.run()
        source = config.source_registry.sources[source_name]
        results.append(
            {
                "source": source_name,
                "cache_id": source_cache_id(
                    source,
                    str(source_manifests[source_name]["sha256"]),
                ),
                "executor": executor_options["executor"],
            }
        )
    return results
