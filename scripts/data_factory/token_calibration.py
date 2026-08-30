"""Estimate Phase 1 token counts from a small deterministic source sample."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.fast_common import parquet_files, stable_fraction, write_fast_report


CALIBRATION_COLUMNS = [
    "doc_id",
    "text",
    "source",
    "char_count",
    "hanzi_count",
    "latin_count",
    "digit_count",
]


def _tokenizer(config: PipelineConfig, rayon_threads: int):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(max(1, rayon_threads))
    from tokenizers import Tokenizer

    path = config.tokenizer_path / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer.json is missing: {path}")
    return Tokenizer.from_file(str(path))


def _fit_linear(features: list[list[float]], targets: list[float]) -> tuple[list[float] | None, float]:
    if len(features) < 100:
        return None, float("nan")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    predictions = x @ coeffs
    mae = float(np.mean(np.abs(predictions - y)))
    # Negative character coefficients are a sign that the sample is too small or ill-conditioned.
    if not np.all(np.isfinite(coeffs)) or np.any(coeffs[:4] < 0):
        return None, mae
    return [float(value) for value in coeffs], mae


def calibrate_tokens(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    del overwrite  # report is intentionally cheap and always reproducibly overwritten
    fast = config.fast_pipeline
    files = parquet_files(config.fast_cache_dir)
    if not files:
        raise FileNotFoundError(
            f"no cached Parquet files under {config.fast_cache_dir}; run the cache stage first"
        )
    tokenizer = _tokenizer(config, workers or fast.tokenizer_rayon_threads)

    # Cache report gives exact per-source document counts, allowing hash-probability
    # sampling without storing a reservoir of raw text in memory.
    cache_report_path = config.reports_dir / "phase1_fast_cache_report.json"
    import json

    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    source_docs: dict[str, int] = defaultdict(int)
    for task in cache_report.get("tasks", []):
        source_docs[str(task["source"])] += int(task.get("kept_records", 0))

    sample_target = fast.calibration_docs_per_source
    probabilities = {
        source: min(1.0, sample_target / max(1, count) * 1.10)
        for source, count in source_docs.items()
    }

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "documents": 0,
            "characters": 0,
            "tokens": 0,
            "features": [],
            "targets": [],
        }
    )
    global_tokens = 0
    global_chars = 0
    batch_rows: list[dict[str, Any]] = []

    def consume() -> None:
        nonlocal global_tokens, global_chars, batch_rows
        if not batch_rows:
            return
        encodings = tokenizer.encode_batch([str(row["text"]) for row in batch_rows], add_special_tokens=False)
        for row, encoding in zip(batch_rows, encodings, strict=True):
            source = str(row["source"])
            token_count = len(encoding.ids)
            char_count = int(row["char_count"])
            hanzi = int(row["hanzi_count"])
            latin = int(row["latin_count"])
            digit = int(row["digit_count"])
            other = max(0, char_count - hanzi - latin - digit)
            value = stats[source]
            value["documents"] += 1
            value["characters"] += char_count
            value["tokens"] += token_count
            value["features"].append([hanzi, latin, digit, other, 1.0])
            value["targets"].append(float(token_count))
            global_tokens += token_count
            global_chars += char_count
        batch_rows = []

    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096, columns=CALIBRATION_COLUMNS):
            for row in batch.to_pylist():
                source = str(row["source"])
                # Once a source has enough samples, continue scanning but avoid retaining more text.
                if stats[source]["documents"] >= sample_target:
                    continue
                if stable_fraction(config.seed, "calibration", str(row["doc_id"])) >= probabilities.get(source, 1.0):
                    continue
                batch_rows.append(row)
                if len(batch_rows) >= fast.calibration_batch_size:
                    consume()
    consume()

    source_report: dict[str, Any] = {}
    for source, value in sorted(stats.items()):
        documents = int(value["documents"])
        characters = int(value["characters"])
        tokens = int(value["tokens"])
        coeffs, mae = _fit_linear(value["features"], value["targets"])
        source_report[source] = {
            "documents": documents,
            "characters": characters,
            "tokens": tokens,
            "tokens_per_char": tokens / max(1, characters),
            "linear_coefficients": coeffs,
            "fit_mae_tokens": None if not math.isfinite(mae) else mae,
        }

    return write_fast_report(
        config.fast_calibration_path,
        {
            "stage": "token_calibration",
            "tokenizer_path": str(config.tokenizer_path),
            "sample_target_per_source": sample_target,
            "sources": source_report,
            "global": {
                "characters": global_chars,
                "tokens": global_tokens,
                "tokens_per_char": global_tokens / max(1, global_chars),
            },
        },
    )
