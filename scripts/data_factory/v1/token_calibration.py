"""Estimate token and Phase 1 intent yields from bounded cache samples."""

from __future__ import annotations

import heapq
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from scripts.data_factory.candidate_features import (
    ALIGNMENT_POOLS,
    AlignmentFeatureMatcher,
    CandidateClassifier,
    CandidateSkip,
)
from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.fast_common import stable_key, write_fast_report


CALIBRATION_COLUMNS = [
    "doc_id",
    "text",
    "source",
    "category",
    "quota_group",
    "char_count",
    "hanzi_count",
    "latin_count",
    "digit_count",
]


def _cache_files_by_source(cache_report: dict[str, Any]) -> dict[str, list[tuple[Path, int]]]:
    files: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for task in cache_report.get("tasks", []):
        source_group = str(task["source"])
        for item in task.get("files", []):
            path = Path(item["path"])
            if not path.is_file():
                raise FileNotFoundError(f"cached Parquet file is missing: {path}")
            files[source_group].append((path, int(item.get("rows", 0))))
    return dict(files)


def _select_calibration_files(
    files: list[tuple[Path, int]],
    *,
    seed: int,
    source: str,
    limit: int,
) -> list[tuple[Path, int]]:
    return sorted(
        files,
        key=lambda item: stable_key(seed, f"calibration-file:{source}", str(item[0])),
    )[:limit]


def _sample_cache_rows(
    files: list[tuple[Path, int]],
    *,
    source: str,
    seed: int,
    target: int,
    scan_multiplier: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return a deterministic bottom-k sample from a bounded set of cache rows."""
    if not files or target <= 0:
        return [], {"scanned_documents": 0, "scanned_files": 0}

    scan_limit = target * scan_multiplier
    rows_per_file = max(1, math.ceil(scan_limit / len(files)))
    reservoir: list[tuple[int, int, dict[str, Any]]] = []
    sequence = 0
    scanned = 0
    scanned_files = 0

    for path, _rows in files:
        scanned_files += 1
        file_rows = 0
        parquet = pq.ParquetFile(path)
        stop_file = False
        for batch in parquet.iter_batches(batch_size=4096, columns=CALIBRATION_COLUMNS):
            for row in batch.to_pylist():
                key = stable_key(seed, f"calibration-row:{source}", str(row["doc_id"]))
                item = (-key, -sequence, row)
                sequence += 1
                if len(reservoir) < target:
                    heapq.heappush(reservoir, item)
                elif key < -reservoir[0][0]:
                    heapq.heapreplace(reservoir, item)
                scanned += 1
                file_rows += 1
                if scanned >= scan_limit or file_rows >= rows_per_file:
                    stop_file = True
                    break
            if stop_file:
                break
        if scanned >= scan_limit:
            break

    selected = [item[2] for item in sorted(reservoir, key=lambda item: -item[0])]
    return selected, {"scanned_documents": scanned, "scanned_files": scanned_files}


def _tokenizer(config: PipelineConfig, rayon_threads: int):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(max(1, rayon_threads))
    from tokenizers import Tokenizer

    path = config.tokenizer_path / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer.json is missing: {path}")
    return Tokenizer.from_file(str(path))


def _fit_linear(
    features: list[list[float]], targets: list[float]
) -> tuple[list[float] | None, float]:
    if len(features) < 100:
        return None, float("nan")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    predictions = x @ coeffs
    mae = float(np.mean(np.abs(predictions - y)))
    if not np.all(np.isfinite(coeffs)) or np.any(coeffs[:4] < 0):
        return None, mae
    return [float(value) for value in coeffs], mae


def _base_intent(pool: str, quota_group: str | None) -> str:
    return f"specialized:{quota_group}" if pool == "specialized" else pool


def calibrate_tokens(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    del overwrite  # The bounded report is cheap and reproducibly overwritten.
    fast = config.fast_pipeline
    cache_report_path = config.reports_dir / "phase1_fast_cache_report.json"
    if not cache_report_path.is_file():
        raise FileNotFoundError(
            f"cache report is missing: {cache_report_path}; run the cache stage first"
        )
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    files_by_group = _cache_files_by_source(cache_report)
    if not files_by_group:
        raise FileNotFoundError(
            f"no cached Parquet files in {cache_report_path}; run the cache stage first"
        )

    tokenizer = _tokenizer(config, workers or fast.tokenizer_rayon_threads)
    classifier = CandidateClassifier(config.quality)
    matcher = AlignmentFeatureMatcher(config)
    sample_target = fast.calibration_docs_per_source

    source_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "documents": 0,
            "characters": 0,
            "tokens": 0,
            "features": [],
            "targets": [],
        }
    )
    group_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "documents": 0,
            "characters": 0,
            "tokens": 0,
            "intent_documents": Counter(),
            "intent_tokens": Counter(),
            "skipped": Counter(),
        }
    )
    scan_stats: dict[str, dict[str, int]] = {}
    bridge_occurrences: Counter[int] = Counter()
    bridge_documents: Counter[int] = Counter()
    new_hanzi_documents: Counter[str] = Counter()
    global_tokens = 0
    global_chars = 0
    batch_rows: list[dict[str, Any]] = []

    def consume() -> None:
        nonlocal global_tokens, global_chars, batch_rows
        if not batch_rows:
            return
        encodings = tokenizer.encode_batch(
            [str(row["text"]) for row in batch_rows],
            add_special_tokens=False,
        )
        for row, encoding in zip(batch_rows, encodings, strict=True):
            source = str(row["source"])
            source_group = str(row.pop("_calibration_group"))
            text = str(row["text"])
            token_count = len(encoding.ids)
            char_count = int(row["char_count"])
            hanzi = int(row["hanzi_count"])
            latin = int(row["latin_count"])
            digit = int(row["digit_count"])
            other = max(0, char_count - hanzi - latin - digit)

            source_value = source_stats[source]
            source_value["documents"] += 1
            source_value["characters"] += char_count
            source_value["tokens"] += token_count
            source_value["features"].append([hanzi, latin, digit, other, 1.0])
            source_value["targets"].append(float(token_count))

            group_value = group_stats[source_group]
            group_value["documents"] += 1
            group_value["characters"] += char_count
            group_value["tokens"] += token_count
            try:
                pool, quota_group, _language = classifier.classify(row, text)
            except CandidateSkip as error:
                group_value["skipped"][error.reason] += 1
            else:
                intent = _base_intent(pool, quota_group)
                group_value["intent_documents"][intent] += 1
                group_value["intent_tokens"][intent] += token_count
                alignment_eligible = pool in ALIGNMENT_POOLS or (
                    pool == "specialized" and quota_group == "math_science"
                )
                if alignment_eligible:
                    bridge_hits = matcher.bridge_hits(text)
                    new_hits = matcher.new_hanzi_hits(text)
                    if bridge_hits:
                        group_value["intent_documents"]["multi_hanzi_bridge"] += 1
                        group_value["intent_tokens"]["multi_hanzi_bridge"] += token_count
                        bridge_occurrences.update(bridge_hits)
                        bridge_documents.update(bridge_hits.keys())
                    if new_hits:
                        group_value["intent_documents"]["new_hanzi_coverage"] += 1
                        group_value["intent_tokens"]["new_hanzi_coverage"] += token_count
                        new_hanzi_documents.update(new_hits.keys())

            global_tokens += token_count
            global_chars += char_count
        batch_rows = []

    for source_group, source_files in sorted(files_by_group.items()):
        selected_files = _select_calibration_files(
            source_files,
            seed=config.seed,
            source=source_group,
            limit=fast.calibration_files_per_source,
        )
        rows, source_scan = _sample_cache_rows(
            selected_files,
            source=source_group,
            seed=config.seed,
            target=sample_target,
            scan_multiplier=fast.calibration_scan_multiplier,
        )
        scan_stats[source_group] = {
            **source_scan,
            "available_documents": sum(count for _path, count in source_files),
            "available_files": len(source_files),
            "selected_files": len(selected_files),
        }
        for row in rows:
            row["_calibration_group"] = source_group
            batch_rows.append(row)
            if len(batch_rows) >= fast.calibration_batch_size:
                consume()
    consume()

    sources_report: dict[str, Any] = {}
    for source, value in sorted(source_stats.items()):
        documents = int(value["documents"])
        characters = int(value["characters"])
        tokens = int(value["tokens"])
        coeffs, mae = _fit_linear(value["features"], value["targets"])
        sources_report[source] = {
            "documents": documents,
            "characters": characters,
            "tokens": tokens,
            "tokens_per_char": tokens / max(1, characters),
            "linear_coefficients": coeffs,
            "fit_mae_tokens": None if not math.isfinite(mae) else mae,
        }

    groups_report: dict[str, Any] = {}
    for source_group, value in sorted(group_stats.items()):
        documents = int(value["documents"])
        tokens = int(value["tokens"])
        intents = {
            intent: {
                "documents": int(value["intent_documents"][intent]),
                "tokens": int(intent_tokens),
                "document_rate": value["intent_documents"][intent] / max(1, documents),
                "token_rate": int(intent_tokens) / max(1, tokens),
            }
            for intent, intent_tokens in sorted(value["intent_tokens"].items())
        }
        groups_report[source_group] = {
            **scan_stats.get(source_group, {}),
            "sampled_documents": documents,
            "sampled_characters": int(value["characters"]),
            "sampled_tokens": tokens,
            "tokens_per_document": tokens / max(1, documents),
            "intents": intents,
            "skipped_by_reason": dict(sorted(value["skipped"].items())),
        }

    top_bridge = [
        {
            "old_token_id": token_id,
            "occurrences": bridge_occurrences[token_id],
            "documents": bridge_documents[token_id],
        }
        for token_id, _count in bridge_occurrences.most_common(
            config.vocab_alignment.bridge_top_token_count
        )
    ]
    return write_fast_report(
        config.fast_calibration_path,
        {
            "stage": "token_calibration",
            "tokenizer_path": str(config.tokenizer_path),
            "sample_target_per_source": sample_target,
            "max_files_per_source": fast.calibration_files_per_source,
            "scan_multiplier": fast.calibration_scan_multiplier,
            "sources": sources_report,
            "source_groups": groups_report,
            "top_bridge_tokens": top_bridge,
            "new_hanzi_document_frequency": dict(new_hanzi_documents.most_common()),
            "global": {
                "characters": global_chars,
                "tokens": global_tokens,
                "tokens_per_char": global_tokens / max(1, global_chars),
            },
        },
    )
