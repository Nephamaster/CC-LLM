"""Bounded token calibration and deterministic file-level run planning."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.data_factory.v2.config import BucketSpec, DataFactoryConfig
from scripts.data_factory.v2.documents import source_cache_id
from scripts.data_factory.v2.quality import sub_bucket


CACHE_COLUMNS = [
    "id",
    "text",
    "parent_doc_id",
    "source",
    "subset",
    "source_path",
    "revision",
    "license",
    "url",
    "language",
    "domain",
    "char_count",
    "hanzi_count",
    "latin_count",
    "digit_count",
    "quality_prior",
    "tags",
    "metadata_json",
]


def stable_key(seed: int, namespace: str, value: str) -> int:
    payload = f"{seed}:{namespace}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def stable_fraction(seed: int, namespace: str, value: str) -> float:
    return stable_key(seed, namespace, value) / float(1 << 64)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_new_characters(path: Path) -> frozenset[str]:
    if not path.is_file():
        raise FileNotFoundError(f"new Hanzi token map is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    characters: set[str] = set()
    values: list[Any]
    if isinstance(raw, dict):
        values = list(raw.keys()) + list(raw.values())
    elif isinstance(raw, list):
        values = raw
    else:
        raise ValueError("new Hanzi token map must be an object or list")
    for value in values:
        if isinstance(value, str) and len(value) == 1:
            characters.add(value)
        elif isinstance(value, dict):
            char = value.get("char") or value.get("token")
            if isinstance(char, str) and len(char) == 1:
                characters.add(char)
    if not characters:
        raise ValueError("new Hanzi token map contains no single characters")
    return frozenset(characters)


def _technical_mixed(text: str, metadata: dict[str, Any]) -> bool:
    if metadata.get("language") == "zh_en_mixed" or "mixed" in set(metadata.get("tags") or []):
        return True
    hanzi = int(metadata.get("hanzi_count") or 0)
    latin = int(metadata.get("latin_count") or 0)
    if hanzi < 10 or latin < 10:
        return False
    signals = ("```", "http://", "https://", " API ", " CLI ", "::", "_", "/")
    return sum(signal in text for signal in signals) >= 1


def selector_matches(
    selector: str,
    text: str,
    metadata: dict[str, Any],
    new_characters: frozenset[str],
) -> bool:
    language = str(metadata.get("language") or "")
    domain = str(metadata.get("domain") or "general")
    tags = set(metadata.get("tags") or [])
    if selector == "new_char_coverage":
        return language in {"zh", "zh_en_mixed"} and any(char in new_characters for char in text)
    if selector == "zh_general":
        return language == "zh" and domain == "general" and "classical" not in tags
    if selector == "zh_knowledge":
        return language == "zh" and domain == "knowledge"
    if selector == "english":
        return language == "en" and domain == "general"
    if selector == "mixed_technical":
        return _technical_mixed(text, metadata)
    if selector == "specialized":
        return domain in {"code", "math", "scientific", "structured"}
    if selector == "english_multilingual_mixed":
        return language not in {"", "unknown", "zh"} or _technical_mixed(text, metadata)
    if selector == "math_code_science":
        return domain in {"code", "math", "scientific", "structured"}
    raise ValueError(f"unsupported bucket selector: {selector}")


def assign_bucket(
    config: DataFactoryConfig,
    text: str,
    metadata: dict[str, Any],
    new_characters: frozenset[str],
) -> str | None:
    buckets = {bucket.name: bucket for bucket in config.buckets}
    for name in config.candidate_priority:
        if name == config.enhancement.bucket:
            continue
        bucket = buckets[name]
        if metadata.get("source") not in bucket.source_weights:
            continue
        if selector_matches(bucket.selector, text, metadata, new_characters) or (
            config.phase == "phase2" and bucket.selector == "zh_general"
            and metadata.get("source") == "fineweb_edu_chinese"
            and metadata.get("language") == "zh" and metadata.get("domain") == "knowledge"
        ):
            return name
    return None


def eligible_buckets(config, text, metadata, new_characters) -> list[str]:
    ordinary = assign_bucket(config, text, metadata, new_characters)
    result = [ordinary] if ordinary else []
    enhancement = next(b for b in config.buckets if b.name == config.enhancement.bucket)
    if metadata.get("source") in enhancement.source_weights and selector_matches(
        "new_char_coverage", text, metadata, new_characters
    ):
        result.append(enhancement.name)
    return result


def calibration_path(config: DataFactoryConfig) -> Path:
    tokenizer_hash = config.hashes.tokenizer or "missing-tokenizer"
    manifest_hash = config.hashes.source_manifest or "missing-manifest"
    new_chars_hash = config.hashes.new_char_tokens or "missing-new-chars"
    return (
        config.corpus_root
        / "metadata"
        / "calibration"
        / tokenizer_hash[:16]
        / (
            f"{config.phase}-{config.hashes.phase_config[:12]}-"
            f"{manifest_hash[:12]}-{new_chars_hash[:12]}.json"
        )
    )


def plan_path(config: DataFactoryConfig, round_index: int = 0) -> Path:
    return config.run_root / "plans" / f"plan-round-{round_index:03d}.json"


def cache_files_for_source(
    config: DataFactoryConfig,
    source_name: str,
    source_manifest: dict[str, Any],
) -> list[Path]:
    source = config.source_registry.sources[source_name]
    cache_id = source_cache_id(source, str(source_manifest["sha256"]))
    root = config.corpus_root / "cache" / source_name / cache_id
    files = sorted(path for path in root.rglob("*.parquet") if path.is_file()) if root.is_dir() else []
    if not files:
        raise FileNotFoundError(f"canonical cache is missing for {source_name}: {root}")
    return files


def _cache_file_rows(files: list[Path]) -> list[tuple[Path, int]]:
    import pyarrow.parquet as pq

    return [(path, pq.ParquetFile(path).metadata.num_rows) for path in files]


def _bounded_sample(
    files: list[tuple[Path, int]],
    *,
    source_name: str,
    seed: int,
    target: int,
    max_files: int,
    scan_multiplier: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    import pyarrow.parquet as pq

    selected_files = sorted(
        files,
        key=lambda item: stable_key(seed, f"calibration-file:{source_name}", str(item[0])),
    )[:max_files]
    scan_limit = target * scan_multiplier
    rows_per_file = max(1, math.ceil(scan_limit / len(selected_files)))
    reservoir: list[tuple[int, int, dict[str, Any]]] = []
    scanned = 0
    sequence = 0
    for path, _rows in selected_files:
        current_file_rows = 0
        stop_file = False
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=4096, columns=CACHE_COLUMNS):
            for row in batch.to_pylist():
                key = stable_key(seed, f"calibration-row:{source_name}", str(row["id"]))
                item = (-key, -sequence, row)
                sequence += 1
                if len(reservoir) < target:
                    heapq.heappush(reservoir, item)
                elif key < -reservoir[0][0]:
                    heapq.heapreplace(reservoir, item)
                scanned += 1
                current_file_rows += 1
                if scanned >= scan_limit or current_file_rows >= rows_per_file:
                    stop_file = True
                    break
            if stop_file:
                break
        if scanned >= scan_limit:
            break
    sample = [item[2] for item in sorted(reservoir, key=lambda item: -item[0])]
    return sample, {
        "selected_files": len(selected_files),
        "scanned_documents": scanned,
        "sampled_documents": len(sample),
    }


def calibrate(
    config: DataFactoryConfig,
    source_manifests: dict[str, dict[str, Any]],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_path = calibration_path(config)
    if output_path.is_file() and not overwrite:
        return json.loads(output_path.read_text(encoding="utf-8"))
    if config.hashes.tokenizer is None:
        raise FileNotFoundError(f"tokenizer.json is missing under {config.tokenizer_path}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(config.tokenizer_path / "tokenizer.json"))
    new_characters = load_new_characters(config.enhancement.token_ids_path)
    sources_report: dict[str, Any] = {}
    global_documents = 0
    global_tokens = 0

    for source_name in sorted(source_manifests):
        files = cache_files_for_source(config, source_name, source_manifests[source_name])
        file_rows = _cache_file_rows(files)
        sample, scan = _bounded_sample(
            file_rows,
            source_name=source_name,
            seed=config.seed,
            target=config.calibration.documents_per_source,
            max_files=config.calibration.max_files_per_source,
            scan_multiplier=config.calibration.scan_multiplier,
        )
        bucket_documents: Counter[str] = Counter()
        bucket_tokens: Counter[str] = Counter()
        sub_tokens: dict[str, Counter[str]] = defaultdict(Counter)
        bucket_characters: Counter[str] = Counter()
        domain_documents: Counter[str] = Counter()
        source_tokens = 0
        source_characters = 0
        for offset in range(0, len(sample), config.calibration.batch_size):
            batch = sample[offset : offset + config.calibration.batch_size]
            encodings = tokenizer.encode_batch(
                [str(row["text"]) for row in batch],
                add_special_tokens=False,
            )
            for row, encoding in zip(batch, encodings, strict=True):
                token_count = len(encoding.ids)
                source_tokens += token_count
                source_characters += int(row["char_count"])
                domain_documents[str(row.get("domain") or "unknown")] += 1
                for bucket in eligible_buckets(config, str(row["text"]), row, new_characters):
                    bucket_documents[bucket] += 1
                    bucket_tokens[bucket] += token_count
                    bucket_characters[bucket] += int(row["char_count"])
                    sub_tokens[bucket][str(sub_bucket(row))] += token_count

        sampled_documents = len(sample)
        available_documents = sum(rows for _path, rows in file_rows)
        tokens_per_document = source_tokens / max(1, sampled_documents)
        sources_report[source_name] = {
            **scan,
            "available_files": len(files),
            "available_documents": available_documents,
            "sampled_characters": source_characters,
            "sampled_tokens": source_tokens,
            "tokens_per_document": tokens_per_document,
            "tokens_per_character": source_tokens / max(1, source_characters),
            "estimated_available_tokens": int(round(available_documents * tokens_per_document)),
            "domains": dict(sorted(domain_documents.items())),
            "buckets": {
                bucket.name: {
                    "documents": bucket_documents[bucket.name],
                    "tokens": bucket_tokens[bucket.name],
                    "document_rate": bucket_documents[bucket.name] / max(1, sampled_documents),
                    "token_rate": bucket_tokens[bucket.name] / max(1, source_tokens),
                    "tokens_per_character": bucket_tokens[bucket.name] / max(1, bucket_characters[bucket.name]),
                    "sub_token_rates": {name: value / max(1, source_tokens) for name, value in sub_tokens[bucket.name].items()},
                }
                for bucket in config.buckets
            },
            "cache_files": [
                {"path": str(path), "rows": rows}
                for path, rows in file_rows
            ],
        }
        global_documents += sampled_documents
        global_tokens += source_tokens

    report: dict[str, Any] = {
        "stage": "calibration",
        "phase": config.phase,
        "run_id": config.run_id,
        "phase_config_sha256": config.hashes.phase_config,
        "source_manifest_sha256": config.hashes.source_manifest,
        "tokenizer_sha256": config.hashes.tokenizer,
        "new_character_count": len(new_characters),
        "sources": sources_report,
        "global": {"sampled_documents": global_documents, "sampled_tokens": global_tokens},
    }
    report["calibration_sha256"] = _hash_json(report)
    _write_json(output_path, report)
    return report


def _select_source_files(
    files: list[dict[str, Any]],
    *,
    source_name: str,
    seed: int,
    tokens_per_document: float,
    required_tokens: float,
) -> tuple[list[dict[str, Any]], int]:
    ordered = sorted(
        files,
        key=lambda item: stable_key(seed, f"plan-file:{source_name}", str(item["path"])),
    )
    selected: list[dict[str, Any]] = []
    accumulated = 0
    for item in ordered:
        estimated = int(round(int(item["rows"]) * tokens_per_document))
        selected.append({**item, "estimated_tokens": estimated})
        accumulated += estimated
        if accumulated >= required_tokens:
            break
    return selected, accumulated


def build_plan(
    config: DataFactoryConfig,
    calibration: dict[str, Any],
    *,
    round_index: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    previous_plans = [json.loads(plan_path(config, i).read_text()) for i in range(round_index)]
    used_files = {item["path"] for p in previous_plans for item in p["selected_files"]}
    deficits = None
    if previous_plans:
        report_path = config.run_root / "reports" / "mixture_report.json"
        report = json.loads(report_path.read_text())
        if report.get("plan_sha256") != previous_plans[-1]["plan_sha256"]:
            raise ValueError("incremental plan requires mixture report for the previous round")
        deficits = report.get("source_shortfalls", {})
        final_path = config.run_root / "reports" / "finalization_report.json"
        if final_path.is_file():
            final = json.loads(final_path.read_text())
            if final.get("plan_sha256") == previous_plans[-1]["plan_sha256"]:
                deficits = final.get("source_shortfalls", deficits)
        if not any(v > 0 for values in deficits.values() for v in values.values()):
            raise ValueError("no token shortfalls to replenish; coverage diagnostics do not trigger resampling")
    output_path = plan_path(config, round_index)
    bucket_targets = {
        b.name: config.bucket_tokens[b.name] + int(round(config.validation_tokens * b.fraction))
        for b in config.buckets
    }
    requirements: dict[str, float] = defaultdict(float)
    requested: dict[str, dict[str, int]] = defaultdict(dict)
    shortfalls: list[dict[str, Any]] = []

    for bucket in config.buckets:
        oversample = (
            config.enhancement_oversample_ratio
            if bucket.name == config.enhancement.bucket
            else config.candidate_oversample_ratio
        )
        for source_name, weight in bucket.source_weights.items():
            target = int(round(bucket_targets[bucket.name] * weight))
            if deficits is not None:
                target = int(deficits.get(bucket.name, {}).get(source_name, 0))
            requested[bucket.name][source_name] = target
            if target <= 0:
                continue
            source_stats = calibration["sources"].get(source_name)
            rate = 0.0 if source_stats is None else float(source_stats["buckets"][bucket.name]["token_rate"])
            if rate <= 0:
                shortfalls.append(
                    {"bucket": bucket.name, "source": source_name, "reason": "zero_calibrated_yield", "target_tokens": target}
                )
                continue
            # Ordinary and enhancement eligibility overlap. Reserve enough source
            # capacity for both uses; only the final mixture assigns ownership.
            requirements[source_name] += target * oversample / rate

    selected_files: list[dict[str, Any]] = []
    selected_capacity: dict[str, int] = {}
    seed = config.seed
    for source_name, required in sorted(requirements.items()):
        source_stats = calibration["sources"][source_name]
        selected, capacity = _select_source_files(
            [item for item in source_stats["cache_files"] if item["path"] not in used_files],
            source_name=source_name,
            seed=seed,
            tokens_per_document=float(source_stats["tokens_per_document"]),
            required_tokens=required,
        )
        selected_capacity[source_name] = capacity
        for item in selected:
            selected_files.append({"source": source_name, **item})
        if capacity < required:
            shortfalls.append(
                {
                    "source": source_name,
                    "reason": "insufficient_cache_capacity",
                    "required_tokens": int(math.ceil(required)),
                    "available_tokens": capacity,
                }
            )

    sampling_rates: dict[str, dict[str, float]] = defaultdict(dict)
    for bucket in config.buckets:
        oversample = (
            config.enhancement_oversample_ratio
            if bucket.name == config.enhancement.bucket
            else config.candidate_oversample_ratio
        )
        for source_name, target in requested[bucket.name].items():
            source_stats = calibration["sources"].get(source_name)
            if source_stats is None or selected_capacity.get(source_name, 0) <= 0:
                continue
            rate = float(source_stats["buckets"][bucket.name]["token_rate"])
            eligible = selected_capacity[source_name] * rate
            reserve = 0
            if bucket.name != config.enhancement.bucket and target > 0:
                reserve = requested.get(config.enhancement.bucket, {}).get(source_name, 0)
            sample_rate = (target + reserve) * oversample / max(1.0, eligible)
            if sample_rate > 1.0 + 1e-9:
                shortfalls.append(
                    {
                        "bucket": bucket.name,
                        "source": source_name,
                        "reason": "sampling_rate_exceeds_one",
                        "required_rate": sample_rate,
                    }
                )
            sampling_rates[bucket.name][source_name] = min(1.0, sample_rate)

    if round_index == 0:
        for bucket in config.buckets:
            for domain, fraction in bucket.sub_buckets.items():
                capacity = 0.0
                for source in bucket.source_weights:
                    stats = calibration["sources"].get(source, {})
                    rates = stats.get("buckets", {}).get(bucket.name, {}).get("sub_token_rates")
                    if rates is None:
                        break
                    capacity += selected_capacity.get(source, 0) * rates.get(domain, 0)
                else:
                    required = bucket_targets[bucket.name] * fraction * config.candidate_oversample_ratio
                    if capacity < required:
                        shortfalls.append({"bucket": bucket.name, "domain": domain,
                                           "reason": "insufficient_sub_bucket_capacity",
                                           "required_tokens": int(required), "available_tokens": int(capacity)})

    fingerprint_payload = {
        "run_id": config.run_id,
        "round_index": round_index,
        "calibration_sha256": calibration["calibration_sha256"],
        "bucket_targets": bucket_targets,
        "selected_files": selected_files,
        "sampling_rates": sampling_rates,
        "previous_plan_hashes": [p["plan_sha256"] for p in previous_plans],
    }
    report: dict[str, Any] = {
        "stage": "plan",
        "phase": config.phase,
        "run_id": config.run_id,
        "round_index": round_index,
        "passed": not shortfalls,
        "calibration_sha256": calibration["calibration_sha256"],
        "plan_sha256": _hash_json(fingerprint_payload),
        "bucket_targets": bucket_targets,
        "source_targets": {name: dict(values) for name, values in requested.items()},
        "previous_plan_hashes": [p["plan_sha256"] for p in previous_plans],
        "sampling_rates": {name: dict(values) for name, values in sampling_rates.items()},
        "selected_files": selected_files,
        "selected_file_count": len(selected_files),
        "selected_estimated_tokens": sum(int(item["estimated_tokens"]) for item in selected_files),
        "shortfalls": shortfalls,
    }
    if output_path.is_file() and not overwrite:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("plan_sha256") != report["plan_sha256"]:
            raise RuntimeError(f"immutable plan already exists with different content: {output_path}")
        return existing
    _write_json(output_path, report)
    return report
