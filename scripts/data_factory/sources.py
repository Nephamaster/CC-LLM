"""Streaming adapters for Phase 1 source datasets."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from scripts.data_factory.config import (
    PipelineConfig,
    VALID_CATEGORIES,
    VALID_MIXED_GROUPS,
    VALID_SUPPLEMENTAL_GROUPS,
)
from scripts.data_factory.io_utils import expand_paths, iter_jsonl


CHID_PLACEHOLDER_RE = re.compile(r"#idiom\d+#")

EXCLUDED_CLUE_FIELDS = frozenset(
    {"answer", "answers", "id", "idx", "index", "label", "label_desc", "label_des", "metadata"}
)


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]+", "_", value.replace("-", "_")).strip("_")


def _text_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() not in EXCLUDED_CLUE_FIELDS:
                yield from _text_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _text_values(nested)


def _join_fields(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for field in fields:
        for value in _text_values(row.get(field)):
            if value not in seen:
                values.append(value)
                seen.add(value)
    return "\n".join(values)


def _chid_texts(row: dict[str, Any]) -> Iterator[tuple[str, str]]:
    contents = list(_text_values(row.get("content")))
    answer_data = row.get("answers")
    answers = list(_text_values(answer_data.get("text"))) if isinstance(answer_data, dict) else []
    placeholder_count = sum(len(CHID_PLACEHOLDER_RE.findall(content)) for content in contents)
    if placeholder_count != len(answers):
        raise ValueError(
            f"CHID row has {placeholder_count} placeholders but {len(answers)} answers"
        )

    answer_index = 0
    for content_index, content in enumerate(contents):
        def replace(_: re.Match[str]) -> str:
            nonlocal answer_index
            answer = answers[answer_index]
            answer_index += 1
            return answer

        yield CHID_PLACEHOLDER_RE.sub(replace, content), f"content-{content_index}"


def _clue_texts(subset: str, row: dict[str, Any]) -> Iterator[tuple[str, str | None]]:
    if subset == "afqmc":
        yield _join_fields(row, ("sentence1", "sentence2")), None
    elif subset == "c3":
        yield _join_fields(row, ("context", "question", "answer")), None
    elif subset == "chid":
        yield from _chid_texts(row)
    elif subset == "cluewsc2020":
        yield _join_fields(row, ("text",)), None
    elif subset == "cmnli":
        for field in ("sentence1", "sentence2"):
            yield _join_fields(row, (field,)), field
    elif subset in {"cmrc2018", "drcd"}:
        yield _join_fields(row, ("context",)), None
    elif subset == "csl":
        yield _join_fields(row, ("abst",)), None
    elif subset in {"iflytek", "tnews"}:
        yield _join_fields(row, ("sentence",)), None


def _clue_context_key(subset: str, row: dict[str, Any]) -> str | None:
    value = str(row.get("id", ""))
    if subset == "cmrc2018":
        match = re.match(r"^TRAIN_([^_]+)", value)
        return f"TRAIN_{match.group(1)}" if match else value or None
    if subset == "drcd":
        return value.rsplit("-", 1)[0] if "-" in value else value or None
    return None


def _iter_parquet_rows(paths: list[Path], columns: list[str] | None = None) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required to read source parquet files") from error

    for path in paths:
        parquet_file = parquet.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        selected = None if columns is None else [column for column in columns if column in available]
        row_index = 0
        for batch in parquet_file.iter_batches(batch_size=4096, columns=selected):
            for row in batch.to_pylist():
                yield path, row_index, row
                row_index += 1


def iter_clue(config: PipelineConfig, source_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    root = Path(source_config["path"])
    if not root.is_absolute():
        root = config.repo_root / root
    paths = sorted(root.glob("*/train-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no CLUE train parquet files found under {root}")

    counters: dict[str, int] = {}
    seen_contexts: dict[str, set[str]] = {"cmrc2018": set(), "drcd": set()}
    for path, _, row in _iter_parquet_rows(paths):
        subset = path.parent.name
        if subset == "ocnli":
            continue
        context_key = _clue_context_key(subset, row)
        if context_key is not None:
            if context_key in seen_contexts[subset]:
                continue
            seen_contexts[subset].add(context_key)
        index = counters.get(subset, 0)
        counters[subset] = index + 1
        for text, part in _clue_texts(subset, row):
            doc_id = f"clue-{_safe_component(subset)}-train-{index}"
            if part is not None:
                doc_id = f"{doc_id}-{_safe_component(part)}"
            yield {
                "text": text,
                "source": "clue_benchmark",
                "doc_id": doc_id,
                "license": source_config.get("license", "unknown"),
                "license_note": source_config.get("license_note"),
                "category": "chinese_general",
                "path": str(path),
                "split": "train",
                "subset": subset,
                "_allow_unverified_license": bool(source_config.get("allow_unverified_license", False)),
            }


def iter_fineweb_chinese(config: PipelineConfig, source_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    paths = expand_paths(source_config["paths"], config.repo_root)
    if not paths:
        raise FileNotFoundError(f"no Chinese FineWeb files match {source_config['paths']}")
    for path, row_index, row in _iter_parquet_rows(paths, ["text", "score", "source"]):
        subset = path.parent.name
        stem = path.stem
        yield {
            "text": row.get("text"),
            "source": "fineweb_edu_chinese_v2.2",
            "doc_id": f"fineweb_edu_chinese_v2.2-{_safe_component(subset)}-{_safe_component(stem)}-{row_index}",
            "license": source_config.get("license", "Apache-2.0"),
            "license_note": source_config.get("license_note"),
            "category": "chinese_high_quality",
            "path": str(path),
            "quality_score": row.get("score"),
            "upstream_source": row.get("source"),
        }


def iter_fineweb_english(config: PipelineConfig, source_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    paths = expand_paths(source_config["paths"], config.repo_root)
    if not paths:
        raise FileNotFoundError(f"no FineWeb-Edu files match {source_config['paths']}")
    columns = ["text", "id", "dump", "url", "file_path", "language", "language_score", "score", "int_score"]
    minimum_score = float(source_config.get("language_score_min", 0.90))
    for path, row_index, row in _iter_parquet_rows(paths, columns):
        if str(row.get("language", "")).lower() != "en":
            continue
        language_score = row.get("language_score")
        if language_score is None or float(language_score) < minimum_score:
            continue
        yield {
            "text": row.get("text"),
            "source": "fineweb_edu",
            "doc_id": f"fineweb_edu-sample_10bt-{_safe_component(path.stem)}-{row_index}",
            "license": source_config.get("license", "ODC-By-1.0"),
            "category": "non_chinese",
            "path": str(path),
            "url": row.get("url"),
            "revision": row.get("dump"),
            "upstream_id": row.get("id"),
            "upstream_path": row.get("file_path"),
            "language": "en",
            "language_score": language_score,
            "quality_score": row.get("score"),
            "quality_bucket": row.get("int_score"),
        }


def iter_external(config: PipelineConfig, source_config: dict[str, Any]) -> Iterator[dict[str, Any]]:
    paths = expand_paths(source_config.get("paths", []), config.repo_root)
    if not paths:
        if source_config.get("optional", True):
            return
        raise FileNotFoundError(f"no external JSONL files match {source_config.get('paths')}")

    default_category = source_config.get("category")
    default_group = source_config.get("quota_group")
    default_license = source_config.get("license")
    source_name = str(source_config["name"])
    for index, row in enumerate(iter_jsonl(paths)):
        category = row.get("category", default_category)
        if category not in VALID_CATEGORIES:
            raise ValueError(f"external source {source_name} has invalid category {category!r}")
        quota_group = row.get("quota_group", default_group)
        if category == "mixed_zh_en" and quota_group not in VALID_MIXED_GROUPS:
            raise ValueError(f"external source {source_name} has invalid mixed quota_group {quota_group!r}")
        if category == "supplemental" and quota_group not in VALID_SUPPLEMENTAL_GROUPS:
            raise ValueError(f"external source {source_name} has invalid supplemental quota_group {quota_group!r}")
        value = dict(row)
        value.setdefault("source", source_name)
        value.setdefault("doc_id", f"{_safe_component(source_name)}-{index}")
        value.setdefault("license", default_license)
        value.setdefault("category", category)
        value.setdefault("quota_group", quota_group)
        value["_allow_unverified_license"] = bool(source_config.get("allow_unverified_license", False))
        yield value


def source_iterators(config: PipelineConfig) -> list[tuple[str, Iterator[dict[str, Any]]]]:
    sources = config.sources
    values: list[tuple[str, Iterator[dict[str, Any]]]] = []
    if sources.get("clue", {}).get("enabled", True):
        values.append(("clue", iter_clue(config, sources["clue"])))
    if sources.get("fineweb_chinese", {}).get("enabled", True):
        values.append(("fineweb_chinese", iter_fineweb_chinese(config, sources["fineweb_chinese"])))
    if sources.get("fineweb_english", {}).get("enabled", True):
        values.append(("fineweb_english", iter_fineweb_english(config, sources["fineweb_english"])))
    for external in sources.get("external", []):
        if external.get("enabled", True):
            values.append((str(external["name"]), iter_external(config, external)))
    return values

