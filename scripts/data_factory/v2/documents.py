"""Source inspection, adaptation, and cheap canonical document tagging."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import tarfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from scripts.data_factory.v2.config import SourceSpec


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
LATIN_RE = re.compile(r"[A-Za-z]")
MATH_RE = re.compile(r"\\(?:frac|sum|prod|int|sqrt|begin)\b|\$\$|[∑∏√∞∫∂∇≈≠≤≥]")
CLASSICAL_MARKERS = frozenset("兮矣焉哉曰於者也其之乎乃遂")
CODE_EXTENSIONS = frozenset(
    {".py", ".js", ".ts", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".go", ".rs", ".sh", ".sql"}
)
STRUCTURED_EXTENSIONS = frozenset({".json", ".yaml", ".yml", ".toml", ".xml", ".md", ".rst"})
TEXT_KEYS = ("text", "content", "body", "document")
ID_KEYS = ("id", "doc_id", "corpus_id", "hexsha", "swhid")
CACHE_SCHEMA_VERSION = 1

PROVENANCE_KEYS = (
    "id",
    "doc_id",
    "corpus_id",
    "hexsha",
    "swhid",
    "repo_name",
    "repository_name",
    "path",
    "file_path",
    "lang",
    "language",
    "language_score",
    "dump",
    "date",
    "title",
    "score",
    "int_score",
    "quality_score",
)


class SourceRecordError(ValueError):
    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class RawRecord:
    row: dict[str, Any]
    path: Path
    row_index: int
    member: str | None = None


@dataclass(frozen=True)
class AdaptedRecord:
    doc_id: str
    text: str
    source: str
    subset: str | None
    source_path: str
    revision: str | None
    license: str
    url: str | None
    language: str | None
    domain: str
    quality_prior: float | None
    metadata_json: str


@dataclass(frozen=True)
class CanonicalRecord:
    doc_id: str
    text: str
    metadata: dict[str, Any]


def expand_source_paths(source: SourceSpec) -> list[Path]:
    paths: list[Path] = []
    for raw_value in source.paths:
        value = os.path.expandvars(raw_value)
        if "$" in value:
            raise ValueError(
                f"source {source.name} has an unresolved environment path: {value}"
            )
        matches = [Path(path) for path in glob.glob(value, recursive=True)]
        if not matches:
            candidate = Path(value)
            if candidate.is_file():
                matches = [candidate]
            elif candidate.is_dir():
                suffixes = {
                    "jsonl": ("*.jsonl", "*.jsonl.gz"),
                    "parquet": ("*.parquet",),
                    "tar_jsonl": ("*.tar", "*.tar.gz", "*.tgz"),
                }[source.reader]
                matches = [path for pattern in suffixes for path in candidate.rglob(pattern)]
        paths.extend(path.resolve() for path in matches if path.is_file())
    result = sorted(set(paths))
    if not result:
        raise FileNotFoundError(f"source {source.name} has no readable input files")
    return result


def source_contract_hash(source: SourceSpec) -> str:
    payload = {
        "name": source.name,
        "reader": source.reader,
        "adapter": source.adapter,
        "paths": source.paths,
        "license": source.license,
        "license_mode": source.license_mode,
        "quality_profile": source.quality_profile,
        "default_domain": source.default_domain,
        "metadata": source.metadata,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_cache_id(source: SourceSpec, manifest_sha256: str) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_contract_sha256": source_contract_hash(source),
        "manifest_sha256": manifest_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _emit_error(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    reason: str,
    path: Path,
    row_index: int | None = None,
    member: str | None = None,
    error: Exception | None = None,
) -> None:
    if callback is not None:
        callback(
            {
                "reason": reason,
                "path": str(path),
                "row_index": row_index,
                "member": member,
                "error": None if error is None else str(error),
            }
        )


def _iter_jsonl(
    path: Path,
    on_error: Callable[[dict[str, Any]], None] | None,
) -> Iterator[RawRecord]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for row_index, raw_line in enumerate(stream):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line.decode("utf-8"))
                if not isinstance(row, dict):
                    raise TypeError("JSONL row is not an object")
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
                _emit_error(
                    on_error,
                    reason="invalid_jsonl_row",
                    path=path,
                    row_index=row_index,
                    error=error,
                )
                continue
            yield RawRecord(row=row, path=path, row_index=row_index)


def _iter_parquet(
    path: Path,
    on_error: Callable[[dict[str, Any]], None] | None,
) -> Iterator[RawRecord]:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                if isinstance(row, dict):
                    yield RawRecord(row=row, path=path, row_index=row_index)
                else:
                    _emit_error(
                        on_error,
                        reason="invalid_parquet_row",
                        path=path,
                        row_index=row_index,
                    )
                row_index += 1
    except Exception as error:
        _emit_error(on_error, reason="invalid_parquet_file", path=path, error=error)
        raise


def _iter_tar_jsonl(
    path: Path,
    on_error: Callable[[dict[str, Any]], None] | None,
) -> Iterator[RawRecord]:
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith((".jsonl", ".json")):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            for row_index, raw_line in enumerate(stream):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                    if not isinstance(row, dict):
                        raise TypeError("archive JSONL row is not an object")
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
                    _emit_error(
                        on_error,
                        reason="invalid_archive_jsonl_row",
                        path=path,
                        row_index=row_index,
                        member=member.name,
                        error=error,
                    )
                    continue
                yield RawRecord(
                    row=row,
                    path=path,
                    row_index=row_index,
                    member=member.name,
                )


def iter_raw_records(
    source: SourceSpec,
    paths: list[Path],
    *,
    limit: int | None = None,
    on_error: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[RawRecord]:
    readers = {
        "jsonl": _iter_jsonl,
        "parquet": _iter_parquet,
        "tar_jsonl": _iter_tar_jsonl,
    }
    reader = readers.get(source.reader)
    if reader is None:
        raise ValueError(f"unsupported reader {source.reader!r} for source {source.name}")
    emitted = 0
    for path in paths:
        for record in reader(path, on_error):
            yield record
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _safe_id(value: Any) -> str:
    text = str(value).strip()
    result = re.sub(r"[^0-9A-Za-z_.:-]+", "_", text).strip("_")
    return result[:160]


def _document_id(source: SourceSpec, record: RawRecord) -> str:
    raw_id = _first(record.row, ID_KEYS)
    if raw_id is None:
        location = f"{record.path}:{record.member or ''}:{record.row_index}"
        raw_id = hashlib.sha256(location.encode("utf-8")).hexdigest()[:24]
    return f"{source.name}:{_safe_id(raw_id)}"


def _license(source: SourceSpec, row: dict[str, Any]) -> str:
    if source.license_mode == "fixed":
        return source.license
    value = row.get("license") or row.get("licenses")
    metadata = row.get("metadata")
    if value is None and isinstance(metadata, dict):
        value = metadata.get("license") or metadata.get("licenses")
    if isinstance(value, list):
        return ",".join(sorted(str(item) for item in value if item))
    return "" if value is None else str(value).strip()


def _metadata_json(row: dict[str, Any]) -> str:
    value = {key: row[key] for key in PROVENANCE_KEYS if row.get(key) is not None}
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _wanjuan_text(row: dict[str, Any], subset: str) -> str | None:
    if "exam" not in subset.lower():
        value = row.get("content")
        return value if isinstance(value, str) else None
    question = row.get("q_main") or row.get("q_mean")
    parts = [str(question).strip()] if question else []
    for label in "abcde":
        option = row.get(f"option_{label}")
        if isinstance(option, str) and option.strip():
            parts.append(f"{label.upper()}. {option.strip()}")
    answer = row.get("std_ans") or row.get("answer")
    if answer:
        parts.append(f"答案：{str(answer).strip()}")
    detail = row.get("answer_detail")
    if isinstance(detail, str) and detail.strip():
        parts.append(detail.strip())
    return "\n".join(parts) or None


def _s2orc_text(row: dict[str, Any]) -> str | None:
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return text
    sections = row.get("sections")
    if not isinstance(sections, list):
        return None
    parts: list[str] = []
    title = row.get("title")
    if title:
        parts.append(str(title))
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = section.get("title") or section.get("heading")
        content = section.get("content") or section.get("text")
        if heading:
            parts.append(str(heading))
        if content:
            parts.append(str(content))
    return "\n\n".join(parts) or None


def adapt_record(source: SourceSpec, record: RawRecord) -> AdaptedRecord:
    row = record.row
    subset = record.path.parent.name
    if source.adapter == "wanjuan":
        text = _wanjuan_text(row, subset)
    elif source.adapter == "the_stack_v2":
        text = row.get("content") or row.get("text")
        if not isinstance(text, str):
            raise SourceRecordError(
                "missing_code_content",
                "The Stack V2 row contains IDs/metadata but no code content",
            )
    elif source.adapter == "s2orc":
        text = _s2orc_text(row)
    elif source.adapter in {"cci3_hq", "fineweb", "openwebmath", "generic_text"}:
        text = _first(row, TEXT_KEYS)
    else:
        raise SourceRecordError("unsupported_adapter", source.adapter)
    if not isinstance(text, str) or not text.strip():
        raise SourceRecordError("missing_text", f"no text found for adapter {source.adapter}")

    path_value = row.get("path") or row.get("file_path") or record.member
    source_path = str(path_value or record.path)
    revision = row.get("revision") or row.get("dump") or row.get("date")
    language = row.get("language") or row.get("lang")
    quality = row.get("quality_score") or row.get("score")
    return AdaptedRecord(
        doc_id=_document_id(source, record),
        text=text,
        source=source.name,
        subset=subset,
        source_path=source_path,
        revision=None if revision is None else str(revision),
        license=_license(source, row),
        url=None if row.get("url") is None else str(row["url"]),
        language=None if language is None else str(language),
        domain=source.default_domain,
        quality_prior=float(quality) if isinstance(quality, (int, float)) else None,
        metadata_json=_metadata_json(row),
    )


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    value = ZERO_WIDTH_RE.sub("", CONTROL_RE.sub("", value))
    lines = [line.rstrip() for line in value.split("\n")]
    return "\n".join(lines).strip()


def is_hanzi(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x323AF
    )


def _language(adapted: AdaptedRecord, hanzi: int, latin: int) -> str:
    if adapted.language:
        value = adapted.language.lower()
        if value in {"zh", "cmn", "zho", "chinese"}:
            return "zh"
        if value in {"en", "eng", "english"}:
            return "en"
        return value
    if hanzi > 0 and latin > 0 and min(hanzi, latin) / max(1, hanzi + latin) >= 0.05:
        return "zh_en_mixed"
    if hanzi > latin:
        return "zh"
    if latin > 0:
        return "en"
    return "unknown"


def _domain(adapted: AdaptedRecord, text: str, language: str) -> tuple[str, set[str]]:
    suffix = Path(adapted.source_path).suffix.lower()
    domain = adapted.domain
    tags: set[str] = set()
    if suffix in CODE_EXTENSIONS:
        domain = "code"
    elif suffix in STRUCTURED_EXTENSIONS:
        domain = "structured"
    elif MATH_RE.search(text):
        domain = "math"
    if language == "zh_en_mixed":
        tags.add("mixed")
    if adapted.source == "fineweb_zhtw":
        tags.add("traditional")
    if adapted.source in {"wikisource", "ect_krp"}:
        tags.add("classical")
    if len(text) >= 8192:
        tags.add("long_doc")
    return domain, tags


def clean_and_tag(source: SourceSpec, adapted: AdaptedRecord) -> CanonicalRecord:
    text = normalize_text(adapted.text)
    minimums = {
        "curated_zh": 20,
        "curated_en": 50,
        "web_zh": 80,
        "web_multilingual": 80,
        "broad_zh": 20,
        "code": 20,
        "math": 80,
        "scientific": 200,
        "classical": 10,
    }
    if len(text) < minimums.get(source.quality_profile, 20):
        raise SourceRecordError("too_short", f"document has {len(text)} characters")
    printable = sum(char.isprintable() or char in "\n\t" for char in text) / len(text)
    if printable < 0.98:
        raise SourceRecordError("low_printable_ratio", f"printable ratio is {printable:.4f}")
    if not adapted.license:
        raise SourceRecordError("missing_license", "document license is missing")

    hanzi = sum(is_hanzi(char) for char in text)
    latin = len(LATIN_RE.findall(text))
    digits = sum(char.isdigit() for char in text)
    language = _language(adapted, hanzi, latin)
    domain, tags = _domain(adapted, text, language)
    if language == "zh" and len(text) > 0:
        marker_count = sum(char in CLASSICAL_MARKERS for char in text)
        if marker_count / len(text) >= 0.025:
            tags.add("classical_candidate")

    metadata = {
        "parent_doc_id": adapted.doc_id,
        "source": adapted.source,
        "subset": adapted.subset,
        "source_path": adapted.source_path,
        "revision": adapted.revision,
        "license": adapted.license,
        "url": adapted.url,
        "language": language,
        "domain": domain,
        "char_count": len(text),
        "hanzi_count": hanzi,
        "latin_count": latin,
        "digit_count": digits,
        "quality_prior": adapted.quality_prior,
        "tags": sorted(tags),
        "metadata_json": adapted.metadata_json,
    }
    return CanonicalRecord(doc_id=adapted.doc_id, text=text, metadata=metadata)


def inspect_source(
    source: SourceSpec,
    *,
    max_files: int = 3,
    max_rows: int = 20,
) -> dict[str, Any]:
    paths = expand_source_paths(source)[:max_files]
    errors: list[dict[str, Any]] = []
    raw_fields: Counter[str] = Counter()
    adapted = 0
    accepted = 0
    rejected: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    for record in iter_raw_records(source, paths, limit=max_rows, on_error=errors.append):
        raw_fields.update(record.row.keys())
        try:
            value = adapt_record(source, record)
            adapted += 1
            canonical = clean_and_tag(source, value)
            accepted += 1
            if len(samples) < 3:
                samples.append(
                    {
                        "doc_id": canonical.doc_id,
                        "text_preview": canonical.text[:200],
                        "metadata": canonical.metadata,
                    }
                )
        except SourceRecordError as error:
            rejected[error.reason] += 1

    file_details: list[dict[str, Any]] = []
    for path in paths:
        detail: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size}
        if source.reader == "parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            detail.update(
                {
                    "rows": parquet.metadata.num_rows,
                    "row_groups": parquet.metadata.num_row_groups,
                    "fields": parquet.schema_arrow.names,
                }
            )
        elif source.reader == "tar_jsonl":
            with tarfile.open(path, "r:*") as archive:
                detail["members"] = [
                    member.name
                    for member in archive
                    if member.isfile() and member.name.lower().endswith((".jsonl", ".json"))
                ][:20]
        file_details.append(detail)

    return {
        "source": source.name,
        "reader": source.reader,
        "adapter": source.adapter,
        "source_contract_sha256": source_contract_hash(source),
        "inspected_files": file_details,
        "rows_requested": max_rows,
        "adapted_rows": adapted,
        "accepted_rows": accepted,
        "raw_fields": dict(raw_fields.most_common()),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "read_errors": errors[:100],
        "samples": samples,
        "passed": accepted > 0,
    }
