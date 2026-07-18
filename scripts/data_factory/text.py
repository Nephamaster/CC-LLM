"""Text normalization, filtering, and category checks."""

from __future__ import annotations

import hashlib
import json
import tomllib
import xml.etree.ElementTree as element_tree
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from scripts.validation.common import is_cjk_hanzi


ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.#/-]*")
CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\s\w]", re.UNICODE)
ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", re.UNICODE)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}"),
)

VERIFIED_LICENSES = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "MPL-2.0",
        "CC0-1.0",
        "CC-BY-2.5",
        "CC-BY-3.0",
        "CC-BY-4.0",
        "CC-BY-SA-2.5",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
        "ODC-By-1.0",
        "Unlicense",
    }
)

LICENSE_ALIASES = {
    "apache-2.0": "Apache-2.0",
    "mit": "MIT",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "isc": "ISC",
    "mpl-2.0": "MPL-2.0",
    "cc0": "CC0-1.0",
    "cc0-1.0": "CC0-1.0",
    "cc-by-2.5": "CC-BY-2.5",
    "cc-by-3.0": "CC-BY-3.0",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-2.5": "CC-BY-SA-2.5",
    "cc-by-sa-3.0": "CC-BY-SA-3.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "odc-by": "ODC-By-1.0",
    "odc-by-1.0": "ODC-By-1.0",
    "unlicense": "Unlicense",
}


def canonical_license(value: Any) -> str:
    text = str(value or "unknown").strip()
    return LICENSE_ALIASES.get(text.lower(), text)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.translate(ZERO_WIDTH)
    text = "".join(char for char in text if char in "\n\t" or ord(char) >= 0x20)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def deduplicate_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    seen: set[str] = set()
    kept: list[str] = []
    for paragraph in paragraphs:
        value = paragraph.strip("\n")
        key = re.sub(r"\s+", " ", value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(value)
    return "\n\n".join(kept)


def redact_pii(text: str) -> tuple[str, int]:
    text, emails = EMAIL_RE.subn("<EMAIL>", text)
    text, phones = PHONE_RE.subn("<PHONE>", text)
    return text, emails + phones


def has_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(char.isprintable() or char in "\n\t" for char in text)
    return printable / len(text)


def mixed_language_stats(text: str) -> dict[str, float | int | bool]:
    hanzi = sum(is_cjk_hanzi(char) for char in text)
    latin_words = len(LATIN_WORD_RE.findall(text))
    denominator = hanzi + latin_words
    ratio = hanzi / denominator if denominator else 0.0
    same_paragraph = any(
        any(is_cjk_hanzi(char) for char in paragraph) and LATIN_WORD_RE.search(paragraph)
        for paragraph in re.split(r"\n\s*\n", text)
    )
    return {"hanzi": hanzi, "latin_words": latin_words, "hanzi_ratio": ratio, "same_paragraph": same_paragraph}


def validate_mixed_text(text: str, quality: dict[str, Any]) -> str | None:
    stats = mixed_language_stats(text)
    if stats["hanzi"] < int(quality.get("mixed_min_hanzi", 20)):
        return "mixed_too_few_hanzi"
    if stats["latin_words"] < int(quality.get("mixed_min_latin_words", 5)):
        return "mixed_too_few_latin_words"
    if not float(quality.get("mixed_min_hanzi_ratio", 0.15)) <= stats["hanzi_ratio"] <= float(
        quality.get("mixed_max_hanzi_ratio", 0.85)
    ):
        return "mixed_ratio_out_of_range"
    if not stats["same_paragraph"]:
        return "mixed_not_cooccurring"
    return None


def validate_structured_text(text: str, path: Any) -> bool:
    suffix = str(path or "").lower().rsplit(".", 1)[-1]
    try:
        if suffix == "json":
            json.loads(text)
        elif suffix == "jsonl":
            for line in text.splitlines():
                if line.strip():
                    json.loads(line)
        elif suffix in {"yaml", "yml"}:
            import yaml

            try:
                yaml.safe_load(text)
            except yaml.YAMLError:
                return False
        elif suffix == "toml":
            tomllib.loads(text)
        elif suffix == "xml":
            element_tree.fromstring(text)
    except (ValueError, TypeError, element_tree.ParseError):
        return False
    return True


def clean_record(row: dict[str, Any], quality: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    text = row.get("text")
    if not isinstance(text, str):
        return None, "missing_text"
    text = deduplicate_paragraphs(normalize_text(text))
    if has_secret(text):
        return None, "secret_detected"
    text, pii_redactions = redact_pii(text)
    if len(text) < int(quality.get("min_chars", 50)):
        return None, "too_short"
    if printable_ratio(text) < float(quality.get("min_printable_ratio", 0.98)):
        return None, "low_printable_ratio"

    license_name = canonical_license(row.get("license"))
    allow_unverified = bool(row.pop("_allow_unverified_license", False))
    if license_name not in VERIFIED_LICENSES and not allow_unverified:
        return None, "unverified_license"
    if row.get("category") == "mixed_zh_en":
        reason = validate_mixed_text(text, quality)
        if reason is not None:
            return None, reason
    if (
        row.get("category") == "supplemental"
        and row.get("quota_group") == "structured"
        and not validate_structured_text(text, row.get("path"))
    ):
        return None, "invalid_structured"

    cleaned = dict(row)
    cleaned["text"] = text
    cleaned["license"] = license_name
    cleaned["license_status"] = "verified" if license_name in VERIFIED_LICENSES else "approved_exception"
    if pii_redactions:
        cleaned["pii_redactions"] = pii_redactions
    return cleaned, None


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def natural_shingles(text: str, width: int = 5) -> Iterable[bytes]:
    normalized = re.sub(r"\s+", " ", normalize_text(text))
    if any(is_cjk_hanzi(char) for char in normalized):
        values = list(normalized)
    else:
        values = ENGLISH_TOKEN_RE.findall(normalized.lower())
    if len(values) < width:
        if values:
            yield "\u241f".join(values).encode("utf-8")
        return
    for index in range(len(values) - width + 1):
        yield "\u241f".join(values[index : index + width]).encode("utf-8")


def code_shingles(text: str, width: int = 5) -> Iterable[bytes]:
    values = CODE_TOKEN_RE.findall(normalize_text(text))
    if len(values) < width:
        if values:
            yield "\u241f".join(values).encode("utf-8")
        return
    for index in range(len(values) - width + 1):
        yield "\u241f".join(values[index : index + width]).encode("utf-8")


def dedup_kind(row: dict[str, Any]) -> str:
    return "code" if row.get("category") == "supplemental" and row.get("quota_group") in {"code", "structured"} else "natural"

