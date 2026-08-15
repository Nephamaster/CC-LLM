"""Build and audit the Han character coverage set used by the tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .unicode_ranges import CJK_BASIC, CJK_EXTENSION_A, UnicodeRange, chars_from_ranges, is_cjk_hanzi


@dataclass(frozen=True)
class HanziSource:
    name: str
    path: Path
    required: bool = False


@dataclass(frozen=True)
class HanziSetBuildConfig:
    output_dir: Path = Path("resources/hanzi")
    vocab_ranges: tuple[UnicodeRange, ...] = (CJK_BASIC, CJK_EXTENSION_A)
    tghz2013_path: Path = Path("resources/hanzi/tghz2013.txt")
    common_traditional_path: Path = Path("resources/hanzi/common_traditional.txt")
    rare_high_freq_path: Path = Path("resources/hanzi/rare_high_freq.txt")
    strict: bool = False
    output_filename: str = "hanzi_set.txt"
    meta_filename: str = "hanzi_set.meta.json"


@dataclass
class SourceMeta:
    name: str
    kind: str
    path: str | None = None
    required: bool = False
    exists: bool = True
    added: int = 0
    duplicates: int = 0
    invalid: int = 0
    total_valid: int = 0


@dataclass
class HanziSetResult:
    chars: set[str]
    meta: dict

    def sorted_chars(self) -> list[str]:
        return sorted(self.chars, key=ord)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _strip_bom(text: str) -> str:
    return text.lstrip("\ufeff")


def _extract_char_from_line(line: str) -> str | None:
    line = _strip_bom(line).strip()
    if not line or line.startswith("#"):
        return None
    first_field = line.split("\t", 1)[0].strip()
    if first_field.lower() == "char":
        return None
    return first_field[0] if first_field else None


def read_hanzi_file(path: Path) -> tuple[list[str], int]:
    chars: list[str] = []
    invalid = 0
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            char = _extract_char_from_line(line)
            if char is None:
                continue
            if is_cjk_hanzi(char):
                chars.append(char)
            else:
                invalid += 1
    return chars, invalid


def write_hanzi_lines(path: Path, chars: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="\n") as f:
        for char in chars:
            f.write(f"{char}\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


class HanziSetBuilder:
    def __init__(self, config: HanziSetBuildConfig):
        self.config = config

    def build(self) -> HanziSetResult:
        chars: set[str] = set()
        source_meta: list[SourceMeta] = []

        for range_ in self.config.vocab_ranges:
            before = len(chars)
            chars.update(chars_from_ranges((range_,)))
            source_meta.append(
                SourceMeta(
                    name=range_.name,
                    kind="unicode_range",
                    added=len(chars) - before,
                    total_valid=range_.size,
                )
            )

        file_sources = (
            HanziSource("tghz2013", self.config.tghz2013_path, required=True),
            HanziSource("common_traditional", self.config.common_traditional_path, required=True),
            HanziSource("rare_high_freq", self.config.rare_high_freq_path, required=False),
        )

        for source in file_sources:
            source_meta.append(self._add_file_source(chars, source))

        meta = self._build_meta(chars, source_meta)
        return HanziSetResult(chars=chars, meta=meta)

    def write(self, result: HanziSetResult) -> tuple[Path, Path]:
        output_path = self.config.output_dir / self.config.output_filename
        meta_path = self.config.output_dir / self.config.meta_filename
        write_hanzi_lines(output_path, result.sorted_chars())
        write_json(meta_path, result.meta)
        return output_path, meta_path

    def build_and_write(self) -> HanziSetResult:
        result = self.build()
        self.write(result)
        return result

    def _add_file_source(self, chars: set[str], source: HanziSource) -> SourceMeta:
        if not source.path.exists():
            if source.required and self.config.strict:
                raise FileNotFoundError(f"Required Hanzi source is missing: {source.path}")
            return SourceMeta(
                name=source.name,
                kind="file",
                path=str(source.path),
                required=source.required,
                exists=False,
            )

        values, invalid = read_hanzi_file(source.path)
        before = len(chars)
        duplicate_count = 0
        for char in values:
            if char in chars:
                duplicate_count += 1
            chars.add(char)

        return SourceMeta(
            name=source.name,
            kind="file",
            path=str(source.path),
            required=source.required,
            exists=True,
            added=len(chars) - before,
            duplicates=duplicate_count,
            invalid=invalid,
            total_valid=len(values),
        )

    def _build_meta(self, chars: set[str], source_meta: list[SourceMeta]) -> dict:
        missing_required = [
            source.name for source in source_meta if source.kind == "file" and source.required and not source.exists
        ]
        return {
            "generated_at": _utc_now_iso(),
            "num_chars": len(chars),
            "missing_required_sources": missing_required,
            "strict": self.config.strict,
            "sources": [source.__dict__ for source in source_meta],
        }


def build_hanzi_set(config: HanziSetBuildConfig | None = None) -> HanziSetResult:
    return HanziSetBuilder(config or HanziSetBuildConfig()).build()
