"""Natural-boundary text windows for Phase 1 candidate construction."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from scripts.data_factory.config import WindowingConfig


FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
LATEX_ENV_RE = re.compile(r"^\s*\\begin\{([^}]+)\}")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
SENTENCE_RE = re.compile(r".*?(?:[。！？!?；;]+[\"'”’）】》」』]*|$)", re.DOTALL)


@dataclass(frozen=True)
class BoundaryUnit:
    text: str
    protected: bool = False
    separator_before: str = ""


@dataclass(frozen=True)
class TextWindow:
    text: str
    token_count: int


class OversizedProtectedBlockError(ValueError):
    pass


def _split_prose(text: str, first_separator: str) -> Iterator[BoundaryUnit]:
    emitted = False
    for paragraph in re.split(r"\n\s*\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = [
            match.group(0).strip()
            for match in SENTENCE_RE.finditer(paragraph)
            if match.group(0).strip()
        ]
        for sentence_index, sentence in enumerate(sentences):
            separator = ""
            if sentence_index == 0:
                separator = first_separator if not emitted else "\n\n"
            yield BoundaryUnit(sentence, separator_before=separator)
            emitted = True


def _protected_block_end(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if stripped.startswith("$$"):
        if stripped == "$$":
            return "$$", "literal"
        if stripped.endswith("$$"):
            return "$$", "closed"
    if stripped.startswith(r"\["):
        if stripped == r"\[":
            return r"\]", "literal"
        if stripped.endswith(r"\]"):
            return r"\]", "closed"
    environment = LATEX_ENV_RE.match(line)
    if environment:
        marker = rf"\end{{{environment.group(1)}}}"
        return marker, "closed" if marker in line[environment.end() :] else "literal"
    fence = FENCE_RE.match(line)
    if fence:
        return fence.group(1), "fence"
    return None


def natural_boundary_units(text: str) -> list[BoundaryUnit]:
    """Split prose at natural boundaries while preserving structured blocks."""
    lines = text.splitlines()
    units: list[BoundaryUnit] = []
    prose: list[str] = []

    def separator() -> str:
        return "\n\n" if units else ""

    def flush_prose() -> None:
        if prose:
            units.extend(_split_prose("\n".join(prose), separator()))
            prose.clear()

    index = 0
    while index < len(lines):
        block_end = _protected_block_end(lines[index])
        if block_end is not None:
            flush_prose()
            marker, kind = block_end
            block = [lines[index]]
            index += 1
            while kind != "closed" and index < len(lines):
                block.append(lines[index])
                stripped = lines[index].strip()
                if (kind == "fence" and stripped == marker) or (
                    kind == "literal" and marker in stripped
                ):
                    index += 1
                    break
                index += 1
            units.append(
                BoundaryUnit(
                    "\n".join(block).strip(),
                    protected=True,
                    separator_before=separator(),
                )
            )
            continue

        line = lines[index]
        if index + 1 < len(lines) and "|" in line and TABLE_SEPARATOR_RE.match(lines[index + 1]):
            flush_prose()
            block = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                block.append(lines[index])
                index += 1
            units.append(
                BoundaryUnit(
                    "\n".join(block).strip(),
                    protected=True,
                    separator_before=separator(),
                )
            )
            continue

        prose.append(line)
        index += 1

    flush_prose()
    return units


def _render_units(units: Sequence[BoundaryUnit]) -> str:
    return "".join(
        (unit.separator_before if index else "") + unit.text
        for index, unit in enumerate(units)
    )


def _split_oversized_prose(
    unit: BoundaryUnit,
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[BoundaryUnit]:
    pieces: list[BoundaryUnit] = []
    remaining = unit.text
    while remaining:
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if count_tokens(remaining[:middle]) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        if count_tokens(remaining[:low]) > max_tokens:
            raise ValueError("a single character exceeds the configured token window")

        split_at = low
        if split_at < len(remaining):
            boundary = max(
                remaining.rfind(mark, 0, split_at + 1)
                for mark in ("。", "！", "？", "；", "，", "、", " ", "\n")
            )
            if boundary >= max(1, split_at // 2):
                split_at = boundary + 1
        pieces.append(
            BoundaryUnit(
                remaining[:split_at].strip(),
                separator_before=unit.separator_before if not pieces else "",
            )
        )
        remaining = remaining[split_at:].strip()
    return [piece for piece in pieces if piece.text]


def build_natural_windows_batched(
    text: str,
    count_many: Callable[[list[str]], list[int]],
    config: WindowingConfig,
) -> list[TextWindow]:
    """Build windows with batched unit counts and exact final verification."""
    units = natural_boundary_units(text)
    if not units:
        return []

    unit_counts = count_many([unit.text for unit in units])
    if len(unit_counts) != len(units):
        raise RuntimeError("tokenizer returned an unexpected number of unit lengths")

    expanded_units: list[BoundaryUnit] = []
    expanded_counts: list[int] = []
    scalar_count = lambda value: count_many([value])[0]
    for unit, token_count in zip(units, unit_counts, strict=True):
        if token_count <= config.max_tokens:
            expanded_units.append(unit)
            expanded_counts.append(token_count)
        elif unit.protected:
            raise OversizedProtectedBlockError(
                f"protected block has {token_count} tokens; maximum is {config.max_tokens}"
            )
        else:
            pieces = _split_oversized_prose(unit, scalar_count, config.max_tokens)
            piece_counts = count_many([piece.text for piece in pieces])
            expanded_units.extend(pieces)
            expanded_counts.extend(piece_counts)

    groups: list[list[BoundaryUnit]] = []
    current: list[BoundaryUnit] = []
    estimated_tokens = 0
    for unit, token_count in zip(expanded_units, expanded_counts, strict=True):
        separator_tokens = 1 if current and unit.separator_before else 0
        candidate_tokens = estimated_tokens + separator_tokens + token_count
        if current and candidate_tokens > config.target_tokens:
            if estimated_tokens >= config.min_tokens or candidate_tokens > config.max_tokens:
                groups.append(current)
                current = []
                estimated_tokens = 0
                separator_tokens = 0
        current.append(unit)
        estimated_tokens += separator_tokens + token_count
    if current:
        groups.append(current)

    texts = [_render_units(group) for group in groups]
    exact_counts = count_many(texts)
    windows: list[TextWindow] = []
    for value, token_count in zip(texts, exact_counts, strict=True):
        if token_count > config.max_tokens:
            windows.extend(build_natural_windows(value, scalar_count, config))
        else:
            windows.append(TextWindow(value, token_count))

    if len(windows) >= 2 and windows[-1].token_count < config.min_tokens:
        merged_text = "\n\n".join([windows[-2].text, windows[-1].text])
        merged_tokens = scalar_count(merged_text)
        if merged_tokens <= config.max_tokens:
            windows[-2:] = [TextWindow(merged_text, merged_tokens)]

    return [window for window in windows if window.token_count >= config.min_tokens]

def build_natural_windows(
    text: str,
    count_tokens: Callable[[str], int],
    config: WindowingConfig,
) -> list[TextWindow]:
    """Build exact-token windows without splitting fenced code, display math, or tables."""
    units: list[BoundaryUnit] = []
    for unit in natural_boundary_units(text):
        token_count = count_tokens(unit.text)
        if token_count <= config.max_tokens:
            units.append(unit)
        elif unit.protected:
            raise OversizedProtectedBlockError(
                f"protected block has {token_count} tokens; maximum is {config.max_tokens}"
            )
        else:
            units.extend(_split_oversized_prose(unit, count_tokens, config.max_tokens))

    windows: list[TextWindow] = []
    current: list[BoundaryUnit] = []

    def flush() -> None:
        if not current:
            return
        value = _render_units(current)
        windows.append(TextWindow(value, count_tokens(value)))
        current.clear()

    for unit in units:
        candidate = _render_units([*current, unit])
        candidate_tokens = count_tokens(candidate)
        if current and candidate_tokens > config.target_tokens:
            current_tokens = count_tokens(_render_units(current))
            if current_tokens >= config.min_tokens or candidate_tokens > config.max_tokens:
                flush()
        current.append(unit)

    flush()
    if len(windows) >= 2 and windows[-1].token_count < config.min_tokens:
        merged_text = "\n\n".join([windows[-2].text, windows[-1].text])
        merged_tokens = count_tokens(merged_text)
        if merged_tokens <= config.max_tokens:
            windows[-2:] = [TextWindow(merged_text, merged_tokens)]

    return [window for window in windows if window.token_count >= config.min_tokens]
