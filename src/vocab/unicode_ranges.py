"""Unicode range utilities for Han character vocabulary construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class UnicodeRange:
    name: str
    start: int
    end: int

    def contains_codepoint(self, codepoint: int) -> bool:
        return self.start <= codepoint <= self.end

    def contains(self, char: str) -> bool:
        return len(char) == 1 and self.contains_codepoint(ord(char))

    @property
    def size(self) -> int:
        return self.end - self.start + 1


CJK_EXTENSION_A = UnicodeRange("cjk_extension_a", 0x3400, 0x4DBF)
CJK_BASIC = UnicodeRange("cjk_basic", 0x4E00, 0x9FFF)
CJK_COMPATIBILITY = UnicodeRange("cjk_compatibility", 0xF900, 0xFAFF)
CJK_EXTENSION_B = UnicodeRange("cjk_extension_b", 0x20000, 0x2A6DF)
CJK_EXTENSION_C = UnicodeRange("cjk_extension_c", 0x2A700, 0x2B73F)
CJK_EXTENSION_D = UnicodeRange("cjk_extension_d", 0x2B740, 0x2B81F)
CJK_EXTENSION_E = UnicodeRange("cjk_extension_e", 0x2B820, 0x2CEAF)
CJK_EXTENSION_F = UnicodeRange("cjk_extension_f", 0x2CEB0, 0x2EBEF)
CJK_EXTENSION_G = UnicodeRange("cjk_extension_g", 0x30000, 0x3134F)
CJK_EXTENSION_H = UnicodeRange("cjk_extension_h", 0x31350, 0x323AF)
CJK_EXTENSION_I = UnicodeRange("cjk_extension_i", 0x2EBF0, 0x2EE5F)

CJK_RANGES: tuple[UnicodeRange, ...] = (
    CJK_EXTENSION_A,
    CJK_BASIC,
    CJK_COMPATIBILITY,
    CJK_EXTENSION_B,
    CJK_EXTENSION_C,
    CJK_EXTENSION_D,
    CJK_EXTENSION_E,
    CJK_EXTENSION_F,
    CJK_EXTENSION_G,
    CJK_EXTENSION_H,
    CJK_EXTENSION_I,
)

DEFAULT_VOCAB_RANGES: tuple[UnicodeRange, ...] = (CJK_BASIC,)


def iter_range_chars(range_: UnicodeRange) -> Iterator[str]:
    for codepoint in range(range_.start, range_.end + 1):
        yield chr(codepoint)


def chars_from_ranges(ranges: Iterable[UnicodeRange]) -> Iterator[str]:
    for range_ in ranges:
        yield from iter_range_chars(range_)


def is_in_ranges(char: str, ranges: Sequence[UnicodeRange]) -> bool:
    return len(char) == 1 and any(range_.contains(char) for range_ in ranges)


def is_cjk_hanzi(char: str) -> bool:
    return is_in_ranges(char, CJK_RANGES)


def is_vocab_hanzi(char: str, ranges: Sequence[UnicodeRange] = DEFAULT_VOCAB_RANGES) -> bool:
    return is_in_ranges(char, ranges)


def hanzi_chars(text: str) -> list[str]:
    return [char for char in text if is_cjk_hanzi(char)]


def count_hanzi(text: str) -> int:
    return sum(1 for char in text if is_cjk_hanzi(char))


def contains_hanzi(text: str) -> bool:
    return any(is_cjk_hanzi(char) for char in text)


def is_single_hanzi(text: str) -> bool:
    return len(text) == 1 and is_cjk_hanzi(text)


def is_all_hanzi(text: str) -> bool:
    return bool(text) and all(is_cjk_hanzi(char) for char in text)
