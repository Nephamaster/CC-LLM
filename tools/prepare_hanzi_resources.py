"""Prepare normalized Hanzi resource files from resources/raw."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

IDS_STRUCTURE_MAP = {
    "⿰": "left_right",
    "⿲": "left_middle_right",
    "⿱": "top_bottom",
    "⿳": "top_middle_bottom",
    "⿴": "full_surround",
    "⿵": "upper_surround",
    "⿶": "lower_surround",
    "⿷": "left_surround",
    "⿸": "upper_left_surround",
    "⿹": "upper_right_surround",
    "⿺": "lower_left_surround",
    "⿻": "overlaid",
}


def is_cjk_hanzi(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x2CEB0 <= codepoint <= 0x2EBEF
        or 0x30000 <= codepoint <= 0x323AF
        or 0x2EBF0 <= codepoint <= 0x2EE5F
    )


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(f"{line}\n")


def parse_unihan_codepoint(raw: str) -> str:
    if not raw.startswith("U+"):
        raise ValueError(f"Bad Unihan codepoint: {raw}")
    return chr(int(raw[2:], 16))


def sort_key_tghz(value: str) -> tuple[int, ...]:
    index = value.split(":", 1)[0]
    parts = re.findall(r"\d+", index)
    return tuple(int(part) for part in parts) if parts else (10**9,)


def build_tghz2013(unihan_dir: Path, output_path: Path) -> int:
    readings_path = unihan_dir / "Unihan_Readings.txt"
    entries: list[tuple[tuple[int, ...], str]] = []
    with readings_path.open("rt", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3 or fields[1] != "kTGHZ2013":
                continue
            char = parse_unihan_codepoint(fields[0])
            if is_cjk_hanzi(char):
                entries.append((sort_key_tghz(fields[2]), char))
    entries.sort(key=lambda item: item[0])
    chars = [char for _key, char in entries]
    write_lines(output_path, chars)
    return len(chars)


def find_opencc_st_characters(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "STCharacters.txt",
        raw_dir / "STChracters.txt",
        raw_dir / "STCharacters.tsv",
        raw_dir / "STCharacters.ocd2",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() == ".txt":
            return candidate
    matches = sorted(raw_dir.glob("ST*Characters*.txt"))
    if matches:
        return matches[0]
    raise FileNotFoundError("Cannot find OpenCC STCharacters.txt under resources/raw")


def build_common_traditional(raw_dir: Path, output_path: Path) -> int:
    source_path = find_opencc_st_characters(raw_dir)
    chars: list[str] = []
    seen: set[str] = set()
    with source_path.open("rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            for char in fields[1]:
                if is_cjk_hanzi(char) and char not in seen:
                    seen.add(char)
                    chars.append(char)
    write_lines(output_path, chars)
    return len(chars)


def structure_from_ids(ids: str) -> str:
    if not ids:
        return "unknown"
    return IDS_STRUCTURE_MAP.get(ids[0], "single" if is_cjk_hanzi(ids[0]) else "unknown")


def build_structure(ids_path: Path, output_path: Path) -> int:
    rows: list[str] = ["char\tstructure"]
    seen: set[str] = set()
    with ids_path.open("rt", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            char = fields[1]
            if len(char) != 1 or not is_cjk_hanzi(char) or char in seen:
                continue
            seen.add(char)
            rows.append(f"{char}\t{structure_from_ids(fields[2])}")
    write_lines(output_path, rows)
    return len(rows) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("resources/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("resources/hanzi"))
    args = parser.parse_args()

    raw_dir = args.raw_dir
    output_dir = args.output_dir
    counts = {
        "tghz2013": build_tghz2013(raw_dir / "Unihan", output_dir / "tghz2013.txt"),
        "common_traditional": build_common_traditional(raw_dir, output_dir / "common_traditional.txt"),
        "structure": build_structure(raw_dir / "cjkvi-ids-master" / "ids.txt", output_dir / "structure.tsv"),
    }
    for name, count in counts.items():
        print(f"{name}: {count}")
    print("rare_high_freq: not generated; provide resources/hanzi/rare_high_freq.txt if needed")


if __name__ == "__main__":
    main()
