"""Build phonetic and structural feature vocabularies for Han characters."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .hanzi_set import read_hanzi_file
from .unicode_ranges import is_cjk_hanzi

NONE_TOKEN = "<none>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

PINYIN_INITIALS = (
    "zh",
    "ch",
    "sh",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "r",
    "z",
    "c",
    "s",
    "y",
    "w",
)

PINYIN_TONE_MARKS = {
    "ā": ("a", 1), "á": ("a", 2), "ǎ": ("a", 3), "à": ("a", 4),
    "ē": ("e", 1), "é": ("e", 2), "ě": ("e", 3), "è": ("e", 4),
    "ī": ("i", 1), "í": ("i", 2), "ǐ": ("i", 3), "ì": ("i", 4),
    "ō": ("o", 1), "ó": ("o", 2), "ǒ": ("o", 3), "ò": ("o", 4),
    "ū": ("u", 1), "ú": ("u", 2), "ǔ": ("u", 3), "ù": ("u", 4),
    "ǖ": ("v", 1), "ǘ": ("v", 2), "ǚ": ("v", 3), "ǜ": ("v", 4), "ü": ("v", 0),
    "Ā": ("a", 1), "Á": ("a", 2), "Ǎ": ("a", 3), "À": ("a", 4),
    "Ē": ("e", 1), "É": ("e", 2), "Ě": ("e", 3), "È": ("e", 4),
    "Ī": ("i", 1), "Í": ("i", 2), "Ǐ": ("i", 3), "Ì": ("i", 4),
    "Ō": ("o", 1), "Ó": ("o", 2), "Ǒ": ("o", 3), "Ò": ("o", 4),
    "Ū": ("u", 1), "Ú": ("u", 2), "Ǔ": ("u", 3), "Ù": ("u", 4),
    "Ǖ": ("v", 1), "Ǘ": ("v", 2), "Ǚ": ("v", 3), "Ǜ": ("v", 4), "Ü": ("v", 0),
}


@dataclass(frozen=True)
class FeatureVocabBuildConfig:
    hanzi_set_path: Path = Path("resources/hanzi/hanzi_set.txt")
    structure_path: Path = Path("resources/hanzi/structure.tsv")
    unihan_path: Path = Path("resources/unihan/Unihan.zip")
    fallback_unihan_dir: Path = Path("resources/raw/Unihan")
    output_dir: Path = Path("models/Qwen3-1.7B-Base-Char/features")
    max_pinyin_per_char: int = 8
    allow_missing_features: bool = True


@dataclass
class FeatureVocabResult:
    feature_vocabs: dict[str, dict[str, int]]
    char_features: dict[str, dict]
    manifest: dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, data, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def iter_unihan_lines(path: Path, fallback_dir: Path) -> Iterable[str]:
    if path.exists() and path.is_file():
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.endswith(".txt"):
                    continue
                with zf.open(name) as f:
                    for raw in f:
                        yield raw.decode("utf-8")
        return

    if path.exists() and path.is_dir():
        source_dir = path
    else:
        source_dir = fallback_dir

    if source_dir.exists():
        for txt_path in sorted(source_dir.glob("Unihan*.txt")):
            with txt_path.open("rt", encoding="utf-8") as f:
                yield from f


def parse_unihan(unihan_path: Path, fallback_dir: Path) -> dict[str, dict[str, str]]:
    props: dict[str, dict[str, str]] = {}
    wanted = {"kMandarin", "kTotalStrokes", "kRSUnicode"}
    for line in iter_unihan_lines(unihan_path, fallback_dir):
        if not line or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3 or fields[1] not in wanted:
            continue
        char = chr(int(fields[0][2:], 16))
        if is_cjk_hanzi(char):
            props.setdefault(char, {})[fields[1]] = fields[2]
    return props


def read_structure_table(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    structures: dict[str, str] = {}
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2 or fields[0].lower() == "char":
                continue
            char, structure = fields[0], fields[1]
            if len(char) == 1 and is_cjk_hanzi(char):
                structures[char] = structure
    return structures


def normalize_pinyin_tone(pinyin: str) -> str:
    raw = pinyin.strip().replace("u:", "v")
    tone = 0
    chars: list[str] = []
    for char in raw:
        mapped = PINYIN_TONE_MARKS.get(char)
        if mapped is None:
            chars.append(char.lower())
            continue
        base, marked_tone = mapped
        chars.append(base)
        if marked_tone:
            tone = marked_tone
    normalized = "".join(chars).replace("ü", "v").lower()
    match = re.match(r"^([a-zv]+)([0-5])?$", normalized)
    if not match:
        return normalized
    base, tone_suffix = match.groups()
    if tone_suffix is not None:
        tone = int(tone_suffix)
    if tone == 5:
        tone = 0
    return f"{base}{tone}"


def split_pinyin(pinyin: str) -> tuple[str, str, int]:
    normalized = normalize_pinyin_tone(pinyin)
    match = re.match(r"^([a-zv]+)([0-4])$", normalized)
    if not match:
        return "", normalized, 0
    body, tone_raw = match.groups()
    initial = ""
    for candidate in PINYIN_INITIALS:
        if body.startswith(candidate):
            initial = candidate
            break
    final = body[len(initial) :]
    return initial, final, int(tone_raw)


def pinyin_candidates(char: str, unihan_props: dict[str, str]) -> list[str]:
    try:
        from pypinyin import Style, pinyin

        values = pinyin(char, heteronym=True, style=Style.TONE3, neutral_tone_with_five=True, strict=True)
        candidates = values[0] if values else []
    except Exception:
        candidates = []

    normalized: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        item = normalize_pinyin_tone(str(value))
        if item and item not in seen:
            seen.add(item)
            normalized.append(item)

    if normalized:
        return normalized

    mandarin = unihan_props.get("kMandarin", "")
    for value in mandarin.split():
        item = normalize_pinyin_tone(value)
        if item and item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def build_vocab(values: Iterable[str], *, include_empty: bool = False) -> dict[str, int]:
    specials = [PAD_TOKEN, NONE_TOKEN, UNK_TOKEN]
    if include_empty:
        specials.append("")
    vocab: dict[str, int] = {}
    for token in specials:
        if token not in vocab:
            vocab[token] = len(vocab)
    for value in sorted({item for item in values if item is not None}):
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def read_hanzi_set(path: Path) -> list[str]:
    chars, invalid = read_hanzi_file(path)
    if invalid:
        raise ValueError(f"Invalid entries in Hanzi set: {path}, invalid={invalid}")
    return sorted(set(chars), key=ord)


class FeatureVocabBuilder:
    def __init__(self, config: FeatureVocabBuildConfig):
        self.config = config

    def build(self) -> FeatureVocabResult:
        chars = read_hanzi_set(self.config.hanzi_set_path)
        unihan = parse_unihan(self.config.unihan_path, self.config.fallback_unihan_dir)
        structures = read_structure_table(self.config.structure_path)

        raw_features: dict[str, dict] = {}
        pinyins: set[str] = set()
        initials: set[str] = set()
        finals: set[str] = set()
        tones: set[str] = {"0", "1", "2", "3", "4"}
        stroke_counts: set[str] = set()
        radical_strokes: set[str] = set()
        structure_values: set[str] = set()

        missing = {"pinyin": 0, "stroke_count": 0, "radical_stroke": 0, "structure": 0}
        for char in chars:
            props = unihan.get(char, {})
            py_values = pinyin_candidates(char, props)[: self.config.max_pinyin_per_char]
            if not py_values:
                missing["pinyin"] += 1
                py_values = [UNK_TOKEN]

            split_values = [split_pinyin(value) for value in py_values]
            stroke_count = props.get("kTotalStrokes", UNK_TOKEN).split()[0]
            radical_stroke = props.get("kRSUnicode", UNK_TOKEN).split()[0]
            structure = structures.get(char, UNK_TOKEN)

            if stroke_count == UNK_TOKEN:
                missing["stroke_count"] += 1
            if radical_stroke == UNK_TOKEN:
                missing["radical_stroke"] += 1
            if structure == UNK_TOKEN:
                missing["structure"] += 1

            pinyins.update(py_values)
            initials.update(initial for initial, _final, _tone in split_values)
            finals.update(final for _initial, final, _tone in split_values)
            tones.update(str(tone) for _initial, _final, tone in split_values)
            stroke_counts.add(stroke_count)
            radical_strokes.add(radical_stroke)
            structure_values.add(structure)

            raw_features[char] = {
                "pinyins": py_values,
                "shengmus": [initial for initial, _final, _tone in split_values],
                "yunmus": [final for _initial, final, _tone in split_values],
                "tones": [tone for _initial, _final, tone in split_values],
                "stroke_count": stroke_count,
                "radical_stroke": radical_stroke,
                "structure": structure,
            }

        vocabs = {
            "pinyin": build_vocab(pinyins),
            "shengmu": build_vocab(initials, include_empty=True),
            "yunmu": build_vocab(finals),
            "tone": build_vocab(tones),
            "stroke_count": build_vocab(stroke_counts),
            "radical_stroke": build_vocab(radical_strokes),
            "structure": build_vocab(structure_values),
        }

        char_features = {
            char: {
                "pinyin_ids": [vocabs["pinyin"].get(value, vocabs["pinyin"][UNK_TOKEN]) for value in feat["pinyins"]],
                "shengmu_ids": [
                    vocabs["shengmu"].get(value, vocabs["shengmu"][UNK_TOKEN]) for value in feat["shengmus"]
                ],
                "yunmu_ids": [vocabs["yunmu"].get(value, vocabs["yunmu"][UNK_TOKEN]) for value in feat["yunmus"]],
                "tone_ids": [vocabs["tone"].get(str(value), vocabs["tone"][UNK_TOKEN]) for value in feat["tones"]],
                "stroke_count_id": vocabs["stroke_count"].get(feat["stroke_count"], vocabs["stroke_count"][UNK_TOKEN]),
                "radical_stroke_id": vocabs["radical_stroke"].get(
                    feat["radical_stroke"], vocabs["radical_stroke"][UNK_TOKEN]
                ),
                "structure_id": vocabs["structure"].get(feat["structure"], vocabs["structure"][UNK_TOKEN]),
                "raw": feat,
            }
            for char, feat in raw_features.items()
        }

        manifest = {
            "generated_at": _utc_now_iso(),
            "hanzi_set_path": str(self.config.hanzi_set_path),
            "structure_path": str(self.config.structure_path),
            "unihan_path": str(self.config.unihan_path),
            "fallback_unihan_dir": str(self.config.fallback_unihan_dir),
            "num_chars": len(chars),
            "max_pinyin_per_char": self.config.max_pinyin_per_char,
            "missing": missing,
            "vocab_sizes": {name: len(vocab) for name, vocab in vocabs.items()},
            "tone_neutral_value": 0,
        }
        return FeatureVocabResult(feature_vocabs=vocabs, char_features=char_features, manifest=manifest)

    def build_and_write(self) -> FeatureVocabResult:
        result = self.build()
        output_dir = self.config.output_dir
        vocab_dir = output_dir / "feature_vocabs"
        write_json(vocab_dir / "pinyin_vocab.json", result.feature_vocabs["pinyin"])
        write_json(vocab_dir / "shengmu_vocab.json", result.feature_vocabs["shengmu"])
        write_json(vocab_dir / "yunmu_vocab.json", result.feature_vocabs["yunmu"])
        write_json(vocab_dir / "tone_vocab.json", result.feature_vocabs["tone"])
        write_json(vocab_dir / "stroke_count_vocab.json", result.feature_vocabs["stroke_count"])
        write_json(vocab_dir / "radical_stroke_vocab.json", result.feature_vocabs["radical_stroke"])
        write_json(vocab_dir / "structure_vocab.json", result.feature_vocabs["structure"])
        write_json(output_dir / "char_features.json", result.char_features)
        write_json(output_dir / "feature_vocab_manifest.json", result.manifest)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hanzi-set-path", type=Path, default=Path("resources/hanzi/hanzi_set.txt"))
    parser.add_argument("--structure-path", type=Path, default=Path("resources/hanzi/structure.tsv"))
    parser.add_argument("--unihan-path", type=Path, default=Path("resources/unihan/Unihan.zip"))
    parser.add_argument("--fallback-unihan-dir", type=Path, default=Path("resources/raw/Unihan"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/Qwen3-1.7B-Base-Char/features"))
    args = parser.parse_args()
    result = FeatureVocabBuilder(
        FeatureVocabBuildConfig(
            hanzi_set_path=args.hanzi_set_path,
            structure_path=args.structure_path,
            unihan_path=args.unihan_path,
            fallback_unihan_dir=args.fallback_unihan_dir,
            output_dir=args.output_dir,
        )
    ).build_and_write()
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
