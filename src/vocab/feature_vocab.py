"""Build phonetic and structural feature vocabularies for Han characters."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .hanzi_set import read_hanzi_file
from .unicode_ranges import is_cjk_hanzi

NONE_TOKEN = "<none>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

FEATURE_SCHEMA_VERSION = 2
SPECIAL_TOKENS = {PAD_TOKEN, NONE_TOKEN, UNK_TOKEN}
POLYPHONE_EXPECTATIONS = {
    "行": {"xing2", "hang2"}, "重": {"zhong4", "chong2"},
    "长": {"chang2", "zhang3"}, "乐": {"le4", "yue4"},
    "还": {"hai2", "huan2"}, "和": {"he2", "he4"},
    "得": {"de2", "dei3", "de0"}, "了": {"le0", "liao3"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_pypinyin():
    try:
        import pypinyin
    except ImportError as exc:
        raise RuntimeError("feature vocab requires pypinyin; no silent fallback") from exc
    return pypinyin


@dataclass(frozen=True)
class FeatureVocabBuildConfig:
    hanzi_set_path: Path = Path("models/.build/hanzi/hanzi_set.txt")
    structure_path: Path = Path("models/.build/hanzi/structure.tsv")
    unihan_path: Path = Path("data/model_build/unihan/Unihan.zip")
    fallback_unihan_dir: Path = Path("data/model_build/unihan")
    output_dir: Path = Path("models/char/features")
    max_pinyin_per_char: int = 16
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
    wanted = {"kMandarin", "kHanyuPinyin", "kTotalStrokes", "kRSUnicode"}
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
                structures[char] = UNK_TOKEN if structure.lower() in {"unknown", "unk", "none", "<unk>"} else structure
    return structures


def normalize_pinyin_tone(pinyin: str) -> str:
    raw = pinyin.strip().lower().replace("u:", "v")
    if raw in SPECIAL_TOKENS:
        return raw
    tone_marks = {"\u0304": 1, "\u0301": 2, "\u030c": 3, "\u0300": 4}
    tones, chars = set(), []
    for char in unicodedata.normalize("NFD", raw):
        if char in tone_marks:
            tones.add(tone_marks[char])
        elif char == "\u0308" and chars and chars[-1] == "u":
            chars[-1] = "v"
        elif char == "\u0302" and chars and chars[-1] == "e":
            chars[-1] = "ê"
        else:
            chars.append(char)
    match = re.fullmatch(r"([a-zê]+)([0-5])?", "".join(chars))
    if not match:
        raise ValueError(f"not a pinyin syllable: {pinyin!r}")
    base, tone_suffix = match.groups()
    if tone_suffix is not None:
        tones.add(int(tone_suffix) % 5)
    if len(tones) > 1:
        raise ValueError(f"conflicting tones: {pinyin!r}")
    tone = next(iter(tones), 0)
    return f"{base}{tone}"


def split_pinyin(pinyin: str) -> tuple[str, str, int | None]:
    normalized = normalize_pinyin_tone(pinyin)
    if normalized in SPECIAL_TOKENS:
        return UNK_TOKEN, UNK_TOKEN, None
    from pypinyin.contrib.tone_convert import to_finals, to_initials

    body, tone = normalized[:-1], int(normalized[-1])
    # Syllabic nasals/interjections have no vowel final in pypinyin's strict helper.
    if body in {"m", "n", "ng", "hm", "hng"}:
        return "", body, tone
    initial, final = to_initials(normalized, strict=True), to_finals(normalized, strict=True)
    if not final:
        raise ValueError(f"no strict final for {pinyin!r}")
    return initial, final.replace("ü", "v"), tone


def pinyin_candidates(char: str, unihan_props: dict[str, str]) -> list[str]:
    return list(collect_pinyin_candidates(char, unihan_props))


def collect_pinyin_candidates(char: str, unihan_props: dict[str, str]) -> dict[str, list[str]]:
    module = require_pypinyin()
    values = module.pinyin(char, heteronym=True, style=module.Style.TONE3,
                          neutral_tone_with_five=True, strict=True, errors="ignore")
    sources = {"pypinyin": values[0] if values else []}
    hanyu = []
    hanyu_value = re.sub(r"([:,])\s+", r"\1", unihan_props.get("kHanyuPinyin", ""))
    for entry in hanyu_value.split():
        if ":" not in entry:
            raise ValueError(f"malformed kHanyuPinyin for {char}: {entry}")
        hanyu.extend(entry.split(":", 1)[1].split(","))
    sources["Unihan.kHanyuPinyin"] = hanyu
    sources["Unihan.kMandarin"] = unihan_props.get("kMandarin", "").split()
    result: dict[str, list[str]] = {}
    for source, candidates in sources.items():
        for value in candidates:
            normalized = normalize_pinyin_tone(value)
            if normalized in SPECIAL_TOKENS:
                raise ValueError(f"special token from {source} for {char}")
            split_pinyin(normalized)
            provenance = result.setdefault(normalized, [])
            if source not in provenance:
                provenance.append(source)
    return result


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
        require_pypinyin()
        if self.config.max_pinyin_per_char < 1:
            raise ValueError("max_pinyin_per_char must be positive")
        chars = read_hanzi_set(self.config.hanzi_set_path)
        if not chars:
            raise ValueError("Hanzi set must not be empty")
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
        candidate_counts, source_counts = Counter(), Counter()
        for char in chars:
            props = unihan.get(char, {})
            provenance = collect_pinyin_candidates(char, props)
            py_values = list(provenance)
            if len(py_values) > self.config.max_pinyin_per_char:
                raise ValueError(f"{char} has {len(py_values)} readings; increase --max-pinyin-per-char "
                                 f"(currently {self.config.max_pinyin_per_char}); refusing silent truncation")
            expected = POLYPHONE_EXPECTATIONS.get(char, set())
            if not expected.issubset(py_values):
                raise ValueError(f"missing common readings for {char}: {sorted(expected - set(py_values))}")
            candidate_counts[str(len(py_values))] += 1
            source_counts.update({source for sources in provenance.values() for source in sources})
            if not py_values:
                missing["pinyin"] += 1
                py_values = [UNK_TOKEN]

            split_values = [split_pinyin(value) for value in py_values]
            stroke_count = (props.get("kTotalStrokes", "").split() or [UNK_TOKEN])[0]
            radical_stroke = (props.get("kRSUnicode", "").split() or [UNK_TOKEN])[0]
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
            tones.update(str(tone) for _initial, _final, tone in split_values if tone is not None)
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
                "pinyin_sources": provenance,
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
                "pinyin_mask": [value not in SPECIAL_TOKENS for value in feat["pinyins"]],
                "stroke_count_id": vocabs["stroke_count"].get(feat["stroke_count"], vocabs["stroke_count"][UNK_TOKEN]),
                "radical_stroke_id": vocabs["radical_stroke"].get(
                    feat["radical_stroke"], vocabs["radical_stroke"][UNK_TOKEN]
                ),
                "structure_id": vocabs["structure"].get(feat["structure"], vocabs["structure"][UNK_TOKEN]),
                "raw": feat,
            }
            for char, feat in raw_features.items()
        }

        if not self.config.allow_missing_features and any(missing.values()):
            raise ValueError(f"missing features are disabled: {missing}")
        source_paths = [self.config.hanzi_set_path, self.config.structure_path]
        if self.config.unihan_path.is_file():
            source_paths.append(self.config.unihan_path)
        else:
            directory = self.config.unihan_path if self.config.unihan_path.is_dir() else self.config.fallback_unihan_dir
            source_paths.extend(sorted(directory.glob("Unihan*.txt")))
        manifest = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "split_convention": "pypinyin-strict; v=ü; syllabic m/n/ng/hm/hng use zero initial",
            "candidate_distribution": dict(candidate_counts),
            "max_observed_candidates": max(map(int, candidate_counts)),
            "pronunciation_source_char_counts": dict(source_counts),
            "source_sha256": {str(path): sha256_file(path) for path in source_paths if path.is_file()},
            "auxiliary_ignore_index": -100,
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
        from .feature_validation import validate_char_features

        report = validate_char_features(result.feature_vocabs, result.char_features)
        if not report["passed"]:
            raise ValueError(f"invalid feature semantics: {report}")
        result.manifest["semantic_validation"] = report
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
        result.manifest["artifact_sha256"] = {
            str(path.relative_to(output_dir)): sha256_file(path)
            for path in [*sorted(vocab_dir.glob("*_vocab.json")), output_dir / "char_features.json"]
        }
        write_json(output_dir / "feature_vocab_manifest.json", result.manifest)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hanzi-set-path", type=Path, default=Path("models/.build/hanzi/hanzi_set.txt"))
    parser.add_argument("--structure-path", type=Path, default=Path("models/.build/hanzi/structure.tsv"))
    parser.add_argument("--unihan-path", type=Path, default=Path("data/model_build/unihan/Unihan.zip"))
    parser.add_argument("--fallback-unihan-dir", type=Path, default=Path("data/model_build/unihan"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/char/features"))
    parser.add_argument("--max-pinyin-per-char", type=int, default=16)
    parser.add_argument("--require-complete-features", action="store_true")
    args = parser.parse_args()
    result = FeatureVocabBuilder(
        FeatureVocabBuildConfig(
            hanzi_set_path=args.hanzi_set_path,
            structure_path=args.structure_path,
            unihan_path=args.unihan_path,
            fallback_unihan_dir=args.fallback_unihan_dir,
            output_dir=args.output_dir,
            max_pinyin_per_char=args.max_pinyin_per_char,
            allow_missing_features=not args.require_complete_features,
        )
    ).build_and_write()
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
