"""Build token-id aligned feature indices from semantic and feature vocab outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .bpe_state import bytes_to_unicode, decode_bpe_piece, write_json
from .feature_vocab import FEATURE_SCHEMA_VERSION, NONE_TOKEN
from .feature_validation import check_vocab_binding, validate_char_features


@dataclass(frozen=True)
class FeatureIndexBuildConfig:
    tokenizer_dir: Path = Path("models/char")
    features_dir: Path = Path("models/char/features")
    output_jsonl: Path = Path("models/char/features/char_feature_index.jsonl")
    output_pt: Path = Path("models/char/features/feature_index.pt")
    max_pinyin_per_char: int | None = None
    write_torch_tensor: bool = True


@dataclass
class FeatureIndexResult:
    jsonl_rows: list[dict]
    manifest: dict


def read_json(path: Path):
    with path.open("rt", encoding="utf-8") as f:
        return json.load(f)


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {int(token_id): token for token, token_id in vocab.items()}


def load_feature_vocabs(features_dir: Path) -> dict[str, dict[str, int]]:
    vocab_dir = features_dir / "feature_vocabs"
    return {
        "pinyin": read_json(vocab_dir / "pinyin_vocab.json"),
        "shengmu": read_json(vocab_dir / "shengmu_vocab.json"),
        "yunmu": read_json(vocab_dir / "yunmu_vocab.json"),
        "tone": read_json(vocab_dir / "tone_vocab.json"),
        "stroke_count": read_json(vocab_dir / "stroke_count_vocab.json"),
        "radical_stroke": read_json(vocab_dir / "radical_stroke_vocab.json"),
        "structure": read_json(vocab_dir / "structure_vocab.json"),
    }


def pad_ids(values: list[int], length: int, pad_id: int) -> tuple[list[int], list[bool]]:
    if len(values) > length:
        raise ValueError("feature candidates exceed the index width; refusing truncation")
    clipped = values[:length]
    mask = [True] * len(clipped)
    if len(clipped) < length:
        pad_len = length - len(clipped)
        clipped = clipped + [pad_id] * pad_len
        mask = mask + [False] * pad_len
    return clipped, mask


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


class FeatureIndexBuilder:
    def __init__(self, config: FeatureIndexBuildConfig):
        self.config = config

    def build(self) -> FeatureIndexResult:
        feature_manifest = check_vocab_binding(self.config.features_dir)
        width = (int(feature_manifest["max_pinyin_per_char"]) if self.config.max_pinyin_per_char is None
                 else self.config.max_pinyin_per_char)
        if width != int(feature_manifest["max_pinyin_per_char"]):
            raise ValueError("index width must match feature-vocab max_pinyin_per_char")
        vocab = read_json(self.config.tokenizer_dir / "vocab.json")
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError("semantic vocabulary IDs must be unique and contiguous")
        id_to_token = invert_vocab(vocab)
        byte_decoder = {piece: byte for byte, piece in bytes_to_unicode().items()}
        char_features = read_json(self.config.features_dir / "char_features.json")
        feature_vocabs = load_feature_vocabs(self.config.features_dir)
        validation = validate_char_features(feature_vocabs, char_features)
        if not validation["passed"]:
            raise ValueError(f"invalid phonetic/glyph semantics: {validation}")

        none_ids = {name: values[NONE_TOKEN] for name, values in feature_vocabs.items()}
        rows: list[dict] = []
        hanzi_token_count = 0

        for token_id in range(len(id_to_token)):
            token = id_to_token[token_id]
            decoded_token = decode_bpe_piece(token, byte_decoder)
            char = decoded_token if len(decoded_token) == 1 and decoded_token in char_features else None
            feat = char_features.get(char) if char is not None else None
            is_hanzi = feat is not None
            if is_hanzi:
                hanzi_token_count += 1
                pinyin_ids, pinyin_mask = pad_ids(
                    [int(value) for value in feat["pinyin_ids"]],
                    width,
                    none_ids["pinyin"],
                )
                shengmu_ids, _ = pad_ids(
                    [int(value) for value in feat["shengmu_ids"]],
                    width,
                    none_ids["shengmu"],
                )
                yunmu_ids, _ = pad_ids(
                    [int(value) for value in feat["yunmu_ids"]],
                    width,
                    none_ids["yunmu"],
                )
                tone_ids, _ = pad_ids(
                    [int(value) for value in feat["tone_ids"]],
                    width,
                    none_ids["tone"],
                )
                stroke_count_id = int(feat["stroke_count_id"])
                radical_stroke_id = int(feat["radical_stroke_id"])
                structure_id = int(feat["structure_id"])
                pinyin_mask = [value > 2 for value in pinyin_ids]
            else:
                pinyin_ids = [none_ids["pinyin"]] * width
                shengmu_ids = [none_ids["shengmu"]] * width
                yunmu_ids = [none_ids["yunmu"]] * width
                tone_ids = [none_ids["tone"]] * width
                pinyin_mask = [False] * width
                stroke_count_id = none_ids["stroke_count"]
                radical_stroke_id = none_ids["radical_stroke"]
                structure_id = none_ids["structure"]

            rows.append(
                {
                    "token": token,
                    "char": char,
                    "token_id": token_id,
                    "is_hanzi": is_hanzi,
                    "pinyin_ids": pinyin_ids,
                    "shengmu_ids": shengmu_ids,
                    "yunmu_ids": yunmu_ids,
                    "tone_ids": tone_ids,
                    "pinyin_mask": pinyin_mask,
                    "stroke_count_id": stroke_count_id,
                    "radical_stroke_id": radical_stroke_id,
                    "structure_id": structure_id,
                    "label_masks": {
                        "pinyin": pinyin_mask,
                        "shengmu": [value > 2 and is_hanzi for value in shengmu_ids],
                        "yunmu": [value > 2 and is_hanzi for value in yunmu_ids],
                        "tone": [value > 2 and is_hanzi for value in tone_ids],
                        "stroke_count": is_hanzi and stroke_count_id > 2,
                        "radical_stroke": is_hanzi and radical_stroke_id > 2,
                        "structure": is_hanzi and structure_id > 2,
                    },
                }
            )

        manifest = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "tokenizer_dir": str(self.config.tokenizer_dir),
            "features_dir": str(self.config.features_dir),
            "vocab_size": len(id_to_token),
            "hanzi_token_count": hanzi_token_count,
            "hanzi_without_pinyin": sum(row["is_hanzi"] and not any(row["pinyin_mask"]) for row in rows),
            "max_pinyin_per_char": width,
            "output_jsonl": str(self.config.output_jsonl),
            "output_pt": str(self.config.output_pt) if self.config.write_torch_tensor else None,
        }
        return FeatureIndexResult(jsonl_rows=rows, manifest=manifest)

    def build_and_write(self) -> FeatureIndexResult:
        result = self.build()
        write_jsonl(self.config.output_jsonl, result.jsonl_rows)
        if self.config.write_torch_tensor:
            self._write_torch_tensor(result.jsonl_rows)
        write_json(self.config.features_dir / "feature_index_manifest.json", result.manifest)
        return result

    def _write_torch_tensor(self, rows: list[dict]) -> None:
        import torch

        self.config.output_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "is_hanzi": torch.tensor([row["is_hanzi"] for row in rows], dtype=torch.bool),
                "pinyin_ids": torch.tensor([row["pinyin_ids"] for row in rows], dtype=torch.long),
                "shengmu_ids": torch.tensor([row["shengmu_ids"] for row in rows], dtype=torch.long),
                "yunmu_ids": torch.tensor([row["yunmu_ids"] for row in rows], dtype=torch.long),
                "tone_ids": torch.tensor([row["tone_ids"] for row in rows], dtype=torch.long),
                "pinyin_mask": torch.tensor([row["pinyin_mask"] for row in rows], dtype=torch.bool),
                "stroke_count_ids": torch.tensor([row["stroke_count_id"] for row in rows], dtype=torch.long),
                "radical_stroke_ids": torch.tensor([row["radical_stroke_id"] for row in rows], dtype=torch.long),
                "structure_ids": torch.tensor([row["structure_id"] for row in rows], dtype=torch.long),
                **{f"{name}_mask": torch.tensor([row["label_masks"][name] for row in rows], dtype=torch.bool)
                   for name in ("shengmu", "yunmu", "tone", "stroke_count", "radical_stroke", "structure")},
            },
            self.config.output_pt,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("models/char"))
    parser.add_argument("--features-dir", type=Path, default=Path("models/char/features"))
    parser.add_argument("--no-torch", action="store_true")
    args = parser.parse_args()
    config = FeatureIndexBuildConfig(
        tokenizer_dir=args.tokenizer_dir,
        features_dir=args.features_dir,
        output_jsonl=args.features_dir / "char_feature_index.jsonl",
        output_pt=args.features_dir / "feature_index.pt",
        write_torch_tensor=not args.no_torch,
    )
    result = FeatureIndexBuilder(config).build_and_write()
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
