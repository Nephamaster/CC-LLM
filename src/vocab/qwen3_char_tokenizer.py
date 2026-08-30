"""Tokenizer wrapper that emits input ids and aligned Hanzi feature ids."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

@dataclass(frozen=True)
class Qwen3CharTokenizerConfig:
    tokenizer_dir: str | Path = Path("models/Qwen3-1.7B-Base-Char")
    features_dir: Path = Path("models/Qwen3-1.7B-Base-Char/features")
    trust_remote_code: bool = True
    use_fast: bool = True


def read_json(path: Path):
    with path.open("rt", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(char for char in text if char == "\n" or char == "\t" or ord(char) >= 0x20)


class Qwen3CharTokenizer:
    def __init__(self, config: Qwen3CharTokenizerConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_dir,
            trust_remote_code=config.trust_remote_code,
            use_fast=config.use_fast,
        )
        self.vocab = self.tokenizer.get_vocab()
        self.feature_rows = self._load_feature_rows(config.features_dir)
        self.char_token_ids = self._load_char_token_ids()
        self.none_feature = self._make_none_feature(config.features_dir)
        self.special_tokens = sorted(
            set(getattr(self.tokenizer, "all_special_tokens", []) or []),
            key=len,
            reverse=True,
        )

    def _load_feature_rows(self, features_dir: Path) -> dict[int, dict]:
        jsonl_path = features_dir / "char_feature_index.jsonl"
        rows: dict[int, dict] = {}
        with jsonl_path.open("rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[int(row["token_id"])] = row
        return rows

    def _load_char_token_ids(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for token_id, row in self.feature_rows.items():
            char = row.get("char")
            if row.get("is_hanzi") and isinstance(char, str) and len(char) == 1:
                mapping[char] = int(token_id)
        return mapping

    def _make_none_feature(self, features_dir: Path) -> dict:
        manifest = read_json(features_dir / "feature_index_manifest.json")
        max_pinyin = int(manifest["max_pinyin_per_char"])
        return {
            "is_hanzi": False,
            "pinyin_ids": [0] * max_pinyin,
            "shengmu_ids": [0] * max_pinyin,
            "yunmu_ids": [0] * max_pinyin,
            "tone_ids": [0] * max_pinyin,
            "pinyin_mask": [False] * max_pinyin,
            "stroke_count_id": 0,
            "radical_stroke_id": 0,
            "structure_id": 0,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> dict[str, list[Any]]:
        text = normalize_text(text)
        input_ids: list[int] = []
        feature_ids: list[dict] = []

        position = 0
        while position < len(text):
            special = self._match_special_token(text, position)
            if special is not None:
                token_id = int(self.vocab[special])
                input_ids.append(token_id)
                feature_ids.append(self._feature_for_token(token_id))
                position += len(special)
                continue

            char = text[position]
            token_id = self.char_token_ids.get(char)
            if token_id is not None:
                # Characters explicitly covered by the Char vocabulary are hard
                # boundaries and must always remain exactly one semantic token.
                input_ids.append(token_id)
                feature_ids.append(self._feature_for_token(token_id))
                position += 1
                continue

            # Everything outside the explicit Char vocabulary, including rare CJK
            # Extension characters, falls back to the preserved Qwen byte/BPE
            # tokenizer.  Stop only at an explicitly covered Hanzi or a special
            # token so the fallback path cannot absorb a target Hanzi.
            next_position = position + 1
            while next_position < len(text):
                if (
                    self._match_special_token(text, next_position) is not None
                    or text[next_position] in self.char_token_ids
                ):
                    break
                next_position += 1

            span = text[position:next_position]
            span_ids = self.tokenizer.encode(span, add_special_tokens=False)
            if not span_ids:
                raise ValueError(f"Fallback tokenizer produced no tokens for span: {span!r}")

            decoded_span = self.tokenizer.decode(
                span_ids,
                clean_up_tokenization_spaces=False,
            )
            if decoded_span != span:
                raise ValueError(
                    "Fallback tokenizer is not reversible for span: "
                    f"span={span!r}, decoded={decoded_span!r}"
                )

            for fallback_token_id in span_ids:
                fallback_token_id = int(fallback_token_id)
                self._assert_fallback_token(fallback_token_id)
                input_ids.append(fallback_token_id)
                feature_ids.append(self._feature_for_token(fallback_token_id))
            position = next_position

        if add_special_tokens:
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            feature_ids = [self._feature_for_token(int(token_id)) for token_id in input_ids]

        return {"input_ids": input_ids, "feature_ids": feature_ids}

    def decode(self, input_ids: list[int], **kwargs) -> str:
        return self.tokenizer.decode(input_ids, **kwargs)

    def _match_special_token(self, text: str, position: int) -> str | None:
        for token in self.special_tokens:
            if token and text.startswith(token, position):
                return token
        return None

    def _feature_for_token(self, token_id: int) -> dict:
        return self.feature_rows.get(int(token_id), self.none_feature)

    def _assert_fallback_token(self, token_id: int) -> None:
        token = self.tokenizer.convert_ids_to_tokens(int(token_id))
        decoded = self.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        target_hanzi = [char for char in decoded if char in self.char_token_ids]
        if target_hanzi:
            raise ValueError(
                "Fallback span produced a token containing an explicitly covered Hanzi: "
                f"id={token_id}, token={token!r}, decoded={decoded!r}, "
                f"target_hanzi={target_hanzi!r}"
            )

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[Any]]:
        return self.encode(text, add_special_tokens=add_special_tokens)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("models/Qwen3-1.7B-Base-Char"))
    parser.add_argument("--features-dir", type=Path, default=Path("models/Qwen3-1.7B-Base-Char/features"))
    args = parser.parse_args()
    tokenizer = Qwen3CharTokenizer(Qwen3CharTokenizerConfig(args.tokenizer_dir, args.features_dir))
    encoded = tokenizer.encode(args.text)
    print(json.dumps(encoded, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()