"""Validation checks for the character-level Qwen3 vocabulary artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from .bpe_state import BpeState, write_json
from .hanzi_set import read_hanzi_file
from .qwen3_char_tokenizer import Qwen3CharTokenizer, Qwen3CharTokenizerConfig
from .unicode_ranges import count_hanzi


@dataclass(frozen=True)
class VocabValidationConfig:
    char_model_path: Path = Path("models/Qwen3-1.7B-Base-Char")
    features_dir: Path = Path("models/Qwen3-1.7B-Base-Char/features")
    hanzi_set_path: Path = Path("resources/hanzi/hanzi_set.txt")
    output_path: Path = Path("models/Qwen3-1.7B-Base-Char/reports/validation_report.json")
    trust_remote_code: bool = True
    use_fast: bool = True
    max_hanzi_checks: int | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    with path.open("rt", encoding="utf-8") as f:
        return json.load(f)


def load_hanzi(path: Path) -> list[str]:
    chars, invalid = read_hanzi_file(path)
    if invalid:
        raise ValueError(f"Invalid Hanzi entries in {path}: {invalid}")
    return sorted(set(chars), key=ord)


class VocabValidator:
    def __init__(self, config: VocabValidationConfig):
        self.config = config

    def validate(self) -> dict:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.char_model_path,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=self.config.use_fast,
        )
        state = BpeState.from_tokenizer(tokenizer)
        hanzi_chars = load_hanzi(self.config.hanzi_set_path)
        if self.config.max_hanzi_checks is not None:
            hanzi_chars = hanzi_chars[: self.config.max_hanzi_checks]

        checks: dict[str, Any] = {
            "hanzi_single_token": self._check_hanzi_single_token(tokenizer, hanzi_chars),
            "no_composite_hanzi_vocab_tokens": self._check_no_composite_hanzi_tokens(state),
            "mapping_coverage": self._check_mapping_coverage(len(tokenizer)),
            "feature_index": self._check_feature_index(len(tokenizer)),
            "wrapper_alignment": self._check_wrapper_alignment(),
            "special_tokens": self._check_special_tokens(tokenizer),
        }
        passed = all(item["passed"] for item in checks.values())
        report = {
            "generated_at": _utc_now_iso(),
            "char_model_path": str(self.config.char_model_path),
            "features_dir": str(self.config.features_dir),
            "hanzi_set_path": str(self.config.hanzi_set_path),
            "passed": passed,
            "checks": checks,
        }
        write_json(self.config.output_path, report)
        return report

    @staticmethod
    def _check_hanzi_single_token(tokenizer, chars: list[str]) -> dict:
        failures: list[dict] = []
        for char in chars:
            ids = tokenizer.encode(char, add_special_tokens=False)
            if len(ids) != 1:
                failures.append({"char": char, "codepoint": f"U+{ord(char):04X}", "ids": ids})
                if len(failures) >= 20:
                    break
        return {"passed": not failures, "checked": len(chars), "failures": failures}

    @staticmethod
    def _check_no_composite_hanzi_tokens(state: BpeState) -> dict:
        failures: list[dict] = []
        for token, token_id in state.vocab.items():
            if token in state.special_tokens:
                continue
            decoded = state.decode_piece(token)
            hanzi_count = count_hanzi(decoded)
            if hanzi_count > 0 and not (hanzi_count == 1 and len(decoded) == 1):
                failures.append({"token": token, "token_id": token_id, "decoded": decoded})
                if len(failures) >= 20:
                    break
        return {"passed": not failures, "failures": failures}

    def _check_mapping_coverage(self, vocab_size: int) -> dict:
        mapping_path = self.config.char_model_path / "new2old_token_id.json"
        init_path = self.config.char_model_path / "new_token_init_token_ids.json"
        if not mapping_path.exists():
            return {"passed": False, "reason": f"missing {mapping_path}"}
        mapping = {int(key) for key in read_json(mapping_path)}
        init_ids = {int(key) for key in read_json(init_path)} if init_path.exists() else set()
        missing = [token_id for token_id in range(vocab_size) if token_id not in mapping and token_id not in init_ids]
        return {"passed": not missing, "vocab_size": vocab_size, "missing_head": missing[:20]}

    def _check_feature_index(self, vocab_size: int) -> dict:
        index_path = self.config.features_dir / "char_feature_index.jsonl"
        if not index_path.exists():
            return {"passed": False, "reason": f"missing {index_path}"}
        count = 0
        hanzi_count = 0
        bad_rows: list[dict] = []
        with index_path.open("rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                if row.get("is_hanzi"):
                    hanzi_count += 1
                    if not row.get("pinyin_mask") or not any(row["pinyin_mask"]):
                        bad_rows.append({"token_id": row.get("token_id"), "reason": "empty pinyin mask"})
        return {
            "passed": count == vocab_size and not bad_rows,
            "rows": count,
            "vocab_size": vocab_size,
            "hanzi_rows": hanzi_count,
            "bad_rows_head": bad_rows[:20],
        }

    def _check_wrapper_alignment(self) -> dict:
        index_path = self.config.features_dir / "char_feature_index.jsonl"
        manifest_path = self.config.features_dir / "feature_index_manifest.json"
        if not index_path.exists() or not manifest_path.exists():
            return {"passed": False, "reason": "feature index artifacts missing"}
        wrapper = Qwen3CharTokenizer(
            Qwen3CharTokenizerConfig(
                tokenizer_dir=self.config.char_model_path,
                features_dir=self.config.features_dir,
                trust_remote_code=self.config.trust_remote_code,
                use_fast=self.config.use_fast,
            )
        )
        samples = [
            "这是一个中文分词测试。",
            "行行重行行，银行行长行不行？",
            "Python 3.11: print(\"你好, Qwen3!\")",
            "URL: https://example.com?a=1",
        ]
        failures: list[dict] = []
        for text in samples:
            encoded = wrapper.encode(text)
            if len(encoded["input_ids"]) != len(encoded["feature_ids"]):
                failures.append({"text": text, "reason": "length mismatch"})
        return {"passed": not failures, "sample_count": len(samples), "failures": failures}

    @staticmethod
    def _check_special_tokens(tokenizer) -> dict:
        return {
            "passed": tokenizer.eos_token_id is not None,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--char-model-path", type=Path, default=Path("models/Qwen3-1.7B-Base-Char"))
    parser.add_argument("--features-dir", type=Path, default=Path("models/Qwen3-1.7B-Base-Char/features"))
    parser.add_argument("--hanzi-set-path", type=Path, default=Path("resources/hanzi/hanzi_set.txt"))
    parser.add_argument("--max-hanzi-checks", type=int, default=None)
    args = parser.parse_args()
    report = VocabValidator(
        VocabValidationConfig(
            char_model_path=args.char_model_path,
            features_dir=args.features_dir,
            hanzi_set_path=args.hanzi_set_path,
            max_hanzi_checks=args.max_hanzi_checks,
        )
    ).validate()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
