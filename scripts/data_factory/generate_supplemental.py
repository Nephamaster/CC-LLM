"""Generate valid chat-template data and missing-Hanzi coverage records."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Iterable

from scripts.data_factory.io_utils import iter_jsonl
from scripts.validation.common import is_cjk_hanzi


def _paths(patterns: Iterable[str]) -> list[Path]:
    return sorted({Path(match) for pattern in patterns for match in glob.glob(pattern, recursive=True)})


def generate_chat(args: argparse.Namespace) -> int:
    from src.vocab.qwen3_char_tokenizer import Qwen3CharTokenizer, Qwen3CharTokenizerConfig

    wrapper = Qwen3CharTokenizer(
        Qwen3CharTokenizerConfig(args.model_path, args.model_path / "features")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("wt", encoding="utf-8", newline="\n") as output:
        for index, row in enumerate(iter_jsonl(_paths(args.inputs))):
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"chat input row {index} must contain a non-empty messages list")
            if not all(isinstance(message, dict) and message.get("role") and isinstance(message.get("content"), str) for message in messages):
                raise ValueError(f"chat input row {index} has invalid messages")
            rendered = wrapper.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            value = {
                "text": rendered,
                "source": row.get("source", "chat_template_generated"),
                "doc_id": row.get("doc_id", f"chat-template-{index}"),
                "license": row.get("license"),
                "url": row.get("url"),
                "path": row.get("path"),
                "revision": row.get("revision"),
                "category": "supplemental",
                "quota_group": "chat",
            }
            output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _read_hanzi(path: Path) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    with path.open("rt", encoding="utf-8") as file:
        for line in file:
            for char in line:
                if is_cjk_hanzi(char) and char not in seen:
                    seen.add(char)
                    values.append(char)
    return values


def _observed_hanzi(paths: list[Path]) -> set[str]:
    observed: set[str] = set()
    for row in iter_jsonl(paths):
        observed.update(char for char in str(row.get("text", "")) if is_cjk_hanzi(char))
    return observed


def _feature_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl([path]):
        char = row.get("char")
        if row.get("is_hanzi") and isinstance(char, str) and len(char) == 1:
            rows[char] = row
    return rows


def generate_hanzi(args: argparse.Namespace) -> int:
    targets = _read_hanzi(args.hanzi_set)
    observed = _observed_hanzi(_paths(args.observed)) if args.observed else set()
    features = _feature_rows(args.feature_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("wt", encoding="utf-8", newline="\n") as output:
        for char in targets:
            if char in observed:
                continue
            feature = features.get(char, {})
            pinyin = [value for value in feature.get("pinyin", []) if value]
            details = f"，可读作{'、'.join(pinyin)}" if pinyin else ""
            text = f"汉字“{char}”是本词表覆盖的单字{details}。在字符级模型中，它必须作为独立汉字编码。"
            row = {
                "text": text,
                "source": "hanzi_coverage_generated",
                "doc_id": f"hanzi-coverage-u{ord(char):x}",
                "license": args.license,
                "revision": args.revision,
                "category": "supplemental",
                "quota_group": "hanzi",
                "char": char,
                "synthetic": True,
            }
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    chat = subparsers.add_parser("chat")
    chat.add_argument("--inputs", nargs="+", required=True)
    chat.add_argument("--output", type=Path, required=True)
    chat.add_argument("--model-path", type=Path, default=Path("models/Qwen3-1.7B-Base-Char"))

    hanzi = subparsers.add_parser("hanzi")
    hanzi.add_argument("--hanzi-set", type=Path, default=Path("resources/hanzi/hanzi_set.txt"))
    hanzi.add_argument(
        "--feature-index",
        type=Path,
        default=Path("models/Qwen3-1.7B-Base-Char/features/char_feature_index.jsonl"),
    )
    hanzi.add_argument("--observed", nargs="*", default=[])
    hanzi.add_argument("--output", type=Path, required=True)
    hanzi.add_argument("--license", required=True)
    hanzi.add_argument("--revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = generate_chat(args) if args.kind == "chat" else generate_hanzi(args)
    print(json.dumps({"output": str(args.output), "records": count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

