"""Validate Phase 0 tokenizer invariants through Transformers AutoTokenizer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from scripts.validation.common import file_sha256, read_json, read_jsonl, utc_now_iso, write_json
from src.vocab.bpe_state import BpeState
from src.vocab.unicode_ranges import count_hanzi, is_cjk_hanzi


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(char for char in text if char in "\n\t" or ord(char) >= 0x20)


def encode(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(normalize_text(text), add_special_tokens=False)


def decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        clean_up_tokenization_spaces=False,
        skip_special_tokens=False,
    )


def encoded_hanzi(tokenizer: Any, token_ids: list[int]) -> list[str]:
    chars: list[str] = []
    for token_id in token_ids:
        value = decode(tokenizer, [token_id])
        if len(value) == 1 and is_cjk_hanzi(value):
            chars.append(value)
    return chars


def check_sample(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any] | None:
    text = str(row["text"])
    try:
        input_ids = encode(tokenizer, text)
        decoded = decode(tokenizer, input_ids)
    except Exception as error:
        return {
            "id": row.get("id"),
            "category": row.get("category"),
            "reason": "encode_error",
            "error": repr(error),
        }

    normalized = normalize_text(text)
    source_hanzi = [char for char in normalized if is_cjk_hanzi(char)]
    token_hanzi = encoded_hanzi(tokenizer, input_ids)
    reasons: list[str] = []
    if decoded != normalized:
        reasons.append("roundtrip_mismatch")
    if token_hanzi != source_hanzi:
        reasons.append("hanzi_not_single_character_aligned")

    special_token = row.get("special_token")
    if special_token is not None:
        expected_id = tokenizer.get_vocab().get(str(special_token))
        if expected_id is None or input_ids != [expected_id]:
            reasons.append("special_token_id_mismatch")

    if not reasons:
        return None
    return {
        "id": row.get("id"),
        "category": row.get("category"),
        "reason": reasons,
        "text": text,
        "decoded": decoded,
        "input_ids": input_ids,
        "source_hanzi": source_hanzi,
        "encoded_hanzi": token_hanzi,
    }


def load_hanzi(hanzi_path: Path) -> list[str]:
    chars: list[str] = []
    seen: set[str] = set()
    with hanzi_path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            for char in line.strip():
                if is_cjk_hanzi(char) and char not in seen:
                    chars.append(char)
                    seen.add(char)
    return chars


def check_hanzi_inventory(tokenizer: Any, hanzi_path: Path) -> dict[str, Any]:
    chars = load_hanzi(hanzi_path)
    token_ids: dict[str, int] = {}
    failures: list[dict[str, Any]] = []

    for char in chars:
        ids = encode(tokenizer, char)
        if len(ids) != 1 or decode(tokenizer, ids) != char:
            failures.append({"char": char, "codepoint": f"U+{ord(char):04X}", "ids": ids})
            if len(failures) >= 20:
                break
        else:
            token_ids[char] = ids[0]

    merge_failures: list[dict[str, Any]] = []
    if not failures:
        for start in range(0, len(chars), 1024):
            chunk = chars[start : start + 1024]
            expected = [token_ids[char] for char in chunk]
            actual = encode(tokenizer, "".join(chunk))
            if actual != expected:
                merge_failures.append(
                    {
                        "start": start,
                        "expected_length": len(expected),
                        "actual_length": len(actual),
                        "codepoint_head": [f"U+{ord(char):04X}" for char in chunk[:8]],
                    }
                )
                if len(merge_failures) >= 20:
                    break

    decode_failures: list[dict[str, Any]] = []
    if not failures:
        for start in range(0, len(chars), 1024):
            chunk = chars[start : start + 1024]
            expected = "".join(chunk)
            actual = decode(tokenizer, [token_ids[char] for char in chunk])
            if actual != expected:
                decode_failures.append(
                    {"start": start, "expected_head": expected[:40], "decoded_head": actual[:40]}
                )
                if len(decode_failures) >= 20:
                    break

    return {
        "passed": not failures and not merge_failures and not decode_failures,
        "checked": len(chars),
        "single_token_failures": failures,
        "merge_failures": merge_failures,
        "decode_failures": decode_failures,
    }


def check_no_composite_hanzi(tokenizer: Any) -> dict[str, Any]:
    state = BpeState.from_tokenizer(tokenizer)
    failures: list[dict[str, Any]] = []
    for token, token_id in state.vocab.items():
        if token in state.special_tokens:
            continue
        decoded = state.decode_piece(token)
        hanzi_count = count_hanzi(decoded)
        if hanzi_count and not (hanzi_count == 1 and len(decoded) == 1):
            failures.append({"token": token, "token_id": token_id, "decoded": decoded})
            if len(failures) >= 20:
                break
    return {"passed": not failures, "checked": len(state.vocab), "failures": failures}


def _flatten_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return value
    return []


def check_special_tokens(tokenizer: Any, model_path: Path) -> dict[str, Any]:
    vocab_size = len(tokenizer)
    failures: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for token in tokenizer.all_special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        input_ids = encode(tokenizer, token)
        decoded = decode(tokenizer, input_ids)
        passed = (
            isinstance(token_id, int)
            and 0 <= token_id < vocab_size
            and input_ids == [token_id]
            and decoded == token
        )
        token_rows.append({"token": token, "token_id": token_id, "passed": passed})
        if not passed:
            failures.append(token_rows[-1])

    config_checks: list[dict[str, Any]] = []
    for filename in ("config.json", "generation_config.json"):
        path = model_path / filename
        if not path.exists():
            failures.append({"file": filename, "reason": "missing_config"})
            continue
        config = read_json(path)
        for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
            for token_id in _flatten_token_ids(config.get(key)):
                passed = 0 <= token_id < vocab_size
                check = {"file": filename, "field": key, "token_id": token_id, "passed": passed}
                config_checks.append(check)
                if not passed:
                    failures.append(check)

    generation_path = model_path / "generation_config.json"
    if generation_path.exists():
        generation = read_json(generation_path)
        expected_ids = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        for key, expected in expected_ids.items():
            actual = generation.get(key)
            if expected is not None and actual != expected:
                failures.append(
                    {"file": "generation_config.json", "field": key, "expected": expected, "actual": actual}
                )
    else:
        expected_ids = {}

    return {
        "passed": not failures,
        "vocab_size": vocab_size,
        "tokenizer_ids": expected_ids,
        "special_tokens": token_rows,
        "config_checks": config_checks,
        "failures": failures,
    }


def check_chat_template(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0
    if not tokenizer.chat_template:
        return {"passed": False, "checked": 0, "failures": [{"reason": "chat_template_missing"}]}

    for row in rows:
        messages = row.get("messages")
        if not messages:
            continue
        checked += 1
        add_generation_prompt = bool(row.get("add_generation_prompt", False))
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            input_ids = encode(tokenizer, rendered)
            decoded = decode(tokenizer, input_ids)
            source_hanzi = [char for char in rendered if is_cjk_hanzi(char)]
            token_hanzi = encoded_hanzi(tokenizer, input_ids)
            reasons: list[str] = []
            if not rendered:
                reasons.append("empty_render")
            if decoded != normalize_text(rendered):
                reasons.append("roundtrip_mismatch")
            if token_hanzi != source_hanzi:
                reasons.append("hanzi_alignment_mismatch")
            if add_generation_prompt:
                without_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if rendered == without_prompt:
                    reasons.append("generation_prompt_not_added")
            if reasons:
                failures.append({"id": row.get("id"), "reason": reasons, "rendered": rendered})
        except Exception as error:
            failures.append({"id": row.get("id"), "reason": "chat_template_error", "error": repr(error)})

    if checked == 0:
        failures.append({"reason": "no_chat_cases"})
    return {"passed": not failures, "checked": checked, "failures": failures}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    sample_failures = [failure for row in rows if (failure := check_sample(tokenizer, row)) is not None]
    category_failures = Counter(str(failure.get("category", "unknown")) for failure in sample_failures)

    checks = {
        "dataset": {
            "passed": len(rows) == args.expected_records and not sample_failures,
            "records": len(rows),
            "expected_records": args.expected_records,
            "failure_count": len(sample_failures),
            "failures_by_category": dict(sorted(category_failures.items())),
            "failures_head": sample_failures[: args.max_failure_details],
        },
        "hanzi_inventory": check_hanzi_inventory(tokenizer, args.hanzi_set),
        "no_composite_hanzi_tokens": check_no_composite_hanzi(tokenizer),
        "special_tokens": check_special_tokens(tokenizer, args.model_path),
        "chat_template": check_chat_template(tokenizer, rows),
    }
    return {
        "generated_at": utc_now_iso(),
        "model_path": str(args.model_path),
        "tokenizer_class": type(tokenizer).__name__,
        "data_path": str(args.data_path),
        "data_sha256": file_sha256(args.data_path),
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen3-1.7B-Base-Char-PGCA"))
    parser.add_argument("--data-path", type=Path, default=Path("data/validation/tokenizer_validation.jsonl"))
    parser.add_argument("--hanzi-set", type=Path, default=Path("resources/hanzi/hanzi_set.txt"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/validation/phase0_tokenizer_report.json"))
    parser.add_argument("--expected-records", type=int, default=200)
    parser.add_argument("--max-failure-details", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(args)
    write_json(args.report_path, report)
    print(json.dumps({"report_path": str(args.report_path), "passed": report["passed"]}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
