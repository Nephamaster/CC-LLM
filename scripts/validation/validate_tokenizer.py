"""Validate Phase 0 tokenizer invariants and chat behavior."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.validation.common import file_sha256, read_json, read_jsonl, utc_now_iso, write_json
from src.vocab.bpe_state import BpeState
from src.vocab.qwen3_char_tokenizer import Qwen3CharTokenizer, Qwen3CharTokenizerConfig, normalize_text
from src.vocab.unicode_ranges import count_hanzi, is_cjk_hanzi


def check_sample(wrapper: Qwen3CharTokenizer, row: dict[str, Any]) -> dict[str, Any] | None:
    text = str(row["text"])
    try:
        encoded = wrapper.encode(text, add_special_tokens=False)
        decoded = wrapper.decode(
            encoded["input_ids"],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
    except Exception as error:
        return {"id": row.get("id"), "category": row.get("category"), "reason": "encode_error", "error": repr(error)}

    source_hanzi = [char for char in normalize_text(text) if is_cjk_hanzi(char)]
    encoded_hanzi = [
        feature.get("char")
        for feature in encoded["feature_ids"]
        if feature.get("is_hanzi")
    ]
    reasons: list[str] = []
    if len(encoded["input_ids"]) != len(encoded["feature_ids"]):
        reasons.append("feature_length_mismatch")
    if decoded != normalize_text(text):
        reasons.append("roundtrip_mismatch")
    if encoded_hanzi != source_hanzi:
        reasons.append("hanzi_not_single_character_aligned")

    special_token = row.get("special_token")
    if special_token is not None:
        expected_id = wrapper.vocab.get(str(special_token))
        if expected_id is None or encoded["input_ids"] != [expected_id]:
            reasons.append("special_token_id_mismatch")

    if not reasons:
        return None
    return {
        "id": row.get("id"),
        "category": row.get("category"),
        "reason": reasons,
        "text": text,
        "decoded": decoded,
        "input_ids": encoded["input_ids"],
        "source_hanzi": source_hanzi,
        "encoded_hanzi": encoded_hanzi,
    }


def check_hanzi_inventory(wrapper: Qwen3CharTokenizer, hanzi_path: Path) -> dict[str, Any]:
    chars: list[str] = []
    seen: set[str] = set()
    with hanzi_path.open("rt", encoding="utf-8") as file:
        for line in file:
            for char in line.strip():
                if is_cjk_hanzi(char) and char not in seen:
                    chars.append(char)
                    seen.add(char)

    missing = [char for char in chars if char not in wrapper.char_token_ids]
    feature_mismatches: list[dict[str, Any]] = []
    for char in chars:
        token_id = wrapper.char_token_ids.get(char)
        if token_id is None:
            continue
        feature = wrapper.feature_rows.get(token_id, {})
        if not feature.get("is_hanzi") or feature.get("char") != char:
            feature_mismatches.append({"char": char, "token_id": token_id})
            if len(feature_mismatches) >= 20:
                break

    decode_failures: list[dict[str, Any]] = []
    available = [char for char in chars if char in wrapper.char_token_ids]
    for start in range(0, len(available), 1024):
        chunk = available[start : start + 1024]
        token_ids = [wrapper.char_token_ids[char] for char in chunk]
        decoded = wrapper.decode(token_ids, clean_up_tokenization_spaces=False, skip_special_tokens=False)
        expected = "".join(chunk)
        if decoded != expected:
            decode_failures.append({"start": start, "expected_head": expected[:40], "decoded_head": decoded[:40]})
            if len(decode_failures) >= 20:
                break

    return {
        "passed": not missing and not feature_mismatches and not decode_failures,
        "checked": len(chars),
        "missing_count": len(missing),
        "missing_head": [{"char": char, "codepoint": f"U+{ord(char):04X}"} for char in missing[:20]],
        "feature_mismatches": feature_mismatches,
        "decode_failures": decode_failures,
    }


def check_no_composite_hanzi(wrapper: Qwen3CharTokenizer) -> dict[str, Any]:
    state = BpeState.from_tokenizer(wrapper.tokenizer)
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


def check_special_tokens(wrapper: Qwen3CharTokenizer, model_path: Path) -> dict[str, Any]:
    tokenizer = wrapper.tokenizer
    vocab_size = len(tokenizer)
    failures: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for token in tokenizer.all_special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = wrapper.encode(token, add_special_tokens=False)["input_ids"]
        decoded = wrapper.decode(encoded, clean_up_tokenization_spaces=False, skip_special_tokens=False)
        passed = isinstance(token_id, int) and 0 <= token_id < vocab_size and encoded == [token_id] and decoded == token
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

    expected_ids = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    generation_path = model_path / "generation_config.json"
    if generation_path.exists():
        generation = read_json(generation_path)
        for key, expected in expected_ids.items():
            actual = generation.get(key)
            if expected is not None and actual != expected:
                failures.append(
                    {"file": "generation_config.json", "field": key, "expected": expected, "actual": actual}
                )

    return {
        "passed": not failures,
        "vocab_size": vocab_size,
        "tokenizer_ids": expected_ids,
        "special_tokens": token_rows,
        "config_checks": config_checks,
        "failures": failures,
    }


def check_chat_template(wrapper: Qwen3CharTokenizer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokenizer = wrapper.tokenizer
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
            encoded = wrapper.encode(rendered, add_special_tokens=False)
            decoded = wrapper.decode(
                encoded["input_ids"],
                clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            source_hanzi = [char for char in rendered if is_cjk_hanzi(char)]
            encoded_hanzi = [feature.get("char") for feature in encoded["feature_ids"] if feature.get("is_hanzi")]
            reasons: list[str] = []
            if not rendered:
                reasons.append("empty_render")
            if decoded != normalize_text(rendered):
                reasons.append("roundtrip_mismatch")
            if encoded_hanzi != source_hanzi:
                reasons.append("hanzi_alignment_mismatch")
            if add_generation_prompt:
                without_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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
    wrapper = Qwen3CharTokenizer(
        Qwen3CharTokenizerConfig(tokenizer_dir=args.model_path, features_dir=args.model_path / "features")
    )
    sample_failures = [failure for row in rows if (failure := check_sample(wrapper, row)) is not None]
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
        "hanzi_inventory": check_hanzi_inventory(wrapper, args.hanzi_set),
        "no_composite_hanzi_tokens": check_no_composite_hanzi(wrapper),
        "special_tokens": check_special_tokens(wrapper, args.model_path),
        "chat_template": check_chat_template(wrapper, rows),
    }
    return {
        "generated_at": utc_now_iso(),
        "model_path": str(args.model_path),
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
