"""Run Phase 0 model forward and migrated-weight diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from scripts.validation.common import file_sha256, numeric_summary, read_json, read_jsonl, utc_now_iso, write_json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def tensor_norms(weight: torch.Tensor, token_ids: list[int]) -> tuple[list[float], bool, int]:
    if not token_ids:
        return [], False, 0
    index = torch.tensor(token_ids, device=weight.device, dtype=torch.long)
    rows = weight.index_select(0, index)
    finite = bool(torch.isfinite(rows).all().item())
    norms = rows.float().norm(dim=1)
    zero_count = int((norms == 0).sum().item())
    return norms.cpu().tolist(), finite, zero_count


def check_migrated_weights(model: Any, model_path: Path) -> dict[str, Any]:
    mapping_path = model_path / "new_token_init_token_ids.json"
    if not mapping_path.exists():
        return {"passed": False, "reason": f"missing {mapping_path}"}

    mapping = read_json(mapping_path)
    new_token_ids = sorted(int(token_id) for token_id in mapping)
    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()
    output_weight = output_module.weight if output_module is not None else None
    vocab_size, hidden_size = input_weight.shape
    invalid_ids = [token_id for token_id in new_token_ids if not 0 <= token_id < vocab_size]
    valid_ids = [token_id for token_id in new_token_ids if 0 <= token_id < vocab_size]
    new_norms, new_finite, new_zero_count = tensor_norms(input_weight, valid_ids)

    new_id_set = set(valid_ids)
    reference_ids = [token_id for token_id in range(vocab_size) if token_id not in new_id_set][:4096]
    reference_norms, reference_finite, reference_zero_count = tensor_norms(input_weight, reference_ids)
    new_summary = numeric_summary(new_norms)
    reference_summary = numeric_summary(reference_norms)
    new_median = new_summary["median"]
    reference_median = reference_summary["median"]
    norm_ratio = (
        float(new_median) / float(reference_median)
        if isinstance(new_median, float) and isinstance(reference_median, float) and reference_median > 0
        else None
    )

    tied_actual = output_weight is not None and input_weight.data_ptr() == output_weight.data_ptr()
    tied_expected = bool(model.config.tie_word_embeddings)
    output_shape_matches = output_weight is not None and tuple(output_weight.shape) == (vocab_size, hidden_size)
    output_new_finite = True
    output_new_zero_count = 0
    if output_weight is not None and not tied_actual:
        _, output_new_finite, output_new_zero_count = tensor_norms(output_weight, valid_ids)

    passed = (
        vocab_size == model.config.vocab_size
        and not invalid_ids
        and bool(valid_ids)
        and new_finite
        and reference_finite
        and new_zero_count == 0
        and reference_zero_count == 0
        and output_shape_matches
        and output_new_finite
        and output_new_zero_count == 0
        and tied_actual == tied_expected
        and norm_ratio is not None
        and 0.1 <= norm_ratio <= 10.0
    )
    return {
        "passed": passed,
        "embedding_shape": list(input_weight.shape),
        "lm_head_shape": list(output_weight.shape) if output_weight is not None else None,
        "vocab_size_matches_config": vocab_size == model.config.vocab_size,
        "new_token_count": len(new_token_ids),
        "invalid_token_ids_head": invalid_ids[:20],
        "new_rows_finite": new_finite,
        "new_zero_norm_count": new_zero_count,
        "new_norms": new_summary,
        "reference_rows_finite": reference_finite,
        "reference_zero_norm_count": reference_zero_count,
        "reference_norms": reference_summary,
        "new_to_reference_median_norm_ratio": norm_ratio,
        "tie_word_embeddings_expected": tied_expected,
        "tie_word_embeddings_actual": tied_actual,
        "output_shape_matches": output_shape_matches,
        "output_new_rows_finite": output_new_finite,
        "output_new_zero_norm_count": output_new_zero_count,
    }


def load_model(model_path: Path, device: torch.device) -> tuple[Any, Any]:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.to(device)
    model.eval()
    return model, config

def run_sample(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    device: torch.device,
    max_length: int,
    max_loss: float,
    max_logit_abs: float,
) -> dict[str, Any]:
    encoded = tokenizer(str(row["text"]), add_special_tokens=False)
    full_length = len(encoded["input_ids"])
    token_ids = encoded["input_ids"][:max_length]
    if len(token_ids) < 2:
        raise ValueError(f"sample {row.get('id')} has fewer than two tokens")
    if min(token_ids) < 0 or max(token_ids) >= model.config.vocab_size:
        raise ValueError(f"sample {row.get('id')} contains a token id outside model vocabulary")

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    if outputs.loss is None:
        raise RuntimeError("model forward did not return loss")

    loss = float(outputs.loss.detach().float().cpu())
    logits_finite = bool(torch.isfinite(outputs.logits).all().item())
    logit_abs_max = float(outputs.logits.detach().abs().max().float().cpu())
    expected_shape = [1, len(token_ids), model.config.vocab_size]
    shape_matches = list(outputs.logits.shape) == expected_shape
    passed = (
        math.isfinite(loss)
        and loss <= max_loss
        and logits_finite
        and logit_abs_max <= max_logit_abs
        and shape_matches
    )
    return {
        "id": row.get("id"),
        "category": row.get("category"),
        "token_count": len(token_ids),
        "full_token_count": full_length,
        "truncated": full_length > max_length,
        "loss": loss,
        "loss_to_uniform_ratio": loss / math.log(model.config.vocab_size),
        "logits_finite": logits_finite,
        "logit_abs_max": logit_abs_max,
        "logits_shape": list(outputs.logits.shape),
        "shape_matches": shape_matches,
        "passed": passed,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.data_path)
    selected_rows = rows[: args.limit] if args.limit is not None else rows
    dataset_complete = len(rows) == args.expected_records
    device = resolve_device(args.device)
    model, config = load_model(args.model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)

    weight_checks = check_migrated_weights(model, args.model_path)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, start=1):
        try:
            result = run_sample(
                model,
                tokenizer,
                row,
                device,
                args.max_length,
                args.max_loss,
                args.max_logit_abs,
            )
            results.append(result)
        except Exception as error:
            errors.append({"id": row.get("id"), "category": row.get("category"), "error": repr(error)})
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if args.log_every and index % args.log_every == 0:
            print(json.dumps({"processed": index, "total": len(selected_rows), "errors": len(errors)}))

    category_losses: dict[str, list[float]] = defaultdict(list)
    for result in results:
        category_losses[str(result["category"])].append(float(result["loss"]))
    failed_results = [result for result in results if not result["passed"]]
    forward_passed = len(results) == len(selected_rows) and not errors and not failed_results
    report = {
        "generated_at": utc_now_iso(),
        "model_path": str(args.model_path),
        "data_path": str(args.data_path),
        "data_sha256": file_sha256(args.data_path),
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "thresholds": {
            "max_loss": args.max_loss,
            "max_logit_abs": args.max_logit_abs,
            "uniform_random_loss": math.log(config.vocab_size),
            "max_length": args.max_length,
        },
        "dataset": {
            "passed": dataset_complete,
            "records": len(rows),
            "expected_records": args.expected_records,
            "selected_records": len(selected_rows),
            "partial": len(selected_rows) != len(rows),
        },
        "weight_checks": weight_checks,
        "forward": {
            "passed": forward_passed,
            "completed": len(results),
            "error_count": len(errors),
            "failed_threshold_count": len(failed_results),
            "loss": numeric_summary([float(result["loss"]) for result in results]),
            "loss_by_category": {
                category: numeric_summary(values) for category, values in sorted(category_losses.items())
            },
            "errors": errors,
            "failed_thresholds": failed_results,
            "samples": results,
        },
        "passed": dataset_complete and weight_checks["passed"] and forward_passed,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen3-1.7B-Base-Char-PGCA"))
    parser.add_argument("--data-path", type=Path, default=Path("data/validation/forward_validation.jsonl"))
    parser.add_argument("--report-path", type=Path, default=Path("reports/validation/phase0_forward_report.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-records", type=int, default=300)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-loss", type=float, default=30.0)
    parser.add_argument("--max-logit-abs", type=float, default=10_000.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
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
