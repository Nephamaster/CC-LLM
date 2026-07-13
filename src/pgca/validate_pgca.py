"""Validate a PGCA-enabled Qwen3 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from src.configuration_qwen3 import Qwen3Config
from src.modeling_qwen3 import Qwen3ForCausalLM
from src.vocab.qwen3_char_tokenizer import Qwen3CharTokenizer, Qwen3CharTokenizerConfig


def _check_config(config: Qwen3Config) -> dict:
    layers = list(getattr(config, "pgca_layers", []) or [])
    return {
        "use_pgca": bool(getattr(config, "use_pgca", False)),
        "layers_in_range": all(0 <= layer_idx < config.num_hidden_layers for layer_idx in layers),
        "hidden_size_matches": config.hidden_size == config.pgca_feature_hidden_size,
        "head_shape_matches": config.num_attention_heads == config.pgca_num_attention_heads
        and config.num_key_value_heads == config.pgca_num_key_value_heads
        and config.head_dim == config.pgca_head_dim,
        "pgca_layers": layers,
    }


def _check_modules(model: Qwen3ForCausalLM) -> dict:
    expected_layers = set(model.config.pgca_layers or [])
    actual_layers = {
        idx
        for idx, layer in enumerate(model.model.layers)
        if getattr(layer, "pgca_attn", None) is not None
    }
    pgca_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".pgca_attn." in name or "feature_memory_builder.feature_embedding." in name
    }
    meta_parameters = [name for name, parameter in pgca_parameters.items() if parameter.is_meta]
    non_finite_parameters = [
        name
        for name, parameter in pgca_parameters.items()
        if not parameter.is_meta and not bool(torch.isfinite(parameter).all().item())
    ]
    gate_values = {
        name: float(parameter.detach().cpu())
        for name, parameter in pgca_parameters.items()
        if name.endswith("pgca_attn.gate")
    }
    expected_gate = float(model.config.pgca_gate_init)
    gates_match = len(gate_values) == len(expected_layers) and all(
        abs(value - expected_gate) < 1e-8 for value in gate_values.values()
    )
    return {
        "expected_layers": sorted(expected_layers),
        "actual_layers": sorted(actual_layers),
        "layers_match": actual_layers == expected_layers,
        "gate_values": gate_values,
        "gates_match": gates_match,
        "meta_parameters": meta_parameters,
        "non_finite_parameters": non_finite_parameters,
        "parameters_finite": not meta_parameters and not non_finite_parameters,
    }


def _check_feature_memory(model: Qwen3ForCausalLM, input_ids: torch.Tensor) -> dict:
    builder = getattr(model.model, "feature_memory_builder", None)
    if builder is None:
        return {"available": False}

    with torch.no_grad():
        memory, mask = builder(input_ids)
    expected_shape = [
        input_ids.shape[0],
        input_ids.shape[1],
        int(model.config.pgca_feature_slots),
        int(model.config.hidden_size),
    ]
    return {
        "available": True,
        "memory_shape": list(memory.shape),
        "mask_shape": list(mask.shape),
        "expected_memory_shape": expected_shape,
        "memory_shape_matches": list(memory.shape) == expected_shape,
        "mask_shape_matches": list(mask.shape) == expected_shape[:-1],
        "has_any_feature": bool(mask.any().item()),
    }


def _check_forward(model: Qwen3ForCausalLM, input_ids: torch.Tensor) -> dict:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids)
    expected_shape = [input_ids.shape[0], input_ids.shape[1], model.config.vocab_size]
    return {
        "logits_shape": list(outputs.logits.shape),
        "expected_logits_shape": expected_shape,
        "logits_shape_matches": list(outputs.logits.shape) == expected_shape,
        "loss_is_finite": bool(torch.isfinite(outputs.loss).item()) if outputs.loss is not None else False,
    }


def validate_pgca_model(model_path: Path, text: str) -> dict:
    config = Qwen3Config.from_pretrained(model_path)
    model = Qwen3ForCausalLM.from_pretrained(model_path, config=config, torch_dtype="auto")
    builder = model.model.reload_feature_memory_builder(model_path)
    if builder is None:
        raise FileNotFoundError(f"PGCA feature artifacts are incomplete under {model_path}")
    model.eval()

    tokenizer = Qwen3CharTokenizer(
        Qwen3CharTokenizerConfig(
            tokenizer_dir=model_path,
            features_dir=model_path / "features",
        )
    )
    encoded = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)

    config_checks = _check_config(config)
    module_checks = _check_modules(model)
    feature_checks = _check_feature_memory(model, input_ids)
    forward_checks = _check_forward(model, input_ids)
    feature_passed = (
        feature_checks["memory_shape_matches"]
        and feature_checks["mask_shape_matches"]
        and feature_checks["has_any_feature"]
        if feature_checks.get("available")
        else False
    )
    passed = (
        config_checks["use_pgca"]
        and config_checks["layers_in_range"]
        and config_checks["hidden_size_matches"]
        and config_checks["head_shape_matches"]
        and module_checks["layers_match"]
        and module_checks["gates_match"]
        and module_checks["parameters_finite"]
        and feature_passed
        and forward_checks["logits_shape_matches"]
        and forward_checks["loss_is_finite"]
    )

    return {
        "model_path": str(model_path),
        "config": config_checks,
        "modules": module_checks,
        "feature_memory": feature_checks,
        "forward": forward_checks,
        "passed": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen3-1.7B-Base-Char-PGCA"))
    parser.add_argument("--text", type=str, default="中国ABC")
    args = parser.parse_args()

    report = validate_pgca_model(args.model_path, args.text)
    report_dir = args.model_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "pgca_validation_report.json"
    with report_path.open("wt", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    # print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
