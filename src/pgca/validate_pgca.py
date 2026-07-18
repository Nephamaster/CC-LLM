"""Validate a PGCA checkpoint through the standard Transformers AutoClass API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


EXPECTED_AUTO_MAP = {
    "AutoConfig": "configuration_qwen3_pgca.Qwen3PGCAConfig",
    "AutoModel": "modeling_qwen3_pgca.Qwen3PGCAModel",
    "AutoModelForCausalLM": "modeling_qwen3_pgca.Qwen3PGCAForCausalLM",
}
REQUIRED_RUNTIME_FILES = (
    "configuration_qwen3_pgca.py",
    "modeling_qwen3_pgca.py",
    "embedding/__init__.py",
    "embedding/config.py",
    "embedding/feature_embedding.py",
    "embedding/feature_memory.py",
    "pgca/__init__.py",
    "pgca/attention.py",
    "pgca/config.py",
)


def _check_repository(model_path: Path, config: Any) -> dict:
    missing_files = [path for path in REQUIRED_RUNTIME_FILES if not (model_path / path).is_file()]
    return {
        "model_type": config.model_type,
        "model_type_matches": config.model_type == "qwen3_pgca",
        "architectures": list(config.architectures or []),
        "architecture_matches": list(config.architectures or []) == ["Qwen3PGCAForCausalLM"],
        "auto_map": dict(config.auto_map or {}),
        "auto_map_matches": dict(config.auto_map or {}) == EXPECTED_AUTO_MAP,
        "missing_runtime_files": missing_files,
        "external_feature_index_absent": not (model_path / "features" / "feature_index.pt").exists(),
        "external_embedding_config_absent": not (model_path / "embedding_config.json").exists(),
    }


def _check_config(config: Any) -> dict:
    layers = list(config.pgca_layers or [])
    feature_vocab_sizes = dict(config.pgca_feature_vocab_sizes or {})
    return {
        "use_pgca": bool(config.use_pgca),
        "layers_in_range": all(0 <= layer_idx < config.num_hidden_layers for layer_idx in layers),
        "hidden_size_matches": config.hidden_size == config.pgca_feature_hidden_size,
        "head_shape_matches": config.num_attention_heads == config.pgca_num_attention_heads
        and config.num_key_value_heads == config.pgca_num_key_value_heads
        and config.head_dim == config.pgca_head_dim,
        "feature_slots_match": config.pgca_feature_slots == config.pgca_max_pinyin_per_char + 1,
        "feature_vocab_sizes_present": bool(feature_vocab_sizes),
        "pgca_layers": layers,
    }


def _check_modules(model: Any) -> dict:
    expected_layers = set(model.config.pgca_layers or [])
    actual_layers = {
        index
        for index, layer in enumerate(model.model.layers)
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


def _check_feature_memory(model: Any, input_ids: torch.Tensor) -> dict:
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
    buffer_prefix = "model.feature_memory_builder."
    persistent_keys = sorted(key for key in model.state_dict() if key.startswith(buffer_prefix))
    expected_buffer_keys = {
        buffer_prefix + key
        for key in (
            "is_hanzi",
            "pinyin_ids",
            "shengmu_ids",
            "yunmu_ids",
            "tone_ids",
            "pinyin_mask",
            "stroke_count_ids",
            "radical_stroke_ids",
            "structure_ids",
            "feature_index_ready",
        )
    }
    return {
        "available": True,
        "memory_shape": list(memory.shape),
        "mask_shape": list(mask.shape),
        "expected_memory_shape": expected_shape,
        "memory_shape_matches": list(memory.shape) == expected_shape,
        "mask_shape_matches": list(mask.shape) == expected_shape[:-1],
        "has_any_feature": bool(mask.any().item()),
        "index_rows": int(builder.is_hanzi.shape[0]),
        "index_rows_match": builder.is_hanzi.shape[0] == model.config.vocab_size,
        "persistent_buffer_keys": persistent_keys,
        "persistent_buffers_match": expected_buffer_keys.issubset(persistent_keys),
        "index_ready": bool(builder.feature_index_ready.item()),
    }


def _check_forward(model: Any, inputs: dict[str, torch.Tensor]) -> dict:
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    input_ids = inputs["input_ids"]
    expected_shape = [input_ids.shape[0], input_ids.shape[1], model.config.vocab_size]
    return {
        "logits_shape": list(outputs.logits.shape),
        "expected_logits_shape": expected_shape,
        "logits_shape_matches": list(outputs.logits.shape) == expected_shape,
        "loss_is_finite": bool(torch.isfinite(outputs.loss).item()) if outputs.loss is not None else False,
    }


def validate_pgca_model(model_path: Path, text: str) -> dict:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()

    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"]

    repository_checks = _check_repository(model_path, config)
    config_checks = _check_config(config)
    module_checks = _check_modules(model)
    feature_checks = _check_feature_memory(model, input_ids)
    forward_checks = _check_forward(model, inputs)
    repository_passed = (
        repository_checks["model_type_matches"]
        and repository_checks["architecture_matches"]
        and repository_checks["auto_map_matches"]
        and not repository_checks["missing_runtime_files"]
        and repository_checks["external_feature_index_absent"]
        and repository_checks["external_embedding_config_absent"]
    )
    feature_passed = (
        feature_checks.get("available", False)
        and feature_checks["memory_shape_matches"]
        and feature_checks["mask_shape_matches"]
        and feature_checks["has_any_feature"]
        and feature_checks["index_rows_match"]
        and feature_checks["persistent_buffers_match"]
        and feature_checks["index_ready"]
    )
    passed = (
        repository_passed
        and config_checks["use_pgca"]
        and config_checks["layers_in_range"]
        and config_checks["hidden_size_matches"]
        and config_checks["head_shape_matches"]
        and config_checks["feature_slots_match"]
        and config_checks["feature_vocab_sizes_present"]
        and module_checks["layers_match"]
        and module_checks["gates_match"]
        and module_checks["parameters_finite"]
        and feature_passed
        and forward_checks["logits_shape_matches"]
        and forward_checks["loss_is_finite"]
    )

    return {
        "model_path": str(model_path),
        "loader": {
            "config_class": type(config).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "model_class": type(model).__name__,
        },
        "repository": repository_checks,
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
    with report_path.open("wt", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
