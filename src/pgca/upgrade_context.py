"""Explicitly upgrade a complete Shared-4L checkpoint without changing its initial function."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from src.configuration_qwen3_pgca import Qwen3PGCAConfig
from src.embedding.feature_memory import FeatureMemoryBuilder
from src.modeling_qwen3_pgca import Qwen3PGCAForCausalLM
from .build_pgca import (
    AUTO_MAP,
    context_parameter_counts,
    copy_runtime_code,
    remove_legacy_runtime_artifacts,
    save_pretrained_without_auto_copy,
    validate_runtime_code,
    write_json,
)


def upgrade_context(model, stage_starts=None):
    """Retain the backbone/PGCA, copy C1 exactly, and initialize C2/C3 as identity residuals."""
    config = model.config
    if (
        not config.use_pgca
        or not config.pgca_context_enabled
        or getattr(config, "pgca_context_mode", "shared") != "shared"
        or config.pgca_context_num_layers != 4
    ):
        raise ValueError("upgrade requires a Shared-4L PGCA model with feature context enabled")
    data = config.to_dict()
    data.update(pgca_context_mode="hierarchical", pgca_context_stage_starts=stage_starts)
    target_config = Qwen3PGCAConfig.from_dict(data)
    original = model.model.feature_memory_builder
    builder = FeatureMemoryBuilder.empty(
        vocab_size=config.vocab_size,
        max_pinyin_per_char=config.pgca_max_pinyin_per_char,
        feature_embedding=original.feature_embedding,
        context_config=target_config,
    )
    builder.context_encoder = original.context_encoder
    for name, value in builder.named_buffers(recurse=False):
        setattr(builder, name, value.to(getattr(original, name).device))
    reference = next(original.context_encoder.parameters())
    for encoder in builder.context_refiners:
        encoder.to(device=reference.device, dtype=reference.dtype)
        encoder.apply(model._init_weights)

    # This scoped migration permits exactly the two new encoders; normal checkpoint loading stays strict.
    expected_missing = {key for key in builder.state_dict() if key.startswith("context_refiners.")}
    info = builder.load_state_dict(original.state_dict(), strict=False)
    if set(info.missing_keys) != expected_missing or info.unexpected_keys:
        raise RuntimeError(f"unexpected feature migration keys: {info}")
    with torch.no_grad():
        for encoder in builder.context_refiners:
            encoder.output_proj.weight.zero_()
    builder.validate_ready(vocab_size=config.vocab_size)
    builder.train(original.training)
    for value in (config, model.model.config):
        value.pgca_context_mode = "hierarchical"
        value.pgca_context_stage_starts = target_config.pgca_context_stage_starts
    model.model.feature_memory_builder = builder
    return {
        "context_mode": "hierarchical",
        "context_stage_starts": target_config.pgca_context_stage_starts,
        "pgca_layers": list(config.pgca_layers),
        "initialization": "C1 copied; C2/C3 zero output projections",
        "allowed_new_keys": sorted(expected_missing),
        "context_stage_parameter_counts": context_parameter_counts(builder),
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "branch_parameter_count": sum(
            p.numel()
            for n, p in model.named_parameters()
            if ".pgca_attn." in n or "feature_memory_builder." in n or "auxiliary_heads." in n
        ),
    }


def upgrade_checkpoint(model_path: Path, output_path: Path, stage_starts=None):
    model_path, output_path = model_path.resolve(), output_path.resolve()
    if model_path == output_path or model_path in output_path.parents or output_path in model_path.parents:
        raise ValueError("upgrade output must be separate from the source checkpoint directory")
    if (model_path / "adapter_config.json").exists():
        raise ValueError("upgrade requires a complete checkpoint; merge the adapter first")
    config = Qwen3PGCAConfig.from_pretrained(model_path)
    model, info = Qwen3PGCAForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype="auto",
        output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"source checkpoint is not a complete Shared-4L model: {info}")
    report = upgrade_context(model, stage_starts)
    shutil.copytree(
        model_path,
        output_path,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "model*.safetensors*",
            "pytorch_model*.bin*",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state*.pth",
            "checkpoint-*",
            "trainer_state.json",
            "training_args.bin",
        ),
    )
    for name in ("trainer_state.json", "training_args.bin"):
        (output_path / name).unlink(missing_ok=True)
    for pattern in ("model*.safetensors*", "pytorch_model*.bin*"):
        for path in output_path.glob(pattern):
            path.unlink()
    model.config.auto_map = dict(AUTO_MAP)
    model.config.architectures = ["Qwen3PGCAForCausalLM"]
    save_pretrained_without_auto_copy(model, output_path)
    copy_runtime_code(output_path)
    remove_legacy_runtime_artifacts(output_path)
    report.update(
        source_checkpoint=str(model_path),
        output_checkpoint=str(output_path),
        runtime_code=validate_runtime_code(output_path),
    )
    write_json(output_path / "hierarchical_upgrade_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--context-stage-starts", help="optional confirmation of the derived layer boundaries"
    )
    args = parser.parse_args()
    starts = [int(i) for i in args.context_stage_starts.split(",")] if args.context_stage_starts else None
    print(
        json.dumps(
            upgrade_checkpoint(args.model_path, args.output_path, starts), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
