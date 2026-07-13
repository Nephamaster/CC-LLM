"""Build a PGCA-enabled Qwen3 checkpoint from a character-level Qwen3 checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from src.configuration_qwen3 import Qwen3Config
from src.modeling_qwen3 import Qwen3PGCAForCausalLM


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_layers(value: str | None, num_hidden_layers: int) -> list[int]:
    if not value:
        width = max(1, num_hidden_layers // 3)
        start = max(0, (num_hidden_layers - width) // 2)
        return list(range(start, start + width))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def update_config(config_path: Path, pgca_layers: list[int]) -> dict:
    with config_path.open("rt", encoding="utf-8") as f:
        data = json.load(f)

    hidden_size = int(data["hidden_size"])
    num_attention_heads = int(data["num_attention_heads"])
    num_key_value_heads = int(data.get("num_key_value_heads", num_attention_heads))
    head_dim = int(data.get("head_dim", hidden_size // num_attention_heads))

    data.update(
        {
            "architectures": ["Qwen3PGCAForCausalLM"],
            "use_pgca": True,
            "pgca_layers": pgca_layers,
            "pgca_num_attention_heads": num_attention_heads,
            "pgca_num_key_value_heads": num_key_value_heads,
            "pgca_head_dim": head_dim,
            "pgca_gate_init": 0.0,
            "pgca_dropout": float(data.get("attention_dropout", 0.0)),
            "pgca_feature_slots": 9,
            "pgca_feature_hidden_size": hidden_size,
            "pgca_build_features_in_model": True,
        }
    )

    with config_path.open("wt", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return data


def reset_pgca_gates(model: Qwen3PGCAForCausalLM, value: float) -> None:
    for name, param in model.named_parameters():
        if name.endswith("pgca_attn.gate"):
            param.data.fill_(value)


def build_pgca_model(model_path: Path, output_path: Path, pgca_layers_arg: str | None) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_path, output_path, dirs_exist_ok=True)

    config_path = output_path / "config.json"
    with config_path.open("rt", encoding="utf-8") as f:
        original_config = json.load(f)
    pgca_layers = parse_layers(pgca_layers_arg, int(original_config["num_hidden_layers"]))
    updated_config = update_config(config_path, pgca_layers)

    config = Qwen3Config.from_pretrained(output_path)
    model, loading_info = Qwen3PGCAForCausalLM.from_pretrained(
        output_path,
        config=config,
        output_loading_info=True,
        torch_dtype="auto",
    )
    reset_pgca_gates(model, float(config.pgca_gate_init))
    model.save_pretrained(output_path)

    gate_values = {
        name: float(param.detach().cpu())
        for name, param in model.named_parameters()
        if name.endswith("pgca_attn.gate")
    }
    report = {
        "base_model_path": str(model_path),
        "output_model_path": str(output_path),
        "pgca_layers": pgca_layers,
        "vocab_size": int(updated_config["vocab_size"]),
        "hidden_size": int(updated_config["hidden_size"]),
        "missing_keys": loading_info.get("missing_keys", []),
        "unexpected_keys": loading_info.get("unexpected_keys", []),
        "mismatched_keys": loading_info.get("mismatched_keys", []),
        "pgca_gate_values": gate_values,
        "pgca_parameter_count": sum(p.numel() for n, p in model.named_parameters() if ".pgca_attn." in n),
    }

    report_path = output_path / "pgca_migration_report.json"
    with report_path.open("wt", encoding="utf-8", newline="\n") as f:
        json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)
        f.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen3-1.7B-Base-Char"))
    parser.add_argument("--output-path", type=Path, default=Path("models/Qwen3-1.7B-Base-Char-PGCA"))
    parser.add_argument("--pgca-layers", type=str, default=None)
    args = parser.parse_args()

    report = build_pgca_model(args.model_path, args.output_path, args.pgca_layers)
    print(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
