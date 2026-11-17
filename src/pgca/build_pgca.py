"""Build a self-contained PGCA-enabled Qwen3 checkpoint."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import unicodedata
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.configuration_qwen3_pgca import (
    DEFAULT_AUX_GLYPH_WEIGHT, DEFAULT_AUX_PHONETIC_WEIGHT, Qwen3PGCAConfig, validate_auxiliary_weights,
)
from src.embedding.config import EmbeddingFeatureConfig
from src.embedding.feature_memory import load_feature_index, validate_feature_index
from src.modeling_qwen3_pgca import Qwen3PGCAForCausalLM


AUTO_MAP = {
    "AutoConfig": "configuration_qwen3_pgca.Qwen3PGCAConfig",
    "AutoModel": "modeling_qwen3_pgca.Qwen3PGCAModel",
    "AutoModelForCausalLM": "modeling_qwen3_pgca.Qwen3PGCAForCausalLM",
}
RUNTIME_FILES = {
    "embedding_config.py": "embedding/config.py",
    "feature_embedding.py": "embedding/feature_embedding.py",
    "feature_memory.py": "embedding/feature_memory.py",
    "context_encoder.py": "embedding/context_encoder.py",
    "pgca_attention.py": "pgca/attention.py",
    "pgca_config.py": "pgca/config.py",
    "pgca_auxiliary.py": "pgca/auxiliary.py",
}
REMOTE_IMPORT_REWRITES = {
    "configuration_qwen3_pgca.py": {".pgca.config": ".pgca_config"},
    "modeling_qwen3_pgca.py": {
        ".embedding.config": ".embedding_config",
        ".embedding.feature_embedding": ".feature_embedding",
        ".embedding.feature_memory": ".feature_memory",
        ".pgca.attention": ".pgca_attention",
        ".embedding.context_encoder": ".context_encoder",
        ".pgca.auxiliary": ".pgca_auxiliary",
    },
    "feature_embedding.py": {".config": ".embedding_config"},
}


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


def read_json(path: Path) -> dict:
    with path.open("rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    with path.open("wt", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def update_config(
    config_path: Path,
    embedding_config_path: Path,
    pgca_layers: list[int],
) -> dict:
    data = read_json(config_path)
    embedding = read_json(embedding_config_path)

    hidden_size = int(data["hidden_size"])
    vocab_size = int(data["vocab_size"])
    num_attention_heads = int(data["num_attention_heads"])
    num_key_value_heads = int(data.get("num_key_value_heads", num_attention_heads))
    head_dim = int(data.get("head_dim", hidden_size // num_attention_heads))
    if int(embedding["semantic_vocab_size"]) != vocab_size:
        raise ValueError("embedding_config semantic_vocab_size does not match config vocab_size")
    if int(embedding["d_model"]) != hidden_size:
        raise ValueError("embedding_config d_model does not match config hidden_size")

    data.pop("pgca_build_features_in_model", None)
    data.pop("pgca_gate_init", None)
    data.update(
        {
            "model_type": Qwen3PGCAConfig.model_type,
            "architectures": ["Qwen3PGCAForCausalLM"],
            "auto_map": AUTO_MAP,
            "use_pgca": True,
            "pgca_layers": pgca_layers,
            "pgca_num_attention_heads": num_attention_heads,
            "pgca_num_key_value_heads": num_key_value_heads,
            "pgca_head_dim": head_dim,
            "pgca_dropout": float(data.get("attention_dropout", 0.0)),
            "pgca_feature_slots": int(embedding["num_feature_slots"]),
            "pgca_feature_hidden_size": hidden_size,
            "pgca_feature_embedding_dim": int(embedding["d_feat"]),
            "pgca_max_pinyin_per_char": int(embedding["max_pinyin_per_char"]),
            "pgca_feature_vocab_sizes": {
                str(key): int(value) for key, value in embedding["feature_vocab_sizes"].items()
            },
            "pgca_use_glyph_image": bool(embedding.get("use_glyph_image", False)),
            "pgca_feature_schema_version": embedding.get("feature_schema_version", 1),
        }
    )
    write_json(config_path, data)
    return data


def initialize_pgca_parameters(model: Qwen3PGCAForCausalLM) -> None:
    std = float(model.config.initializer_range)

    with torch.no_grad():
        if model.auxiliary_heads is not None:
            model.auxiliary_trained_tokens.zero_()
        for layer_idx in model.config.pgca_layers:
            attention = model.model.layers[layer_idx].pgca_attn
            for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj):
                torch.nn.init.normal_(projection.weight, mean=0.0, std=std)
                if projection.bias is not None:
                    torch.nn.init.zeros_(projection.bias)
            attention.q_norm.weight.fill_(1.0)
            attention.k_norm.weight.fill_(1.0)
            if model.config.pgca_zero_init_output:
                attention.o_proj.weight.zero_()


def check_pgca_parameters(model: Qwen3PGCAForCausalLM) -> dict:
    pgca_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".pgca_attn." in name or "feature_memory_builder." in name or "auxiliary_heads." in name
    }
    meta_parameters = [name for name, parameter in pgca_parameters.items() if parameter.is_meta]
    non_finite_parameters = [
        name
        for name, parameter in pgca_parameters.items()
        if not parameter.is_meta and not bool(torch.isfinite(parameter).all().item())
    ]
    all_zero_weights = [
        name
        for name, parameter in pgca_parameters.items()
        if parameter.ndim > 1 and not parameter.is_meta and not bool(torch.count_nonzero(parameter).item())
        and not (model.config.pgca_zero_init_output and name.endswith(".pgca_attn.o_proj.weight"))
    ]
    invalid_norm_parameters = [
        name
        for name, parameter in pgca_parameters.items()
        if (name.endswith("q_norm.weight") or name.endswith("k_norm.weight"))
        and not parameter.is_meta
        and not bool(torch.all(parameter == 1).item())
    ]
    nonzero_biases = [
        name
        for name, parameter in pgca_parameters.items()
        if name.endswith(".bias")
        and not parameter.is_meta
        and bool(torch.count_nonzero(parameter).item())
    ]
    gate_parameters = [
        name
        for name in pgca_parameters
        if name.endswith("pgca_attn.gate")
    ]

    return {
        "parameter_count": sum(parameter.numel() for parameter in pgca_parameters.values()),
        "tensor_count": len(pgca_parameters),
        "meta_parameters": meta_parameters,
        "non_finite_parameters": non_finite_parameters,
        "all_zero_weights": all_zero_weights,
        "invalid_norm_parameters": invalid_norm_parameters,
        "nonzero_biases": nonzero_biases,
        "gate_parameters": gate_parameters,
        "gate_free": not gate_parameters,
        "zero_initialized_pgca_output": model.config.pgca_zero_init_output,
        "passed": (
            not meta_parameters
            and not non_finite_parameters
            and not all_zero_weights
            and not invalid_norm_parameters
            and not nonzero_biases
            and not gate_parameters
        ),
    }


def check_feature_index(model: Qwen3PGCAForCausalLM) -> dict:
    builder = model.model.feature_memory_builder
    if builder is None:
        return {"passed": False, "reason": "feature_memory_builder is missing"}

    expected_keys = {
        f"model.feature_memory_builder.{key}"
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
    state_keys = set(model.state_dict())
    missing_state_keys = sorted(expected_keys - state_keys)
    return {
        "passed": (
            not missing_state_keys
            and builder.is_hanzi.shape[0] == model.config.vocab_size
            and bool(builder.is_hanzi.any().item())
            and bool(builder.feature_index_ready.item())
        ),
        "rows": int(builder.is_hanzi.shape[0]),
        "hanzi_rows": int(builder.is_hanzi.sum().item()),
        "missing_state_keys": missing_state_keys,
        "ready": bool(builder.feature_index_ready.item()),
    }


def _copy_remote_file(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    for old, new in REMOTE_IMPORT_REWRITES.get(destination.name, {}).items():
        text = text.replace(old, new)
    destination.write_text(text, encoding="utf-8")


def copy_runtime_code(output_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    for filename in ("configuration_qwen3_pgca.py", "modeling_qwen3_pgca.py"):
        _copy_remote_file(source_root / filename, output_path / filename)
    for destination, source in RUNTIME_FILES.items():
        _copy_remote_file(source_root / source, output_path / destination)

    init_path = output_path / "__init__.py"
    init_path.write_text(
        "from .configuration_qwen3_pgca import Qwen3PGCAConfig\n"
        "from .modeling_qwen3_pgca import Qwen3PGCAForCausalLM, Qwen3PGCAModel\n\n"
        "__all__ = [\"Qwen3PGCAConfig\", \"Qwen3PGCAModel\", \"Qwen3PGCAForCausalLM\"]\n",
        encoding="utf-8",
    )


def validate_runtime_code(output_path: Path) -> dict:
    relative_paths = [
        "__init__.py",
        "configuration_qwen3_pgca.py",
        "modeling_qwen3_pgca.py",
        *RUNTIME_FILES,
    ]
    errors: list[dict[str, str]] = []
    for relative_path in relative_paths:
        path = output_path / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "src" or alias.name.startswith("src."):
                        errors.append({"file": relative_path, "import": alias.name})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and (module == "src" or module.startswith("src.")):
                    errors.append({"file": relative_path, "import": module})
                if node.level > 0 and module:
                    target = path.parent
                    for _ in range(node.level - 1):
                        target = target.parent
                    target = target.joinpath(*module.split("."))
                    if not target.with_suffix(".py").is_file() and not (target / "__init__.py").is_file():
                        errors.append({"file": relative_path, "import": "." * node.level + module})

    if errors:
        raise RuntimeError(f"remote code contains non-portable imports: {errors}")
    return {"files": relative_paths, "portable": True}

def remove_legacy_runtime_artifacts(output_path: Path) -> None:
    for filename in ("configuration_qwen3.py", "modeling_qwen3.py", "embedding_config.json"):
        (output_path / filename).unlink(missing_ok=True)
    for directory in ("embedding", "features", "pgca"):
        shutil.rmtree(output_path / directory, ignore_errors=True)


def save_pretrained_without_auto_copy(model: Qwen3PGCAForCausalLM, output_path: Path) -> None:
    """Save weights while leaving remote-code copying to copy_runtime_code()."""
    model_class = type(model)
    config_class = type(model.config)
    model_auto_class = getattr(model_class, "_auto_class", None)
    config_auto_class = getattr(config_class, "_auto_class", None)
    model_class._auto_class = None
    config_class._auto_class = None
    try:
        model.save_pretrained(output_path, safe_serialization=True)
    finally:
        model_class._auto_class = model_auto_class
        config_class._auto_class = config_auto_class


def build_token_type_index(model_path: Path, builder) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    specials = set(tokenizer.all_special_ids)
    types = torch.zeros(builder.is_hanzi.shape[0], dtype=torch.long)
    for token_id in range(len(types)):
        value = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if token_id in specials:
            types[token_id] = 5
        elif bool(builder.is_hanzi[token_id]):
            types[token_id] = 1
        elif value and value.isspace():
            types[token_id] = 6
        elif value.strip().isdecimal():
            types[token_id] = 3
        elif value.strip() and all("LATIN" in unicodedata.name(char, "") for char in value.strip()):
            types[token_id] = 2
        elif value.strip() and all(unicodedata.category(char).startswith("P") for char in value.strip()):
            types[token_id] = 4
    builder.token_type_ids.copy_(types)


def build_pgca_model(model_path: Path, output_path: Path, pgca_layers_arg: str | None,
                     *, context_encoder=False, auxiliary_heads=False, zero_init_output=False,
                     aux_phonetic_weight=DEFAULT_AUX_PHONETIC_WEIGHT,
                     aux_glyph_weight=DEFAULT_AUX_GLYPH_WEIGHT) -> dict:
    validate_auxiliary_weights(aux_phonetic_weight, aux_glyph_weight)
    model_path = model_path.resolve()
    output_path = output_path.resolve()
    if model_path == output_path:
        raise ValueError("model_path and output_path must be different")
    if model_path in output_path.parents:
        raise ValueError("output_path must not be inside model_path")
    original_config = read_json(model_path / "config.json")
    if original_config.get("use_pgca"):
        raise ValueError("rebuild from Char weights, not an already-trained PGCA checkpoint with old feature IDs")
    embedding = read_json(model_path / "embedding_config.json")
    expected = EmbeddingFeatureConfig.from_artifacts(
        model_path, d_model=int(embedding["d_model"]), d_feat=int(embedding["d_feat"]),
        initializer_range=float(embedding.get("initializer_range", 0.02)),
    ).to_dict()
    if {key: embedding.get(key) for key in expected} != expected:
        raise ValueError("stale embedding_config.json; regenerate from the current feature artifacts")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_path, output_path, dirs_exist_ok=True)

    config_path = output_path / "config.json"
    embedding_config_path = output_path / "embedding_config.json"
    feature_index_path = output_path / "features" / "feature_index.pt"
    if not embedding_config_path.exists() or not feature_index_path.exists():
        raise FileNotFoundError("embedding_config.json and features/feature_index.pt are required for migration")

    original_config = read_json(config_path)
    pgca_layers = parse_layers(pgca_layers_arg, int(original_config["num_hidden_layers"]))
    updated_config = update_config(
        config_path,
        embedding_config_path,
        pgca_layers,
    )
    updated_config.update(pgca_context_enabled=context_encoder, pgca_aux_enabled=auxiliary_heads,
                          pgca_zero_init_output=zero_init_output,
                          pgca_aux_phonetic_weight=aux_phonetic_weight,
                          pgca_aux_glyph_weight=aux_glyph_weight)
    if (context_encoder or auxiliary_heads) and updated_config["pgca_feature_schema_version"] < 2:
        raise ValueError("context/auxiliary experiments require repaired feature schema v2")
    if auxiliary_heads:
        from src.vocab.feature_index import load_feature_vocabs
        vocabs = load_feature_vocabs(output_path / "features")
        updated_config["pgca_auxiliary_class_names"] = {
            name: [value for value, index in sorted(vocabs[name].items(), key=lambda item: item[1]) if index > 2]
            for name in ("pinyin", "stroke_count", "radical_stroke", "structure")
        }
    write_json(config_path, updated_config)

    config = Qwen3PGCAConfig.from_pretrained(output_path)
    model, loading_info = Qwen3PGCAForCausalLM.from_pretrained(
        model_path,
        config=config,
        output_loading_info=True,
        torch_dtype="auto",
        validate_pgca_feature_index=False,
    )

    feature_index = validate_feature_index(
        load_feature_index(feature_index_path),
        vocab_size=config.vocab_size,
        max_pinyin_per_char=config.pgca_max_pinyin_per_char,
    )
    builder = model.model.feature_memory_builder
    if builder is None:
        raise RuntimeError("PGCA model did not construct feature_memory_builder")
    builder.set_feature_index(feature_index)
    if context_encoder:
        build_token_type_index(model_path, builder)

    initialize_pgca_parameters(model)
    parameter_checks = check_pgca_parameters(model)
    feature_checks = check_feature_index(model)
    if not parameter_checks["passed"]:
        raise RuntimeError(f"PGCA parameter initialization failed: {parameter_checks}")
    if not feature_checks["passed"]:
        raise RuntimeError(f"Feature index embedding failed: {feature_checks}")

    save_pretrained_without_auto_copy(model, output_path)
    copy_runtime_code(output_path)
    runtime_code_checks = validate_runtime_code(output_path)
    remove_legacy_runtime_artifacts(output_path)

    saved_config = read_json(output_path / "config.json")
    if saved_config.get("model_type") != Qwen3PGCAConfig.model_type:
        raise RuntimeError("saved config has an incorrect model_type")
    if saved_config.get("auto_map") != AUTO_MAP:
        raise RuntimeError("saved config has an incorrect auto_map")

    report = {
        "base_model_path": str(model_path),
        "output_model_path": str(output_path),
        "pgca_layers": pgca_layers,
        "vocab_size": int(updated_config["vocab_size"]),
        "hidden_size": int(updated_config["hidden_size"]),
        "model_type": saved_config["model_type"],
        "auto_map": saved_config["auto_map"],
        "feature_index": feature_checks,
        "runtime_code": runtime_code_checks,
        "missing_keys": loading_info.get("missing_keys", []),
        "unexpected_keys": loading_info.get("unexpected_keys", []),
        "mismatched_keys": loading_info.get("mismatched_keys", []),
        "pgca_initialization": parameter_checks,
        "context_parameter_count": sum(p.numel() for p in builder.context_encoder.parameters()) if context_encoder else 0,
        "auxiliary_parameter_count": sum(p.numel() for p in model.auxiliary_heads.parameters()) if auxiliary_heads else 0,
        "pgca_aux_phonetic_weight": saved_config["pgca_aux_phonetic_weight"],
        "pgca_aux_glyph_weight": saved_config["pgca_aux_glyph_weight"],
        "pgca_aux_warmup_tokens": saved_config["pgca_aux_warmup_tokens"],
        "token_type_schema": ["other", "hanzi", "latin", "digit", "punctuation", "special", "whitespace"] if context_encoder else None,
    }

    report_path = output_path / "pgca_migration_report.json"
    write_json(report_path, to_jsonable(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/char"))
    parser.add_argument("--output-path", type=Path, default=Path("models/pgca"))
    parser.add_argument("--pgca-layers", type=str, default=None)
    parser.add_argument("--context-encoder", action="store_true")
    parser.add_argument("--auxiliary-heads", action="store_true")
    parser.add_argument("--zero-init-output", action="store_true")
    parser.add_argument("--aux-phonetic-weight", type=float, default=DEFAULT_AUX_PHONETIC_WEIGHT)
    parser.add_argument("--aux-glyph-weight", type=float, default=DEFAULT_AUX_GLYPH_WEIGHT)
    args = parser.parse_args()

    report = build_pgca_model(args.model_path, args.output_path, args.pgca_layers,
                             context_encoder=args.context_encoder, auxiliary_heads=args.auxiliary_heads,
                             zero_init_output=args.zero_init_output,
                             aux_phonetic_weight=args.aux_phonetic_weight,
                             aux_glyph_weight=args.aux_glyph_weight)
    print(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
