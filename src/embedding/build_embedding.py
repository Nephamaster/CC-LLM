"""Command facade for embedding migration and feature embedding setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import EmbeddingFeatureConfig
from .semantic_embedding import EmbeddingMigrationConfig, SemanticEmbeddingMigrator
from .validate_embedding import EmbeddingValidationConfig, EmbeddingValidator


DEFAULT_BASE_MODEL = "/share/project/wuhaiming/data/models/Qwen3-1.7B-Base/"
DEFAULT_CHAR_MODEL = Path("models/Qwen3-1.7B-Base-Char")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--char-model-path", type=Path, default=DEFAULT_CHAR_MODEL)


def migrate(args) -> dict:
    return SemanticEmbeddingMigrator(
        EmbeddingMigrationConfig(
            base_model_path=args.base_model_path,
            char_model_path=args.char_model_path,
            torch_dtype=getattr(args, "torch_dtype", "auto"),
        )
    ).migrate()


def build_config(args) -> dict:
    config = EmbeddingFeatureConfig.from_artifacts(
        args.char_model_path,
        d_model=args.d_model,
        d_feat=args.d_feat,
        initializer_range=args.initializer_range,
    )
    output_path = args.char_model_path / "embedding_config.json"
    config.save_pretrained(output_path)
    return {"embedding_config": str(output_path), **config.to_dict()}


def validate(args) -> dict:
    return EmbeddingValidator(
        EmbeddingValidationConfig(
            char_model_path=args.char_model_path,
            torch_dtype=getattr(args, "torch_dtype", "auto"),
            validate_model_weights=not getattr(args, "skip_model_weights", False),
        )
    ).validate()


def run_all(args) -> dict:
    summary = {}
    if not getattr(args, "skip_migrate", False):
        summary["migration"] = migrate(args)
    summary["config"] = build_config(args)
    if not getattr(args, "skip_validate", False):
        summary["validation"] = validate(args)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    all_parser = subparsers.add_parser("all")
    add_common_args(all_parser)
    all_parser.add_argument("--torch-dtype", default="auto")
    all_parser.add_argument("--d-model", type=int, default=2048)
    all_parser.add_argument("--d-feat", type=int, default=256)
    all_parser.add_argument("--initializer-range", type=float, default=0.02)
    all_parser.add_argument("--skip-migrate", action="store_true")
    all_parser.add_argument("--skip-validate", action="store_true")
    all_parser.add_argument("--skip-model-weights", action="store_true")
    all_parser.set_defaults(func=run_all)

    migrate_parser = subparsers.add_parser("migrate")
    add_common_args(migrate_parser)
    migrate_parser.add_argument("--torch-dtype", default="auto")
    migrate_parser.set_defaults(func=migrate)

    config_parser = subparsers.add_parser("config")
    add_common_args(config_parser)
    config_parser.add_argument("--d-model", type=int, default=2048)
    config_parser.add_argument("--d-feat", type=int, default=256)
    config_parser.add_argument("--initializer-range", type=float, default=0.02)
    config_parser.set_defaults(func=build_config)

    validate_parser = subparsers.add_parser("validate")
    add_common_args(validate_parser)
    validate_parser.add_argument("--torch-dtype", default="auto")
    validate_parser.add_argument("--skip-model-weights", action="store_true")
    validate_parser.set_defaults(func=validate)

    args = parser.parse_args()
    result = args.func(args)
    # print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
