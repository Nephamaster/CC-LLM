"""Command facade for the character-level Qwen3 vocabulary pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .feature_index import FeatureIndexBuildConfig, FeatureIndexBuilder
from .feature_vocab import FeatureVocabBuildConfig, FeatureVocabBuilder
from .hanzi_set import HanziSetBuildConfig, HanziSetBuilder
from .migrate_embeddings import EmbeddingMigrationConfig, EmbeddingMigrator
from .semantic_vocab import SemanticVocabBuildConfig, SemanticVocabBuilder
from .validate_vocab import VocabValidationConfig, VocabValidator


DEFAULT_BASE_MODEL = "/share/project/wuhaiming/data/models/Qwen3-1.7B-Base/"
DEFAULT_CHAR_MODEL = Path("models/Qwen3-1.7B-Base-Char")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--char-model-path", type=Path, default=DEFAULT_CHAR_MODEL)
    parser.add_argument("--hanzi-dir", type=Path, default=Path("resources/hanzi"))
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_CHAR_MODEL / "features")
    parser.add_argument("--unihan-path", type=Path, default=Path("resources/unihan/Unihan.zip"))
    parser.add_argument("--fallback-unihan-dir", type=Path, default=Path("resources/raw/Unihan"))


def build_hanzi_set(args) -> dict:
    result = HanziSetBuilder(
        HanziSetBuildConfig(
            output_dir=args.hanzi_dir,
            tghz2013_path=args.hanzi_dir / "tghz2013.txt",
            common_traditional_path=args.hanzi_dir / "common_traditional.txt",
            rare_high_freq_path=args.hanzi_dir / "rare_high_freq.txt",
            strict=getattr(args, "strict", False),
        )
    ).build_and_write()
    return result.meta


def build_semantic_vocab(args) -> dict:
    result = SemanticVocabBuilder(
        SemanticVocabBuildConfig(
            base_tokenizer_path=args.base_model_path,
            output_dir=args.char_model_path,
            hanzi_set_path=args.hanzi_dir / "hanzi_set.txt",
            use_fast=not getattr(args, "allow_slow", False),
        )
    ).build_and_write()
    return result.manifest


def build_feature_vocab(args) -> dict:
    result = FeatureVocabBuilder(
        FeatureVocabBuildConfig(
            hanzi_set_path=args.hanzi_dir / "hanzi_set.txt",
            structure_path=args.hanzi_dir / "structure.tsv",
            unihan_path=args.unihan_path,
            fallback_unihan_dir=args.fallback_unihan_dir,
            output_dir=args.features_dir,
        )
    ).build_and_write()
    return result.manifest


def build_feature_index(args) -> dict:
    result = FeatureIndexBuilder(
        FeatureIndexBuildConfig(
            tokenizer_dir=args.char_model_path,
            features_dir=args.features_dir,
            output_jsonl=args.features_dir / "char_feature_index.jsonl",
            output_pt=args.features_dir / "feature_index.pt",
            write_torch_tensor=not getattr(args, "no_torch_index", False),
        )
    ).build_and_write()
    return result.manifest


def migrate_embeddings(args) -> dict:
    return EmbeddingMigrator(
        EmbeddingMigrationConfig(
            base_model_path=args.base_model_path,
            char_model_path=args.char_model_path,
            torch_dtype=getattr(args, "torch_dtype", "auto"),
        )
    ).migrate()


def validate_vocab(args) -> dict:
    return VocabValidator(
        VocabValidationConfig(
            char_model_path=args.char_model_path,
            features_dir=args.features_dir,
            hanzi_set_path=args.hanzi_dir / "hanzi_set.txt",
            max_hanzi_checks=getattr(args, "max_hanzi_checks", None),
        )
    ).validate()


def run_all(args) -> dict:
    summary = {
        "hanzi_set": build_hanzi_set(args),
        "semantic_vocab": build_semantic_vocab(args),
        "feature_vocab": build_feature_vocab(args),
        "feature_index": build_feature_index(args),
    }
    if not getattr(args, "skip_migrate", False):
        summary["embedding_migration"] = migrate_embeddings(args)
    if not getattr(args, "skip_validate", False):
        summary["validation"] = validate_vocab(args)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    all_parser = subparsers.add_parser("all")
    add_common_args(all_parser)
    all_parser.add_argument("--strict", action="store_true")
    all_parser.add_argument("--allow-slow", action="store_true")
    all_parser.add_argument("--no-torch-index", action="store_true")
    all_parser.add_argument("--torch-dtype", default="auto")
    all_parser.add_argument("--skip-migrate", action="store_true")
    all_parser.add_argument("--skip-validate", action="store_true")
    all_parser.add_argument("--max-hanzi-checks", type=int, default=None)
    all_parser.set_defaults(func=run_all)

    hanzi_parser = subparsers.add_parser("hanzi-set")
    add_common_args(hanzi_parser)
    hanzi_parser.add_argument("--strict", action="store_true")
    hanzi_parser.set_defaults(func=build_hanzi_set)

    semantic_parser = subparsers.add_parser("semantic-vocab")
    add_common_args(semantic_parser)
    semantic_parser.add_argument("--allow-slow", action="store_true")
    semantic_parser.set_defaults(func=build_semantic_vocab)

    feature_parser = subparsers.add_parser("feature-vocab")
    add_common_args(feature_parser)
    feature_parser.set_defaults(func=build_feature_vocab)

    index_parser = subparsers.add_parser("feature-index")
    add_common_args(index_parser)
    index_parser.add_argument("--no-torch-index", action="store_true")
    index_parser.set_defaults(func=build_feature_index)

    migrate_parser = subparsers.add_parser("migrate")
    add_common_args(migrate_parser)
    migrate_parser.add_argument("--torch-dtype", default="auto")
    migrate_parser.set_defaults(func=migrate_embeddings)

    validate_parser = subparsers.add_parser("validate")
    add_common_args(validate_parser)
    validate_parser.add_argument("--max-hanzi-checks", type=int, default=None)
    validate_parser.set_defaults(func=validate_vocab)

    args = parser.parse_args()
    result = args.func(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
