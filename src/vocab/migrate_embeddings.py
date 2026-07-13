"""Backward-compatible wrapper for semantic embedding migration."""

from __future__ import annotations

from src.embedding.semantic_embedding import (
    EmbeddingMigrationConfig,
    SemanticEmbeddingMigrator,
    main,
    read_init_json,
    read_int_key_json,
    validate_id_coverage,
)


EmbeddingMigrator = SemanticEmbeddingMigrator

__all__ = [
    "EmbeddingMigrationConfig",
    "EmbeddingMigrator",
    "SemanticEmbeddingMigrator",
    "main",
    "read_init_json",
    "read_int_key_json",
    "validate_id_coverage",
]


if __name__ == "__main__":
    main()