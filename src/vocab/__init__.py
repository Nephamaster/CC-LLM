"""Vocabulary construction helpers for the character-level Chinese model."""

from .bpe_state import (
    BpeState,
    decode_bpe_piece,
    encode_text_as_bpe_piece,
    extract_bpe_state,
    normalize_merges,
    normalize_vocab,
)
from .feature_index import FeatureIndexBuildConfig, FeatureIndexBuilder
from .feature_vocab import FeatureVocabBuildConfig, FeatureVocabBuilder
from .hanzi_set import HanziSetBuildConfig, HanziSetBuilder, build_hanzi_set
from .migrate_embeddings import EmbeddingMigrationConfig, EmbeddingMigrator
from .qwen3_char_tokenizer import Qwen3CharTokenizer, Qwen3CharTokenizerConfig
from .semantic_vocab import SemanticVocabBuildConfig, SemanticVocabBuilder, build_semantic_vocab
from .validate_vocab import VocabValidationConfig, VocabValidator
from .unicode_ranges import (
    CJK_BASIC,
    CJK_COMPATIBILITY,
    CJK_EXTENSION_A,
    CJK_RANGES,
    DEFAULT_VOCAB_RANGES,
    contains_hanzi,
    count_hanzi,
    is_cjk_hanzi,
    is_single_hanzi,
    iter_range_chars,
)

__all__ = [
    "BpeState",
    "CJK_BASIC",
    "CJK_COMPATIBILITY",
    "CJK_EXTENSION_A",
    "CJK_RANGES",
    "DEFAULT_VOCAB_RANGES",
    "EmbeddingMigrationConfig",
    "EmbeddingMigrator",
    "FeatureIndexBuildConfig",
    "FeatureIndexBuilder",
    "FeatureVocabBuildConfig",
    "FeatureVocabBuilder",
    "HanziSetBuildConfig",
    "HanziSetBuilder",
    "Qwen3CharTokenizer",
    "Qwen3CharTokenizerConfig",
    "SemanticVocabBuildConfig",
    "SemanticVocabBuilder",
    "VocabValidationConfig",
    "VocabValidator",
    "build_hanzi_set",
    "build_semantic_vocab",
    "contains_hanzi",
    "count_hanzi",
    "decode_bpe_piece",
    "encode_text_as_bpe_piece",
    "extract_bpe_state",
    "is_cjk_hanzi",
    "is_single_hanzi",
    "iter_range_chars",
    "normalize_merges",
    "normalize_vocab",
]