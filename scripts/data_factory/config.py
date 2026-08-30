"""Configuration loading and validation for Phase 1 data construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PHASE1_TOTAL_TOKENS = 1_000_000_000
PHASE1_DEFAULT_CANDIDATE_TOKENS = 1_100_000_000
PHASE1_QUOTAS = {
    "chinese_natural": 450_000_000,
    "multi_hanzi_bridge": 150_000_000,
    "new_hanzi_coverage": 50_000_000,
    "non_chinese": 150_000_000,
    "mixed_zh_en": 100_000_000,
    "specialized": 100_000_000,
}

SPECIALIZED_QUOTAS = {
    "code": 40_000_000,
    "math_science": 40_000_000,
    "structured": 20_000_000,
}

VALID_SPECIALIZED_GROUPS = frozenset(SPECIALIZED_QUOTAS)

# Existing normalized artifacts and external manifests still use these names.
LEGACY_SOURCE_CATEGORIES = frozenset(
    {"chinese_general", "chinese_high_quality", "mixed_zh_en", "non_chinese", "supplemental"}
)
VALID_CATEGORIES = frozenset(PHASE1_QUOTAS) | LEGACY_SOURCE_CATEGORIES
VALID_MIXED_GROUPS = frozenset({"github", "huggingface", "wikimedia_openalex", "web_allowlist"})
VALID_SUPPLEMENTAL_GROUPS = frozenset({"code", "structured", "math", "hanzi", "chat"})


@dataclass(frozen=True)
class VocabAlignmentConfig:
    removed_multi_hanzi_tokens_path: Path
    new_hanzi_token_ids_path: Path
    bridge_top_token_count: int
    bridge_min_contexts: int
    priority_hanzi_min_documents: int
    priority_hanzi_coverage: float
    priority_hanzi_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        if self.bridge_top_token_count <= 0:
            raise ValueError("bridge_top_token_count must be positive")
        if self.bridge_min_contexts <= 0:
            raise ValueError("bridge_min_contexts must be positive")
        if self.priority_hanzi_min_documents <= 0:
            raise ValueError("priority_hanzi_min_documents must be positive")
        if not 0 < self.priority_hanzi_coverage <= 1:
            raise ValueError("priority_hanzi_coverage must be in (0, 1]")
        if not self.priority_hanzi_paths:
            raise ValueError("priority_hanzi_paths must not be empty")


@dataclass(frozen=True)
class WindowingConfig:
    min_tokens: int
    target_tokens: int
    max_tokens: int

    def __post_init__(self) -> None:
        if not 0 < self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("window token limits must satisfy 0 < min <= target <= max")


@dataclass(frozen=True)
class ValidationConfig:
    natural_tokens: int
    alignment_tokens: int

    def __post_init__(self) -> None:
        if self.natural_tokens <= 0 or self.alignment_tokens <= 0:
            raise ValueError("validation token targets must be positive")




@dataclass(frozen=True)
class FastPipelineConfig:
    cache_rows_per_shard: int
    parquet_compression: str
    calibration_docs_per_source: int
    calibration_batch_size: int
    candidate_rows_per_shard: int
    tokenized_rows_per_shard: int
    oversample_ratio: float
    feature_oversample_ratio: float
    source_cap_ratio: float
    tokenizer_rayon_threads: int
    exact_batch_size: int
    exact_batch_chars: int
    pack_sequence_length: int
    pack_shard_tokens: int
    keep_candidate_text: bool

    def __post_init__(self) -> None:
        positive = {
            "cache_rows_per_shard": self.cache_rows_per_shard,
            "calibration_docs_per_source": self.calibration_docs_per_source,
            "calibration_batch_size": self.calibration_batch_size,
            "candidate_rows_per_shard": self.candidate_rows_per_shard,
            "tokenized_rows_per_shard": self.tokenized_rows_per_shard,
            "tokenizer_rayon_threads": self.tokenizer_rayon_threads,
            "exact_batch_size": self.exact_batch_size,
            "exact_batch_chars": self.exact_batch_chars,
            "pack_sequence_length": self.pack_sequence_length,
            "pack_shard_tokens": self.pack_shard_tokens,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"fast_pipeline.{name} must be positive")
        if not 1 < self.oversample_ratio <= 2:
            raise ValueError("fast_pipeline.oversample_ratio must be in (1, 2]")
        if not self.oversample_ratio <= self.feature_oversample_ratio <= 3:
            raise ValueError(
                "fast_pipeline.feature_oversample_ratio must be >= oversample_ratio and <= 3"
            )
        if not 0 < self.source_cap_ratio <= 1:
            raise ValueError("fast_pipeline.source_cap_ratio must be in (0, 1]")

@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    repo_root: Path
    phase_root: Path
    tokenizer_path: Path
    seed: int
    tolerance: float
    candidate_tokens: int
    normalized_shard_records: int
    deduplicated_shard_records: int
    final_shard_tokens: int
    sample_batch_size: int
    sample_batch_chars: int
    sample_workers: int
    prepare_workers: int
    dedup_workers: int
    preselection_buffer_ratio: float
    fast_pipeline: FastPipelineConfig
    sources: dict[str, Any]
    quotas: dict[str, int]
    specialized_quotas: dict[str, int]
    vocab_alignment: VocabAlignmentConfig
    windowing: WindowingConfig
    validation: ValidationConfig
    quality: dict[str, Any]
    dedup: dict[str, Any]

    def __post_init__(self) -> None:
        if self.sample_batch_size <= 0:
            raise ValueError("sample_batch_size must be positive")
        if self.sample_batch_chars <= 0:
            raise ValueError("sample_batch_chars must be positive")
        if self.sample_workers <= 0:
            raise ValueError("sample_workers must be positive")
        if self.prepare_workers <= 0:
            raise ValueError("prepare_workers must be positive")
        if self.dedup_workers <= 0:
            raise ValueError("dedup_workers must be positive")
        target_tokens = sum(self.quotas.values())

        if target_tokens <= 0:
            raise ValueError("Phase 1 quotas must sum to a positive token target")

        required_tokens = (
            target_tokens
            + self.validation.natural_tokens
            + self.validation.alignment_tokens
        )

        if self.candidate_tokens < required_tokens:
            raise ValueError(
                "candidate_tokens must be at least "
                f"{required_tokens:,} "
                f"(train={target_tokens:,}, "
                f"validation={self.validation.natural_tokens + self.validation.alignment_tokens:,})"
            )

        if not 1 < self.preselection_buffer_ratio <= 2:
            raise ValueError("preselection_buffer_ratio must be in (1, 2]")

        if sum(self.specialized_quotas.values()) != self.quotas["specialized"]:
            raise ValueError("specialized_quotas must sum to the specialized quota")

    @property
    def raw_manifest_dir(self) -> Path:
        return self.phase_root / "raw_manifest"

    @property
    def normalized_dir(self) -> Path:
        return self.phase_root / "normalized"

    @property
    def deduplicated_dir(self) -> Path:
        return self.phase_root / "deduplicated"

    @property
    def final_dir(self) -> Path:
        return self.phase_root / "final"

    @property
    def reports_dir(self) -> Path:
        return self.phase_root / "reports"

    @property
    def fast_cache_dir(self) -> Path:
        return self.phase_root / "cache_parquet"

    @property
    def fast_candidate_dir(self) -> Path:
        return self.phase_root / "fast_candidates"

    @property
    def fast_tokenized_dir(self) -> Path:
        return self.phase_root / "fast_tokenized"

    @property
    def fast_final_dir(self) -> Path:
        return self.phase_root / "fast_final"

    @property
    def fast_calibration_path(self) -> Path:
        return self.reports_dir / "phase1_token_calibration.json"


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _integer_mapping(value: Any, default: dict[str, int], name: str) -> dict[str, int]:
    mapping = default if value is None else value
    if not isinstance(mapping, dict) or set(mapping) != set(default):
        raise ValueError(f"{name} must contain exactly: {sorted(default)}")
    result = {str(key): int(count) for key, count in mapping.items()}
    if any(count <= 0 for count in result.values()):
        raise ValueError(f"{name} values must be positive")
    return result


def _object(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def load_config(path: Path) -> PipelineConfig:
    path = path.resolve()
    with path.open("rt", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("phase1 config must be a JSON object")

    repo_root = _resolve_path(raw.get("repo_root", "."), path.parent).resolve()
    phase_root = _resolve_path(raw.get("phase_root", "data/semantic_alignment"), repo_root)
    tokenizer_path = _resolve_path(raw.get("tokenizer_path", "models/Qwen3-1.7B-Base-Char"), repo_root)
    quotas = _integer_mapping(raw.get("quotas"), PHASE1_QUOTAS, "quotas")
    specialized_quotas = _integer_mapping(
        raw.get("specialized_quotas"), SPECIALIZED_QUOTAS, "specialized_quotas"
    )

    vocab_raw = _object(raw.get("vocab_alignment"), "vocab_alignment")
    vocab_alignment = VocabAlignmentConfig(
        removed_multi_hanzi_tokens_path=_resolve_path(
            vocab_raw.get(
                "removed_multi_hanzi_tokens_path",
                "models/Qwen3-1.7B-Base-Char/removed_multi_hanzi_tokens.json",
            ),
            repo_root,
        ),
        new_hanzi_token_ids_path=_resolve_path(
            vocab_raw.get(
                "new_hanzi_token_ids_path",
                "models/Qwen3-1.7B-Base-Char/new_hanzi_token_ids.json",
            ),
            repo_root,
        ),
        bridge_top_token_count=int(vocab_raw.get("bridge_top_token_count", 5_000)),
        bridge_min_contexts=int(vocab_raw.get("bridge_min_contexts", 256)),
        priority_hanzi_min_documents=int(vocab_raw.get("priority_hanzi_min_documents", 128)),
        priority_hanzi_coverage=float(vocab_raw.get("priority_hanzi_coverage", 0.95)),
        priority_hanzi_paths=tuple(
            _resolve_path(value, repo_root)
            for value in vocab_raw.get(
                "priority_hanzi_paths",
                [
                    "resources/hanzi/tghz2013.txt",
                    "resources/hanzi/common_traditional.txt",
                    "resources/hanzi/rare_high_freq.txt",
                ],
            )
        ),
    )

    window_raw = _object(raw.get("windowing"), "windowing")
    windowing = WindowingConfig(
        min_tokens=int(window_raw.get("min_tokens", 128)),
        target_tokens=int(window_raw.get("target_tokens", 512)),
        max_tokens=int(window_raw.get("max_tokens", 1_024)),
    )

    validation_raw = _object(raw.get("validation"), "validation")
    validation = ValidationConfig(
        natural_tokens=int(validation_raw.get("natural_tokens", 2_500_000)),
        alignment_tokens=int(validation_raw.get("alignment_tokens", 1_000_000)),
    )


    fast_raw = _object(raw.get("fast_pipeline"), "fast_pipeline")
    fast_pipeline = FastPipelineConfig(
        cache_rows_per_shard=int(fast_raw.get("cache_rows_per_shard", 250_000)),
        parquet_compression=str(fast_raw.get("parquet_compression", "zstd")),
        calibration_docs_per_source=int(fast_raw.get("calibration_docs_per_source", 50_000)),
        calibration_batch_size=int(fast_raw.get("calibration_batch_size", 1024)),
        candidate_rows_per_shard=int(fast_raw.get("candidate_rows_per_shard", 100_000)),
        tokenized_rows_per_shard=int(fast_raw.get("tokenized_rows_per_shard", 25_000)),
        oversample_ratio=float(fast_raw.get("oversample_ratio", 1.20)),
        feature_oversample_ratio=float(fast_raw.get("feature_oversample_ratio", 1.50)),
        source_cap_ratio=float(fast_raw.get("source_cap_ratio", 0.40)),
        tokenizer_rayon_threads=int(fast_raw.get("tokenizer_rayon_threads", 8)),
        exact_batch_size=int(fast_raw.get("exact_batch_size", 1024)),
        exact_batch_chars=int(fast_raw.get("exact_batch_chars", 2_000_000)),
        pack_sequence_length=int(fast_raw.get("pack_sequence_length", 1024)),
        pack_shard_tokens=int(fast_raw.get("pack_shard_tokens", 100_000_000)),
        keep_candidate_text=bool(fast_raw.get("keep_candidate_text", True)),
    )

    tolerance = float(raw.get("tolerance", 0.01))
    if not 0 <= tolerance < 1:
        raise ValueError("tolerance must be in [0, 1)")
    sources = raw.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("sources must be a JSON object")

    return PipelineConfig(
        path=path,
        repo_root=repo_root,
        phase_root=phase_root,
        tokenizer_path=tokenizer_path,
        seed=int(raw.get("seed", 20260714)),
        tolerance=tolerance,
        candidate_tokens=int(
            raw.get("candidate_tokens", PHASE1_DEFAULT_CANDIDATE_TOKENS)
        ),
        normalized_shard_records=int(raw.get("normalized_shard_records", 100_000)),
        deduplicated_shard_records=int(raw.get("deduplicated_shard_records", 100_000)),
        final_shard_tokens=int(raw.get("final_shard_tokens", 100_000_000)),
        sample_batch_size=int(raw.get("sample_batch_size", 512)),
        sample_batch_chars=int(raw.get("sample_batch_chars", 1_000_000)),
        sample_workers=int(raw.get("sample_workers", raw.get("tokenizer_workers", 8))),
        prepare_workers=int(raw.get("prepare_workers", 8)),
        dedup_workers=int(raw.get("dedup_workers", 8)),
        preselection_buffer_ratio=float(raw.get("preselection_buffer_ratio", 1.25)),
        fast_pipeline=fast_pipeline,
        sources=sources,
        quotas=quotas,
        specialized_quotas=specialized_quotas,
        vocab_alignment=vocab_alignment,
        windowing=windowing,
        validation=validation,
        quality=dict(raw.get("quality", {})),
        dedup=dict(raw.get("dedup", {})),
    )
