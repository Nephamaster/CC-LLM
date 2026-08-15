"""Configuration loading and validation for Phase 1 data construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PHASE1_QUOTAS = {
    "chinese_general": 800_000_000,
    "chinese_high_quality": 400_000_000,
    "mixed_zh_en": 300_000_000,
    "non_chinese": 300_000_000,
    "supplemental": 200_000_000,
}

MIXED_QUOTAS = {
    "github": 135_000_000,
    "huggingface": 45_000_000,
    "wikimedia_openalex": 60_000_000,
    "web_allowlist": 60_000_000,
}

SUPPLEMENTAL_QUOTAS = {
    "code": 104_000_000,
    "structured": 50_000_000,
    "math": 36_000_000,
    "hanzi": 8_000_000,
    "chat": 2_000_000,
}

VALID_CATEGORIES = frozenset(PHASE1_QUOTAS)
VALID_MIXED_GROUPS = frozenset(MIXED_QUOTAS)
VALID_SUPPLEMENTAL_GROUPS = frozenset(SUPPLEMENTAL_QUOTAS)


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    repo_root: Path
    phase_root: Path
    tokenizer_path: Path
    seed: int
    tolerance: float
    normalized_shard_records: int
    deduplicated_shard_records: int
    final_shard_tokens: int
    sample_batch_size: int
    sample_batch_chars: int
    sample_workers: int
    sources: dict[str, Any]
    quotas: dict[str, int]
    mixed_quotas: dict[str, int]
    supplemental_quotas: dict[str, int]
    quality: dict[str, Any]
    dedup: dict[str, Any]

    def __post_init__(self) -> None:
        if self.sample_batch_size <= 0:
            raise ValueError("sample_batch_size must be positive")
        if self.sample_batch_chars <= 0:
            raise ValueError("sample_batch_chars must be positive")
        if self.sample_workers <= 0:
            raise ValueError("sample_workers must be positive")

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
    mixed_quotas = _integer_mapping(raw.get("mixed_quotas"), MIXED_QUOTAS, "mixed_quotas")
    supplemental_quotas = _integer_mapping(
        raw.get("supplemental_quotas"), SUPPLEMENTAL_QUOTAS, "supplemental_quotas"
    )
    if sum(mixed_quotas.values()) != quotas["mixed_zh_en"]:
        raise ValueError("mixed_quotas must sum to the mixed_zh_en quota")
    if sum(supplemental_quotas.values()) != quotas["supplemental"]:
        raise ValueError("supplemental_quotas must sum to the supplemental quota")

    tolerance = float(raw.get("tolerance", 0.05))
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
        normalized_shard_records=int(raw.get("normalized_shard_records", 100_000)),
        deduplicated_shard_records=int(raw.get("deduplicated_shard_records", 100_000)),
        final_shard_tokens=int(raw.get("final_shard_tokens", 100_000_000)),
        sample_batch_size=int(raw.get("sample_batch_size", 512)),
        sample_batch_chars=int(raw.get("sample_batch_chars", 1_000_000)),
        sample_workers=int(raw.get("sample_workers", raw.get("tokenizer_workers", 8))),
        sources=sources,
        quotas=quotas,
        mixed_quotas=mixed_quotas,
        supplemental_quotas=supplemental_quotas,
        quality=dict(raw.get("quality", {})),
        dedup=dict(raw.get("dedup", {})),
    )

