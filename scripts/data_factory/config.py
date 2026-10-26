"""Configuration schema and run identity for Data Factory V2."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_VERSION = 2
PHASE_TARGETS = {"phase1": 1_000_000_000, "phase2": 10_000_000_000}
VALID_PROFILES = frozenset({"default", "exact_only"})


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _fraction(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if (result < 0.0 if allow_zero else result <= 0.0) or result > 1.0:
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return result


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str, base: Path) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser()
    return expanded if expanded.is_absolute() else (base / expanded).resolve()


@dataclass(frozen=True)
class SourceSpec:
    name: str
    reader: str
    adapter: str
    paths: tuple[str, ...]
    phases: frozenset[str]
    license: str
    license_mode: str
    quality_profile: str
    default_domain: str
    metadata: dict[str, Any]
    homepage: str | None = None

    @classmethod
    def from_raw(cls, name: str, raw: dict[str, Any]) -> "SourceSpec":
        paths = raw.get("paths")
        if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
            raise ValueError(f"sources.{name}.paths must be a non-empty string list")
        phases = frozenset(str(value) for value in raw.get("phases", []))
        if not phases or not phases <= PHASE_TARGETS.keys():
            raise ValueError(f"sources.{name}.phases contains unsupported phases")
        license_mode = str(raw.get("license_mode", "fixed"))
        if license_mode not in {"fixed", "per_record"}:
            raise ValueError(f"sources.{name}.license_mode must be fixed or per_record")
        license_value = str(raw.get("license", "")).strip()
        if license_mode == "fixed" and not license_value:
            raise ValueError(f"sources.{name}.license is required for fixed licensing")
        reader = str(raw.get("reader", "")).strip()
        if not reader:
            raise ValueError(f"sources.{name}.reader is required")
        adapter = str(raw.get("adapter", "")).strip()
        if not adapter:
            raise ValueError(f"sources.{name}.adapter is required")
        return cls(
            name=name,
            reader=reader,
            adapter=adapter,
            paths=tuple(paths),
            phases=phases,
            license=license_value,
            license_mode=license_mode,
            quality_profile=str(raw.get("quality_profile", "default")),
            default_domain=str(raw.get("default_domain", "general")),
            metadata=dict(_mapping(raw.get("metadata", {}), f"sources.{name}.metadata")),
            homepage=None if raw.get("homepage") is None else str(raw["homepage"]),
        )


@dataclass(frozen=True)
class SourceRegistry:
    path: Path
    sha256: str
    sources: dict[str, SourceSpec]


@dataclass(frozen=True)
class BucketSpec:
    name: str
    fraction: float
    source_weights: dict[str, float]
    selector: str
    sub_buckets: dict[str, float]

    @classmethod
    def from_raw(cls, index: int, raw: dict[str, Any]) -> "BucketSpec":
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError(f"buckets[{index}].name is required")
        if "bridge" in name.lower() or "桥接" in name:
            raise ValueError("multi-token bridge buckets are not part of Data Factory V2")
        weights_raw = _mapping(raw.get("source_weights"), f"buckets.{name}.source_weights")
        weights = {
            str(source): _fraction(weight, f"buckets.{name}.source_weights.{source}")
            for source, weight in weights_raw.items()
        }
        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError(f"buckets.{name}.source_weights must sum to 1")
        sub_raw = _mapping(raw.get("sub_buckets", {}), f"buckets.{name}.sub_buckets")
        sub_buckets = {
            str(key): _fraction(value, f"buckets.{name}.sub_buckets.{key}")
            for key, value in sub_raw.items()
        }
        if sub_buckets and abs(sum(sub_buckets.values()) - 1.0) > 1e-9:
            raise ValueError(f"buckets.{name}.sub_buckets must sum to 1")
        return cls(
            name=name,
            fraction=_fraction(raw.get("fraction"), f"buckets.{name}.fraction"),
            source_weights=weights,
            selector=str(raw.get("selector", name)),
            sub_buckets=sub_buckets,
        )


@dataclass(frozen=True)
class DedupSpec:
    exact_algorithm: str
    minhash_enabled: bool
    minhash_profiles: dict[str, dict[str, Any]]
    decontamination_registry: Path


@dataclass(frozen=True)
class CalibrationSpec:
    documents_per_source: int
    max_files_per_source: int
    scan_multiplier: int
    batch_size: int


@dataclass(frozen=True)
class EnhancementSpec:
    bucket: str
    token_ids_path: Path
    natural_only: bool
    synthetic_enabled: bool
    synthetic_max_fraction: float
    feature_score_weight: float
    coverage_targets: dict[int, float]
    constraints: dict[str, float]


@dataclass(frozen=True)
class SequenceSpec:
    lengths: dict[int, float]
    insert_eos: bool
    preserve_long_document_order: bool


@dataclass(frozen=True)
class DependencyHashes:
    phase_config: str
    source_registry: str
    tokenizer: str | None
    new_char_tokens: str | None
    source_manifest: str | None


@dataclass(frozen=True)
class DataFactoryConfig:
    path: Path
    version: int
    phase: str
    profile: str
    repo_root: Path
    corpus_root: Path
    tokenizer_path: Path
    seed: int
    target_tokens: int
    validation_tokens: int
    calibration: CalibrationSpec
    candidate_oversample_ratio: float
    enhancement_oversample_ratio: float
    candidate_priority: tuple[str, ...]
    buckets: tuple[BucketSpec, ...]
    attributes: dict[str, dict[str, float]]
    dedup: DedupSpec
    enhancement: EnhancementSpec
    sequence: SequenceSpec
    source_registry: SourceRegistry
    hashes: DependencyHashes
    run_id: str

    @property
    def bucket_tokens(self) -> dict[str, int]:
        return {
            bucket.name: int(round(self.target_tokens * bucket.fraction))
            for bucket in self.buckets
        }

    @property
    def run_root(self) -> Path:
        return self.corpus_root / "runs" / self.phase / self.run_id

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "phase": self.phase,
            "profile": self.profile,
            "run_id": self.run_id,
            "repo_root": str(self.repo_root),
            "corpus_root": str(self.corpus_root),
            "tokenizer_path": str(self.tokenizer_path),
            "seed": self.seed,
            "target_tokens": self.target_tokens,
            "validation_tokens": self.validation_tokens,
            "calibration": {
                "documents_per_source": self.calibration.documents_per_source,
                "max_files_per_source": self.calibration.max_files_per_source,
                "scan_multiplier": self.calibration.scan_multiplier,
                "batch_size": self.calibration.batch_size,
            },
            "candidate_oversample_ratio": self.candidate_oversample_ratio,
            "enhancement_oversample_ratio": self.enhancement_oversample_ratio,
            "candidate_priority": self.candidate_priority,
            "buckets": [
                {
                    "name": bucket.name,
                    "fraction": bucket.fraction,
                    "target_tokens": self.bucket_tokens[bucket.name],
                    "source_weights": bucket.source_weights,
                    "selector": bucket.selector,
                    "sub_buckets": bucket.sub_buckets,
                }
                for bucket in self.buckets
            ],
            "attributes": self.attributes,
            "dedup": {
                "exact_algorithm": self.dedup.exact_algorithm,
                "minhash_enabled": self.dedup.minhash_enabled,
                "minhash_profiles": self.dedup.minhash_profiles,
                "decontamination_registry": str(self.dedup.decontamination_registry),
            },
            "enhancement": {
                "bucket": self.enhancement.bucket,
                "token_ids_path": str(self.enhancement.token_ids_path),
                "natural_only": self.enhancement.natural_only,
                "synthetic_enabled": self.enhancement.synthetic_enabled,
                "synthetic_max_fraction": self.enhancement.synthetic_max_fraction,
                "feature_score_weight": self.enhancement.feature_score_weight,
                "coverage_targets": self.enhancement.coverage_targets,
                "constraints": self.enhancement.constraints,
            },
            "sequence": {
                "lengths": self.sequence.lengths,
                "insert_eos": self.sequence.insert_eos,
                "preserve_long_document_order": self.sequence.preserve_long_document_order,
            },
            "source_registry": str(self.source_registry.path),
            "hashes": {
                "phase_config": self.hashes.phase_config,
                "source_registry": self.hashes.source_registry,
                "tokenizer": self.hashes.tokenizer,
                "new_char_tokens": self.hashes.new_char_tokens,
                "source_manifest": self.hashes.source_manifest,
            },
        }

    def write_snapshot(self, path: Path) -> None:
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wt", encoding="utf-8") as stream:
            yaml.safe_dump(
                self.to_snapshot(),
                stream,
                allow_unicode=True,
                sort_keys=False,
            )


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("rt", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return _mapping(raw, str(path))


def _load_registry(path: Path) -> SourceRegistry:
    raw = _load_yaml(path)
    if int(raw.get("version", 0)) != CONFIG_VERSION:
        raise ValueError(f"unsupported source registry version: {raw.get('version')}")
    sources_raw = _mapping(raw.get("sources"), "sources")
    sources = {
        str(name): SourceSpec.from_raw(str(name), _mapping(value, f"sources.{name}"))
        for name, value in sources_raw.items()
    }
    if not sources:
        raise ValueError("source registry must not be empty")
    return SourceRegistry(path=path, sha256=_canonical_hash(raw), sources=sources)


def _load_dedup(raw: dict[str, Any], repo_root: Path, profile: str) -> DedupSpec:
    exact = _mapping(raw.get("exact"), "dedup.exact")
    algorithm = str(exact.get("algorithm", "")).lower()
    if algorithm != "sha256":
        raise ValueError("dedup.exact.algorithm must be sha256")
    minhash = _mapping(raw.get("minhash"), "dedup.minhash")
    enabled = bool(minhash.get("enabled", True)) and profile != "exact_only"
    profiles = _mapping(minhash.get("profiles"), "dedup.minhash.profiles")
    decontamination = _mapping(raw.get("decontamination"), "dedup.decontamination")
    registry = _resolve_path(str(decontamination.get("registry")), repo_root)
    return DedupSpec(
        exact_algorithm=algorithm,
        minhash_enabled=enabled,
        minhash_profiles={str(key): dict(_mapping(value, str(key))) for key, value in profiles.items()},
        decontamination_registry=registry,
    )


def _load_enhancement(
    raw: dict[str, Any], bucket_names: set[str], repo_root: Path
) -> EnhancementSpec:
    bucket = str(raw.get("bucket", ""))
    if bucket not in bucket_names:
        raise ValueError("new_char_enhancement.bucket must reference an existing bucket")
    synthetic = _mapping(raw.get("synthetic", {}), "new_char_enhancement.synthetic")
    enabled = bool(synthetic.get("enabled", False))
    max_fraction = _fraction(
        synthetic.get("max_fraction", 0.0),
        "new_char_enhancement.synthetic.max_fraction",
        allow_zero=True,
    )
    if enabled and max_fraction > 0.01:
        raise ValueError("synthetic enhancement data must not exceed 1%")
    natural_only = bool(raw.get("natural_only", True))
    if natural_only and enabled:
        raise ValueError("natural_only and synthetic.enabled cannot both be true")
    coverage_raw = _mapping(raw.get("coverage_targets"), "new_char_enhancement.coverage_targets")
    coverage_targets = {
        _positive_int(documents, "coverage target documents"): _fraction(
            fraction,
            f"new_char_enhancement.coverage_targets.{documents}",
        )
        for documents, fraction in coverage_raw.items()
    }
    constraints_raw = _mapping(raw.get("constraints", {}), "new_char_enhancement.constraints")
    constraints = {
        str(name): _fraction(value, f"new_char_enhancement.constraints.{name}", allow_zero=True)
        for name, value in constraints_raw.items()
    }
    return EnhancementSpec(
        bucket=bucket,
        token_ids_path=_resolve_path(str(raw.get("token_ids_path")), repo_root),
        natural_only=natural_only,
        synthetic_enabled=enabled,
        synthetic_max_fraction=max_fraction,
        feature_score_weight=float(raw.get("feature_score_weight", 0.25)),
        coverage_targets=coverage_targets,
        constraints=constraints,
    )


def _load_sequence(raw: dict[str, Any]) -> SequenceSpec:
    lengths_raw = _mapping(raw.get("lengths"), "sequence.lengths")
    lengths = {
        _positive_int(length, "sequence length"): _fraction(weight, f"sequence.lengths.{length}")
        for length, weight in lengths_raw.items()
    }
    if abs(sum(lengths.values()) - 1.0) > 1e-9:
        raise ValueError("sequence.lengths weights must sum to 1")
    return SequenceSpec(
        lengths=lengths,
        insert_eos=bool(raw.get("insert_eos", True)),
        preserve_long_document_order=bool(raw.get("preserve_long_document_order", True)),
    )


def load_data_factory_config(
    path: Path,
    *,
    profile: str = "default",
    source_manifest_sha256: str | None = None,
    require_tokenizer: bool = False,
) -> DataFactoryConfig:
    """Load and validate one immutable Phase configuration."""
    if profile not in VALID_PROFILES:
        raise ValueError(f"profile must be one of {sorted(VALID_PROFILES)}")
    path = path.resolve()
    raw = _load_yaml(path)
    if int(raw.get("version", 0)) != CONFIG_VERSION:
        raise ValueError(f"unsupported config version: {raw.get('version')}")
    phase = str(raw.get("phase", ""))
    if phase not in PHASE_TARGETS:
        raise ValueError(f"phase must be one of {sorted(PHASE_TARGETS)}")
    target_tokens = _positive_int(raw.get("target_tokens"), "target_tokens")
    if target_tokens != PHASE_TARGETS[phase]:
        raise ValueError(f"{phase} target_tokens must be {PHASE_TARGETS[phase]:,}")

    repo_root = _resolve_path(str(raw.get("repo_root", ".")), path.parent)
    corpus_root = _resolve_path(str(raw.get("corpus_root", "data/corpus")), repo_root)
    tokenizer_path = _resolve_path(str(raw.get("tokenizer_path")), repo_root)
    tokenizer_hash = _file_hash(tokenizer_path / "tokenizer.json")
    if require_tokenizer and tokenizer_hash is None:
        raise FileNotFoundError(f"tokenizer.json is missing under {tokenizer_path}")

    registry_path = _resolve_path(str(raw.get("source_registry")), path.parent)
    registry = _load_registry(registry_path)
    buckets_raw = raw.get("buckets")
    if not isinstance(buckets_raw, list) or not buckets_raw:
        raise ValueError("buckets must be a non-empty list")
    buckets = tuple(
        BucketSpec.from_raw(index, _mapping(value, f"buckets[{index}]"))
        for index, value in enumerate(buckets_raw)
    )
    if len({bucket.name for bucket in buckets}) != len(buckets):
        raise ValueError("bucket names must be unique")
    if abs(sum(bucket.fraction for bucket in buckets) - 1.0) > 1e-9:
        raise ValueError("bucket fractions must sum to 1")
    for bucket in buckets:
        for source_name in bucket.source_weights:
            source = registry.sources.get(source_name)
            if source is None:
                raise ValueError(f"bucket {bucket.name} references unknown source {source_name}")
            if phase not in source.phases:
                raise ValueError(f"source {source_name} does not allow {phase}")

    candidate = _mapping(raw.get("candidate"), "candidate")
    oversample = float(candidate.get("oversample_ratio", 0.0))
    enhancement_oversample = float(candidate.get("enhancement_oversample_ratio", 0.0))
    if not 1.0 < oversample <= 2.0:
        raise ValueError("candidate.oversample_ratio must be in (1, 2]")
    if not oversample <= enhancement_oversample <= 3.0:
        raise ValueError("enhancement oversample ratio must be >= default and <= 3")
    priority = tuple(str(value) for value in candidate.get("priority", []))
    bucket_names = {bucket.name for bucket in buckets}
    if len(priority) != len(bucket_names) or set(priority) != bucket_names:
        raise ValueError("candidate.priority must contain every bucket exactly once")

    calibration_raw = _mapping(raw.get("calibration"), "calibration")
    calibration = CalibrationSpec(
        documents_per_source=_positive_int(
            calibration_raw.get("documents_per_source"),
            "calibration.documents_per_source",
        ),
        max_files_per_source=_positive_int(
            calibration_raw.get("max_files_per_source"),
            "calibration.max_files_per_source",
        ),
        scan_multiplier=_positive_int(
            calibration_raw.get("scan_multiplier"),
            "calibration.scan_multiplier",
        ),
        batch_size=_positive_int(
            calibration_raw.get("batch_size"),
            "calibration.batch_size",
        ),
    )

    attributes_raw = _mapping(raw.get("attributes", {}), "attributes")
    attributes: dict[str, dict[str, float]] = {}
    for name, spec in attributes_raw.items():
        values = {
            str(key): _fraction(
                value,
                f"attributes.{name}.{key}",
                allow_zero=True,
            )
            for key, value in _mapping(spec, f"attributes.{name}").items()
        }
        minimum = values.get("min_fraction", 0.0)
        maximum = values.get("max_fraction", 1.0)
        if minimum > maximum:
            raise ValueError(f"attributes.{name} min_fraction exceeds max_fraction")
        attributes[str(name)] = values
    dedup = _load_dedup(_mapping(raw.get("dedup"), "dedup"), repo_root, profile)
    enhancement = _load_enhancement(
        _mapping(raw.get("new_char_enhancement"), "new_char_enhancement"),
        bucket_names,
        repo_root,
    )
    sequence = _load_sequence(_mapping(raw.get("sequence"), "sequence"))

    enhancement_weights = next(b.source_weights for b in buckets if b.name == enhancement.bucket)
    source_cap = enhancement.constraints.get("single_source_max_fraction", 1.0)
    if max(enhancement_weights.values()) > source_cap + 1e-9:
        raise ValueError("enhancement source_weights exceed single_source_max_fraction")
    phase_hash = _canonical_hash({"config": raw, "profile": profile, "pipeline_revision": 3})
    new_char_tokens_hash = _file_hash(enhancement.token_ids_path)
    if require_tokenizer and new_char_tokens_hash is None:
        raise FileNotFoundError(
            f"new Hanzi token map is missing: {enhancement.token_ids_path}"
        )
    hashes = DependencyHashes(
        phase_config=phase_hash,
        source_registry=registry.sha256,
        tokenizer=tokenizer_hash,
        new_char_tokens=new_char_tokens_hash,
        source_manifest=source_manifest_sha256,
    )
    identity = {
        "phase": phase,
        "profile": profile,
        "target_tokens": target_tokens,
        "hashes": hashes.__dict__,
    }
    size_label = f"{target_tokens // 1_000_000_000}b"
    run_id = f"{phase}-{size_label}-{profile}-{_canonical_hash(identity)[:12]}"
    config = DataFactoryConfig(
        path=path,
        version=CONFIG_VERSION,
        phase=phase,
        profile=profile,
        repo_root=repo_root,
        corpus_root=corpus_root,
        tokenizer_path=tokenizer_path,
        seed=int(raw.get("seed", 20260714)),
        target_tokens=target_tokens,
        validation_tokens=_positive_int(raw.get("validation_tokens"), "validation_tokens"),
        calibration=calibration,
        candidate_oversample_ratio=oversample,
        enhancement_oversample_ratio=enhancement_oversample,
        candidate_priority=priority,
        buckets=buckets,
        attributes=attributes,
        dedup=dedup,
        enhancement=enhancement,
        sequence=sequence,
        source_registry=registry,
        hashes=hashes,
        run_id=run_id,
    )
    if sum(config.bucket_tokens.values()) != target_tokens:
        raise ValueError("rounded bucket token targets do not sum to target_tokens")
    return config
