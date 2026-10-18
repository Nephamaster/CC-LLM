"""Configuration helpers for semantic and feature embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingFeatureConfig:
    semantic_vocab_size: int
    d_model: int = 2048
    d_feat: int = 256
    max_pinyin_per_char: int = 8
    num_feature_slots: int = 9
    feature_vocab_sizes: dict[str, int] | None = None
    initializer_range: float = 0.02
    use_glyph_image: bool = False
    feature_schema_version: int = 1
    feature_vocab_manifest_sha256: str | None = None
    feature_index_sha256: str | None = None

    @classmethod
    def from_artifacts(
        cls,
        char_model_path: str | Path,
        *,
        d_model: int = 2048,
        d_feat: int = 256,
        initializer_range: float = 0.02,
    ) -> "EmbeddingFeatureConfig":
        root = Path(char_model_path)
        feature_manifest = _read_json(root / "features" / "feature_vocab_manifest.json")
        index_manifest = _read_json(root / "features" / "feature_index_manifest.json")
        if feature_manifest.get("schema_version") != 2 or index_manifest.get("schema_version") != 2:
            raise ValueError("rebuild feature-vocab and feature-index with schema v2 before embedding config")
        expected_vocab_files = {f"feature_vocabs/{name}_vocab.json" for name in
                                ("pinyin", "shengmu", "yunmu", "tone", "stroke_count", "radical_stroke", "structure")}
        if set(feature_manifest.get("artifact_sha256", {})) != expected_vocab_files | {"char_features.json"}:
            raise ValueError("incomplete feature vocabulary hashes")
        if set(index_manifest.get("artifact_sha256", {})) != {"char_feature_index.jsonl", "feature_index.pt"}:
            raise ValueError("both JSONL and PT feature indices with hashes are required")
        for manifest in (feature_manifest, index_manifest):
            for relative, expected in manifest["artifact_sha256"].items():
                if _sha256(root / "features" / relative) != expected:
                    raise ValueError(f"feature artifact hash mismatch: {relative}")
        manifest_hash = _sha256(root / "features" / "feature_vocab_manifest.json")
        if index_manifest["feature_vocab_manifest_sha256"] != manifest_hash:
            raise ValueError("feature index does not belong to the feature vocabulary")
        if index_manifest["semantic_vocab_sha256"] != _sha256(root / "vocab.json"):
            raise ValueError("feature index does not belong to this semantic vocabulary")
        vocab_size = int(index_manifest["vocab_size"])
        max_pinyin = int(index_manifest["max_pinyin_per_char"])
        return cls(
            semantic_vocab_size=vocab_size,
            d_model=d_model,
            d_feat=d_feat,
            max_pinyin_per_char=max_pinyin,
            num_feature_slots=max_pinyin + 1,
            feature_vocab_sizes=dict(feature_manifest["vocab_sizes"]),
            initializer_range=initializer_range,
            use_glyph_image=False,
            feature_schema_version=2,
            feature_vocab_manifest_sha256=manifest_hash,
            feature_index_sha256=index_manifest["artifact_sha256"]["feature_index.pt"],
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save_pretrained(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wt", encoding="utf-8", newline="\n") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            f.write("\n")


def _read_json(path: Path) -> dict:
    with path.open("rt", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
