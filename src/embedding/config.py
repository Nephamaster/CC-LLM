"""Configuration helpers for semantic and feature embeddings."""

from __future__ import annotations

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
