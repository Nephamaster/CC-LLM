"""Lookup token-aligned feature ids and build feature memory tensors."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .feature_embedding import PhoneticGlyphFeatureEmbedding


FEATURE_INDEX_KEYS = (
    "is_hanzi",
    "pinyin_ids",
    "shengmu_ids",
    "yunmu_ids",
    "tone_ids",
    "pinyin_mask",
    "stroke_count_ids",
    "radical_stroke_ids",
    "structure_ids",
)


def load_feature_index(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    feature_index = torch.load(path, map_location=map_location)
    missing = [key for key in FEATURE_INDEX_KEYS if key not in feature_index]
    if missing:
        raise ValueError(f"feature_index is missing keys: {missing}")
    return {key: feature_index[key] for key in FEATURE_INDEX_KEYS}


class FeatureMemoryBuilder(nn.Module):
    """Build feature memory from semantic input ids."""

    def __init__(
        self,
        feature_index: dict[str, torch.Tensor],
        feature_embedding: PhoneticGlyphFeatureEmbedding,
    ):
        super().__init__()
        self.feature_embedding = feature_embedding
        for key, value in feature_index.items():
            self.register_buffer(key, value, persistent=False)

    @classmethod
    def from_pretrained(
        cls,
        feature_index_path: str | Path,
        feature_embedding: PhoneticGlyphFeatureEmbedding,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "FeatureMemoryBuilder":
        return cls(load_feature_index(feature_index_path, map_location=map_location), feature_embedding)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_ids = self.lookup(input_ids)
        return self.feature_embedding(feature_ids)

    def lookup(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        return {
            key: getattr(self, key).to(input_ids.device).index_select(0, input_ids.reshape(-1)).reshape(
                *input_ids.shape, *getattr(self, key).shape[1:]
            )
            for key in FEATURE_INDEX_KEYS
        }
