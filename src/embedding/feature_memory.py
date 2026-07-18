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
SEQUENCE_FEATURE_KEYS = (
    "pinyin_ids",
    "shengmu_ids",
    "yunmu_ids",
    "tone_ids",
    "pinyin_mask",
)
BOOLEAN_FEATURE_KEYS = {"is_hanzi", "pinyin_mask"}


def load_feature_index(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    feature_index = torch.load(path, map_location=map_location)
    missing = [key for key in FEATURE_INDEX_KEYS if key not in feature_index]
    if missing:
        raise ValueError(f"feature_index is missing keys: {missing}")
    return {key: feature_index[key] for key in FEATURE_INDEX_KEYS}


def validate_feature_index(
    feature_index: dict[str, torch.Tensor],
    *,
    vocab_size: int,
    max_pinyin_per_char: int,
) -> dict[str, torch.Tensor]:
    missing = [key for key in FEATURE_INDEX_KEYS if key not in feature_index]
    if missing:
        raise ValueError(f"feature_index is missing keys: {missing}")

    normalized: dict[str, torch.Tensor] = {}
    for key in FEATURE_INDEX_KEYS:
        value = feature_index[key]
        if not torch.is_tensor(value):
            raise TypeError(f"feature_index[{key!r}] must be a tensor")
        expected_shape = (
            (vocab_size, max_pinyin_per_char)
            if key in SEQUENCE_FEATURE_KEYS
            else (vocab_size,)
        )
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"feature_index[{key!r}] has shape {tuple(value.shape)}; expected {expected_shape}"
            )
        normalized[key] = value.to(dtype=torch.bool if key in BOOLEAN_FEATURE_KEYS else torch.long)
    return normalized


def empty_feature_index(vocab_size: int, max_pinyin_per_char: int) -> dict[str, torch.Tensor]:
    feature_index: dict[str, torch.Tensor] = {}
    for key in FEATURE_INDEX_KEYS:
        shape = (
            (vocab_size, max_pinyin_per_char)
            if key in SEQUENCE_FEATURE_KEYS
            else (vocab_size,)
        )
        dtype = torch.bool if key in BOOLEAN_FEATURE_KEYS else torch.long
        feature_index[key] = torch.zeros(shape, dtype=dtype)
    return feature_index


class FeatureMemoryBuilder(nn.Module):
    """Build feature memory from semantic input ids."""

    def __init__(
        self,
        feature_index: dict[str, torch.Tensor],
        feature_embedding: PhoneticGlyphFeatureEmbedding,
        *,
        persistent: bool = True,
    ):
        super().__init__()
        self.feature_embedding = feature_embedding
        for key in FEATURE_INDEX_KEYS:
            self.register_buffer(key, feature_index[key], persistent=persistent)
        self.register_buffer("feature_index_ready", torch.tensor(False), persistent=persistent)
        self._feature_index_checked = False

    @classmethod
    def empty(
        cls,
        *,
        vocab_size: int,
        max_pinyin_per_char: int,
        feature_embedding: PhoneticGlyphFeatureEmbedding,
    ) -> "FeatureMemoryBuilder":
        return cls(
            empty_feature_index(vocab_size, max_pinyin_per_char),
            feature_embedding,
            persistent=True,
        )

    @classmethod
    def from_pretrained(
        cls,
        feature_index_path: str | Path,
        feature_embedding: PhoneticGlyphFeatureEmbedding,
        *,
        map_location: str | torch.device = "cpu",
        persistent: bool = True,
    ) -> "FeatureMemoryBuilder":
        return cls(
            load_feature_index(feature_index_path, map_location=map_location),
            feature_embedding,
            persistent=persistent,
        )

    def set_feature_index(self, feature_index: dict[str, torch.Tensor]) -> None:
        reference = self.is_hanzi
        normalized = validate_feature_index(
            feature_index,
            vocab_size=reference.shape[0],
            max_pinyin_per_char=self.pinyin_ids.shape[1],
        )
        with torch.no_grad():
            for key, value in normalized.items():
                target = getattr(self, key)
                target.copy_(value.to(device=target.device, dtype=target.dtype))
            self.feature_index_ready.fill_(True)
            self._feature_index_checked = True

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.feature_embedding(self.lookup(input_ids))

    def lookup(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self._feature_index_checked:
            if not bool(self.feature_index_ready.item()):
                raise RuntimeError("PGCA feature index is not initialized in this checkpoint")
            self._feature_index_checked = True
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        return {
            key: getattr(self, key).to(input_ids.device).index_select(0, input_ids.reshape(-1)).reshape(
                *input_ids.shape, *getattr(self, key).shape[1:]
            )
            for key in FEATURE_INDEX_KEYS
        }
