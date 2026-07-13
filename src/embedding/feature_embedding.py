"""Phonetic and structural feature embedding modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .config import EmbeddingFeatureConfig


class PhoneticGlyphFeatureEmbedding(nn.Module):
    """Convert token-aligned feature ids into per-position feature memory."""

    def __init__(self, config: EmbeddingFeatureConfig):
        super().__init__()
        if config.feature_vocab_sizes is None:
            raise ValueError("feature_vocab_sizes must be provided")
        self.config = config
        d_feat = config.d_feat
        d_model = config.d_model
        sizes = config.feature_vocab_sizes

        self.pinyin_embed = nn.Embedding(sizes["pinyin"], d_feat)
        self.shengmu_embed = nn.Embedding(sizes["shengmu"], d_feat)
        self.yunmu_embed = nn.Embedding(sizes["yunmu"], d_feat)
        self.tone_embed = nn.Embedding(sizes["tone"], d_feat)
        self.stroke_count_embed = nn.Embedding(sizes["stroke_count"], d_feat)
        self.radical_stroke_embed = nn.Embedding(sizes["radical_stroke"], d_feat)
        self.structure_embed = nn.Embedding(sizes["structure"], d_feat)

        self.pinyin_projector = nn.Sequential(
            nn.Linear(d_feat * 4, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.shape_projector = nn.Sequential(
            nn.Linear(d_feat * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.reset_parameters(config.initializer_range)

    def reset_parameters(self, std: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, feature_ids: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pinyin_memory = self._build_pinyin_memory(feature_ids)
        shape_memory = self._build_shape_memory(feature_ids)

        is_hanzi = feature_ids["is_hanzi"].bool()
        pinyin_mask = feature_ids["pinyin_mask"].bool() & is_hanzi.unsqueeze(-1)
        shape_mask = is_hanzi.unsqueeze(-1)

        memory = torch.cat([pinyin_memory, shape_memory], dim=2)
        mask = torch.cat([pinyin_mask, shape_mask], dim=2)
        memory = memory.masked_fill(~mask.unsqueeze(-1), 0)
        return memory, mask

    def _build_pinyin_memory(self, feature_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            self.pinyin_embed(feature_ids["pinyin_ids"]),
            self.shengmu_embed(feature_ids["shengmu_ids"]),
            self.yunmu_embed(feature_ids["yunmu_ids"]),
            self.tone_embed(feature_ids["tone_ids"]),
        ]
        return self.pinyin_projector(torch.cat(parts, dim=-1))

    def _build_shape_memory(self, feature_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            self.stroke_count_embed(feature_ids["stroke_count_ids"]),
            self.radical_stroke_embed(feature_ids["radical_stroke_ids"]),
            self.structure_embed(feature_ids["structure_ids"]),
        ]
        shape = self.shape_projector(torch.cat(parts, dim=-1))
        return shape.unsqueeze(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build",), nargs="?", default="build")
    parser.add_argument("--char-model-path", type=Path, default=Path("models/Qwen3-1.7B-Base-Char"))
    parser.add_argument("--d-model", type=int, default=2048)
    parser.add_argument("--d-feat", type=int, default=256)
    parser.add_argument("--initializer-range", type=float, default=0.02)
    args = parser.parse_args()

    config = EmbeddingFeatureConfig.from_artifacts(
        args.char_model_path,
        d_model=args.d_model,
        d_feat=args.d_feat,
        initializer_range=args.initializer_range,
    )
    config_path = args.char_model_path / "embedding_config.json"
    config.save_pretrained(config_path)
    print(json.dumps({"embedding_config": str(config_path), **config.to_dict()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()