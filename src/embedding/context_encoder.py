"""Shared causal context over text positions, with candidate-equivariant fusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache


class PGCACompositeCache(DynamicCache):
    """Keep backbone and feature KV caches in one per-generation cache object."""

    def __init__(self, config):
        self.backbone_config = config
        super().__init__(config=config)
        self.feature_cache = DynamicCache()

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        self.feature_cache.reorder_cache(beam_idx)

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        self.feature_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self.feature_cache.batch_select_indices(indices)

    def crop(self, max_length):
        super().crop(max_length)
        self.feature_cache.crop(max_length)

    def reset(self):
        # DynamicLayer.reset retains its old length/batch storage; recreate lazy layers.
        super().__init__(config=self.backbone_config)
        self.feature_cache = DynamicCache()


class FeatureRMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, value):
        value_float = value.float()
        normalized = value_float * torch.rsqrt(value_float.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(value.dtype)


class CausalFeatureBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        size = config.pgca_context_hidden_size
        self.num_heads = config.pgca_context_num_heads
        self.head_dim = size // self.num_heads
        self.layer_idx = layer_idx
        self.input_norm = FeatureRMSNorm(size, config.rms_norm_eps)
        self.post_norm = FeatureRMSNorm(size, config.rms_norm_eps)
        self.q_proj = nn.Linear(size, size, bias=False)
        self.k_proj = nn.Linear(size, size, bias=False)
        self.v_proj = nn.Linear(size, size, bias=False)
        self.o_proj = nn.Linear(size, size, bias=False)
        self.gate_proj = nn.Linear(size, config.pgca_context_intermediate_size, bias=False)
        self.up_proj = nn.Linear(size, config.pgca_context_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.pgca_context_intermediate_size, size, bias=False)
        self.rope_theta = config.pgca_context_rope_theta

    def _rotate(self, value, position_ids):
        # Recompute the tiny constant on-device: meta loading does not initialize custom nonpersistent buffers.
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=value.device).float() / self.head_dim))
        angles = position_ids.float().unsqueeze(-1) * inv_freq
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        half = self.head_dim // 2
        rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
        return value * angles.cos().to(value.dtype) + rotated * angles.sin().to(value.dtype)

    def forward(self, hidden, position_ids, allowed, cache=None):
        normed = self.input_norm(hidden)
        shape = (*hidden.shape[:2], self.num_heads, self.head_dim)
        query = self._rotate(self.q_proj(normed).view(shape).transpose(1, 2), position_ids)
        key = self._rotate(self.k_proj(normed).view(shape).transpose(1, 2), position_ids)
        value = self.v_proj(normed).view(shape).transpose(1, 2)
        if cache is not None:
            key, value = cache.update(key, value, self.layer_idx)
        # An explicit rectangular mask also handles cached decoding and left padding.
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed, dropout_p=0.0)
        hidden = hidden + self.o_proj(attended.transpose(1, 2).reshape_as(hidden))
        normed = self.post_norm(hidden)
        return hidden + self.down_proj(F.silu(self.gate_proj(normed)) * self.up_proj(normed))


class CausalFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, width = config.hidden_size, config.pgca_context_hidden_size
        self.input_proj = nn.Linear(2 * dim, width, bias=False)
        self.type_embedding = nn.Embedding(config.pgca_context_num_token_types, width)
        self.missing_pinyin_embedding = nn.Embedding(2, width)
        self.layers = nn.ModuleList(CausalFeatureBlock(config, i) for i in range(config.pgca_context_num_layers))
        self.final_norm = FeatureRMSNorm(width, config.rms_norm_eps)
        self.slot_norm = FeatureRMSNorm(dim, config.rms_norm_eps)
        self.feature_proj = nn.Linear(dim, width, bias=False)
        self.context_proj = nn.Linear(width, width, bias=False)
        self.slot_type_embedding = nn.Embedding(2, width)
        self.output_proj = nn.Linear(width, dim, bias=False)

    def forward(self, memory, feature_mask, token_types, position_ids, attention_mask=None, cache=None):
        batch, length, slots, _ = memory.shape
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("feature context currently supports only unpacked 2D padding masks")
        past_length = cache.get_seq_length() if cache is not None else 0
        total_length = past_length + length
        if attention_mask is None:
            padding = torch.ones(batch, total_length, dtype=torch.bool, device=memory.device)
        else:
            if attention_mask.shape != (batch, total_length):
                raise ValueError("feature context needs the full prefix padding mask during cached decoding")
            padding = attention_mask.bool()
        feature_mask = feature_mask.bool() & padding[:, -length:, None]
        masked_memory = memory.masked_fill(~feature_mask[..., None], 0)
        pinyin_mask = feature_mask[:, :, :-1]
        pinyin = masked_memory[:, :, :-1].sum(2) / pinyin_mask.sum(2, keepdim=True).clamp_min(1)
        hidden = self.input_proj(torch.cat((pinyin, masked_memory[:, :, -1]), dim=-1))
        hidden = hidden + self.type_embedding(token_types)
        hidden = hidden + self.missing_pinyin_embedding((~pinyin_mask.any(-1)).long())
        query_positions = torch.arange(past_length, total_length, device=memory.device)
        key_positions = torch.arange(total_length, device=memory.device)
        allowed = (key_positions[None, :] <= query_positions[:, None])[None, None]
        allowed = allowed & padding[:, None, None, :]
        for layer in self.layers:
            hidden = layer(hidden, position_ids, allowed, cache)
        context = self.context_proj(self.final_norm(hidden))[:, :, None]
        slot_types = torch.zeros(slots, dtype=torch.long, device=memory.device)
        slot_types[-1] = 1
        delta = self.output_proj(F.silu(self.feature_proj(self.slot_norm(masked_memory)) + context
                                       + self.slot_type_embedding(slot_types)))
        return (masked_memory + delta).masked_fill(~feature_mask[..., None], 0), feature_mask
