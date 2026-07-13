"""Position-wise phonetic-glyph cross attention."""

from __future__ import annotations

import torch
from torch import nn


class PGCACrossAttention(nn.Module):
    """Cross-attend each token hidden state to its own feature memory."""

    def __init__(self, config, rms_norm_cls: type[nn.Module]):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(getattr(config, "pgca_num_attention_heads", config.num_attention_heads))
        self.num_key_value_heads = int(getattr(config, "pgca_num_key_value_heads", config.num_key_value_heads))
        self.head_dim = int(getattr(config, "pgca_head_dim", getattr(config, "head_dim", self.hidden_size // self.num_heads)))
        self.feature_hidden_size = int(getattr(config, "pgca_feature_hidden_size", self.hidden_size))
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("pgca_num_attention_heads must be divisible by pgca_num_key_value_heads")
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError("pgca_num_attention_heads * pgca_head_dim must equal hidden_size")

        bias = bool(getattr(config, "attention_bias", False))
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.feature_hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.feature_hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)
        self.q_norm = rms_norm_cls(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = rms_norm_cls(self.head_dim, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(float(getattr(config, "pgca_dropout", 0.0)))
        self.gate = nn.Parameter(torch.tensor(float(getattr(config, "pgca_gate_init", 0.0))))

    def forward(
        self,
        hidden_states: torch.Tensor,
        feature_memory: torch.Tensor,
        feature_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if feature_memory.ndim != 4:
            raise ValueError("feature_memory must have shape (batch, seq, slots, hidden)")

        batch_size, seq_len, num_slots, _ = feature_memory.shape
        query_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        kv_shape = (batch_size, seq_len, num_slots, self.num_key_value_heads, self.head_dim)

        query = self.q_norm(self.q_proj(hidden_states).view(query_shape))
        key = self.k_norm(self.k_proj(feature_memory).view(kv_shape))
        value = self.v_proj(feature_memory).view(kv_shape)

        key = self._repeat_kv(key)
        value = self._repeat_kv(value)

        scores = torch.einsum("bshd,bsmhd->bshm", query, key) * self.scaling
        if feature_mask is None:
            mask = torch.ones((batch_size, seq_len, num_slots), dtype=torch.bool, device=feature_memory.device)
        else:
            mask = feature_mask.to(device=feature_memory.device, dtype=torch.bool)

        scores = scores.masked_fill(~mask.unsqueeze(2), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        attn = attn.masked_fill(~mask.unsqueeze(2), 0)
        attn = self.dropout(attn)

        context = torch.einsum("bshm,bsmhd->bshd", attn, value)
        context = context.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.gate * self.o_proj(context)

    def _repeat_kv(self, states: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return states
        batch_size, seq_len, num_slots, num_kv_heads, head_dim = states.shape
        states = states[:, :, :, :, None, :].expand(
            batch_size,
            seq_len,
            num_slots,
            num_kv_heads,
            self.num_key_value_groups,
            head_dim,
        )
        return states.reshape(batch_size, seq_len, num_slots, self.num_heads, head_dim)
