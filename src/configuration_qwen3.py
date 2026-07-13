# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3 configuration with PGCA settings."""

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters

from .pgca.config import default_pgca_layers


class Qwen3Config(PreTrainedConfig):
    """Qwen3 configuration extended with PGCA fields."""

    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.pgca_attn.q_proj": "colwise",
        "layers.*.pgca_attn.k_proj": "colwise",
        "layers.*.pgca_attn.v_proj": "colwise",
        "layers.*.pgca_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.pgca_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.pgca_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 32,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: RopeParameters | dict | None = None,
        attention_bias: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int | None = 4096,
        max_window_layers: int = 28,
        layer_types: list[str] | None = None,
        attention_dropout: float | int = 0.0,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        use_pgca: bool = False,
        pgca_layers: list[int] | None = None,
        pgca_num_attention_heads: int | None = None,
        pgca_num_key_value_heads: int | None = None,
        pgca_head_dim: int | None = None,
        pgca_gate_init: float = 0.0,
        pgca_dropout: float = 0.0,
        pgca_feature_slots: int = 9,
        pgca_feature_hidden_size: int | None = None,
        pgca_build_features_in_model: bool = True,
        **kwargs,
    ):
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        sliding_window = sliding_window if use_sliding_window else None
        if layer_types is None:
            layer_types = [
                "sliding_attention" if sliding_window is not None and i >= max_window_layers else "full_attention"
                for i in range(num_hidden_layers)
            ]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.layer_types = layer_types
        self.attention_dropout = attention_dropout

        self.use_pgca = use_pgca
        self.pgca_layers = (
            default_pgca_layers(num_hidden_layers) if pgca_layers is None and use_pgca else list(pgca_layers or [])
        )
        self.pgca_layers = [int(layer_idx) for layer_idx in self.pgca_layers]
        self.pgca_num_attention_heads = pgca_num_attention_heads or num_attention_heads
        self.pgca_num_key_value_heads = pgca_num_key_value_heads or num_key_value_heads
        self.pgca_head_dim = pgca_head_dim or head_dim
        self.pgca_gate_init = pgca_gate_init
        self.pgca_dropout = pgca_dropout
        self.pgca_feature_slots = pgca_feature_slots
        self.pgca_feature_hidden_size = pgca_feature_hidden_size or hidden_size
        self.pgca_build_features_in_model = pgca_build_features_in_model

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Qwen3Config"]
