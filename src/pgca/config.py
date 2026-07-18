"""Configuration helpers for PGCA layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass


def default_pgca_layers(num_hidden_layers: int) -> list[int]:
    """Select the middle third layers for PGCA by default."""
    if num_hidden_layers <= 0:
        return []
    width = max(1, num_hidden_layers // 3)
    start = max(0, (num_hidden_layers - width) // 2)
    return list(range(start, start + width))


@dataclass(frozen=True)
class PGCAConfig:
    use_pgca: bool = True
    pgca_layers: list[int] | None = None
    pgca_num_attention_heads: int = 16
    pgca_num_key_value_heads: int = 8
    pgca_head_dim: int = 128
    pgca_gate_init: float = 0.0
    pgca_dropout: float = 0.0
    pgca_feature_slots: int = 9
    pgca_feature_hidden_size: int = 2048
    pgca_feature_embedding_dim: int = 256
    pgca_max_pinyin_per_char: int = 8
    pgca_feature_vocab_sizes: dict[str, int] | None = None
    pgca_use_glyph_image: bool = False

    @classmethod
    def from_model_config(cls, config) -> "PGCAConfig":
        layers = config.pgca_layers
        if layers is None:
            layers = default_pgca_layers(int(config.num_hidden_layers))
        return cls(
            use_pgca=bool(config.use_pgca),
            pgca_layers=list(layers),
            pgca_num_attention_heads=int(config.pgca_num_attention_heads),
            pgca_num_key_value_heads=int(config.pgca_num_key_value_heads),
            pgca_head_dim=int(config.pgca_head_dim),
            pgca_gate_init=float(config.pgca_gate_init),
            pgca_dropout=float(config.pgca_dropout),
            pgca_feature_slots=int(config.pgca_feature_slots),
            pgca_feature_hidden_size=int(config.pgca_feature_hidden_size),
            pgca_feature_embedding_dim=int(config.pgca_feature_embedding_dim),
            pgca_max_pinyin_per_char=int(config.pgca_max_pinyin_per_char),
            pgca_feature_vocab_sizes=dict(config.pgca_feature_vocab_sizes),
            pgca_use_glyph_image=bool(config.pgca_use_glyph_image),
        )

    def to_dict(self) -> dict:
        return asdict(self)
