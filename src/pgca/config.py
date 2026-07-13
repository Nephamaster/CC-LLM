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
    pgca_build_features_in_model: bool = True

    @classmethod
    def from_model_config(cls, config) -> "PGCAConfig":
        layers = getattr(config, "pgca_layers", None)
        if layers is None:
            layers = default_pgca_layers(int(config.num_hidden_layers))
        return cls(
            use_pgca=bool(getattr(config, "use_pgca", True)),
            pgca_layers=list(layers),
            pgca_num_attention_heads=int(getattr(config, "pgca_num_attention_heads", config.num_attention_heads)),
            pgca_num_key_value_heads=int(
                getattr(config, "pgca_num_key_value_heads", config.num_key_value_heads)
            ),
            pgca_head_dim=int(getattr(config, "pgca_head_dim", config.head_dim)),
            pgca_gate_init=float(getattr(config, "pgca_gate_init", 0.0)),
            pgca_dropout=float(getattr(config, "pgca_dropout", 0.0)),
            pgca_feature_slots=int(getattr(config, "pgca_feature_slots", 9)),
            pgca_feature_hidden_size=int(getattr(config, "pgca_feature_hidden_size", config.hidden_size)),
            pgca_build_features_in_model=bool(getattr(config, "pgca_build_features_in_model", True)),
        )

    def to_dict(self) -> dict:
        return asdict(self)
