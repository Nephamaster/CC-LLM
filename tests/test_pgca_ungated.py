from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from src.configuration_qwen3_pgca import FEATURE_VOCAB_KEYS, Qwen3PGCAConfig
from src.embedding.feature_memory import empty_feature_index
from src.modeling_qwen3_pgca import Qwen3PGCAForCausalLM, Qwen3RMSNorm
from src.pgca.attention import PGCACrossAttention
from src.pgca.build_pgca import (
    AUTO_MAP,
    check_pgca_parameters,
    copy_runtime_code,
    initialize_pgca_parameters,
    save_pretrained_without_auto_copy,
    update_config,
    validate_runtime_code,
)
from src.pgca.config import PGCAConfig
from src.pgca.validate_pgca import _check_modules


def tiny_config(**kwargs) -> Qwen3PGCAConfig:
    return Qwen3PGCAConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        use_cache=False,
        use_pgca=True,
        pgca_layers=[0, 1],
        pgca_feature_embedding_dim=4,
        pgca_max_pinyin_per_char=2,
        pgca_feature_vocab_sizes={key: 4 for key in FEATURE_VOCAB_KEYS},
        attn_implementation="eager",
        **kwargs,
    )


def tiny_model() -> Qwen3PGCAForCausalLM:
    model = Qwen3PGCAForCausalLM(tiny_config())
    index = empty_feature_index(model.config.vocab_size, model.config.pgca_max_pinyin_per_char)
    for values in index.values():
        values[2] = 1
    model.model.feature_memory_builder.set_feature_index(index)
    initialize_pgca_parameters(model)
    return model


class UngatedPGCATest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_attention_returns_unscaled_projection(self) -> None:
        attention = PGCACrossAttention(tiny_config(), Qwen3RMSNorm)
        hidden = torch.randn(1, 2, 8)
        memory = torch.randn(1, 2, 3, 8)
        mask = torch.tensor([[[True, False, False], [True, False, False]]])

        output = attention(hidden, memory, mask)
        value = attention.v_proj(memory[:, :, 0]).view(1, 2, 1, 4)
        context = value.repeat_interleave(2, dim=2).reshape(1, 2, 8)
        torch.testing.assert_close(output, attention.o_proj(context))
        self.assertGreater(output.abs().sum().item(), 0)
        self.assertNotIn("gate", dict(attention.named_parameters()))
        self.assertNotIn("gate", attention.state_dict())

    def test_all_masked_positions_remain_zero_and_finite(self) -> None:
        attention = PGCACrossAttention(tiny_config(), Qwen3RMSNorm)
        hidden = torch.randn(1, 2, 8, requires_grad=True)
        memory = torch.randn(1, 2, 3, 8, requires_grad=True)
        mask = torch.zeros(1, 2, 3, dtype=torch.bool)

        output = attention(hidden, memory, mask)
        torch.testing.assert_close(output, torch.zeros_like(output))
        output.sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all().item())
        self.assertTrue(torch.isfinite(memory.grad).all().item())

    def test_attention_projections_receive_gradients_immediately(self) -> None:
        attention = PGCACrossAttention(tiny_config(), Qwen3RMSNorm)
        output = attention(torch.randn(2, 3, 8), torch.randn(2, 3, 3, 8), None)
        output.square().sum().backward()

        for name, parameter in attention.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_config_drops_legacy_gate_setting(self) -> None:
        config = tiny_config(pgca_gate_init=0.75)
        self.assertNotIn("pgca_gate_init", config.to_dict())
        self.assertNotIn("pgca_gate_init", PGCAConfig.from_model_config(config).to_dict())

    def test_migration_config_removes_legacy_gate_setting(self) -> None:
        config = tiny_config()
        data = config.to_dict()
        data["pgca_gate_init"] = 0.0
        embedding = {
            "semantic_vocab_size": config.vocab_size,
            "d_model": config.hidden_size,
            "d_feat": config.pgca_feature_embedding_dim,
            "num_feature_slots": config.pgca_feature_slots,
            "max_pinyin_per_char": config.pgca_max_pinyin_per_char,
            "feature_vocab_sizes": config.pgca_feature_vocab_sizes,
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            embedding_path = Path(directory) / "embedding_config.json"
            config_path.write_text(json.dumps(data), encoding="utf-8")
            embedding_path.write_text(json.dumps(embedding), encoding="utf-8")
            updated = update_config(config_path, embedding_path, [0, 1], "test-hash")
            self.assertNotIn("pgca_gate_init", updated)
            self.assertNotIn("pgca_gate_init", json.loads(config_path.read_text(encoding="utf-8")))

    def test_decoder_adds_both_branches_and_residual(self) -> None:
        layer = tiny_model().model.layers[0]
        hidden = torch.randn(1, 3, 8)
        self_output = torch.randn_like(hidden)
        pgca_output = torch.randn_like(hidden)
        memory = torch.randn(1, 3, 3, 8)
        mask = torch.ones(1, 3, 3, dtype=torch.bool)
        layer.post_attention_layernorm = nn.Identity()
        layer.mlp = nn.Identity()

        with patch.object(layer.self_attn, "forward", return_value=(self_output, None)) as self_attention:
            with patch.object(layer.pgca_attn, "forward", return_value=pgca_output) as pgca_attention:
                output = layer(hidden, feature_memory=memory, feature_mask=mask)

        torch.testing.assert_close(output, 2 * (hidden + self_output + pgca_output))
        self.assertIs(self_attention.call_args.kwargs["hidden_states"], pgca_attention.call_args.args[0])

    def test_parameter_checks_require_gate_free_model(self) -> None:
        model = tiny_model()
        self.assertTrue(check_pgca_parameters(model)["passed"])
        self.assertTrue(_check_modules(model)["gate_free"])

        model.model.layers[0].pgca_attn.gate = nn.Parameter(torch.tensor(1.0))
        self.assertFalse(check_pgca_parameters(model)["passed"])
        self.assertFalse(_check_modules(model)["gate_free"])

    def test_causal_lm_forward_and_backward(self) -> None:
        model = tiny_model()
        input_ids = torch.tensor([[2, 3, 2, 4]])
        output = model(input_ids, labels=input_ids)
        self.assertEqual(output.logits.shape, (1, 4, 16))
        self.assertTrue(torch.isfinite(output.loss).item())
        output.loss.backward()
        self.assertFalse(any(name.endswith(".pgca_attn.gate") for name in model.state_dict()))
        for layer in model.model.layers:
            gradient = layer.pgca_attn.o_proj.weight.grad
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_remote_checkpoint_round_trip(self) -> None:
        model = tiny_model().eval()
        model.config.auto_map = dict(AUTO_MAP)
        input_ids = torch.tensor([[2, 3, 2, 4]])
        with torch.no_grad():
            expected = model(input_ids).logits

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            save_pretrained_without_auto_copy(model, output_path)
            copy_runtime_code(output_path)
            self.assertTrue(validate_runtime_code(output_path)["portable"])
            loaded = AutoModelForCausalLM.from_pretrained(output_path, trust_remote_code=True).eval()
            self.assertTrue(_check_modules(loaded)["gate_free"])
            self.assertNotIn("pgca_gate_init", loaded.config.to_dict())
            with torch.no_grad():
                torch.testing.assert_close(loaded(input_ids).logits, expected)


if __name__ == "__main__":
    unittest.main()
