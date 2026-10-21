from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM

from src.configuration_qwen3_pgca import FEATURE_VOCAB_KEYS, Qwen3PGCAConfig
from src.embedding.feature_memory import empty_feature_index
from src.modeling_qwen3_pgca import Qwen3PGCAForCausalLM
from src.pgca.auxiliary import GLYPH_ATTRIBUTES, PGCAAuxiliaryOutput
from src.pgca.build_pgca import (AUTO_MAP, check_pgca_parameters, copy_runtime_code,
                                 initialize_pgca_parameters, save_pretrained_without_auto_copy)


def context_model(*, zero_output=False):
    config = Qwen3PGCAConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=32,
        use_cache=False, use_pgca=True, pgca_layers=[0, 1], pgca_feature_embedding_dim=4,
        pgca_max_pinyin_per_char=2, pgca_feature_vocab_sizes={key: 8 for key in FEATURE_VOCAB_KEYS},
        pgca_context_enabled=True, pgca_context_num_layers=2, pgca_context_hidden_size=8,
        pgca_context_num_heads=2, pgca_context_intermediate_size=16,
        pgca_aux_enabled=True, pgca_aux_hidden_size=4, pgca_zero_init_output=zero_output,
        attn_implementation="eager", pad_token_id=0, eos_token_id=15,
    )
    model = Qwen3PGCAForCausalLM(config)
    index = empty_feature_index(16, 2)
    for values in index.values():
        values[2] = 3 if values.dtype == torch.long else True
        values[4] = 4 if values.dtype == torch.long else True
    index["pinyin_ids"][2, 1] = 4
    builder = model.model.feature_memory_builder
    builder.set_feature_index(index)
    builder.token_type_ids[index["is_hanzi"]] = 1
    initialize_pgca_parameters(model)
    return model


class ContextPGCATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_round2_auxiliary_defaults_and_legacy_config_round_trip(self):
        model = context_model()
        self.assertEqual(model.config.pgca_aux_phonetic_weight, 0.03)
        self.assertEqual(model.config.pgca_aux_glyph_weight, 0.03)
        self.assertEqual(model.config.pgca_aux_warmup_tokens, 10_000_000)
        model.config.pgca_aux_phonetic_weight = 0.05
        model.config.pgca_aux_glyph_weight = 0.02
        with tempfile.TemporaryDirectory() as directory:
            save_pretrained_without_auto_copy(model, Path(directory))
            restored = Qwen3PGCAConfig.from_pretrained(directory)
        self.assertEqual(restored.pgca_aux_phonetic_weight, 0.05)
        self.assertEqual(restored.pgca_aux_glyph_weight, 0.02)
        self.assertEqual(restored.pgca_aux_warmup_tokens, 10_000_000)
        for name in ("pgca_aux_phonetic_weight", "pgca_aux_glyph_weight"):
            for value in (-0.1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "auxiliary weight"):
                    Qwen3PGCAConfig(**{name: value})

    def test_prefix_invariance_and_candidate_permutation(self):
        encoder = context_model().model.feature_memory_builder.context_encoder.eval()
        memory = torch.randn(2, 5, 3, 8)
        mask = torch.ones(2, 5, 3, dtype=torch.bool)
        mask[0, 1] = False
        types = torch.ones(2, 5, dtype=torch.long)
        positions = torch.arange(5)[None]
        reference, _ = encoder(memory, mask, types, positions)
        changed = memory.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100
        modified, _ = encoder(changed, mask, types, positions)
        torch.testing.assert_close(reference[:, :3], modified[:, :3])
        order = [1, 0, 2]
        permuted, _ = encoder(memory[:, :, order], mask[:, :, order], types, positions)
        torch.testing.assert_close(permuted, reference[:, :, order])
        self.assertEqual(reference[0, 1].abs().sum().item(), 0)

    def test_suffix_tokens_cannot_change_prefix_logits(self):
        model = context_model().eval()
        ids = torch.tensor([[2, 3, 4, 2]])
        reference = model(ids).logits
        ids[:, -1] = 8
        torch.testing.assert_close(reference[:, :-1], model(ids).logits[:, :-1])

    def test_bfloat16_context_is_computed_once_and_remains_finite(self):
        model = context_model().to(torch.bfloat16).eval()
        encoder = model.model.feature_memory_builder.context_encoder
        with patch.object(encoder, "forward", wraps=encoder.forward) as forward:
            output = model(torch.tensor([[2, 3, 4, 2]]), return_aux_logits=True)
        self.assertEqual(forward.call_count, 1)
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.isfinite(output.phonetic_logits).all())

    def test_phase1_bypass_never_calls_feature_modules(self):
        model = context_model()
        model.config.pgca_enabled = False
        with patch.object(model.model.feature_memory_builder, "forward", side_effect=AssertionError("feature call")):
            with patch.object(model.model.layers[0].pgca_attn, "forward", side_effect=AssertionError("PGCA call")):
                output = model(torch.tensor([[2, 3, 4]]))
        self.assertIsNone(output.phonetic_logits)
        self.assertTrue(any("context_encoder" in key for key in model.state_dict()))

    def test_zero_projection_and_delayed_internal_gradient(self):
        model = context_model(zero_output=True)
        model.auxiliary_trained_tokens.fill_(123)
        initialize_pgca_parameters(model)
        self.assertEqual(model.auxiliary_trained_tokens.item(), 0)
        self.assertTrue(check_pgca_parameters(model)["passed"])
        ids = torch.tensor([[2, 3, 4, 2]])
        model.config.pgca_enabled = False
        reference = model(ids).logits
        model.config.pgca_enabled = True
        torch.testing.assert_close(model(ids).logits, reference)
        model(ids, labels=ids).loss.backward()
        attention = model.model.layers[0].pgca_attn
        self.assertGreater(attention.o_proj.weight.grad.abs().sum().item(), 0)
        self.assertEqual(attention.v_proj.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            attention.o_proj.weight.add_(attention.o_proj.weight.grad, alpha=-0.1)
        model.zero_grad()
        model(ids, labels=ids).loss.backward()
        self.assertGreater(attention.v_proj.weight.grad.abs().sum().item(), 0)
        gradients = [p.grad for p in model.model.feature_memory_builder.context_encoder.parameters()]
        self.assertGreater(sum(g.abs().sum().item() for g in gradients if g is not None), 0)

    def test_empty_slots_zero_even_with_output_bias(self):
        model = context_model()
        attention = model.model.layers[0].pgca_attn
        attention.o_proj.bias = torch.nn.Parameter(torch.ones(8))
        result = attention(torch.randn(1, 2, 8), torch.randn(1, 2, 3, 8), torch.zeros(1, 2, 3, dtype=torch.bool))
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_incremental_cache_matches_full_with_left_padding(self):
        model = context_model().eval()
        ids = torch.tensor([[0, 2, 3, 4], [2, 4, 2, 3]])
        mask = ids.ne(0).long()
        positions = mask.cumsum(-1).sub(1).clamp_min(0)
        full = model(ids, attention_mask=mask, position_ids=positions).logits
        cache = None
        for index in range(ids.shape[1]):
            output = model(ids[:, index:index + 1], attention_mask=mask[:, :index + 1],
                           position_ids=positions[:, index:index + 1], use_cache=True, past_key_values=cache)
            cache = output.past_key_values
            valid = mask[:, index].bool()
            torch.testing.assert_close(output.logits[valid, 0], full[valid, index], atol=2e-6, rtol=2e-5)
            self.assertEqual(cache.get_seq_length(), cache.feature_cache.get_seq_length())
        cache.reorder_cache(torch.tensor([1, 0]))
        cache.batch_repeat_interleave(2)
        cache.batch_select_indices(torch.tensor([0, 2]))
        cache.crop(2)
        self.assertEqual(cache.get_seq_length(), 2)
        self.assertEqual(cache.feature_cache.get_seq_length(), 2)
        cache.reset()
        self.assertEqual(cache.get_seq_length(), 0)
        self.assertEqual(cache.feature_cache.get_seq_length(), 0)
        restarted = model(torch.tensor([[2, 4]]), use_cache=True, past_key_values=cache)
        self.assertEqual(restarted.logits.shape[:2], (1, 2))

    def test_generate_beam_search(self):
        model = context_model().eval()
        ids = torch.tensor([[2, 3, 4]])
        generated = model.generate(ids, max_new_tokens=3, num_beams=2, use_cache=True, do_sample=False)
        self.assertGreater(generated.shape[-1], 3)

    def test_shift_candidate_set_and_unknown_masks(self):
        model = context_model()
        labels = torch.tensor([[-100, 2, 3, 4]])
        output = PGCAAuxiliaryOutput(
            logits=torch.zeros(1, 4, 16, requires_grad=True),
            phonetic_logits=torch.zeros(1, 4, 5, requires_grad=True),
            glyph_logits={key: torch.zeros(1, 4, 5, requires_grad=True) for key in GLYPH_ATTRIBUTES})
        sums = model.multitask_loss_sums(output, labels)
        self.assertEqual(sums["token_count"].item(), 3)
        self.assertEqual(sums["phon_count"].item(), 2)
        expected = -torch.log(torch.tensor(2 / 5)) - torch.log(torch.tensor(1 / 5))
        torch.testing.assert_close(sums["phon_sum"], expected)
        sums["phon_sum"].backward()
        self.assertGreater(output.phonetic_logits.grad[:, 0].abs().sum().item(), 0)
        self.assertEqual(output.phonetic_logits.grad[:, 1].abs().sum().item(), 0)
        self.assertGreater(output.phonetic_logits.grad[:, 2].abs().sum().item(), 0)
        self.assertEqual(output.phonetic_logits.grad[:, 3].abs().sum().item(), 0)

    def test_empty_supervision_has_connected_zero_gradients(self):
        model = context_model()
        ids = torch.tensor([[3, 5, 7]])
        output = model(ids, return_aux_logits=True)
        sums = model.multitask_loss_sums(output, torch.full_like(ids, -100))
        total = sums["clm_sum"] + sums["phon_sum"] + sum(sums[f"{name}_sum"] for name in GLYPH_ATTRIBUTES)
        self.assertEqual(total.item(), 0)
        total.backward()
        for parameter in model.auxiliary_heads.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(model(torch.tensor([[2]]), labels=torch.tensor([[2]])).loss.item(), 0)

    def test_accumulation_matches_large_batch(self):
        model = context_model()
        other = copy.deepcopy(model)
        ids = torch.tensor([[2, 3, 4, 2], [3, 2, 5, 4]])
        labels = ids.clone()
        labels[:, 0] = -100
        labels[1, 2] = -100
        count = labels[:, 1:].ne(-100).sum()

        def loss(current, token_ids, targets):
            sums = current.multitask_loss_sums(current(token_ids, return_aux_logits=True), targets)
            return (sums["clm_sum"] + 0.05 * sums["phon_sum"]
                    + 0.02 / 3 * sum(sums[f"{name}_sum"] for name in GLYPH_ATTRIBUTES)) / count

        loss(model, ids, labels).backward()
        for index in range(2):
            loss(other, ids[index:index + 1], labels[index:index + 1]).backward()
        for (name, first), (_, second) in zip(model.named_parameters(), other.named_parameters()):
            if first.grad is not None:
                torch.testing.assert_close(first.grad, second.grad, rtol=2e-4, atol=2e-6, msg=name)

    def test_context_checkpoint_round_trip(self):
        model = context_model().eval()
        model.config.auto_map = dict(AUTO_MAP)
        ids = torch.tensor([[2, 3, 4]])
        expected = model(ids, return_aux_logits=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_pretrained_without_auto_copy(model, path)
            copy_runtime_code(path)
            loaded = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).eval()
            actual = loaded(ids, return_aux_logits=True)
            torch.testing.assert_close(expected.logits, actual.logits)
            torch.testing.assert_close(expected.phonetic_logits, actual.phonetic_logits)


if __name__ == "__main__":
    unittest.main()
