from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.diagnostics.pgca_observers import LayerObserver, gradient_comparison, gradient_probe
from scripts.diagnostics.pgca_probe import (
    batch_from_row, checkpoint_files, encoded_row, evaluate_pairs, evaluate_polyphones, git_state, load_polyphones,
    rank_readings, run, tokenizer_digest, validate_output_location,
)
from scripts.diagnostics.probe_metrics import METRICS, paired_summary, tensor_digest
from scripts.diagnostics.uniform_pgca_probe import (
    UniformAttention, load_reference, run as run_uniform, summarize as summarize_uniform,
)
from src.configuration_qwen3_pgca import FEATURE_VOCAB_KEYS, Qwen3PGCAConfig
from src.embedding.feature_memory import empty_feature_index
from src.modeling_qwen3_pgca import Qwen3PGCAForCausalLM
from src.pgca.build_pgca import AUTO_MAP, copy_runtime_code, save_pretrained_without_auto_copy


def tiny_model():
    config = Qwen3PGCAConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=32,
        use_cache=False, use_pgca=True, pgca_layers=[0, 1], pgca_feature_embedding_dim=4,
        pgca_max_pinyin_per_char=2, pgca_feature_vocab_sizes={key: 8 for key in FEATURE_VOCAB_KEYS},
        pgca_context_enabled=True, pgca_context_num_layers=1, pgca_context_hidden_size=8,
        pgca_context_num_heads=2, pgca_context_intermediate_size=16,
        pgca_aux_enabled=True, pgca_aux_hidden_size=4, attn_implementation="eager",
        pad_token_id=0, eos_token_id=15,
    )
    config.pgca_auxiliary_class_names = {"pinyin": ["zhong1", "zhong4", "x1", "x2", "x3"]}
    model = Qwen3PGCAForCausalLM(config).eval()
    index = empty_feature_index(16, 2)
    for values in index.values():
        values[2] = 3 if values.dtype == torch.long else True
    index["pinyin_ids"][2, 1] = 4
    builder = model.model.feature_memory_builder
    builder.set_feature_index(index)
    builder.token_type_ids[index["is_hanzi"]] = 1
    return model


def fixture_row(ids=(3, 2, 3, 2)):
    return encoded_row({"input_ids": list(ids), "labels": [-100, *ids[1:]]})


class ProbeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_encoded_row_keeps_boundary_labels_and_rejects_dummy(self):
        original = {"input_ids": [2, 3, 2], "labels": [-100, -100, 2]}
        snapshot = copy.deepcopy(original)
        result = encoded_row(original)
        self.assertEqual(original, snapshot)
        self.assertEqual(result["labels"], original["labels"])
        with self.assertRaisesRegex(ValueError, "zero-target"):
            encoded_row({"input_ids": [2, 3], "labels": [-100, -100]})
        with self.assertRaisesRegex(ValueError, "length"):
            encoded_row({"input_ids": [2, 3], "labels": [-100, 2, 3]})

    def test_weighted_paired_bootstrap_and_empty_auxiliary_counts(self):
        samples = []
        for count, on_loss in ((1, 1.), (9, 4.)):
            on, off = {}, {}
            for numerator, denominator in METRICS.values():
                on[numerator] = off[numerator] = 0
                on[denominator] = off[denominator] = 0
            on.update(clm_sum=on_loss * count, token_count=count, clm_correct=0)
            off.update(clm_sum=(on_loss + .5) * count, token_count=count, clm_correct=0)
            samples.append({"on": on, "off": off})
        result = paired_summary(samples, bootstrap=32)
        self.assertAlmostEqual(result["clm_nll"]["on"], 3.7)
        self.assertEqual(result["clm_nll"]["paired_bootstrap_ci95"], [.5, .5])
        self.assertIsNone(result["phonetic_accuracy"]["delta"])
        self.assertIsNone(result["phonetic_accuracy"]["paired_bootstrap_ci95"])
        samples[0]["off"]["token_count"] = 2
        with self.assertRaisesRegex(ValueError, "denominator mismatch"):
            paired_summary(samples, bootstrap=4)

    def test_on_off_repeat_hooks_and_state_unchanged(self):
        model = tiny_model()
        before = tensor_digest(model.state_dict().items())
        rows = [fixture_row(), fixture_row((2, 3, 2))]
        samples, layers, repeat = evaluate_pairs(model, rows, [4, 7], "cpu", [0, 1])
        self.assertEqual(len(samples), 2)
        self.assertTrue(repeat["passed"])
        self.assertEqual(before, tensor_digest(model.state_dict().items()))
        self.assertTrue(model.config.pgca_enabled)
        for values in layers.values():
            self.assertEqual(values["active_tokens"], 4)
            self.assertAlmostEqual(values["phonetic_mass"]["mean"] + values["glyph_mass"]["mean"], 1., places=6)
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_gradient_comparison_zero_and_unused_coordinates(self):
        value = gradient_comparison({"a": torch.ones(2)}, {}, ["a"])
        self.assertEqual(value["gradient_norm"], 0)
        self.assertIsNone(value["cosine_with_clm"])
        value = gradient_comparison({"a": torch.ones(2)}, {"a": -torch.ones(2)}, ["a"])
        self.assertAlmostEqual(value["cosine_with_clm"], -1)

    def test_three_losses_reach_branch_without_changing_weights(self):
        model = tiny_model()
        before = tensor_digest(model.state_dict().items())
        inputs, labels = batch_from_row(fixture_row(), "cpu")
        report = gradient_probe(model, inputs, labels)
        for task in ("clm", "phonetic", "glyph"):
            self.assertGreater(report["tasks"][task]["groups"]["shared_pgca"]["gradient_norm"], 0)
        self.assertEqual(report["tasks"]["clm"]["groups"]["auxiliary_phonetic"]["gradient_norm"], 0)
        self.assertIsNone(report["tasks"]["clm"]["groups"]["auxiliary_phonetic"]["cosine_with_clm"])
        self.assertEqual(before, tensor_digest(model.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_no_hanzi_means_zero_auxiliary_gradients(self):
        inputs, labels = batch_from_row(fixture_row((3, 5, 7)), "cpu")
        report = gradient_probe(tiny_model(), inputs, labels)
        self.assertIn("phon_count", report["missing_auxiliary_supervision"])
        for task in ("phonetic", "glyph"):
            self.assertEqual(report["tasks"][task]["loss"], 0)
            for group in report["tasks"][task]["groups"].values():
                self.assertEqual(group["gradient_norm"], 0)
                self.assertIsNone(group["cosine_with_clm"])

    def test_candidate_deduplication_ties_and_missing_gold(self):
        names = ["a1", "a2"]
        result = rank_readings([.2, .3, .4], [3, 3, 4], [True] * 3, names, "a1")
        self.assertTrue(result["top1_correct"])
        result = rank_readings([.5, .5], [3, 4], [True, True], names, "a1")
        self.assertFalse(result["top1_correct"])
        self.assertEqual(result["top1_tie_count"], 2)
        self.assertFalse(rank_readings([1., 0], [3, 4], [True, False], names, "a1")["eligible"])
        self.assertFalse(rank_readings([.5, .5], [3, 4], [True, True], names, "b1")["eligible"])

    def test_polyphones_use_prefix_only_and_exclude_unreviewed(self):
        class Tokenizer:
            def __call__(self, text, **_):
                self.seen.append(text)
                return {"input_ids": [3, 2], "offset_mapping": [(0, 1), (1, 2)]}
        tokenizer = Tokenizer()
        tokenizer.seen = []
        cases = [{"id": str(i), "left_context": "a", "target": "中", "pinyin": "zhong1", "reviewed": flag}
                 for i, flag in enumerate([False, True])]
        samples, report = evaluate_polyphones(tiny_model(), tokenizer, cases, "cpu", [0, 1])
        self.assertEqual(tokenizer.seen, ["a中", "a中"])
        self.assertEqual(report["pending_review"], 1)
        self.assertEqual(report["layers"]["0"]["eligible_reviewed"], 1)
        self.assertFalse(samples[0]["layers"]["0"]["included_in_formal_metric"])

    def test_review_set_has_100_distinct_cases_and_explicit_review_status(self):
        rows = load_polyphones(Path("scripts/diagnostics/polyphones_review.tsv"))
        self.assertEqual(len(rows), 100)
        self.assertTrue(all(isinstance(row["reviewed"], bool) for row in rows))
        self.assertEqual(len({row["target"] for row in rows}), 50)
        for target in {row["target"] for row in rows}:
            self.assertEqual(len({row["pinyin"] for row in rows if row["target"] == target}), 2)

    def test_outputs_cannot_overlap_input(self):
        with self.assertRaises(ValueError):
            validate_output_location(Path("/tmp/model/probe"), [Path("/tmp/model")])
        with self.assertRaises(ValueError):
            validate_output_location(Path("/tmp"), [Path("/tmp/model")])

    def test_git_metadata_when_available(self):
        with patch("scripts.diagnostics.pgca_probe.subprocess.check_output",
                   side_effect=["abc123\n", " M file.py\n"]) as query:
            result = git_state()
        self.assertTrue(result["available"])
        self.assertEqual(result["commit"], "abc123")
        self.assertEqual(result["status"], "M file.py")
        self.assertNotIn("tracked_diff_sha256", result)
        self.assertEqual([call.args[0] for call in query.call_args_list],
                         [["git", "rev-parse", "HEAD"], ["git", "status", "--short"]])

    def test_optional_git_failures_do_not_block_provenance(self):
        for error in (FileNotFoundError("git executable missing"),
                      subprocess.CalledProcessError(128, ["git"], stderr="fatal: not a git repository\n"),
                      subprocess.TimeoutExpired(["git", "status"], 10)):
            with self.subTest(error=type(error).__name__):
                with patch("scripts.diagnostics.pgca_probe.subprocess.check_output", side_effect=error):
                    result = git_state()
                self.assertFalse(result["available"])
                self.assertIsNone(result["commit"])
                self.assertIsNone(result["status"])
                self.assertNotIn("tracked_diff_sha256", result)
                self.assertTrue(result["reason"])

    def test_checkpoint_hashes_never_read_source_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("model.safetensors", "config.json", "modeling.py", "configuration.py", "run.sh"):
                (root / name).touch()
            with patch("scripts.diagnostics.pgca_probe.file_digest", return_value="artifact-hash") as digest:
                result = checkpoint_files(root)
            self.assertEqual(set(result), {"model.safetensors", "config.json"})
            self.assertEqual({call.args[0].name for call in digest.call_args_list}, set(result))

    def test_uniform_preserves_values_masks_and_output_projection(self):
        for dtype in (torch.float32, torch.bfloat16):
            for slots in (1, 4, 17):
                with self.subTest(dtype=dtype, slots=slots), torch.no_grad():
                    model = tiny_model().to(dtype)
                    attention = model.model.layers[0].pgca_attn
                    attention.o_proj.bias = torch.nn.Parameter(torch.ones(8, dtype=dtype))
                    hidden = torch.randn(1, 3, 8, dtype=dtype)
                    memory = torch.randn(1, 3, slots, 8, dtype=dtype)
                    mask = torch.zeros(1, 3, slots, dtype=torch.bool)
                    mask[:, 1, 0] = True
                    mask[:, 2] = True
                    old_memory, old_mask = memory.clone(), mask.clone()
                    before = tensor_digest(model.state_dict().items())
                    uniform = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
                    weights = uniform.unsqueeze(2).expand(1, 3, attention.num_heads, slots).to(dtype)
                    values = attention._repeat_kv(attention.v_proj(memory).view(
                        1, 3, slots, attention.num_key_value_heads, attention.head_dim))
                    expected = attention.o_proj(
                        torch.einsum("bshm,bsmhd->bshd", weights, values).reshape(1, 3, 8))
                    expected = expected.masked_fill(~mask.any(-1, keepdim=True), 0)
                    seen = []
                    with UniformAttention(model, [0]) as intervention:
                        handle = attention.dropout.register_forward_hook(
                            lambda module, inputs, output: seen.append(output.clone()))
                        try:
                            actual = attention(hidden, feature_memory=memory, feature_mask=mask)
                        finally:
                            handle.remove()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(seen[0], weights, rtol=0, atol=0)
                    self.assertEqual(intervention.calls, {0: 1})
                    self.assertEqual(actual[:, 0].abs().sum(), 0)
                    self.assertTrue(torch.equal(memory, old_memory))
                    self.assertTrue(torch.equal(mask, old_mask))
                    self.assertEqual(before, tensor_digest(model.state_dict().items()))

    def test_uniform_handles_absent_mask_and_ignores_qk_scores(self):
        model = tiny_model()
        attention = model.model.layers[0].pgca_attn
        memory, hidden = torch.randn(1, 2, 3, 8), torch.randn(1, 2, 8)
        seen = []
        with torch.no_grad(), UniformAttention(model, [0]):
            handle = attention.dropout.register_forward_hook(lambda m, i, o: seen.append(o.clone()))
            try:
                first = attention(hidden, memory, None)
                attention.q_proj.weight.fill_(2)
                attention.k_proj.weight.fill_(-3)
                second = attention(hidden * 4, memory, None)
            finally:
                handle.remove()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(seen[0], torch.full_like(seen[0], 1 / 3))

    def test_uniform_restores_hooks_after_exception_and_rejects_training(self):
        model = tiny_model()
        before = model(torch.tensor([[3, 2, 3, 2]])).logits
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with UniformAttention(model, [0, 1]):
                model(torch.tensor([[3, 2, 3, 2]]))
                raise RuntimeError("intentional")
        after = model(torch.tensor([[3, 2, 3, 2]])).logits
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        with self.assertRaisesRegex(ValueError, "eval-only"):
            with UniformAttention(model.train(), [0, 1]):
                pass

    def test_uniform_leaves_padding_and_non_hanzi_without_feature_attention(self):
        model = tiny_model()
        seen = []
        ids = torch.tensor([[0, 3, 2]])
        with torch.no_grad(), UniformAttention(model, [0, 1]):
            handle = model.model.layers[0].pgca_attn.dropout.register_forward_hook(
                lambda m, i, o: seen.append(o.clone()))
            try:
                model(ids, attention_mask=torch.tensor([[0, 1, 1]]))
            finally:
                handle.remove()
        self.assertEqual(seen[0][:, :2].abs().sum(), 0)
        torch.testing.assert_close(seen[0][:, 2].sum(-1), torch.ones(1, 2))

    def test_uniform_summary_recomputes_paired_subset_ratios(self):
        samples = []
        for count, loss in ((1, 1.), (9, 4.)):
            samples.append({name: {"clm_sum": (loss + offset) * count, "token_count": count}
                            for name, offset in (("normal", 0), ("uniform", 1), ("off", 3))})
        result = summarize_uniform(samples, seed=42, bootstrap=32)
        self.assertAlmostEqual(result["normal_nll"], 3.7)
        self.assertAlmostEqual(result["uniform_nll"], 4.7)
        self.assertAlmostEqual(result["off_nll"], 6.7)
        for name, value in (("uniform_minus_normal", 1.), ("off_minus_normal", 3.), ("off_minus_uniform", 2.)):
            self.assertEqual(result["paired_comparisons"][name]["paired_bootstrap_ci95"], [value, value])
        samples[0]["uniform"]["token_count"] = 2
        with self.assertRaisesRegex(ValueError, "target counts differ"):
            summarize_uniform(samples, seed=42, bootstrap=4)

    def test_end_to_end_saved_runtime_and_reproducible_summary(self):
        from datasets import Dataset
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-11500"
            model = tiny_model()
            model.config.auto_map = dict(AUTO_MAP)
            save_pretrained_without_auto_copy(model, checkpoint)
            copy_runtime_code(checkpoint)
            vocabulary = {"[PAD]": 0, "[UNK]": 1, "中": 2, "a": 3}
            raw = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
            raw.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]")
            tokenizer.save_pretrained(checkpoint)
            cache = root / "cache"
            dataset = Dataset.from_list([fixture_row(), fixture_row((2, 3, 2)), fixture_row((3, 5, 7))])
            for split in ("val", "train_b"):
                dataset.save_to_disk(str(cache / split))
            (cache / "preparation_report.json").write_text(json.dumps({
                "tokenizer_sha256": tokenizer_digest(checkpoint), "max_length": 32,
            }))
            cases = root / "polyphones.tsv"
            cases.write_text("id\tleft_context\ttarget\tpinyin\treviewed\n1\ta\t中\tzhong1\tfalse\n")
            args = argparse.Namespace(model=checkpoint, cache=cache, output=root / "probe1",
                                      polyphones=cases, samples=3, bootstrap=16, seed=42,
                                      gradient_row_index=0, layers="0,1", device="cpu", dtype="float32",
                                      attention="eager")
            # The server deploys a source copy without .git: exercise that real layout.
            with patch("scripts.diagnostics.pgca_probe.ROOT", root):
                run(args)
                args.output = root / "probe2"
                run(args)
            manifest = json.loads((root / "probe1/probe_manifest.json").read_text())
            self.assertFalse(manifest["git"]["available"])
            self.assertIsNone(manifest["git"]["commit"])
            self.assertTrue(manifest["git"]["reason"])
            self.assertNotIn("diagnostic_code_sha256", manifest)
            self.assertTrue(manifest["checkpoint_files_sha256"])
            self.assertFalse(any(name.endswith((".py", ".sh"))
                                 for name in manifest["checkpoint_files_sha256"]))
            first = json.loads((root / "probe1/probe_summary.json").read_text())
            second = json.loads((root / "probe2/probe_summary.json").read_text())
            self.assertEqual(first, second)
            self.assertTrue(first["weights_unchanged"])
            self.assertEqual(first["status"], "partial")
            self.assertIn("polyphone_review", first["pending_acceptance"])
            self.assertIsNone(first["polyphones"]["layers"]["0"]["polyphone_top1"])
            self.assertEqual((root / "probe1/gradient_alignment.json").read_bytes(),
                             (root / "probe2/gradient_alignment.json").read_bytes())
            self.assertEqual((root / "probe1/probe_samples.jsonl").read_bytes(),
                             (root / "probe2/probe_samples.jsonl").read_bytes())
            with self.assertRaises(FileExistsError):
                run(args)
            p0 = root / "probe1"
            p0_bytes = {path.name: path.read_bytes() for path in p0.iterdir()}
            uniform_args = argparse.Namespace(p0=p0, model=None, output=root / "uniform1",
                                              device="cpu", limit=0)
            run_uniform(uniform_args)
            uniform_args.output = root / "uniform2"
            run_uniform(uniform_args)
            result = json.loads((root / "uniform1/uniform_summary.json").read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["validation_sequences"], 2)
            self.assertEqual(result["effective_targets"], 5)
            self.assertEqual(result["checks"]["uniform_hook_calls"], {"0": 3, "1": 3})
            self.assertTrue(result["weights_unchanged"])
            self.assertTrue(result["checks"]["normal_restored_after_hooks"])
            for name in ("uniform_summary.json", "uniform_samples.jsonl"):
                self.assertEqual((root / "uniform1" / name).read_bytes(), (root / "uniform2" / name).read_bytes())
            self.assertEqual(p0_bytes, {path.name: path.read_bytes() for path in p0.iterdir()})
            _, selected, eligible = load_reference(p0, limit=1)
            self.assertEqual(len(selected), 1)
            self.assertEqual(eligible, 2)
            with self.assertRaises(FileExistsError):
                run_uniform(uniform_args)
            uniform_args.output = root / "mismatch"
            with patch("scripts.diagnostics.uniform_pgca_probe.tensor_digest", return_value="wrong"):
                with self.assertRaisesRegex(ValueError, "model state differs from P0"):
                    run_uniform(uniform_args)
            self.assertFalse(uniform_args.output.exists())
            sample_file = p0 / "probe_samples.jsonl"
            sample_rows = [json.loads(line) for line in sample_file.read_text().splitlines()]
            sample_rows[0]["input_sha256"] = "changed"
            sample_file.write_text("".join(json.dumps(row) + "\n" for row in sample_rows))
            with self.assertRaisesRegex(ValueError, "input/labels changed"):
                load_reference(p0)


if __name__ == "__main__":
    unittest.main()
