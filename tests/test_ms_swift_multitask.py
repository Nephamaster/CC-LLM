from __future__ import annotations

import importlib.util
import json
import math
import os
import runpy
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from test_pgca_context import context_model
from src.pgca.build_pgca import AUTO_MAP, copy_runtime_code, save_pretrained_without_auto_copy


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "scripts/train/ms_swift/cpt"
PREPARE = runpy.run_path(str(BRIDGE / "prepare_data.py"))


def ddp_gradient_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = context_model()
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
        ids = torch.tensor([[2, 3, 4, 2], [3, 2, 5, 4]])
        labels = ids.clone()
        labels[:, 0] = -100
        labels[1, 2] = -100
        local_labels = labels[rank:rank + 1]
        denominator = local_labels[:, 1:].ne(-100).sum()
        dist.all_reduce(denominator)
        def combine(sums):
            return (sums["clm_sum"] + 0.05 * sums["phon_sum"] + 0.02 / 3 *
                    sum(sums[f"{name}_sum"] for name in ("stroke_count", "radical_stroke", "structure"))) / denominator
        sums = model.multitask_loss_sums(wrapped(ids[rank:rank + 1], return_aux_logits=True), local_labels)
        (2 * combine(sums)).backward()
        torch.manual_seed(42)
        reference = context_model()
        combine(reference.multitask_loss_sums(reference(ids, return_aux_logits=True), labels)).backward()
        for (name, first), (_, second) in zip(model.named_parameters(), reference.named_parameters()):
            if second.grad is not None:
                torch.testing.assert_close(first.grad, second.grad, rtol=3e-4, atol=2e-6, msg=name)
    finally:
        dist.destroy_process_group()


class PreparedDatasetTest(unittest.TestCase):
    def test_token_ranges_are_disjoint_and_drop_empty_rows(self):
        rows = {"input_ids": [[2], [2, 3, 4], [4, 3, 2, 4]],
                "labels": [[-100], [-100, 3, 4], [-100, 3, 2, 4]]}
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "raw", Path(directory) / "prepared"
            dataset = Dataset.from_dict(rows)
            dataset.save_to_disk(str(source / "train"))
            dataset.save_to_disk(str(source / "val"))
            report = PREPARE["prepare"](source, target, adaptation_tokens=3, total_tokens=5)
            self.assertEqual(report["dropped_train_rows"], 1)
            self.assertEqual(report["splits"]["train_a"]["targets"], 3)
            self.assertEqual(report["splits"]["train_b"]["targets"], 2)
            from datasets import load_from_disk
            first, second = load_from_disk(str(target / "train_a")), load_from_disk(str(target / "train_b"))
            self.assertEqual(sum(PREPARE["target_count"](row["labels"]) for row in first), 3)
            self.assertEqual(sum(PREPARE["target_count"](row["labels"]) for row in second), 2)
            def supervised_tokens(data):
                return {(tuple(row["input_ids"]), i) for row in data for i, label in enumerate(row["labels"])
                        if label != -100}
            self.assertFalse(supervised_tokens(first) & supervised_tokens(second))
            with self.assertRaises(FileExistsError):
                PREPARE["prepare"](source, target)

    @unittest.skipUnless(os.environ.get("CC_LLM_TEST_DDP") == "1", "requires local Gloo sockets")
    def test_ddp_gradients_match_single_large_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_gradient_worker, args=(str(Path(directory) / "rendezvous"),), nprocs=2)


@unittest.skipUnless(importlib.util.find_spec("swift"), "requires external ms-swift checkout")
class SwiftMultitaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = runpy.run_path(str(BRIDGE / "plugin.py"))

    def test_loss_ramp_and_shared_denominator(self):
        model = context_model()
        model.auxiliary_trained_tokens.fill_(10_000_000)
        trainer = SimpleNamespace(model=model, accelerator=SimpleNamespace(unwrap_model=lambda value: value),
                                  custom_metrics={"train": {}, "eval": {}})
        loss_fn = self.plugin["CCLLMLoss"](SimpleNamespace(), trainer)
        ids = torch.tensor([[2, 3, 4]])
        labels = torch.tensor([[-100, 3, 4]])
        with patch.dict(os.environ, CC_LLM_STAGE="phase2b"):
            output = model(ids, return_aux_logits=True)
            sums = model.multitask_loss_sums(output, labels)
            expected = (sums["clm_sum"] + 0.05 * sums["phon_sum"] + 0.02 / 3 *
                        sum(sums[f"{name}_sum"] for name in ("stroke_count", "radical_stroke", "structure"))) / 7
            actual = loss_fn(output, labels, num_items_in_batch=7)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(trainer.cc_llm_pending_tokens.item(), 2)
            model.auxiliary_trained_tokens.zero_()
            actual = loss_fn(output, labels, num_items_in_batch=7)
            torch.testing.assert_close(actual, sums["clm_sum"] / 7)
            self.assertNotIn("loss", trainer.custom_metrics["train"])

    def test_eval_loss_is_token_weighted_and_resets(self):
        for run_stage in ("phase1", "phase2a", "phase2b"):
            with self.subTest(stage=run_stage), patch.dict(os.environ, CC_LLM_STAGE=run_stage):
                model = context_model().eval()
                model.config.pgca_enabled = run_stage != "phase1"
                trainer = SimpleNamespace(
                    model=model, custom_metrics={"train": {}, "eval": {}},
                    accelerator=SimpleNamespace(unwrap_model=lambda value: value,
                                                gather_for_metrics=lambda value: value),
                )
                loss_fn = self.plugin["CCLLMLoss"](SimpleNamespace(), trainer)
                expected_sum, expected_count = 0.0, 0
                for tokens in ([2, 3, 4, 2], [4, 5]):
                    ids = torch.tensor([tokens])
                    labels = ids.clone()
                    labels[:, 0] = -100
                    with torch.no_grad():
                        output = model(ids, return_aux_logits=run_stage != "phase1")
                        sums = model.multitask_loss_sums(output, labels)
                        total = sums["clm_sum"]
                        if run_stage != "phase1":
                            total = total + 0.05 * sums["phon_sum"] + 0.02 / 3 * sum(
                                sums[f"{name}_sum"] for name in self.plugin["GLYPH_NAMES"])
                        expected_sum += total.item()
                        expected_count += sums["token_count"].item()
                        loss_fn(output, labels, num_items_in_batch=99)
                metrics = trainer.custom_metrics["eval"]
                self.assertAlmostEqual(metrics["loss"].compute(), expected_sum / expected_count, places=6)
                self.assertEqual(metrics["loss"].compute(), metrics["total_loss"].compute())
                self.assertFalse(hasattr(trainer, "cc_llm_pending_tokens"))
                metrics["loss"].reset()
                self.assertIsNone(metrics["loss"].compute())

    def test_real_swift_cpu_train_eval_save(self):
        self._run_swift_smoke(1)

    def test_real_swift_phase1_bypass_and_freeze(self):
        self._run_swift_smoke(1, "phase1")

    def test_real_swift_phase2a_frozen_backbone_checkpointing(self):
        self._run_swift_smoke(1, "phase2a")

    @unittest.skipUnless(os.environ.get("CC_LLM_TEST_DDP") == "1", "requires local Gloo sockets")
    def test_real_swift_ddp_train_eval_save(self):
        for run_stage in ("phase1", "phase2a", "phase2b"):
            with self.subTest(stage=run_stage):
                self._run_swift_smoke(2, run_stage)

    def _run_swift_smoke(self, world_size, run_stage="phase2b"):
        torch.manual_seed(42)
        model = context_model(zero_output=run_stage == "phase2a")
        if run_stage == "phase2b":
            model.auxiliary_trained_tokens.fill_(10_000_000)
        model.config.auto_map = dict(AUTO_MAP)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model"
            save_pretrained_without_auto_copy(model, model_path)
            copy_runtime_code(model_path)
            vocab = {f"word{i}": i for i in range(16)}
            vocab["<pad>"] = vocab.pop("word0")
            vocab["<unk>"] = vocab.pop("word1")
            vocab["<|endoftext|>"] = vocab.pop("word15")
            backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
                                                pad_token="<pad>", unk_token="<unk>", eos_token="<|endoftext|>")
            tokenizer.save_pretrained(model_path)
            dataset = Dataset.from_dict({"input_ids": [[2, 3, 4, 2], [4, 5, 2], [3, 2, 4, 4]],
                                         "labels": [[-100, 3, 4, 2], [-100, 5, 2], [-100, 2, 4, 4]],
                                         "lengths": [4, 3, 4], "is_padding_dummy": [False, False, False]})
            train = dataset
            if world_size > 1:
                dummy = dict(dataset[0], labels=[-100] * 4, is_padding_dummy=True)
                train = train.add_item(dummy)
            train.save_to_disk(str(root / "train"))
            dataset.save_to_disk(str(root / "val"))
            command = [sys.executable, "-B", "-m", "swift.cli.main", "pt", "--external_plugins", str(BRIDGE / "plugin.py"),
                       "--model", str(model_path), "--model_type", "qwen3", "--template", "cc_llm_multitask",
                       "--use_hf", "true", "--use_cpu", "true", "--torch_dtype", "float32", "--tuner_type", "full",
                       "--fp16", "false", "--bf16", "false", "--disable_ignore_empty_think", "true",
                       "--cached_dataset", str(root / "train"), "--cached_val_dataset", str(root / "val"),
                       "--loss_type", "cc_llm_multitask", "--optimizer", "cc_llm_multitask", "--callbacks", "cc_llm_multitask",
                       "--loss_scale", "all", "--use_chat_template", "false", "--max_length", "16", "--truncation_strategy", "split",
                       "--max_steps", "1", "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "2",
                       "--per_device_eval_batch_size", "1", "--average_tokens_across_devices", "true",
                       "--use_logits_to_keep", "false", "--padding_free", "false", "--packing", "false",
                       "--remove_unused_columns", "false",
                       "--gradient_checkpointing", "false", "--optim", "adamw_torch", "--learning_rate", "1e-5",
                       "--eval_strategy", "steps", "--eval_steps", "1", "--save_steps", "1", "--logging_steps", "1",
                       "--load_best_model_at_end", "true", "--metric_for_best_model", "eval_clm_loss",
                       "--greater_is_better", "false", "--add_version", "false", "--check_model", "false",
                       "--report_to", "none", "--dataloader_num_workers", "0", "--output_dir", str(root / "output")]
            if run_stage == "phase1":
                command.extend(["--freeze_parameters_regex", r"(^model\.layers\.[0-9]+\.pgca_attn\.|^model\.feature_memory_builder\.|^auxiliary_heads\.)"])
            elif run_stage == "phase2a":
                command.extend(["--freeze_parameters_ratio", "1", "--trainable_parameters_regex",
                                r"(^model\.layers\.[0-9]+\.pgca_attn\.|^model\.feature_memory_builder\.|^auxiliary_heads\.)"])
                command[command.index("--gradient_checkpointing") + 1] = "true"
                command.extend(["--gradient_checkpointing_kwargs", '{"use_reentrant": false}'])
            env = dict(os.environ, CC_LLM_STAGE=run_stage, PGCA_ENABLED=str(run_stage != "phase1").lower(), CUDA_VISIBLE_DEVICES="",
                       TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", MODELSCOPE_CACHE=str(root / "ms-cache"))
            for key in ("NPROC_PER_NODE", "NNODES", "NODE_RANK", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
                env.pop(key, None)
            if world_size == 1 and run_stage == "phase2b":
                raw = root / "raw.jsonl"
                raw.write_text(json.dumps({"messages": [{"role": "assistant", "content": "word2 word3 word4 word2"}]}) + "\n")
                export = [sys.executable, "-B", "-m", "swift.cli.main", "export", "--external_plugins", str(BRIDGE / "plugin.py"),
                          "--model", str(model_path), "--model_type", "qwen3", "--template", "cc_llm_multitask",
                          "--dataset", str(raw), "--val_dataset", str(raw), "--dataset_num_proc", "1",
                          "--use_hf", "true", "--to_cached_dataset", "true", "--truncation_strategy", "split",
                          "--max_length", "2", "--use_chat_template", "false", "--loss_scale", "all",
                          "--disable_ignore_empty_think", "true", "--output_dir", str(root / "exported")]
                exported = subprocess.run(export, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                self.assertEqual(exported.returncode, 0, exported.stdout[-7000:])
                report = PREPARE["prepare"](root / "exported", root / "normalized", max_length=2)
                self.assertGreater(report["train_targets"], 0)
            if world_size > 1:
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                env.update(NPROC_PER_NODE=str(world_size), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
                command.extend(["--ddp_backend", "gloo", "--ddp_find_unused_parameters", "false"])
            result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout[-7000:])
            state = json.loads((root / "output/checkpoint-1/trainer_state.json").read_text())
            evaluations = [item for item in state["log_history"] if "eval_clm_loss" in item]
            self.assertTrue(evaluations, result.stdout[-6000:])
            self.assertTrue(math.isfinite(evaluations[-1]["eval_clm_loss"]))
            self.assertEqual(evaluations[-1]["eval_clm_tokens"], 8)
            self.assertAlmostEqual(evaluations[-1]["eval_loss"], evaluations[-1]["eval_total_loss"], places=6)
            if run_stage != "phase1":
                self.assertGreater(evaluations[-1]["eval_total_loss"], evaluations[-1]["eval_clm_loss"])
            else:
                self.assertAlmostEqual(evaluations[-1]["eval_loss"], evaluations[-1]["eval_clm_loss"], places=6)
            self.assertTrue((root / "output/next_stage_model.txt").is_file())
            self.assertTrue((root / "output/checkpoint-1/cc_llm_state.json").is_file())


if __name__ == "__main__":
    unittest.main()
