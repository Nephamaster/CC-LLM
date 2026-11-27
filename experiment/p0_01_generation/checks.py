"""Focused CPU checks of token tracing and real HF/Swift entry points."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from .records import HFTrace, complete_record, observe_vllm, write_json
from .summarize import config_fields, summarize

ROOT = Path(__file__).resolve().parents[2]


class ExperimentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.temp = tempfile.TemporaryDirectory(prefix="cc-p0-01-check-")
        cls.root = Path(cls.temp.name)
        cls.model_path = cls.root / "tiny"
        cls.model_path.mkdir()
        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(["A B abcdef user assistant Context Question"], trainers.BpeTrainer(
            vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"], show_progress=False))
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                               eos_token="<|endoftext|>", pad_token="<|endoftext|>",
                                               additional_special_tokens=["<|im_start|>", "<|im_end|>"])
        cls.tokenizer.save_pretrained(cls.model_path)
        config = Qwen3Config(vocab_size=cls.tokenizer.vocab_size, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                            max_position_embeddings=128, eos_token_id=2, pad_token_id=3)
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.model.generation_config.forced_eos_token_id = 2
        cls.model.save_pretrained(cls.model_path)
        write_json(cls.model_path / "args.json", {
            "template": "cc_llm_instruction", "model_type": "qwen3", "use_chat_template": True,
            "enable_thinking": False, "add_non_thinking_prefix": False, "loss_scale": "default",
            "external_plugins": [str(ROOT / "scripts/train/downstream/plugin.py")],
        })

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_trace_preserves_actual_generate_and_restores_methods(self):
        import torch
        from transformers.generation.stopping_criteria import EosTokenCriteria

        ids = torch.tensor([[4, 5]])
        config = deepcopy(self.model.generation_config)
        config.max_new_tokens = 4
        original_stop = EosTokenCriteria.__call__
        expected = self.model.generate(input_ids=ids, generation_config=config)
        record = {}
        with HFTrace(self.model).observe(record, [4, 5]):
            actual = self.model.generate(input_ids=ids, generation_config=config)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(record["generated_ids"], actual[0, 2:].tolist())
        self.assertTrue(record["effective_config_observed"])
        self.assertEqual(record["effective_generation_config"]["max_new_tokens"], 4)
        self.assertTrue(any(e["criterion"] == "EosTokenCriteria" for e in record["stop_events"]))
        self.assertIs(EosTokenCriteria.__call__, original_stop)
        self.assertNotIn("generate", self.model.__dict__)
        self.assertNotIn("_get_stopping_criteria", self.model.__dict__)

    def test_wrong_prompt_fails_and_restores(self):
        import torch

        with self.assertRaisesRegex(ValueError, "prompt IDs differ"):
            with HFTrace(self.model).observe({}, [4, 5]):
                self.model.generate(input_ids=torch.tensor([[5, 4]]), max_new_tokens=2)
        self.assertNotIn("generate", self.model.__dict__)

    def test_budget_and_eos_can_coincide(self):
        record = {"generated_ids": [4, 5, 2], "budget": 3,
                  "stop_events": [{"criterion": "EosTokenCriteria"}]}
        complete_record(record, self.tokenizer)
        self.assertTrue(record["hit_budget"])
        self.assertEqual(record["end_token_positions"], [{"position": 2, "id": 2}])
        self.assertEqual(record["observed_finish_reason"], ["EosTokenCriteria"])

    def test_vllm_trace_preserves_arguments_and_captures_engine_update(self):
        class Params:
            max_tokens = 4
            stop = ["<|im_end|>"]
            def update_from_generation_config(self, generation_config, eos_token_id=None):
                self.stop_token_ids = [eos_token_id]

        params = Params()
        def generate(requests, sampling_params=None, **kwargs):
            self.assertEqual(requests, [[4, 5]])
            sampling_params[0].update_from_generation_config({"eos_token_id": 2}, 2)
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[4, 2], finish_reason="stop",
                                                             stop_reason=2, text="A")])]
        backend = SimpleNamespace(_model_generate=generate)
        record = {}
        with patch.dict("sys.modules", {"vllm": SimpleNamespace(SamplingParams=Params)}):
            with observe_vllm(backend, record, [4, 5]):
                backend._model_generate([[4, 5]], sampling_params=[params])
        self.assertTrue(record["effective_config_observed"])
        self.assertEqual(record["engine_sampling_updates"][0]["effective_sampling"]["stop_token_ids"], [2])
        self.assertIs(backend._model_generate, generate)

    def test_real_prepare_hf_and_swift_cells(self):
        from .worker import prepare, harness, native, swift

        folder = Path(tempfile.mkdtemp(dir=self.root))
        cases = folder / "source.jsonl"
        case = {"id": "fixture/0", "benchmark": "fixture", "reference": "A", "document": {},
                "ordinary_messages": [{"role": "user", "content": "A"}],
                "historical": {"shared": {"prompt": "A B", "available": True, "source": "fixture",
                                            "response": "A", "generation": {"max_gen_toks": 4, "until": ["<|im_end|>"]}}}}
        cases.write_text(json.dumps(case) + "\n")
        prepared = folder / "prepare"
        prepared.mkdir()
        args = SimpleNamespace(model=self.model_path, alias="shared", cases=cases, output=prepared,
                               prepared=prepared, group="prepare", device="cpu")
        prepare(args)
        for group, action in (("A", harness), ("B", native), ("C", swift), ("D", swift)):
            args.output = folder / group
            args.output.mkdir()
            args.group = group
            action(args)
            record = json.loads((args.output / "generations.jsonl").read_text())
            self.assertTrue(record["completed"], record.get("error"))
            self.assertTrue(record["prompt_ids_match"])
            self.assertTrue(record["effective_config_observed"])
            self.assertEqual(record["generated_ids"][-1], 2)
        records = [json.loads((folder / group / "generations.jsonl").read_text()) for group in ("B", "C", "D")]
        harness_record = json.loads((folder / "A/generations.jsonl").read_text())
        self.assertEqual(harness_record["actual_prompt_ids"], records[0]["actual_prompt_ids"])
        self.assertEqual(records[0]["actual_prompt_ids"], records[1]["actual_prompt_ids"])
        self.assertNotEqual(records[1]["actual_prompt_ids"], records[2]["actual_prompt_ids"])

    def test_summary_uses_latest_failed_cell(self):
        folder = Path(tempfile.mkdtemp(dir=self.root))
        (folder / "inputs").mkdir()
        (folder / "inputs/cases.jsonl").write_text('{"id":"fixture/0"}\n')
        write_json(folder / "experiment.json", {"gpu": "cpu"})
        for name, state in (("v0", "complete"), ("v1", "failed")):
            path = folder / "base/A" / name
            write_json(path / "status.json", {"state": state})
        summarize(folder)
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual(summary["cells"][0]["state"], "failed")
        self.assertFalse(summary["all_cells_completed"])
        self.assertTrue((folder / "summary.csv").exists())

    def test_configuration_comparison_preserves_unknowns(self):
        native = config_fields({"effective_generation_config": {"eos_token_id": 2, "do_sample": False}})
        engine = config_fields({"engine_sampling_updates": [{"effective_sampling": {
            "temperature": 0, "all_stop_token_ids": [2]}}]})
        self.assertEqual(native["stop_token_ids"], engine["stop_token_ids"])
        self.assertEqual(native["do_sample"], engine["do_sample"])
        self.assertIsNone(config_fields({})["do_sample"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
