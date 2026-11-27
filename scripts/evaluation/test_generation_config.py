"""Focused checks for decoding policy, effective observations and resume identity."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluation.generation_config import GENERATION_PROTOCOL, GenerationConfigLog


class GenerationConfigChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cc-generation-config-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def records(self, log):
        return [json.loads(line) for line in log.path.read_text().splitlines()]

    def test_real_squad_requests_preserve_scoring_and_stops(self):
        from lm_eval.tasks.squadv2.task import SQuAD2
        from scripts.evaluation.lmeh.squad import BudgetedSQuAD2

        doc = {"id": "fixture"}
        reference = SQuAD2.construct_requests(object.__new__(SQuAD2), doc, "Question: fixture\nAnswer:")
        actual = BudgetedSQuAD2.construct_requests(object.__new__(BudgetedSQuAD2), doc, "Question: fixture\nAnswer:")
        self.assertEqual(actual[0].args[0], reference[0].args[0])
        self.assertEqual(actual[0].args[1], {**reference[0].args[1], "max_gen_toks": 256,
                                           "do_sample": False, "temperature": 0.0})
        self.assertEqual(actual[1].args, reference[1].args)
        self.assertEqual(actual[1].request_type, "loglikelihood")

    def test_common_policy_preserves_tasks_and_checks_overflow(self):
        from lm_eval.api.instance import Instance
        from scripts.evaluation.lmeh.backend import generation_requests

        backend = SimpleNamespace(max_length=30, max_gen_toks=4, tok_encode=lambda text: list(text),
                                  instruct_protocol={})
        for seed in (None, 42):
            request = Instance("generate_until", {}, ("question", {"max_gen_toks": 6, "until": ["\n"],
                "n": 2, "seed": 7, "num_beams": 2, "num_return_sequences": 2}), idx=0)
            original = dict(request.args[1])
            generation_requests(backend, [request], vllm_seed=seed)
            options = request.args[1]
            self.assertEqual(options["max_gen_toks"], 6)
            self.assertEqual(options["until"], ["\n", "<|im_end|>", "<|endoftext|>"])
            self.assertIs(options["do_sample"], False)
            self.assertEqual(options["temperature"], 0)
            self.assertNotIn("do_sample", original)
            self.assertEqual(options.get("seed"), seed)
            if seed is None:
                self.assertNotIn("n", options)
                self.assertEqual(options["num_beams"], 1)
            else:
                self.assertNotIn("num_beams", options)
                self.assertNotIn("num_return_sequences", options)
                self.assertEqual(options["n"], 1)
        request = Instance("generate_until", {}, ("x"*28, {"max_gen_toks": 6}), idx=0)
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            generation_requests(backend, [request])

    def test_vllm_engine_and_request_seed(self):
        from lm_eval.api.instance import Instance
        from scripts.evaluation.lmeh.backend import build_vllm_backend

        class Parent:
            def __init__(self, **kwargs):
                self.model_args = kwargs
                self.max_length = kwargs["max_model_len"]
                self.max_gen_toks = 4

            def tok_encode(self, text):
                return list(text)

            def generate_until(self, requests, **kwargs):
                return [request.args[1] for request in requests]

        (self.root / "config.json").write_text('{"model_type":"qwen3"}')
        with patch.dict("sys.modules", {"lm_eval.models.vllm_causallms": SimpleNamespace(VLLM=Parent)}):
            backend = build_vllm_backend(self.root, dtype="bfloat16", batch_size=1,
                                         max_length=100, tensor_parallel_size=1)
        self.assertEqual(backend.model_args["seed"], 42)
        request = Instance("generate_until", {}, ("fixture", {"max_gen_toks": 4}), idx=0)
        options = backend.generate_until([request])[0]
        self.assertEqual(options["seed"], 42)
        self.assertEqual(options["temperature"], 0)

    def test_real_hf_config_is_captured_after_length_resolution(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                    max_position_embeddings=64, eos_token_id=2, pad_token_id=0)).eval()
        inputs = torch.tensor([[4, 5, 6], [4, 5, 7]])
        expected = model.generate(inputs, max_new_tokens=3, do_sample=False)
        log = GenerationConfigLog(self.root / "hf.jsonl", "hf")
        with log.capture(model=model, request={"batch_size": 2}):
            actual = model.generate(inputs, max_new_tokens=3, do_sample=False)
        self.assertTrue(torch.equal(expected, actual))
        effective = [row for row in self.records(log) if row["phase"] == "effective"]
        self.assertEqual(len(effective), 1)
        config = effective[0]["config"]
        self.assertEqual(config["max_length"], 6)
        self.assertEqual(config["max_new_tokens"], 3)
        self.assertIs(config["do_sample"], False)
        self.assertEqual(config["eos_token_id"], 2)
        self.assertTrue(log.summary()["all_calls_observed"])
        self.assertNotIn("_get_stopping_criteria", vars(model))

    def test_vllm_logs_defaults_after_merge_and_skips_loglikelihood(self):
        class Params:
            temperature = 0
            seed = 42
            def update_from_generation_config(self, defaults, eos_token_id=None):
                self.all_stop_token_ids = defaults["eos_token_id"]

        original_update = Params.update_from_generation_config
        def generate(requests, generate=False, sampling_params=None):
            if sampling_params:
                for params in sampling_params:
                    params.update_from_generation_config({"eos_token_id": [2, 3]})
            return "result"

        backend = SimpleNamespace(_model_generate=generate, model_args={"seed": 42})
        log = GenerationConfigLog(self.root / "vllm.jsonl", "vllm")
        with patch.dict("sys.modules", {"vllm": SimpleNamespace(SamplingParams=Params)}):
            with log.harness(backend):
                self.assertEqual(backend._model_generate([[4]], generate=False), "result")
                self.assertEqual(backend._model_generate([[4], [5]], generate=True,
                                                          sampling_params=[Params(), Params()]), "result")
        effective = [r for r in self.records(log) if r["phase"] == "effective"]
        self.assertEqual(len(effective), 2)
        self.assertTrue(all(r["config"]["all_stop_token_ids"] == [2, 3] for r in effective))
        self.assertTrue(all(r["config"]["seed"] == 42 for r in effective))
        self.assertEqual(log.summary()["generation_calls"], 1)
        self.assertIs(backend._model_generate, generate)
        self.assertIs(Params.update_from_generation_config, original_update)

    def test_missing_observation_and_failure_are_not_reported_as_passed(self):
        model = SimpleNamespace(_get_stopping_criteria=lambda **kwargs: [])
        original = model._get_stopping_criteria
        log = GenerationConfigLog(self.root / "missing.jsonl", "hf")
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with log.capture(model=model):
                raise RuntimeError("fixture")
        self.assertIs(model._get_stopping_criteria, original)
        self.assertFalse(log.summary()["all_calls_observed"])
        record = self.records(log)[-1]
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertFalse(record["effective_config_observed"])

    def test_hf_harness_hook_is_restored_on_failure(self):
        def generate(*args, **kwargs):
            raise RuntimeError("fixture")
        backend = SimpleNamespace(_model_generate=generate, model=SimpleNamespace())
        log = GenerationConfigLog(self.root / "failed-harness.jsonl", "hf")
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with log.harness(backend):
                backend._model_generate([[4]], max_length=5)
        self.assertIs(backend._model_generate, generate)
        self.assertFalse(log.summary()["all_calls_observed"])

    def test_old_prediction_protocol_is_rejected_before_reuse(self):
        from scripts.evaluation.inference import infer

        output = self.root / "predictions.jsonl"
        (self.root / "config.json").write_text('{"model_type":"qwen3"}')
        source = self.root / "input.jsonl"
        source.write_text('{"id":"fixture","task":"csc","messages":[]}\n')
        output.write_text('{"id":"fixture","predict_raw":"old"}\n')
        manifest = output.with_suffix('.manifest.json')
        manifest.write_text(json.dumps({"identity": {"decoder": "greedy"}, "complete": True}))
        with patch('scripts.evaluation.inference.training_contract', return_value={}):
            with self.assertRaisesRegex(ValueError, "start a new evaluation RUN_DIR"):
                infer(source, output, self.root, predictor=lambda rows: [])
        self.assertEqual(json.loads(output.read_text())["predict_raw"], "old")
        self.assertNotIn("generation_protocol", json.loads(manifest.read_text())["identity"])

    def test_swift_vllm_engine_request_and_recording_are_wired(self):
        import swift
        from scripts.evaluation.inference import TrainingTemplatePredictor

        class Params:
            def __init__(self, request):
                self.seed, self.temperature, self.max_tokens = request.seed, request.temperature, request.max_tokens
            def update_from_generation_config(self, defaults):
                self.all_stop_token_ids = defaults['eos_token_id']

        class Engine:
            def __init__(self, *args, **kwargs):
                self.seed = kwargs['seed']
            def _prepare_generation_config(self, request):
                return Params(request)
            def infer(self, requests, config, **kwargs):
                params = self._prepare_generation_config(config)
                params.update_from_generation_config({'eos_token_id': [2, 3]})
                return [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='fixture'),
                        finish_reason='stop')], usage=SimpleNamespace(completion_tokens=1)) for _ in requests]

        contract = {'external_plugins': [], 'model_type': 'qwen3'}
        with patch.dict(vars(swift), {'VllmEngine': Engine}), \
             patch('swift.model.get_model_processor', return_value=(None, None)), \
             patch('transformers.AutoConfig.from_pretrained', return_value=SimpleNamespace(max_position_embeddings=128)), \
             patch('scripts.evaluation.inference.training_template', return_value=object()):
            predictor = TrainingTemplatePredictor(self.root, None, contract, 2, 8, backend='vllm',
                                                 generation_log=self.root/'swift.jsonl')
        row = {'dataset': 'csc', 'messages': [{'role': 'user', 'content': 'fixture'}]}
        with patch.dict('sys.modules', {'vllm': SimpleNamespace(SamplingParams=Params)}):
            result = predictor([row, row])
        self.assertEqual(len(result), 2)
        self.assertEqual(predictor.engine.seed, 42)
        effective = [r for r in self.records(predictor.config_log) if r['phase']=='effective']
        self.assertEqual(effective[0]['config']['seed'], 42)
        self.assertEqual(effective[0]['config']['temperature'], 0)
        self.assertTrue(predictor.info['generation_trace']['all_calls_observed'])

    def test_real_swift_hf_pipeline_records_effective_config(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        from scripts.evaluation.inference import TrainingTemplatePredictor, training_contract

        tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(['fixture user assistant'], trainers.BpeTrainer(vocab_size=280,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
            special_tokens=['<unk>', '<|im_start|>', '<|im_end|>', '<|endoftext|>']))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>',
            eos_token='<|endoftext|>', pad_token='<|endoftext|>',
            additional_special_tokens=['<|im_start|>', '<|im_end|>'])
        tokenizer.save_pretrained(self.root)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=128, eos_token_id=3, pad_token_id=3)).eval()
        model.generation_config.forced_eos_token_id = 2
        model.save_pretrained(self.root)
        (self.root/'args.json').write_text(json.dumps({'template':'qwen3_nothinking', 'model_type':'qwen3',
            'use_chat_template':True, 'enable_thinking':False, 'add_non_thinking_prefix':False}))
        predictor = TrainingTemplatePredictor(self.root, None, training_contract(self.root), 1, 4,
                                             backend='hf', generation_log=self.root/'real-swift.jsonl')
        result = predictor([{'dataset':'csc', 'messages':[{'role':'user','content':'fixture'}]}])
        self.assertEqual(len(result), 1)
        effective = [r for r in self.records(predictor.config_log) if r['phase']=='effective']
        self.assertTrue(effective)
        self.assertFalse(effective[-1]['config']['do_sample'])
        self.assertEqual(effective[-1]['request']['seed'], 42)
        self.assertTrue(predictor.info['generation_trace']['all_calls_observed'])

    def test_vllm_hook_is_restored_when_generation_raises(self):
        class Params:
            def update_from_generation_config(self, defaults):
                pass
        original = Params.update_from_generation_config
        log = GenerationConfigLog(self.root/'vllm-error.jsonl', 'vllm')
        with patch.dict('sys.modules', {'vllm': SimpleNamespace(SamplingParams=Params)}):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with log.capture():
                    raise RuntimeError('fixture')
        self.assertIs(Params.update_from_generation_config, original)
        self.assertFalse(log.summary()['all_calls_observed'])

    def test_same_protocol_reuses_without_relabeling_old_outputs(self):
        from scripts.evaluation.inference import infer

        (self.root/'config.json').write_text('{"model_type":"qwen3"}')
        source, output = self.root/'input.jsonl', self.root/'predictions.jsonl'
        source.write_text('{"id":"fixture","task":"csc","messages":[]}\n')
        output.write_text('{"id":"fixture","predict_raw":"existing","error":null}\n')
        output.with_suffix('.manifest.json').write_text(json.dumps({
            'identity': {'generation_protocol': GENERATION_PROTOCOL}, 'complete':True}))
        with patch('scripts.evaluation.inference.training_contract', return_value={}):
            result = infer(source, output, self.root, predictor=lambda rows: self.fail('no inference needed'))
        self.assertEqual(result['reused'], 1)
        self.assertEqual(json.loads(output.read_text())['predict_raw'], 'existing')

    def test_interruption_keeps_current_trace_coverage_in_manifest(self):
        from scripts.evaluation.inference import infer

        source, output = self.root/'input.jsonl', self.root/'predictions.jsonl'
        source.write_text('{"id":"fixture","task":"csc","messages":[]}\n')
        class Predictor:
            info = {'generation_trace': {'generation_calls': 0}}
            def __call__(self, rows):
                self.info['generation_trace'] = {'generation_calls': 1, 'observed_calls': 1}
                raise KeyboardInterrupt()
        with patch('scripts.evaluation.inference.training_contract', return_value={}):
            with self.assertRaises(KeyboardInterrupt):
                infer(source, output, self.root, backend='hf', predictor=Predictor())
        manifest = json.loads(output.with_suffix('.manifest.json').read_text())
        self.assertFalse(manifest['complete'])
        self.assertEqual(manifest['predictor']['generation_trace']['generation_calls'], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
