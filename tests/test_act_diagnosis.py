"""Local checks of diagnostic correctness; these do not establish 1.7B GPU results."""
import json

import pytest

from scripts.evaluation.ancient.diagnostics.checks import compare_cache, eos_evidence, make_batch, select_cases


@pytest.mark.parametrize("labels,passed", [([-100, 3, 9, 4], True), ([-100, 3, -100, 4], False)])
def test_assistant_end_label_is_checked(labels, passed):
    assert eos_evidence({"input_ids": [9, 3, 9, 4], "labels": labels}, 9)["passed"] is passed


def test_user_terminator_is_not_assistant_terminator():
    ids = [9] + list(range(20, 40))
    assert not eos_evidence({"input_ids": ids, "labels": ids}, 9)["passed"]


def test_task_selection_is_reproducible_and_rejects_missing_task():
    rows = [{"task": t, "messages": [{"role": "assistant", "content": "文"}]}
            for t in ("c2m", "m2c", "punctuate") for _ in range(3)]
    encoded = [{"input_ids": [1, 2, 3]} for _ in rows]
    assert select_cases(rows, encoded, 2, 42) == select_cases(rows, encoded, 2, 42)
    assert len(select_cases(rows, encoded, 2, 42)) == 6
    with pytest.raises(ValueError, match="Not enough"):
        select_cases(rows[:6], encoded[:6], 1, 42)


def test_cache_check_with_real_small_pgca_and_nonzero_lora():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from test_pgca_context import context_model

    torch.manual_seed(42)
    model = get_peft_model(context_model(), LoraConfig(r=2, lora_alpha=4,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.02)
    model.eval()
    inputs = make_batch([[2, 3, 4, 2], [4, 2]], 0, "cpu")
    report = compare_cache(model, inputs, 2e-6, 2e-5)
    assert report["passed"]
    assert report["steps"][-1]["feature"] == report["steps"][-1]["backbone"] == 4
    # A defective feature-cache length must be detected even if logits still match.
    from unittest.mock import patch
    from scripts.evaluation.ancient.diagnostics.checks import cache_lengths

    def incorrect_length(cache):
        return {**cache_lengths(cache), "feature": 0}
    with patch("scripts.evaluation.ancient.diagnostics.checks.cache_lengths", side_effect=incorrect_length):
        assert not compare_cache(model, inputs, 2e-6, 2e-5)["passed"]


def test_actual_swift_adapter_generation_trace(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("swift")
    from peft import LoraConfig, get_peft_model
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from test_pgca_context import context_model
    from src.pgca.build_pgca import AUTO_MAP, copy_runtime_code, save_pretrained_without_auto_copy
    from scripts.evaluation.ancient.diagnostics.checks import read_contract, traced_generation
    from scripts.evaluation.inference import TrainingTemplatePredictor
    from test_finetune import SFT

    base_path, adapter_path = tmp_path / "base", tmp_path / "adapter"
    torch.manual_seed(42)
    base = context_model().eval()
    base.config.auto_map = dict(AUTO_MAP)
    base.config.max_position_embeddings = 256
    save_pretrained_without_auto_copy(base, base_path)
    copy_runtime_code(base_path)
    vocab = {f"word{i}": i for i in range(16)}
    for i, token in ((0, "<pad>"), (1, "<unk>"), (13, "<|im_start|>"), (14, "<|im_end|>"), (15, "<|endoftext|>")):
        del vocab[f"word{i}"]
        vocab[token] = i
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
                                       eos_token="<|endoftext|>", additional_special_tokens=["<|im_start|>", "<|im_end|>"])
    tokenizer.save_pretrained(base_path)
    base.name_or_path = str(base_path)
    adapter = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.02)
    adapter.save_pretrained(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = str(base_path)
    config_path.write_text(json.dumps(config))
    (adapter_path / "args.json").write_text(json.dumps({
        "template": "cc_llm_ancient", "use_chat_template": True, "enable_thinking": False,
        "add_non_thinking_prefix": True, "model_type": "qwen3",
        "external_plugins": [str(SFT / "plugin.py")], "loss_scale": "default+ignore_empty_think",
        "max_length": 128}))
    _, contract = read_contract(base_path, adapter_path)
    predictor = TrainingTemplatePredictor(base_path, adapter_path, contract, 1, 3)
    row = {"messages": [{"role": "user", "content": "word2 word3"},
                        {"role": "assistant", "content": "word4"}]}
    for disabled in (False, True):
        from contextlib import nullcontext
        with predictor.engine.model.disable_adapter() if disabled else nullcontext():
            result = traced_generation(predictor, row)
        assert result["generation_use_cache"] is True
        assert len(result["raw_token_ids"]) > 0
        assert len(result["forward_trace"]) > 0
        if len(result["forward_trace"]) > 1:
            assert result["incremental_cache_observed"] is True
    off = traced_generation(predictor, row, use_cache=False)
    assert off["generation_use_cache"] is False
    assert off["uncached_full_prefix_observed"] is True
    assert off["batch_size"] == 1 and off["padding_tokens"] == 0
    on = traced_generation(predictor, row, use_cache=True)
    assert on["generation_use_cache"] is True
    assert on["prompt_token_ids"] == off["prompt_token_ids"]
    assert predictor.engine.generation_config.use_cache is True
    # Exercise the complete reporting workflow on CPU with this disposable small model.
    # Only bypass the server-GPU preflight; all model/template/adapter computations are real.
    from types import SimpleNamespace
    from scripts.evaluation.ancient.diagnostics.checks import run
    dev = tmp_path / "dev.metadata.jsonl"
    dev.write_text("".join(json.dumps(dict(row, task=task)) + "\n" for task in ("c2m", "m2c", "punctuate")))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # The model loader must still observe a CPU-only environment.
    from unittest.mock import patch
    checks = iter([True])
    def first_preflight_only():
        return next(checks, False)
    with patch.object(torch.cuda, "is_available", side_effect=first_preflight_only), \
            patch.object(torch.cuda, "get_device_name", return_value="CPU test fixture"):
        status = run(SimpleNamespace(model=base_path, adapter=adapter_path, dev=dev, output=tmp_path / "diagnosis",
                                     samples_per_task=1, max_new_tokens=3, cache_max_tokens=16,
                                     seed=42, atol=2e-6, rtol=2e-5))
    report = json.loads((tmp_path / "diagnosis/v0/report.json").read_text())
    assert status == 0 and report["passed"]
    assert report["eos_supervision"]["counts"] == dict.fromkeys(("c2m", "m2c", "punctuate"), 1)
    assert len(report["cache_consistency"]["cases"]) == 3
    # Follow-up uses exactly the six prior cases and compares EOS with a real single token.
    from scripts.evaluation.ancient.diagnostics.followup import run as run_followup
    cases = [dict(row, task=task, id=str(i)) for i, task in enumerate(
        ("c2m", "c2m", "m2c", "m2c", "punctuate", "punctuate"))]
    dev.write_text("".join(json.dumps(c) + "\n" for c in cases))
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps([dict(c, index=i) for i, c in enumerate(cases)]))
    checks = iter([True])
    with patch.object(torch.cuda, "is_available", side_effect=first_preflight_only):
        run_followup(SimpleNamespace(model=base_path, adapter=adapter_path, dev=dev, cases=selected,
                                     output=tmp_path / "followup", candidate="word2", max_new_tokens=3))
    followup = json.loads((tmp_path / "followup/v0/report.json").read_text())
    assert followup["status"] == "completed"
    assert len(followup["end_probability"]) == 6 and len(followup["cache_on_off"]) == 3
    assert all(x["cpt"]["target_label"] == x["act"]["im_end"]["id"] == 14
               for x in followup["end_probability"])
    assert all(x["off"]["uncached_full_prefix_observed"] for x in followup["cache_on_off"])
