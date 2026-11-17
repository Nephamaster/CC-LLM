"""CPU regression tests for Shared-4L and the serial Hierarchical-12L branch."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from src.configuration_qwen3_pgca import Qwen3PGCAConfig
from src.embedding.config import EmbeddingFeatureConfig
from src.embedding.context_encoder import PGCACompositeCache
from src.modeling_qwen3_pgca import Qwen3PGCAForCausalLM
from src.pgca.build_pgca import (
    AUTO_MAP,
    build_pgca_model,
    copy_runtime_code,
    save_pretrained_without_auto_copy,
)
from src.pgca.upgrade_context import upgrade_checkpoint, upgrade_context
from src.pgca.validate_pgca import validate_pgca_model


torch.set_num_threads(1)
KEYS = ("pinyin", "shengmu", "yunmu", "tone", "stroke_count", "radical_stroke", "structure")


def config(mode="hierarchical", **changes):
    values = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=2,
        use_pgca=True,
        pgca_layers=[1, 2, 3, 4],
        pgca_feature_vocab_sizes={key: 5 for key in KEYS},
        pgca_feature_embedding_dim=4,
        pgca_max_pinyin_per_char=2,
        pgca_context_enabled=True,
        pgca_context_mode=mode,
        pgca_context_hidden_size=8,
        pgca_context_num_heads=2,
        pgca_context_intermediate_size=16,
        pgca_aux_enabled=True,
        pgca_aux_hidden_size=8,
        pgca_feature_schema_version=2,
    )
    values.update(changes)
    return Qwen3PGCAConfig(**values)


def model(mode="hierarchical", **changes):
    torch.manual_seed(42)
    result = Qwen3PGCAForCausalLM(config(mode, **changes)).eval()
    builder = result.model.feature_memory_builder
    with torch.no_grad():
        builder.is_hanzi[3:6] = True
        for name in ("pinyin_ids", "shengmu_ids", "yunmu_ids", "tone_ids"):
            getattr(builder, name)[3, :] = torch.tensor([3, 4])
            getattr(builder, name)[4, 0] = 3
        builder.pinyin_mask[3, :] = True
        builder.pinyin_mask[4, 0] = True
        for name in ("stroke_count_ids", "radical_stroke_ids", "structure_ids"):
            getattr(builder, name)[3:6] = 3
        builder.token_type_ids[3:6] = 1
        builder.feature_index_ready.fill_(True)
    return result


def tokenizer(path):
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "<|endoftext|>": 2,
        "行": 3,
        "银": 4,
        "中": 5,
        "A": 6,
        "B": 7,
        "1": 8,
        "<|im_start|>": 9,
        "<|im_end|>": 10,
        "system": 11,
        "user": 12,
        "assistant": 13,
    }
    vocab.update({f"word{i}": i for i in range(14, 32)})
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    result = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    result.save_pretrained(path)
    return result


def save_model(result, path):
    path.mkdir(parents=True, exist_ok=True)
    result.config.auto_map = dict(AUTO_MAP)
    save_pretrained_without_auto_copy(result, path)
    copy_runtime_code(path)
    tokenizer(path)


def write_char(path):
    """A complete tiny Char fixture with real V2 artifact files, not mocked build calls."""
    path.mkdir(parents=True, exist_ok=True)
    native = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=6,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            pad_token_id=0,
            eos_token_id=2,
            max_position_embeddings=128,
            pgca_context_hidden_size=8,
            pgca_context_num_heads=2,
            pgca_context_intermediate_size=16,
            pgca_aux_hidden_size=8,
        )
    )
    native.save_pretrained(path)
    tokenizer(path)
    features = path / "features"
    (features / "feature_vocabs").mkdir(parents=True)
    source = model("shared").model.feature_memory_builder
    index = {
        key: getattr(source, key)
        for key in (
            "is_hanzi",
            "pinyin_ids",
            "shengmu_ids",
            "yunmu_ids",
            "tone_ids",
            "pinyin_mask",
            "stroke_count_ids",
            "radical_stroke_ids",
            "structure_ids",
        )
    }
    torch.save(index, features / "feature_index.pt")
    for key in KEYS:
        (features / "feature_vocabs" / f"{key}_vocab.json").write_text(
            json.dumps({"<pad>": 0, "<none>": 1, "<unk>": 2, "a": 3, "b": 4})
        )
    (features / "feature_vocab_manifest.json").write_text(
        json.dumps({"schema_version": 2, "vocab_sizes": {key: 5 for key in KEYS}})
    )
    (features / "feature_index_manifest.json").write_text(
        json.dumps({"schema_version": 2, "vocab_size": 32, "max_pinyin_per_char": 2})
    )
    for name in ("char_features.json", "char_feature_index.jsonl"):
        (features / name).write_text("{}\n")
    EmbeddingFeatureConfig.from_artifacts(path, d_feat=4).save_pretrained(path / "embedding_config.json")
    return path


@pytest.mark.parametrize(
    "changes",
    [
        {"pgca_context_mode": "invalid"},
        {"use_pgca": False},
        {"pgca_context_enabled": False},
        {"pgca_context_num_layers": 2},
        {"pgca_context_stage_starts": [1, 3, 4]},
        {"pgca_context_stage_starts": [1, 1, 4]},
        {"pgca_layers": [0, 1, 5]},
        {"pgca_layers": [1, 2]},
    ],
)
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        config(**changes)


def test_midpoint_and_routing():
    result = model(pgca_layers=list(range(9, 17)), num_hidden_layers=18)
    assert result.config.pgca_context_stage_starts == [9, 12, 16]
    builder = result.model.feature_memory_builder
    assert len(builder.context_encoders) == 3
    assert all(len(encoder.layers) == 4 for encoder in builder.context_encoders)
    assert len({p.data_ptr() for e in builder.context_encoders for p in e.parameters()}) == sum(
        len(list(e.parameters())) for e in builder.context_encoders
    )
    inputs, outputs, routed, handles = [], [], {}, []
    for encoder in builder.context_encoders:
        handles.append(encoder.register_forward_pre_hook(lambda _, args: inputs.append(args[0])))
        handles.append(encoder.register_forward_hook(lambda _, args, out: outputs.append(out[0])))
    for index in result.config.pgca_layers:
        handles.append(
            result.model.layers[index].pgca_attn.register_forward_pre_hook(
                lambda _, args, i=index: routed.update({i: args[1]})
            )
        )
    try:
        with torch.no_grad():
            result(torch.tensor([[3, 4, 6, 3]]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert inputs[1] is outputs[0] and inputs[2] is outputs[1]
    assert not torch.equal(outputs[0], outputs[1]) and not torch.equal(outputs[1], outputs[2])
    for index, memory in routed.items():
        stage = 0 if index < 12 else 1 if index < 16 else 2
        torch.testing.assert_close(memory, outputs[stage], atol=0, rtol=0)
    assert all(result.model.layers[i].pgca_attn is None for i in range(9))


@pytest.mark.parametrize("mode", ["shared", "hierarchical"])
def test_causality_masks_and_missing_pinyin(mode):
    result = model(mode)
    ids = torch.tensor([[3, 4, 5, 6, 3]])
    changed = ids.clone()
    changed[:, -1] = 4
    with torch.no_grad():
        left = result(ids, use_cache=False).logits
        right = result(changed, use_cache=False).logits
        memories, mask = result.model.feature_memory_builder(
            ids, position_ids=torch.arange(5)[None], attention_mask=torch.tensor([[1, 1, 1, 1, 0]])
        )
    torch.testing.assert_close(left[:, :-1], right[:, :-1], atol=0, rtol=0)
    assert not mask[:, 3:].any() and not mask[:, 2, :-1].any() and mask[:, 2, -1].all()
    for memory in memories if isinstance(memories, tuple) else (memories,):
        assert not memory[:, 3:].any()
        assert torch.isfinite(memory).all()


def test_gradient_flow_and_identity_start():
    result = model("shared")
    ids = torch.tensor([[3, 4, 5, 6, 3, 4]])
    with torch.no_grad():
        before = result(ids, use_cache=False).logits
    original = {key: value.clone() for key, value in result.state_dict().items()}
    upgrade_context(result)
    with torch.no_grad():
        after = result(ids, use_cache=False).logits
        memories, _ = result.model.feature_memory_builder(ids, position_ids=torch.arange(6)[None])
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert torch.equal(memories[0], memories[1]) and torch.equal(memories[1], memories[2])
    for key, value in original.items():
        assert torch.equal(value, result.state_dict()[key]), key
    result.train()
    optimizer = torch.optim.AdamW(result.parameters(), lr=1e-3)
    result(ids, labels=ids, use_cache=False).loss.backward()
    for encoder in result.model.feature_memory_builder.context_refiners:
        assert encoder.output_proj.weight.grad.abs().sum() > 0
        assert encoder.input_proj.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad()
    result(ids, labels=ids, use_cache=False).loss.backward()
    for encoder in result.model.feature_memory_builder.context_encoders:
        assert encoder.input_proj.weight.grad.abs().sum() > 0
    with torch.no_grad():
        memories, _ = result.model.feature_memory_builder(ids, position_ids=torch.arange(6)[None])
        assert not torch.equal(memories[0], memories[1]) and not torch.equal(memories[1], memories[2])


def test_upgrade_preserves_feature_branch_dtype():
    result = model("shared")
    result.model.feature_memory_builder.feature_embedding.double()
    result.model.feature_memory_builder.context_encoder.double()
    ids = torch.tensor([[3, 4, 6]])
    with torch.no_grad():
        expected = result(ids, use_cache=False).logits
    original = result.model.feature_memory_builder.context_encoder
    upgrade_context(result)
    assert result.model.feature_memory_builder.context_encoder is original
    assert next(result.model.feature_memory_builder.feature_embedding.parameters()).dtype == torch.float64
    with torch.no_grad():
        torch.testing.assert_close(result(ids, use_cache=False).logits, expected, atol=0, rtol=0)


def test_in_memory_legacy_config_without_mode_fields():
    result = model("shared")
    del result.config.pgca_context_mode
    del result.config.pgca_context_stage_starts
    with torch.no_grad():
        output = result(torch.tensor([[3, 4]]), use_cache=True)
    assert len(output.past_key_values.feature_caches) == 1


def test_validator_checks_disabled_branch_without_changing_it():
    from src.pgca.validate_pgca import _check_cache

    result = model(pgca_enabled=False)
    report = _check_cache(result, {"input_ids": torch.tensor([[3, 4, 6]])})
    assert report["passed"] and report["original_pgca_enabled"] is False
    assert all(len(row["features"]) == 3 for row in report["lengths"])
    assert result.config.pgca_enabled is False


@pytest.mark.parametrize("mode", ["shared", "hierarchical"])
@pytest.mark.parametrize("padding", [False, True])
def test_incremental_and_cache_lifecycle(mode, padding):
    result = model(mode)
    ids = torch.tensor([[3, 4, 5, 6, 3, 4], [6, 3, 4, 3, 5, 6]])
    mask = torch.ones_like(ids)
    if padding:
        ids[1, :2] = 0
        mask[1, :2] = 0
    positions = mask.cumsum(-1).sub(1).clamp_min(0)
    with torch.no_grad():
        full = result(ids, attention_mask=mask, position_ids=positions, use_cache=False).logits
        output = result(ids[:, :2], attention_mask=mask[:, :2], position_ids=positions[:, :2], use_cache=True)
        actual = [output.logits]
        for start, end in [(2, 4), (4, 5), (5, 6)]:
            output = result(
                ids[:, start:end],
                attention_mask=mask[:, :end],
                position_ids=positions[:, start:end],
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            actual.append(output.logits)
            assert output.past_key_values.get_seq_length() == end
            assert [c.get_seq_length() for c in output.past_key_values.feature_caches] == [end] * (
                3 if mode == "hierarchical" else 1
            )
    actual = torch.cat(actual, dim=1)
    torch.testing.assert_close(actual, full, atol=1e-6, rtol=1e-5)
    assert torch.equal(actual.argmax(-1), full.argmax(-1))
    with torch.no_grad():
        cropped = result(ids, attention_mask=mask, position_ids=positions, use_cache=True).past_key_values
        cropped.crop(3)
        continued = result(
            ids[:, 3:5],
            attention_mask=mask[:, :5],
            position_ids=positions[:, 3:5],
            past_key_values=cropped,
            use_cache=True,
        )
        torch.testing.assert_close(continued.logits, full[:, 3:5], atol=1e-6, rtol=1e-5)
    cache = output.past_key_values
    all_caches = (cache, *cache.feature_caches)
    snapshots = [c.layers[0].keys.clone() for c in all_caches]
    cache.batch_repeat_interleave(2)
    cache.reorder_cache(torch.tensor([3, 0, 2, 1]))
    cache.batch_select_indices(torch.tensor([1, 0]))
    for c, old in zip(all_caches, snapshots):
        expected = old.repeat_interleave(2, 0)[torch.tensor([3, 0, 2, 1])][torch.tensor([1, 0])]
        assert torch.equal(c.layers[0].keys, expected)
    cache.crop(-3)
    assert cache.get_seq_length() == 3 and all(c.get_seq_length() == 3 for c in cache.feature_caches)
    with torch.no_grad():
        replayed = result(
            ids[:, 3:5],
            attention_mask=mask[:, :5],
            position_ids=positions[:, 3:5],
            past_key_values=cache,
            use_cache=True,
        )
        torch.testing.assert_close(replayed.logits, full[:, 3:5], atol=1e-6, rtol=1e-5)
    cache.reset()
    assert cache.get_seq_length() == 0 and all(c.get_seq_length() == 0 for c in cache.feature_caches)
    with torch.no_grad():
        fresh = result(
            ids[:, :2],
            attention_mask=mask[:, :2],
            position_ids=positions[:, :2],
            past_key_values=cache,
            use_cache=True,
        )
        torch.testing.assert_close(fresh.logits, full[:, :2], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["shared", "hierarchical"])
@pytest.mark.parametrize("beams", [1, 2])
def test_generation_cache_on_off(mode, beams):
    result = model(mode)
    ids = torch.tensor([[3, 4, 6]])
    args = dict(attention_mask=torch.ones_like(ids), max_new_tokens=3, do_sample=False, num_beams=beams)
    with torch.no_grad():
        cached = result.generate(ids, use_cache=True, **args)
        uncached = result.generate(ids, use_cache=False, **args)
    assert torch.equal(cached, uncached)


def test_cache_rejects_wrong_stage_count_and_training():
    result = model()
    wrong = PGCACompositeCache(config("shared"))
    with pytest.raises(ValueError, match="stage count"):
        result(torch.tensor([[3]]), past_key_values=wrong, use_cache=True)
    duplicate = PGCACompositeCache(result.config)
    duplicate.feature_caches = (duplicate.feature_cache,) * 3
    with pytest.raises(ValueError, match="independent"):
        result(torch.tensor([[3]]), past_key_values=duplicate, use_cache=True)
    result.train()
    with pytest.raises(ValueError, match="training"):
        result(torch.tensor([[3]]), use_cache=True)


@pytest.mark.parametrize("mode", ["shared", "hierarchical"])
def test_reject_unsupported_feature_masks(mode):
    result = model(mode)
    encoder = result.model.feature_memory_builder.context_encoder
    args = (
        torch.zeros(1, 2, 3, 16),
        torch.ones(1, 2, 3, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.long),
        torch.arange(2)[None],
    )
    for mask in (torch.ones(1, 1, 2, 2), {"full_attention": None}):
        with pytest.raises(ValueError, match="unpacked 2D"):
            encoder(*args, attention_mask=mask)


@pytest.mark.parametrize("tied", [False, True])
def test_checkpoint_upgrade_autoload_and_missing_weight(tmp_path, tied):
    source, upgraded = tmp_path / "shared", tmp_path / "hierarchical"
    original = model("shared", tie_word_embeddings=tied)
    save_model(original, source)
    data = json.loads((source / "config.json").read_text())
    data.pop("pgca_context_mode", None)
    data.pop("pgca_context_stage_starts", None)
    (source / "config.json").write_text(json.dumps(data))
    before = (source / "config.json").read_bytes()
    (source / "trainer_state.json").write_text('{"global_step": 6}')
    legacy = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True).eval()
    assert legacy.config.pgca_context_mode == "shared"
    report = upgrade_checkpoint(source, upgraded)
    assert report["context_stage_starts"] == [1, 2, 4]
    assert not (upgraded / "trainer_state.json").exists()
    assert (source / "trainer_state.json").exists()
    loaded = AutoModelForCausalLM.from_pretrained(upgraded, trust_remote_code=True).eval()
    assert loaded.config.pgca_context_mode == "hierarchical"
    assert AutoConfig.from_pretrained(upgraded, trust_remote_code=True).pgca_context_stage_starts == [1, 2, 4]
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        expected = legacy(ids, use_cache=False).logits
        torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected, atol=0, rtol=0)
        torch.testing.assert_close(loaded(ids, use_cache=True).logits, expected, atol=0, rtol=0)
    assert (source / "config.json").read_bytes() == before
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        assert (source / name).exists() == (upgraded / name).exists()
        if (source / name).exists():
            assert (source / name).read_bytes() == (upgraded / name).read_bytes()
    assert validate_pgca_model(upgraded, "行 银 A")["passed"]
    weights = load_file(upgraded / "model.safetensors")
    key = "model.feature_memory_builder.context_refiners.1.output_proj.weight"
    reference = weights.pop(key)
    save_file(weights, upgraded / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="missing contextual"):
        Qwen3PGCAForCausalLM.from_pretrained(upgraded)
    weights[key] = reference[:-1]
    save_file(weights, upgraded / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(RuntimeError):
        Qwen3PGCAForCausalLM.from_pretrained(upgraded, ignore_mismatched_sizes=True)


def test_from_char_build(tmp_path):
    char = write_char(tmp_path / "char")
    output = tmp_path / "hierarchical"
    report = build_pgca_model(
        char,
        output,
        "1,2,3,4",
        context_encoder=True,
        context_mode="hierarchical",
        auxiliary_heads=True,
        zero_init_output=True,
    )
    assert set(report["context_stage_parameter_counts"]) == {"C1", "C2", "C3"}
    assert report["context_stage_starts"] == [1, 2, 4]
    assert validate_pgca_model(output, "行 银 A")["passed"]
    native = Qwen3ForCausalLM.from_pretrained(char).eval()
    built = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True).eval()
    ids = torch.tensor([[3, 4, 6]])
    with torch.no_grad():
        torch.testing.assert_close(native(ids).logits, built(ids, use_cache=False).logits, atol=0, rtol=0)


@pytest.mark.parametrize("stage", ["phase1", "phase2a", "phase2b"])
@pytest.mark.parametrize("auxiliary", [True, False])
def test_cpt_freezing_and_optimizer(stage, auxiliary, monkeypatch, tmp_path):
    from transformers import TrainingArguments
    from scripts.train.cpt.plugin import (
        CCLLMCallback,
        CCLLMOptimizer,
        branch_gradient_group,
        is_branch_parameter,
    )

    monkeypatch.setenv("CC_LLM_STAGE", stage)
    monkeypatch.setenv("ENABLE_AUXILIARY_TASKS", str(auxiliary).lower())
    monkeypatch.setenv("PHASE2_B_BRANCH_LR", "0.0003")
    result = model()
    result.config.pgca_enabled = stage != "phase1"
    for name, parameter in result.named_parameters():
        parameter.requires_grad_(
            not is_branch_parameter(name)
            if stage == "phase1"
            else is_branch_parameter(name)
            if stage == "phase2a"
            else True
        )
    args = TrainingArguments(
        output_dir=str(tmp_path),
        learning_rate=1e-4,
        optim="adamw_torch",
        weight_decay=0.01,
        use_cpu=True,
        report_to=[],
        per_device_train_batch_size=1,
    )
    args.use_liger_kernel = False
    args.packing = False
    args.average_tokens_across_devices = True
    trainer = SimpleNamespace(
        model=result,
        train_dataset=list(range(4)),
        template=SimpleNamespace(padding_free=False, sequence_parallel_size=1),
        accelerator=SimpleNamespace(unwrap_model=lambda m: m, num_processes=1),
        get_decay_parameter_names=lambda m: [name for name, _ in m.named_parameters()],
    )
    callback = CCLLMCallback(args, trainer)
    callback.on_init_end(args, SimpleNamespace(), None, model=result)
    callback.on_train_begin(args, SimpleNamespace(), None, model=result)
    optimizer = CCLLMOptimizer(args, trainer).create_optimizer()
    by_parameter = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
    for name, parameter in result.named_parameters():
        if name.startswith("auxiliary_heads."):
            assert parameter.requires_grad == (auxiliary and stage != "phase1")
        assert (id(parameter) in by_parameter) == parameter.requires_grad
        if parameter.requires_grad:
            assert by_parameter[id(parameter)] == (
                3e-4 if stage == "phase2b" and is_branch_parameter(name) else 1e-4
            )
        if "context_refiners.0." in name:
            assert branch_gradient_group(name) == "context_encoder_c2"
        if "context_refiners.1." in name:
            assert branch_gradient_group(name) == "context_encoder_c3"


@pytest.mark.parametrize("stage", ["phase1", "phase2a", "phase2b", "char_cpt"])
def test_auxiliary_switch_defaults_and_values(stage, monkeypatch):
    from scripts.train.cpt.plugin import auxiliary_enabled

    monkeypatch.setenv("CC_LLM_STAGE", stage)
    monkeypatch.delenv("ENABLE_AUXILIARY_TASKS", raising=False)
    assert auxiliary_enabled() == (stage in {"phase2a", "phase2b"})
    monkeypatch.setenv("ENABLE_AUXILIARY_TASKS", "false")
    assert not auxiliary_enabled()
    monkeypatch.setenv("ENABLE_AUXILIARY_TASKS", "invalid")
    with pytest.raises(ValueError, match="ENABLE_AUXILIARY_TASKS"):
        auxiliary_enabled()


@pytest.mark.parametrize("mode", ["shared", "hierarchical"])
@pytest.mark.parametrize("stage", ["phase2a", "phase2b"])
@pytest.mark.parametrize("heads", [True, False])
def test_auxiliary_off_is_pure_clm(mode, stage, heads, monkeypatch):
    from unittest.mock import patch
    from scripts.train.cpt.plugin import CCLLMCallback, CCLLMLoss, CCLLMTemplate, is_branch_parameter

    monkeypatch.setenv("CC_LLM_STAGE", stage)
    monkeypatch.setenv("ENABLE_AUXILIARY_TASKS", "false")
    result = model(mode, pgca_aux_enabled=heads).train()
    for name, parameter in result.named_parameters():
        parameter.requires_grad_(is_branch_parameter(name) if stage == "phase2a" else True)
    trainer = SimpleNamespace(
        model=result,
        custom_metrics={"train": {}, "eval": {}},
        train_dataset=list(range(4)),
        accelerator=SimpleNamespace(
            unwrap_model=lambda m: m, num_processes=1, gather_for_metrics=lambda x: x
        ),
        template=SimpleNamespace(padding_free=False, sequence_parallel_size=1),
    )
    args = SimpleNamespace(
        use_liger_kernel=False,
        packing=False,
        average_tokens_across_devices=True,
        per_device_train_batch_size=1,
    )
    callback = CCLLMCallback(args, trainer)
    callback.on_init_end(args, SimpleNamespace(), None, model=result)
    callback.on_train_begin(args, SimpleNamespace(), None, model=result)
    loss_function = CCLLMLoss(args, trainer)
    ids = torch.tensor([[3, 4, 5, 3]])
    labels = ids.clone()
    labels[0, 2] = -100
    # If a stored head exists, prove its forward is never invoked rather than just weighting it by zero.
    guard = (
        patch.object(result.auxiliary_heads, "forward", side_effect=AssertionError("auxiliary head executed"))
        if heads
        else nullcontext()
    )
    with guard:
        output = CCLLMTemplate.compute_sft_loss(None, result, {"input_ids": ids, "labels": labels})
    assert output.phonetic_logits is None and output.glyph_logits is None
    loss = loss_function(output, labels)
    expected = (
        torch.nn.functional.cross_entropy(
            output.logits[:, :-1].reshape(-1, 32),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        / labels[:, 1:].ne(-100).sum()
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for encoder in result.model.feature_memory_builder.context_encoders:
        assert encoder.output_proj.weight.grad.abs().sum() > 0
    assert all(
        p.grad is None and not p.requires_grad
        for n, p in result.named_parameters()
        if n.startswith("auxiliary_heads.")
    )
    assert set(trainer.custom_metrics["train"]) == {"clm_loss", "clm_tokens", "total_loss"}
    assert not hasattr(trainer, "cc_llm_pending_tokens")
    if hasattr(result, "auxiliary_trained_tokens"):
        before = result.auxiliary_trained_tokens.clone()
        callback.on_step_end(args, SimpleNamespace(), None, model=result)
        assert torch.equal(before, result.auxiliary_trained_tokens)


def test_auxiliary_on_keeps_existing_weighted_loss(monkeypatch):
    from scripts.train.cpt.plugin import CCLLMLoss, CCLLMTemplate

    monkeypatch.setenv("CC_LLM_STAGE", "phase2b")
    monkeypatch.delenv("ENABLE_AUXILIARY_TASKS", raising=False)
    result = model(pgca_aux_warmup_tokens=0).train()
    trainer = SimpleNamespace(
        model=result,
        custom_metrics={"train": {}, "eval": {}},
        accelerator=SimpleNamespace(unwrap_model=lambda m: m),
    )
    ids = torch.tensor([[3, 4, 5, 3]])
    outputs = CCLLMTemplate.compute_sft_loss(None, result, {"input_ids": ids})
    assert outputs.phonetic_logits is not None
    sums = result.multitask_loss_sums(outputs, ids)
    expected = (
        sums["clm_sum"]
        + 0.03 * sums["phon_sum"]
        + 0.03 * sum(sums[f"{name}_sum"] for name in ("stroke_count", "radical_stroke", "structure")) / 3
    ) / sums["token_count"]
    torch.testing.assert_close(CCLLMLoss(None, trainer)(outputs, ids), expected)
    assert trainer.cc_llm_pending_tokens.item() == sums["token_count"].item()


@pytest.mark.parametrize("task", ["instruction", "csc"])
def test_all_linear_lora_roundtrip(task, monkeypatch, tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    from scripts.train.downstream.plugin import CorrectionCallback

    monkeypatch.setenv("CC_LLM_SFT_TASK", task)
    result = model()
    counts = [
        sum(isinstance(m, torch.nn.Linear) for m in e.modules())
        for e in result.model.feature_memory_builder.context_encoders
    ]
    adapted = get_peft_model(
        result,
        LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear", r=2, lora_alpha=4, lora_dropout=0),
    )
    callback = CorrectionCallback(
        None, SimpleNamespace(accelerator=SimpleNamespace(unwrap_model=lambda m: m))
    )
    state = SimpleNamespace(is_world_process_zero=False, log_history=[])
    callback.on_init_end(None, state, None, model=adapted)
    callback.on_train_begin(None, state, None, model=adapted)
    for encoder, expected in zip(
        adapted.base_model.model.model.feature_memory_builder.context_encoders, counts
    ):
        assert sum(hasattr(m, "lora_A") for m in encoder.modules()) == expected
    assert all(not p.requires_grad for n, p in adapted.named_parameters() if "auxiliary_heads" in n)
    ids = torch.tensor([[3, 4, 5, 3]])
    optimizer = torch.optim.AdamW((p for p in adapted.parameters() if p.requires_grad), lr=1e-3)
    adapted.train()
    adapted(ids, labels=ids, use_cache=False).loss.backward()
    optimizer.step()
    adapted.eval()
    with torch.no_grad():
        expected = adapted(ids, use_cache=False).logits
    adapted.save_pretrained(tmp_path)
    loaded = PeftModel.from_pretrained(model(), tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    os.environ.get("CC_LLM_RUN_SWIFT_SMOKE") != "1",
    reason="enable the real ms-swift CPU stage smoke explicitly",
)
def test_swift_cpu_three_phase_smoke(tmp_path):
    root = Path(__file__).resolve().parents[2]
    char = write_char(tmp_path / "char")
    initial = tmp_path / "hierarchical"
    build_pgca_model(
        char,
        initial,
        "1,2,3,4",
        context_encoder=True,
        context_mode="hierarchical",
        auxiliary_heads=True,
        zero_init_output=True,
    )
    original_aux = {
        key: value
        for key, value in load_file(initial / "model.safetensors").items()
        if key.startswith("auxiliary_heads.") or key == "auxiliary_trained_tokens"
    }
    for phase in ("phase1", "phase2"):
        data = tmp_path / phase
        for split, count in (("train", 8), ("validation", 2)):
            (data / split).mkdir(parents=True)
            (data / split / "tiny.jsonl").write_text(
                "".join(
                    json.dumps({"text": "行 银 中 A 行 银 B 中 " * 2}, ensure_ascii=False) + "\n"
                    for _ in range(count)
                )
            )
    current = initial
    # Python 3.14 defaults to forkserver; match the supported Linux/Python 3.12 worker context in this CPU test.
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / "sitecustomize.py").write_text(
        "import multiprocessing as mp\nmp.set_start_method('fork', force=True)\n"
    )
    for stage in ("phase1", "phase2_a", "phase2_b"):
        output = tmp_path / "training" / stage
        env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_ENDPOINT": "https://hf-mirror.com",
            "PYTHON_CMD": sys.executable,
            "MODEL": str(current),
            "MODEL_NAME": "tiny-PGCA",
            "PYTHONPATH": os.pathsep.join([str(startup), str(root), os.environ.get("PYTHONPATH", "")]),
            "DATA_DIR": str(tmp_path / ("phase1" if stage == "phase1" else "phase2")),
            "OUT": str(output),
            "CPT_ROOT": str(tmp_path / "training/CPT"),
            "CACHE_DIR": str(tmp_path / "cache" / stage),
            "RAW_CACHE": str(tmp_path / "raw" / ("phase1" if stage == "phase1" else "phase2")),
            "MAX_LENGTH": "32",
            "PHASE1_TOKENS": "0",
            "PHASE2_TOKENS": "0",
            "ADAPTATION_TOKENS": "16",
            "CUDA_VISIBLE_DEVICES": "",
            "NPROC_PER_NODE": "1",
            "ACCELERATE_USE_CPU": "true",
            "OMP_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRAIN_BATCH_SIZE": "1",
            "EVAL_BATCH_SIZE": "1",
            "GRAD_ACCUMULATION": "1",
            "DATASET_NUM_PROC": "1",
            "TORCH_DTYPE": "float32",
            "MAX_STEPS": "4",
            "SAVE_STEPS": "4",
            "EVAL_STEPS": "4",
            "LOGGING_STEPS": "1",
            "OPTIM": "adamw_torch",
            "SAVE_ONLY_MODEL": "true",
            "GRADIENT_CHECKPOINTING": "true",
        }
        log = tmp_path / f"{stage}.log"
        with log.open("w") as handle:
            result = subprocess.run(
                ["bash", str(root / f"scripts/train/cpt/{stage}.sh")],
                cwd=root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        assert result.returncode == 0, f"{stage} failed: {log}\n{log.read_text()[-6000:]}"
        current = Path((output / "next_stage_model.txt").read_text().strip())
        state = json.loads((current / "trainer_state.json").read_text())
        assert state["global_step"] == 4
        assert all(
            torch.isfinite(torch.tensor(row["eval_clm_loss"]))
            for row in state["log_history"]
            if "eval_clm_loss" in row
        )
        if stage != "phase1":
            for name in (
                "context_encoder_grad_norm",
                "context_encoder_c2_grad_norm",
                "context_encoder_c3_grad_norm",
            ):
                assert any(row.get(name, 0) > 0 for row in state["log_history"]), name
        saved = Qwen3PGCAForCausalLM.from_pretrained(current)
        assert len(saved.model.feature_memory_builder.context_encoders) == 3
        if os.environ.get("ENABLE_AUXILIARY_TASKS", "true").lower() == "false":
            for key, value in original_aux.items():
                assert torch.equal(value, saved.state_dict()[key]), key
            switch_state = json.loads((current / "cc_llm_state.json").read_text())
            assert switch_state["auxiliary_tasks_enabled"] is False
            assert not any(
                key in {"phon_loss", "glyph_loss", "eval_phon_loss", "eval_glyph_loss"}
                for row in state["log_history"]
                for key in row
            )
