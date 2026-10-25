from __future__ import annotations

import json
import math
import os
import runpy
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "scripts/train/ms_swift/finetune"


def option(args, name):
    return args[args.index(name) + 1]


@pytest.fixture
def shell_inputs(tmp_path):
    model = tmp_path / "base model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    for path in (train, dev):
        path.write_text('{"messages":[{"role":"user","content":"原句"},{"role":"assistant","content":"改正句"}]}\n')
    capture = tmp_path / "calls.json"
    python = tmp_path / "python"
    python.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
if sys.argv[1:3] == ["-B", "-c"]:
    code = sys.argv[3]
    sys.argv = ["-c", *sys.argv[4:]]
    exec(code)
    raise SystemExit(0)
if len(sys.argv) > 2 and sys.argv[1] == "-B" and sys.argv[2].endswith("/prepare_ancient_views.py"):
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    output.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "dev.jsonl"):
        (output / name).write_text('{"messages":[],"channel":"c2m"}\\n')
    raise SystemExit(0)
Path(os.environ["CAPTURE"]).write_text(json.dumps({"args": sys.argv[1:], "pgca": os.getenv("PGCA_ENABLED"),
    "root": os.getenv("CC_LLM_SFT_OUTPUT_ROOT"), "world": os.getenv("NPROC_PER_NODE"),
    "macro_channels": os.getenv("CC_LLM_MACRO_CHANNELS")}))
''')
    python.chmod(0o755)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for key in ("MODEL", "MODEL_NAME", "BASE_MODEL_NAME", "OUT", "TRAIN", "DEV", "MAX_LENGTH", "LEARNING_RATE",
                "NUM_TRAIN_EPOCHS", "RESUME_FROM_CHECKPOINT", "NPROC_PER_NODE", "LOAD_FROM_CACHE_FILE",
                "ADD_NON_THINKING_PREFIX", "TRUNCATION_STRATEGY", "ENABLE_CHANNEL_LOSS",
                "DISABLE_IGNORE_EMPTY_THINK", "METRIC_FOR_BEST_MODEL", "TEMPLATE_TYPE",
                "CC_LLM_MACRO_CHANNELS", "ANCIENT_DATA_DIR", "ANCIENT_VIEW_DIR"):
        env.pop(key, None)
    env.update(CC_ROOT=str(tmp_path), TRAIN_ROOT=str(tmp_path / "outputs/train"), PYTHON_CMD=str(python),
               TRAIN=str(train), DEV=str(dev), CAPTURE=str(capture), CUDA_VISIBLE_DEVICES="4,5")
    return model, capture, env


@pytest.mark.parametrize("task,suffix", [("csc", "CSC"), ("cgec", "CGEC")])
@pytest.mark.parametrize("name", ["ExampleBase", "ExampleBase-Char-ALIGN", "ExampleBase-PGCA-CPT"])
def test_scripts_use_native_sft_and_canonical_names(shell_inputs, tmp_path, task, suffix, name):
    model, capture, env = shell_inputs
    pointer = Path(env["TRAIN_ROOT"]) / name / "next_stage_model.txt"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(str(model) + "\n")
    env["MODEL_NAME"] = name
    subprocess.run(["bash", str(SFT / f"{task}.sh")], cwd=tmp_path, env=env, check=True)
    call = json.loads(capture.read_text())
    args = call["args"]
    assert args[:4] == ["-B", "-m", "swift.cli.main", "sft"]
    assert option(args, "--model") == str(model)
    assert option(args, "--output_dir") == str(Path(env["TRAIN_ROOT"]) / f"{name.removesuffix('-ALIGN')}-{suffix}")
    for flag, value in {"--tuner_type": "lora", "--target_modules": "all-linear", "--lora_rank": "16",
                        "--lora_alpha": "32", "--lora_dropout": "0.05", "--template": "cc_llm_correction",
                        "--use_chat_template": "true", "--loss_scale": "default", "--load_args": "false",
                        "--add_version": "true", "--load_from_cache_file": "true", "--split_dataset_ratio": "0"}.items():
        assert option(args, flag) == value
    assert option(args, "--dataset") == env["TRAIN"]
    assert option(args, "--val_dataset") == env["DEV"]
    assert "--cached_dataset" not in args and "--loss_type" not in args and "--optimizer" not in args
    assert "--logging_dir" not in args
    assert call["pgca"] == "true" and call["world"] == "2"
    assert call["root"] == option(args, "--output_dir")


def test_explicit_model_and_options_take_precedence(shell_inputs, tmp_path):
    model, capture, env = shell_inputs
    env.update(MODEL=str(model), MODEL_NAME="ExampleBase-Char", OUT=str(tmp_path / "custom output"),
               LOAD_FROM_CACHE_FILE="false", LORA_RANK="8", MAX_LENGTH="512", LEARNING_RATE="2e-4")
    subprocess.run(["bash", str(SFT / "cgec.sh")], env=env, check=True)
    args = json.loads(capture.read_text())["args"]
    for flag, value in {"--model": str(model), "--output_dir": env["OUT"], "--load_from_cache_file": "false",
                        "--lora_rank": "8", "--max_length": "512", "--learning_rate": "2e-4"}.items():
        assert option(args, flag) == value


def test_ancient_script_uses_task_channels_and_macro_selection(shell_inputs, tmp_path):
    model, capture, env = shell_inputs
    data_dir = tmp_path / "ancient/final"
    data_dir.mkdir(parents=True)
    env.update(MODEL=str(model), MODEL_NAME="ExampleBase-PGCA-CPT", ANCIENT_DATA_DIR=str(data_dir),
               OUT=str(tmp_path / "outputs/train/ExampleBase-PGCA-CPT-ACT"), CUDA_VISIBLE_DEVICES="4,5,6,7")
    subprocess.run(["bash", str(SFT / "ancient.sh")], env=env, check=True)
    call = json.loads(capture.read_text())
    args = call["args"]
    assert option(args, "--dataset") == str(Path(env["OUT"]) / ".inputs/train.jsonl")
    assert option(args, "--val_dataset") == str(Path(env["OUT"]) / ".inputs/dev.jsonl")
    for flag, value in {"--max_length": "2048", "--per_device_train_batch_size": "1",
                        "--gradient_accumulation_steps": "8", "--add_non_thinking_prefix": "true",
                        "--disable_ignore_empty_think": "false",
                        "--truncation_strategy": "delete", "--enable_channel_loss": "true",
                        "--metric_for_best_model": "eval_macro_task_loss",
                        "--template": "cc_llm_ancient"}.items():
        assert option(args, flag) == value
    assert call["world"] == "4"
    assert call["macro_channels"] == "c2m,m2c,punctuate"


@pytest.mark.parametrize("explicit_model", [True, False])
def test_resume_retains_base_model_and_original_version(shell_inputs, tmp_path, explicit_model):
    model, capture, env = shell_inputs
    output = tmp_path / "outputs/train/ExampleBase-CSC"
    checkpoint = output / "v0-test/checkpoint-1"
    checkpoint.mkdir(parents=True)
    for name in ("trainer_state.json", "adapter_config.json"):
        (checkpoint / name).write_text("{}")
    (checkpoint / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(model)}))
    env.update(MODEL_NAME="ExampleBase", RESUME_FROM_CHECKPOINT=str(checkpoint))
    if explicit_model:
        env["MODEL"] = str(model)
    subprocess.run(["bash", str(SFT / "csc.sh")], env=env, check=True)
    args = json.loads(capture.read_text())["args"]
    assert option(args, "--model") == str(model)
    assert option(args, "--resume_from_checkpoint") == str(checkpoint)
    assert option(args, "--output_dir") == str(checkpoint.parent)
    assert option(args, "--add_version") == "false"


def test_missing_dataset_fails_before_launch(shell_inputs):
    model, capture, env = shell_inputs
    env.update(MODEL=str(model), DEV=env["DEV"] + ".missing")
    result = subprocess.run(["bash", str(SFT / "csc.sh")], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "JSONL missing" in result.stderr
    assert not capture.exists()


@pytest.fixture
def plugin():
    pytest.importorskip("swift")
    return runpy.run_path(str(SFT / "plugin.py"))


def test_pgca_lora_has_branch_gradients_without_auxiliary_loss(plugin, tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from swift.utils import find_all_linears
    from test_pgca_context import context_model

    base = context_model()
    base.model_meta = SimpleNamespace(model_arch=None)
    base.config.pgca_enabled = False
    base.auxiliary_trained_tokens.fill_(123)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=find_all_linears(base),
                                          task_type="CAUSAL_LM", lora_dropout=0))
    trainer = SimpleNamespace(accelerator=SimpleNamespace(unwrap_model=lambda m: m))
    callback = plugin["CorrectionCallback"](SimpleNamespace(), trainer)
    callback.on_init_end(None, None, None, model=model)
    callback.on_train_begin(SimpleNamespace(output_dir=str(tmp_path)), SimpleNamespace(is_world_process_zero=True),
                            None, model=model)
    template = object.__new__(plugin["CorrectionTemplate"])
    ids = torch.tensor([[2, 3, 4, 2]])
    labels = torch.tensor([[-100, -100, 4, 2]])
    with patch.object(base.auxiliary_heads, "forward", side_effect=AssertionError("auxiliary forward")):
        outputs = template.compute_sft_loss(model, {"input_ids": ids, "labels": labels})
        outputs.loss.backward()
    assert model.config.pgca_enabled
    assert outputs.phonetic_logits is None and outputs.glyph_logits is None
    assert base.auxiliary_trained_tokens.item() == 123
    for name, parameter in model.named_parameters():
        if "auxiliary_heads" in name.split("."):
            assert not parameter.requires_grad and parameter.grad is None
        elif parameter.requires_grad:
            assert "lora_" in name
    for group in (".pgca_attn.", ".context_encoder."):
        assert any(group in n and p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters())
    model.config.pgca_enabled = False
    with patch.object(plugin["CORRECTION_META"].template_cls, "generate", return_value="generated") as generate:
        assert template.generate(model, input_ids=ids) == "generated"
        assert generate.call_args.kwargs["return_aux_logits"] is False
    assert model.config.pgca_enabled


def test_callback_adds_macro_task_loss_before_selection(plugin, monkeypatch):
    monkeypatch.setenv("CC_LLM_MACRO_CHANNELS", "c2m,m2c,punctuate")
    callback = plugin["CorrectionCallback"](SimpleNamespace(), SimpleNamespace())
    logs = {"eval_loss_c2m": 1.0, "eval_loss_m2c": 2.0, "eval_loss_punctuate": 3.0}
    state = SimpleNamespace(log_history=[dict(logs)])
    callback.on_log(None, state, None, logs=logs)
    assert logs["eval_macro_task_loss"] == 2.0
    assert state.log_history[-1]["eval_macro_task_loss"] == 2.0


@pytest.mark.parametrize("kind,world_size", [("base", 1), ("pgca", 1), ("pgca", 2)])
def test_real_swift_lora_sft_and_evaluator_contract(plugin, tmp_path, monkeypatch, kind, world_size):
    if world_size > 1 and os.getenv("CC_LLM_TEST_DDP") != "1":
        pytest.skip("requires local Gloo sockets")
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from test_pgca_context import context_model
    from src.pgca.build_pgca import AUTO_MAP, copy_runtime_code, save_pretrained_without_auto_copy
    from scripts.evaluation.inference import training_contract

    torch.manual_seed(42)
    model_path = tmp_path / "base"
    if kind == "pgca":
        model = context_model()
        model.config.pgca_enabled = False
        model.config.auto_map = dict(AUTO_MAP)
        save_pretrained_without_auto_copy(model, model_path)
        copy_runtime_code(model_path)
    else:
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                                           num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                                           max_position_embeddings=64, pad_token_id=0, eos_token_id=15))
        model.save_pretrained(model_path)
    vocab = {f"word{i}": i for i in range(16)}
    for index, token in ((0, "<pad>"), (1, "<unk>"), (13, "<|im_start|>"), (14, "<|im_end|>"), (15, "<|endoftext|>")):
        del vocab[f"word{index}"]
        vocab[token] = index
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
                                       eos_token="<|endoftext|>", additional_special_tokens=["<|im_start|>", "<|im_end|>"])
    tokenizer.save_pretrained(model_path)
    rows = [{"messages": [{"role": "user", "content": "word2 word3"},
                          {"role": "assistant", "content": text}]} for text in ("word2 word4", "word4", "word2", "word4 word2")]
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dev.write_text("".join(json.dumps(row) + "\n" for row in rows[:3]))
    root = tmp_path / "sft-output"
    command = [sys.executable, "-B", "-m", "swift.cli.main", "sft", "--external_plugins", str(SFT / "plugin.py"),
               "--model", str(model_path), "--model_type", "qwen3", "--template", "cc_llm_correction",
               "--callbacks", "cc_llm_correction", "--use_hf", "true", "--use_cpu", "true",
               "--torch_dtype", "float32", "--bf16", "false", "--fp16", "false", "--tuner_type", "lora",
               "--target_modules", "all-linear", "--lora_rank", "2", "--lora_alpha", "4", "--lora_dropout", "0",
               "--use_chat_template", "true", "--enable_thinking", "false", "--add_non_thinking_prefix", "false",
               "--disable_ignore_empty_think", "true", "--loss_scale", "default", "--check_model", "false", "--load_args", "false",
               "--dataset", str(train), "--val_dataset", str(dev), "--split_dataset_ratio", "0", "--dataset_num_proc", "1",
               "--load_from_cache_file", "false", "--max_length", "32", "--truncation_strategy", "delete", "--strict", "true",
               "--max_steps", "2", "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "1",
               "--gradient_accumulation_steps", "2", "--gradient_checkpointing", "true",
               "--gradient_checkpointing_kwargs", '{"use_reentrant": false}', "--optim", "adamw_torch",
               "--ddp_find_unused_parameters", "true", "--packing", "false", "--padding_free", "false",
               "--use_logits_to_keep", "false", "--eval_strategy", "steps", "--eval_steps", "1", "--save_steps", "1",
               "--logging_steps", "1", "--load_best_model_at_end", "true", "--metric_for_best_model", "eval_loss",
               "--greater_is_better", "false", "--report_to", "none", "--add_version", "true", "--output_dir", str(root)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CC_LLM_SFT_OUTPUT_ROOT=str(root), PGCA_ENABLED="false",
               HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
               MODELSCOPE_CACHE=str(tmp_path / "ms-cache"))
    for name in ("NPROC_PER_NODE", "NNODES", "NODE_RANK", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        env.pop(name, None)
    if world_size > 1:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env.update(NPROC_PER_NODE=str(world_size), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        command.extend(["--ddp_backend", "gloo"])
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, result.stdout[-12000:]
    checkpoint = Path((root / "next_stage_model.txt").read_text().strip())
    assert checkpoint.parent.name.startswith("v0-")
    assert (checkpoint / "adapter_config.json").is_file()
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    assert all(math.isfinite(item["eval_loss"]) for item in state["log_history"] if "eval_loss" in item)
    report = json.loads((checkpoint.parent / "finetune_report.json").read_text())
    assert report["auxiliary_loss_enabled"] is False
    assert report["pgca_enabled"] == (kind == "pgca")
    contract = training_contract(model_path, checkpoint)
    assert contract["template"] == "cc_llm_correction"
    assert contract["settings"]["use_chat_template"] is True
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(str(model_path), model_type="qwen3", load_model=False, use_hf=True)
    encoded = []
    for template_name in ("qwen3_nothinking", "cc_llm_correction"):
        template = get_template(processor, template_type=template_name, use_chat_template=True,
                                enable_thinking=False, add_non_thinking_prefix=False,
                                loss_scale="default")
        template.set_mode("train")
        encoded.append(template.encode(rows[0]))
    assert encoded[0]["input_ids"] == encoded[1]["input_ids"]
    assert encoded[0]["labels"] == encoded[1]["labels"]
    assert any(label == -100 for label in encoded[1]["labels"])
    assert any(label != -100 for label in encoded[1]["labels"])
    ancient_encoded = []
    for template_name in ("qwen3", "cc_llm_ancient"):
        template = get_template(processor, template_type=template_name, use_chat_template=True,
                                enable_thinking=False, add_non_thinking_prefix=True,
                                loss_scale="default")
        template.set_mode("train")
        ancient_encoded.append(template.encode(rows[0]))
    assert ancient_encoded[0]["input_ids"] == ancient_encoded[1]["input_ids"]
    assert ancient_encoded[0]["labels"] == ancient_encoded[1]["labels"]
    if kind == "pgca" and world_size == 1:
        from scripts.evaluation.inference import TrainingTemplatePredictor
        monkeypatch.setenv("PGCA_ENABLED", "false")
        predictor = TrainingTemplatePredictor(model_path, checkpoint, contract, batch_size=1, max_new_tokens=2)
        prediction = predictor([{"messages": [{"role": "user", "content": "word2 word3"}]}])
        assert len(prediction) == 1 and isinstance(prediction[0]["text"], str)
