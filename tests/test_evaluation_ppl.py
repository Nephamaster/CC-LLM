from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.evaluation.ppl.evaluate import checkpoint_identity, evaluate_ppl, inspect_inputs
from scripts.evaluation.ppl.metrics import aggregate, score_document, target_windows, validate_window, with_metrics


class CharacterTokenizer:
    model_max_length = 64

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False, "verbose": False}
        return {"input_ids": [ord(char) % 7 + 1 for char in text]}


class TransitionModel(torch.nn.Module):
    """Predict solely from the preceding token, so a whole-document oracle is exact."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 2)
        self.config = SimpleNamespace(max_position_embeddings=64)
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        assert not use_cache and return_dict and torch.all(attention_mask == 1)
        assert not torch.is_grad_enabled()
        self.calls.append(input_ids.tolist()[0])
        logits = -(torch.arange(8)[None, None, :] - input_ids[:, :, None]).float().square()
        return SimpleNamespace(logits=logits, loss=torch.tensor(float("nan")))


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 21, 32, 51])
def test_sliding_targets_are_shifted_and_counted_once(length):
    for window in (2, 4, 7):
        for stride in range(1, window):
            scored = []
            for begin, end, first in target_windows(length, window, stride):
                assert begin < first < end and end - begin <= window
                scored.extend(range(first, end))
            assert scored == list(range(1, length))


@pytest.mark.parametrize("window,stride", [(2, 1), (4, 3), (4, 2), (7, 1), (7, 6)])
def test_score_matches_next_token_oracle_not_model_loss(window, stride):
    model, tokenizer = TransitionModel(), CharacterTokenizer()
    text = "甲𠀀乙 丙\n丁abc🙂末尾文档"
    result = score_document(model, tokenizer, text, window, stride, loss_chunk_size=2)
    ids = torch.tensor([ord(char) % 7 + 1 for char in text])
    logits = -(torch.arange(8)[None, :] - ids[:-1, None]).float().square()
    expected = F.cross_entropy(logits, ids[1:], reduction="none").double().sum().item()
    assert result["nll"] == pytest.approx(expected, abs=1e-6)
    assert result["token_count"] == len(text) - 1
    assert result["char_count"] == len(text)
    assert result["byte_count"] == len(text.encode("utf-8"))
    assert max(map(len, model.calls)) <= window


def test_byte_tokens_do_not_duplicate_character_denominator():
    class ByteTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3, 4], "offset_mapping": [(0, 1)] * 4}
    result = score_document(TransitionModel(), ByteTokenizer(), "𠀀", window=4, stride=2)
    assert result["char_count"] == 1 and result["byte_count"] == 4
    assert result["token_count"] == 3 and result["encoded_token_count"] == 4


def test_empty_one_token_and_document_boundaries():
    model, tokenizer = TransitionModel(), CharacterTokenizer()
    empty = score_document(model, tokenizer, "", 4, 2)
    short = score_document(model, tokenizer, "字", 4, 2)
    assert not model.calls
    assert empty["empty_documents"] == 1 and short["no_target_documents"] == 1
    assert short["char_count"] == 1 and short["nll"] == 0
    docs = [score_document(model, tokenizer, text, 4, 2) for text in ("甲乙", "丙丁戊")]
    assert aggregate(docs)["token_count"] == 3
    assert [len(call) for call in model.calls] == [2, 3]


@pytest.mark.parametrize("window,stride", [(4, 4), (4, 5), (4, 0), (1, 1), (65, 2)])
def test_invalid_protocol_does_not_silently_cap_or_skip(window, stride):
    with pytest.raises(ValueError):
        validate_window(TransitionModel(), CharacterTokenizer(), window, stride)


def test_weighted_aggregation_and_overflow_are_json_safe():
    model, tokenizer = TransitionModel(), CharacterTokenizer()
    rows = [score_document(model, tokenizer, text, 4, 2) for text in ("短文", "比较长的一段文本内容")]
    result = aggregate(rows)
    assert result["token_ppl"] == pytest.approx(math.exp(sum(r["nll"] for r in rows) / result["token_count"]))
    assert result["bpc"] == pytest.approx(result["nll"] / result["char_count"] / math.log(2))
    assert result["tokens_per_char"] == 1
    assert result["scored_tokens_per_char"] < 1
    overflow = with_metrics({**rows[0], "nll": 1e6})
    assert overflow["token_ppl"] is None and "token_ppl" in overflow["ppl_overflow"]
    json.dumps(overflow, allow_nan=False)


def test_nonfinite_nll_fails_instead_of_publishing_nan():
    class BrokenModel(TransitionModel):
        def forward(self, **kwargs):
            output = super().forward(**kwargs)
            output.logits[:] = float("nan")
            return output
    with pytest.raises(ValueError, match="non-finite"):
        score_document(BrokenModel(), CharacterTokenizer(), "原始文本", 4, 2)


def write_input(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_input_inspection_is_strict_and_preserves_whitespace(tmp_path):
    path = tmp_path / "zh_test.jsonl"
    write_input(path, [{"text": " 甲\n", "domain": "test"}, {"text": ""}])
    source, = inspect_inputs(tmp_path)
    assert source["documents"] == 2 and source["char_count"] == 3
    assert source["byte_count"] == 5 and source["empty_documents"] == 1
    for bad in ({"text": 1}, {"text": "甲", "domain": "wrong"}, {"text": ""}):
        write_input(path, [bad])
        with pytest.raises(ValueError):
            inspect_inputs(tmp_path)
    path.write_text('{"text":', encoding="utf-8")
    with pytest.raises(ValueError):
        inspect_inputs(tmp_path)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(42)
    path = tmp_path / "model"
    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", model_max_length=16)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=5, hidden_size=8, intermediate_size=16,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                      head_dim=4, max_position_embeddings=16, tie_word_embeddings=True))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def test_real_hf_checkpoint_end_to_end(tmp_path, tiny_checkpoint):
    source = tmp_path / "data"
    write_input(source / "zh_alpha.jsonl", [{"id": "one", "text": "a b c d a b c d a b c"}, {"text": "d a"}])
    write_input(source / "zh_beta.jsonl", [{"text": "b c d"}])
    before = inspect_inputs(source)
    output = tmp_path / "out/metrics.json"
    result = evaluate_ppl(tiny_checkpoint, source, output, window=4, stride=2, dtype="float32", device="cpu")
    assert result["complete"] and result["overall"]["token_count"] == 13
    assert result["overall"]["encoded_token_count"] == 16
    assert result["overall"]["document_count"] == 3
    assert result["overall"]["token_ppl"] > 1
    assert inspect_inputs(source) == before
    actual = Path(result["output"])
    manifest = json.loads(actual.with_suffix(".manifest.json").read_text())
    assert manifest["complete"] and manifest["protocol"]["prepend_bos"] is False
    assert manifest["runtime"]["tokenizer_backend_sha256"]
    assert len(actual.with_suffix(".documents.jsonl").read_text().splitlines()) == 3
    before = actual.read_bytes()
    repeated = evaluate_ppl(tiny_checkpoint, source, output, window=4, stride=2, device="cpu")
    assert actual.parent.name == "v0" and Path(repeated["output"]).parent.name == "v1"
    assert actual.read_bytes() == before
    with pytest.raises(ValueError, match="outside source"):
        evaluate_ppl(tiny_checkpoint, source, source / "metrics.json")


def test_checkpoint_identity_rejects_tokenizer_only_and_adapter(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="no model weights"):
        checkpoint_identity(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="LoRA"):
        checkpoint_identity(tmp_path)


def test_context_pgca_logits_can_be_scored_without_auxiliary_loss(tmp_path, tiny_checkpoint):
    from transformers import AutoTokenizer
    from src.configuration_qwen3_pgca import Qwen3PGCAConfig, FEATURE_VOCAB_KEYS
    from src.embedding.feature_memory import empty_feature_index
    from src.modeling_qwen3_pgca import Qwen3PGCAForCausalLM
    from src.pgca.build_pgca import AUTO_MAP, copy_runtime_code, save_pretrained_without_auto_copy

    model = Qwen3PGCAForCausalLM(Qwen3PGCAConfig(
        vocab_size=8, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=16,
        use_pgca=True, pgca_layers=[0], pgca_feature_embedding_dim=4, pgca_max_pinyin_per_char=2,
        pgca_feature_vocab_sizes={key: 8 for key in FEATURE_VOCAB_KEYS},
        pgca_context_enabled=True, pgca_context_num_layers=1, pgca_context_hidden_size=8,
        pgca_context_num_heads=2, pgca_context_intermediate_size=16, pgca_aux_enabled=True,
        pgca_aux_hidden_size=4, attn_implementation="sdpa",
    )).eval()
    index = empty_feature_index(8, 2)
    for values in index.values():
        values[2] = 3 if values.dtype == torch.long else True
    model.model.feature_memory_builder.set_feature_index(index)
    model.model.feature_memory_builder.token_type_ids[2] = 1
    def unexpected_aux(*args, **kwargs):
        pytest.fail("PPL must not request auxiliary logits")
    model.auxiliary_heads.forward = unexpected_aux
    result = score_document(model, CharacterTokenizer(), "古文测试语料", 4, 2)
    assert result["token_count"] == 5 and math.isfinite(result["nll"])
    path = tmp_path / "pgca"
    model.config.auto_map = dict(AUTO_MAP)
    save_pretrained_without_auto_copy(model, path)
    copy_runtime_code(path)
    AutoTokenizer.from_pretrained(tiny_checkpoint).save_pretrained(path)
    write_input(tmp_path / "data/zh_test.jsonl", [{"text": "a b c d a b c"}])
    report = evaluate_ppl(path, tmp_path / "data", tmp_path / "out/metrics.json", 4, 2, device="cpu")
    assert report["overall"]["token_count"] == 6
    manifest = json.loads(Path(report["manifest"]).read_text())
    assert manifest["runtime"]["model_settings"]["pgca_context_enabled"] is True
    assert manifest["model"]["runtime_code_sha256"]


def test_interrupted_evaluation_leaves_diagnostics_not_success(tmp_path, tiny_checkpoint, monkeypatch):
    from scripts.evaluation.ppl import evaluate

    source = tmp_path / "data"
    write_input(source / "zh_test.jsonl", [{"text": "a b"}, {"text": "b c"}])
    original_score = evaluate.score_document
    calls = 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated failure")
        return original_score(*args, **kwargs)
    monkeypatch.setattr(evaluate, "score_document", fail_second)
    output = tmp_path / "out/metrics.json"
    with pytest.raises(RuntimeError, match="simulated"):
        evaluate_ppl(tiny_checkpoint, source, output, 4, 2, device="cpu")
    failed = output.parent / "v0" / output.name
    assert not failed.exists()
    assert json.loads(failed.with_suffix(".manifest.json").read_text())["complete"] is False
    assert len(failed.with_suffix(".documents.jsonl").read_text().splitlines()) == 1
    repeated = evaluate_ppl(tiny_checkpoint, source, output, 4, 2, device="cpu")
    assert Path(repeated["output"]).parent.name == "v1"
    assert json.loads(failed.with_suffix(".manifest.json").read_text())["complete"] is False


@pytest.mark.parametrize("overrides,relative", [
    ({"MODEL_NAME": "Example-Char-ALIGN", "RUN_NAME": "trial"},
     "outputs/evaluation/ppl/Example-Char-ALIGN/trial/metrics.json"),
    ({"OUT": "explicit-output"}, "explicit-output/metrics.json"),
])
def test_launcher_uses_active_environment_and_model_scoped_output(tmp_path, overrides, relative):
    root = Path(__file__).resolve().parents[1]
    recorder = tmp_path / "python"
    recorder.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    recorder.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in {"OUT", "MODEL_NAME", "RUN_NAME", "DATA_DIR", "WINDOW", "STRIDE", "DTYPE", "DEVICE", "ATTENTION", "LOSS_CHUNK_SIZE"}}
    env.update(PATH=f"{tmp_path}{os.pathsep}{env['PATH']}", PYTHON="/must/not/be/used",
               MODEL="/path/to/checkpoint", **overrides)
    result = subprocess.run(["bash", str(root / "scripts/evaluation/ppl/run.sh")], cwd=tmp_path,
                            env=env, check=True, capture_output=True, text=True)
    args = json.loads(result.stdout)
    assert args[:3] == ["-B", "-m", "scripts.evaluation.ppl.evaluate"]
    assert args[args.index("--input-dir") + 1] == str(root / "data/ppl")
    expected = str(root / relative) if "OUT" not in overrides else relative
    assert args[args.index("--output") + 1] == expected


@pytest.mark.parametrize("overrides", [{}, {"MODEL_NAME": "../escape"},
                                      {"MODEL_NAME": "Example", "RUN_NAME": "../escape"}])
def test_launcher_rejects_missing_or_unsafe_output_names(tmp_path, overrides):
    root = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items() if key not in {"OUT", "MODEL_NAME", "RUN_NAME"}}
    env.update(MODEL="/path/to/checkpoint", **overrides)
    result = subprocess.run(["bash", str(root / "scripts/evaluation/ppl/run.sh")], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "MODEL_NAME" in result.stderr or "single directory components" in result.stderr
