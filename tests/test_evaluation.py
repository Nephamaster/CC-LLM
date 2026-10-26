from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import scripts.evaluation.inference as inference_module
from scripts.evaluation.common import aligned_predictions, iter_jsonl, prepare_file, read_journal, signature, write_json
from scripts.evaluation.inference import enable_generation_cache, infer, training_contract
from scripts.evaluation.cgec.prepare import fcgec_references, fcgec_rows, nacgec_rows
from scripts.evaluation.cgec.score import score as score_cgec
from scripts.evaluation.ancient.metrics import c3_metrics, ci_format, couplet, fuxi_metrics
from scripts.evaluation.ancient.judge import judge_file, parse_judgment
from scripts.evaluation.gec_metrics import reference_edits


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sample(uid="1", text="原句"):
    return {"id": uid, "dataset": "cscd_ns", "subset": "all", "task": "csc", "src": text, "tgt": "正确句",
            "messages": [{"role": "user", "content": text}]}


@pytest.fixture
def model_dir(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "args.json", {"template": "qwen3_nothinking", "use_chat_template": True,
                                      "enable_thinking": False, "hub_token": "DO_NOT_LEAK"})
    return model


def test_template_is_required_and_not_overridable(model_dir, tmp_path):
    contract = training_contract(model_dir)
    assert contract["template"] == "qwen3_nothinking"
    assert "DO_NOT_LEAK" not in str(contract)
    write_json(model_dir / "args.json", {"template": "qwen3", "use_chat_template": False})
    with pytest.raises(ValueError, match="CPT"):
        training_contract(model_dir)
    with pytest.raises(FileNotFoundError, match="args.json"):
        training_contract(tmp_path / "unknown" / "model")


def test_lora_uses_adapter_training_args(model_dir, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    write_json(adapter / "args.json", {"template": "qwen3", "use_chat_template": True, "enable_thinking": False})
    assert training_contract(model_dir, adapter)["template"] == "qwen3"


def test_generation_cache_is_enabled_for_model_and_generation_config():
    from types import SimpleNamespace

    model = SimpleNamespace(config=SimpleNamespace(use_cache=False),
                            generation_config=SimpleNamespace(use_cache=False))
    enable_generation_cache(model)
    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True


def test_prepare_is_non_destructive_and_checks_identity(tmp_path):
    source = tmp_path / "source.jsonl"
    jsonl(source, [sample()])
    out = tmp_path / "input.jsonl"
    first = prepare_file(iter_jsonl(source), out, [source], "fixture")
    assert prepare_file(iter_jsonl(source), out, [source], "fixture") == first
    assert list(iter_jsonl(source)) == [sample()]
    with pytest.raises(ValueError, match="changed"):
        prepare_file(iter_jsonl(source), out, [source], "different")


def test_infer_resume_and_metadata_alignment(model_dir, tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "prediction.jsonl"
    jsonl(source, [sample("1"), sample("2")])
    called = []

    def predictor(rows):
        called.extend(row["id"] for row in rows)
        return [{"text": "输出：正确句\n解释内容"} for row in rows]

    assert infer(source, output, model_dir, predictor=predictor)["complete"]
    assert called == ["1", "2"]
    assert infer(source, output, model_dir, predictor=predictor)["reused"] == 2
    originals, results = aligned_predictions(source, output)
    assert [r["predict_clean"] for r in results] == ["正确句", "正确句"]
    assert "DO_NOT_LEAK" not in output.with_suffix(".manifest.json").read_text()
    assert json.loads(output.with_suffix(".manifest.json").read_text())["identity"]["use_cache"] is True
    with pytest.raises(ValueError, match="identity"):
        infer(source, output, model_dir, max_new_tokens=99, predictor=predictor)


def test_inference_progress_counts_reused_and_new_samples(model_dir, tmp_path, monkeypatch):
    source, output = tmp_path / "input.jsonl", tmp_path / "prediction.jsonl"
    jsonl(source, [sample("1"), sample("2"), sample("3")])
    events = {}

    class Progress:
        def __init__(self, *, total, initial, **kwargs):
            events.update(total=total, initial=initial, updates=[], closed=False)

        def update(self, count):
            events["updates"].append(count)

        def set_postfix(self, **kwargs):
            events["postfix"] = kwargs

        def close(self):
            events["closed"] = True

    monkeypatch.setattr(inference_module, "tqdm", Progress)
    predictor = lambda rows: [{"text": "正确句"} for _ in rows]
    assert infer(source, output, model_dir, batch_size=2, predictor=predictor)["complete"]
    assert events == {"total": 3, "initial": 0, "updates": [2, 1], "closed": True,
                      "postfix": {"errors": 0}}


def test_inference_errors_are_retried_not_scored(model_dir, tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "prediction.jsonl"
    jsonl(source, [sample()])

    def bad(rows):
        raise RuntimeError("fixture error")

    assert not infer(source, output, model_dir, predictor=bad)["complete"]
    with pytest.raises(ValueError, match="inference error"):
        aligned_predictions(source, output)
    assert infer(source, output, model_dir, predictor=lambda rs: [{"text": "正确句"} for _ in rs])["complete"]
    assert list(iter_jsonl(output))[0]["error"] is None


def test_journal_cannot_be_reused_with_changed_checkpoint(model_dir, tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "prediction.jsonl"
    jsonl(source, [sample()])
    infer(source, output, model_dir, predictor=lambda rs: [{"text": "正确句"}])
    output.unlink()  # Simulated interrupted publication, journal remains.
    write_json(model_dir / "config.json", {"changed": True})
    with pytest.raises(ValueError, match="identity"):
        infer(source, output, model_dir, predictor=lambda rs: [])


def test_journal_recovers_only_an_incomplete_tail(tmp_path):
    path = tmp_path / "retry.jsonl"
    path.write_bytes(b'{"id":1}\n{"id":')
    assert list(read_journal(path)) == [{"id": 1}]
    assert path.read_bytes() == b'{"id":1}\n'
    path.write_bytes(b'broken\n{"id":1}\n')
    with pytest.raises(ValueError, match="before EOF"):
        list(read_journal(path))


def test_alignment_rejects_partial_or_changed_predictions(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "prediction.jsonl"
    jsonl(source, [sample()])
    jsonl(output, [])
    with pytest.raises(ValueError, match="incomplete"):
        aligned_predictions(source, output)
    jsonl(output, [{**sample(), "input_sha256": "wrong", "predict_clean": "正确句"}])
    with pytest.raises(ValueError, match="different input"):
        aligned_predictions(source, output)


def test_fcgec_keeps_all_labels_and_operations():
    operations = [{"Delete": [2], "Insert": [{"pos": -1, "tag": "INS_1", "label": ["头", "首"]}],
                   "Modify": [{"pos": 1, "tag": "MOD_1", "label": ["新", "另"]}]}]
    assert set(fcgec_references("甲乙丙", operations)) == {"头甲新", "首甲新", "头甲另", "首甲另"}
    assert fcgec_references("甲乙丙", [{"Switch": [1, 0, 2]}, {"Switch": [2, 0, 1]}]) == ["乙甲丙", "丙甲乙"]


def test_fcgec_real_annotation_edge_cases():
    assert fcgec_references("甲乙丙", [{"Delete": [1, 2], "Modify": [{"pos": 1, "tag": "MOD_2", "label": "新"}]}]) == ["甲新"]
    assert fcgec_references("甲乙", [{"Modify": [{"pos": 1, "tag": "INS_2", "label": "观众"}]}]) == ["甲乙观众"]
    assert fcgec_references("甲乙", [{"Insert": [{"pos": 0, "tag": "INS_职业", "label": "职业"}]}]) == ["甲职业乙"]
    assert fcgec_references("甲乙。", [{"Switch": [1, 0]}]) == ["乙甲"]


def test_fcgec_unlabeled_test_exports_without_fake_score(tmp_path):
    raw = tmp_path / "test.json"
    write_json(raw, {"uid": {"sentence": "原文"}})
    row = next(fcgec_rows(raw, "test"))
    assert row["references"] is None
    source, predictions = tmp_path / "input.jsonl", tmp_path / "prediction.jsonl"
    jsonl(source, [row])
    jsonl(predictions, [{**row, "input_sha256": signature(row), "predict_clean": "修改文", "error": None}])
    report = score_cgec(source, predictions, tmp_path / "metrics.json")
    assert report["metrics"] is None and report["status"] == "exported_unlabeled_test"
    assert json.loads((tmp_path / "fcgec_test_predictions.json").read_text()) == {"uid": "修改文"}


def test_nacgec_multi_reference_reader(tmp_path):
    path = tmp_path / "test.para"
    path.write_text("7\t错句\t正句\t另一正确句\n")
    assert next(nacgec_rows(path, "test"))["references"] == ["正句", "另一正确句"]


def test_c3_scoring_not_target_leaking_and_invalid_punctuation_zero():
    rows = [dict(task="classification", gold="史", predict_clean="史。"),
            dict(task="retrieval", gold="史记·本纪", predict_clean="《史记》"),
            dict(task="ner", gold="甲、乙", predict_clean="甲、丙"),
            dict(task="punctuation", src="甲乙", gold="甲，乙。", predict_clean="甲，丙。")]
    report = c3_metrics(rows)
    assert report["tasks"]["classification"]["accuracy"] == 100
    assert report["tasks"]["retrieval"]["levels"]["1"]["accuracy"] == 100
    assert report["tasks"]["retrieval"]["levels"]["2"]["accuracy"] == 0
    assert report["tasks"]["ner"]["f1"] == 50
    assert report["tasks"]["punctuation"]["f1"] == 0
    assert report["official_code_equivalence"] is False


def test_fuxi_pending_judge_is_not_zero_or_em():
    rows = [dict(id="j", task="author", metric="judge", gold="李白", predict_clean="李白"),
            dict(id="a", task="choice", metric="acc", gold="B", predict_clean="答案是B")]
    result = fuxi_metrics(rows)
    assert not result["complete"]
    assert result["tasks"]["author"]["score"] is None
    assert result["tasks"]["choice"]["score"] == 100
    partial = fuxi_metrics([rows[0], {**rows[0], "id": "pending"}], {"j": {"score": 1}})
    assert partial["tasks"]["author"]["score"] is None
    assert partial["tasks"]["author"]["partial_score"] == 100


def test_fuxi_format_rules():
    assert couplet("春风得意", "秋雨无声")
    assert not couplet("", "秋雨无声")
    assert ci_format("一二三四五六七，一二三四五六七。一二三四五六七。\n一二三四五六七，一二三四五六七。一二三四五六七。", "浣溪沙")


def test_judge_cache_and_invalid_results(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "fixture")
    monkeypatch.setenv("JUDGE_BASE_URL", "http://localhost:12345/v1")
    row = dict(id="j", task="author", metric="judge", src="作者是谁", gold="李白", predict_clean="李白")
    source, output = tmp_path / "predictions.jsonl", tmp_path / "judgments.jsonl"
    jsonl(source, [row])
    result = judge_file(source, output, client=lambda prompt: "无法判断")
    assert not result["complete"]
    assert list(iter_jsonl(output))[0]["score"] is None
    assert judge_file(source, output, client=lambda prompt: "Y。")["complete"]
    assert judge_file(source, output, client=lambda prompt: pytest.fail("already cached"))["complete"]
    with pytest.raises(ValueError):
        parse_judgment("Yes, probably")


def test_official_m2_core_and_gold_creator():
    from scripts.evaluation.vendor.m2scorer import levenshtein as m2

    source, target = "我 是 学生 。", "我 是 一个 学生 。"
    edits = reference_edits(source, target)
    assert m2.batch_multi_pre_rec_f1([target], [source], [{0: edits}]) == (1, 1, 1)
    assert reference_edits(source, source) == []


def test_m2_matches_published_upstream_example():
    from scripts.evaluation.vendor.m2scorer import levenshtein as m2

    sources = ["The cat sat at mat .", "The dog .", "Giant otters is an apex predator ."]
    hypotheses = ["A cat sat on the mat .", "The dog .", "Giant otters are apex predator ."]
    gold = [{0: [(3, 4, "at", ["on"]), (4, 4, "", ["the", "a"])]},
            {0: [(1, 2, "dog", ["dogs"])], 1: []},
            {0: [(2, 3, "is", ["are"]), (3, 4, "an", [""]), (5, 6, "predator", ["predators"])],
             1: [(1, 2, "otters", ["otter"])]}]
    assert m2.batch_multi_pre_rec_f1(hypotheses, sources, gold) == pytest.approx((.8, .8, .8))


def test_real_cherrant_with_empty_generation(tmp_path):
    from scripts.evaluation.gec_metrics import cherrant_score

    rows = [dict(src="我喜换学习。", references=["我喜欢学习。"], predict_clean="我喜欢学习。"),
            dict(src="这是原句。", references=["这是原句。"], predict_clean="")]
    report = cherrant_score(rows, tmp_path / "scores")
    assert report["tp"] >= 1 and report["fp"] >= 1


def test_real_csc_migrated_scoring_pipeline(tmp_path):
    from scripts.evaluation.common import VENDOR
    from scripts.evaluation.csc.metrics import evaluate_files

    predictions = tmp_path / "predictions.jsonl"
    rows = [{"id": "1", "dataset": "cscd_ns", "subset": "all", "src": "错别子", "tgt": "错别字", "predict_clean": "错别字"},
            {"id": "2", "dataset": "cscd_ns", "subset": "all", "src": "正确", "tgt": "正确", "predict_clean": "正确"}]
    jsonl(predictions, rows)
    report = evaluate_files([predictions], tmp_path / "metrics.json", VENDOR / "cherrant", tmp_path / "adjusted.jsonl")
    assert report["raw"]["sentence_correction"]["f1"] == 100
    assert report["datasets_cherrant"]["cscd_ns"]["mean"]["table"]["correct-f1"] == 100


@pytest.mark.skipif(os.getenv("EVALUATION_INTEGRATION") != "1", reason="THULAC model and ms-swift required")
def test_real_thulac_maxmatch(tmp_path):
    from scripts.evaluation.gec_metrics import maxmatch_score

    result = maxmatch_score([dict(src="我喜换学习。", references=["我喜欢学习。"], predict_clean="我喜欢学习。")], tmp_path)
    assert result["f0.5"] == 100


@pytest.mark.skipif(os.getenv("EVALUATION_INTEGRATION") != "1", reason="THULAC model required")
def test_real_nacgec_dual_metric_pipeline(tmp_path):
    row = {"id": "nacgec/1", "original_id": "1", "dataset": "nacgec", "task": "cgec", "split": "test",
           "src": "我喜换学习。", "references": ["我喜欢学习。"], "messages": [{"role": "user", "content": "纠错"}]}
    inputs, predictions = tmp_path / "input.jsonl", tmp_path / "predictions.jsonl"
    jsonl(inputs, [row])
    jsonl(predictions, [{**row, "predict_clean": "我喜欢学习。", "error": None, "input_sha256": signature(row)}])
    report = score_cgec(inputs, predictions, tmp_path / "metrics.json")
    assert report["cherrant"]["f0.5"] == report["maxmatch"]["f0.5"] == report["score"] == 100


@pytest.mark.skipif(os.getenv("EVALUATION_INTEGRATION") != "1", reason="ms-swift and Transformers required")
def test_real_training_template_inference(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer

    checkpoint = tmp_path / "tiny_sft"
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-1.7B-Base-Char", local_files_only=True)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=4, max_position_embeddings=256,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    Qwen3ForCausalLM(config).save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    write_json(checkpoint / "args.json", {"template": "qwen3_nothinking", "model_type": "qwen3",
                                          "use_chat_template": True, "enable_thinking": False})
    source, output = tmp_path / "input.jsonl", tmp_path / "predictions.jsonl"
    jsonl(source, [sample(), sample("long", "这是一个长度不同的输入句子")])
    report = infer(source, output, checkpoint, max_new_tokens=4, batch_size=2)
    assert report["complete"]
    assert report["predictor"]["template"]["template"] == "qwen3_nothinking"
    from peft import LoraConfig, get_peft_model

    adapter = tmp_path / "adapter"
    get_peft_model(Qwen3ForCausalLM.from_pretrained(checkpoint),
                   LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM")).save_pretrained(adapter)
    write_json(adapter / "args.json", {"template": "qwen3", "model_type": "qwen3",
                                       "use_chat_template": True, "enable_thinking": False})
    result = infer(source, tmp_path / "lora_predictions.jsonl", checkpoint, adapter,
                   max_new_tokens=4, batch_size=2)
    assert result["complete"]
    assert result["predictor"]["template"]["template"] == "qwen3"
