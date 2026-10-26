from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from datasets import Dataset, DatasetDict

from scripts.evaluation.lmeh.datasets import load_cmmlu
from scripts.evaluation.lmeh.protocol import BENCHMARKS, DATA_REVISIONS, build_bundle, select_benchmarks
from scripts.evaluation.lmeh.run import evaluate_one, load_tasks, output_directory


@pytest.fixture(autouse=True)
def single_thread_cpu():
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_rows(path, name):
    if path == "cais/mmlu":
        return [{"question": "1 + 1?", "choices": ["2", "3", "4", "5"], "answer": 0}]
    if path == "allenai/ai2_arc":
        return [{"id": "one", "question": "1 + 1?", "choices": {"label": ["A", "B", "C", "D"],
                                                                  "text": ["2", "3", "4", "5"]}, "answerKey": "A"}]
    if path == "openai/gsm8k":
        return [{"question": "1 + 1?", "answer": "1 + 1 = 2.\n#### 2"}]
    if path == "lighteval/squad_v2":
        return [{"id": "one", "title": "France", "context": "Paris is the capital.",
                 "question": "What is the capital?", "answers": {"text": ["Paris"], "answer_start": [0]}}]
    assert path == "clue/clue"
    if name in {"afqmc", "cmnli"}:
        row = {"sentence1": "甲", "sentence2": "乙", "label": 0}
    elif name in {"tnews", "iflytek"}:
        row = {"sentence": "这是一段文本", "label": 0}
    elif name == "cluewsc2020":
        row = {"text": "小明说他要出门。", "target": {"span1_text": "小明", "span2_text": "他"}, "label": 0}
    elif name == "csl":
        row = {"keyword": ["学习"], "abst": "学习知识", "label": 1}
    elif name == "cmrc2018":
        row = {"id": "one", "context": "小明住北京。", "question": "小明住哪里？",
               "answers": {"text": ["北京"], "answer_start": [3]}}
    elif name == "chid":
        row = {"idx": "one", "content": ["这里#idiom000000#。"], "candidates": ["一心一意", "三心二意"],
               "answers": {"candidate_id": [0]}}
    else:
        assert name == "c3"
        row = {"context": ["我喜欢读书。"], "question": "我喜欢什么？", "choice": ["读书", "运动"], "answer": "读书"}
    return [row]


@pytest.fixture
def fake_hub(monkeypatch):
    calls = []
    def load(path, name=None, **kwargs):
        assert kwargs["revision"] == DATA_REVISIONS["datasets"][path]
        calls.append((path, name, kwargs))
        rows = fixture_rows(path, name)
        return DatasetDict({"train": Dataset.from_list(rows * 6), "dev": Dataset.from_list(rows * 5),
                            "test": Dataset.from_list(rows), "validation": Dataset.from_list(rows)})
    monkeypatch.setattr("datasets.load_dataset", load)
    return calls


def cmmlu_csv(root, subject="agronomy", examples=5):
    for split, count in (("dev", examples), ("test", 1)):
        path = root / split / f"{subject}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "Question", "A", "B", "C", "D", "Answer"])
            writer.writerows([[i, "问题", "0.80", "1/10", "(1) (2)", "其他", "A"] for i in range(count)])


def test_cmmlu_retains_text_and_requires_five_dev_examples(tmp_path):
    cmmlu_csv(tmp_path)
    loaded = load_cmmlu("agronomy", tmp_path)
    assert loaded["test"][0]["A"] == "0.80"
    assert loaded["test"][0]["B"] == "1/10"
    cmmlu_csv(tmp_path, examples=4)
    with pytest.raises(ValueError, match="five"):
        load_cmmlu("agronomy", tmp_path)
    with pytest.raises(FileNotFoundError, match="CMMLU_DIR"):
        load_cmmlu("missing", tmp_path)


@pytest.mark.parametrize("benchmark", list(BENCHMARKS))
def test_all_task_bundles_load_with_correct_protocol(tmp_path, fake_hub, benchmark):
    bundle = build_bundle(tmp_path / "tasks", benchmark, tmp_path / "data")
    if benchmark == "cmmlu":
        for path in bundle.glob("cmmlu_*.yaml"):
            name = yaml.safe_load(path.read_text())["dataset_name"]
            cmmlu_csv(tmp_path / "data", name)
    manager, loaded, inventory = load_tasks(bundle, benchmark)
    assert len(inventory) == BENCHMARKS[benchmark]["leaves"]
    assert all(row["num_fewshot"] == (5 if benchmark in {"mmlu", "cmmlu", "gsm8k"} else 0)
               for row in inventory.values())
    for task in loaded["tasks"].values():
        text = task.doc_to_text(task.eval_docs[0])
        assert isinstance(text, str) and text
        if benchmark == "cmmlu":
            assert text.endswith("答案：")
            assert task.config.description.startswith("以下是关于")
        if benchmark == "mmlu":
            assert task.config.description.startswith("The following are multiple choice")
        if benchmark == "clue":
            assert task.config.test_split is None and task.config.validation_split == "validation"
        if benchmark == "squadv2":
            requests = task.construct_requests(task.eval_docs[0], text)
            assert [req.request_type for req in requests] == ["generate_until", "loglikelihood"]
    if benchmark != "cmmlu":
        assert fake_hub


def test_task_selection_and_output_paths():
    assert select_benchmarks("arc,mmlu,arc_easy") == ["arc_easy", "arc_challenge", "mmlu"]
    assert len(select_benchmarks("all")) == 7
    assert str(output_directory("/tmp/output", "arc_easy", "Base", "trial")) == "/tmp/output/arc/Base/easy/trial"
    with pytest.raises(ValueError):
        select_benchmarks("gsm8k_cot")
    with pytest.raises(ValueError):
        output_directory("/tmp/output", "mmlu", "../bad")


@pytest.fixture
def tiny_model(tmp_path):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(42)
    path = tmp_path / "model"
    raw = Tokenizer(models.WordLevel({"[UNK]": 0, "A": 1, "B": 2, "C": 3, "D": 4,
                                      "1": 5, "2": 6, "3": 7, "4": 8, "5": 9, "[EOS]": 10}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]", model_max_length=1024)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=11, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                                      num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                                      max_position_embeddings=1024, eos_token_id=10))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def test_real_hf_run_writes_results_and_never_overwrites(tmp_path, tiny_model, fake_hub):
    args = SimpleNamespace(model=tiny_model, model_name="TinyBase", run_name="smoke", output_root=tmp_path / "out",
                           cmmlu_dir=tmp_path / "data", max_length=1024, batch_size="1", device="cpu", dtype="float32", limit=1)
    backend, summary = evaluate_one(args, "arc_easy", None)
    output = Path(summary["output"])
    report = json.loads((output / "results.json").read_text())
    assert "acc,none" in report["results"]["arc_easy"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] and manifest["smoke_only"]
    assert manifest["protocol"]["shots"] == 0 and manifest["sample_sha256"]
    assert manifest["actual_model"]["class"] == "Qwen3ForCausalLM"
    with pytest.raises(FileExistsError):
        evaluate_one(args, "arc_easy", backend)
    # Reuse the loaded model but retain 5-shot MMLU groups and aggregation.
    _, summary = evaluate_one(args, "mmlu", backend)
    result = json.loads((Path(summary["output"]) / "results.json").read_text())
    assert result["groups"]["mmlu"]["acc,none"] >= 0
    mmlu_manifest = json.loads((Path(summary["output"]) / "manifest.json").read_text())
    assert len(mmlu_manifest["datasets"]) == 57
    assert all(result["n-shot"][name] == 5 for name in mmlu_manifest["datasets"])


def test_backend_rejects_truncation_and_generates(tiny_model):
    from lm_eval.api.instance import Instance
    from scripts.evaluation.lmeh.backend import StrictHFLM
    from scripts.evaluation.ppl.evaluate import load_model
    model, tokenizer, _ = load_model(str(tiny_model), device="cpu", dtype="float32")
    backend = StrictHFLM(pretrained=model, tokenizer=tokenizer, backend="causal", max_length=16, batch_size=1)
    with pytest.raises(ValueError, match="refusing to truncate"):
        backend._loglikelihood_tokens([(("context", "answer"), [1] * 16, [2, 3])])
    request = Instance(request_type="generate_until", doc={}, arguments=("A " * 16, {"max_gen_toks": 2}), idx=0)
    with pytest.raises(ValueError, match="refusing to truncate"):
        backend.generate_until([request])
    request = Instance(request_type="generate_until", doc={}, arguments=("A B", {"max_gen_toks": 2, "until": ["\n"],
                                                                                "do_sample": False}), idx=0)
    assert len(backend.generate_until([request])) == 1


def test_remaining_protocols_end_to_end(tmp_path, tiny_model, fake_hub, monkeypatch):
    import lm_eval
    from scripts.evaluation.lmeh import squad
    # The official metric is separately validated online; keep this regression offline.
    keys = ("exact", "f1", "HasAns_exact", "HasAns_f1", "NoAns_exact", "NoAns_f1", "best_exact", "best_f1")
    monkeypatch.setattr(squad, "squad_metric", lambda: SimpleNamespace(compute=lambda **kw: dict.fromkeys(keys, 0.0)))
    data = tmp_path / "data"
    for path in (Path(lm_eval.__file__).parent / "tasks/cmmlu").glob("cmmlu_*.yaml"):
        cmmlu_csv(data, yaml.safe_load(path.read_text())["dataset_name"])
    args = SimpleNamespace(model=tiny_model, model_name="TinyBase", run_name="smoke", output_root=tmp_path / "out",
                           cmmlu_dir=data, max_length=1024, batch_size="1", device="cpu", dtype="float32", limit=1)
    backend = None
    for benchmark in ("arc_challenge", "gsm8k", "cmmlu", "clue", "squadv2"):
        backend, summary = evaluate_one(args, benchmark, backend)
        manifest = json.loads((Path(summary["output"]) / "manifest.json").read_text())
        assert manifest["complete"] and len(manifest["datasets"]) == BENCHMARKS[benchmark]["leaves"]
    # An error must not create successful final scores.
    def fail(**kwargs):
        raise RuntimeError("simulated model failure")
    monkeypatch.setattr("lm_eval.evaluator.simple_evaluate", fail)
    args.run_name = "failure"
    with pytest.raises(RuntimeError, match="simulated"):
        evaluate_one(args, "arc_easy", backend)
    output = output_directory(args.output_root, "arc_easy", args.model_name, args.run_name)
    assert not (output / "results.json").exists()
    assert json.loads((output / "manifest.json").read_text())["complete"] is False
    # Disk metadata cannot silently relabel weights already in memory.
    args.run_name = "changed"
    backend.checkpoint_identity = {"changed": True}
    with pytest.raises(RuntimeError, match="changed between benchmarks"):
        evaluate_one(args, "arc_easy", backend)


def test_squad_metric_is_revision_pinned(monkeypatch):
    from scripts.evaluation.lmeh import squad
    calls = []
    monkeypatch.setattr(squad.evaluate, "load", lambda *a, **kw: calls.append((a, kw)) or object())
    squad.squad_metric.cache_clear()
    squad.squad_metric()
    squad.squad_metric()
    assert calls == [(("squad_v2",), {"revision": DATA_REVISIONS["squad_metric_revision"]})]
    squad.squad_metric.cache_clear()


def test_squad_missing_answerability_group_matches_upstream(monkeypatch):
    from scripts.evaluation.lmeh import squad
    monkeypatch.setattr(squad, "squad_metric", lambda: SimpleNamespace(compute=lambda **kwargs: {"exact": 100}))
    assert squad.aggregate_squad("NoAns_exact", [({"id": "one"}, {"id": "one"})]) == 0


def test_self_contained_pgca_hf_likelihood_and_generation(tmp_path, tiny_model):
    from lm_eval.api.instance import Instance
    from transformers import AutoTokenizer
    from test_pgca_context import context_model
    from src.pgca.build_pgca import AUTO_MAP, copy_runtime_code, save_pretrained_without_auto_copy
    from scripts.evaluation.lmeh.backend import StrictHFLM
    from scripts.evaluation.ppl.evaluate import load_model

    model = context_model().eval()
    model.config.auto_map = dict(AUTO_MAP)
    path = tmp_path / "pgca"
    save_pretrained_without_auto_copy(model, path)
    copy_runtime_code(path)
    AutoTokenizer.from_pretrained(tiny_model).save_pretrained(path)
    loaded, tokenizer, _ = load_model(str(path), device="cpu", dtype="float32")
    backend = StrictHFLM(pretrained=loaded, tokenizer=tokenizer, backend="causal", max_length=16, batch_size=2,
                         add_bos_token=False)
    likelihoods = [Instance(request_type="loglikelihood", doc={}, arguments=(text, " A"), idx=0)
                   for text in ("A B", "A B C")]
    assert len(backend.loglikelihood(likelihoods)) == 2
    generations = [Instance(request_type="generate_until", doc={},
                            arguments=(text, {"max_gen_toks": 2, "until": ["\n"], "do_sample": False}), idx=0)
                   for text in ("A B", "A B C")]
    assert len(backend.generate_until(generations)) == 2


def test_launcher_binds_its_environment_without_cwd(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/evaluation/lmeh/run.sh"
    assert '${PYTHON:-$TASK_DIR/env/bin/python}' in script.read_text()
    recorder = tmp_path / "python"
    recorder.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    recorder.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in {"BENCHMARK", "STAGE", "MODEL_NAME", "MODEL", "OUT", "RUN_NAME", "CMMLU_DIR", "LIMIT"}}
    env.update(PYTHON=str(recorder), STAGE="inspect", BENCHMARK="all")
    proc = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    args = json.loads(proc.stdout)
    assert args[:3] == ["-B", "-m", "scripts.evaluation.lmeh.run"]
    assert args[args.index("--stage") + 1] == "inspect"
