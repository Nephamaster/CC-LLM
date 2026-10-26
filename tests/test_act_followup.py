import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation.ancient.diagnostics.followup import end_distribution, load_cases, teacher_forced_end
from scripts.evaluation.ancient.diagnostics.audit_data import run, structural_flags


def test_teacher_forcing_predicts_eos_without_feeding_eos():
    torch = pytest.importorskip("torch")

    class Model:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def __call__(self, **kwargs):
            assert kwargs["input_ids"].tolist() == [[2, 3, 4]]
            assert kwargs["use_cache"] is False
            return SimpleNamespace(logits=torch.tensor([[[0., 4., 2., 0., 0., 0.]]]))

    tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kw: str(ids[0]))
    result = teacher_forced_end(Model(), tokenizer, {"input_ids": [2, 3, 4, 2], "labels": [-100, -100, 4, 2]}, 2, 1)
    assert result["prefix_tokens"] == 3 and result["candidate_above_eos"]
    assert result["candidate_minus_eos_logit"] == 2
    assert result["candidate"]["rank"] == 1 and result["im_end"]["rank"] == 2
    with pytest.raises(ValueError, match="Non-finite"):
        end_distribution(torch.tensor([float("nan"), 0.]), tokenizer, 0, 1)


def test_fixed_cases_reject_reference_changes(tmp_path):
    rows = [{"id": str(i), "task": task, "messages": []} for i, task in enumerate(
        ("c2m", "c2m", "m2c", "m2c", "punctuate", "punctuate"))]
    dev, selected = tmp_path / "dev.jsonl", tmp_path / "selected.json"
    dev.write_text("".join(json.dumps(r) + "\n" for r in rows))
    selected.write_text(json.dumps([dict(r, index=i) for i, r in enumerate(rows)]))
    assert len(load_cases(selected, dev)) == 6
    rows[0]["messages"] = [{"role": "assistant", "content": "changed"}]
    dev.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="differs"):
        load_cases(selected, dev)


def test_data_audit_preserves_pending_semantics_even_when_raw_pair_matches(tmp_path):
    data, raw = tmp_path / "final", tmp_path / "raw"
    data.mkdir()
    raw.mkdir()
    (raw / "chinese-classical-corpus").mkdir()
    rows = []
    raw_rows = []
    for i, task in enumerate(("c2m", "m2c", "punctuate")):
        source, target = ("甲乙", "甲，乙。") if task == "punctuate" else ("古文", "不相干的译文")
        ancient, modern = (target, source) if task == "m2c" else (source, target)
        rows.append({"id": str(i), "task": task, "ancient": ancient, "modern": modern,
                     "metadata": {"source_location": f"old/translate.jsonl:{i + 1}"},
                     "messages": [{"role": "user", "content": "任务：" + source},
                                  {"role": "assistant", "content": target}]})
        raw_rows.append({"input": source, "output": target})
    for split in ("train", "dev"):
        (data / f"{split}.jsonl").write_text("".join(json.dumps({"messages": r["messages"]}) + "\n" for r in rows))
        (data / f"{split}.metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (raw / "chinese-classical-corpus/translate.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw_rows))
    out = tmp_path / "audit"
    run(SimpleNamespace(data=data, raw=raw, output=out, per_task=1, seed=42))
    report = json.loads((out / "report.json").read_text())
    assert len(report["groups"]) == 6
    assert all(v["source_pair_mismatch"] == 0 and v["semantic_pending"] == 1 for v in report["groups"].values())
    broken = dict(rows[-1], modern="甲丙。")
    assert "punctuation_changes_characters" in structural_flags(broken)
