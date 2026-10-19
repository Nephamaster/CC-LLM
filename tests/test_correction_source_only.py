import json
from pathlib import Path

import pytest

from scripts.data_factory.prepare_cgec import messages
from scripts.data_factory.strip_correction_prompts import (
    CGEC_PREFIX, CSC_PREFIX, CSC_SUFFIX, convert_file, target_files, unwrap,
)
from scripts.evaluation.cgec.prepare import fcgec_rows, nacgec_rows
from scripts.evaluation.csc.prepare import rows as csc_rows


@pytest.mark.parametrize("task,prefix,suffix", [("csc", CSC_PREFIX, CSC_SUFFIX), ("cgec", CGEC_PREFIX, "")])
def test_conversion_only_removes_outer_wrapper_and_is_repeatable(tmp_path, task, prefix, suffix):
    source = " 句子中写着“输出: ”，保留它。\n另一行 "
    row = {"messages": [{"role": "user", "content": prefix + source + suffix},
                        {"role": "assistant", "content": "正确答案", "loss": True}], "id": "preserved"}
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    original = path.read_bytes()
    assert convert_file(path, task)["changed"] == 1
    assert path.read_bytes() == original
    report = convert_file(path, task, apply=True)
    assert report["changed"] == 1 and Path(report["backup"]).read_bytes() == original
    row["messages"][0]["content"] = source
    assert json.loads(path.read_text()) == row
    assert convert_file(path, task, apply=True)["changed"] == 0
    assert Path(report["backup"]).read_bytes() == original


def test_malformed_wrapper_does_not_modify_source(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": CSC_PREFIX + "句子"},
                                            {"role": "assistant", "content": "答案"}]}))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="suffix"):
        convert_file(path, "csc", apply=True)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.bak")) and not list(tmp_path.glob(".*.tmp"))


def test_existing_backup_is_never_overwritten(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(messages(CGEC_PREFIX + "原句", "正确句")) + "\n")
    backup = path.with_name(path.name + ".with-prompts.bak")
    backup.write_text("previous backup")
    with pytest.raises(FileExistsError):
        convert_file(path, "cgec", apply=True)
    assert backup.read_text() == "previous backup"
    assert json.loads(path.read_text())["messages"][0]["content"] == CGEC_PREFIX + "原句"


def test_target_discovery_ignores_raw_and_backups(tmp_path):
    wanted = ["csc/sft/train.jsonl", "csc/sft/cscd_dev.jsonl", "csc/sft/final/extra.jsonl",
              "cgec/sft/final/train.jsonl", "cgec/sft/final/validation.jsonl"]
    for name in wanted + ["csc/sft/raw/raw.jsonl", "cgec/sft/raw/raw.jsonl", "csc/sft/train.jsonl.with-prompts.bak"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert {str(path.relative_to(tmp_path)) for _, path in target_files(tmp_path)} == set(wanted)


def test_new_training_and_evaluation_inputs_are_source_only(tmp_path):
    source = "原始 句子\n下一行"
    assert messages(source, "正确句")["messages"][0]["content"] == source
    csc = tmp_path / "csc.jsonl"
    csc.write_text(json.dumps({"id": "one", "dataset": "cscd_ns", "subset": "all", "src": source, "tgt": "正确句"}) + "\n")
    assert next(csc_rows([csc]))["messages"] == [{"role": "user", "content": source}]
    fcgec = tmp_path / "fcgec.json"
    fcgec.write_text(json.dumps({"one": {"sentence": source, "error_flag": 0, "operation": []}}))
    assert next(fcgec_rows(fcgec, "valid"))["messages"] == [{"role": "user", "content": source}]
    nacgec = tmp_path / "nacgec.ref.para"
    nacgec.write_text("1\t带错句\t正确句\n", encoding="utf-8")
    assert next(nacgec_rows(nacgec, "test"))["messages"] == [{"role": "user", "content": "带错句"}]


def test_unwrapped_text_is_not_cleaned():
    text = "解释：这是原句，含任务: 纠错文本\n输出: "
    assert unwrap(text, "csc") == text
    assert unwrap(text, "cgec") == text
