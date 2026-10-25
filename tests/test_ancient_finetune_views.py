from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train.ms_swift.finetune.prepare_ancient_views import TASKS, prepare


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _fixture(root: Path) -> None:
    tasks = list(TASKS)
    rows = [{"messages": [{"role": "user", "content": task},
                           {"role": "assistant", "content": f"{task}-answer"}]} for task in tasks]
    metadata = [dict(row, task=task, audit_only="not forwarded") for row, task in zip(rows, tasks)]
    for split in ("train", "dev"):
        _write_rows(root / f"{split}.jsonl", rows)
        _write_rows(root / f"{split}.metadata.jsonl", metadata)
    report = {
        "ready_for_training": True,
        "tasks": {f"{split}/{task}": {"selected": 1} for split in ("train", "dev") for task in tasks},
    }
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_prepare_builds_minimal_channel_views(tmp_path):
    data, output = tmp_path / "final", tmp_path / "views"
    data.mkdir()
    _fixture(data)
    report = prepare(data, output)
    assert report["counts"] == {"train": dict.fromkeys(TASKS, 1), "dev": dict.fromkeys(TASKS, 1)}
    rows = [json.loads(line) for line in (output / "dev.jsonl").read_text().splitlines()]
    assert [row["channel"] for row in rows] == list(TASKS)
    assert all(set(row) == {"messages", "channel"} for row in rows)


def test_prepare_rejects_unaccepted_or_misaligned_data(tmp_path):
    data, output = tmp_path / "final", tmp_path / "views"
    data.mkdir()
    _fixture(data)
    report_path = data / "report.json"
    report = json.loads(report_path.read_text())
    report["ready_for_training"] = False
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="not accepted"):
        prepare(data, output)

    report["ready_for_training"] = True
    report_path.write_text(json.dumps(report))
    rows = [json.loads(line) for line in (data / "dev.metadata.jsonl").read_text().splitlines()]
    rows[0]["messages"][0]["content"] = "different"
    _write_rows(data / "dev.metadata.jsonl", rows)
    with pytest.raises(ValueError, match="messages mismatch"):
        prepare(data, output)
