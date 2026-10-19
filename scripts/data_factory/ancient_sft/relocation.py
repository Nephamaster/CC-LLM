"""Recognize the documented directory move without relaxing content checks."""

from __future__ import annotations

import copy
from pathlib import Path

from .common import ROOT

PATH_MOVES = (
    ("data/downstream/ancient/chinese-classical-corpus", "data/downstream/ancient/raw/chinese-classical-corpus"),
    ("data/downstream/ancient/HistoryTrans", "data/downstream/ancient/raw/HistoryTrans"),
    ("data/downstream/ancient_sft_v1", "data/downstream/ancient/train"),
    ("data/downstream/ancient_sft_v2", "data/downstream/ancient/train"),
)


def canonical_path(value):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return str(path)
    name = path.as_posix()
    for old, new in PATH_MOVES:
        if name == old or name.startswith(old + "/"):
            return new + name[len(old):]
    for split in ("train", "validation", "test"):
        if name == f"data/downstream/ancient/HistoryTrans-{split}.jsonl":
            return f"data/downstream/ancient/raw/HistoryTrans/HistoryTrans-{split}.jsonl"
    return name


def canonical_config(config):
    result = copy.deepcopy(config)
    # Old prepared manifests contain this obsolete environment pin, not a data parameter.
    result.pop("swift_commit", None)
    for name in ("sources", "benchmarks"):
        result[name] = {key: canonical_path(value) for key, value in result[name].items()}
    for name in ("output_dir", "tokenizer_path", "reviews"):
        if name in result:
            result[name] = canonical_path(result[name])
    return result


def canonical_inventory(inventory):
    result = copy.deepcopy(inventory)
    result["missing"] = sorted(canonical_path(path) for path in result["missing"])
    for entries in result["files"].values():
        for entry in entries:
            entry["path"] = canonical_path(entry["path"])
        entries.sort(key=lambda entry: entry["path"])
    return result
