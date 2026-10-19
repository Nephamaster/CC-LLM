"""Recognize the documented directory move without relaxing content checks."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

from .common import ROOT, digest

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


def preparation_contract(root: Path) -> str:
    files = {}
    for name in ("__init__.py", "common.py", "encoding.py", "overlap.py", "partition.py",
                 "selection.py", "sources.py", "run.py"):
        source = (root / name).read_text(encoding="utf-8")
        if name == "common.py":
            source = source.replace("ROOT = Path(__file__).resolve().parents[3]",
                                    "ROOT = Path(__file__).resolve().parents[2]")
        if name == "run.py":
            source = source.replace("scripts.data_factory.ancient_sft", "scripts.ancient_sft").replace(
                "scripts/data_factory/ancient_sft", "scripts/ancient_sft")
        files[name] = digest(source)
    source = (root / "pipeline.py").read_text(encoding="utf-8")
    segments = [ast.get_source_segment(source, node) for node in ast.parse(source).body
                if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name in {"finalize", "inspect_inputs"})]
    return digest(json.dumps({"files": files, "pipeline_prepare_source": "\n".join(segments)}, sort_keys=True))


def is_verified_relocation(recorded: str, root: Path, expected: dict) -> bool:
    original = {"__init__.py", "common.py", "encoding.py", "overlap.py", "partition.py", "pipeline.py",
                "provenance.py", "repartition.py", "run.py", "selection.py", "sources.py"}
    return (recorded == expected["implementation_sha256"]
            and {path.name for path in root.glob("*.py")} == original | {"relocation.py"}
            and preparation_contract(root) == expected["preparation_contract_sha256"])
