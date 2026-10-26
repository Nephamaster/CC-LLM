"""Resolve installed built-in tasks; retain upstream prompts, groups and metrics."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from ..common import sha256

HERE = Path(__file__).resolve().parent
DEFAULT_CMMLU_DIR = Path("/share/project/wuhaiming/data/dataset/cmmlu")
DATA_REVISIONS = json.loads((HERE / "dataset_revisions.json").read_text(encoding="utf-8"))
BENCHMARKS = {
    "mmlu": {"task": "mmlu", "shots": 5, "leaves": 57, "output": "mmlu"},
    "arc_challenge": {"task": "arc_challenge", "shots": 0, "leaves": 1, "output": "arc/challenge"},
    "squadv2": {"task": "squadv2", "shots": 0, "leaves": 1, "output": "squadv2", "max_gen_toks": 256},
    "gsm8k": {"task": "gsm8k", "shots": 5, "leaves": 1, "output": "gsm8k", "max_gen_toks": 512},
    "cmmlu": {"task": "cmmlu", "shots": 5, "leaves": 67, "output": "cmmlu"},
    "clue": {"task": "clue", "shots": 0, "leaves": 9, "output": "clue"},
}


def select_benchmarks(value):
    selected = []
    for name in value.split(","):
        names = list(BENCHMARKS) if name == "all" else ["arc_challenge"] if name == "arc" else [name]
        for item in names:
            if item not in BENCHMARKS:
                raise ValueError(f"unknown benchmark {item!r}; choose all, arc or {', '.join(BENCHMARKS)}")
            if item not in selected:
                selected.append(item)
    return selected


def build_bundle(destination, benchmark, cmmlu_dir=DEFAULT_CMMLU_DIR):
    """Snapshot only the selected protocol, independent of external checkout/CWD."""
    import lm_eval

    source = Path(lm_eval.__file__).resolve().parent / "tasks"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if benchmark == "clue":
        for path in (HERE / "tasks/clue").iterdir():
            if path.is_file():
                shutil.copyfile(path, destination / path.name)
        for name in ("_classification_template_yaml", "_qa_template_yaml"):
            with (destination / name).open("a", encoding="utf-8") as handle:
                handle.write("\nnum_fewshot: 0\n")
        return destination
    if benchmark == "squadv2":
        (destination / "squadv2.yaml").write_text(
            "task: squadv2\nclass: !function scripts.evaluation.lmeh.squad.PinnedSQuAD2\n", encoding="utf-8")
        return destination
    family = "arc" if benchmark.startswith("arc_") else benchmark
    folder = source / ("mmlu/default" if family == "mmlu" else family)
    names = [p.name for p in folder.iterdir() if p.suffix == ".yaml" or p.name == "_default_template_yaml"]
    if family == "gsm8k":
        names = ["gsm8k.yaml"]
    if family == "arc":
        names = ["arc_challenge.yaml"]
    for name in sorted(names):
        config = yaml.safe_load((folder / name).read_text(encoding="utf-8"))
        if family == "arc":
            # Challenge inherits the prompt/scorer from Easy; inline it without registering Easy.
            common = yaml.safe_load((folder / "arc_easy.yaml").read_text(encoding="utf-8"))
            config = {**common, **config}
            config.pop("include", None)
        extra = ""
        if "dataset_path" in config:
            if family == "cmmlu":
                config["dataset_path"] = None
                extra = "custom_dataset: !function scripts.evaluation.lmeh.datasets.load_cmmlu\n"
            else:
                config["dataset_kwargs"] = {"revision": DATA_REVISIONS["datasets"][config["dataset_path"]]}
            config["num_fewshot"] = BENCHMARKS[benchmark]["shots"]
        if family == "cmmlu" and "dataset_name" in config:
            config["dataset_kwargs"] = {"subject": config["dataset_name"], "data_dir": str(Path(cmmlu_dir).resolve())}
        if family == "gsm8k":
            config["generation_kwargs"]["max_gen_toks"] = BENCHMARKS[benchmark]["max_gen_toks"]
        (destination / name).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False) + extra,
                                        encoding="utf-8")
    return destination


def bundle_hashes(path):
    return {p.name: sha256(p) for p in sorted(Path(path).iterdir()) if p.is_file()}
