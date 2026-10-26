"""Resolve installed built-in tasks; retain upstream prompts, groups and metrics."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE.parents[2] / "data/evaluation/lmeh"
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


def _drop_yaml_block(text, key):
    lines = text.splitlines()
    output = []
    index = 0
    while index < len(lines):
        if lines[index].startswith(f"{key}:"):
            index += 1
            while index < len(lines) and (not lines[index] or lines[index][0].isspace()):
                index += 1
            continue
        output.append(lines[index])
        index += 1
    return "\n".join(output) + "\n"


def build_bundle(destination, benchmark, data_dir=DEFAULT_DATA_DIR):
    """Snapshot only the selected protocol, independent of external checkout/CWD."""
    import lm_eval

    os.environ["CC_LLM_LMEH_DATA_DIR"] = str(Path(data_dir).resolve())
    source = Path(lm_eval.__file__).resolve().parent / "tasks"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if benchmark == "clue":
        for path in (HERE / "tasks/clue").iterdir():
            if path.is_file():
                shutil.copyfile(path, destination / path.name)
        for name in ("_classification_template_yaml", "_qa_template_yaml"):
            path = destination / name
            text = _drop_yaml_block(path.read_text(encoding="utf-8"), "dataset_kwargs")
            text = re.sub(r"^dataset_path:.*$", "dataset_path: null", text, count=1, flags=re.MULTILINE)
            text = re.sub(r"^  dataset_revision:.*\n?", "", text, flags=re.MULTILINE)
            text += "num_fewshot: 0\ncustom_dataset: !function scripts.evaluation.lmeh.datasets.load_local_parquet\n"
            path.write_text(text, encoding="utf-8")
        for path in destination.glob("clue_*.yaml"):
            text = path.read_text(encoding="utf-8")
            match = re.search(r"^dataset_name:\s*([^\s#]+)", text, re.MULTILINE)
            if match:
                text += ("dataset_kwargs:\n"
                         f"  data_dir: {str((Path(data_dir) / 'clue').resolve())!r}\n"
                         f"  name: {match.group(1)!r}\n")
            path.write_text(text, encoding="utf-8")
        return destination
    if benchmark == "squadv2":
        (destination / "squadv2.yaml").write_text(
            "task: squadv2\nclass: !function scripts.evaluation.lmeh.squad.LocalSQuAD2\n"
            f"dataset_kwargs:\n  data_dir: {str((Path(data_dir) / 'squadv2').resolve())!r}\n  name: default\n",
            encoding="utf-8",
        )
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
                config["dataset_path"] = None
                extra = "custom_dataset: !function scripts.evaluation.lmeh.datasets.load_local_parquet\n"
            config["num_fewshot"] = BENCHMARKS[benchmark]["shots"]
        if family == "cmmlu" and "dataset_name" in config:
            config["dataset_kwargs"] = {
                "subject": config["dataset_name"],
                "data_dir": str((Path(data_dir) / "cmmlu").resolve()),
            }
        elif "dataset_name" in config:
            local_family = "arc_challenge" if family == "arc" else family
            config["dataset_kwargs"] = {
                "data_dir": str((Path(data_dir) / local_family).resolve()),
                "name": config["dataset_name"],
            }
        elif family == "mmlu" and name == "_default_template_yaml":
            config["dataset_kwargs"] = {"data_dir": str((Path(data_dir) / "mmlu").resolve())}
        if family == "gsm8k":
            config["generation_kwargs"]["max_gen_toks"] = BENCHMARKS[benchmark]["max_gen_toks"]
        (destination / name).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False) + extra,
                                        encoding="utf-8")
    return destination
