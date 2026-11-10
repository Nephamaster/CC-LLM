"""Resolve installed benchmark tasks without pinning package or dataset revisions."""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BENCHMARKS = {
    "mmlu": {"task": "mmlu", "shots": 5, "leaves": 57, "output": "mmlu"},
    "arc_challenge": {"task": "arc_challenge", "shots": 0, "leaves": 1, "output": "arc/challenge"},
    "squadv2": {"task": "squadv2", "shots": 0, "leaves": 1, "output": "squadv2", "max_gen_toks": 256},
    "gsm8k": {"task": "gsm8k", "shots": 5, "leaves": 1, "output": "gsm8k", "max_gen_toks": 512},
    "cmmlu": {"task": "cmmlu", "shots": 5, "leaves": 67, "output": "cmmlu"},
    "clue": {"task": "clue", "shots": 0, "leaves": 9, "output": "clue"},
}


def table_metrics(benchmark, results):
    """Explicit metrics for the experiment sheet; never mix score scales."""
    def metric(task, name, filter_name="none"):
        return results.get(task, {}).get(f"{name},{filter_name}")

    if benchmark == "clue":
        members = {
            "semantic_inference": [("clue_afqmc", "acc"), ("clue_cmnli", "acc"), ("clue_wsc2020", "acc")],
            "text_classification": [("clue_tnews", "acc"), ("clue_iflytek", "acc")],
            "semantic_consistency": [("clue_csl", "acc")],
            "word_comprehension": [("clue_chid", "acc")],
            "reading_comprehension": [("clue_c3", "acc"), ("clue_cmrc2018", "f1")],
        }
        scores = {}
        for group, tasks in members.items():
            values = [metric(task, name) for task, name in tasks]
            scores[group] = sum(values) / len(values) if all(v is not None for v in values) else None
        return {"scores": scores, "scale": "0..1", "aggregation": "equal-weight task mean within each group",
                "members": members, "note": "Reading combines C3 accuracy and CMRC2018 F1; not an official CLUE total."}
    task, name, filter_name, scale = {
        "mmlu": ("mmlu", "acc", "none", 1),
        "arc_challenge": ("arc_challenge", "acc_norm", "none", 1),
        "squadv2": ("squadv2", "f1", "none", 100),
        "gsm8k": ("gsm8k", "exact_match", "strict-match", 1),
        "cmmlu": ("cmmlu", "acc", "none", 1),
    }[benchmark]
    value = metric(task, name, filter_name)
    return {"score": value / scale if value is not None else None, "scale": "0..1",
            "task": task, "metric": name, "filter": filter_name}


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


def build_bundle(destination, benchmark):
    """Copy the selected task protocol from the installed lm-eval package."""
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
            "task: squadv2\nclass: !function scripts.evaluation.lmeh.squad.BudgetedSQuAD2\n",
            encoding="utf-8",
        )
        return destination

    family = "arc" if benchmark.startswith("arc_") else benchmark
    folder = source / ("mmlu/default" if family == "mmlu" else family)
    names = [path.name for path in folder.iterdir()
             if path.suffix == ".yaml" or path.name == "_default_template_yaml"]
    if family == "gsm8k":
        names = ["gsm8k.yaml"]
    if family == "arc":
        names = ["arc_challenge.yaml"]
    for name in sorted(names):
        config = yaml.safe_load((folder / name).read_text(encoding="utf-8"))
        if family == "arc":
            common = yaml.safe_load((folder / "arc_easy.yaml").read_text(encoding="utf-8"))
            config = {**common, **config}
            config.pop("include", None)
        if "dataset_path" in config:
            config.pop("dataset_kwargs", None)
            config["num_fewshot"] = BENCHMARKS[benchmark]["shots"]
        if family == "gsm8k":
            config["generation_kwargs"]["max_gen_toks"] = BENCHMARKS[benchmark]["max_gen_toks"]
        (destination / name).write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destination
