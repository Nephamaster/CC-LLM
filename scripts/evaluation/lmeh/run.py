"""Run local benchmark protocols with HF or vLLM inference."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

from ..common import ROOT
from ..runs import new_run_directory
from ..ppl.evaluate import checkpoint_identity, load_model, write_report
from .protocol import BENCHMARKS, build_bundle, select_benchmarks


def validate_component(value):
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("MODEL_NAME/RUN_NAME must be single directory components")
    return value


def output_directory(root, benchmark, model_name, run_name=None):
    family, *subset = BENCHMARKS[benchmark]["output"].split("/")
    path = Path(root) / family / validate_component(model_name)
    if subset:
        path /= subset[0]
    if run_name:
        path /= validate_component(run_name)
    return path


def load_tasks(bundle, benchmark):
    from lm_eval.tasks import TaskManager

    manager = TaskManager(include_defaults=False, include_path=str(bundle))
    loaded = manager.load([BENCHMARKS[benchmark]["task"]])
    leaves = loaded["tasks"]
    if len(leaves) != BENCHMARKS[benchmark]["leaves"]:
        raise ValueError(f"unexpected number of tasks for {benchmark}: {len(leaves)}")
    inventory = {}
    for name, task in leaves.items():
        # SQuAD is a Python task whose upstream constructor ignores most YAML overrides.
        task.set_config("num_fewshot", BENCHMARKS[benchmark]["shots"])
        task.set_fewshot_seed(42)
        docs = task.eval_docs
        if not len(docs):
            raise ValueError(f"empty evaluation split: {name}")
        if benchmark == "clue" and "label" in docs.column_names:
            if any(label < 0 for label in docs["label"]):
                raise ValueError(f"unlabelled CLUE data: {name}; validation is required")
        dataset = getattr(task, "dataset", {})
        inventory[name] = {"evaluation_documents": len(docs),
                           "num_fewshot": BENCHMARKS[benchmark]["shots"],
                           "dataset_path": task.DATASET_PATH, "dataset_name": task.DATASET_NAME,
                           "dataset_kwargs": task.config.dataset_kwargs,
                           "splits": {split: {"rows": len(data)} for split, data in dataset.items()}}
    return manager, loaded, inventory


def resolve_backend(args):
    if args.backend != "auto":
        return args.backend
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    return "hf" if config.get("model_type") == "qwen3_pgca" else "vllm"


def json_ready(value):
    """Keep harness numpy scalars serializable without permitting NaN in reports."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite harness result")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)  # Harness resolved task configs may contain callable descriptions.


def evaluate_one(args, benchmark, backend):
    from lm_eval import evaluator

    output = output_directory(args.output_root, benchmark, args.model_name, args.run_name).resolve()
    if output.is_relative_to(args.model.resolve()):
        raise ValueError("evaluation output must not be inside the model directory")
    output = new_run_directory(output)
    print(f"[evaluation][{benchmark}] output={output}", flush=True)
    model_identity = checkpoint_identity(args.model)
    if backend is not None and backend.checkpoint_identity != model_identity:
        raise RuntimeError("checkpoint changed between benchmarks; refusing to relabel an already loaded model")
    backend_name = resolve_backend(args)
    manifest = {"complete": False, "benchmark": benchmark, "protocol": BENCHMARKS[benchmark],
                "model": model_identity, "backend": backend_name,
                "max_length": args.max_length, "dtype": args.dtype, "device": args.device,
                "batch_size": args.batch_size, "limit": args.limit, "smoke_only": args.limit is not None,
                "seed": 42, "bootstrap_iters": 0 if args.limit else 1000,
                "apply_chat_template": False, "add_bos_token": False,
                "tensor_parallel_size": args.tensor_parallel_size,
                "overflow_policy": "error; no prompt/few-shot truncation",
                "HF_ENDPOINT": os.getenv("HF_ENDPOINT", "https://hf-mirror.com")}
    manifest_path = output / "manifest.json"
    write_report(manifest_path, manifest)
    started = time.monotonic()
    bundle = build_bundle(output / "tasks", benchmark)
    manager, loaded, inventory = load_tasks(bundle, benchmark)
    manifest["datasets"] = inventory
    write_report(manifest_path, manifest)
    if backend is None:
        if backend_name == "vllm":
            from .backend import build_vllm_backend
            backend = build_vllm_backend(
                args.model, dtype=args.dtype, batch_size=args.batch_size,
                max_length=args.max_length, tensor_parallel_size=args.tensor_parallel_size)
        else:
            from .backend import StrictHFLM
            from ..ppl.metrics import validate_window

            model, tokenizer, _ = load_model(str(args.model.resolve()), args.dtype, args.device, "sdpa")
            validate_window(model, tokenizer, args.max_length, 1)
            backend = StrictHFLM(pretrained=model, tokenizer=tokenizer, backend="causal",
                                 max_length=args.max_length, batch_size=args.batch_size,
                                 device=args.device, add_bos_token=False, softmax_dtype="float32")
            if backend.world_size != 1:
                raise ValueError("this launcher is single-process; do not use torchrun/accelerate launch")
        backend.checkpoint_identity = model_identity
    manifest["actual_model"] = {"backend_class": type(backend).__name__,
                                "tokenizer": type(backend.tokenizer).__name__}
    if backend_name == "hf":
        manifest["actual_model"].update(
            class_name=type(backend.model).__name__, dtype=str(backend.model.dtype),
            pgca_enabled=getattr(backend.model.config, "pgca_enabled", None),
            pgca_context_enabled=getattr(backend.model.config, "pgca_context_enabled", None),
            checkpoint_max_new_tokens_ignored=backend.checkpoint_max_new_tokens)
    write_report(manifest_path, manifest)
    # Pass already loaded root groups/tasks, retaining harness's own group aggregation.
    task_name = BENCHMARKS[benchmark]["task"]
    root = loaded.get("groups", {}).get(task_name) or loaded["tasks"].get(task_name)
    result = evaluator.simple_evaluate(
        model=backend, tasks=[root], task_manager=manager, num_fewshot=BENCHMARKS[benchmark]["shots"],
        limit=args.limit, log_samples=True, apply_chat_template=False, fewshot_as_multiturn=False,
        random_seed=42, numpy_random_seed=42, torch_random_seed=42, fewshot_random_seed=42,
        bootstrap_iters=0 if args.limit else 1000, cache_requests=False,
    )
    if result is None:
        raise RuntimeError("harness returned no results")
    if checkpoint_identity(args.model) != model_identity:
        raise RuntimeError("checkpoint changed during evaluation")
    result = json_ready(result)
    samples = result.pop("samples")
    samples_dir = output / "samples"
    samples_dir.mkdir()
    for name, rows in samples.items():
        with (samples_dir / f"{validate_component(name)}.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_report(output / "results.json", result)
    manifest.update(complete=True, elapsed_seconds=time.monotonic() - started)
    write_report(manifest_path, manifest)
    return backend, {"benchmark": benchmark, "output": str(output), "smoke_only": args.limit is not None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="all")
    parser.add_argument("--stage", choices=("inspect", "check", "run"), default="run")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--run-name")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--backend", choices=("auto", "hf", "vllm"), default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--limit", type=int, help="smoke only: examples per leaf task, NOT a corpus-wide limit")
    args = parser.parse_args(argv)
    benchmarks = select_benchmarks(args.benchmark)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_length < 2:
        parser.error("--max-length must be at least two")
    if args.tensor_parallel_size < 1:
        parser.error("--tensor-parallel-size must be positive")
    if args.stage == "run" and (args.model is None or args.model_name is None):
        parser.error("--model and --model-name are required for run")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("run this launcher directly, not via torchrun/accelerate")
    if args.stage == "run":
        validate_component(args.model_name)
        if args.run_name:
            validate_component(args.run_name)
        checkpoint_identity(args.model)
    backend = None
    summaries = []
    for benchmark in benchmarks:
        if args.stage == "run":
            backend, summary = evaluate_one(args, benchmark, backend)
        else:
            with tempfile.TemporaryDirectory(prefix="ccllm-lmeh-") as temporary:
                bundle = build_bundle(Path(temporary) / "tasks", benchmark)
                summary = {"benchmark": benchmark, **BENCHMARKS[benchmark]}
                if args.stage == "check":
                    _, _, summary["datasets"] = load_tasks(bundle, benchmark)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summaries


if __name__ == "__main__":
    main()
