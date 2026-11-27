"""Sequence P0-01 cells, keeping every model/framework attempt in its own version."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from scripts.evaluation.runs import new_run_directory
from .records import write_json
from .summarize import summarize

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def launch(command, output):
    print(f"[P0-01] log={output / 'worker.log'}", flush=True)
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            status_path = output / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if code and status.get("state") != "failed":
                write_json(status_path, {**status, "state": "failed", "exit_code": code})
            return code
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            write_json(output / "status.json", {"state": "interrupted"})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "outputs/train/Qwen3-1.7B-Base-INS/benchmark-checkpoint-1641")
    parser.add_argument("--shared", type=Path, default=ROOT / "outputs/train/Qwen3-1.7B-Base-PGCA-Shared-CPT-INS/benchmark-checkpoint-1000")
    parser.add_argument("--models", nargs="+", choices=("base", "shared"), default=["base", "shared"])
    parser.add_argument("--groups", nargs="+", choices=("A", "B", "C", "D"), default=["A", "B", "C", "D"])
    parser.add_argument("--run-dir", type=Path, help="retry selected cells in an existing P0-01 run")
    args = parser.parse_args()
    models = {"base": args.base.resolve(), "shared": args.shared.resolve()}
    for alias in args.models:
        model = models[alias]
        if not (model / "config.json").is_file():
            parser.error(f"{alias} full INS model config missing: {model / 'config.json'}")
        if not (model / "args.json").is_file():
            parser.error(f"{alias} saved INS template contract missing: {model / 'args.json'}")
        if not list(model.glob("*.safetensors")) and not list(model.glob("pytorch_model*.bin")):
            parser.error(f"{alias} full INS weights missing: {model}; use the merged model, not an adapter")
    if args.run_dir:
        run = args.run_dir.resolve()
        if not run.is_relative_to(HERE / "runs") or not (run / "experiment.json").exists():
            parser.error("--run-dir must name an existing experiment/p0_01_generation/runs/vN")
        saved = json.loads((run / "experiment.json").read_text())
        if saved["models"] != {k: str(v) for k, v in models.items()}:
            parser.error("model paths changed; start a new run for a different model pair")
    else:
        run = new_run_directory(HERE / "runs")
        (run / "inputs").mkdir()
        shutil.copyfile(HERE / "cases.jsonl", run / "inputs/cases.jsonl")
        write_json(run / "experiment.json", {
            "task": "P0-01", "models": {k: str(v) for k, v in models.items()},
            "gpu": os.getenv("CUDA_VISIBLE_DEVICES"), "python": sys.executable,
            "batch_size": 1, "seed": 42, "scope": "nine fixed generation cases; no task-score rerun",
            "groups": {"A": "Harness original backend: Base vLLM / Shared HF",
                       "B": "native HF, identical prompt IDs, native StopStringCriteria",
                       "C": "Swift TransformersEngine, injected identical prompt IDs",
                       "D": "Swift TransformersEngine, ordinary dialogue with saved INS template"},
        })
    print(f"[P0-01] run={run}", flush=True)
    failed = False
    try:
        for alias in args.models:
            versions = [p for p in (run / alias / "prepare").glob("v*") if p.name[1:].isdigit()]
            prepare_root = max(versions, key=lambda p: int(p.name[1:])) if versions else None
            status_file = prepare_root / "status.json" if prepare_root else None
            prepared = (status_file is not None and status_file.exists()
                        and json.loads(status_file.read_text()).get("state") == "complete"
                        and (prepare_root / "cases.jsonl").exists())
            if not prepared:
                prepare_root = new_run_directory(run / alias / "prepare")
                code = launch([sys.executable, "-B", "-m", "experiment.p0_01_generation.worker",
                               "--model", str(models[alias]), "--alias", alias, "--group", "prepare",
                               "--cases", str(run / "inputs/cases.jsonl"), "--output", str(prepare_root)], prepare_root)
                if code:
                    failed = True
                    continue
            for group in args.groups:
                output = new_run_directory(run / alias / group)
                code = launch([sys.executable, "-B", "-m", "experiment.p0_01_generation.worker",
                               "--model", str(models[alias]), "--alias", alias, "--group", group,
                               "--prepared", str(prepare_root), "--output", str(output)], output)
                failed = bool(code) or failed
    finally:
        summarize(run)
    print(f"[P0-01] summary={run / 'summary.md'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
