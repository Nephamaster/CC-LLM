"""Process launching, stage logs and checkpoint publication for training scripts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import wraps
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[2]


def option(arguments, *names, default=None):
    for name in names:
        if name in arguments:
            return arguments[arguments.index(name) + 1]
    return default


@contextmanager
def phase(name):
    started = time.monotonic()
    primary = os.environ.get("RANK", "0") == "0"
    if primary:
        print(f"[{name}] start", flush=True)
    try:
        yield
    finally:
        if primary:
            error = sys.exc_info()[1]
            failed = error is not None and not (isinstance(error, SystemExit) and error.code in (None, 0))
            status = "failed" if failed else "done"
            print(f"[{name}] {status}, {time.monotonic() - started:.1f}s", flush=True)


def worker(stage, method, arguments):
    sys.path.insert(0, str(ROOT))
    if method == "python":
        sys.argv = arguments
        with phase(stage):
            runpy.run_path(arguments[0], run_name="__main__")
        return
    sys.argv = [f"swift.{method}", *arguments]
    if method in {"pt", "sft"} and os.environ.get("RANK", "0") == "0":
        print(f"[{stage}] distributed world_size={os.environ.get('WORLD_SIZE', '1')} "
              f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}", flush=True)
    with phase(f"{stage}/import"):
        if method in {"pt", "sft"}:
            from swift.cli.utils import try_use_single_device_mode
            try_use_single_device_mode()
        if method == "sft":
            from swift.cli.sft import try_init_unsloth
            try_init_unsloth()
            from swift.ray_utils import try_init_ray
            try_init_ray()
        from swift.pipelines.train.sft import SwiftSft
        pipelines = importlib.import_module("swift.pipelines")
        entry = getattr(pipelines, {"pt": "pretrain_main", "sft": "sft_main", "export": "export_main"}[method])

    def measured(function, label):
        @wraps(function)
        def call(*args, **kwargs):
            with phase(f"{stage}/{label}"):
                return function(*args, **kwargs)
        return call

    for name, label in (("_prepare_model_tokenizer", "model"), ("_get_dataset", "load"),
                        ("_encode_dataset", "encode"), ("train", "train")):
        setattr(SwiftSft, name, measured(getattr(SwiftSft, name), label))
    with phase(stage):
        entry(arguments)


def oom_kills():
    try:
        values = dict(line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines())
        return int(values["oom_kill"])
    except (OSError, KeyError, ValueError):
        return None


def launch(stage, method, arguments):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONFAULTHANDLER="1",
                       CC_LLM_RUN_STAGE=stage)
    if not environment.get("MASTER_PORT"):
        environment.pop("MASTER_PORT", None)
    environment["CC_LLM_INPUT_MODEL"] = option(arguments, "--model", default="")
    if method not in {"pt", "sft"}:
        environment.pop("NPROC_PER_NODE", None)
    command = [sys.executable, "-u", "-B"]
    if method in {"pt", "sft"}:
        command += ["-m", "torch.distributed.run", "--nproc_per_node", environment.get("NPROC_PER_NODE", "1")]
        if environment.get("NNODES", "1") == "1" and not environment.get("MASTER_PORT"):
            command += ["--standalone", "--nnodes=1"]
        else:
            for name in ("NNODES", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
                if environment.get(name):
                    command += [f"--{name.lower()}", environment[name]]
    command += [str(Path(__file__).resolve()), "_worker", method, stage, *arguments]

    output = option(arguments, "--output_dir", "--output-dir", "--output", default=str(ROOT / "outputs/train"))
    log_dir = Path(output) if method in {"pt", "sft"} else Path(output).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=stage.replace("/", "-") + "-", suffix=".log",
                                     dir=log_dir, delete=False) as log:
        def emit(message):
            print(message, flush=True)
            log.write((message + "\n").encode())
            log.flush()

        emit(f"[{stage}] python={sys.executable} ({sys.version.split()[0]})")
        emit(f"[{stage}] model={environment['CC_LLM_INPUT_MODEL'] or '-'} output={output}")
        data = option(arguments, "--cached_dataset", "--dataset", "--input", "--data-dir", default="-")
        emit(f"[{stage}] data={data} log={log.name}")
        if method in {"pt", "sft"}:
            processes = environment.get("NPROC_PER_NODE", "1")
            nodes = environment.get("NNODES", "1")
            batch = option(arguments, "--per_device_train_batch_size", default="1")
            accumulation = option(arguments, "--gradient_accumulation_steps", default="1")
            world = int(nodes) * int(processes) if nodes.isdigit() and processes.isdigit() else "elastic"
            effective = world * int(batch) * int(accumulation) if isinstance(world, int) else "elastic"
            emit(f"[{stage}] GPUs={environment.get('CUDA_VISIBLE_DEVICES', 'all')} world_size={world} "
                 f"batch/GPU={batch} accumulation={accumulation} effective_batch={effective} "
                 f"port={environment.get('MASTER_PORT') or 'automatic'}")
        before, started = oom_kills(), time.monotonic()
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)

        def forward(signum, frame):
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

        handlers = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}

        def copy_output():
            while chunk := process.stdout.read1(65536):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                log.write(chunk)
                log.flush()

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        try:
            code = process.wait()
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            reader.join()
            process.stdout.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
        after = oom_kills()
        delta = after - before if before is not None and after is not None else "unavailable"
        status = signal.Signals(-code).name if code < 0 else f"exit_code={code}"
        emit(f"[{stage}] {status}, elapsed={time.monotonic() - started:.1f}s, "
             f"container_oom_kill_delta={delta} (container-wide, not PID attribution)")
    return 128 - code if code < 0 else code


def finite_number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def validation_result(args, metrics):
    metric = args.metric_for_best_model or "loss"
    metric = metric if metric.startswith("eval_") else f"eval_{metric}"
    count = metrics.get("eval_clm_tokens", metrics.get("eval_supervised_tokens"))
    value = finite_number(metrics.get(metric))
    status = "finite" if value is not None else "missing or non-finite"
    if count is not None:
        count = finite_number(count)
        if count is None or count < 0:
            status = "invalid target count"
        elif count == 0:
            status = "no supervised targets"
    return metric, value, count, status


def report_validation(args, metrics):
    metric, _, count, status = validation_result(args, metrics)
    print(f"[validation] {metric}={metrics.get(metric)} supervised_tokens={count} status={status}", flush=True)


class TokenCountMetric:
    def __init__(self):
        self.reset()

    def update(self, count):
        self.total += int(count)

    def compute(self):
        return self.total

    def reset(self):
        self.total = 0


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def publish_checkpoint(args, state, output_root=None, *, prefer_last=False):
    run = Path(args.output_dir).resolve()
    checkpoints = [path for path in run.glob("checkpoint-*")
                   if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()]
    last = max(checkpoints, key=lambda path: int(path.name.split("-")[-1])) if checkpoints else None
    if last is None:
        raise RuntimeError("training ended without a saved checkpoint")
    metric, _, _, _ = validation_result(args, {})
    evaluations = [row for row in state.log_history if metric in row]
    results = [validation_result(args, row) for row in evaluations]
    best = Path(state.best_model_checkpoint).resolve() if state.best_model_checkpoint else None
    score = finite_number(state.best_metric)
    best_step = int(best.name.split("-")[-1]) if best and best.name.split("-")[-1].isdigit() else None
    best_rows = [row for row in evaluations if row.get("step") == best_step]
    valid_targets = not best_rows or validation_result(args, best_rows[-1])[3] == "finite"
    use_best = not prefer_last and best is not None and best.is_dir() and score is not None and valid_targets
    selected = best if use_best else last
    reason = ("Phase2-A selects the last checkpoint" if prefer_last else
              "finite best metric" if use_best else "no usable finite best checkpoint; use last")
    report = dict(checkpoint=str(selected), selection="best" if use_best else "last", reason=reason,
                  metric=metric, best_metric=score, evaluations=len(results),
                  valid_evaluations=sum(row[3] == "finite" for row in results),
                  last_effective_targets=finite_number(results[-1][2]) if results else None,
                  input_model=os.environ.get("CC_LLM_INPUT_MODEL"), global_step=state.global_step)
    atomic_write(run / "checkpoint_selection.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for directory in dict.fromkeys([run, Path(output_root).resolve() if output_root else run]):
        atomic_write(directory / "next_stage_model.txt", str(selected) + "\n")
    print(f"[checkpoint] {report['selection']}: {selected}; {reason}", flush=True)
    return report


if __name__ == "__main__":
    if sys.argv[1] == "_worker":
        worker(sys.argv[3], sys.argv[2], sys.argv[4:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("stage")
        parser.add_argument("method", choices=("pt", "sft", "export", "python"))
        parser.add_argument("arguments", nargs=argparse.REMAINDER)
        args = parser.parse_args()
        raise SystemExit(launch(args.stage, args.method, args.arguments))
