"""Build real models, then smoke-test the three experiment routes on dedicated GPUs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time

from scripts.evaluation.runs import latest_run_directory, new_run_directory

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("phase1", "phase2", "instruction_sft", "csc_sft", "cgec_sft", "ancient_sft", "ppl",
         "csc", "fcgec_valid", "fcgec_test", "nacgec", "nacgec_test", "c3", "fuxi")
BENCHMARKS = ("mmlu", "arc", "squadv2", "gsm8k", "cmmlu", "clue")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Smoke:
    def __init__(self, args):
        self.args = args
        self.output = new_run_directory(args.output_root)
        self.cases = []
        self.lock = threading.RLock()
        self.processes = set()
        self.cancelled = threading.Event()
        self.finished = False
        self.devices = queue.Queue()
        for device in args.devices.split(","):
            self.devices.put(device.strip())
        self.base_name = args.base_model.name
        self.models = {"base": args.base_model,
                       "char": self.output / "models" / f"{self.base_name}-Char",
                       "pgca": self.output / "models" / f"{self.base_name}-PGCA"}

    def report(self):
        with self.lock:
            required = [c for c in self.cases if c.get("required", True)]
            failures = any(c["status"] in {"failed", "blocked"} for c in required)
            pending = any(c["status"] == "pending" for c in self.cases)
            running = any(c["status"] == "running" for c in self.cases)
            status = "running" if not self.finished or running else "failed" if failures else "partial" if pending else "passed"
            result = {"smoke_only": True, "status": status, "base_model": str(self.args.base_model),
                      "required_passed": self.finished and not failures and not running,
                      "complete": status == "passed", "cases": self.cases,
                      "not_covered": ["full-data memory use and throughput", "long-run model quality/EOS learning",
                                      "multi-GPU DDP", "optimizer-state resume (model-only smoke checkpoints)"]}
            text = json.dumps(result, ensure_ascii=False, indent=2)
            temporary = self.output / ".summary.json.tmp"
            temporary.write_text(text + "\n", encoding="utf-8")
            temporary.replace(self.output / "summary.json")
            lines = ["# 集中冒烟结果", "", f"状态：{status}；仅少量真实数据，不代表正式实验分数。", "",
                     "| 项目 | GPU | 状态 | 秒 | 说明 / 日志 |", "| --- | --- | --- | --- | --- |"]
            for case in self.cases:
                note = case.get("note", "").replace("|", "/").replace("\n", " ")[:240]
                log = case.get("log")
                if log:
                    note += f" [日志]({log})"
                lines.append(f"| {case['name']} | {case.get('gpu', '—')} | {case['status']} | {case.get('seconds', '')} | {note} |")
            (self.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            return result

    def add(self, name, status, note, *, required=True):
        with self.lock:
            self.cases.append(dict(name=name, status=status, note=note, required=required))
            self.report()
        print(f"[smoke] {status.upper()} {name}: {note}", flush=True)
        return False

    def environment(self, gpu=None, **values):
        env = dict(os.environ)
        # Do not inherit stage pointers, distributed ranks, or a judge configured only halfway.
        for key in ("MODEL", "MODEL_NAME", "MODEL_VARIANT", "OUT", "RUN_DIR", "STAGE", "STAGES", "CACHE_DIR",
                    "RAW_CACHE", "ALIGN_OUT", "CPT_ROOT", "ADAPTATION_OUT", "RESUME_FROM_CHECKPOINT",
                    "PHASE1_MODEL", "PHASE2_A_MODEL", "PHASE2_B_MODEL", "CPT_MODEL", "CSC_MODEL", "CGEC_MODEL",
                    "ANCIENT_MODEL", "LORA", "PGCA_ENABLED", "CC_LLM_STAGE", "RANK", "LOCAL_RANK", "WORLD_SIZE",
                    "MASTER_PORT", "MASTER_ADDR", "NODE_RANK", "NNODES", "NPROC_PER_NODE", "CC_LLM_MACRO_CHANNELS",
                    "DEV_CHECK_DEV", "DEV_CHECK_DEVICE", "DEV_CHECK_REPORT_ROOT",
                    "INSTRUCTION_MODEL", "INSTRUCTION_DATA", "INS_OUT", "SUPERVISED_TOKENS"):
            env.pop(key, None)
        env.update(PYTHONDONTWRITEBYTECODE="1", HF_ENDPOINT=env.get("HF_ENDPOINT", "https://hf-mirror.com"),
                   CUDA_VISIBLE_DEVICES="" if gpu is None else gpu, CC_ROOT=str(ROOT), TOKENIZERS_PARALLELISM="false")
        env.update({key: str(value) for key, value in values.items()})
        return env

    @staticmethod
    def stop(process):
        try:
            os.killpg(process.pid, signal.SIGTERM if process.poll() is None else signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            for process in list(self.processes):
                self.stop(process)

    def step(self, name, command, env, *, needs=True, check=None):
        if not needs or self.cancelled.is_set():
            return self.add(name, "blocked", "前置步骤未通过或已中断")
        log = self.output / "logs" / (name.replace("/", "--") + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        case = dict(name=name, status="running", gpu=env.get("CUDA_VISIBLE_DEVICES") or "CPU", log=str(log))
        with self.lock:
            self.cases.append(case)
            self.report()
        print(f"[smoke] RUN {name} GPU={case['gpu']} log={log}", flush=True)
        started = time.monotonic()
        try:
            with log.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                with self.lock:
                    self.processes.add(process)
                    if self.cancelled.is_set():
                        self.stop(process)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=15)
                            break
                        except subprocess.TimeoutExpired:
                            print(f"[smoke] RUNNING {name} {time.monotonic()-started:.0f}s", flush=True)
                finally:
                    self.stop(process)  # Also release any surviving workers from this job's process group.
                    with self.lock:
                        self.processes.discard(process)
                if code:
                    raise RuntimeError(f"进程退出码 {code}；查看日志")
                if check:
                    check()
            case.update(status="passed", note="命令与产物检查通过")
        except KeyboardInterrupt:
            case.update(status="failed", note="用户中断")
            raise
        except Exception as error:
            case.update(status="failed", note=str(error))
        finally:
            with self.lock:
                case["seconds"] = round(time.monotonic() - started, 1)
                self.report()
            print(f"[smoke] {case['status'].upper()} {name} {case['seconds']}s", flush=True)
        return case["status"] == "passed"

    def checkpoint(self, output, adapter=False):
        path = Path((output / "next_stage_model.txt").read_text().strip())
        if not (path / ("adapter_config.json" if adapter else "config.json")).is_file():
            raise ValueError(f"检查点未完整发布：{path}")
        report = read_json(path.parent / "checkpoint_selection.json")
        if (report["global_step"] < self.args.train_steps or not report["valid_evaluations"]
                or report["valid_evaluations"] != report["evaluations"]):
            raise ValueError("未完成指定训练步数，或没有有限的验证 loss")
        if adapter and read_json(path.parent / "finetune_report.json")["auxiliary_loss_enabled"]:
            raise ValueError("下游辅助损失未关闭")
        return path

    def instruction_checkpoint(self, output):
        path = Path((output / "next_stage_model.txt").read_text().strip())
        report = read_json(path.parent / "instruction_selection.json")
        if not report["complete"] or not (path / "config.json").is_file():
            raise ValueError("General-SFT 合并模型或验收报告不完整")
        adapter = Path(report["selected_adapter"])
        state = read_json(adapter.parent / "checkpoint_selection.json")
        if (state["global_step"] < self.args.train_steps or not state["valid_evaluations"]
                or state["valid_evaluations"] != state["evaluations"]):
            raise ValueError("General-SFT 未完成指定训练步数或验证 loss 不全为有限值")
        return path

    def train(self, variant, gpu, stage, model, prepared):
        name = self.models[variant].name
        root = self.output / "training"
        cpt = root / f"{name}-CPT"
        ins = root / f"{name if variant == 'base' else name+'-CPT'}-INS"
        outputs = {"phase1": root / f"{name}-ALIGN", "cpt": cpt, "instruction": ins,
                   "phase2_a": cpt / "adaptation", "phase2_b": cpt / "joint"}
        suffix = {"csc": "CSC", "cgec": "CGEC", "ancient": "ACT"}
        adapter = stage in suffix
        if adapter:
            outputs[stage] = root / f"{name if variant == 'base' else name+'-CPT'}-INS-{suffix[stage]}"
        output = outputs[stage]
        values = dict(MODEL_VARIANT=variant, MODEL_NAME=name, PYTHON_CMD=self.args.train_python, TRAIN_ROOT=root,
                      OUT=output, STAGES=stage, FORCE="true", NPROC_PER_NODE=1, TRAIN_BATCH_SIZE=1,
                      EVAL_BATCH_SIZE=1, GRAD_ACCUMULATION=1, DATASET_NUM_PROC=1, MAX_STEPS=self.args.train_steps,
                      EVAL_STEPS=self.args.train_steps, SAVE_STEPS=self.args.train_steps, LOGGING_STEPS=1,
                      SAVE_TOTAL_LIMIT=1, SAVE_ONLY_MODEL="true", LOAD_FROM_CACHE_FILE="false",
                      DEV_CHECK_SAMPLES=1, DEV_CHECK_SAVE_SAMPLES=1, DEV_CHECK_MAX_NEW_TOKENS=self.args.max_new_tokens,
                      ALIGN_OUT=root / f"{name}-ALIGN", CPT_ROOT=cpt, ADAPTATION_OUT=cpt / "adaptation")
        if stage == "instruction":
            values.update(INSTRUCTION_MODEL=model or "", INSTRUCTION_DATA=self.output / "data/instruction_sft",
                          INS_OUT=ins, SUPERVISED_TOKENS=0, NUM_TRAIN_EPOCHS=1, MAX_LENGTH=4096,
                          DEV_CHECK_MODE="report")
        elif adapter:
            values.update({f"{stage.upper()}_MODEL": model or "", "NUM_TRAIN_EPOCHS": 1,
                           "MAX_NEW_TOKENS": self.args.max_new_tokens,
                           "MAX_LENGTH": 2048 if stage == "ancient" else 768})
            data = self.output / "data" / f"{stage}_sft"
            if stage == "ancient":
                values.update(ANCIENT_DATA_DIR=data, ANCIENT_VIEW_DIR=output / ".inputs")
            else:
                values.update(TRAIN=data / "train.jsonl", DEV=data / "dev.jsonl")
        else:
            phase = "phase1" if stage == "phase1" else "phase2"
            input_key = {"phase1": "PHASE1_MODEL", "cpt": "CPT_MODEL", "phase2_a": "PHASE2_A_MODEL",
                         "phase2_b": "PHASE2_B_MODEL"}[stage]
            values.update({input_key: model or "", "DATA_DIR": self.output / "data" / phase,
                           "MAX_LENGTH": self.args.cpt_max_length,
                           "CACHE_DIR": self.output / "cache" / variant / phase,
                           "PHASE1_TOKENS": 0, "PHASE2_TOKENS": 0, "ADAPTATION_TOKENS": 512})
        get_checkpoint = (lambda: self.instruction_checkpoint(output)) if stage == "instruction" else (lambda: self.checkpoint(output, adapter))
        ok = self.step(f"{variant}/train/{stage}", ["bash", str(ROOT / "scripts/train/run.sh")],
                       self.environment(gpu, **values), needs=model is not None and prepared,
                       check=get_checkpoint)
        return get_checkpoint() if ok else None

    def ppl(self, variant, gpu, model, name, available):
        output = self.output / "evaluation/ppl" / name

        def check():
            report = read_json(latest_run_directory(output) / "metrics.json")
            if not report["complete"] or len(report["domains"]) != 6:
                raise ValueError("PPL 未完成六域评估")

        self.step(f"{variant}/ppl/{name}", ["bash", str(ROOT / "scripts/evaluation/ppl/run.sh")],
                  self.environment(gpu, MODEL=model or "", MODEL_NAME=name, PYTHON=self.args.eval_python,
                                   DATA_DIR=self.output / "data/ppl", OUT=output, WINDOW=128, STRIDE=64),
                  needs=model is not None and available, check=check)

    def local_evaluation(self, variant, gpu, model, adapter, kind, prepared):
        output = self.output / "evaluation" / variant / kind
        output.mkdir(parents=True, exist_ok=True)
        if prepared:
            for name in ("input.jsonl", "input.manifest.json"):
                shutil.copy2(self.output / "data" / kind / name, output / name)
        task = "csc" if kind == "csc" else "ancient" if kind in {"c3", "fuxi"} else "cgec"
        model_name = self.models[variant].name + ("" if variant == "base" else "-CPT") + "-INS"
        suffix = {"csc": "CSC", "cgec": "CGEC", "ancient": "ACT"}[task]
        values = dict(MODEL=model or "", LORA=adapter or "", MODEL_NAME=f"{model_name}-{suffix}",
                      PYTHON=self.args.eval_python, RUN_DIR=output, STAGE="all", BATCH_SIZE=1,
                      MAX_NEW_TOKENS=self.args.max_new_tokens, INFERENCE_BACKEND=self.args.backend,
                      DEV_CHECK_MODE="report", DEV_CHECK_SAMPLES=1, DEV_CHECK_MAX_NEW_TOKENS=self.args.max_new_tokens,
                      TENSOR_PARALLEL_SIZE=1, FEWSHOT=0)
        if kind.startswith("fcgec_"):
            values.update(DATASET="fcgec", SPLIT=kind.removeprefix("fcgec_"))
        elif kind in {"nacgec", "nacgec_test"}:
            values.update(DATASET="nacgec", SPLIT="dev" if kind == "nacgec" else "test")
        elif task == "ancient":
            values.update(DATASET=kind)
        env = self.environment(gpu, **values)
        judge = bool(env.get("JUDGE_MODEL") and env.get("JUDGE_BASE_URL"))
        if not judge:
            env.pop("JUDGE_MODEL", None)
            env.pop("JUDGE_BASE_URL", None)

        def check():
            if not read_json(output / "predictions.manifest.json")["complete"]:
                raise ValueError("推理有失败样本")
            report = read_json(output / "metrics.json")
            if kind in {"fcgec_test", "nacgec_test"} and report.get("status") != "exported_unlabeled_test":
                raise ValueError("此 test 路线应只导出提交文件")
            if kind == "fuxi" and judge and not report["complete"]:
                raise ValueError("Fuxi 裁判未完成全部少量样本")

        ok = self.step(f"{variant}/evaluate/{kind}", ["bash", str(ROOT / f"scripts/evaluation/{task}/run.sh")],
                       env, needs=model is not None and adapter is not None and prepared, check=check)
        if kind == "fuxi" and ok and not judge:
            self.add(f"{variant}/evaluate/fuxi_judge", "pending", "未配置 JUDGE_MODEL/JUDGE_BASE_URL；自动评分已检查，裁判未测", required=False)

    def route(self, variant, built, prepared, train_ok, eval_ok):
        gpu = self.devices.get()
        try:
            initial = self.models[variant] if variant == "base" or built else None
            model = initial
            name = self.models[variant].name
            print(f"[smoke] {variant} reserves GPU {gpu}", flush=True)
            if variant in {"base", "char"}:
                self.ppl(variant, gpu, initial, name, eval_ok and prepared["ppl"])
            stages = [] if variant == "base" else ["phase1", "cpt"] if variant == "char" else ["phase1", "phase2_a", "phase2_b"]
            for stage in stages:
                model = self.train(variant, gpu, stage, model,
                                   train_ok and prepared["phase1" if stage == "phase1" else "phase2"])
                if variant == "pgca" and stage == "phase1":
                    self.ppl(variant, gpu, model, name + "-ALIGN", eval_ok and prepared["ppl"])
            final_name = name if variant == "base" else name + "-CPT"
            if stages:
                self.ppl(variant, gpu, model, final_name, eval_ok and prepared["ppl"])
            model = self.train(variant, gpu, "instruction", model, train_ok and prepared["instruction_sft"])
            final_name += "-INS"
            adapters = {task: self.train(variant, gpu, task, model, train_ok and prepared[task + "_sft"])
                        for task in ("csc", "cgec", "ancient")}
            for benchmark in BENCHMARKS:
                family = "arc" if benchmark == "arc" else benchmark
                output = self.output / "evaluation/lmeh" / family / final_name
                if benchmark == "arc":
                    output /= "challenge"

                def check(directory=output):
                    run = latest_run_directory(directory)
                    if not read_json(run / "manifest.json")["complete"] or not (run / "table_metrics.json").is_file():
                        raise ValueError("LMEH 评估或表格指标未完成")

                self.step(f"{variant}/lmeh/{benchmark}", ["bash", str(ROOT / "scripts/evaluation/lmeh/run.sh")],
                          self.environment(gpu, MODEL=model or "", MODEL_NAME=final_name, PYTHON=self.args.eval_python,
                                           OUT=self.output / "evaluation/lmeh", BENCHMARK=benchmark, STAGE="run",
                                           LIMIT=self.args.eval_samples, MAX_LENGTH=8192, BATCH_SIZE=1,
                                           EVAL_MODE="instruct",
                                           INFERENCE_BACKEND=self.args.backend, TENSOR_PARALLEL_SIZE=1),
                          needs=model is not None and eval_ok, check=check)
            for kind in ("csc", "fcgec_valid", "fcgec_test", "nacgec", "nacgec_test", "c3", "fuxi"):
                task = "csc" if kind == "csc" else "ancient" if kind in {"c3", "fuxi"} else "cgec"
                self.local_evaluation(variant, gpu, model, adapters[task], kind, eval_ok and prepared[kind])
        finally:
            self.devices.put(gpu)

    def run(self):
        print(f"[smoke] output={self.output}", flush=True)
        probe = ("import importlib.util as u, sys, torch; "
                 "print(sys.executable, sys.version); print('torch', torch.__version__); "
                 "assert torch.cuda.is_available(), 'CUDA unavailable in this environment'; "
                 "missing=[m for m in sys.argv[1:] if u.find_spec(m) is None]; "
                 "assert not missing, 'Missing packages: '+','.join(missing)")
        gpu = self.args.devices.split(",")[0].strip()
        train_ok = self.step("environment/train", [str(self.args.train_python), "-B", "-c", probe,
                                                   "transformers", "swift", "datasets", "peft", "pypinyin"], self.environment(gpu))
        eval_ok = self.step("environment/evaluation", [str(self.args.eval_python), "-B", "-c", probe,
                                                      "transformers", "swift", "lm_eval", "vllm", "sacrebleu", "thulac", "opencc", "Levenshtein"], self.environment(gpu))
        prepared = {}
        for kind in KINDS:
            prepared[kind] = self.step("prepare/" + kind, [str(self.args.train_python), "-B", "-m", "scripts.smoke.prepare",
                                        "--kind", kind, "--data-root", str(self.args.data_root),
                                        "--output", str(self.output / "data" / kind),
                                        "--train-samples", str(self.args.train_samples), "--eval-samples", str(self.args.eval_samples)],
                                       self.environment())
        built = self.step("build/char_and_pgca", ["bash", str(ROOT / "scripts/model_build/run.sh")],
                          self.environment(BASE_MODEL=self.args.base_model, BASE_MODEL_NAME=self.base_name,
                                           MODEL_ROOT=self.output / "models", CHAR_MODEL=self.models["char"],
                                           PGCA_MODEL=self.models["pgca"], DATA_ROOT=self.args.data_root,
                                           BUILD_RESOURCES=self.output / "models/.build/hanzi", PYTHON_CMD=self.args.train_python),
                          needs=train_ok)
        executor = ThreadPoolExecutor(max_workers=min(3, self.devices.qsize()))
        try:
            jobs = [executor.submit(self.route, variant, built, prepared, train_ok, eval_ok)
                    for variant in ("base", "char", "pgca")]
            for variant, job in zip(("base", "char", "pgca"), jobs):
                try:
                    job.result()
                except Exception as error:
                    self.add(variant + "/runner", "failed", str(error))
        except KeyboardInterrupt:
            self.cancel()
            self.add("interrupted", "failed", "用户中断，已停止本次冒烟子进程")
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        self.finished = True
        report = self.report()
        print(f"[smoke] {report['status'].upper()} {dict(Counter(c['status'] for c in self.cases))}", flush=True)
        print(f"[smoke] report={self.output / 'summary.md'}", flush=True)
        return 130 if self.cancelled.is_set() else 0 if report["required_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--train-python", type=Path, required=True)
    parser.add_argument("--eval-python", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/smoke")
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--cpt-max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--backend", choices=("auto", "hf", "vllm"), default="auto")
    args = parser.parse_args()
    for name in ("base_model", "data_root", "output_root"):
        setattr(args, name, getattr(args, name).resolve())
    # Resolving the executable symlink bypasses a venv and loads the base Python's packages.
    for name in ("train_python", "eval_python"):
        setattr(args, name, getattr(args, name).absolute())
    devices = args.devices.split(",")
    if any(not d.strip() for d in devices) or len(set(d.strip() for d in devices)) != len(devices):
        parser.error("devices must be a nonempty list without repeated GPUs")
    if min(args.train_steps, args.train_samples, args.eval_samples, args.cpt_max_length, args.max_new_tokens) < 1:
        parser.error("steps, samples and lengths must be positive")
    if not (args.base_model / "config.json").is_file():
        parser.error("base-model must be a local full HF model directory")
    for python in (args.train_python, args.eval_python):
        if not os.access(python, os.X_OK):
            parser.error(f"missing Python environment: {python}; run scripts/environment/run.sh first")
    smoke = Smoke(args)
    try:
        code = smoke.run()
    except KeyboardInterrupt:
        smoke.cancel()
        smoke.add("interrupted", "failed", "用户中断，已停止本次冒烟子进程")
        smoke.finished = True
        smoke.report()
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
