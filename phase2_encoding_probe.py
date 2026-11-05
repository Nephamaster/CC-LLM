import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

parser = argparse.ArgumentParser(description="Phase2 数据编码临时诊断；不启动训练。")
parser.add_argument("--full-input", action="store_true",
                    help="加载全部原始分片，再仅编码训练前 1000 条、验证前 8 条；batch=1000。")
parser.add_argument("--mode", choices=("main", "pool"), default="pool",
                    help="--full-input 的编码方式：main 为当前主进程，pool 为单个子进程。")
options = parser.parse_args()
root = Path.cwd()
import_timeout = float(os.environ.get("PROBE_IMPORT_TIMEOUT", "900"))
export_timeout = float(os.environ.get("PROBE_EXPORT_TIMEOUT", "900"))
encoding_timeout = float(os.environ.get("PROBE_ENCODING_TIMEOUT", "300"))
print("诊断脚本:", Path(__file__).resolve(), flush=True)
print(f"阶段时限: 导入={import_timeout:g}秒, 初始化/导出={export_timeout:g}秒, 编码={encoding_timeout:g}秒", flush=True)
print("输入:", "加载全量分片后仅编码首批" if options.full_input else "单文件前 1000 条小样本对照", flush=True)
if options.full_input:
    print("首批编码方式:", options.mode, "batch=1000", flush=True)
name = os.environ["MODEL_NAME"]
pointer = root / "outputs/train" / f"{name}-ALIGN/next_stage_model.txt"
model = pointer.read_text().strip()
work = Path(tempfile.mkdtemp(prefix="encoding-check-", dir=root / "outputs"))

input_files = {}
for split, count in (("train", 1000), ("validation", 8)):
    sources = sorted((root / "data/pretrain/phase2" / split).glob("*.jsonl"))
    if options.full_input:
        input_files[split] = [str(path) for path in sources]
    else:
        sample = work / f"{split}.jsonl"
        with sources[0].open(encoding="utf-8") as src, sample.open("w", encoding="utf-8") as dst:
            dst.writelines(itertools.islice(src, count))
        input_files[split] = [str(sample)]
    print(split, "输入文件数:", len(sources) if options.full_input else 1, flush=True)

program = r'''
import faulthandler
import json
import os
import resource
import sys
import time
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
mode, batch, status_path = sys.argv[1:4]
status_path = Path(status_path)
state = {"encoding_started": False, "encodings": []}
def mark(stage, note):
    if state.get("stage") != stage:
        state["phase_started"] = time.monotonic()
    state.update(stage=stage, note=note,
                 process_peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
                 cgroup_memory_gib=round(int(Path("/sys/fs/cgroup/memory.current").read_text()) / 1024**3, 3))
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state))
    temporary.replace(status_path)
    print("[probe]", json.dumps(state), flush=True)

mark("imports", "import swift.dataset")
started = time.monotonic()
from swift.dataset import EncodePreprocessor
mark("imports", "import export pipeline")
from swift.pipelines.export.export import export_main
state["import_seconds"] = round(time.monotonic() - started, 1)
mark("export", "initialize arguments, tokenizer and dataset")

original = EncodePreprocessor.__call__
def controlled(self, dataset, **kwargs):
    kwargs["num_proc"] = None if mode == "main" else 1
    kwargs["batch_size"] = int(batch)
    input_rows = len(dataset)
    if os.environ["PROBE_FULL_INPUT"] == "true":
        mark("export", f"full dataset loaded: rows={input_rows}; select first encoding batch")
        limit = 1000 if not state["encodings"] else 8
        dataset = dataset.select(range(min(limit, input_rows)))
    state["encodings"].append(dict(input_rows=input_rows, selected_rows=len(dataset),
                                   num_proc=kwargs["num_proc"], batch_size=int(batch)))
    state["encoding_started"] = True
    mark("encoding", f"rows={len(dataset)}, num_proc={kwargs['num_proc']}, batch_size={batch}")
    print("[encoding]", len(dataset), kwargs, flush=True)
    result = original(self, dataset, **kwargs)
    state["encodings"][-1]["output_rows"] = len(result)
    mark("export", f"encoded_rows={len(result)}; continue export")
    return result
EncodePreprocessor.__call__ = controlled

if __name__ == "__main__":
    export_main(sys.argv[4:])
    mark("done", "export completed")
'''
arguments = [
    "--external_plugins", str(root / "scripts/train/cpt/plugin.py"),
    "--model", model, "--dataset", *input_files["train"],
    "--val_dataset", *input_files["validation"],
] + """--model_type qwen3 --template cc_llm_multitask --load_args false
--dataset_num_proc 1 --data_seed 42 --to_cached_dataset true --strict true
--load_from_cache_file false --split_dataset_ratio 0 --truncation_strategy split
--max_length 4096 --use_chat_template false --loss_scale all
--disable_ignore_empty_think true""".split()
environment = dict(os.environ, CC_LLM_STAGE="phase2a", PGCA_ENABLED="true",
                   PYTHONDONTWRITEBYTECODE="1", PROBE_FULL_INPUT=str(options.full_input).lower())
environment.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
environment.pop("NPROC_PER_NODE", None)

def oom_kills():
    events = Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
    return int(dict(line.split() for line in events)["oom_kill"])

results = []
print("报告目录:", work, flush=True)
cases = ((options.mode, 1000),) if options.full_input else (("main", 16), ("pool", 16), ("main", 1000))
for mode, batch in cases:
    case = f"{mode}-batch{batch}"
    log_path = work / f"{case}.log"
    status_path = work / f"{case}.status.json"
    before, started, timed_out = oom_kills(), time.monotonic(), False
    status = {"stage": "starting", "phase_started": started, "encoding_started": False}
    previous_note = None
    print("开始:", case, "日志:", log_path, flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-B", "-c", program, mode, str(batch), str(status_path),
             *arguments, "--output_dir", str(work / case)],
            env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while True:
            if status_path.exists():
                status = json.loads(status_path.read_text())
            note = (status["stage"], status.get("note"))
            if note != previous_note:
                print("阶段:", case, *note, flush=True)
                previous_note = note
            if process.poll() is not None:
                break
            timeout = (encoding_timeout if status["stage"] == "encoding" else
                       import_timeout if status["stage"] in {"starting", "imports"} else export_timeout)
            if time.monotonic() - status["phase_started"] >= timeout:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        returncode = process.wait()
        if returncode:
            # The probe owns this session; clean up pool children if their parent crashed.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    result = dict(case=case, returncode=returncode, timed_out=timed_out,
                  seconds=round(time.monotonic() - started, 1),
                  stage=status["stage"], encoding_started=status["encoding_started"],
                  import_seconds=status.get("import_seconds"),
                  timeout_stage=status["stage"] if timed_out else None,
                  encodings=status.get("encodings", []),
                  process_peak_rss_gib=status.get("process_peak_rss_gib"),
                  oom_kill_delta=oom_kills() - before, log=str(log_path))
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if returncode:
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-35:]), flush=True)
    (work / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    if result["oom_kill_delta"] > 0:
        print("本组期间 OOM 计数增加，停止后续组；需结合日志判断归属。", flush=True)
        break
    if returncode and not result["encoding_started"]:
        print("本组尚未进入编码，停止后续组，先检查导入或初始化日志。", flush=True)
        break

