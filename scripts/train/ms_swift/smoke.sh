cd /share/project/wuhaiming/spaces/CC-LLM
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHON_CMD="$PWD/scripts/train/ms_swift/env/bin/python"
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2

export MODEL_NAME=Qwen3-1.7B-Base-PGCA
export PGCA_MODEL="$PWD/models/$MODEL_NAME"

export TRAIN_BATCH_SIZE=1
export EVAL_BATCH_SIZE=1
export GRAD_ACCUMULATION=1
export DATASET_NUM_PROC=2
export LOGGING_STEPS=1
export EVAL_STEPS=4
export SAVE_STEPS=4

# 每次执行创建新目录，不复用正式缓存。
export SMOKE_ROOT="$(mktemp -d "$PWD/outputs/pgca-smoke.XXXXXX")"
export SMOKE_ALIGN="$PWD/outputs/cpt/${MODEL_NAME}-ALIGN/smoke/"
export SMOKE_CPT="$PWD/outputs/cpt/${MODEL_NAME}-CPT/smoke/"

"$PYTHON_CMD" -m pip check

"$PYTHON_CMD" -B - <<'PY'
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

assert torch.cuda.device_count() == int(os.environ["NPROC_PER_NODE"])
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)

for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        assert torch.cuda.is_bf16_supported(), f"GPU {i} 不支持 BF16"
        x = torch.randn(
            128, 128, device=f"cuda:{i}",
            dtype=torch.bfloat16, requires_grad=True,
        )
        (x @ x).float().square().mean().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(x.grad).all()
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}，BF16 前反向通过")
        del x
        torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained(
    os.environ["PGCA_MODEL"], trust_remote_code=True,
)
unit = (
    "这是音形上下文模型的训练链路测试。"
    "银行行长正在检查账目，学生重新阅读文章，重庆的音乐令人快乐。"
    "汉字具有不同的读音、笔画、部首和结构。\n"
    "This is a small training smoke test. Calculate 12 + 35 = 47.\n"
)
text = unit
while len(tokenizer.encode(text, add_special_tokens=False)) < 8192:
    text *= 2

root = Path(os.environ["SMOKE_ROOT"]) / "dataset"
for split, count in (("train", 32), ("validation", 2)):
    folder = root / split
    folder.mkdir(parents=True)
    with (folder / "smoke.jsonl").open("x", encoding="utf-8") as f:
        for i in range(count):
            body = text if split == "train" else unit * 8
            row = {"text": f"{split} sample {i}\n{body}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("冒烟数据：", root)
print("注意：合成训练/验证内容有重叠，仅用于链路验证，不能用于效果评估。")
PY

# Phase1：主干训练，音形分支旁路并冻结
MODEL="$PGCA_MODEL" \
DATA_DIR="$SMOKE_ROOT/dataset" \
OUT="$SMOKE_ALIGN" \
CACHE_DIR="$SMOKE_ROOT/cache_phase1" \
PHASE1_TOKENS=32768 \
RESUME_FROM_CHECKPOINT="" \
bash scripts/train/ms_swift/cpt/phase1.sh \
  2>&1 | tee "$SMOKE_ROOT/phase1.console.log"

# Phase2：自动读取刚完成的 Phase1 checkpoint，依次运行 A/B
MODEL="" \
ALIGN_OUT="$SMOKE_ALIGN" \
DATA_DIR="$SMOKE_ROOT/dataset" \
OUT="$SMOKE_CPT" \
CACHE_DIR="$SMOKE_ROOT/cache_phase2" \
PHASE2_TOKENS=131072 \
ADAPTATION_TOKENS=65536 \
RUN_STAGE=all \
RESUME_A="" \
RESUME_B="" \
bash scripts/train/ms_swift/cpt/phase2.sh \
  2>&1 | tee "$SMOKE_ROOT/phase2.console.log"

"$PYTHON_CMD" -B - <<'PY'
import json
import math
import os
from pathlib import Path

stages = {
    "Phase1": Path(os.environ["SMOKE_ALIGN"]),
    "Phase2-A": Path(os.environ["SMOKE_CPT"]) / "adaptation",
    "Phase2-B": Path(os.environ["SMOKE_CPT"]) / "joint",
}

for name, folder in stages.items():
    pointer = folder / "next_stage_model.txt"
    assert pointer.is_file(), f"{name}: 未生成阶段完成指针"
    assert Path(pointer.read_text().strip()).is_dir()

    checkpoints = list(folder.glob("checkpoint-*"))
    assert checkpoints, f"{name}: 没有 checkpoint"
    latest = max(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[1]))
    state = json.loads((latest / "trainer_state.json").read_text())
    history = state["log_history"]

    losses = [r["eval_clm_loss"] for r in history if "eval_clm_loss" in r]
    assert losses and all(math.isfinite(v) for v in losses), (
        f"{name}: eval_clm_loss 缺失或出现 NaN/Inf"
    )

    gradients = {
        k: v
        for row in history for k, v in row.items()
        if k.endswith("grad_norm")
    }
    print(name, "step =", state["global_step"],
          "eval_clm_loss =", losses[-1], "梯度记录 =", gradients)

print("三个阶段均已保存 checkpoint，且已记录的 eval_clm_loss 全部有限。")
print("日志和缓存：", os.environ["SMOKE_ROOT"])
PY
