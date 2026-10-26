# 通用 benchmark：lm-evaluation-harness

统一评估 Base、Char、PGCA 及其 ALIGN/CPT checkpoint，使用 HF 后端及 benchmark 原始提示词，不套聊天模板，不要求 SFT args.json。完整模型和 tokenizer 从同一 checkpoint 加载；PGCA 通过 checkpoint 自带 AutoClass 加载，保留模型分支配置。不支持直接传 LoRA adapter，也不使用旧脚本的 vLLM 后端。

## 当前实验协议

| BENCHMARK | harness 任务 | few-shot | 数据/评分 split | 指标 |
| --- | --- | --- | --- | --- |
| `mmlu` | `mmlu`，57 学科 | 5，dev 中 first_n | HF `cais/mmlu`，test | acc，保留官方分组加权汇总 |
| `arc` / `arc_challenge` | `arc_challenge` | 0 | HF `allenai/ai2_arc`，ARC-Challenge/test | acc、acc_norm |
| `squadv2` | `squadv2` | 0 | HF `lighteval/squad_v2`，validation | EM/F1、HasAns/NoAns、best 阈值指标 |
| `gsm8k` | `gsm8k`，不是 `gsm8k_cot` | 5，train，seed=42 | HF `openai/gsm8k` main/test | strict-match 与 flexible-extract EM |
| `cmmlu` | `cmmlu`，67 学科 | 5，dev 中 first_n | 本地官方 CSV，test | acc、acc_norm，保留内置分组加权汇总 |
| `clue` | Nepham 的九个 `clue_*` 任务 | 0 | HF `clue/clue`，validation | 分类/选择 acc、acc_norm；CMRC2018 EM/F1 |

只有 MMLU、CMMLU、GSM8K 使用 5-shot，所有待比较模型都相同；启动入口不提供临时覆盖 few-shot 的参数。MMLU 不再用旧合并任务 `mmlu_all`，CMMLU 使用内置中文提示词/学科 description，不沿用旧 `cmmlu_local` 的英文 Answer 提示。旧 0/5-shot 混用结果不可直接并表。

CLUE 范围为 AFQMC、TNEWS、IFLYTEK、CMNLI、CLUEWSC2020、CSL、CMRC2018、CHID、C3，不含 OCNLI、DRCD、diagnostics。公共 test 没有可用金标，因此在 validation 评分；这不是监督训练后的 CLUE 官方隐藏测试榜。CHID 按原 Nepham 逻辑展开成每空一道题，CMRC2018 的归一化/LCS-F1 保持迁移实现，不伪称另一套官方 scorer。不同 CLUE 子任务分开报告，不混合分类 acc 和 QA F1 制造总分。

## 环境

`run.sh` 默认调用已有 `scripts/evaluation/lmeh/env/bin/python`。直接从 PyPI 或可访问的 pip 镜像安装：

```bash
# 先确认此 env 的 PyTorch 与服务器 GPU/CUDA 匹配。
scripts/evaluation/lmeh/env/bin/python -m pip install -r scripts/evaluation/lmeh/requirements.txt
scripts/evaluation/lmeh/env/bin/python -m pip check
```

依赖为普通 `lm_eval[hf]` 包，不要求 GitHub、Git commit 或特定包版本。
运行入口不检查包版本、不扫描框架源码、不做源码哈希匹配。实际安装版本仅随结果记录，不作为启动条件。
不使用 editable 安装、外部仓库 PYTHONPATH 或外部 `lm_eval` 命令。

可以显式 `PYTHON=/path/to/python` 用另一个兼容环境测试，但默认仍绑定本目录 env。代码不会自动安装依赖、读取仓库 HF_TOKEN 文件或改动现有环境。

## 数据：通常无需手动准备

除 CMMLU 外，上表数据由 harness/datasets 首次运行自动从 HF 拉取，之后复用缓存；既定 HF 数据与评分脚本 revision 记录在 `dataset_revisions.json`，它仅选择评估数据，不校验安装环境。沿用调用环境的 `HF_ENDPOINT`、`HF_HOME` 等配置，不强行指定镜像，也不会在下载失败时偷偷换数据来源。SQuAD v2 的 `evaluate` 官方评分脚本首次使用会下载到缓存。

CMMLU 沿用已确认的 CSV 方案，默认：

```text
/share/project/wuhaiming/data/dataset/cmmlu/
  dev/<subject>.csv       # 每科至少 5 条，共 67 科
  test/<subject>.csv
```

可设置 `CMMLU_DIR` 覆盖根目录。这是 dashboard 的服务器路径，不是旧 harness/Nepham/tasks/cmmlu 路径。harness 有 CMMLU 任务定义，但其原 HF 源包含旧式 `cmmlu.py` 加载脚本，当前 datasets>=4 不执行这类脚本，因此保留本地 CSV 适配；所有选项按字符串读取，避免 `0.80` 等被转成数值。已有上述文件便不需要再准备；只有 test 而没有 dev 时不能完成 5-shot，会报出具体缺失文件。

源码不附带、搬移或删除任何原始数据；MMLU/ARC 不再读取旧 `/spaces/sce/data/raw`。也不自动删除已缓存的数据。

## 启动

脚本可以从任意工作目录通过绝对路径调用，下面示例在仓库根目录。

```bash
# 只检查环境/生成任务配置；不加载模型、不下载 benchmark 数据。
STAGE=inspect bash scripts/evaluation/lmeh/run.sh

# 实际加载并检查选定 benchmark 数据；可能触发 HF 下载，但不加载模型。
STAGE=check BENCHMARK=clue bash scripts/evaluation/lmeh/run.sh
STAGE=check BENCHMARK=cmmlu bash scripts/evaluation/lmeh/run.sh

# 冒烟：LIMIT 是每个子任务的条数，不是整个 benchmark 的总条数。
# 必须使用单独 RUN_NAME；报告 smoke_only=true，不能作为完整评估。
CUDA_VISIBLE_DEVICES=0 MODEL=/path/to/checkpoint MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
  BENCHMARK=arc LIMIT=10 RUN_NAME=smoke bash scripts/evaluation/lmeh/run.sh

# 正式运行全部六项 benchmark（ARC 仅 Challenge），无 LIMIT。
CUDA_VISIBLE_DEVICES=0 MODEL=/path/to/checkpoint MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
  RUN_NAME=full bash scripts/evaluation/lmeh/run.sh

# 也可以分开运行，避免一项的数据问题阻断后续项目。
CUDA_VISIBLE_DEVICES=0 MODEL=/path/to/checkpoint MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
  BENCHMARK=mmlu,cmmlu,gsm8k RUN_NAME=full bash scripts/evaluation/lmeh/run.sh
```

环境变量：`BENCHMARK=all`（或逗号分隔；`arc` 仅选择 `arc_challenge`，不再提供 `arc_easy`）、`STAGE=run|inspect|check`、`MODEL`、`MODEL_NAME`、`RUN_NAME`、`CMMLU_DIR`、`DEVICE=cuda:0`、`DTYPE=bfloat16`、`BATCH_SIZE=1`（也可 auto）、`MAX_LENGTH=8192`、`LIMIT`（仅冒烟）。随机种子均为 42；正式统计 bootstrap_iters=1000，冒烟关闭 bootstrap。

默认单进程、单设备，在一次 `all` 调用中复用已加载模型；不要用 torchrun/accelerate launch 启动。多个可见 GPU 不会自动数据并行。CPU 小模型可用 `DEVICE=cpu DTYPE=float32`。PGCA 的 HF 似然评分和带 KV cache 生成均有小模型回归，但服务器大模型 CUDA 仍需冒烟确认。

所有模型比较时使用相同 MAX_LENGTH/DTYPE。上下文容量不足或“prompt + 生成预算”超长时直接报错，不删题目、不截 few-shot、不自动降 shot。需要增大 MAX_LENGTH 时应对所有对照模型一致设置，且不得超过各自容量。GSM8K 生成预算 512、SQuAD v2 为 256、CLUE CMRC2018 为 64；greedy，不开启 thinking/chat template。后端会忽略检查点 `generation_config.json` 中可能覆盖任务预算的 `max_new_tokens`，实际任务预算记录在 manifest 的 `protocol` 中。更换生成预算会改变实验协议，应修改协议并单独记录，不能混用分数。

## 输出与可追溯性

```text
outputs/evaluation/mmlu/<MODEL_NAME>/<RUN_NAME>/v0/
outputs/evaluation/arc/<MODEL_NAME>/challenge/<RUN_NAME>/v0/
outputs/evaluation/{squadv2,gsm8k,cmmlu,clue}/<MODEL_NAME>/<RUN_NAME>/v0/
```

RUN_NAME 未设置时省略该层。每次运行自动创建下一个 `v0`、`v1`、`v2` 目录，并在开始时打印实际路径。
`OUT` 覆盖的是上述 `outputs/evaluation` 根目录，不是某个 benchmark 的最终目录。

每项保存 `results.json`（harness 原生结果/组汇总）、`samples/<task>.jsonl`（题目、prompt、输出及评分）、`tasks/`（本次展开的任务 YAML/辅助文件）和 `manifest.json`。从已安装的 harness 包展开内置任务，不复制第二套 57/67 学科配置到源码目录；CLUE 自定义源码保存在本仓库 `tasks/clue/`。

manifest 记录数据 revision/Arrow fingerprint、CMMLU 文件 SHA256、任务快照、模型元数据、实际依赖版本、协议、模型开关和产物哈希；不记录框架/评估源码哈希或安装版本锁。大权重只记录大小/mtime，不宣称逐字节权重哈希。不复制可能含凭据的训练 args.json，也不上传结果。

重复执行原命令即可创建新版本，不必修改 RUN_NAME/OUT。已有结果、旧式未分版本的文件和失败目录都保留；失败的版本占用其序号。并发启动通过原子创建目录分配不同版本。失败时 manifest 不标记 complete，不会把部分子任务当完整报告。`complete=true` 仅表示该评估执行完成，不代表模型质量达标；带 LIMIT 的结果始终标记 smoke_only。

## 验证与来源

```bash
PYTHONDONTWRITEBYTECODE=1 scripts/evaluation/lmeh/env/bin/python -B -m pytest -q tests/test_evaluation_lmeh.py
```

测试覆盖全部任务解析/shot/split/提示词、CSV 文本保真、SQuAD 双请求和固定评分器、原生分组汇总、超长拒绝、结果保护、CPU Qwen3 评估、自包含 PGCA 的似然和生成。离线 fixture 测试不等于所有线上数据或服务器 GPU 全量评估完成。

依据：[harness 接口](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)、[MMLU 任务](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu)、[CMMLU 数据源](https://huggingface.co/datasets/haonan-li/cmmlu/tree/main)。评估数据选择见 `dataset_revisions.json`。
