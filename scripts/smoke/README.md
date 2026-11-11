# 集中冒烟

在仓库根目录执行，唯一必填项是本地完整 Base 模型路径：

```bash
bash scripts/smoke/run.sh /path/to/Qwen3-Base
```

先准备好 `.venv/train`、`.venv/evaluation` 和根目录 `data/`。冒烟直接使用现有模型构建、训练与评估入口，不要求预先构建 Char/PGCA。Base 可以位于仓库之外。

默认使用 GPU `0,1,2,3`，三条路线各占一张卡并行，第四张卡闲置；每条路线内部串行执行。模型构建在 CPU 完成。只用一张卡时，三条路线会自动串行：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke/run.sh /path/to/Qwen3-Base
```

## 覆盖范围

| 路线 | 模型构建与训练 | 评估 |
| --- | --- | --- |
| Base | 原基座 → General-SFT/合并 → 三项专项 LoRA | 原 Base PPL、INS 通用 benchmark、专项 adapter |
| Char | Base 构建 Char → Phase1 → CPT → General-SFT/合并 → 三项专项 LoRA | 初始 Char/CPT PPL、INS 通用 benchmark、专项 adapter |
| PGCA | Char 构建 PGCA → Phase1 → Phase2-A/B → General-SFT/合并 → 三项专项 LoRA | ALIGN/CPT PPL、INS 通用 benchmark、专项 adapter |

六项通用 benchmark 为 MMLU、ARC-Challenge、SQuAD v2、GSM8K、CMMLU、CLUE；LMEH 的 `LIMIT=1` 是每个叶子任务各一条，所以 MMLU/CMMLU/CLUE 的全部子任务都会检查。保留正式脚本的 few-shot 协议。

下游覆盖 CSCD-NS 与 LEMON 七域、FCGEC valid/Test 提交、NaCGEC Dev 评分/Test 提交、C³Bench 五任务、Fuxi 全部 21 任务。评分使用现有入口和真实参考答案。微调每个阶段只训练两步，输出质量与短生成上限不作正式性能结论。

指令数据需先按 [General-SFT](../train/instruction/README.md) 准备；缺少该数据时其后训练/评估会标为失败或阻断，不会用假数据补齐。冒烟的 General-SFT 合并验收使用 report 模式并明确标记 smoke_only，不代表正式生成门槛通过。

训练通过不仅看进程退出码：还检查发布的 checkpoint、实际训练步数、所有已记录验证 loss 有限，以及下游辅助损失关闭。评估检查推理是否完整、评分文件是否生成；PPL 要完成六域，LMEH 要生成表格指标。

下游保存时执行每任务 1 条固定 Dev 生成；下游评估前再重新加载实际 adapter，检查 EOS 监督、LoRA 开关及 Cache 数值/生成状态。冒烟显式使用 `DEV_CHECK_MODE=report`、每任务 1 条、与冒烟一致的生成上限：生成质量不设硬门槛，结构、LoRA 或 Cache 错误仍会使步骤失败。质量报告保留在各 checkpoint 或评估目录下的 `dev_check/vN/`；少步冒烟通过不表示停止率已达到正式门槛。

## 数据与参数

运行时只在本次输出目录抽取少量样本：

- 每阶段预训练取 32 篇，正文最多 512 字符，训练序列长度 128；`tar.xz` 直接顺序读取小段内容，原始文件保持原样。正式训练仍按根 README 原地解压。
- CSC/CGEC 微调各取 32 条训练、1 条验证；古汉语按古译今/今译古/断句各取 10 条训练、1 条验证，并保留对应 metadata。
- 本地评估每域/任务一条，PPL 六域各一篇；纠错与古汉语评估保留完整样本和原始 gold。
- 使用单卡、每卡 batch 1、梯度累积 1；每阶段训练 2 步，生成最多 64 token。仅保存模型/adapter，减少 checkpoint I/O。

入口顶部列出可调参数，同名环境变量优先：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 \
TRAIN_STEPS=3 TRAIN_SAMPLES=64 EVAL_SAMPLES=2 \
MAX_NEW_TOKENS=96 \
bash scripts/smoke/run.sh /path/to/Qwen3-Base
```

`DATA_ROOT`、`SMOKE_ROOT`、`TRAIN_PYTHON`、`EVAL_PYTHON` 可覆盖数据目录、输出根目录及两套解释器。默认 `INFERENCE_BACKEND=auto`：Base/Char 检查 vLLM，PGCA 检查 HF；显式设 `INFERENCE_BACKEND=hf` 会统一走 HF，此时不代表验证了 vLLM。

通用 benchmark 通过 `HF_ENDPOINT=https://hf-mirror.com` 获取/复用缓存。首次下载完整数据集可能比少量推理耗时更长。

## 结果与失败排查

每次自动创建新版本，默认结构：

```text
outputs/smoke/vN/
├── summary.md                 # 集中的通过/失败/阻断/裁判待测表
├── summary.json               # 机器可读状态；smoke_only=true
├── logs/                      # 每个步骤的完整输出
├── models/                    # 本轮构建的 Char/PGCA 和独立构建指针
├── data/                      # 本轮少量输入
├── cache/                     # 本轮 CPT 编码与准备缓存
├── training/                  # 本轮 checkpoint 和 LoRA
└── evaluation/                # 各项少量预测、评分和通用 benchmark 结果
```

终端显示每项开始、GPU、耗时、日志位置，长步骤每 15 秒更新状态。失败步骤及其依赖项会分别标记 `failed` / `blocked`，独立项目继续执行。最终先看 `summary.md`；需要排查时提供 `summary.json` 及失败步骤日志即可。

Fuxi 未配置 `JUDGE_MODEL` 与 `JUDGE_BASE_URL` 时，自动评分仍执行，三个模型的裁判步骤单独标记 `pending`，整体状态为 `partial`、`complete=false`；所有必测步骤通过时退出码仍为 0。配置两项后会调用兼容 OpenAI 协议的裁判接口；需要鉴权时设置 `JUDGE_API_KEY`，脚本不会把密钥写进汇总。

任何必测项目失败/阻断时退出码为 1；用户中断为 130。正常实验的模型与完成指针不参与本轮写入。此次单卡少步运行不验收全量预处理内存、长期性能与 EOS 学习、多卡 DDP，或优化器状态恢复。
