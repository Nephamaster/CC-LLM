# CC-LLM Phase1 / Phase2 训练

本目录固定由 CC-LLM 管理，不移动到 ms-swift 仓库。训练环境位于 `scripts/train/ms_swift/env/`，通过普通安装提供 ms-swift；插件调用模型 checkpoint 自带的 `multitask_loss_sums`，不依赖外部框架源码目录。

目标接口：ms-swift `4.7.0.dev0`，交付时本地源码 HEAD 为 `ce28e0288657d6472a439ea219f1bcc77310e764`（其 `swift/` 与审查起点 `1516d78d627381c2072be80037a7a64c3811480d` 无差异）。隔离验证环境使用 PyTorch `2.10.0+cpu`、Transformers `5.12.1`。这不是服务器 CUDA/显存验证；不要为了照抄测试版本在 GPU 环境安装 CPU wheel。

先在 CC-LLM 根目录准备环境（GPU 环境先安装匹配 CUDA 的 PyTorch）：

```bash
python3.11 -m venv scripts/train/ms_swift/env
scripts/train/ms_swift/env/bin/python -m pip install -r scripts/train/ms_swift/requirements.txt
scripts/train/ms_swift/env/bin/python -m pip check
```

## 1. 先构建新模型

前提：`models/Qwen3-1.7B-Base-Char` 包含原有语义 tokenizer/权重、修复后的 schema v2 音形资源、匹配的 `embedding_config.json`。schema 版本记录在 manifest，不写成模型名称后缀。不需要重新生成语义词表，也不能把新音形索引直接替换进旧 PGCA 权重；旧目录存在时请准备独立输入并显式指定路径，不自动覆盖。

在 CC-LLM 仓库根目录执行，输出目录必须尚不存在：

```bash
scripts/train/ms_swift/env/bin/python -B -m src.pgca.build_pgca \
  --model-path models/Qwen3-1.7B-Base-Char \
  --output-path models/Qwen3-1.7B-Base-PGCA \
  --pgca-layers 11,12,13,14,15,16 \
  --context-encoder --auxiliary-heads --zero-init-output

scripts/train/ms_swift/env/bin/python -B -m src.pgca.validate_pgca \
  --model-path models/Qwen3-1.7B-Base-PGCA
```

核对构建报告的 `context_parameter_count`、`auxiliary_parameter_count`、`pgca_initialization.passed`，以及验证报告 `passed=true`。实际候选宽度来自 manifest，修复后的默认值为 16＋1 slots；辅助类别从合法 ID 3 开始映射，排除 pad/none/unknown。模型保存类别名称、资源哈希和 token 类型索引指纹。

## 2. 训练日程与完整脚本

| 入口/阶段 | 可训练参数 | 分支/损失 | 学习率 | 长度 |
| --- | --- | --- | --- | --- |
| `phase1.sh` / Phase1 | 主干及语义输入输出层 | PGCA 完全旁路，纯 CLM | `1e-5` | 2048 |
| `phase2.sh` / A | Feature Embedding、Context Encoder、PGCA、辅助头 | 全音形路径，三任务 | `5e-5` | 4096 |
| `phase2.sh` / B | 全参数 | 三任务 | 主干 `1e-5`，新增分支 `3e-5` | 4096 |

Phase2 默认按有效 next-token 监督目标划分为 100M＋9.9B。字音/字形权重在 A 的前 10M 有效目标内，按优化步从 0 升至 0.05/0.02。A/B 重新创建优化器和调度器。A 的最后 checkpoint 进入 B；Phase1/B 按 `eval_clm_loss` 选优。三个阶段使用同一套模型源码，不通过修改源码切换行为。

在 CC-LLM 根目录执行：

```bash
bash scripts/train/ms_swift/cpt/phase1.sh
bash scripts/train/ms_swift/cpt/phase2.sh
```

脚本自动定位本仓库，不要求在外部框架目录运行。模型命名采用根 README 的规范，可通过环境变量覆盖：

- `CC_ROOT`：默认从脚本位置定位的 CC-LLM 根目录。
- `PYTHON_CMD`：默认 `scripts/train/ms_swift/env/bin/python`；数据准备与 Swift 分发使用同一解释器，不依赖 PATH 中的 `swift`。
- `BASE_MODEL_NAME`：Base 原名，默认 `Qwen3-1.7B-Base`；`MODEL_NAME` 表示本轮基础模型 M，默认 `${BASE_MODEL_NAME}-PGCA`，不是 checkpoint 路径。
- `TRAIN_ROOT`：默认 `outputs/train`；Phase1 输出 `M-ALIGN`，Phase2 输出 `M-CPT`，不形成 `M-ALIGN-CPT`。
- `MODEL`：Phase1 初始模型，或 Phase2 的 Phase1 checkpoint。Phase2 未设置时读取 ALIGN 输出的 `next_stage_model.txt`，不猜测 `best/` 路径。
- `DATA_DIR`：含 `train/*.jsonl` 和 `validation/*.jsonl` 的目录，自动读取所有分片。
- `OUT`、`CACHE_DIR`：独立实验输出/缓存目录。默认拒绝覆盖已有训练目录。

- `CUDA_VISIBLE_DEVICES`：默认 8 卡；`NPROC_PER_NODE` 默认按可见卡列表推导。脚本按单机 DDP 编写。
- `TRAIN_BATCH_SIZE`、`EVAL_BATCH_SIZE`、`GRAD_ACCUMULATION`：默认 1、1、8。
- `DATASET_NUM_PROC`：默认 8。
- `PHASE1_TOKENS`：默认 1B，语料不足则停止；显式设为 0 表示接受现有 Phase1 语料跑一遍并报告实际目标数，不把不足 1B 冒充为 1B。
- `PHASE2_TOKENS`、`ADAPTATION_TOKENS`：默认 10B、100M。语料不足时会报错，需显式选择较小预算，且保持总预算大于适配预算。
- `EVAL_STEPS`、`SAVE_STEPS`、`LOGGING_STEPS`：默认 500、500、10；短冒烟可都设为 1。保存步数必须符合 HF 的最佳模型加载约束。
- `RUN_STAGE=all|a|b`：Phase2 默认依次跑 A/B，可单独跑某阶段。

默认例子：初始化权重 `models/Qwen3-1.7B-Base-PGCA/`；Phase1 为 `outputs/train/Qwen3-1.7B-Base-PGCA-ALIGN/`；Phase2 为 `outputs/train/Qwen3-1.7B-Base-PGCA-CPT/`，其 `adaptation/`、`joint/` 下分别保存两个阶段的 checkpoint 和 `logs/`。阶段结束的 `next_stage_model.txt` 指向实际模型，Phase2 根目录也发布此指针。

例如较小 Phase2 筛选实验（仍需在前面完成 Phase1）：

```bash
PHASE2_TOKENS=500000000 ADAPTATION_TOKENS=100000000 \
CACHE_DIR=/实际路径/cache/phase2_500m \
OUT=/实际仓库/outputs/train/Qwen3-1.7B-Base-PGCA-CPT/experiment-500m \
bash scripts/train/ms_swift/cpt/phase2.sh
```

默认脚本交付完整方案 E；正式跑五组消融、下游 SFT 和 benchmark 仍需另行安排，不能把单个完整方案的结果当作机制验证。

## 3. 数据准备与 token 预算

首次运行脚本会调用官方 `swift export`，用本插件的模板执行无 chat template 的预训练编码，再调用 `prepare_data.py`。不重采语料，不执行逐文 G2P，不改原始 JSONL。

准备过程：

1. 在模板切分后过滤零有效目标分片；再次检查缓存 labels/input_ids 和序列长度。
2. 用 Arrow/NumPy 分批计数 `labels[1:] != -100`，固定 seed 打乱训练样本；验证集不打乱。
3. A/B 使用不重叠的监督目标区间。跨分界的同一输入片段可能出现在两段中，但 labels 使用互补 mask，不重复训练目标，也不丢掉边界目标。
4. 训练样本数补齐到 `world_size × per_device_train_batch_size` 的倍数。补齐行复制输入但全标签置 -100，携带 `is_padding_dummy=true`，贡献图连接的零损失；避免 DDP 自动重复真实样本。改变卡数/batch 时必须重新准备视图。
5. 保存 `preparation_report.json`：实际 token 数、丢弃行数、A/B 区间、补齐行数、tokenizer 指纹、长度及 batch 几何。

缓存以 `--cached_dataset` / `--cached_val_dataset` 加载，必须保持 `truncation_strategy=split`。当前 Swift 对非 split 缓存会走 LazyLLMDataset，不能把编码后的行重新当作原始 messages。

原始官方缓存保存在 `${CACHE_DIR}.raw`，分段缓存和中间 Arrow 文件保存在 `CACHE_DIR`。**需预留额外磁盘空间**，不能把它们视为仅有几个字节的索引文件。已有兼容官方缓存可直接调用：

```bash
scripts/train/ms_swift/env/bin/python -B scripts/train/ms_swift/cpt/prepare_data.py \
  --input /已有官方缓存根目录 \
  --output /新的分段缓存目录 \
  --model /本轮模型目录 --max-length 4096 \
  --world-size 8 --batch-size 1 \
  --adaptation-tokens 100000000 --total-tokens 10000000000
```

2048 缓存不能伪装成 4096 长上下文训练数据。新模型必须复用相同语义 tokenizer；改变 tokenizer 后不能复用旧 token IDs。

## 4. 断点恢复

同一阶段恢复使用 checkpoint 的模型、优化器、调度器和数据游标；不能把阶段切换写成 resume。

```bash
# Phase1
RESUME_FROM_CHECKPOINT=/本轮ALIGN输出/checkpoint-N \
bash scripts/train/ms_swift/cpt/phase1.sh

# Phase2-A 恢复；完成后自动进入 B
RESUME_A=/本轮CPT输出/adaptation/checkpoint-N \
bash scripts/train/ms_swift/cpt/phase2.sh

# A 已完成，只恢复 B
RUN_STAGE=b RESUME_B=/本轮CPT输出/joint/checkpoint-N \
bash scripts/train/ms_swift/cpt/phase2.sh
```

恢复时保持原 `OUT`、`CACHE_DIR`、模型/数据版本、卡数、batch、累积步数及预算。缓存不完整时脚本会停止，不会自动删掉重建。每个 checkpoint 的 `cc_llm_state.json` 记录阶段、step 和累计辅助训练 tokens；模型 buffer 同时保存该计数，恢复后不会重新开始辅助权重 warmup。

## 5. 验证指标与工程边界

- 以 `eval_clm_loss` 选优，PPL 只能用 `exp(eval_clm_loss)`。`eval_total_loss` 是含辅助任务的联合指标。
- `eval_phon_loss`、`eval_glyph_loss` 使用 CLM token 分母；`*_supervised_loss`、`*_accuracy` 使用各自有效监督分母；同时记录 `*_tokens`。
- 验证指标先通过 Accelerate 去掉 DDP 尾部重复样本，再按 token 累计。优化损失只由 Swift 乘一次 world size，不在插件里重复放大。
- 训练记录 `weighted_aux_to_clm_ratio`、主干和分支梯度范数。PGCA 零输出初始化后，第一步内部 Q/K/V/上下文梯度为零是预期现象。
- 首版仅验证 HF DynamicCache 路径，支持 prefill、增量解码和 beam 重排。不支持 packing、padding-free、sequence parallel、Liger loss、logits-to-keep、StaticCache 或 vLLM；不能把旧 vLLM 后端当作已经支持新编码器。
- 正式 GPU 训练前应先用少量独立 JSONL 和显式小 token 预算做冒烟，检查真实显存、吞吐、三阶段冻结状态及有效验证指标。本地 CPU 测试不替代该步骤。
