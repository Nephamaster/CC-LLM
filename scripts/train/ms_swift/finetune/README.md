# CSC / CGEC / 古汉语 LoRA 微调

三项任务都使用 ms-swift all-linear LoRA，PGCA 分支参与 LoRA，字音/字形辅助损失关闭。

## 数据与训练方式

| 任务 | Train | Dev |
| --- | --- | --- |
| CSC | `data/downstream/csc/sft/csc_mix_dedup.jsonl` | `data/downstream/csc/sft/cscd_dev.jsonl` |
| CGEC | `data/downstream/cgec/sft/final/train.jsonl` | `data/downstream/cgec/sft/final/validation.jsonl` |
| 古汉语 | `data/downstream/ancient/train/final/train.jsonl` | `data/downstream/ancient/train/final/dev.jsonl` |

直接读取已有 `messages` JSONL：user 只有原句，assistant 为纠正句。不添加任务提示、不改原数据、不重新划分、不调用 CPT 的 `prepare_data.py`，也不另建 `.raw`/分段缓存。`LOAD_FROM_CACHE_FILE` 直接控制 ms-swift 原生预处理缓存复用，默认 true。

主干和 PGCA 使用 `--target_modules all-linear` 的 PEFT LoRA，默认 rank=16、alpha=32、dropout=0.05；embedding、norm 等非 linear 参数仍冻结。**不启用字音/字形辅助损失**，只对 assistant 文本计算标准 SFT loss。all-linear 挂到闲置辅助头上的 LoRA 参数在优化器/DDP 初始化前冻结，不参与训练；完整音形分支及上下文编码器的线性层仍训练 LoRA。

CSC/CGEC 的 `cc_llm_correction` 模板基于 `qwen3_nothinking`，不加入额外 non-thinking 前缀。古汉语使用独立的 `cc_llm_ancient`，基于数据验收时的 `qwen3 + enable_thinking=false + add_non_thinking_prefix=true`，并保持 `disable_ignore_empty_think=false`，保证长度与 assistant loss 口径一致。两个模板都关闭 PGCA 辅助 logits。插件不注册自定义训练 loss/优化器，只负责 PGCA 启用、辅助头冻结、古汉语宏平均指标和完成指针。已有评估入口会从 SFT 的 `args.json` 恢复对应模板。

古汉语入口根据已验收的 `train/dev.metadata.jsonl` 为运行视图加入 `channel=c2m|m2c|punctuate`，不改写 final 数据。ms-swift 原生 channel loss 产生三项验证 loss，插件取等权宏平均 `eval_macro_task_loss`选择最佳 checkpoint。古汉语按数据验收口径使用 `max_length=2048` 和 non-thinking 前缀。锁定版 ms-swift 的 CLI 使用 `truncation_strategy=delete` 映射模板内部 `raise`；本入口同时设置 `strict=true`，因此超长样本会直接报错，不会静默删除。

## 环境与参数入口

复用 `scripts/train/ms_swift/env/` 及上一级 `requirements.txt`，无需新环境、无需外部 ms-swift 仓库。GPU PyTorch 安装沿用训练环境的配置。

- `common.sh` 顶部：GPU、batch、LoRA、优化器、缓存开关、eval/save/logging 间隔等。
- `csc.sh` / `cgec.sh` / `ancient.sh` 顶部：任务数据、输出目录、长度、学习率、epoch及任务特有参数。
- 环境变量优先于脚本默认值。默认沿用旧 CSC 配置：8 卡、每卡 batch=8、累积=1、学习率 1e-4、5 epochs、max_length=768。CGEC 使用同一套可覆盖默认值；这不是经过调优后的最佳配置。
- `truncation_strategy=delete` 沿用旧 CSC：超长样本整条跳过，不拆开输入/答案。启动时检查框架日志中的实际样本数量；不同 tokenizer 的删除数量可能不同，需要调整 `MAX_LENGTH` 并记录，不能只比较原始文件条数。
- `MAX_STEPS` 默认为 -1（按 epochs）；短冒烟可设正数。`SAVE_STEPS` 应是 `EVAL_STEPS` 的整数倍。
- 古汉语默认面向四卡：每卡 batch 1、累积 8，有效 batch 32；学习率 `1e-4`、5 epochs 是待验证的起点，不声称已调优。

## 启动与模型命名

`MODEL_NAME` 是**输入模型的规范名称**，默认 `Qwen3-1.7B-Base-PGCA-CPT`。新训练优先使用显式 `MODEL`；未指定时，先读取 `outputs/train/$MODEL_NAME/next_stage_model.txt`，没有该指针再尝试 `models/$MODEL_NAME`。不会按目录最大版本号猜模型。

输出按输入名称追加 `-CSC` / `-CGEC`；输入末尾的 `-ALIGN` 会去掉，例如 `B-Char-ALIGN → B-Char-CSC`、`B-PGCA-CPT → B-PGCA-CPT-CGEC`。自定义 `MODEL` 时同时设置正确的 `MODEL_NAME`，不要把 Base 的结果标成 PGCA。

```bash
# 默认使用最近完成的 PGCA-CPT checkpoint；两项任务独立训练。
bash scripts/train/ms_swift/finetune/csc.sh
bash scripts/train/ms_swift/finetune/cgec.sh
bash scripts/train/ms_swift/finetune/ancient.sh

# 固定输入 checkpoint；一组对照实验应固定同一路径，不随 CPT 指针更新切换基座。
CUDA_VISIBLE_DEVICES=0,1 MODEL=/实际路径/checkpoint-N \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
bash scripts/train/ms_swift/finetune/csc.sh

# 原 Base 对照；Char/ALIGN 对照同理。
MODEL=/实际路径/Qwen3-1.7B-Base MODEL_NAME=Qwen3-1.7B-Base \
bash scripts/train/ms_swift/finetune/cgec.sh
```

输出根目录默认 `outputs/train/<输入名称去掉-ALIGN>-CSC|CGEC|ACT/`；由 ms-swift 自动创建 `v0-时间戳`、`v1-时间戳`，日志随版本保存。CSC/CGEC 按 `eval_loss` 选优，古汉语按 `eval_macro_task_loss` 选优。完成后在 run 目录及输出根目录写入 `next_stage_model.txt`。`finetune_report.json` 记录实际可训练参数，方便核对 PGCA/context LoRA 是否启用。并发实验使用不同 `OUT`，避免共享完成指针。

## 恢复与评估

```bash
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT \
RESUME_FROM_CHECKPOINT=/实际路径/outputs/train/Qwen3-1.7B-Base-PGCA-CPT-CSC/v0-时间戳/checkpoint-N \
bash scripts/train/ms_swift/finetune/csc.sh
```

恢复时原版本目录继续写入，不创建新版本。未传 `MODEL` 时，从 adapter_config.json 读取原基座路径，不使用后来更新的 CPT 指针；基座搬迁后可显式传 `MODEL`，但必须指向同一份权重。保持原模型名称、数据、卡数、batch、梯度累积及训练配置。

**产物是 LoRA adapter，不是完整模型权重。** 现有评估入口接受 `MODEL=原始基座` 与 `LORA=选出的adapter`；保留 adapter 的 `args.json` 和当前插件文件，不复制 CPT 的训练参数冒充 SFT 参数。

```bash
MODEL=/实际路径/原始PGCA-CPT基座 LORA=/实际路径/CSC的adapter-checkpoint \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-CSC \
bash scripts/evaluation/csc/run.sh

MODEL=/实际路径/原始PGCA-CPT基座 LORA=/实际路径/CGEC的adapter-checkpoint \
MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-CGEC DATASET=fcgec SPLIT=valid \
bash scripts/evaluation/cgec/run.sh
```

评估入口仍使用其自己的环境。这里关闭的是辅助损失，不是 PGCA 分支；辅助头结构保留以兼容基座 checkpoint。若需要“关 PGCA”消融，不能用本入口误当成该组实验。

## 验证边界

使用普通安装的项目固定 ms-swift 和 PyTorch 2.10.0 CPU 验证：Transformers 5.19.0 下 14 项测试通过（含 Base/PGCA 真实小模型 SFT、双进程 Gloo、LoRA 梯度和 adapter 评估生成）；仓库锁定的 Transformers 5.12.1 下 12 项相关测试通过（含 PGCA 单进程训练/推理）。未执行服务器 1.7B GPU 正式训练，先按实际空闲 GPU/显存调整 batch，再做短跑检查。
