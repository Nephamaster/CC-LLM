# 下游 LoRA

CSC、CGEC 和古汉语均使用 ms-swift all-linear LoRA，PGCA 分支参与 LoRA，字音/字形辅助损失关闭。

```bash
bash scripts/train/finetune/csc.sh
bash scripts/train/finetune/cgec.sh
bash scripts/train/finetune/ancient.sh
```

数据分别位于 `data/finetune/{csc,cgec,ancient}`。CSC 默认读取 `csc_mix_dedup.jsonl` 和 `cscd_dev.jsonl`；其余任务使用各目录下的 train/validation 或 train/dev。任务脚本顶部直接设置长度、学习率和 epoch；GPU、batch、LoRA 和优化器公共参数位于 `common.sh` 顶部。

默认读取完成的 CPT 指针。显式固定基座时同时传入名称：

```bash
MODEL=/path/to/checkpoint MODEL_NAME=Qwen3-Base-PGCA-CPT \
bash scripts/train/finetune/csc.sh
```

产物是 LoRA adapter，并在任务输出根目录发布 `next_stage_model.txt`。

单机训练默认通过 torchrun 自动分配通信端口；指定端口时在原命令前加 `MASTER_PORT=23457`。启动时打印模型、输出路径、world size、每卡 batch、梯度累积及有效 batch，并显示阶段日志文件路径。

每个版本目录的 `checkpoint_selection.json` 说明最终选择 best 还是 last。验证日志增加有效监督 token 总数；没有可用的有限 best 指标时，会明确记录回退到最后检查点的原因。原有 LoRA、辅助损失与数据编码并发设置保持原值。
