# 下游 LoRA

CSC、CGEC 和古汉语均使用 ms-swift all-linear LoRA，PGCA 分支参与 LoRA，字音/字形辅助损失关闭。

```bash
bash scripts/train/finetune/csc.sh
bash scripts/train/finetune/cgec.sh
bash scripts/train/finetune/ancient.sh
```

数据分别位于 `data/finetune/{csc,cgec,ancient}`。任务脚本顶部直接设置长度、学习率和 epoch；GPU、batch、LoRA 和优化器公共参数位于 `common.sh` 顶部。

默认读取完成的 CPT 指针。显式固定基座时同时传入名称：

```bash
MODEL=/path/to/checkpoint MODEL_NAME=Qwen3-Base-PGCA-CPT \
bash scripts/train/finetune/csc.sh
```

产物是 LoRA adapter，并在任务输出根目录发布 `next_stage_model.txt`。
