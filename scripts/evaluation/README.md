# 评估

完整入口：

```bash
bash scripts/evaluation/run.sh
```

所有任务使用 `.venv/evaluation`，数据位于 `data/evaluation/`，结果写入 `outputs/evaluation/`。子入口包括：

```text
ppl/run.sh       原始文本 PPL
lmeh/run.sh      MMLU、ARC-Challenge、SQuAD v2、GSM8K、CMMLU、CLUE
csc/run.sh       CSCD-NS 与 LEMON
cgec/run.sh      FCGEC 与 NaCGEC
ancient/run.sh   C³Bench 与 Fùxì
```

下游 LoRA 评估使用 `MODEL=完整CPT基座` 和 `LORA=adapter checkpoint`。`STAGE=prepare|infer|score|all` 可单独执行阶段，`RUN_DIR` 可接续指定版本。

`INFERENCE_BACKEND=auto` 对标准模型使用 vLLM，对当前 PGCA 使用 HF。显式 `INFERENCE_BACKEND=vllm` 会通过 vLLM Transformers backend 加载本地自定义模型；未完成真实一致性验证前不能替代 PGCA 的 HF 正式结果。
