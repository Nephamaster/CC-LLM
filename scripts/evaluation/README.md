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

下游 LoRA 评估使用 `MODEL=完整基座` 和 `LORA=adapter checkpoint`，基座可以是原 Base、Char-CPT 或 PGCA-CPT。总入口用 `MODEL_NAME` 区分三条路线，并读取 `outputs/train/<MODEL_NAME>-{CSC,CGEC,ACT}/next_stage_model.txt`；`TRAIN_ROOT` 可以更改训练根目录。`STAGE=prepare|infer|score|all` 可单独执行阶段，`RUN_DIR` 可接续指定版本。

`INFERENCE_BACKEND=auto` 对标准模型使用 vLLM，对当前 PGCA 使用 HF。显式 `INFERENCE_BACKEND=vllm` 会通过 vLLM Transformers backend 加载本地自定义模型；未完成真实一致性验证前不能替代 PGCA 的 HF 正式结果。

正式评估默认 `BATCH_SIZE=32 TENSOR_PARALLEL_SIZE=4`，下游 vLLM 的 `max_num_seqs` 随 batch 取 32。TP 只对 vLLM 生效；HF 后端不会因此启用四卡张量并行。单卡使用 vLLM 时显式设置 `CUDA_VISIBLE_DEVICES=<索引> TENSOR_PARALLEL_SIZE=1`。集中冒烟继续显式使用 batch 1、TP 1。

## 下游评估前的自动验收

CSC、CGEC、Ancient 的 `STAGE=all|infer` 会先在当前可见的第一张卡上单独加载完整基座和已保存 LoRA，执行固定 Dev 检查，结束进程释放模型后才启动 benchmark 推理。PPL/LMEH 不受这项 SFT 检查影响；只准备输入、评分或补裁判也不需要重新加载模型。

默认每任务 10 条（ACT 三任务共 30 条），正常停止及 `<|im_end|>` 结束率各至少 90%，截断、重复、异常率各不超过 10%，按任务分别判定。还检查真实模板中的 EOS label、LoRA 开关 logits，以及同一前缀在 Cache 开关下的数值、缓存长度和生成状态。任一要求未通过会在正式推理前退出；结果在本轮 `dev_check/vN/report.json` 和 `samples.jsonl`，原命令无需增加步骤。

若迁移了训练数据，可在原评估命令前设置 `DEV_CHECK_DEV=/path/to/original-dev.jsonl`。阈值、采样量、生成上限、Cache 容差与独立验收命令见 [下游训练说明](../train/finetune/README.md#固定-dev-生成验收)。默认 `DEV_CHECK_MODE=required`；`report` 仅用于冒烟或排障，允许生成质量未达标继续运行，不能当作通过正式验收，结构、LoRA 或 Cache 失败仍阻止运行。推理 manifest 记录检查模式及报告位置。

此验收使用 HF 单序列、无 padding 的真实模型路径。它能发现明显非终止和加载/缓存异常，不证明回答准确性，也不代替实际 vLLM、批量推理的一致性检查。

## 实验表取数

LMEH 每次运行额外写入 `table_metrics.json`；其他评估的 `metrics.json` 增加 `table_metrics` 字段，原始指标保留。缺失任务保持空值，不当作 0 分，也不会自动把估算写入实测报告。

| 表格 | 取数口径 |
| --- | --- |
| 通用 benchmark | MMLU/CMMLU：harness 汇总 acc；ARC-Challenge：acc_norm；GSM8K：strict-match exact_match；SQuAD v2：F1 除以 100。全部为 0–1。 |
| CLUE 五组 | 语义理解与推理：AFQMC/CMNLI/WSC2020 acc 等权；分类：TNEWS/IFLYTEK acc 等权；一致性：CSL acc；词语：CHID acc；阅读：C3 acc 与 CMRC2018 F1 等权。全部为 0–1，这是本表的分组汇总，不是官方 CLUE 总分。 |
| CSC | `table_metrics.lemon_v2` / `cscd_ns`：原始输出的句子级纠正 P/R/F1，0–100。LEMON 为七域分别计数后等权平均，不能从宏平均 P/R 反算实测 F1。ChERRANT 调整后结果另存，避免混用。 |
| CGEC | `table_metrics`：ChERRANT 字符级编辑 P/R/F1，0–100；F1 从同次比对的 TP/FP/FN 计算。原有 F0.5 和 NaCGEC 双评分器平均分保留，不填进本表的 F1 栏。FCGEC 用有标签 valid；公开 test 不能本地填分。 |
| C³Bench | 分类 acc、一级出处检索 acc、NER 微平均 F1、标点 F1、中文 corpus BLEU，0–100。检索其他层级仍在 `tasks.retrieval.levels`；当前评分是明确记录的重建协议，未证明等价于官方实现。 |
| PPL | `domain_token_ppl` 按政务/影视/通用/游戏/科技/金融填六域；`avg_domain_token_ppl` 为六域算术平均，对应表内 AVERAGE。正式整体 PPL 仍按总 NLL/总目标数计算，不能与六域平均混淆。Base/Char tokenizer 不同，跨 tokenizer 比较还需查看 BPC/BPB。 |

表内 Base 指原基座；CSC/CGEC/Ancient 的 Base 行均评估从原基座训练出的对应 LoRA，而不是把未微调的 Base 直接用于下游任务。Char-CPT 与完整 PGCA-CPT 行同理。
