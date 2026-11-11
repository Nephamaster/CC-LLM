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

下游 LoRA 评估使用 `MODEL=完整基座` 和 `LORA=专项 adapter checkpoint`，正式新增路线从 Base-INS、Char-CPT-INS、PGCA-CPT-INS 合并模型开始。总入口用 `MODEL_NAME` 区分路线，并读取 `outputs/train/<MODEL_NAME>-{CSC,CGEC,ACT}/next_stage_model.txt`；`TRAIN_ROOT` 可以更改训练根目录。`STAGE=prepare|infer|score|all` 可单独执行阶段，`RUN_DIR` 可接续指定版本。

`INFERENCE_BACKEND=auto` 对标准模型使用 vLLM，对当前 PGCA 使用 HF。显式 `INFERENCE_BACKEND=vllm` 会通过 vLLM Transformers backend 加载本地自定义模型；未完成真实一致性验证前不能替代 PGCA 的 HF 正式结果。

正式评估默认 `BATCH_SIZE=32 TENSOR_PARALLEL_SIZE=4`，下游 vLLM 的 `max_num_seqs` 随 batch 取 32。TP 只对 vLLM 生效；HF 后端不会因此启用四卡张量并行。单卡使用 vLLM 时显式设置 `CUDA_VISIBLE_DEVICES=<索引> TENSOR_PARALLEL_SIZE=1`。集中冒烟继续显式使用 batch 1、TP 1。

LMEH 默认 `EVAL_MODE=instruct`，实际使用保存的非思考 Swift 模板，并固定 few-shot 为多轮消息；原始模型对照使用 `EVAL_MODE=base`。详见 [通用协议](lmeh/README.md)。

## CGEC 数据与主指标

默认 `BENCHMARKS=nacgec` 在总入口表示 Dev 本地评分，`nacgec_test` 表示 Test 提交导出。子入口默认 NaCGEC `SPLIT=dev`、FCGEC `SPLIT=valid`：

```bash
# 沿用 MODEL、LORA、MODEL_NAME、GPU 等配置。
DATASET=nacgec SPLIT=dev bash scripts/evaluation/cgec/run.sh
DATASET=nacgec SPLIT=test bash scripts/evaluation/cgec/run.sh
```

Dev 读取 `nacgec.dev.ref.para`；Test 默认只读 `nacgec.test.input`，按原顺序生成 `nacgec.test.output`，不调用本地评分器。NaCGEC 官方已于 2023 年公开 Test gold，因此“官方没有 Test 参考答案”不成立；确需复现公开 Test 指标时显式设置 `REFERENCE_FILE=/path/to/nacgec.test.ref.para`。旧的隐式 Test gold 输入会在推理/评分前被拒绝，需新建运行；已有历史成绩不因默认协议变化就自动变成错误成绩。[官方说明](https://github.com/masr2000/NaCGEC)

FCGEC 主报 ChERRANT F0.5，F1 为补充；NaCGEC 有标签评分主报 ChERRANT 与 MaxMatch F0.5 算术平均，并保留两个评分器的 P/R/F0.5。当前分词和金标编辑构建口径写入结果，不把本地实现直接宣称为已验证的排行榜等价实现。

## 专项生成可靠性

C³Bench 默认上限：分类 32、出处 128、NER 256、标点 512、翻译 1024；Fuxi 选择题 32、其他任务 1024。可用 `CLASSIFICATION_MAX_NEW_TOKENS`、`RETRIEVAL_MAX_NEW_TOKENS`、`NER_MAX_NEW_TOKENS`、`PUNCTUATION_MAX_NEW_TOKENS`、`TRANSLATION_MAX_NEW_TOKENS`、`CHOICE_MAX_NEW_TOKENS`、`GENERATION_MAX_NEW_TOKENS` 覆盖，均受全局 `MAX_NEW_TOKENS` 上限约束。输出逐条记录 `generation_budget`。

三类专项评分的 `inference` 均记录停止、截断、重复率及分任务统计，分母为全部样本，未知停止原因不会算作正常停止。后端 `finish_reason=stop` 不等于已证明生成了 im_end；精确 EOS ID 在固定 Dev 验收中检查。CSC 主表用完整 `predict_raw` 评分，第一行清理和 ChERRANT 调整结果单独保留，避免用清理掩盖多余生成。

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
| CGEC | `score` 为主指标：FCGEC ChERRANT F0.5，NaCGEC 双评分器 F0.5 平均。`table_metrics` 同时保留 P/R/F0.5/F1，0–100；F1 只作补充。默认 FCGEC valid、NaCGEC Dev；默认 Test 导出不填分。 |
| C³Bench | 分类 acc、一级出处检索 acc、NER 微平均 F1、标点 F1、中文 corpus BLEU，0–100。检索其他层级仍在 `tasks.retrieval.levels`；当前评分是明确记录的重建协议，未证明等价于官方实现。 |
| PPL | `domain_token_ppl` 按政务/影视/通用/游戏/科技/金融填六域；`avg_domain_token_ppl` 为六域算术平均，对应表内 AVERAGE。正式整体 PPL 仍按总 NLL/总目标数计算，不能与六域平均混淆。Base/Char tokenizer 不同，跨 tokenizer 比较还需查看 BPC/BPB。 |

正式新实验三条路线均先完成 General-SFT，再训练独立专项 LoRA；模型名保留 INS 后缀。旧的原基座直接专项微调结果属于不同训练协议，应单独标注。C³Bench 的 `official_code_equivalence=false`：NER、层级出处、标点仍为内部重建口径，未经等价核验不得作为已确认官方分数。
