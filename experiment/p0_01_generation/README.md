# P0-01：生成异常定位

当前阶段交付固定样本和可运行的对照工具，最终根因结论需真实 Base-INS/Shared-INS 的服务器结果。实验代码、输入、日志与汇总均在本目录。该任务不重新训练模型。

## 固定对象

| 对象 | 默认路径，相对项目根目录 |
|---|---|
| Base-INS | outputs/train/Qwen3-1.7B-Base-INS/benchmark-checkpoint-1641 |
| Shared-INS | outputs/train/Qwen3-1.7B-Base-PGCA-Shared-CPT-INS/benchmark-checkpoint-1000 |

两者均是实际参与此前评估的完整合并权重。目录须含 config.json、args.json、Tokenizer、generation_config.json 及模型权重。使用已保存的 INS 模板，辅助损失关闭。

`cases.jsonl` 固定九条输入，来源均为最新正式评估，排除 smoke，不用旧版本补齐缺失。

| Benchmark | doc_id | 选择目的 |
|---|---|---|
| GSM8K | 0 | 给出正确答案后仍续写 |
| GSM8K | 19 | 两组数学推理及拖尾不同 |
| GSM8K | 40 | Shared 算出答案但缺少 strict 提取格式 |
| CMRC2018 | 0 | 正确实体后重复整句 |
| CMRC2018 | 1 | 答非所问和拖尾对照 |
| CMRC2018 | 4 | 短答案后追加无关片段 |
| SQuAD v2 | 0 | France 后出现无关尾缀 |
| SQuAD v2 | 3 | Rollo 后追加长解释 |
| SQuAD v2 | 6 | 不可回答问题仍作答 |

Shared 的历史 SQuAD 文件未完成，因此对应三个样本复用 Base 的题目与 Prompt 文本，再由 Shared 自己的 Tokenizer 编码。记录标记 historical_available=false，不声称这三条 Shared 异常已经在旧结果中观测到。

## 对照定义

| 组 | 路径 | 输入 | 停止方式 |
|---|---|---|---|
| A | Harness，Base 使用 vLLM，Shared 使用 HF | 历史 Benchmark Prompt | 原任务停止串与原 Harness 生成逻辑 |
| B | 原生 HF model.generate | 与该模型 A 组相同 Token IDs | 保存的 GenerationConfig 与原生 StopStringCriteria |
| C | Swift TransformersEngine | 与该模型 A 组相同 Token IDs | Swift 生成配置及停止处理 |
| D | Swift TransformersEngine | 同一道题的普通对话，使用保存的 INS 模板 | 与 C 同请求参数、任务停止串及模板停止处理 |

每次仅一个样本，greedy，Cache 开启，生成预算沿用历史任务（GSM8K 512、CMRC2018 64、SQuAD 256）。D 组去除 few-shot 和 Benchmark 外壳，阅读标题、文章和问题保留，不增加新的拒答提示或重复惩罚。C/D 的差异衡量 Prompt 整体影响，不能直接归因于 Chat Template 的某一个片段。

C 使用真实 Engine 的 pre_infer_hook，在生成前替换 input_ids/attention_mask，避免再次套模板。实际 model.generate 输入会逐 Token 检查，输入不一致时该样本失败。A 的历史文本重新编码后也检查实际送入模型的 IDs，未复用其他 Tokenizer 的数字 IDs。

当前单样本控制有意消除 Padding/同批请求影响，不能替代 P0-06 的完整批处理和多后端一致性验收。如果历史批量异常未在此复现，结论需保留这一限制。

## 服务器执行

先同步本仓库的整个 `experiment/`。在包含这两份完整合并模型的服务器项目根目录执行，固定使用物理 GPU 1：

```bash
GPU=1 bash experiment/p0_01_generation/run.sh
```

默认使用 `.venv/evaluation/bin/python`，四组在同一套现有依赖环境运行。每个模型/框架启动独立进程，顺序执行，释放上一组占用后再加载下一组。原始结果与完整样本记录不需要重新下载 Benchmark。

模型目录变更时可显式指定：

```bash
GPU=1 bash experiment/p0_01_generation/run.sh \
  --base /absolute/path/to/Base-INS-merged \
  --shared /absolute/path/to/Shared-INS-merged
```

终端显示正在执行的组、样本和日志路径。Ctrl+C 会终止当前子进程组，并汇总已经保存的记录。

## 失败重试与部分运行

某组失败会记录状态，继续独立组，不用另一个后端偷偷替代。依赖完整配置无法捕获时汇总也会标记。

例如只重跑 v0 实验的 Base A 组：

```bash
GPU=1 bash experiment/p0_01_generation/run.sh \
  --run-dir experiment/p0_01_generation/runs/v0 \
  --models base --groups A
```

该组写到 A/v1 等新版本，保留 A/v0；汇总取新版本，失败也不会回退到旧成功结果。`--run-dir` 重试沿用最初冻结的九条样本，模型路径必须一致。

## 输出与同步范围

```text
runs/vN/
├── experiment.json
├── inputs/cases.jsonl
├── base/
│   ├── prepare/vN/{cases.jsonl,model.json,runtime.json,status.json,worker.log}
│   └── A|B|C|D/vN/{generations.jsonl,runtime.json,status.json,worker.log}
├── shared/...
├── summary.json
├── summary.csv
└── summary.md
```

请把服务器生成的完整 `runs/vN/` 同步回来，包含失败组日志，随后继续分析 P0-01。无需复制模型权重。第一次本地目录若已含其他运行，实际版本号以终端打印为准。

`generations.jsonl` 每条保存完整 Prompt、预期与实际 IDs、原始生成 IDs、原始解码文本、框架处理后文本、有效生成配置、停止条件及触发事件、长度、重复特征和耗时。默认模型元数据和模板契约仅记录必要字段，不复制完整训练 args 或读取 HF_TOKEN/GITHUB_TOKEN 文件。

HF/Swift 没有显式返回的原生 finish_reason 保持 null，使用实际触发的 Transformers 停止条件解释结束。Swift 提供的 finish_reason 单独记录。vLLM 记录 RequestOutput.finish_reason/stop_reason，并在 SamplingParams.update_from_generation_config 被实际调用时捕获引擎默认值合入后的参数；没捕获就标记，不能假装已拿到完整配置。

停止条件通过观察原对象调用结果记录，保留原对象类型及其全部属性，运行结束恢复临时观察函数。不会设置新的 EOS、强制 EOS、重复惩罚或改变原有字符串停止逻辑。达到预算和遇到 EOS 可以同时发生，两种事实分别记录。

summary 生成的比较是观察事实，根因字段初始始终为 pending evidence review。不同输出不能自动等同于框架错误，需进一步查看配置和 Token 轨迹。统计耗时包含记录开销，不用于正式吞吐比较。

## 本地验证与当前状态

`verification/` 保存本地验证结果，使用临时小型 CPU 模型验证工具，不作为真实检查点验收。`checks.py` 覆盖真实 HF、Swift 输入注入、结束符/长度同时触发、观察后恢复、错误输入拒绝、vLLM API 观察的隔离单元检查、最新失败版本汇总。

当前已通过 7 项定向检查，真实 CPU 小模型跑通 A(Harness-HF)、B、C、D，A/B/C 的实际 Prompt IDs 一致。vLLM 的采样配置观察只做了模拟 API 单元检查，尚无实卡执行。本地临时环境为 Python 3.14，服务器仍使用现有 Python 3.12 环境。初次检查使用 WordLevel 测试 Tokenizer，不满足 Transformers 原生 StopStringCriteria 的词表清理假设；已改为 ByteLevel BPE 测试夹具并重跑通过。失败日志保留，不能把该测试夹具失败当作真实模型问题。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m experiment.p0_01_generation.checks
```

本地尚无实际 Base/Shared 模型权重和服务器 GPU。P0-01 的模型级结论与验收需要返回上述运行目录后完成。本项不开展官方模型对照、SFT 标签全量审计或 CLUE 协议整改。
