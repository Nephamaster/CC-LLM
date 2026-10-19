# 通用 benchmark 落地

## 已确认的实验约定

- 入口 `scripts/evaluation/lmeh/run.sh`，默认使用该目录已有 `env/bin/python`；允许 PYTHON 显式覆盖。PPL 继续使用激活环境，不受此改动影响。
- MMLU、CMMLU、GSM8K 5-shot，其余 0-shot；所有模型条件一致。MMLU 57 科，CMMLU 67 科；ARC Easy/Challenge 分开报告；CLUE 九项、不含 extra，使用 validation；SQuAD v2 同样 validation。
- 采用 HF 后端、原始任务提示词、不套 chat template。支持本地完整 Base/Char/PGCA 和 ALIGN/CPT checkpoint，不支持直接传 LoRA adapter 或用 vLLM 跑新 PGCA。
- 默认单进程单设备、bf16、batch=1、MAX_LENGTH=8192、seed=42。超长输入报错，不静默截断或减少 shot。生成预算 GSM8K 512 / SQuAD 256 / CLUE QA 64。

## 框架、任务与数据身份

本地 harness HEAD `0d3003be83db889d9ddbe4f8bf0fb2d09c3f32ba` 相对 upstream `d6de81643928d653435c431bae19945d41d32520` 仅有 gitignore 改动；固定普通安装 upstream，版本字符串 `0.4.14.dev0` 不能单独识别源码。protocol.lock.json 校验 150 个相关源码/内置任务文件，固定 HF 数据和 SQuAD evaluate metric revision。

运行时从包展开内置 MMLU/ARC/GSM8K/CMMLU 快照，保留 prompt、description、过滤器、分组加权算法。CLUE 从 Nepham/tasks 迁移到本仓库，不再依赖被源仓库 gitignore 忽略的任务文件。自定义 CMMLU CSV 加载保留所有字段字符串，防止 0.80 等被推断成数值。

除 CMMLU 外数据自动从 HF 拉取/复用缓存。CMMLU 默认 `/share/project/wuhaiming/data/dataset/cmmlu/{dev,test}/*.csv`，可用 CMMLU_DIR 覆盖；harness 原 HF 源有旧式 cmmlu.py，datasets>=4 不执行加载脚本，因此采用已确认的服务器 CSV。dev 每科至少 5 条；只有 test 不足以完成 5-shot。

## 值得保留的发现

- 旧 Nepham 的 Base/Char 5-shot 与 PGCA 0-shot 混用，不能直接比较。旧 cmmlu_local 与内置 cmmlu 的 prompt 不同。新结果不要直接合并旧结果表。
- `HFLM.generate_until` 会调用 left_truncate_len，不能仅靠传入 truncation=False 保证不截断。局部 StrictHFLM 在似然和生成入口检查实际 token 长度，超长先报错；未修改包的评分公式。
- SQuAD v2 的内置 Python task 构造器不直接接受所有 YAML 覆盖；通过子类仅固定 download revision 与 evaluate metric revision，保留 generate_until + unanswerable loglikelihood 双请求。某个小样本子集完全没有 NoAns/HasAns 时，继承原聚合器缺失项返回 0 的行为。
- MMLU results 的 n-shot 不只含 57 个叶任务，还可能含分组键；验证叶任务数量应查实际 task inventory，不直接断言 len(n-shot)==57。
- 本轮没有服务器 env 的本地副本。测试在 `/tmp/cc-llm-ppl-tests.2nhChi/env` 普通安装依赖；没有重建项目 env 或改服务器环境。安装锁不匹配会报错，先 STAGE=inspect 再按 README 补齐依赖。

## 验证状态

Python 3.13.15、torch 2.10.0+cpu、Transformers 5.12.1、datasets 4.8.5、evaluate 0.4.6、固定 harness。相关依赖检查通过。

- lmeh + PPL + 已有下游评估回归：82 passed、3 skipped；跳过的是已有可选集成测试。离线 fixture 覆盖七个入口的真实 harness 任务加载、few-shot、结果聚合、失败保护；CPU Qwen3 跑通七项流程，自包含 PGCA 跑通双样本似然与带 KV cache 的生成。
- 真实本地 CMMLU CSV：67 科、dev 335、test 11,582，全部加载通过。
- 真实 HF 固定 revision：MMLU abstract_algebra（test 100/dev 5）、ARC-Easy（test 2,376）、ARC-Challenge（test 1,172）、GSM8K（test 1,319）、SQuAD v2（validation 11,873）、CLUE AFQMC（validation 4,316）完整配置加载通过。不是全部 57 科 MMLU 或全部九项 CLUE 的线上验证。
- 官方固定 SQuAD metric：有答案/无答案两条自造样例 EM/F1=100，验证下载与评分链路，不代表真实模型得分。

尚未验证服务器真实 1.7B/PGCA 权重、CUDA 显存/速度或全量 benchmark 得分。LIMIT 为每叶任务限制，报告标记 smoke_only；需要独立 RUN_NAME，不能把冒烟当正式结果。输出目录存在即拒绝覆盖，不自动断点续算。
