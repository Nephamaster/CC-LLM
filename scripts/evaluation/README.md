# 下游评估

入口位于 `scripts/evaluation/{csc,cgec,ancient}/run.sh`。通用 benchmark 统一归入 `scripts/evaluation/lmeh/`，PPL 归入 `scripts/evaluation/ppl/`；本轮只统一布局，不补齐它们尚缺的脚本。训练入口位于 `scripts/train/ms_swift/`。

原始 `data/downstream/` 只读。prepare 只在 OUT 下生成评估输入，不重采样、不重新划分、不删除 rSIGHAN/ECSpell 或任何其他数据。

## 环境与训练模板

每个下游评估子目录建立自己的 `env/`，通过普通安装获得框架与评分依赖，不使用外部源码路径或 editable 安装。以下以 CSC 为例；CGEC、古汉语分别将环境路径换成 `cgec/env`、`ancient/env`：

```bash
python3.11 -m venv scripts/evaluation/csc/env
# GPU 环境先安装与服务器 CUDA 匹配的 PyTorch
scripts/evaluation/csc/env/bin/python -m pip install -r scripts/evaluation/requirements.txt
scripts/evaluation/csc/env/bin/python -m pip check
```

这里的推理仍使用普通安装的 ms-swift 的 TransformersEngine 和实际 get_template，保留自定义 PGCA AutoClass 加载路径，支持完整 SFT 模型和基座＋LoRA。该依赖与 lmeh 的通用 benchmark 环境相互独立。

模板只来自 SFT checkpoint 的 `args.json`（也允许其直接父目录）；LoRA 优先使用 adapter 的训练参数。继承 template、system、use_chat_template、enable_thinking、response_prefix 等，不提供 `--template` 覆盖或旧裸文本模式。不猜测训练设置：缺失 args.json、插件文件或 use_chat_template=false 的 CPT checkpoint 会报错。导出合并模型时应保留正确的 SFT args.json，不能复制无关的 CPT 参数。

任务指令和 chat template 是两层：CSC 指令保持 CLLM 原句式及尾部空格；C³Bench 用论文中文任务指令；Fùxì 用原数据 instruction/input。这些 user 消息再由训练模板编码。参考答案不会传入待测模型；Fùxì 明确启用的官方 few-shot 示例除外。

评估不沿用训练时的截断策略：限制在模型位置容量减去生成预算内，超长请求报错而不是切掉题目。默认 greedy，参数和训练模板写入推理 Manifest；需要与其他论文推理设置比较时另行说明。

## 启动

所有脚本可在任意工作目录通过绝对路径调用；下面示例在本仓库根目录。`MODEL` 是真实 checkpoint 路径；`MODEL_NAME` 是评估模型的规范名称，不能用 `checkpoint-1000` 代替模型名。默认结果写入 `outputs/evaluation/<评估项>/<MODEL_NAME>/`，split/few-shot 在其下继续分目录。下方使用占位 checkpoint 路径：

```bash
# CSC：CSCD-NS + LEMON
CUDA_VISIBLE_DEVICES=0 MODEL=/path/to/csc-sft MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-CSC \
  bash scripts/evaluation/csc/run.sh

# 基座 + LoRA：MODEL 指向基座，LORA 指向 adapter checkpoint
CUDA_VISIBLE_DEVICES=0 MODEL=/path/to/base LORA=/path/to/csc-adapter \
  MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-CSC RUN_NAME=lora-run1 bash scripts/evaluation/csc/run.sh

# FCGEC valid 本地评分
CUDA_VISIBLE_DEVICES=0 MODEL=/path/to/cgec-sft DATASET=fcgec SPLIT=valid \
  MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-CGEC bash scripts/evaluation/cgec/run.sh

# FCGEC test：只导出预测，不伪造本地分数
CUDA_VISIBLE_DEVICES=0 MODEL=/path/to/cgec-sft DATASET=fcgec SPLIT=test \
  MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-CGEC bash scripts/evaluation/cgec/run.sh

# NaCGEC test：官方双指标
CUDA_VISIBLE_DEVICES=0 MODEL=/path/to/cgec-sft DATASET=nacgec SPLIT=test \
  MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-CGEC bash scripts/evaluation/cgec/run.sh

# 古汉语：两个 benchmark 分开运行、分开报告
CUDA_VISIBLE_DEVICES=0 MODEL=/path/to/ancient-sft DATASET=c3 \
  MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-ACT bash scripts/evaluation/ancient/run.sh
CUDA_VISIBLE_DEVICES=0 MODEL=/path/to/ancient-sft DATASET=fuxi \
  MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-ACT bash scripts/evaluation/ancient/run.sh
```

公共环境变量：`PYTHON`（默认各任务子目录的 `env/bin/python`）、`MODEL_NAME`、`RUN_NAME`（可选模型内实验子目录）、`OUT`（可显式覆盖默认路径）、`BATCH_SIZE`（默认 8）、`MAX_NEW_TOKENS`（CSC/CGEC 1024，古汉语 2048）、`STAGE=all|prepare|infer|score`。古汉语额外支持 `STAGE=judge`，Fùxì 可设 `FEWSHOT=5`，默认 0。LoRA 的 MODEL_NAME 应描述 adapter 所属模型，不是基座；同一模型不同 checkpoint 比较时设置不同 RUN_NAME，避免结果身份冲突。

已有输入和成功预测可复用。输入、checkpoint 路径/元数据/权重文件 stat、训练参数、模板依赖、代码或生成参数改变时，拒绝复用推理结果，应使用新 OUT。权重身份记录的是路径/大小/mtime，而非对全部大权重逐字节哈希；配置、tokenizer、输入按 SHA256 校验。只有本程序 retry journal 的未写完末行允许恢复，原始数据和损坏中间记录不做修复。推理异常保留记录并在重跑时重试，不把异常伪装成模型空答案参与评分。

当前推理缓存身份尚未把所有框架版本纳入拒绝复用条件；切换 env 或升级 Torch/Transformers/ms-swift 时应另设 RUN_NAME/OUT。评分报告会记录依赖版本，但这不等于旧预测已在新环境重新生成。

只做输入准备，不需要加载模型：

```bash
STAGE=prepare MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-CSC bash scripts/evaluation/csc/run.sh
```

## 评分口径

### CSC

从 `~/Lab/CLLM` 迁移原 metrics、预测清理及 ChERRANT 等长替换逻辑，不改变指标公式。启动脚本仅选择 `cscd_test.jsonl` 和 LEMON 的 `car/cot/enc/gam/mec/new/nov.jsonl`，保留 dataset/subset/src/tgt 与文件内顺序。

输出 raw 与 ChERRANT 两套句级/字级检测纠正指标、FRP、空/不等长输出统计及分数据集宏平均。注意原 CLLM 字级 raw 统计会跳过不等长预测，因此必须同时看不等长率及 ChERRANT 表，不能只挑 raw 字级 F1。ChERRANT 后处理仅保留等长 substitution，**不是 CGEC 的 span-level F0.5**。

### CGEC

- FCGEC valid：解析每条 operation 的全部多参考标签。按官方 tag 优先级处理 Switch/Delete/Insert/Modify，包括真实数据中的覆盖操作、Modify 内 INS 标签及不含尾标点的 Switch；不使用 eval 解析 tag。按官方说明先去标点，报告错误句子上的多参考 Exact Match 和 ChERRANT 字级 span P/R/F0.5。
- FCGEC test：公开文件无 gold，生成 UUID→预测 JSON 和 `UUID,Sentence,Correction` CSV。它们是 correction 结果交付文件；当前平台的最终提交容器/列要求仍需在提交前核对，未验证可直接上传的 ZIP，不自动上传。不会输出 test 指标，不生成未经预测的错误类型标签。
- NaCGEC：保留 `.ref.para` 的所有参考答案；字级使用固定 ChERRANT，词级使用 THULAC `seg_only=True`＋固定 NUS MaxMatch，最终 `score=(word_F0.5+char_F0.5)/2`。本地只有平行参考，没有官方固定 word-gold M2，故用 NUS 自带 edit_creator 的双距离图算法生成 gold（max_unchanged_words=0），系统评分 max_unchanged_words=2。报告保留这些参数及生成的 M2，不宣称未经验证地逐位复现官方排行榜。

ChERRANT CGEC 路径保留上游默认 target 简体化及 all 最短编辑策略；CSC 迁移路径保留 CLLM 的 `--no_simplified --multi_cheapest_strategy first`。二者不要互换。MaxMatch 核心从官方 Python 2 源码机械转换到 Python 3，许可证与修改记录保留。

### 古汉语

- C³Bench：原始 10,000 条记录分别形成五任务共 50,000 条请求。分类 accuracy、分层书名检索 accuracy、实体集合 micro-F1、标点位置＋符号 micro-F1、corpus SacreBLEU（zh）。标点任务若改动正文字符，其正确标点数按 0 计，不能仅按偏移碰巧得分。官方仓库未提供完整 scorer，以上按论文重建，具体分词/后处理/聚合口径写入报告，`official_code_equivalence=false`。
- Fùxì：21 任务，3 项选择题 accuracy、6 项 sentence SacreBLEU 均值、2 项格式检查、10 项模型裁判。选择题提取沿用官方实现；词牌使用官方首个 longyusheng_template 及标题/重复段规则。对联和词牌分数只反映格式，**不验证完整文学质量或平仄音韵**。
- 默认 Fùxì zero-shot。可选 5-shot 只读取给定 `dev_fewshot.json`，不会从 test 临时挑示例。本地提供的示例与 119 条测试输入有文本重合，逐条记录 demonstration_overlap，不删除任何测试样本；使用 few-shot 结果时必须披露，不能视作严格未见测试。

## 可选模型裁判

不配置裁判时，Fùxì 自动指标正常评分，10 项裁判任务显示 pending、score=null，complete=false；不使用 Exact Match 代替，不计算伪完整总分。

需要时设置独立的 `JUDGE_MODEL`、`JUDGE_BASE_URL`、`JUDGE_API_KEY` 环境变量，然后：

```bash
DATASET=fuxi STAGE=judge MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-ACT \
  bash scripts/evaluation/ancient/run.sh
DATASET=fuxi STAGE=score MODEL_NAME=Qwen3-1.7B-Base-PGCA-CPT-ACT \
  bash scripts/evaluation/ancient/run.sh
```

`STAGE=all` 仅当 DATASET=fuxi 且显式设置 JUDGE_MODEL 时自动调用裁判。接口为 `JUDGE_BASE_URL/chat/completions`，需要配置到 `/v1` 等 API base。凭据不写脚本、参数文件或日志，不读取仓库 `HF_TOKEN`。调用会将对应测试题、参考答案及待测输出发送至所配置服务，并可能产生费用。

每条裁判结果绑定预测内容、裁判模型/地址、prompt 哈希，成功结果可复用；格式不合法/API 错误保持 pending，不计为错误答案。不同裁判配置不可混合使用，原始判断保留在 judgments.jsonl。

## 产物与验证

OUT 下保存 input.jsonl/manifest、predictions.jsonl/manifest、retry journal、metrics.json，以及评分所需的中间 M2/分词文件。例如 `outputs/evaluation/fcgec/Qwen3-1.7B-Base-PGCA-CPT-CGEC/valid/`。原始回答与清理后回答分别保存，报告统计空输出、finish_reason 和依赖/评分代码哈希；代码指纹排除 env/.venv，不扫描或绑定整套虚拟环境。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
  tests/test_evaluation.py tests/test_evaluation_csc_migration.py

# 含真实 THULAC 与 ms-swift CPU 小模型推理
EVALUATION_INTEGRATION=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
  tests/test_evaluation.py tests/test_evaluation_csc_migration.py
```

完整数据适配验证不等于模型评测完成。CPU 小模型测试不代表服务器 PGCA 权重、GPU 显存或全量 benchmark 分数已经验证。

官方依据与第三方版本见 [THIRD_PARTY.md](THIRD_PARTY.md)。
