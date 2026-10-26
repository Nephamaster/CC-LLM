# P0：第一轮 PGCA 离线机制诊断

仅诊断第一轮 Phase2-B 的 **checkpoint-11500**，不训练、不更新权重、不改模型源码或训练插件、不改缓存，也不读取会随训练变化的 `next_stage_model.txt`。使用 checkpoint 自带的运行代码，不用当前 `src/` 替换它。P1/P2 尚未实施。

P0 完成后的最小补充对照见 [Uniform-Attention 实验](UNIFORM_ATTENTION.md)：复用已保存的相关输入和 Normal/Off 结果，只新增 Uniform 推理，不需重跑本页完整流程。

## 先做这两件事

1. 在服务器保留 **完整、已保存完毕的 checkpoint-11500**（权重分片及索引、config、tokenizer、配套 `.py`）。当前训练仍会轮换清理 checkpoint，建议将诊断副本放在训练目录之外，运行时用 `MODEL` 指向副本。无需上传本地，也无需停止 Phase2-B。
2. 审核 [polyphones_review.tsv](polyphones_review.tsv)。初稿为 100 条、50 个字的两种读音对照，**全部 `reviewed=false`，不是已验证的金标准**。确认该行读音可由左侧语境判断后改为 `true`，有歧义就先修改语境或读音。输入严格是 `left_context + target`，不追加后文；读音使用 checkpoint 的数字调号格式，例如 `hang2`、`zhe0`。没有右侧文字的前缀可以不是完整句子。

未审核的样本也输出候选 attention 排名供核验，但**不进入正式 `polyphone_top1`**；全部未审核时该值为 `null`。单读音、真实读音不在模型候选中、目标不能对应一个 token 等情况单独记录，不伪装成消歧错误，也不自动替换标注。审核后用新 `OUT` 重跑，不能覆盖旧报告。

## 运行

复用 `scripts/train/env/` 的 torch / transformers / datasets / numpy，不需要安装额外框架或启动 Swift/DDP。先确保指定 GPU 空闲；**如果目前所有卡都在训练，先完成审核并保留 checkpoint，等空闲卡或另一台机器再运行**。

在 CC-LLM 根目录，以下 `0` 仅为示例，确认空闲后再执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/diagnostics/run_pgca_probe.sh
```

模型已另存、输出需区分轮次时：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL=/实际路径/诊断副本/checkpoint-11500 \
OUT=outputs/experiments/rnd2/p0_pgca_probe_reviewed \
bash scripts/diagnostics/run_pgca_probe.sh
```

路径、样本数、种子、梯度 batch 行号等集中在 [run_pgca_probe.sh](run_pgca_probe.sh) 顶部，可直接改默认值，也可传同名环境变量。默认 `CACHE_DIR=data/cc_llm_multitask/phase2_4096`，读取 `val/`、`train_b/` 及 `preparation_report.json`。核验 tokenizer 哈希，不重新 tokenize 验证集，不丢掉 A/B 边界的 label mask。

可先用 `SAMPLES=2 BOOTSTRAP=100 OUT=outputs/experiments/rnd2/p0_preflight` 验证服务器加载与显存；这不是 512 条正式诊断。预检仍会执行 100 条短多音字前向和一次固定 batch 的三任务求梯度。默认单卡、一次一条序列；梯度阶段开销高于纯验证，不宣称已经通过真实 1.7B GPU 验证。显存不够时不要偷偷截短输入改变协议；可选一个较短的真实训练行并显式设置 `GRADIENT_ROW_INDEX`，报告会记录这一选择。

## 固定统计口径

- 验证集：seed=42，无放回抽取 512 条**已经编码的序列**，不是原始文档。每条原样用于 On/Off，`eval()`、`use_cache=False`。汇总 NLL 是 NLL 总和 / 有效 next-token 总数；准确率也按各自有效分母累计。
- `delta_clm_nll = off - on`，正数表示此 checkpoint 开启 PGCA 更好；准确率差为 `on - off`。所有差异报告 2,000 次配对序列 bootstrap 的 95% percentile CI。来自同一原文的分片可能相关，此区间不声称处理了文档内相关性。
- 辅助读音准确率是下一字**合法读音集合命中率**，不代表语境消歧能力。多音字诊断则观察输入中**当前目标字**的 attention，二者位置和含义不同。
- 层号是 **0-based 11–16**，启动时核对 checkpoint 配置。Self-Attention、PGCA 输出来自实际 forward；slot 权重捕获 PGCA 内现有 dropout 模块的输出（eval 下无随机丢弃），不重算替代 attention。
- 残差范数比与余弦只统计有有效音形 slot 的位置，不用非汉字的零输出稀释。主 attention 范数为零的位置不计算比值；任一分支零范数时不计算余弦，单独计数。输出 P50/P95/均值；`pgca_residual_cosine` 实际指 PGCA 与主 attention 输出的余弦，不是与整个 residual stream 的余弦。
- 模态权重先平均 attention heads，再合计拼音/字形 slot。额外报告“两模态都有效”子集，以及按有效拼音 slot 数分组的权重，避免把缺失模态误判为数量偏置。
- 多音字排名在每层先平均 heads，再合并重复读音 slot；至少两个**不同的合法候选**才可消歧。并列最大不算唯一正确 Top-1，保存 tie 数量与真实读音名次。
- 梯度 batch 默认 `train_b[0]`，不截断、不修改标签。独立计算 CLM、phonetic、glyph 三次 forward/梯度；均除以 CLM 有效监督数，glyph 为三项 CE 之和 / 3，保持当前插件口径。**不乘辅助权重、不做 warm-up 或 optimizer step**；另记录 checkpoint 权重和已训练 token 数，方便解释加权贡献。
- 梯度覆盖 Feature Embedding、Context Encoder、PGCA Q/K/V/O、PGCA Norm、字音/字形辅助头，以及所有 PGCA 参数的合并向量。CLM 与辅助目标的余弦使用相同参数坐标，未参与的梯度视作零；零范数时 `cosine_with_clm=null`。这只是一个 eval-mode batch 的方向，不能代表整个训练分布。

## 产物与验收边界

默认 `outputs/experiments/rnd2/p0_pgca_probe/`：

| 文件 | 内容 |
| --- | --- |
| `probe_manifest.json` | 固定 checkpoint、权重/模型配置 SHA256、可选 Git 信息、tokenizer/音形索引、环境、缓存准备报告、样本行号及协议 |
| `probe_inputs.jsonl` | 抽样验证输入和固定训练 batch 的原始 token IDs / labels，可精确追溯 |
| `probe_samples.jsonl` | 每条 On/Off 的损失总和、计数、差异；多音字候选排名和审核状态 |
| `probe_summary.json` | 配对汇总/CI、分层统计、多音字审核覆盖、重复前向和权重不变检查 |
| `gradient_alignment.json` | 三任务 loss、有效目标数、分模块梯度范数及与 CLM 的余弦 |

不对诊断脚本、checkpoint 配套源码或 Git diff 计算/校验哈希。仅保留模型权重与配置、tokenizer、数据和模型状态指纹。

Git 仅用于补充 commit 和工作区状态，不是运行前提。服务器只同步源码、没有 `.git`，或没有安装 Git、Git 查询失败/超时时，manifest 记录 `git.available=false`、空的 Git 字段及原因，继续正常诊断。无需 `git init` 或补传 `.git`。

脚本禁止覆盖任何旧结果，禁止输出目录与 checkpoint/cache 互相嵌套。若中途失败，保留已有产物，但**没有成功生成最终 summary 不算完成**；改用新的 `OUT` 重跑。

启用确定性算法，关闭 TF32；对第一条验证输入显式重复 On/Off，精确比较损失/计数和分层 attention/statistic 指纹；最终比较全部持久化模型状态的前后 SHA256。环境和硬件不同不保证逐位一致。再次以同参数、同输入运行到新输出目录，应得到一致的统计报告。

`status=partial` 的原因列在 `pending_acceptance`：审核未完成、候选覆盖不足，或固定训练 batch 缺少某项辅助监督。**不是其余计算失败，也不代表 P0 全部验收通过**。没有辅助监督的 batch 出现零梯度，不能证明分支断路；需显式换一个有有效汉字辅助目标的 `GRADIENT_ROW_INDEX` 并用新输出目录重跑。本次只固定 P0 manifest；第二轮整体 `experiment_manifest.json` 等 P0 结果、P1/P2 选择及新模型确定后再冻结，不把尚未确定的参数写成既成事实。

本地回归：

```bash
PYTHONDONTWRITEBYTECODE=1 scripts/train/env/bin/python -B \
  -m unittest discover -s tests -p test_pgca_probe.py -v
```
