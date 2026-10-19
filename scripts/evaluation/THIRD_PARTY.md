# 迁移来源与协议

## CSC / CLLM

来源：用户本地 `/home/nepham/Lab/CLLM`，HEAD `c7cf30a5ff92e13e68c1d2a07c4cccc50f10f02c`。

- 原 `src/cllm/metrics.py` SHA256：`aac9ce2759b9ccd409803c3f807cb6677ead52f3a32fa8f3df5b35c1befd0197`。
- 原 `src/cllm/inference.py` SHA256：`e393c03abbcc381bfc6a0c8cb86f34eb31ae342c1c52deae218451ed4d66ad47`。
- 迁移评分与清理函数、回归测试；改包路径和 Python 子进程解释器。原始裸 prompt 推理不迁移，改为训练时的 ms-swift 模板。评分公式保持原样。

## PPL / CLLM

来源同上，原 `src/cllm/ppl.py` SHA256：`9b9179727d09d7ef89482f754fb7c55958277b1685520425cadf50b268cc2adb`。

沿用逐文档滑窗、无额外 BOS/EOS 的设计；修正字符分母和边界计数，改为直接从语言模型 logits 计算 NLL，增加全语料加权汇总、字节指标和身份记录。不是逐字迁移或旧报告数值完全等价实现。滑窗方法参考 [Transformers 官方 PPL 文档](https://huggingface.co/docs/transformers/perplexity)。

## 通用 benchmark / lm-evaluation-harness

框架使用普通安装的 [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)，固定 upstream `d6de81643928d653435c431bae19945d41d32520`，MIT 授权随包保留。用户本地 Nepham 仓库 HEAD `0d3003be83db889d9ddbe4f8bf0fb2d09c3f32ba` 相对此 upstream 只改了 gitignore，任务配置另在被忽略的 Nepham/tasks 中。

MMLU/ARC/GSM8K/CMMLU 以固定包的内置任务为准，运行时展开任务快照；CMMLU 将数据加载替换为 dashboard 路径的字符串 CSV 读取，保留内置中文 prompt、description 和汇总。SQuAD v2 继承内置 Python task，仅固定数据和官方 evaluate scorer revision，不改双请求及评分算法。数据、评分器 commit 和 150 个相关源码/任务文件 SHA256 见 `lmeh/protocol.lock.json`。

CLUE 的九项 YAML、两个模板、组配置和 utils.py 从 Nepham/tasks/clue 迁移，提示词/标签/CHID 展开/QA 指标保持原样；utils.py SHA256 `788ee2089769c35cb083d9d67fc022c59f1984f49378f8455be1e354a70e7216`。不迁移旧数据加载脚本 clue.py、未选中的额外任务或原始数据；OCNLI/DRCD 不默认加入，不删除源仓库任何文件。CLUE 是该本地协议在公开 validation 上的 LLM 测评，不声称等同官方隐藏测试榜。

## ChERRANT

来源：[HillZhang1999/MuCGEC](https://github.com/HillZhang1999/MuCGEC)，revision `39221934ee4ccf07801879136a75dd7e11319977`。

Python 代码使用 CLLM 已保留的版本（包含 char 模式下延迟导入 LTP 的兼容修复）；缺失字典从同一上游 revision 补齐。Apache-2.0 许可证保留在 vendor/cherrant/LICENSE。

本次局部改动：字典目录改为 assets（避免仓库 data/ 忽略规则漏传）；空字符串初始化为空 token 列表；parallel_to_m2 不再用 strip() 吞掉最后一条空预测的 TAB。未更改最短路径、编辑合并或评分算法。

## MaxMatch

来源：[NUS M2Scorer](https://github.com/nusnlp/m2scorer)，revision `cb874a7ecdcb360b08635538a299cf868258a4c1`。

仅迁移 levenshtein.py/util.py 和上游 LICENSE.md；源码版权/GPL 版本声明按原文保留。Python 3 适配为 lib2to3 机械转换＋包内相对 import，未重写 MaxMatch 算法。上游发布样例 P/R/F0.5=0.8 用作回归。word gold 使用上游 edit_creator.py 的图合并/无参考最短路径算法。

## CGEC

- [FCGEC](https://github.com/xlxwalex/FCGEC)，核对 revision `851cfb043e9cca8bc2f866d46987e10b24b1b9d6`；[操作格式](https://github.com/xlxwalex/FCGEC/tree/main/data)、[评分定义](https://github.com/xlxwalex/FCGEC/tree/main/scorer)。操作转换按原字段定义重新实现，避免源码中的 eval、跨参考累积修改及展开多个 label 时丢失其他操作。
- [NaCGEC](https://github.com/masr2000/NaCGEC)，核对 revision `da54f1e3ccb98ac5600fc3beac11c71fdc43ccb3`；按 README 的 THULAC/MaxMatch、ChERRANT 及均值公式实现。

## 古汉语

- [C³Bench 官方数据](https://github.com/SCUT-DLVCLab/C3bench)，[论文 v2](https://arxiv.org/html/2405.17732v2)：使用 Table 4 中文指令与 §4.3 指标类别；没有取得完整官方 scorer，具体实现约定见 README，不能把重建口径标为原论文代码。
- [Fùxì](https://github.com/cubenlp/FuxiBench)：核对本地 evaluate.py、src/utils.py、src/cipai_utils.py、src/eval_prompts.py。utils.py SHA256 `893dfad77c9e5a98209e85b54a6bccc8604664c6c4bc198d384d4eb3c815408c`；eval_prompts.py SHA256 `367bbb87ed2ec26c5a5fc5b9ac30fc6522377e42015894c4ddc4e959c3c8ecf3`。迁移选择题提取，保留词牌表，重现自动指标和 sft 裁判 prompt；移除对 vLLM/langchain 的强制依赖，不使用上游硬编码裁判地址。

第三方目录保留来源说明与能够取得的许可证；本文件不改变上游代码/数据的授权条件。
