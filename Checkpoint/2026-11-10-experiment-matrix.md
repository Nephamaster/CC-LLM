# 实验结果表与三条模型路线

基线：`master / c072524` 加本轮工作区改动。目标是让 `experiment.xlsx` 的「Phase1&2」「CSC&CGEC」「Ancient」均有对应的训练、推理和取数入口；本轮没有执行真实模型训练或全量评估。

## 已核对并实施

| 表格内容 | 原能力 / 缺口 | 当前实现 |
| --- | --- | --- |
| Base | 下游 LoRA 插件本来就支持普通 Qwen3，总入口却默认拼 CPT 名称 | `MODEL_VARIANT=base` 直接从原基座做三项 LoRA，产物为 `B-CSC/CGEC/ACT` |
| Char-CPT | PGCA 预训练插件依赖分支、辅助头和专用 loss，不能只换模型路径 | `MODEL_VARIANT=char`：Phase1 主干训练 → `cpt/char.sh` 的 Phase2 全参数 CLM → 三项 LoRA；插件增加标准 Qwen3 的 CLM 路径 |
| 完整 PGCA-CPT | 原 Phase1、Phase2-A/B、LoRA 流程已有 | 保持原路线，默认 `MODEL_VARIANT=pgca` |
| 通用与 CLUE 指标 | 有原始结果，缺少表中五类 CLUE 汇总 | 每次 LMEH 额外写 `table_metrics.json`，统一 0–1 分数和明示的分组均值 |
| CSC / CGEC | CSC 已有句子级纠正指标；CGEC 原输出以 F0.5 为主 | 增加 `table_metrics`，CSC 取原始输出、LEMON 七域等权；CGEC 保留 F0.5 并用同一 TP/FP/FN 补 F1 |
| Ancient | 五类任务已有，但检索有多个层级 | `table_metrics` 明确取一级出处准确率；其他层级原始结果保留 |
| PPL | 已有六域和整体加权 PPL | 增加表格六域算术均值；整体加权结果和 BPC/BPB 保留 |

额外发现并修复评估总入口原有问题：`local suffix=... pointer=...` 在同一声明中引用尚未赋值的 suffix；同时 `LORA="$(...)" bash ...` 可能让 adapter 查询失败被后续命令状态掩盖。现在分开赋值，缺失指针会在启动下游推理前退出。Base 默认输入也不再误读 CPT 指针。

## 用户确认的比较设计

- Char 同样使用 Phase1 与 Phase2 数据。默认 Phase1 监督上限 1B、Phase2 10B，关闭 PGCA 与辅助任务，均训练主干。与完整路线数据池和预算对齐，但完整模型的 Phase2-A 冻结主干，优化过程并不相同。
- Phase1 后 PPL 比较原 Base、刚构建的 Char、完成 Phase1 的 PGCA-ALIGN。五个规模块的错误行名 `Char-PGCA-CPT` 改为 `Char-PGCA-ALIGN`；原有数值不改，其真实 checkpoint 身份未独立核验。
- 4B 预期值用批注逐格标为「情景估算，非实测」，原有两项 Base 数值保留。

详细指标对应与单位见 `scripts/evaluation/README.md`。FCGEC 表格采用有标签 valid，公开 test 无法本地填分；C³Bench 采用现有重建评分协议，不宣称官方实现等价。跨 Base/Char tokenizer 不能直接比较 token PPL，需同时查看 BPC/BPB。

## 命令入口

将 `B` 替换为实际原基座名称，GPU、batch、学习率继续由各阶段顶部参数和环境变量设置：

```bash
MODEL_VARIANT=base MODEL_NAME=B BASE_MODEL=/path/to/B bash scripts/train/run.sh
MODEL_VARIANT=char MODEL_NAME=B-Char PHASE1_MODEL="$PWD/models/B-Char" bash scripts/train/run.sh
MODEL_VARIANT=pgca MODEL_NAME=B-PGCA PHASE1_MODEL="$PWD/models/B-PGCA" bash scripts/train/run.sh
```

Char 的默认阶段为 `phase1,cpt,csc,cgec,ancient`；可用 `STAGES` 只选其中阶段。其 Phase2 输入用 `CPT_MODEL` 覆盖，默认读取 `B-Char-ALIGN/next_stage_model.txt`。三条路线统一评估入口：

```bash
MODEL_NAME=B MODEL=/path/to/B bash scripts/evaluation/run.sh
MODEL_NAME=B-Char-CPT bash scripts/evaluation/run.sh
MODEL_NAME=B-PGCA-CPT bash scripts/evaluation/run.sh
```

## Excel 修改与预期值依据

只修改目标三张表的数据/批注：100 个 4B 预期单元格（含三个 Avg 公式），以及五处 PPL 行名。其余原有数据、格式和公式保留。逐项对照了原文件的样式、行高、列宽、合并单元格及其他工作表属性，原 `styles.xml` 完全保留。

预期值是用于展示的情景，不是模型性能预测的实证结论：通用项根据原表已有两项 Base 的跨规模差值均值平移，再转移原表较小模型的改造差值；CLUE Char 缺少参考项，采用明示的替代假设；PPL 假设降低 15%；纠错预期 P/R 加 5 个百分点后构造 F1；古文指标加 5 个百分点，BLEU 加 3。所有假设均写入对应单元格批注，未经实验校准，也不是置信区间。

原文件备份、填表脚本与验证证据在 `/tmp/cc-experiment-matrix.oGwnWY/`。不新增仓库内测试/诊断脚本，也未安装新的仓库依赖。

## 验证与边界

- Shell 语法、Python AST 与 `git diff --check` 通过。
- 42 次隔离脚本调用验证 Base/Char/PGCA 阶段顺序、命名、指针、adapter 参数；另验证缺失 adapter 时停止。使用临时替身，不代表真实训练/推理通过。
- 小型原生 Qwen3 CPU 模型验证 Char CLM 前反向、全屏蔽补齐样本 loss 为 0、验证有效 token 计数，以及 Base/Char all-linear LoRA 回调。指标测试覆盖严格 GSM8K 筛选、SQuAD 单位、CLUE 缺失任务、ChERRANT F1 与古文检索层级。
- 临时环境：Python 3.11.15、CPU Torch 2.15.0、Transformers 5.19.0、ms-swift 4.6.0、openpyxl 3.1.5。它不是正式服务器的 Python 3.12/GPU 环境；真实 4B 权重的训练、vLLM 推理和全量评测仍需在服务器执行。
- 工作簿经 ZIP/XML、重新加载及逐项格式检查；没有用 Excel 客户端人工验收显示效果。
- 无提交、推送或分支切换；用户此前的大小写 Checkpoint 目录移动保持原状。
