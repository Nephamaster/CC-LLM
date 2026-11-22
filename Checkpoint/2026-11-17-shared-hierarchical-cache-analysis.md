# Shared / Hierarchical 最新构建的 Cache 数值对照失败

输入：models/Qwen3-1.7B-Base-PGCA-Shared/reports/pgca_validation_report.json，与之前 Hierarchical 模型同名报告对照。Shared 报告比 Hierarchical 新；只据最终 PGCA 报告判断，未使用继承的旧词表/embedding 报告。

两者结构、配置、权重/Feature Index 就绪、普通前向与 loss 检查均为 true，唯一失败字段为 cache.passed（及由它汇总的顶层 passed）。两者 Cache 最大 logits 误差均 0.5，Top-1 一致率均 1.0，使用 BF16 比较阈值 atol=0.1/rtol=0.02。Shared 的单 feature cache、Hierarchical 的三个 feature cache 均与主干按 1/2/3 同步递增。

因此该现象不是新增三级结构独有，需优先调查两者共有的主干/缓存路径和数值验证环境。当前校验器默认 CPU、dtype=auto；CPU BF16 的整段与逐 Token 计算差异值得优先对照，但报告本身不能证明只是精度问题，也不能仅凭 Top-1 一致放行。

两者配置都开启 PGCA 输出零初始化。按构建实现，若权重确实对应本轮新建产物，初始 PGCA 分支不应改变主干 logits；应在有权重的服务器核对这一点，并做同模型 BF16/FP32 × PGCA Off/On 的四组 Cache 对照。Off 需调用 _compare_cache，不能调用会临时启用 PGCA 的 _check_cache，否则对照失效。

本地没有实际模型权重，无法复现真实规模数值；尚不能确定误差起于哪个主干层或是何种算子。未修改源码、阈值、模型或报告，未建议再次构建模型或直接开始正式训练。

## 四组实测结果补充

用户提供根目录 pgca_cache_comparison.json，测试 Shared 模型，CPU、SDPA，输入 IDs `[15946,28392,25403]`。四组均确认实际 PGCA 输出投影全零，Top-1 一致率 100%，Cache 长度同步：

| 精度 | PGCA | Cache 对照 | 最大绝对差 |
|---|---|---|---|
| BF16 | Off | false | 0.5 |
| BF16 | On | false | 0.5 |
| FP32 | Off | true | 2.5987625122070312e-05 |
| FP32 | On | true | 2.5987625122070312e-05 |

确认本次失败在不启用 PGCA 时仍复现，且在相同 CPU/SDPA 条件下改用 FP32 后满足更严格的 1e-4/1e-4 门槛。此前关于 CPU BF16 数值路径差异的假设获得对照支持；不能据此指认某个具体算子，也不应把本次失败归因于新增三级结构。

建议构建期结构/Cache 校验使用 CPU FP32，并将 BF16 作为独立诊断；不通过直接放宽容差来解决。已有模型权重不需要重建。尚需在 Hierarchical 实际权重上做 FP32 复核，以及真实 GPU/训练后非零 PGCA 输出时的检查；当前零投影对照不能证明所有音形缓存内容在训练后均正确。本轮只更新分析记录，未改程序或用户结果文件。
