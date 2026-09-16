# Source Inspection抽样决策

## 跨分片均衡扫描（2026-09-16）

旧Inspection只取排序后的前`max_files`个文件，并在这些文件上共享一份总扫描行数上限。对于按来源或内容类型聚集的分片，这会让扫描长期停留在第一个文件开头，例如peS2o前20,000条均为非`s2orc`时误报来源不可用。

现改为在全部输入文件中确定性等距选择首、中、末等代表分片，并为每个选中分片分别应用`max_rows * inspect_scan_multiplier`扫描预算。达到所需有效样本数后仍立即停止。报告新增`available_files`、`selected_files`和`scan_limit_per_file`，便于区分数据缺失与抽样偏差。
