# Benchmark去污染读取决策

## 多格式读取修复（2026-09-15）

`_benchmark_index` 原先将除Parquet外的所有评测文件按JSONL逐行解析，导致CMMLU CSV首先触发`JSONDecodeError`，后续LEMON TXT、CSCD-NS TSV、NaCGEC para以及FCGEC/Fuxi普通JSON也无法正确读取。

现统一按扩展名处理Parquet、CSV、TSV/para、TXT、JSON和JSONL。普通JSON支持FCGEC的ID映射结构、Fuxi的记录列表及嵌套任务分组；路径匹配只计入真实文件，避免目录被当作文本读取。六项格式回归测试通过。
