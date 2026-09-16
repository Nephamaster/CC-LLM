# Mixture写出类型决策

## `sample_key`无符号整数修复（2026-09-16）

Candidate及去污染产物将稳定采样键声明为Arrow `uint64`。Mixture曾使用`Table.from_pylist()`重新推断类型，导致大于`2^63-1`的合法采样键被按有符号`int64`转换并触发`OverflowError`。

Mixture Writer现直接复用上游去污染Parquet的完整Schema，并在启动时要求`sample_key`存在且为`uint64`。这样同时保留所有字段类型，避免后续阶段再次依赖Python值推断Arrow类型。
