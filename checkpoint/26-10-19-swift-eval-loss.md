# ms-swift 验证 loss 口径修正

## 现象与边界

三阶段双卡冒烟中，默认 `eval_loss` 是 `eval_total_loss` 的两倍。插件的 `eval_clm_loss`、`eval_total_loss` 已按去除 DDP 尾部重复样本后的全局有效 token 统计；框架默认 loss 的分布式缩放及 batch 均值聚合不能直接替代这些指标。

训练路径使用每卡损失总和、全局监督目标分母，以及框架的 world-size 补偿；本次不修改这条路径，也不关闭 `average_tokens_across_devices`。选模仍使用 `eval_clm_loss`。

## 实施

在 `scripts/train/ms_swift/cpt/plugin.py` 中，仅验证模式额外注册 token 加权的 `loss` 指标。ms-swift 的 `SwiftMixin.log` 会在写入日志、触发验证回调和返回 `evaluate()` 结果前，将其覆盖到 `eval_loss`。

- Phase1：`eval_loss = eval_total_loss = eval_clm_loss`。
- Phase2：`eval_loss = eval_total_loss`，包含辅助损失；PPL 仍只取 `exp(eval_clm_loss)`。
- 使用独立的指标累加器，避免多个指标共享对象、被框架依次 reset 后丢失统计。
- 不对旧日志做除以卡数的追溯修正：不同长度 batch 还涉及聚合权重，简单除法不构成通用修复。

## 验证

隔离环境使用 Python 3.13、PyTorch 2.10.0 CPU、普通安装的项目锁定 ms-swift commit `ce28e0288657d6472a439ea219f1bcc77310e764`，没有使用外部框架源码的 editable/PYTHONPATH 安装。

- Transformers 5.12.1：`test_ms_swift_multitask.py`、`test_pgca_context.py`、`test_pgca_ungated.py`、`test_project_layout.py` 共 39 项和 12 个子测试通过。
- Transformers 5.19.0（服务器日志记录的版本）：`test_ms_swift_multitask.py` 的 8 项和 6 个子测试通过；隔离环境依赖检查通过。
- 覆盖三阶段单进程/双进程 Gloo 训练、验证、保存，变长验证样本及 DDP 尾部去重，训练 DDP 梯度与单大 batch 等价；新增断言检查 `eval_loss` 与 token 加权指标一致。
- CPU 回归不替代服务器 CUDA/NCCL 复验。

## 使用

同步插件后重新启动训练/验证即可；不需要重建模型、词表或数据，也不需要因此重训旧 checkpoint。历史报告不会自动更新。服务器复验需使用新的冒烟输出目录，确认上述指标关系。
