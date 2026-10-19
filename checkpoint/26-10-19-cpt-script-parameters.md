# CPT 训练参数入口

- `common.sh` 顶部维护两阶段共享超参数；`phase1.sh`、`phase2.sh` 顶部维护阶段路径、长度、预算、学习率与恢复入口。环境变量优先于脚本默认值，默认训练方案不变。
- `PHASE2_B_BRANCH_LR` 通过环境传给插件，仅控制 Phase2-B 新增分支，默认仍为 `3e-5`。它与 `PHASE2_B_LR`（主干）独立；同步到服务器时不能只更新 shell 而遗漏 `plugin.py`。
- `DATA_SEED` 同时传给 Swift 导出、缓存分段打乱和训练采样；`prepare_data.py --verify` 会检查已存 seed。修改 seed 后使用新缓存，不静默复用旧排列。默认 seed=42 的已有兼容缓存仍可使用。
- `MAX_LENGTH` 同时控制准备与训练，并参与默认缓存目录名。packing、单轮 token 预算、阶段冻结和选优规则仍是执行区固定约束。
- 辅助任务权重及 warmup 仍取模型 config 的 `pgca_aux_*`，不是学习率参数。

验证：Bash 语法及差异检查通过；42 项测试、21 个子测试通过，覆盖默认值/环境覆盖、参数分组学习率、缓存 seed 校验和三阶段 CPU 单/双进程回归。未重新执行服务器 GPU 训练。
