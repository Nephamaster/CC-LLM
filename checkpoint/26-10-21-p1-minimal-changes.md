# P1 最小改动：辅助权重与日志

## 范围

- 仅落实 TODO 第 1 项。未创建/覆盖实际模型，未启动服务器训练，未调整三个 CPT 阶段的 batch/LR/WD，未实施 P2 或新的诊断条件。
- 新构建默认字音/字形权重为 0.03/0.03。构建 CLI 提供 --aux-phonetic-weight、--aux-glyph-weight；scripts/build_pgca.sh 顶部提供对应 AUX_PHONETIC_WEIGHT、AUX_GLYPH_WEIGHT。显式值写入 config.json 和构建报告。
- 旧 checkpoint 的显式 0.05/0.02 保持不变，不因源码默认值变化自动迁移；旧 config 保存/加载回归已覆盖。实际第二轮初始模型准备仍是 TODO 第 2 项。
- 保持 10M-token warm-up、合法读音集合、next-token shift、mask、CLM 有效目标分母、字形三项平均、模型前向及辅助头结构不变。

## 日志口径

- 新增 weighted_phon_to_clm、weighted_glyph_to_clm，保留 weighted_aux_to_clm_ratio。先累计包含训练 ramp 的加权损失分子及 CLM loss 分母，再取比值；不是逐 batch 比率的平均。eval 仍用去重后的全局有效统计。
- 梯度组：Feature Embedding、Context Encoder、PGCA Q/K/V/O、PGCA Norms、字音头、字形头；保留 backbone/branch 总组。每个参数的梯度只求一次平方和，不增加独立任务反向，不修改梯度或参数。
- 当前 Transformers Trainer 的 on_pre_optimizer_step 在裁剪后、更新前触发。因此记录的是累积/同步后的联合梯度范数，并在日志窗口对优化步取均值，不同于通常裁剪前的默认 grad_norm，也不是 Adam 更新量。
- 无梯度/冻结组记录 0；零初始化 O 后第一步内部 Q/K/V/context 梯度为零仍属预期。口径针对单卡/普通 DDP，不冒充 FSDP/ZeRO 的全局分片范数。

## 验证

- 105 项模型、feature 构建、诊断及脚本回归通过；另 14 项 ms-swift 回归通过，合计 119 项（不重复计数）。
- ms-swift 回归包含真实单进程三阶段训练/验证/保存、双进程 Gloo 三阶段流程、分布式梯度对单大 batch、warm-up 0/5M/10M/20M、新旧权重、英文/未知属性/无目标 batch、损失比例、冻结组及梯度不变性。
- 构建回归覆盖默认与自定义权重保存、remote-code 重载、CLI/bash 参数透传、输入 config 不变及已有输出保护。
- 本地环境为普通安装的 ms-swift，PyTorch 2.10.0 CPU / Transformers 5.12.1；不是服务器 1.7B CUDA 冒烟。Shell 语法、diff 空白检查通过，未生成项目 __pycache__。
- 初次 Phase2-B 数据导出测试因沙箱禁止本地 IPC 报 SyncManager PermissionError；在允许本地通信的隔离临时测试中重跑通过，没有为绕过沙箱修改框架或数据逻辑。

## 下一步

先完成 TODO 第 2 项：独立初始模型及其配置核验；之后做第 3 项真实模型实卡冒烟。直接运行原阶段脚本仍可能加载首轮模型或采用旧训练超参数，不能据 P1 代码完成就视为第二轮已经准备好。
