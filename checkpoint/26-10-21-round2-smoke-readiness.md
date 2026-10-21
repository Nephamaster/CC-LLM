# 第二轮第 3 项：本地验收与服务器冒烟交接

## 已完成

- 修订 scripts/train/ms_swift/smoke.sh：去掉硬编码服务器 cwd、旧初始模型和旧 outputs/cpt 路径；显式指定空闲 GPU，默认读取 models/rnd2/Qwen3-1.7B-Base-PGCA。
- 每次自动生成 outputs/experiments/rnd2/smoke/pgca-smoke.XXXXXX，所有数据、缓存、阶段模型和完成指针留在本次目录。正式数据/模型与第一轮产物不变；正式三阶段脚本默认超参数未修改。
- 新增 smoke_check.py：实际模型加载、0.03/0.03、10M warm-up、初始计数/O 投影、汉字/英文/未知属性/零目标 loss 与有限梯度检查；后处理验证冻结梯度、分项比例、验证 token 数、eval_loss 缩放和 A/B 累计进度。不新增源码哈希或 Git 依赖。
- 短训练使用 2048/4096 长度、BF16、gradient checkpointing、WD=0.01 及第二轮 LR/warm-up；降低梯度累积与 token 预算。两卡、累积 2 时预算为 32,768 / 65,536 / 65,536；原 10M 辅助 warm-up 不改。
- 123 项模型/数据/脚本/冒烟检查回归及 14 项 ms-swift 集成回归通过，共 137 项；后者包含真实 CPU 三阶段和双进程 Gloo 测试。新预检检查权重不变，失败报告不能伪装成 passed。Shell 语法及空白检查通过。

## 必须保留的验收边界

- 本地没有第二轮 1.7B 权重和 GPU，没有执行服务器实卡冒烟；TODO 第 3 项尚未整体完成。
- 零 O、零 Phase2 计数并不能单独证明主干未训练；模型来源仍必须是保留的未训练初始目录，不能用 ALIGN/CPT。
- 实际索引没有缺失属性汉字时，报告明确标记未知样本不可用，不假称已覆盖。短冒烟不会跑完 10M-token warm-up，报告保存实际覆盖比例；完整 ramp 数学逻辑由本地测试覆盖。
- 冻结组应为零，可训练组在整个短跑中应观察到非零梯度。初期 O=0/学习率 warm-up 导致内部梯度为零是预期，不要求每步非零；日志舍入也可能隐藏极小梯度，失败后先核对日志。
- 合成训练/验证内容重叠，不能评价模型效果；冒烟 checkpoint 不用于正式第二轮。完整 checkpoint/optimizer 仍需数十 GB 磁盘空间。

## 用户动作

按 scripts/train/ms_swift/SMOKE.md 选择两张空闲 GPU 执行。回传 preflight_report.json、smoke_report.json、三阶段 console log 与准备报告；必要时追加 trainer_state.json、cc_llm_state.json、config.json、args.json/logging.jsonl。不要上传大模型权重、optimizer 或 Arrow 缓存。
