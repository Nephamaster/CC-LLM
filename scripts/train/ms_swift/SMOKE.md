# 第二轮验收与实卡冒烟

本地数值与 CPU/DDP 回归不替代服务器真实 1.7B、BF16、NCCL 和显存检查。使用已准备好的第二轮**未训练初始模型**；不能用第一轮 ALIGN/CPT/for_exp，不能把冒烟后的模型接到正式第二轮。

## 服务器执行

先同步最新 smoke.sh、smoke_check.py、P1 插件及三个 CPT 脚本。在 CC-LLM 根目录，确认两张卡空闲后执行（0、1 仅为示例）：

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 GRAD_ACCUMULATION=2 \
PGCA_MODEL="$PWD/models/rnd2/Qwen3-1.7B-Base-PGCA" \
bash scripts/train/ms_swift/smoke.sh
```

参数集中在 smoke.sh 顶部。默认每卡 batch=1、梯度累积=2，以约四个满长优化步估算各阶段 token 预算；实际步数因切分/边界可能略多。两卡默认预算为 Phase1 32,768、A 65,536、B 65,536 个有效目标。使用 Phase1 2048、A/B 4096 的真实序列长度和 BF16、gradient checkpointing；WD=0.01，LR 沿用第二轮，LR warm-up 为 3%/3%/1%。不改三个正式脚本的默认超参数。

每次自动创建 outputs/experiments/rnd2/smoke/pgca-smoke.XXXXXX，所有数据、缓存、训练输出和完成指针都在其中。可通过 SMOKE_PARENT 改变父目录，不复用已有冒烟目录；不接入正式数据缓存，不覆盖第一轮产物。保存完整模型及 optimizer 状态，仍可能占用**数十 GB**磁盘；SAVE_TOTAL_LIMIT=1 在保留 best/last 时也可能留下两个 checkpoint。

## 自动检查什么

1. pip 依赖检查及每张可见 GPU 的 BF16 小矩阵前反向。
2. 用 checkpoint 配套代码加载模型，拒绝缺失/错配权重。检查完整 PGCA、上下文编码器、辅助头、0.03/0.03、10M-token warm-up、零 Phase2 计数与零初始化 O 投影。
3. 在真实模型上分别运行汉字、英文、零目标和可用的未知属性样本，检查损失、辅助 mask 和梯度有限；不做 optimizer step，不修改初始模型。未知样本从实际音形索引选择；若没有此类汉字，会明确记录 unavailable，不伪称该场景已覆盖。
4. 使用合成小数据依次运行 Phase1 → A → B；阶段开始时由现有插件检查每个参数的冻结/解冻状态。收尾检查冻结组梯度为零、可训练组至少在某步出现非零梯度、验证 loss 有限及新增 P1 日志。
5. 检查验证有效 token 数与准备报告一致、eval_loss 与 eval_total_loss 一致、分项加权比例正确，以及 A/B 区间不重叠、阶段累计辅助 token 数正确。

初始 O=0 和计数=0 **不能单独证明主干从未训练**，仍需保证模型确实来自你保留的未训练初始目录。预检不负责重建模型或重置任何权重。

## 不要误读结果

- 合成训练/验证内容有重叠：loss 和准确率只用于链路检查，不是效果评估。
- 保持真实 10M-token 辅助 warm-up，不加速、不伪造训练计数。两卡默认短跑结束只到 131,072 个 Phase2 目标，辅助 ramp 约 1.31%；报告会标记尚未覆盖完整 warm-up。0/5M/10M/20M 的数值逻辑由本地测试覆盖。
- 预检使用 eval 模式下的目标权重检查 loss/梯度，无更新；短训练仍按正常 ramp。第一次 O 更新前内部 Q/K/V/context 梯度为零正常，脚本按整个日志检查是否出现过非零梯度，而不要求每一步都非零。
- 部分梯度太小可能在日志舍入后显示为零。若报告没有观察到预期非零梯度，不要直接宣布断路；保留日志，必要时增加 SMOKE_STEPS 后重跑新目录。
- 两卡验证 NCCL/DDP 链路，但不等于八卡完整预算的吞吐或长期稳定性验收；单卡执行时 ddp_exercised=false。分布式数学归一化另由本地双进程测试验证。

## 跑完需要回传

优先同步本次目录中的：

- preflight_report.json、smoke_report.json；
- environment.log、preflight.console.log、phase1.console.log、phase2a.console.log、phase2b.console.log；
- 两个 cache_phase*/preparation_report.json；
- 若需深入核对，再同步各阶段最新 checkpoint 的 trainer_state.json、cc_llm_state.json、config.json，以及版本目录 args.json/logging.jsonl。

**不用上传模型权重、optimizer.pt 或 Arrow 缓存。** 中途失败也不要删除现场；没有最终 smoke_report.json，或 passed 不为 true，都不能算本轮冒烟完成。结果确认后，正式训练仍从未被冒烟修改的初始模型开始，用独立正式输出目录。
