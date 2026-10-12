# PGCA 直接相加实验

## 决策与边界

用户要求移除 PGCA 门控，并从头训练 Phase 1/2，两阶段保持同一份模型源码。不采用 gate 固定为 1 的阶段切换方案。

当前融合公式：

```text
X = input_layernorm(H)
A = SelfAttention(X)
P = o_proj(PGCAAttention(X, feature_memory, feature_mask))
U = H + A + P
H_next = U + MLP(post_attention_layernorm(U))
```

`src/modeling_qwen3_pgca.py` 已按以上方式融合两个分支，无需修改；门控原本位于 `src/pgca/attention.py` 内。本次删除该模块的 `gate` 参数和输出乘法，同步清理配置、构建初始化及验收条件。保留主干 MLP 的 `gate_proj`，它不是 PGCA 融合门控。

旧配置中的 `pgca_gate_init` 会被丢弃，不会写入新配置。构建与验证报告改用 `gate_free` 和 `gate_parameters`，不再要求 gate 等于初始化值。旧带 gate 的模型不满足当前验收条件。

本次未修改 loss、数据切分或评估逻辑；`26-10-12-phase2-inspection.md` 记录的验证 NaN 问题仍需独立处理，不能视为本次变更已修复。

## 原版备份

改动前逐字复制，使用 `.py.backup` 后缀，避免被当成 Python 模块加载：

- `src/pgca/attention.py.backup`
- `src/pgca/config.py.backup`
- `src/configuration_qwen3_pgca.py.backup`
- `src/pgca/build_pgca.py.backup`
- `src/pgca/validate_pgca.py.backup`

五份备份均在修改前通过 `cmp` 验证与原文件一致，修改后再次与改动前 HEAD 比对通过。未修改原有的训练检查记录。

## 两阶段训练的实际影响

- PGCA 从 Phase 1 开始就参与前向计算。即使 Phase 1 仅训练语义 embedding/lm_head，冻结的 PGCA 分支仍有输出；不再具有 gate=0 时的 Char-only 初始行为。
- Phase 2 可以继续使用全参训练，不需要 gate 设值、冻结正则或阶段切换插件。
- 模型目录使用独立打包的 remote code。修改仓库 `src` 不会自动更新已生成的模型目录，也不会更新旧的 ALIGN/CPT checkpoint。
- 新一轮应从 Char 迁移输入构建到新的输出目录，并在两阶段沿用这份无门控实现。不要直接运行会覆盖旧输出目录的构建命令。

以下命令在服务器仓库根目录执行，仅当新目标不存在时构建；本次没有执行真实模型构建或训练：

```bash
test ! -e models/Qwen3-1.7B-Base-Char-PGCA-ADD && \
python -B -m src.pgca.build_pgca \
  --model-path models/Qwen3-1.7B-Base-Char \
  --output-path models/Qwen3-1.7B-Base-Char-PGCA-ADD \
  --pgca-layers 11,12,13,14,15,16
```

构建成功后单独验证：

```bash
python -B -m src.pgca.validate_pgca \
  --model-path models/Qwen3-1.7B-Base-Char-PGCA-ADD
```

## 验证

`tests/test_pgca_ungated.py` 的 9 项测试全部通过：

- 输出等于未缩放的 PGCA 投影，参数与 state dict 均无 gate。
- 全 mask 位置输出为零，前向与反向有限；各 PGCA 投影从第一步即可获得梯度。
- decoder 保留 residual，并直接相加两个使用同一归一化输入的分支。
- 配置与迁移清理旧 gate 字段，初始化/模块检查拒绝残留 gate。
- 小型 causal LM 的 loss 和梯度有限。
- 扁平化 remote code 可独立保存、重载，重载前后 logits 一致。

测试环境为 `/tmp` 隔离依赖，Python 3.11.15、PyTorch 2.14.1+cpu、Transformers 5.19.0。没有修改现有 Python 环境，没有验证服务器依赖版本、真实 1.7B 权重或分布式 Swift 训练。测试使用 `PYTHONDONTWRITEBYTECODE=1` / `-B`，结束后确认项目无 `__pycache__`。`git diff --check` 通过。
