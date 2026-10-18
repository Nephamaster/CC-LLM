# 音形上下文与辅助预训练实施检查点

## 当前交付

- 基于现有无门控 PGCA 与音形 schema v2，不重新生成语义 tokenizer/token ID。
- 新增共享 `CausalFeatureEncoder`：默认 4 层、512 维、8 头、1536 SwiGLU；按文本位置因果建模，保留候选级 slots，所有 PGCA 层复用一次 forward 的结果。候选宽度读取 manifest，当前默认 16＋1，而不是旧方案的 8＋1。
- HF `PGCACompositeCache` 分别维护主干/特征 KV，支持增量解码、beam 重排、裁剪和重置。首版只支持 DynamicCache；不宣称支持 StaticCache、packing 或 vLLM。
- 辅助头注册于模型内部：带调拼音集合似然、三个字形属性分类。合法输出类排除 ID 0/1/2；所有任务对齐 `h_t -> labels[t+1]`，使用共同 CLM 有效目标分母。
- 新模型构建使用 `--context-encoder --auxiliary-heads --zero-init-output`，PGCA 输出投影仅在构建时清零。加载训练 checkpoint 时不重新初始化。
- Phase1 沿用 `pgca_enabled` 完全旁路音形分支，冻结整个 `feature_memory_builder`、PGCA 与辅助头；Phase2-A 训练新增分支，B 全参联合训练。三阶段源码一致，旧配置默认不创建新模块。
- 构建/验证同步打包新运行文件，绑定 token 类型索引指纹，记录参数量、辅助类别名称、特征版本/哈希。正常加载拒绝缺失的分支、上下文或辅助权重。

## 独立 ms-swift 交付包

`external/ms_swift/cc_llm_multitask/` 内含插件、缓存准备器、公共参数与完整 Phase1/Phase2 脚本、README。整体移动到 ms-swift 后无需导入 CC-LLM checkout。没有修改本地 ms-swift 仓库已有的未提交脚本。

核对 ms-swift `4.7.0.dev0`、审查起点 HEAD `1516d78d627381c2072be80037a7a64c3811480d`，并读取用户提供的 GitHub `semantic_alignment.sh` / `full_cpt.sh`。工作期间外部仓库 HEAD 更新为 `ce28e0288657d6472a439ea219f1bcc77310e764`，已确认两者 `swift/` 无差异，没有撤销用户的同步。新脚本按新方案的学习率日程实施，不沿用旧脚本的统一 `2e-5`。

关键行为：

- 自定义 Template 显式请求辅助 logits，BaseLoss 负责 shift/查表/组合损失，不依赖模型收到 labels。
- 梯度累积沿用 Trainer 的有效 token 分母，world size 只由 Swift 乘一次。
- A/B 分别消费 100M/剩余目标，输入上下文可以在边界复用，但监督目标不重复。补齐 DDP 行时使用有标记的全忽略标签样本，不重复真实目标。
- 验证按样本聚合后经 `gather_for_metrics` 去掉 DDP 尾部重复样本，再累计 NLL/目标数。选优看 `eval_clm_loss`，不能用联合 loss 算 PPL。
- 累计辅助训练 tokens 保存在模型 buffer 与 checkpoint 的 `cc_llm_state.json`。A 取最后 checkpoint 进入 B；Phase1/B 按 CLM 选优。恢复同阶段与新建下一阶段的优化器明确分开。
- 默认按 Phase1 1B、Phase2 10B 有效目标检查预算，不足则报错，不静默重复数据填充；明确设置 `PHASE1_TOKENS=0` 可接受现有 Phase1 语料跑一遍并报告实数。

## 定位并修正的工程问题

1. **自定义 RoPE 的 meta 加载**：仅在 `__init__` 注册非持久 `inv_freq`，HF 从 Char 权重初始化新分支时不会替自定义模块正确恢复该 buffer。实测加载后频率出现 `7.65e6` 等未初始化值，预期首项为 `1.0`，可导致 NaN。改为在设备上以 FP32 确定性计算这个很小的常量，并通过真实构建、重载及 Phase2-A 回归。辅助 token 计数 buffer 同样在新构建时显式清零，不能依赖 meta 构造中的初值。
2. **动态缓存 reset**：上游 `DynamicLayer.reset()` 清零存储但保留长度/批次形状。组合缓存重建懒初始化层，避免清理后把零 KV 当作旧前缀，也支持新 batch 尺寸。
3. **无效监督与 slot**：切分后过滤零有效目标行；模型纯 CLM loss 也使用 sum/非零分母保护。拼音特殊 ID 不进入有效候选，字形三个属性均未知时 slot 无效；PGCA 即使存在输出 bias，全 mask 位置也严格为零。
4. **Swift 缓存契约**：编码缓存需保留 `lengths`，训练使用 `truncation_strategy=split`；否则当前 Swift 会重新按 messages 编码，导致 `KeyError: messages`。未用模仿私有格式的缓存代替官方导出。
5. **验证统计**：直接 all-reduce 已重复尾部样本的指标会产生偏差。真实双进程测试验证了 3 条验证样本仍只统计 8 个目标，而非包含补齐的重复目标。

这些是本次新增实现的回归发现，不应混同于旧训练中全部 eval_loss=NaN 的最终根因确认；旧真实模型/验证数据仍未在本机重现。

## 验证范围与结果

隔离环境：Python 3.11.15、PyTorch 2.10.0+cpu、Transformers 5.12.1，本地 ms-swift 4.7.0.dev0。依赖、微型模型与训练输出均在 `/tmp/cc-llm-context-tests.xDh4ZU/`。

相关回归：**64 passed、6 subtests passed**。包括词表/索引回归、旧 ungated 模型、真实 tiny Char→上下文模型构建和 AutoClass 重载、因果/候选重排、旁路/零初始化/梯度、BF16 前向、KV/beam/reset，以及：

- 单大 batch 与梯度累积梯度一致。
- 两进程 CPU/Gloo DDP 与单大 batch 梯度一致。
- 真正运行 `swift export`，再读取和规范化缓存。
- 真正运行 `swift pt` 的 Phase1、冻结主干且启用 gradient checkpointing 的 Phase2-A、Phase2-B，完成验证、保存和 CLM 选优。
- 两进程真实 Swift 训练、验证、保存与尾部重复去除。

仅有上游 `torch.jit.script` 弃用告警。最后减少 RMSNorm 重复 FP32 副本后，另跑 12 项上下文专项测试全部通过。`git diff --check` 与三个 Bash 脚本语法检查通过；使用 `-B` / `PYTHONDONTWRITEBYTECODE=1`，结束后源码、测试和交付包无 `__pycache__`。

未运行正式 1.7B 构建、CUDA/NCCL、多卡真实数据训练、10B 缓存导出、benchmark 或消融矩阵。服务器操作和所需动作清单见交付包 README；先完成 GPU 小样本冒烟，再跑正式训练。
