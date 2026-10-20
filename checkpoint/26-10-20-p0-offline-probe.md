# P0 离线诊断边界与复现协议

## 已确认的决策

- 第一轮 Phase2-B 继续运行，不修改训练脚本、插件、模型源码或权重。
- P0 固定 **checkpoint-11500**，替代方案原先指定的 10000；不使用会变化的阶段指针。
- 入口 `scripts/diagnostics/run_pgca_probe.sh`，复用 ms-swift Python 环境但不启动 Swift/DDP；实际加载 checkpoint 自带运行代码。
- 100 条多音字候选由 Agent 编写、用户审核；初稿全部待审核，不计入正式 Top-1。候选表与协议见 `scripts/diagnostics/README.md`。

## 避免再次混淆的口径

- 辅助读音准确率是下一字合法集合命中率，多音字消歧是当前输入字的上下文候选选择，不可互相替代。
- 512 条取自本轮编码后的 `phase2_4096/val`，不重编码；保持 labels。按有效 token 加权的 On/Off 差异使用配对序列 bootstrap，不能以两个 batch 均值作比较。
- hook 现有 `pgca_attn.dropout` 可获得实际 slot 权重，无需改 Attention 源码。只统计有特征的 token，另外报告双模态齐全子集及候选数量分组。
- 固定训练行默认 `train_b[0]`；三任务梯度沿用当前 CLM token 分母、字形三项平均，未乘辅助权重，无 optimizer step。零范数对应余弦 null；没有辅助目标不代表梯度断路，需要换有监督的显式行号。
- `partial` 不伪装成 P0 全部通过；未审核、候选不覆盖、辅助监督不足逐项标明。实际结论必须等待服务器真实报告。

## 验证与剩余工作

- 本地 tiny contextual PGCA 的完整保存/加载、缓存输入、On/Off、hook、三任务梯度、重复运行、结果保护通过；相关 PGCA 回归共 32 项通过，shell 语法检查通过。
- 没有运行真实 1.7B 权重或 GPU；未判断 PGCA 收益、未实施 P1/P2。
- 用户先保留 checkpoint-11500 完整副本以免被训练轮换删除、审核 TSV；有空闲 GPU 后再运行，回传 summary / samples / gradient / manifest。
- 第二轮整体 experiment manifest 需要等诊断结果、P1/P2 选择、新初始模型固定后再生成，不能用现在的未定参数代填。

## 服务器源码副本无 Git 元数据

- 实际部署目录未包含 `.git`，原 `git_state()` 无条件执行 `git rev-parse HEAD`，在模型加载后因退出码 128 中断；这是诊断工具把可选追溯信息误设为硬依赖，不是权重或缓存错误。
- Git 查询改为可选：捕获命令失败、Git 不存在及超时，manifest 明确记录不可用和原因，Git 字段为 null，不虚构 commit 或干净工作区。
- 补测无 Git 程序/查询失败/超时，以及真实无 `.git` 临时目录中的完整小模型诊断。该次报错发生在输出目录创建之前，同步修复文件后可直接重跑原命令，无需初始化 Git、重建缓存或重训。
- 用户随后明确要求不再对代码文件计算/校验哈希：已移除诊断脚本、checkpoint 配套源码和 Git diff 的哈希；仅保留模型/配置、tokenizer、数据和模型状态指纹。不得重新引入源码哈希作为运行前提。
