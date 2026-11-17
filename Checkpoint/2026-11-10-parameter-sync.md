# 参数一览同步

基线：`master / 341860b`。依据用户编辑的 `experiment.xlsx`「实验参数一览」同步正式训练和评估默认值，保留冒烟的独立小参数。用户已有的 central-smoke 文档修改保持原样，未提交或切换分支。

## 已落实

| 阶段 | 每卡 batch | 梯度累积 | 编码进程 | 默认 8 卡有效 batch |
| --- | --- | --- | --- | --- |
| Char / PGCA Phase1 | 8 | 4 | 32 | 256 |
| Char CPT | 8 | 4 | 32 | 256 |
| PGCA-A | 8 | 1 | 32 | 64 |
| PGCA-B | 16 | 2 | 32 | 256 |

- LMEH 和本地下游评估默认 batch 32、vLLM TP 4；Python 直接入口与 shell 默认一致。vLLM max_num_seqs 继续随下游 batch 绑定，为 32。
- 用户确认：PGCA-B 保留有效 batch 256，将表中每卡 64 改为 16；下游 batch/max_num_seqs 统一为 32。只回写这两个单元格，其余工作簿内容及格式不变。
- A/B 默认 prepared 目录分别为 `phase2_<长度>_adaptation` 与 `phase2_<长度>_joint`，复用同一个既有 `phase2_<长度>.raw`。显式 CACHE_DIR 仍默认搭配其自身的 .raw，显式 RAW_CACHE 优先。保持原有缓存几何校验，不添加通用的 batch/卡数变更自动修复。
- 训练继续支持 CUDA_VISIBLE_DEVICES 索引列表，默认按列表长度推导 NPROC_PER_NODE。表中 8 卡不是硬编码进程数；换卡数会改变有效 batch。
- TP 4 只对 vLLM 生效，不会为 PGCA 的 HF 后端启用四卡张量并行。单卡 vLLM 需要显式 TP 1。冒烟仍使用每卡 batch 1、TP 1。

## 验证与限制

- 工作簿与脚本默认值、有效 batch 计算对照通过；原工作簿格式及其他数值保留。
- 原生 shell 配临时 Python 替身，验证 GPU `1,3` 转发、自动推导两进程、A/B 共用一次原始导出而分开准备、显式缓存参数优先、评估 32/4 默认和冒烟 1/1 覆盖。
- 使用真实小型编码数据实际生成两份 8 卡 batch 8/16 的 prepared 视图，核对补齐几何、A/B 监督范围及有效目标总数一致。
- Shell 语法、Python AST 与 diff 空白检查通过；未执行 GPU 训练或推理，新 batch 的显存和吞吐尚未实卡验证。冒烟 batch 1 不能证明正式 batch 的显存可用性。
- 临时脚本、检查数据与工作簿备份：`/tmp/cc-parameter-sync.7MVkDI/`；未添加仓库测试目录或运行时 Excel 依赖。
