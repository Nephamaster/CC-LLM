# CPT 修改 batch / 卡数后自动重建训练视图

基线：master，981b38f；本次修改未提交。用户反馈 Phase1 更换 batch 后，在 prepare_data.py 的 --verify 分支因 distributed batch geometry mismatch 直接退出，要求自动处理不兼容的缓存。

原因：prepared 训练集按 world_size × batch_size 补齐零监督 dummy，参数变化后不能直接使用旧补齐结果；原始 .raw 编码无需重做。

修改 scripts/train/cpt/prepare_data.py：在既有准备锁内校验已完成缓存；卡数或每卡 batch 变化时，复用原始编码，在临时目录重新准备视图，成功后替换并清理旧视图，失败时保留旧缓存。配置相同时继续复用。现有四个调用脚本不必修改：Phase1、Phase2-A、Phase2-B、Char CPT 都经由 --verify 使用该行为。长度、种子、Token 预算不匹配继续拒绝复用。同步更新根 README 与 CPT README。

验证：临时脚本 /tmp/cc-cache-batch-20261117/check.py，日志同目录 check.log。4 项测试通过，覆盖无 A/B 划分的 CLI 重建、有 A/B 区间的多次 batch/卡数变更、匹配后不重建、其他配置不匹配仍拒绝、准备失败保留原视图。检查真实样本及监督区间不变、dummy 全部无监督、各训练 split 行数满足新的补齐要求、原始编码文件的大小/修改时间不变、成功后没有遗留旧视图。git diff --check 通过。

使用临时 CPU Python 3.14.6、datasets 4.8.4 环境验证小型真实 Arrow 数据集；未在正式 Python 3.12、服务器全量数据或 GPU 正式训练上执行。未改用户结果 JSON 或其他已有未提交记录。用户同步修改后，直接重跑原训练命令，无需手动删除缓存。
