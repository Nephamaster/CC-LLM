# Ancient SFT v2：Dev 覆盖与标点监督预算

## 根因与决策

v1 全量 report 显示核心 Train 72K 达标，但 Dev 只有 197 条，全部来自《笑林广记》，没有标点。完整 metadata 哈希验证后确认：74 部作品通过跨书重复的传递连接被并成一个组，包含 Train 68,718 条（95.4%）。对少数超大组执行 2% 哈希划分无法保证任务覆盖，Dev 再受单书上限压缩。标点 24K 占样本 33.3%，却占 assistant loss token 71.9%。

用户批准改为篇章优先、分任务 Dev、标点 token 预算。版本标签为 `ancient_sft_v2`，不覆盖 v1。本文操作路径已按 2026-10-19 的统一布局更新；历史验证数字不变。

## 实现边界

- 卷/章/可追溯 corpus 记录优先，SQLite FTS5 trigram 召回并以完整规范化文本包含验证；无匹配再用来源篇章标签，最后才回退整书。不能定位或 HistoryTrans 作品归属歧义则隔离。检索先用 16 字精确锚点减少长串 FTS 位置连接开销，不放宽完整包含验证。
- 不再跨书传递合并整书。先选 Dev 候选组，Train 排除同篇章、多出处交集、同近重复组、同译文、双向完整包含以及共享 24 字片段。章名未对齐的 corpus 内来源不进入 Dev，避免迫使同书全部训练记录被排除；这些记录若和已留出书籍存在边界不确定性，仍从严隔离。
- Dev 三项各 600；每项至少 5 部作品、10 个独立组，单书比例不超过 20%。分组阶段预留约 1.5 倍候选容量，最终按作品/篇章/长度档/简繁类型交错采样并编码验收。
- Train c2m 38K、m2c 10K、标点初始 5K。标点实际监督 token 比例限制在 30%–40%；5K 后低于下限时最多补到 6K，超过上限的完整样本跳过，不截断。不能达到配额/比例则 ready_for_training=false；可选知识任务仍必须审核。
- `ready_for_training` 不再等价于 Train/Dev 非空，要求核心 Train 配额、Dev 各任务覆盖与集中度、标点预算全部通过。额外导出 `dev_by_task`；宏平均 Dev loss 需要训练端分别评估，数据管线不改 Trainer。
- 新增 `repartition --from-prepared`。只复用已知实现或当前实现的 hash-verified eligible 表，不读取 raw candidates 来恢复已排除数据，不改变旧 manifest。配置只能改变划分/采样字段，输入/Benchmark/审核文件/数据库、归一化版本、过滤规则均需一致。之前被过滤的数据不会恢复。

## 验证与限制

- 54 项定向测试通过，包括真实无 Git Swift 编码、Train/Dev 核心任务覆盖、同书不同章、跨书共享片段不合并整书、边界包含检查、Dev 低覆盖拒绝、标点预算/补采、确定性划分及安全重划分。PyTorch 有一条弃用警告。
- 另用当前 v2 实现完成各原始训练来源前 3,000 条的 fresh prepare 诊断，载入全部 Benchmark 与 HistoryTrans 保留分区；诊断产物位于临时目录，不允许正式 finalize。
- 用 hash-verified 的 v1 已选 Train/Dev 共 72,197 条 metadata 做独立诊断，未触碰正式产物。改进后 Dev 三任务各 600，分别覆盖 82/89/29 部作品，492/578/308 个主要篇章组；单书最高比例 3.83%/2.67%/6.83%。
- 此诊断复用旧记录的实际编码长度，不代表再次执行了全量 Swift 编码；候选只覆盖旧已选数据，不是服务器上的完整 eligible 池。诊断 Train 只有 28,702/7,190/4,570 条，标点预算控制在约 40%，数量未满足因此正确拒绝 ready_for_training。不能拿这个诊断结果宣称正式 v2 已通过。
- 本地没有服务器的完整 prepared SQLite，尚未运行全量 v2 repartition/finalize；服务器应使用 v1/prepared 的完整已准入候选，而不是只对旧 train.jsonl 重划分。

## 服务器执行

```bash
python -m pip install -r scripts/data_factory/ancient_sft/requirements.txt
python -m scripts.data_factory.ancient_sft.run repartition \
  --from-prepared data/downstream/ancient/train/prepared \
  --output-dir data/downstream/ancient/train/rebuild
python -m scripts.data_factory.ancient_sft.run finalize \
  --output-dir data/downstream/ancient/train/rebuild
```

默认输出 `data/downstream/ancient/train`。上面的重划分示例使用独立 rebuild 子目录，不能读写同一个 prepared。若没有完整旧 prepared 或输入/过滤规则已变，使用新目录执行 prepare → finalize。已有部分新阶段输出时换新 output-dir，不自动覆盖。
