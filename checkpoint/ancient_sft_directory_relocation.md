# 古汉语数据及源码目录规范化（2026-10-19）

- 用户已将模块移入 `scripts/data_factory/ancient_sft`，入口改为 `python -m scripts.data_factory.ancient_sft.run`。common.py 的项目根定位必须从 parents[2] 改为 parents[3]，否则相对数据/模型路径会错误指向 scripts 下。
- 原始数据统一到 `data/downstream/ancient/raw/{chinese-classical-corpus,HistoryTrans}`；Benchmark 仍在 `ancient/benchmark`；默认输出为 `ancient/train/{prepared,final}`。prepared 在服务器已有，本地为空不代表服务器需要重建。
- 配置、模块文档相对链接、测试导入、根 README、方案、Dashboard 和历史检查点中的操作命令已同步。数据版本标签仍是 ancient_sft_v2，不再用版本标签充当默认物理目录名。
- 迁移前 v2 实现指纹为 `c5a98e3708860e3ee4ddca5571fa5c61379c142726ad7fe0e39b2a733cfb2997`。新增有界兼容：验证准备逻辑未变（仅允许项目根与入口路径的等价替换），配置仅识别明确旧→新路径映射。库存仍逐文件比较角色、路径身份、SHA256、大小；数据库哈希检查保留。未知实现、参数变化、内容变化仍拒绝。
- 旧 manifest/metadata 的历史来源位置不原地重写，避免破坏审核 ID 和来源证据。新 final manifest 保存 effective_config/effective_inventory；repartition 的新 manifest 保存当前位置和父版本指纹。
- 目录迁移本身不需要重新分组；已完成 v2 prepare 可接续 finalize，但已有 final 不覆盖。确需 repartition 时必须指定另一个空输出目录，例如 `ancient/train/rebuild`，不可从 `ancient/train/prepared` 读入又写回同一目录。
- 按用户要求只做轻量验证：7 项标准库迁移测试约 0.1 秒通过，覆盖新入口/根路径、配置与库存映射、旧 prepared 接续、内容变动拒绝、准备逻辑变更拒绝、repartition 路径适配；没有重扫原始数据、运行全套测试、加载模型或访问服务器。
