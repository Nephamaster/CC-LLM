# Ancient finalize：环境版本只记录，不作为运行门槛

2026-10-19，服务器正常安装的 ms-swift 位于 site-packages，源码指纹与历史 Git snapshot 不同，导致 finalize 在实际编码前被阻断。用户明确要求撤销这类限制，而不是更新白名单或重新锁定另一份源码。

- 已删除 ms-swift 源码扫描、Git/commit/dirty 校验、模板源码哈希及固定 swift_commit 配置。运行时记录已加载包和依赖版本，元数据缺失记录 null，不因此报错。
- finalize 不再要求 prepared 的整个实现源码指纹等于当前代码，防止这次移除环境校验反而要求重跑 repartition。旧 swift_commit 字段从配置语义比较中忽略，旧 manifest 不修改。
- 保留真正的数据约束：正式非诊断数据、v2 篇章划分策略、数据配置、原始输入/排除集的哈希及大小、SQLite 哈希、实际 labels 和序列长度。v1 未重分区数据仍不能直接 finalize。repartition 的候选过滤复用检查不在本次移除范围内。
- 替换旧“必须拒绝不同 Swift 源码”的测试；11 项轻量标准库测试通过，覆盖任意已安装版本、元数据缺失、无 Git/源码扫描、旧 prepared 接续、实际编码标签异常与数据变化拒绝。未跑全套、未重扫语料、未执行服务器全量 finalize。
- 同步整个 scripts/data_factory/ancient_sft 后直接执行原 finalize 命令。该错误发生在 final 目录创建之前，不需要清理输出或重跑 repartition。

本记录覆盖先前 checkpoint 中关于固定 Swift commit/源码指纹的旧要求，后续不要恢复此类环境准入。
