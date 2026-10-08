# 接受现有Phase2数据与古文缺额

2026-10-08，用户明确接受本次补采后约9.789B估算Token与整体古文6.63%的Mixture，准备继续数据导出及之后训练。

原--accept-existing-mixture仅接受数量/来源/子类偏差，仍拒绝整体古文低于8%。新增--accept-classical-shortfall，必须与前者同时传给tokenize和finalize。仅接受attribute/classical_chinese的最低比例不足，不接受最高比例超限、其他属性失败、增强约束失败或跨Run/Plan误用。

Tokenization与Finalization报告保存明确接受标记及原始古文检查快照。原Mixture报告和passed不修改，运行参数不进入配置Hash，保持当前Run/Plan并复用全部上游。最终真实规模以分词报告为准，训练验证按父文档隔离。

后续顺序：当前轮次tokenize → finalize → 检查accepted_for_use/实际Token与重叠 → export_ms_swift.sh生成训练框架缓存 → 使用相同Tokenizer及序列长度启动Phase2训练。本次未启动服务器任务或模型训练。

验证：7项定向测试通过，包括真实临时Parquet/Tokenizer的分词与导出、原配额失败仍保留、显式接受古文最低比例不足、默认拒绝、不允许绕过其他属性或古文上限、身份及完整性检查、既有Packing/Swift输出。另用用户本次Mixture报告验证默认模式拒绝6.63246%古文，而双开关接受并原样记录该检查；CLI帮助及git diff --check通过。
