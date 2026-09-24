# 显式接受现有Mixture

2026-09-23：用户决定使用Phase1 round1现有数据继续处理。报告总量984.146169M为估算Token，专项桶未达到原配额，不能通过篡改passed或修改Phase配置让它看似达标。

新增仅供tokenize/finalize使用的--accept-existing-mixture，不进入配置Hash，不变更Run/Plan。Tokenize可接受数量/来源/子类偏差，仍检查报告Run/Plan及增强和横向约束。Finalize核对Tokenization报告身份及Token总数；整篇预留验证文档，剩余全部输出训练集，不按原配额继续裁剪。

最终报告保留原配额passed和缺额，另记accepted_for_use、execution_passed和真实训练/验证Token数。执行成功不等于原实验配额达标。默认无该参数时保持原门禁与配额行为。

验证：5项定向测试通过，覆盖真实Parquet与临时Tokenizer导出、缺额/超额均保留全部Token、训练验证隔离、默认拒绝失败Mixture、跨Plan及其他约束拒绝、Token化产物不完整拒绝及既有Packing/Swift输出测试。服务器数据未在本机执行。
