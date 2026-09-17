# Phase2来源容量与配比调整

2026-09-17，用户授权根据实际容量自行重新分配来源，保持10B总量、五个一级类别比例及自然语料、来源/长文/古文约束。

- 知识桶（2.5B）：Cosmopedia49.97%、Wikipedia15%、FineWeb-zhtw20%、Wikisource15%、ect-krp0.03%。
- 增强桶（1B）：CCI3-HQ20%、FineWeb中文20%、WanJuan15%、Cosmopedia9.95%、Wikipedia5%、FineWeb-zhtw15%、Wikisource15%、ect-krp0.05%。
- ect-krp约4.94M缓存容量，原175M正式配额降为1.25M；知识0.75M，增强0.50M，预留解压/筛选损耗余量。
- Wikipedia原725M正式配额降为425M；知识375M，增强50M。根据服务器报错反推知识产率约0.5796，减少配额后避免逼近1.262B缓存容量。
- 相比旧配比，Wikisource增加175M，FineWeb-zhtw增加50M，Cosmopedia增加248.75M，分别承接古文、繁体与知识数据；其他三个一级桶不变。

原Plan计算文件需求时没有按抽样率公式补普通桶的增强预留，造成FineWeb-zhtw有大量缓存仍要求抽样率101.42%。文件预算与抽样率现在均使用`(普通来源目标+增强预留)*oversample/rate`，增强单独使用`增强目标*enhancement_oversample/rate`；同一来源取最大扫描需求，不将各类别完整扫描量相加。

验证：V2测试36项通过、1项因本地缺服务器Char Tokenizer跳过。新增预算回归复现旧公式在窄普通产率时出现超过100%抽样率的错误；新增容量压力场景采用已有容量、Wikipedia反推产率及其他知识来源的保守场景产率，Plan passed=true且所有抽样率不超过1。该场景不是服务器最新Calibration结果；服务器仍须重新校准确认实际产率。

配比改动生成新的Phase2 Run ID与Calibration路径，从calibrate→plan→candidate继续，不需要重建Cache或Manifest；Phase1 YAML未修改。保留旧Run，首次执行新配置无需overwrite。本次未改benchmarks.required，也未触碰V1。
