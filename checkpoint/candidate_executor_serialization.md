# Candidate执行器序列化修复

2026-09-17：CandidateSelector新增的类别Token密度映射使用`(source, bucket)`作为字典键。DataTrove在启动任务之前将PipelineStep属性写入executor.json，JSON不支持tuple键，因此报`TypeError: keys must be str, int, float, bool or None, not tuple`。

映射改为`source -> bucket -> density`两层字符串键字典，保留有效类别密度及缺失/零密度时的来源均值回退。回归使用DataTrove实际执行器调用save_executor_as_json并读取结果，同时验证估算Token值和候选资格并集；两项定向测试通过。

服务器同步scripts/data_factory/v2/candidate.py后直接重新执行原Candidate命令，不需要重建Cache、Calibration或Plan，也不需要overwrite。此次错误发生在任务启动前，残留executor.json会在下次启动时重新写出。
