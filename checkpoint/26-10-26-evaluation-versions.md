# 评估输出自动分版本

用户要求重复执行相同命令自动创建 v0/v1，不因旧结果或失败目录存在而阻断。
统一分配实现为 scripts/evaluation/runs.py：扫描最大序号后原子 mkdir，竞争失败递增重试。
旧文件不搬迁、不覆盖；失败运行也占一个版本。

- lmeh：每个 benchmark 的模型/RUN_NAME 父目录下自动创建 vN。
- PPL：output=parent/metrics.json 实际生成 parent/vN/metrics.json，结果 output 字段返回真实路径。
- CSC/CGEC/古汉语：all/prepare 创建新版本；infer/score/judge 选最大序号版本；RUN_DIR 显式接续。
- ACT baseline/followup/P1 抽查也使用 vN，取消目录必须不存在的启动限制。
- 底层 prepare/infer 的同一运行身份检查保留，避免接续时混入不同模型或参数；新运行不会遇到它们。

已验证并发16次分配互不冲突、保留旧式结果/失败版本、分阶段接续/指定RUN_DIR、重复真实小模型评估。
联合检查86 passed，另2个既有CPT测试仍断言weight_decay=0.0，而当前训练脚本为0.01；本次不改训练参数。
未在服务器跑全量GPU评估；用户temp.sh修改保留。
