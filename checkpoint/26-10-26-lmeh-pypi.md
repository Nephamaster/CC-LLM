# lmeh 改为普通 PyPI 安装

按用户要求取消 lmeh 环境版本与源码校验：requirements 使用普通包名，删除 Git commit 绑定、
包版本约束、harness_identity 及150文件源码匹配表，不增加版本兼容分支。
运行信息仍记录实际包版本，但不据此阻断启动。

原 protocol.lock.json 删除；其中评估数据和 SQuAD 评分脚本 revision 原样移到 dataset_revisions.json，
这些值只选择既定数据，不校验环境。few-shot、任务与评分流程未调整。

验证：在隔离测试环境普通安装 PyPI lm_eval[hf]（解析为0.4.13），
tests/test_evaluation_lmeh.py 的16项全部通过。涵盖任务加载、分组汇总、小模型HF/PGCA似然与生成。
测试数据使用离线fixture；未在服务器执行全量GPU benchmark。

删除的源码锁表可从 Git 历史恢复；未修改服务器环境，未提交代码。
