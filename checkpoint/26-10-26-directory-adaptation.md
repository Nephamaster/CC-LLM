# 用户目录调整后的路径适配

## 范围

用户将 data_factory/v2 内容上移到 scripts/data_factory，将 train/ms_swift 内容上移到 scripts/train；
同时删除旧 smoke.sh/smoke_check.py/SMOKE.md 和 tests/v1。只适配这次目录变化，不改训练超参、
模型计算、数据选择或依赖约束，不恢复被删功能。

## 适配

- 数据工厂入口为 `python -m scripts.data_factory.run`，内部导入和测试导入同步上移；
  configs 位置没变，不调整 YAML 的 repo_root 或 run 身份。
- cpt/finetune shell 根目录向上层数从4改为3；数据导出脚本从3改为2。
  训练环境默认 scripts/train/env，评估 requirements 的相对引用同步。
- 诊断与 temp.sh 中只替换训练环境路径；文档中的当前入口同步。
- 旧 checkpoint 的已知训练插件路径缺失时映射到搬迁后的插件；不改历史 args.json。
- 删除仅测试已删 smoke 功能的 test_round2_smoke.py 和对应 layout 测试；可从 Git 历史恢复。
  v1 源码及既有 checkpoint 文档不批量修改。

## 验证

Python3.13 / torch2.10 CPU / Transformers5.12.1 / ms-swift ce28e028 / DataTrove0.10.1 隔离环境。
12个新路径数据工厂模块可导入，run --help、Phase1/2 repo_root、相关 shell 语法和差异检查通过。
联合检查103 passed、3 skipped、49 subtests passed；另3项既有失败保留：
两项 CPT 默认 weight_decay 断言仍为0.0（脚本0.01），一项 finetune 测试桩缺 metric_for_best_model。
新增路径与插件恢复等定向检查8 passed。迁移前后两份训练插件与数据工厂 config 内容比对一致。
未执行服务器GPU训练/评估，未搬动服务器虚拟环境、模型或数据；未暂存/提交。
