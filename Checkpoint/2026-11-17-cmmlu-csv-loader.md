# CMMLU 直接读取官方 CSV

基线：`master / 8c51af6`，开始时工作区干净。针对去污染收集阶段 `Dataset scripts are no longer supported, but found cmmlu.py`，只修改共享任务生成模块与对应 LMEH README，未修改环境依赖、其他任务协议或原始数据。

## 实现

- `protocol.build_bundle()` 通过 `hf_hub_download` 获取官方 `haonan-li/cmmlu` 的 `cmmlu_v1_0_1.zip`，默认 HF 镜像并复用库缓存，无 Git 调用、数据 revision 锁或自行计算哈希。
- CMMLU 模板使用 csv 数据入口；67 个叶子任务移除旧的 dataset_name，配置压缩包内 dev/{subject}.csv、test/{subject}.csv，仅保留 Question/A/B/C/D/Answer 六列。
- 叶子任务的 custom_dataset 指向本模块的 `cmmlu_dataset`，内部调用 datasets 内置 CSV 加载器，在解析前明确六列 string Features。该配置同时供正式评估和 instruction.decontamination 的共享 load_tasks/build_bundle 链路使用。
- 提示模板、学科描述、学科/组名称、Dev first_n 5-shot、Test 评估、acc/acc_norm 与汇总方式均保持原配置。

## 验证及发现

临时环境、脚本、官方 ZIP 缓存与日志均在 `/tmp/cc-cmmlu-csv-20261117/`，没有写入代码目录。

- 真实通过 HF 镜像下载官方 ZIP，无需 token。
- 在 datasets 5.2.0 和 4.8.4 两个临时测试基线上，读取全部 67 学科，得到 Dev 335 条、Test 11,582 条。逐行核对六字段、渲染提示和答案目标，与官方旧脚本的 pandas.read_csv(header=0, index_col=0) 加 Features 编码逻辑一致。
- 使用本机 lm-evaluation-harness checkout `0d3003be8` 的 CMMLU YAML 模板；4.8.4 测试还使用其原生 `_yaml_loader.load_yaml` 验证 include 合并与 custom_dataset 函数解析。六字段最终均为 string，各 Dev 仍为 5 条；ARC-Challenge 生成配置回归检查通过。
- 初版“先自动推断 CSV，再转 string”的方法在部分学科出现 ArrowInvalid：Dev 数值选项被推断为 double，Test 含文本，加载时即失败。已改为在 CSV 解析前传入 Features，不能在加载完成后才 cast。
- Ruff F/E9、Python AST、diff 空白检查通过。使用 python -B/PYTHONDONTWRITEBYTECODE，未产生项目 __pycache__。

本机测试解释器 Python 3.14，未运行用户服务器 Python 3.12 或完整 TaskManager/GPU 模型评估；真实数据读取及原生 YAML 配置解析已验证，不将其描述为正式分数验收。没有安装或降级项目的 train/evaluation 环境，所有测试依赖变动仅发生在上述 /tmp 环境。

## 接续

同步改动后重跑 `.venv/train/bin/python -B -m scripts.train.instruction.prepare`；若此前使用 HF_TOKEN 或 --evaluation-python，继续保留相同设置。不需要降级 datasets 或删除已有 HF 缓存。
