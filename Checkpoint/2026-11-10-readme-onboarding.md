# README 从零执行与环境安装补齐

基线：`master / 341860b` 加此前未提交的参数同步改动。用户要求补齐 README 的环境、数据、基座、三路训练评估、Phase1 PPL 和故障处理缺口；本轮保留已有工作区改动，未修改用户的 `pipeline.md` 或 `experiment.xlsx`，未提交/推送。

## 实施与决定

- README 改为顺序执行手册：解释器与两套环境 → 外层数据包/训练分片解压 → 统一 B/BASE/GPU 变量及 HF 镜像下载 → 正式构建 → 冒烟 → 三路训练 → 三路评估 → 初始 Char 与 PGCA-ALIGN PPL → 换规模。
- 保留并集中说明阶段输入变量、缓存/产物位置、重跑/跳过/resume 区别、训练与评估的继续命令、Fuxi pending，以及常见错误的处理路径。正式 batch、多卡与完整模型质量仍需真实验收。
- 环境脚本移除具体 Torch 版本映射、Torch 约束文件和精确版本相等检查。Python 3.12 要求保留；CUDA 渠道仍按驱动选择，不等于固定包版本。
- 在线由各 venv 内的 uv 使用 CUDA 索引优先、first-index 策略联合解析全部依赖；普通包源仍为清华，CUDA wheel 仍为南京镜像。只使用 pip install / uv pip install，不生成项目依赖锁、不调用 Git。
- 实际本地索引实验发现：只加 --upgrade 仍可能保留已安装的错误 CUDA 渠道 Torch；加入 --reinstall-package torch 后，正确选择目标索引的 Torch，并联合选择兼容的 vLLM。重跑安装因此会重新选择/安装 Torch，README 已明确此行为。
- 离线 WHEELHOUSE 继续使用 pip --no-index --find-links，不额外要求 uv。保留 pip check、模块导入及 CUDA 张量检查，不用固定版本号作为验收条件。

## 验证

证据、一次性脚本与本地模拟包均在 `/tmp/cc-readme-env.WdHpWS/`。

- README 26 个 Bash 代码块语法及相对链接检查通过。
- 对 4B/8B 两组变量实际展开并隔离执行三路训练、三路评估、两项补充 PPL 命令，检查模型名称、初始权重路径和 ALIGN 指针衔接；这些命令使用替身，不代表实际训练/评测通过。
- 使用小型真实 tar.xz 执行 README 原文解压命令两次，验证 JSONL 正确发布、原包保留和重复执行跳过行为。
- 实际 uv 0.12.10 与本地包索引验证联合解析：CUDA 索引提供的候选优先于通用源的更新 CUDA 构建；存在错误渠道已安装包时能够修复。使用的是模拟包，不是真正的 Torch/vLLM CUDA 二进制。
- 复制原生环境脚本到临时工程，隔离依赖/GPU 操作验证两套环境、驱动到 CUDA 源的映射及显式覆盖、在线/离线分支、不创建 Torch 约束、Python3.12 要求和验证命令调用。
- Shell 语法与 diff 空白检查通过。当前本地可用 Python 为 3.11，未做真实 Python3.12/GPU 新环境安装，也未执行完整 4B/8B 实验；不声明任意驱动与浮动依赖组合都兼容。
