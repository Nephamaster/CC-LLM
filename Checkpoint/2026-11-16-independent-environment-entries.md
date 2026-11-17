# 独立训练、评估环境安装入口

基线：`master / 696a935`，开始时工作区干净。用户已删除服务器上错误的评估环境，明确要求仅保留 train 和 evaluation 两个独立脚本，不要 run 门面或 common。

## 改动

- 删除 `scripts/environment/run.sh`，仅保留 `train.sh`、`evaluation.sh`；没有 common，也不互相调用。每个入口只检查、创建、安装和验证对应的 venv。
- 训练自动选渠道的行为保持不变；评估在驱动报告 CUDA >=13.0 时优先选 cu130。两个脚本均接受 TORCH_CHANNEL 覆盖，新增 cu130 支持，无具体包版本锁或哈希校验。
- 背景：服务器 Torch 2.11.0+cu128 与 vLLM 0.26.0 的默认 CUDA 13 构建不配套，导入缺 libcudart.so.13。已核对 vLLM v0.26.0 的 requirements/cuda.txt 要求 Torch 2.11.0，Dockerfile 默认 CUDA 13.0.2；南京 cu130 源包含对应 Python 3.12 的 Torch wheel。新入口在该服务器的驱动 590 / CUDA 能力 13.1 下选择 cu130，不动训练环境。
- 保留从 CUDA 源选择候选、普通源预解析、冲突回退、显式 wheel 联合安装及 Torch 最后收尾的流程。评估独有的 lm_eval/vLLM 导入仅存在评估脚本中。
- 根 README、pipeline 的环境命令与模型构建/训练/评估/冒烟的缺环境报错提示均更新为实际入口。未修改它们的执行逻辑、requirements 或模型代码。

执行：`TORCH_CHANNEL=cu130 bash scripts/environment/evaluation.sh`。已有训练环境不需要重装。

## 验证与边界

临时脚本与日志在 `/tmp/cc-env-split-20261116/`。

- 真实 pip、临时纯 Python wheel、本地 HTTP 源：cu130 候选冲突回退、在线新建、已有错误 Torch 替换、离线解析、全候选无解、无目标渠道 wheel 与临时文件清理均通过。HTTP 请求检查验证 CUDA 源只提供 Torch，普通依赖来自普通源。
- 隔离执行两个完整入口，使用解释器/GPU/安装命令替身：CUDA 13.1 默认 evaluation=cu130、train=cu128；CUDA 12.8 默认 evaluation=cu128；两入口显式覆盖均通过。故意放置不完整的另一环境，当前入口仍能运行，且未触碰另一环境目录。
- 两脚本 Shell 语法、受影响 Python 文件 AST、diff 空白、LF 换行与活动代码/文档中旧入口引用检查通过。
- 实际 pip 测试使用本机 Python 3.14，临时测试绕过 Python 3.12 检查；模拟包不含真实 CUDA 扩展。这不能证明服务器 vLLM 导入、GPU 推理或 CUDA JIT 编译通过，仍需在服务器执行新入口及模型冒烟。

本记录取代此前环境记录中的统一 run 入口方案，保留旧记录作为历史证据。没有修改用户服务器的目录，也没有执行提交或推送。
