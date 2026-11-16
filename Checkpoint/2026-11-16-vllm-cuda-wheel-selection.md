# 评估环境同时选择 vLLM CUDA 构建与 Torch

基线：`master / 94ea5d1`，开始时工作区干净。仅修改独立的 evaluation.sh 和根 README，train.sh、requirements、模型及评测代码未修改，无提交/推送。

## 原因与当前实现

服务器此前安装 vLLM 0.26.0 和 Torch 2.11.0+cu128，依赖声明可解，但 vLLM 扩展需要 libcudart.so.13。仅选择 Torch CUDA 渠道不足以选择配套的 vLLM 二进制。

- 从配置的普通包源枚举适合当前 Python/平台、非预发布且未撤回的 vLLM wheel。使用版本的 +cuXXX/+cpu 标签，或包内 envs.py 的 VLLM_MAIN_CUDA_VERSION 默认构建声明确认 CUDA 渠道；不导入/执行候选包代码。
- 未识别的默认构建跳过，不凭 Torch 发布版本或“依赖可解”推定其 CUDA。没有候选时在安装应用包前明确失败。
- 读取 wheel 的 METADATA 中实际 Torch 依赖与环境标记，再从指定 CUDA 源寻找满足该要求的 Torch；按需配套 torchvision、torchaudio，避免其他 Torch 扩展仍混用渠道。
- 候选较新优先。匹配构建后，下载明确选中的 wheel 并与 requirements 一起 dry-run；冲突时尝试其他组合。成功后在实际安装阶段继续传入这些 wheel，最后使用同一 Torch wheel 收尾。
- 在线通过 HTTP Range 读取 ZIP 元数据，源忽略 Range 时暂存该 wheel 后读取；离线直接读取 WHEELHOUSE。只访问配置的包源及其 wheel 链接，不调用 Git，不查询 GitHub 发布或编译源码。
- 筛选代码嵌在 evaluation.sh 中，无新增 common/门面/独立 Python 模块，无静态版本表、哈希校验或依赖锁。保留原有 pip check、导入和 CUDA 张量检查。

## 验证

临时脚本与日志：`/tmp/cc-vllm-cuda-20261116/`。

- 真实 pip、隔离 venv、纯 Python 模拟 wheel、本地 HTTP 源验证 cu130、cu128、带 +cu126 标签的构建选择。模拟 vLLM 在导入时检查所配 Torch CUDA，确保没有仅靠版本约束通过后继续混用。
- cu128 能跳过默认 CUDA 13 wheel；未知构建跳过；配套 Torch/torchaudio/torchvision 的渠道与 vLLM 一致。
- 离线读取、没有匹配 cu124 候选时不安装应用包、HTTP Range 与不支持 Range 的暂存读取、临时目录清理均通过。请求记录确认 Torch 系列不从普通源选择，其他包不访问 CUDA 源。
- 只读探针读取官方 vLLM 0.26.0 wheel：识别到 cu130，METADATA 要求 torch==2.11.0；使用 Range，没有下载完整 wheel。此探针读取官方 PyPI 仅用于开发核验，正式脚本仍使用配置的国内镜像。
- Shell 语法、嵌入 Python 的 AST/Ruff F/E9、diff 空白、LF 换行通过。测试初次因临时 select.py 遮蔽标准库模块失败，已重命名为 selection_fixture.py 后完成验证，仓库没有同名模块。

## 限制与使用

执行仍为 `bash scripts/environment/evaluation.sh`，也可显式设置 TORCH_CHANNEL。

不是所有历史版本或 CUDA 渠道都有适合当前 Python 的可识别 wheel。旧默认包缺少构建声明时会被跳过，可在 WHEELHOUSE 提供官方带 CUDA 标签的配套构建及依赖；不会自动退回未经确认的包。默认构建声明与标签属于包提供的信息，不能替代真实扩展导入和模型冒烟。

本机实际 pip 测试使用 Python 3.14，临时调用绕过解释器 3.12 检查，模拟包不含真实 CUDA 扩展。没有执行真实 Torch/vLLM 安装或服务器 GPU 推理，不声称所有旧驱动/软件组合均已验收。
