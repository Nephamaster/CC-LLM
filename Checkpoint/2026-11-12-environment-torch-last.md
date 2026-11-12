# 环境安装：普通依赖先装，Torch CUDA 构建最后处理

基线：`master / 5c576f9`，开始时工作区干净。本轮只调整环境安装流程及对应 README，未修改 requirements 或训练/评估代码，未提交。

## 原因与决定

服务器日志显示 Torch `2.11.0+cu128`、Transformers `5.22.2`、PEFT `0.6.2`、ms-swift `1.4.0`。旧 Swift 要求 PEFT 不高于 0.6.2，与 Transformers 所需的 PeftMixedModel 接口不兼容；仅升级 PEFT 后虽能导入，又留下旧 Swift 的声明依赖冲突。尚未获得最初解析日志，不能认定 CUDA 镜像导致 Swift 被选旧。

用户明确要求撤销 CUDA 镜像作为全局优先源的设计，仅解决 Torch/CUDA 适配，并将 Torch 放在最后安装。本记录取代 `2026-11-10-readme-onboarding.md` 中关于 uv/CUDA 全局优先解析的当前方案；历史检查结果保留。

## 当前流程

1. 创建/复用 Python 3.12 的 train、evaluation 环境。
2. 正常使用 pip 从清华源安装各自 requirements；移除 uv 和整套依赖的全局 upgrade。
3. 读取刚安装的 Torch 发布版本，保留该版本，下载目标 CUDA/CPU 构建的 wheel。在线只向 CUDA 镜像请求 Torch wheel；离线只查 WHEELHOUSE。
4. 从本地 wheel 安装 Torch，正常解析其依赖；在线依赖仍走清华源。download 的 no-deps 仅用于限制下载对象，不用于跳过安装依赖。临时 wheel 自动清理。
5. 执行 pip check、原有关键模块导入、CUDA 渠道与张量检查。Torch 之后不再执行其他包安装。

发布版本来自当次 requirements 的实际解析，不是写死的版本基线，也不生成约束或锁文件。此举保留 vLLM 声明要求的 Torch 版本，但不能证明其预编译 CUDA 扩展必然兼容。目标 CUDA wheel 不存在时明确失败，不自动调整其他包。

## 核验与边界

- `bash -n scripts/environment/run.sh`、`git diff --check` 通过。
- `/tmp/cc-torch-last-20261112/check.py` 隔离执行真实脚本，用解释器/安装命令替身核对两套环境的在线和离线顺序、源分离、Torch 版本保留、失败退出与临时目录清理，均通过。
- 对脚本内实际 Python 完成检查使用控制模块执行，验证正确 CUDA、错误 CUDA、CPU 与误装 GPU 构建分支，均通过。
- 上述替身检查不是实际依赖安装或 GPU/vLLM 验收；本机未执行大包安装，服务器仍需重跑安装并检查真实结果。
- 已有旧 Swift 等包不会因为本轮重跑被全局升级。既有非 Torch 接口冲突需单独修复，本轮未扩大到普通包的升级或版本范围设计。
