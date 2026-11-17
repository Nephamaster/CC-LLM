# 从目标 CUDA 渠道选择 Torch，再解析其余依赖

基线：`master / 3a67ef4`，开始时工作区干净。本轮仅修改 `scripts/environment/run.sh` 与根 README，未修改 requirements、训练或评估代码，未提交。

## 问题与修正

服务器先从普通源选出 Torch 2.15.0，随后尝试下载不存在的 2.15.0+cu128。只把 Torch 放到最后并不能解决版本选择错误。本方案取代 `2026-11-12-environment-torch-last.md` 中“先安装 requirements 再替换同版本 CUDA 构建”的流程。

- 用 pip 查询目标 CUDA 渠道中与当前 Python/平台兼容的二进制候选，只接受指定渠道，从新到旧尝试。离线从 WHEELHOUSE 查询。
- 只下载候选 Torch wheel，再让 pip 从普通源用 requirements 和该 wheel 一起进行 dry-run；此时不安装应用包。
- 依赖冲突时尝试下一候选；其他解析错误直接报告，下载/索引失败直接退出。全部候选不兼容也在实际安装前退出。
- 找到可解组合后，安装 requirements 时继续明确指定该 wheel，最后再单独安装同一个 wheel 收尾。pip 可在联合安装阶段提前安装 Torch，收尾不强制重装。
- CUDA 源仅用于 Torch，其余在线依赖继续用清华源。无静态版本基线、哈希校验、约束文件或持久依赖锁，不增加全局 upgrade。
- 保留原有 Python 3.12、依赖、导入及 CUDA 检查；没有扩展普通依赖的升级策略。

## 验证

一次性脚本和日志：`/tmp/cc-cuda-resolution-20261116/`。

使用真实 pip、隔离 venv、临时纯 Python wheel 与本地 HTTP 包索引执行当前安装函数，检查通过：

1. 较新 CUDA Torch 与 vLLM 声明冲突，回退到较早候选后正常安装。
2. 新环境在线解析、安装完成；HTTP 请求确认普通依赖未访问 CUDA 源、Torch 未访问普通源。
3. 已安装错误渠道/较新 Torch 时，仍选择指定 CUDA 候选并替换。
4. 离线完成同样的候选回退和安装。
5. 全部候选冲突、没有目标渠道 wheel 时失败，未安装应用包。
6. 临时 wheel 目录在成功/失败后均清理；Shell 语法、diff 空白检查通过。

这些测试使用本机 Python 3.14 和模拟包，仅在临时测试中绕过解释器 3.12 检查；没有修改正式要求。不代表真实 Python 3.12 的 Torch/vLLM CUDA 二进制、镜像完整性或 GPU 模型推理通过，服务器仍需实际执行安装与冒烟。索引查询文本读取使用 pip 的 `Available versions:` 输出格式。
