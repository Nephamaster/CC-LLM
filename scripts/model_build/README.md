# 模型构建

完整入口：

```bash
bash scripts/model_build/run.sh
```

默认读取 `models/base` 与 `data/model_build`，依次调用 `build_vocab.sh`、`build_embedding.sh`、`build_pgca.sh`。隐藏维度从 Base 的 `config.json` 读取，PGCA 默认选择模型中间三分之一层，不绑定参数规模。

已有输出不会被覆盖；重新构建时指定新的 `CHAR_MODEL` 和 `PGCA_MODEL`。
