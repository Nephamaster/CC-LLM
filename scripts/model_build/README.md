# 模型构建

完整入口：

```bash
bash scripts/model_build/run.sh
```

默认读取 `models/base`、`data/model_build/hanzi` 与 `data/model_build/unihan`，依次调用 `build_vocab.sh`、`build_embedding.sh`、`build_pgca.sh`。隐藏维度从 Base 的 `config.json` 读取，PGCA 默认选择模型中间三分之一层，不绑定参数规模。

允许使用已有的 `CHAR_MODEL` 和 `PGCA_MODEL` 目录；重新构建会覆盖同名构建产物。
