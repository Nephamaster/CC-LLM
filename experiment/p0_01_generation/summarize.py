"""Summarize observed differences without asserting an untested root cause."""

import argparse
import csv
import json
from pathlib import Path

from .records import write_json


def read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def config_fields(record):
    """Comparable fields only; full framework-specific configurations remain in raw records."""
    updates = record.get("engine_sampling_updates", [])
    if updates:
        config = updates[-1]["effective_sampling"]
        temperature = config.get("temperature")
        fields = {"max_tokens": config.get("max_tokens"), "do_sample": None if temperature is None else temperature != 0,
                "top_p": config.get("top_p"), "top_k": config.get("top_k"),
                "repetition_penalty": config.get("repetition_penalty"),
                "stop_token_ids": config.get("all_stop_token_ids", config.get("stop_token_ids"))}
    else:
        config = record.get("effective_generation_config") or {}
        fields = {"max_tokens": config.get("max_new_tokens"), "do_sample": config.get("do_sample"),
                  "top_p": config.get("top_p"), "top_k": config.get("top_k"),
                  "repetition_penalty": config.get("repetition_penalty"),
                  "stop_token_ids": config.get("eos_token_id")}
    ids = fields["stop_token_ids"]
    if ids is not None:
        fields["stop_token_ids"] = sorted(set([ids] if isinstance(ids, int) else ids))
    return fields


def summarize(run):
    run = Path(run)
    metadata = json.loads((run / "experiment.json").read_text())
    case_ids = [c["id"] for c in read(run / "inputs/cases.jsonl")]
    cells, rows, comparisons = [], [], []
    lookup = {}
    for alias in ("base", "shared"):
        for group in ("A", "B", "C", "D"):
            versions = [p for p in (run / alias / group).glob("v*") if p.name[1:].isdigit()]
            if not versions:
                cells.append({"model": alias, "group": group, "state": "not_run"})
                continue
            latest = max(versions, key=lambda p: int(p.name[1:]))
            state = json.loads((latest / "status.json").read_text()) if (latest / "status.json").exists() else {"state": "interrupted"}
            records = read(latest / "generations.jsonl")
            cells.append({"model": alias, "group": group, **state, "output": str(latest.relative_to(run)),
                          "completed_cases": sum(bool(r.get("completed")) for r in records),
                          "expected_cases": len(case_ids)})
            for record in records:
                lookup[(alias, group, record["case_id"])] = record
                rows.append({"model": alias, "group": group, "case": record["case_id"],
                             "completed": record.get("completed", False),
                             "prompt_ids_match": record.get("prompt_ids_match", False),
                             "generated_length": record.get("generated_length"),
                             "repetition": record.get("repetition"), "hit_budget": record.get("hit_budget"),
                             "end_token_count": len(record.get("end_token_positions", [])),
                             "finish_reason": record.get("framework_finish_reason") or ",".join(record.get("observed_finish_reason", [])),
                             "effective_config_observed": record.get("effective_config_observed", False),
                             "output": str((latest / "generations.jsonl").relative_to(run))})
        for case in case_ids:
            for left, right in (("A", "B"), ("B", "C"), ("C", "D")):
                x, y = lookup.get((alias, left, case)), lookup.get((alias, right, case))
                if not x or not y or not x.get("completed") or not y.get("completed"):
                    continue
                a, b = x["generated_ids"], y["generated_ids"]
                position = next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), min(len(a), len(b)))
                first_difference = None if a == b else {
                    "generated_position": position,
                    "token_ids": [a[position] if position < len(a) else None,
                                  b[position] if position < len(b) else None]}
                xc, yc = config_fields(x), config_fields(y)
                comparisons.append({"model": alias, "case": case, "groups": [left, right],
                                    "same_prompt_ids": x["actual_prompt_ids"] == y["actual_prompt_ids"],
                                    "same_generated_ids": x["generated_ids"] == y["generated_ids"],
                                    "repetition": [x["repetition"], y["repetition"]],
                                    "lengths": [x["generated_length"], y["generated_length"]],
                                    "first_difference": first_difference,
                                    "configuration_differences": {k: [xc.get(k), yc.get(k)] for k in xc if xc.get(k) != yc.get(k)}})
    ready = all(c["state"] == "complete" and c.get("completed_cases") == len(case_ids) for c in cells)
    instrumented = bool(rows) and all(r["effective_config_observed"] for r in rows)
    write_json(run / "summary.json", {"task": "P0-01", "all_cells_completed": ready,
                                       "effective_configs_captured": instrumented,
                                       "root_cause": "pending evidence review", "cells": cells,
                                       "comparisons": comparisons})
    fields = list(rows[0]) if rows else ["model", "group", "case", "completed"]
    with (run / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# P0-01 生成对照", "", f"可见 GPU：`{metadata['gpu']}`。每次单样本，最大生成量沿用历史任务。", "",
             f"全部组运行完成：{ready}。最终生成配置均已捕获：{instrumented}。", "",
             "根因结论待结合原始 Token、停止事件和配置差异审阅，运行完成不等于根因定位完成。", "",
             "| 模型 | 组 | 状态 | 完成样本 | 记录 |", "|---|---|---|---|---|"]
    for cell in cells:
        output = cell.get("output", "")
        link = f"[{output}]({output}/worker.log)" if output else ""
        lines.append(f"| {cell['model']} | {cell['group']} | {cell['state']} | {cell.get('completed_cases',0)}/{len(case_ids)} | {link} |")
    lines += ["", "## 逐样本观察", "", "| 模型 | 组 | 样本 | 生成 Token | 重复 | 达到预算 | 结束符数 | 结束依据 |",
              "|---|---|---|---:|---|---|---:|---|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['group']} | {row['case']} | {row['generated_length']} | {row['repetition']} | {row['hit_budget']} | {row['end_token_count']} | {row['finish_reason']} |")
    lines += ["", "A/B、B/C 必须同 Prompt IDs。C/D 改变普通对话输入，不能要求 Prompt 相同。",
              "达到预算与停止同时发生时，需查看原始停止事件，不仅凭生成长度判定不终止。",
              "完整输入、生成 IDs、原始文本、处理后文本与配置均在各组 generations.jsonl。"]
    lines += ["", "## 配对差异", "", "| 模型 | 样本 | 对照 | 输入相同 | 生成相同 | 首次分歧位置 | 配置差异字段 |",
              "|---|---|---|---|---|---|---|"]
    for pair in comparisons:
        first = pair["first_difference"]
        position = first["generated_position"] if first else ""
        fields = ", ".join(pair["configuration_differences"])
        lines.append(f"| {pair['model']} | {pair['case']} | {'/'.join(pair['groups'])} | {pair['same_prompt_ids']} | {pair['same_generated_ids']} | {position} | {fields} |")
    lines.append("\n配置差异只列观察值。Greedy 时 top_p/top_k 等参数可能不参与选择，不能将字段不同直接当作根因。")
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ready


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    summarize(parser.parse_args().run)
