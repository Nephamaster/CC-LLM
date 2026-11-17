"""Select by finite Dev loss plus generation acceptance, then merge and recheck."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.evaluation.common import write_json
from scripts.evaluation.runs import new_run_directory
from scripts.train.runtime import atomic_write


def check(model, adapter, dev, output):
    command = [
        sys.executable,
        "-B",
        "-m",
        "scripts.train.downstream.dev_check",
        "--model",
        str(model),
        "--task",
        "instruction",
        "--dev",
        str(dev),
        "--output",
        str(output),
        "--mode",
        os.getenv("DEV_CHECK_MODE", "required"),
    ]
    if adapter is not None:
        command += ["--adapter", str(adapter)]
    result = subprocess.run(command, check=False)
    reports = sorted(output.glob("v*/report.json"), key=lambda p: int(p.parent.name[1:]))
    report = json.loads(reports[-1].read_text()) if reports else {"passed": False, "error": "no report"}
    return result.returncode == 0, report, str(reports[-1]) if reports else None


def merge_model(base, adapter, destination, dev):
    from peft import PeftModel
    from swift.model import get_model_processor
    from scripts.evaluation.inference import TEMPLATE_KEYS
    from src.pgca.build_pgca import save_pretrained_without_auto_copy

    model, processor = get_model_processor(
        str(base), model_type="qwen3", device_map={"": "cpu"}, download_model=False
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).merge_and_unload()
    tokenizer = getattr(processor, "tokenizer", processor)
    end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(end) != 1:
        raise ValueError("im_end must be exactly one token")
    stop = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    model.generation_config.eos_token_id = list(dict.fromkeys(end + (stop if len(stop) == 1 else [])))
    model.generation_config.max_new_tokens = None
    model.generation_config.do_sample = False
    model.config.use_cache = True
    model.generation_config.use_cache = True
    if getattr(model.config, "model_type", None) == "qwen3_pgca":
        model.config.pgca_enabled = True
        save_pretrained_without_auto_copy(model, destination)
        for file in base.glob("*.py"):
            shutil.copy2(file, destination / file.name)
    else:
        model.save_pretrained(destination, safe_serialization=True)
    processor.save_pretrained(destination)
    model.generation_config.save_pretrained(destination)
    saved = json.loads((adapter.parent / "args.json").read_text())
    keys = (
        *TEMPLATE_KEYS,
        "model_type",
        "template",
        "loss_scale",
        "is_binary_loss_scale",
        "padding_side",
        "max_length",
        "truncation_strategy",
        "external_plugins",
    )
    args = {key: saved[key] for key in keys if key in saved}
    args["val_dataset"] = [str(dev.resolve())]
    write_json(destination / "args.json", args)


def finish(base, adapter_root, output, dev):
    pointer = adapter_root / "next_stage_model.txt"
    selected = Path(pointer.read_text().strip())
    run = selected.parent
    losses = {}
    actual_tokens = 0
    for state_path in run.glob("checkpoint-*/trainer_state.json"):
        state = json.loads(state_path.read_text())
        actual_tokens = max(
            actual_tokens,
            max((row.get("train_supervised_tokens", 0) for row in state["log_history"]), default=0),
        )
        losses.update(
            {
                int(row["step"]): float(row["eval_loss"])
                for row in state["log_history"]
                if row.get("eval_loss") is not None and math.isfinite(float(row["eval_loss"]))
            }
        )
    candidates = [
        (losses[int(path.name.split("-")[-1])], path)
        for path in run.glob("checkpoint-*")
        if path.is_dir() and path.name.split("-")[-1].isdigit() and int(path.name.split("-")[-1]) in losses
    ]
    destination = new_run_directory(output)
    report = {
        "complete": False,
        "base": str(base),
        "dev": str(dev),
        "candidates": [],
        "actual_train_supervised_tokens": actual_tokens,
        "mode": os.getenv("DEV_CHECK_MODE", "required"),
        "smoke_only": os.getenv("DEV_CHECK_MODE") == "report",
        "selection": "lowest finite Dev loss among saved checkpoints passing full generation/adapter/cache acceptance",
    }
    write_json(destination / "instruction_selection.json", report)
    qualified = None
    for loss, path in sorted(candidates, key=lambda item: (item[0], int(item[1].name.split("-")[-1]))):
        passed, metrics, location = check(base, path, dev, destination / "candidate_checks" / path.name)
        report["candidates"].append(
            {
                "checkpoint": str(path),
                "eval_loss": loss,
                "passed": passed,
                "checks": metrics.get("checks"),
                "groups": metrics.get("groups"),
                "report": location,
            }
        )
        write_json(destination / "instruction_selection.json", report)
        if passed:
            qualified = path
            break
    if qualified is None:
        raise RuntimeError(
            f"no saved checkpoint passes finite-loss + generation acceptance; see {destination}"
        )
    selected_state = json.loads((qualified / "trainer_state.json").read_text())
    selected_tokens = max(
        (row.get("train_supervised_tokens", 0) for row in selected_state["log_history"]), default=0
    )
    budget_file = dev.with_name("train.report.json")
    budget = json.loads(budget_file.read_text()) if budget_file.is_file() else {}
    requested = budget.get("requested_tokens")
    args_file = qualified.parent / "args.json"
    training_args = json.loads(args_file.read_text()) if args_file.is_file() else {}
    epochs = training_args.get("num_train_epochs")
    requested_total = requested * epochs if requested and epochs is not None else None
    report["input_tokens_per_epoch"] = budget.get("input_tokens")
    report["configured_epochs"] = epochs
    report["planned_input_tokens"] = (
        budget["input_tokens"] * epochs
        if budget.get("input_tokens") is not None and epochs is not None
        else None
    )
    report["selected_checkpoint_supervised_tokens"] = selected_tokens
    report["supervised_budget"] = {
        "requested_per_epoch": requested,
        "prepared_per_epoch": budget.get("supervised_tokens"),
        "requested_total": requested_total,
        "actual_run": actual_tokens,
        "actual_selected": selected_tokens,
        "selected_reached_target": selected_tokens >= requested_total if requested_total else None,
    }
    report["completion_scope"] = (
        "selection, merging and generation checks; strict cross-route token equality is not established"
    )
    merged = destination / "merged"
    merged.mkdir()
    merge_model(base, qualified, merged, dev)
    passed, _, location = check(merged, None, dev, destination / "merged_check")
    report.update(selected_adapter=str(qualified), merged=str(merged), merged_check=location, complete=passed)
    write_json(destination / "instruction_selection.json", report)
    if not passed:
        raise RuntimeError(f"merged model fails generation acceptance; see {location}")
    atomic_write(output / "next_stage_model.txt", str(merged.resolve()) + "\n")
    print(f"[instruction] published merged model (mode={report['mode']}): {merged}", flush=True)
    return merged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    a = p.parse_args()
    finish(a.model.resolve(), a.adapter_root.resolve(), a.output.resolve(), a.dev.resolve())


if __name__ == "__main__":
    main()
