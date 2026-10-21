"""Read-only model preflight and artifact-based checks for the isolated GPU smoke run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ATTRIBUTES = ("stroke_count", "radical_stroke", "structure")
BRANCH_GROUPS = ("feature_embedding", "context_encoder", "pgca_q_proj", "pgca_k_proj",
                 "pgca_v_proj", "pgca_o_proj", "pgca_norms", "auxiliary_phonetic", "auxiliary_glyph")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_config(config):
    for name in ("use_pgca", "pgca_context_enabled", "pgca_aux_enabled", "pgca_zero_init_output"):
        require(config.get(name) is True, f"expected {name}=true")
    for name in ("pgca_aux_phonetic_weight", "pgca_aux_glyph_weight"):
        require(config.get(name) == 0.03, f"expected {name}=0.03; prepare the round-two model first")
    require(config.get("pgca_aux_warmup_tokens") == 10_000_000, "auxiliary warm-up must remain 10M")


def check_initial_model(model, tokenizer):
    check_config(model.config.to_dict())
    require(model.config.pgca_enabled, "initial PGCA must be enabled; do not use an ALIGN checkpoint")
    require(model.auxiliary_trained_tokens.item() == 0, "initial model already has Phase2 training progress")
    for index in model.config.pgca_layers:
        output = model.model.layers[index].pgca_attn.o_proj
        require(not torch.count_nonzero(output.weight).item(), f"layer {index}: O projection is not zero-initialized")
        if output.bias is not None:
            require(not torch.count_nonzero(output.bias).item(), f"layer {index}: O bias is not zero")
    model.eval()
    device = next(model.parameters()).device
    builder = model.model.feature_memory_builder
    has_phon = (builder.pinyin_mask.bool() & builder.pinyin_ids.gt(2)).any(-1)
    complete = builder.is_hanzi.bool() & has_phon
    for name in ATTRIBUTES:
        complete &= getattr(builder, f"{name}_ids").gt(2)
    hanzi = torch.where(complete)[0][:4].tolist()
    require(hanzi, "no fully supervised Hanzi found")
    english = tokenizer.encode("This is an English smoke test with numbers 123.", add_special_tokens=False)[:24]
    require(len(english) >= 2 and not builder.is_hanzi[english].any().item(), "English case must have no Hanzi")
    missing = torch.where(builder.is_hanzi.bool() & ~complete)[0][:4].tolist()
    cases = {"hanzi": hanzi * 2, "english": english, "zero_targets": hanzi * 2}
    if missing:
        cases["unknown_attributes"] = [english[0], *missing]
    results = {}
    for name, ids in cases.items():
        inputs = torch.tensor([ids], device=device)
        labels = inputs.clone()
        labels[:, 0] = -100
        if name == "zero_targets":
            labels.fill_(-100)
        model.zero_grad(set_to_none=True)
        outputs = model(input_ids=inputs, use_cache=False, return_aux_logits=True)
        sums = model.multitask_loss_sums(outputs, labels)
        require(sums["token_count"].item() == (0 if name == "zero_targets" else len(ids) - 1),
                f"{name}: incorrect next-token supervision count")
        denominator = sums["token_count"].clamp_min(1)
        phon = sums["phon_sum"] / denominator
        glyph = sum(sums[f"{key}_sum"] for key in ATTRIBUTES) / (3 * denominator)
        clm = sums["clm_sum"] / denominator
        total = clm + model.config.pgca_aux_phonetic_weight * phon + model.config.pgca_aux_glyph_weight * glyph
        require(all(torch.isfinite(value).all().item() for value in sums.values()), f"{name}: non-finite loss")
        total.backward()
        require(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
                f"{name}: non-finite gradient")
        if name in ("english", "zero_targets"):
            require(phon.item() == 0 and glyph.item() == 0, f"{name}: auxiliary loss must be zero")
        if name == "zero_targets":
            require(total.item() == 0, "zero-target batch must have zero total loss")
        results[name] = {"input_ids": ids, "clm_loss": clm.item(), "phon_loss": phon.item(),
                         "glyph_loss": glyph.item(), "total_loss": total.item(), "gradients_finite": True,
                         "counts": {key: int(value) for key, value in sums.items() if key.endswith("count")}}
        if name == "unknown_attributes":
            count = int(sums["token_count"])
            require(any(int(sums[f"{key}_count"]) < count for key in ("phon", *ATTRIBUTES)),
                    "unknown case did not exercise a missing auxiliary label")
        del outputs, sums, total, clm, phon, glyph
    model.zero_grad(set_to_none=True)
    require(model.auxiliary_trained_tokens.item() == 0, "preflight must not advance training progress")
    return {"passed": True, "cases": results, "unknown_case_available": bool(missing),
            "optimizer_step": False, "initial_auxiliary_trained_tokens": 0,
            "unknown_case_note": None if missing else "No Hanzi with missing attributes exists in this model index.",
            "note": "Zero O and zero progress do not by themselves prove that the backbone was never trained."}


def prepare(args):
    source, root = args.model.resolve(), args.root.resolve()
    require(source != root and source not in root.parents and root not in source.parents,
            "smoke output and model directories must be separate")
    require(not (root / "dataset").exists(), "use a fresh smoke directory")
    require(args.world_size > 0 and args.batch_size > 0 and args.accumulation > 0 and args.steps >= 4,
            "positive batch geometry and at least four budgeted steps are required")
    require(torch.cuda.device_count() == args.world_size, "visible GPU count must equal NPROC_PER_NODE")
    devices = []
    for index in range(args.world_size):
        with torch.cuda.device(index):
            require(torch.cuda.is_bf16_supported(), f"GPU {index} does not support BF16")
            x = torch.randn(128, 128, device=f"cuda:{index}", dtype=torch.bfloat16, requires_grad=True)
            (x @ x).float().square().mean().backward()
            require(torch.isfinite(x.grad).all().item(), f"GPU {index}: BF16 backward is not finite")
            devices.append(torch.cuda.get_device_name(index))
            del x
            torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    model, loading = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", output_loading_info=True,
    )
    require(not any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")),
            f"initial checkpoint did not load exactly: {loading}")
    model.to("cuda:0")
    report = check_initial_model(model, tokenizer)
    report.update(model=str(source), devices=devices, world_size=args.world_size,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  preflight_peak_allocated_gib=torch.cuda.max_memory_allocated(0) / 2**30)
    del model
    torch.cuda.empty_cache()
    chinese = ("这是音形上下文模型的链路测试。银行行长重新阅读重庆的音乐教材。"
               "汉字有不同的读音、笔画、部首和结构。\n")
    english = "This is an English training smoke test. Calculate 12 + 35 = 47.\n"
    unknown = report["cases"].get("unknown_attributes", {}).get("input_ids", [])[1:]
    if unknown:
        chinese += tokenizer.decode(unknown, skip_special_tokens=True) + "\n"
    texts = []
    for unit in (chinese + english, english):
        text = unit
        while len(tokenizer.encode(text, add_special_tokens=False)) < 8192:
            text *= 2
        texts.append(text)
    train_rows = max(32, 2 * args.world_size * args.batch_size * args.accumulation * args.steps)
    for split, count in (("train", train_rows), ("validation", 3)):
        folder = root / "dataset" / split
        folder.mkdir(parents=True)
        with (folder / "smoke.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(count):
                text = texts[int(index % 8 == 0)] if split == "train" else (english if index == 0 else chinese) * 8
                stream.write(json.dumps({"text": f"{split} sample {index}\n{text}"}, ensure_ascii=False) + "\n")
    report["synthetic_train_rows"] = train_rows
    report["limitations"] = ["Synthetic training/validation content overlaps: not an effectiveness evaluation.",
                             "Preflight uses eval-mode gradients at configured auxiliary weights, without an update."]
    return report


def stage_report(folder, stage, expected_tokens, validation_tokens):
    selected = Path((folder / "next_stage_model.txt").read_text().strip()).resolve()
    require(folder.resolve() in selected.parents and selected.is_dir(), f"{stage}: invalid completion pointer")
    latest = max(selected.parent.glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[1]))
    state = json.loads((latest / "trainer_state.json").read_text())
    config = json.loads((latest / "config.json").read_text())
    check_config(config)
    require(config["pgca_enabled"] == (stage != "phase1"), f"{stage}: incorrect PGCA switch")
    require(state["global_step"] >= 3, f"{stage}: too few steps to check delayed gradients")
    progress = json.loads((latest / "cc_llm_state.json").read_text())
    require(progress["stage"] == stage, f"{stage}: wrong saved stage")
    require(progress["auxiliary_trained_tokens"] == expected_tokens, f"{stage}: incorrect effective token progress")
    history = state["log_history"]
    for row in history:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                require(math.isfinite(value), f"{stage}: non-finite log field {key}")
    evaluations = [row for row in history if "eval_clm_loss" in row]
    require(evaluations and any(row.get("step", 0) > 0 for row in evaluations), f"{stage}: missing post-update validation")
    for row in evaluations:
        require(row["eval_clm_tokens"] == validation_tokens, f"{stage}: validation token count mismatch")
        require(math.isclose(row["eval_loss"], row["eval_total_loss"], rel_tol=1e-6, abs_tol=1e-6),
                f"{stage}: eval_loss scaling differs from eval_total_loss")
        if stage == "phase1":
            require(math.isclose(row["eval_loss"], row["eval_clm_loss"], rel_tol=1e-6, abs_tol=1e-6),
                    "Phase1 must have CLM-only loss")
        else:
            ratio = row["eval_weighted_phon_to_clm"] + row["eval_weighted_glyph_to_clm"]
            require(math.isclose(ratio, row["eval_weighted_aux_to_clm_ratio"], rel_tol=1e-6, abs_tol=1e-6),
                    f"{stage}: weighted auxiliary ratios disagree")
            require(math.isclose(row["eval_total_loss"], row["eval_clm_loss"] * (1 + ratio),
                                 rel_tol=1e-6, abs_tol=1e-6), f"{stage}: weighted total loss disagrees")
    training = [row for row in history if "backbone_grad_norm" in row]
    require(training, f"{stage}: missing gradient logs")
    peaks = {}
    for group in ("backbone", "branch", *BRANCH_GROUPS):
        key = f"{group}_grad_norm"
        require(all(key in row for row in training), f"{stage}: missing P1 metric {key}")
        values = [row[key] for row in training]
        require(all(isinstance(value, (int, float)) and math.isfinite(value) for value in values),
                f"{stage}: non-finite gradient metric {key}")
        frozen = (group != "backbone" if stage == "phase1" else group == "backbone" if stage == "phase2a" else False)
        require(all(value == 0 for value in values) if frozen else any(value > 0 for value in values),
                f"{stage}: unexpected {'frozen' if frozen else 'trainable'} gradient group {group}")
        peaks[key] = max(values)
    memory = [row["memory(GiB)"] for row in history if "memory(GiB)" in row]
    return {"selected_checkpoint": str(selected), "latest_checkpoint": str(latest),
            "global_step": state["global_step"], "auxiliary_trained_tokens": progress["auxiliary_trained_tokens"],
            "last_eval_clm_loss": evaluations[-1]["eval_clm_loss"], "gradient_peaks": peaks,
            "memory_gib": max(memory) if memory else None}


def summarize(root, model_name):
    preflight = json.loads((root / "preflight_report.json").read_text())
    require(preflight["passed"], "preflight did not pass")
    phase1 = json.loads((root / "cache_phase1/preparation_report.json").read_text())
    phase2 = json.loads((root / "cache_phase2/preparation_report.json").read_text())
    a, b = phase2["splits"]["train_a"], phase2["splits"]["train_b"]
    require(a["range"][1] == b["range"][0] and a["range"][0] == 0 and b["range"][1] == phase2["train_targets"],
            "A/B supervised ranges are not a contiguous partition")
    train = root / "training"
    stages = {
        "phase1": stage_report(train / f"{model_name}-ALIGN", "phase1", 0, phase1["validation_targets"]),
        "phase2a": stage_report(train / f"{model_name}-CPT/adaptation", "phase2a", a["targets"], phase2["validation_targets"]),
        "phase2b": stage_report(train / f"{model_name}-CPT/joint", "phase2b", phase2["train_targets"], phase2["validation_targets"]),
    }
    warmup_fraction = min(1., phase2["train_targets"] / 10_000_000)
    return {"passed": True, "scope": "small GPU smoke, not formal training", "stages": stages,
            "world_size": preflight["world_size"], "ddp_exercised": preflight["world_size"] > 1,
            "phase1_targets": phase1["train_targets"], "phase2_targets": phase2["train_targets"],
            "final_auxiliary_warmup_fraction": warmup_fraction, "full_auxiliary_warmup_exercised": warmup_fraction == 1,
            "unknown_case_available": preflight["unknown_case_available"],
            "limitations": ["Synthetic overlap invalidates any effectiveness conclusion.",
                             "Reduced gradient accumulation and token budgets do not validate full-run throughput.",
                             "Finite gradients are checked; this does not assert every weight changes in BF16."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-name", default="Qwen3-1.7B-Base-PGCA")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare" and args.model is not None:
        source, root = args.model.resolve(), args.root.resolve()
        require(source != root and source not in root.parents and root not in source.parents,
                "smoke output and model directories must be separate")
    report_path = args.root / ("preflight_report.json" if args.action == "prepare" else "smoke_report.json")
    require(not report_path.exists(), f"refusing to overwrite {report_path}")
    try:
        if args.action == "prepare":
            require(args.model is not None, "prepare requires --model")
            report = prepare(args)
        else:
            report = summarize(args.root, args.model_name)
    except Exception as error:
        report = {"passed": False, "error": str(error)}
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        raise
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
