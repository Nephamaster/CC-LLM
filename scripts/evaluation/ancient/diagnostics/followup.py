"""P0: teacher-forced end-token probabilities and real cache on/off generation."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from importlib.metadata import version
import json
from pathlib import Path

from .checks import TASKS, eos_evidence, make_batch, read_contract, save, traced_generation


def load_cases(path, dev_path):
    cases = json.loads(path.read_text(encoding="utf-8"))
    dev = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if Counter(c["task"] for c in cases) != Counter(dict.fromkeys(TASKS, 2)):
        raise ValueError("Expected the six existing Dev cases: two per task")
    for case in cases:
        current = dev[case["index"]]
        if case["messages"] != current["messages"] or case.get("id") != current.get("id"):
            raise ValueError(f"Existing selected case differs from current Dev at {case['index']}")
    return cases


def token_mapping(tokenizer, candidate, vocab_size):
    mapping = {}
    for text in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", candidate):
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if len(ids) != 1 or not 0 <= ids[0] < vocab_size or decoded != text:
            raise ValueError(f"Expected single-token round trip for {text!r}: ids={ids}, decoded={decoded!r}")
        mapping[text] = {"id": ids[0], "decoded": decoded,
                         "special": ids[0] in tokenizer.all_special_ids}
    if mapping[candidate]["id"] == mapping["<|im_end|>"]["id"]:
        raise ValueError("Candidate and EOS map to the same token")
    return mapping


def end_distribution(logits, tokenizer, end_id, candidate_id):
    import torch

    values = logits.float().reshape(-1)
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite teacher-forced logits")
    log_probs = values.log_softmax(-1)

    def entry(token_id):
        return {"id": token_id, "token": tokenizer.decode([token_id], skip_special_tokens=False),
                "logit": values[token_id].item(), "probability": log_probs[token_id].exp().item(),
                "log_probability": log_probs[token_id].item(),
                "rank": int((values > values[token_id]).sum().item()) + 1}
    return {"im_end": entry(end_id), "candidate": entry(candidate_id),
            "candidate_minus_eos_logit": (values[candidate_id] - values[end_id]).item(),
            "candidate_above_eos": bool(values[candidate_id] > values[end_id]),
            "top10": [entry(i) for i in values.topk(min(10, values.numel())).indices.tolist()]}


def teacher_forced_end(model, tokenizer, encoded, end_id, candidate_id):
    evidence = eos_evidence(encoded, end_id)
    if not evidence["passed"]:
        raise ValueError(f"End label check failed: {evidence}")
    # Position end-1 predicts the terminator at end. Do not include the target EOS itself.
    end = evidence["end_position"]
    inputs = make_batch([encoded["input_ids"][:end]], tokenizer.pad_token_id,
                        model.get_input_embeddings().weight.device)
    outputs = model(**inputs, use_cache=False, return_aux_logits=False, logits_to_keep=1)
    return {"end_position": end, "target_label": encoded["labels"][end], "prefix_tokens": end,
            **end_distribution(outputs.logits[0, -1], tokenizer, end_id, candidate_id)}


def repetition_fraction(tokens, n=4):
    grams = [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def run(args):
    import torch
    from swift.template import get_template
    from ...inference import TrainingTemplatePredictor

    cases = load_cases(args.cases, args.dev)
    if not torch.cuda.is_available():
        raise RuntimeError("Run P0 on the server GPU with the real 1.7B base and ACT adapter")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "act-p0-followup-v1", "status": "running", "model": str(args.model.resolve()),
              "adapter": str(args.adapter.resolve()), "cases_file": str(args.cases.resolve()),
              "reference_quality": "unverified; known mismatch at Dev index 25 retained as supplied",
              "versions": {p: version(p) for p in ("torch", "transformers", "ms-swift", "peft")},
              "end_probability": [], "cache_on_off": [], "max_new_tokens": args.max_new_tokens}
    path = args.output / "report.json"
    save(path, report)
    try:
        train_args, contract = read_contract(args.model, args.adapter)
        print("[P0] loading the same base + ACT LoRA evaluation engine", flush=True)
        predictor = TrainingTemplatePredictor(args.model, args.adapter, contract, 1, args.max_new_tokens)
        model = predictor.engine.model.eval()
        tokenizer = predictor.engine.template.tokenizer
        report["dtype"] = str(next(model.parameters()).dtype)
        report["active_adapters"] = list(model.active_adapters)
        if not model.config.pgca_enabled or not model.config.pgca_context_enabled:
            raise ValueError("PGCA/context must remain enabled for both cache conditions")
        report["mapping"] = token_mapping(tokenizer, args.candidate, model.get_output_embeddings().weight.shape[0])
        report["tokenizer_size"] = len(tokenizer)
        report["output_vocab_size"] = model.get_output_embeddings().weight.shape[0]
        report["input_vocab_size"] = model.get_input_embeddings().weight.shape[0]
        settings = dict(contract["settings"])
        settings["default_system"] = settings.pop("system", None)
        if isinstance(settings["default_system"], str) and settings["default_system"].endswith(".txt"):
            settings["default_system"] = Path(settings["default_system"]).read_text(encoding="utf-8")
        template = get_template(predictor.engine.template.processor, template_type=contract["template"],
                                loss_scale=train_args["loss_scale"], max_length=train_args["max_length"],
                                truncation_strategy="raise", **settings)
        template.set_mode("train")
        end_id, candidate_id = (report["mapping"][k]["id"] for k in ("<|im_end|>", args.candidate))
        with torch.inference_mode():
            for case in cases:
                print(f"[P0] end probabilities: {case['task']} Dev index {case['index']}", flush=True)
                encoded = template.encode({"messages": case["messages"]})
                item = {"id": case["id"], "index": case["index"], "task": case["task"]}
                for name, context in (("cpt", model.disable_adapter()), ("act", nullcontext())):
                    with context:
                        item[name] = teacher_forced_end(model, tokenizer, encoded, end_id, candidate_id)
                report["end_probability"].append(item)
                save(path, report)
            # Three existing cases, one per task, batch=1, no left or right padding.
            for task in TASKS:
                case = next(c for c in cases if c["task"] == task)
                print(f"[P0] actual cache On/Off generation: {task} Dev index {case['index']}", flush=True)
                on = traced_generation(predictor, case, use_cache=True)
                off = traced_generation(predictor, case, use_cache=False)
                if on["prompt_token_ids"] != off["prompt_token_ids"]:
                    raise ValueError("Cache On/Off prompts differ")
                if any(x["batch_size"] != 1 or x["padding_tokens"] != 0 for x in (on, off)):
                    raise ValueError("P0 cache comparison requires single sequence without padding")
                if on["generation_use_cache"] is not True or off["generation_use_cache"] is not False:
                    raise ValueError("Final generate call ignored cache override")
                for result in (on, off):
                    result["repeated_4gram_fraction"] = repetition_fraction(result["raw_token_ids"])
                report["cache_on_off"].append({"task": task, "index": case["index"], "id": case["id"],
                    "same_tokens": on["raw_token_ids"] == off["raw_token_ids"], "on": on, "off": off,
                    "cache_controls_verified": on["incremental_cache_observed"] is True
                                               and off["uncached_full_prefix_observed"] is True})
                save(path, report)
        report["status"] = "completed"
        report["summary"] = {
            "cpt_candidate_above_eos": sum(x["cpt"]["candidate_above_eos"] for x in report["end_probability"]),
            "act_candidate_above_eos": sum(x["act"]["candidate_above_eos"] for x in report["end_probability"]),
            "cache_off_stops_on_hits_limit": [x["index"] for x in report["cache_on_off"]
                if x["off"]["finish_reason"] == "stop" and x["on"]["finish_reason"] == "length"],
            "both_hit_limit": [x["index"] for x in report["cache_on_off"]
                if x["off"]["finish_reason"] == x["on"]["finish_reason"] == "length"],
            "cache_control_verified": all(x["cache_controls_verified"] for x in report["cache_on_off"])}
        save(path, report)
        print(f"[P0] report: {path}", flush=True)
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        save(path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "adapter", "dev", "cases", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--candidate", default="𫟦")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 96:
        parser.error("P0 generation budget must be between 1 and 96")
    run(args)


if __name__ == "__main__":
    main()
