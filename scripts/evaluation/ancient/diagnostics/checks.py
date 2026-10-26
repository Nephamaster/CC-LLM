"""Read-only ACT diagnosis on the real ms-swift base + LoRA inference path."""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
import random
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from ...runs import new_run_directory

TASKS = ("c2m", "m2c", "punctuate")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_contract(model, adapter):
    from ...inference import TEMPLATE_KEYS

    path = adapter / "args.json"
    if not path.exists():
        path = adapter.parent / "args.json"
    args = json.loads(path.read_text(encoding="utf-8"))
    if args.get("template") != "cc_llm_ancient" or args.get("use_chat_template") is not True:
        raise ValueError("Expected an ACT adapter trained with cc_llm_ancient and use_chat_template=true")
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if Path(adapter_config["base_model_name_or_path"]).resolve() != model.resolve():
        raise ValueError("MODEL differs from adapter_config.base_model_name_or_path; verify the base checkpoint")
    # Only read settings needed for diagnosis. No code/weight hashing or credentials in reports.
    contract = {"template": args["template"], "settings": {k: args[k] for k in TEMPLATE_KEYS if k in args},
                "external_plugins": args.get("external_plugins", []), "model_type": args.get("model_type")}
    return args, contract


def eos_evidence(encoded, end_id):
    ids, labels = encoded["input_ids"], encoded["labels"]
    if len(ids) != len(labels):
        raise ValueError("input_ids/labels lengths differ")
    ends = [i for i, token in enumerate(ids) if token == end_id]
    end = ends[-1] if ends else None
    # The final assistant terminator should be at the end (possibly followed by a newline).
    terminal = end is not None and end > 0 and len(ids) - end <= 8
    passed = terminal and labels[end] == end_id
    return {"passed": bool(passed), "sequence_tokens": len(ids), "end_position": end,
            "end_label": labels[end] if end is not None else None,
            "supervised_targets": sum(x != -100 for x in labels[1:]),
            "tail_ids": ids[-12:], "tail_labels": labels[-12:]}


def check_labels(template, rows, output):
    from tqdm import tqdm

    tokenizer = template.tokenizer
    encoded_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(encoded_end) != 1:
        raise ValueError(f"<|im_end|> must be one token, got {encoded_end}")
    template.set_mode("train")
    counts, failures = Counter(), []
    encoded_rows = []
    with (output / "eos_labels.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(tqdm(rows, desc="Dev EOS labels", unit="sample")):
            if row["messages"][-1]["role"] != "assistant":
                raise ValueError(f"Dev row {index} does not end with assistant")
            encoded = template.encode({"messages": row["messages"]})
            evidence = eos_evidence(encoded, encoded_end[0])
            evidence.update(index=index, task=row["task"], id=row.get("id", str(index)))
            handle.write(json.dumps(evidence, ensure_ascii=False) + "\n")
            counts[row["task"]] += 1
            if not evidence["passed"]:
                failures.append(index)
            encoded_rows.append(encoded)
    return {"passed": not failures, "counts": dict(counts), "end_token_id": encoded_end[0],
            "failed_indices": failures}, encoded_rows


def select_cases(rows, encoded, per_task, seed):
    rng = random.Random(seed)
    chosen = []
    for task in TASKS:
        candidates = [i for i, row in enumerate(rows) if row["task"] == task
                      and len(encoded[i]["input_ids"]) <= 512
                      and len(row["messages"][-1]["content"]) <= 80]
        if len(candidates) < per_task:
            raise ValueError(f"Not enough short Dev examples for {task}: {len(candidates)}")
        chosen.extend(rng.sample(candidates, per_task))
    return chosen


def make_batch(sequences, pad_id, device):
    import torch

    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, -len(sequence):] = torch.tensor(sequence, device=device)
        mask[i, -len(sequence):] = 1
    return {"input_ids": ids, "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp_min(0)}


def differences(reference, actual, atol, rtol):
    import torch

    reference, actual = reference.float(), actual.float()
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        return {"finite": False, "close": False}
    delta = (reference - actual).abs()
    return {"finite": True, "close": bool(torch.allclose(reference, actual, atol=atol, rtol=rtol)),
            "max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "rms": delta.square().mean().sqrt().item(),
            "argmax_agreement": (reference.argmax(-1) == actual.argmax(-1)).float().mean().item()}


def cache_lengths(cache):
    if cache is None:
        return {"type": None, "backbone": None, "feature": None}
    feature = getattr(cache, "feature_cache", None)
    return {"type": type(cache).__name__, "backbone": int(cache.get_seq_length()),
            "feature": int(feature.get_seq_length()) if feature is not None else None}


def compare_cache(model, inputs, atol, rtol):
    import torch
    from tqdm import trange

    # Compare the entire vocabulary at every valid position, using identical input tokens.
    with torch.inference_mode():
        full = model(**inputs, use_cache=False, return_aux_logits=False, logits_to_keep=0).logits.float().cpu()
        cache, steps = None, []
        width = inputs["input_ids"].shape[1]
        for i in trange(width, desc="token-by-token cache", unit="token"):
            result = model(input_ids=inputs["input_ids"][:, i:i + 1],
                           attention_mask=inputs["attention_mask"][:, :i + 1],
                           position_ids=inputs["position_ids"][:, i:i + 1],
                           past_key_values=cache, use_cache=True, return_aux_logits=False, logits_to_keep=0)
            cache = result.past_key_values
            lengths = cache_lengths(cache)
            valid = inputs["attention_mask"][:, i].bool().cpu()
            diff = differences(full[valid, i], result.logits[:, 0].float().cpu()[valid], atol, rtol)
            steps.append({"position": i, **lengths, **diff,
                          "lengths_match": lengths["backbone"] == lengths["feature"] == i + 1})
    return {"passed": all(x["close"] and x["lengths_match"] for x in steps),
            "atol": atol, "rtol": rtol, "steps": steps}


def traced_generation(predictor, row, *, use_cache=None):
    """Observe the production engine without replacing its generation or stopping logic."""
    engine = predictor.engine
    model = engine.model
    base = model.get_base_model()
    template = engine.template
    original = template.generate
    traces, generations = [], []

    def forward_hook(module, args, kwargs, result):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        traces.append({"input_tokens": int(ids.shape[1]) if ids is not None else None,
                       "use_cache": kwargs.get("use_cache"), **cache_lengths(result.past_key_values)})

    def generate_with_trace(model, *args, **kwargs):
        config = deepcopy(kwargs["generation_config"])
        if use_cache is not None:
            # Set the final call, not engine defaults: Swift may rebuild GenerationConfig per request.
            config.use_cache = use_cache
            kwargs["use_cache"] = use_cache
        kwargs["generation_config"] = config
        result = original(model, *args, **kwargs)
        ids = kwargs.get("input_ids")
        if ids is None:
            raise RuntimeError("Expected input_ids in actual template.generate call")
        mask = kwargs.get("attention_mask")
        tokens = result.sequences[0, ids.shape[1]:].tolist()
        generations.append({"raw_token_ids": tokens,
                            "prompt_token_ids": ids[0].tolist(),
                            "batch_size": int(ids.shape[0]),
                            "padding_tokens": int((mask == 0).sum().item()) if mask is not None else 0,
                            "raw_text_with_special_tokens": template.tokenizer.decode(tokens, skip_special_tokens=False),
                            "im_end_positions": [i for i, x in enumerate(tokens)
                                                 if x == template.tokenizer.convert_tokens_to_ids("<|im_end|>")],
                            "generation_use_cache": config.use_cache,
                            "eos_token_id": config.eos_token_id,
                            "stop_words": getattr(config, "stop_words", None)})
        return result

    handle = base.register_forward_hook(forward_hook, with_kwargs=True)
    template.generate = generate_with_trace
    try:
        response = predictor([{"messages": row["messages"][:-1]}])[0]
    finally:
        template.generate = original
        handle.remove()
    if len(generations) != 1:
        raise RuntimeError(f"Expected one generate call, observed {len(generations)}")
    cache_ok = len(traces) > 1 and all(
        x["use_cache"] is True and x["backbone"] is not None and x["feature"] == x["backbone"]
        and (i == 0 or x["input_tokens"] == 1 and x["backbone"] == traces[i - 1]["backbone"] + 1)
        for i, x in enumerate(traces))
    cache_off_ok = bool(traces) and all(
        x["use_cache"] is False and x["backbone"] is None and x["feature"] is None
        and (i == 0 or x["input_tokens"] == traces[i - 1]["input_tokens"] + 1)
        for i, x in enumerate(traces))
    return {**response, **generations[0], "forward_trace": traces,
            "incremental_cache_observed": cache_ok if len(traces) > 1 else None,
            "uncached_full_prefix_observed": cache_off_ok}


def run(args):
    import torch
    from swift.template import get_template
    from ...inference import TrainingTemplatePredictor

    if not torch.cuda.is_available():
        raise RuntimeError("Run this diagnostic in the server GPU training environment")
    args.output = new_run_directory(args.output)
    print(f"[diagnose] output={args.output}", flush=True)
    report = {"status": "running", "model": str(args.model.resolve()), "adapter": str(args.adapter.resolve()),
              "versions": {p: version(p) for p in ("torch", "transformers", "ms-swift", "peft")},
              "gpu": torch.cuda.get_device_name(), "seed": args.seed,
              "cache_max_tokens": args.cache_max_tokens, "generation_max_tokens": args.max_new_tokens}
    report_path = args.output / "report.json"
    save(report_path, report)
    try:
        training_args, contract = read_contract(args.model, args.adapter)
        print("[diagnose] loading real base + ACT adapter", flush=True)
        predictor = TrainingTemplatePredictor(args.model, args.adapter, contract, 1, args.max_new_tokens)
        model = predictor.engine.model.eval()
        if not callable(getattr(model, "disable_adapter", None)) or not callable(getattr(model, "get_base_model", None)):
            raise RuntimeError("Expected PEFT adapter with disable_adapter/get_base_model")
        if not model.config.pgca_enabled or not model.config.pgca_context_enabled:
            raise RuntimeError("Expected PGCA and causal feature encoder enabled")
        base = model.get_base_model()
        report["runtime"] = {"model_class": type(base).__name__, "runtime_file": inspect.getfile(type(base)),
                             "dtype": str(next(base.parameters()).dtype),
                             "attention_implementation": getattr(base.config, "_attn_implementation", None),
                             "device": str(base.get_input_embeddings().weight.device),
                             "active_adapters": list(model.active_adapters),
                             "lora_tensor_count": sum("lora_" in n for n, _ in model.named_parameters())}
        settings = dict(contract["settings"])
        settings["default_system"] = settings.pop("system", None)
        if isinstance(settings["default_system"], str) and settings["default_system"].endswith(".txt"):
            settings["default_system"] = Path(settings["default_system"]).read_text(encoding="utf-8")
        settings.update(loss_scale=training_args["loss_scale"], max_length=training_args["max_length"],
                        truncation_strategy="raise")
        training_template = get_template(predictor.engine.template.processor,
                                         template_type=contract["template"], **settings)
        report["training_encoding"] = {"template": contract["template"], "settings": settings}
        rows = [json.loads(line) for line in args.dev.read_text(encoding="utf-8").splitlines() if line.strip()]
        report["eos_supervision"], encoded = check_labels(training_template, rows, args.output)
        save(report_path, report)
        chosen = select_cases(rows, encoded, args.samples_per_task, args.seed)
        save(args.output / "selected_dev.json", [{"index": i, **rows[i]} for i in chosen])
        adapter_results, cache_results, generation_checks = [], [], []
        report["adapter_checks"] = adapter_results
        report["cache_checks"] = cache_results
        report["generation_checks"] = generation_checks
        device = base.get_input_embeddings().weight.device
        with torch.inference_mode(), (args.output / "generations.jsonl").open("w", encoding="utf-8") as handle:
            for i in chosen:
                print(f"[diagnose] adapter on/off: {rows[i]['task']} row={i}", flush=True)
                template = predictor.engine.template
                template.set_mode("transformers")
                prompt = template.encode({"messages": rows[i]["messages"][:-1]})["input_ids"]
                inputs = make_batch([prompt], template.tokenizer.pad_token_id, device)
                enabled = model(**inputs, use_cache=False, return_aux_logits=False, logits_to_keep=1).logits.cpu()
                on_generation = traced_generation(predictor, rows[i])
                with model.disable_adapter():
                    disabled = model(**inputs, use_cache=False, return_aux_logits=False, logits_to_keep=1).logits.cpu()
                    off_generation = traced_generation(predictor, rows[i])
                restored = model(**inputs, use_cache=False, return_aux_logits=False, logits_to_keep=1).logits.cpu()
                effect = differences(enabled, disabled, 0, 0)
                restore = differences(enabled, restored, args.atol, args.rtol)
                item = {"index": i, "task": rows[i]["task"], "logit_difference": effect, "restoration": restore,
                        "passed": effect["finite"] and effect.get("max_abs", 0) > 0 and restore["close"],
                        "generation_changed": on_generation["raw_token_ids"] != off_generation["raw_token_ids"]}
                adapter_results.append(item)
                generation_checks.append({"index": i, "task": rows[i]["task"],
                                          "enabled_incremental_cache": on_generation["incremental_cache_observed"],
                                          "disabled_incremental_cache": off_generation["incremental_cache_observed"],
                                          "enabled_finish_reason": on_generation["finish_reason"],
                                          "disabled_finish_reason": off_generation["finish_reason"]})
                handle.write(json.dumps({**item, "enabled": on_generation, "disabled": off_generation},
                                        ensure_ascii=False) + "\n")
                handle.flush()
                save(report_path, report)
            # One unequal-length, left-padded pair per task, with LoRA enabled.
            for task in TASKS:
                i = next(i for i in chosen if rows[i]["task"] == task)
                sequence = encoded[i]["input_ids"][:args.cache_max_tokens]
                inputs = make_batch([sequence, sequence[:max(1, len(sequence) - 3)]],
                                    training_template.tokenizer.pad_token_id, device)
                print(f"[diagnose] full vs incremental with LoRA: {task}", flush=True)
                cache_results.append({"task": task, "index": i,
                                      **compare_cache(model, inputs, args.atol, args.rtol)})
                save(report_path, report)
        report["adapter"] = {"path": str(args.adapter.resolve()), "cases": adapter_results,
                             "passed": all(x["passed"] for x in adapter_results)}
        report["cache_consistency"] = {"cases": cache_results, "passed": all(x["passed"] for x in cache_results),
                                       "scope": "three bounded left-padded Dev pairs, real base + enabled LoRA"}
        report["status"] = "completed"
        report["passed"] = all(report[k]["passed"] for k in ("eos_supervision", "adapter", "cache_consistency"))
        report["passed"] = report["passed"] and all(
            x["enabled_incremental_cache"] is not False and x["disabled_incremental_cache"] is not False
            for x in generation_checks)
        save(report_path, report)
        print(f"[diagnose] completed passed={report['passed']} report={report_path}", flush=True)
        return 0 if report["passed"] else 1
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        save(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "adapter", "dev", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cache-max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()
    if min(args.samples_per_task, args.max_new_tokens, args.cache_max_tokens) < 1 or args.atol < 0 or args.rtol < 0:
        parser.error("counts must be positive and tolerances nonnegative")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
