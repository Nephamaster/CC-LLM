"""Fixed-Dev generation, saved-LoRA effect and real-prefix KV-cache acceptance."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import copy
from dataclasses import asdict
from itertools import zip_longest
import json
import os
from pathlib import Path
import traceback

from scripts.evaluation.common import ROOT, iter_jsonl
from scripts.evaluation.runs import new_run_directory
from .dev_metrics import DevPolicy, generation_stats, summarize


def fixed_samples(path, task, count):
    channels = ("c2m", "m2c", "punctuate") if task == "ancient" else (task,)
    selected, counts = [], Counter()
    metadata = path.with_name(path.stem + ".metadata.jsonl")
    metas = iter_jsonl(metadata) if task == "ancient" and metadata.is_file() else None
    rows = zip_longest(iter_jsonl(path), metas) if metas is not None else ((row, None) for row in iter_jsonl(path))
    for index, (row, meta) in enumerate(rows):
        if row is None or (metas is not None and (meta is None or meta["messages"] != row["messages"])):
            raise ValueError("Dev messages and metadata do not match")
        channel = row.get("channel", meta.get("task") if meta else task) if task != "instruction" else task
        if channel not in channels:
            raise ValueError(f"Dev has unexpected channel {channel}; ancient Dev needs channel or matching metadata")
        messages = row.get("messages", [])
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError("Dev must end with the actual assistant reference")
        if counts[channel] < count:
            selected.append({"index": index, "channel": channel, "messages": messages})
            counts[channel] += 1
        if all(counts[c] >= count for c in channels):
            break
    if any(not counts[c] for c in channels):
        raise ValueError(f"Dev is missing required channels: {dict(counts)}")
    return selected


@contextmanager
def evaluation_state(model):
    training = [(module, module.training) for module in model.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    cache = getattr(model.config, "use_cache", None)
    generation_cache = getattr(model.generation_config, "use_cache", None)
    model.eval()
    try:
        yield
    finally:
        for module, active in training:
            module.training = active
        for parameter, active in gradients:
            parameter.requires_grad_(active)
        model.config.use_cache = cache
        model.generation_config.use_cache = generation_cache


def encoded_case(template, row):
    training = copy(template)
    training.set_mode("train")
    inference = copy(template)
    inference.set_mode("transformers")
    full = training.encode({"messages": row["messages"]})
    prompt = inference.encode({"messages": row["messages"][:-1]})
    if isinstance(full, list) or isinstance(prompt, list):
        raise ValueError("SFT Dev acceptance requires unsplit examples")
    ids = full["input_ids"]
    prefix = prompt["input_ids"]
    eos = template.tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(eos) != 1:
        raise ValueError("<|im_end|> must map to exactly one token")
    positions = [i for i, value in enumerate(ids) if value == eos[0]]
    supervised = (bool(positions and full["labels"][positions[-1]] == eos[0])
                  and sum(full["labels"][i] == eos[0] for i in positions)
                  == sum(message["role"] == "assistant" for message in row["messages"]))
    aligned = ids[:len(prefix)] == prefix
    return inference, prefix, ids[len(prefix):], eos[0], supervised, aligned


def forward(model, ids, *, cache=False, past=None, keep=1):
    import torch

    device = model.get_input_embeddings().weight.device
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    length = len(ids) + (past.get_seq_length() if past is not None else 0)
    kwargs = dict(input_ids=tokens, attention_mask=torch.ones((1, length), dtype=torch.long, device=device),
                  use_cache=cache, past_key_values=past, logits_to_keep=keep, return_dict=True)
    if getattr(model.config, "model_type", None) == "qwen3_pgca":
        kwargs["return_aux_logits"] = False
    output = model(**kwargs)
    if not torch.isfinite(output.logits).all():
        raise ValueError("non-finite Dev logits")
    return output


def generate(model, template, ids, reference, im_end, policy, cache=True):
    import torch
    from swift import RequestConfig
    from swift.infer_engine.utils import prepare_generation_config

    reference_tokens = len(template.tokenizer.encode(reference, add_special_tokens=False))
    budget = min(policy.max_new_tokens, max(64, 2 * reference_tokens + 16))
    if len(ids) + budget > model.config.max_position_embeddings:
        raise ValueError("Dev prompt plus generation budget exceeds the model context")
    request = RequestConfig(max_tokens=budget, temperature=0, stream=False)
    config = prepare_generation_config(model.generation_config, request, template.tokenizer)
    config.use_cache = cache
    config.return_dict_in_generate = True
    config.output_scores = False
    device = model.get_input_embeddings().weight.device
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    kwargs = dict(input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=config,
                  synced_gpus=False, logits_to_keep=1)
    kwargs = template.prepare_generate_kwargs(kwargs, model=model)
    generated = template.generate(model, **kwargs).sequences[0, len(ids):].tolist()
    raw = template.tokenizer.decode(generated, skip_special_tokens=False)
    body = template.tokenizer.decode(template.skip_stop_tokens(list(generated)), skip_special_tokens=False)
    stop_ids = config.eos_token_id or []
    stop_ids = [stop_ids] if isinstance(stop_ids, int) else list(stop_ids)
    for word in template.template_meta.stop_words:
        values = template.tokenizer.encode(word, add_special_tokens=False) if isinstance(word, str) else word
        if values is not None and len(values) == 1:
            stop_ids.extend(values)
    stats = generation_stats(generated, body, reference,
                             template.tokenizer.decode(ids), budget, im_end, set(stop_ids))
    return {**stats, "raw": raw, "text": body, "token_ids": generated, "use_cache": cache,
            "reference_tokens": reference_tokens, "budget_covers_reference": budget > reference_tokens}


def cache_probe(model, prefix, continuation, policy):
    import torch

    # Same real prompt followed by teacher-forced reference tokens; no padding.
    tail = continuation[:policy.cache_steps]
    if not tail:
        raise ValueError("no continuation tokens for cache verification")
    expected = forward(model, prefix + tail, keep=len(tail) + 1).logits.float().transpose(0, 1)
    output = forward(model, prefix, cache=True)
    actual = [output.logits[:, -1].float()]
    past = output.past_key_values
    lengths = []
    feature_context = (getattr(model.config, "model_type", None) == "qwen3_pgca"
                       and getattr(model.config, "pgca_context_enabled", False))

    def check_length(total):
        backbone = past.get_seq_length()
        feature = getattr(past, "feature_cache", None)
        feature_length = feature.get_seq_length() if feature is not None else None
        lengths.append({"expected": total, "backbone": backbone, "feature": feature_length})
        if backbone != total or (feature_context and feature_length != total):
            raise ValueError(f"backbone/feature cache length mismatch: {lengths[-1]}")

    check_length(len(prefix))
    for index, token in enumerate(tail, 1):
        output = forward(model, [token], cache=True, past=past)
        past = output.past_key_values
        actual.append(output.logits[:, -1].float())
        check_length(len(prefix) + index)
    actual = torch.stack(actual)
    dtype = model.get_input_embeddings().weight.dtype
    default_atol, default_rtol = ((0.1, 0.02) if dtype == torch.bfloat16 else
                                  (0.02, 0.005) if dtype == torch.float16 else (1e-4, 1e-4))
    atol = default_atol if policy.cache_atol is None else policy.cache_atol
    rtol = default_rtol if policy.cache_rtol is None else policy.cache_rtol
    difference = (actual.double() - expected.double()).abs()
    return {"passed": bool(torch.isclose(actual, expected, atol=atol, rtol=rtol).all()),
            "dtype": str(dtype), "atol": atol, "rtol": rtol, "max_abs_error": difference.max().item(),
            "rms_error": difference.square().mean().sqrt().item(),
            "argmax_agreement": actual.argmax(-1).eq(expected.argmax(-1)).float().mean().item(),
            "steps": len(tail), "cache_lengths": lengths}


def audit(model, template, samples, policy, *, full):
    import torch

    if not samples:
        raise ValueError("cannot accept a checkpoint without Dev samples")
    results, probes = [], []
    channels_seen = set()
    adapter_present = bool(getattr(model, "peft_config", None))
    nonzero_b = sum(bool(torch.count_nonzero(p).item()) for name, p in model.named_parameters()
                    if ".lora_B." in name) if adapter_present and full else None
    with evaluation_state(model), torch.inference_mode():
        for sample in samples:
            print(f"[dev-check] {sample['channel']} index={sample['index']}", flush=True)
            inference, prefix, tail, im_end, supervised, aligned = encoded_case(template, sample)
            reference = sample["messages"][-1]["content"]
            answer = generate(model, inference, prefix, reference, im_end, policy)
            result = {**sample, **answer, "reference": reference, "im_end_id": im_end,
                      "im_end_supervised": supervised, "prompt_matches_training": aligned}
            results.append(result)
            if sample["channel"] in channels_seen:
                continue
            channels_seen.add(sample["channel"])
            enabled = forward(model, prefix).logits[:, -1].float()
            if full:
                cache = cache_probe(model, prefix, tail, policy)
                cache_off = generate(model, inference, prefix, reference, im_end, policy, cache=False)
                probe = {"channel": sample["channel"], "cache": cache,
                         "cache_on_off_same_tokens": answer["token_ids"] == cache_off["token_ids"],
                         "cache_generation_health_match": all(answer[key] == cache_off[key]
                             for key in ("stopped", "im_end", "hit_limit", "repetition", "abnormal")),
                         "cache_off_generation": cache_off}
                if adapter_present:
                    with model.disable_adapter():
                        disabled = forward(model, prefix).logits[:, -1].float()
                        base_generation = generate(model, inference, prefix, reference, im_end, policy)
                    probe["adapter"] = {"logits_max_abs_difference": (enabled.double() - disabled.double()).abs().max().item(),
                                        "same_generation": answer["token_ids"] == base_generation["token_ids"],
                                        "disabled_generation": base_generation}
                probes.append(probe)
    groups = summarize(results, policy)
    cache_passed = all(p["cache"]["passed"] and p["cache_generation_health_match"] for p in probes)
    adapter_passed = (not adapter_present or not full or
                      (bool(nonzero_b) and any(p["adapter"]["logits_max_abs_difference"] > 0 for p in probes)))
    checks = {"im_end_supervision": all(r["im_end_supervised"] for r in results),
              "prompt_alignment": all(r["prompt_matches_training"] for r in results),
              "cache": cache_passed if full else None,
              "adapter_effect": adapter_passed if full and adapter_present else None}
    integrity = all(value is not False for value in checks.values())
    quality = all(group["passed"] for group in groups.values())
    return {"completed": True, "full_audit": full, "passed": integrity and quality,
            "integrity_passed": integrity, "quality_passed": quality, "checks": checks, "settings": asdict(policy),
            "groups": groups, "adapter": {"applicable": adapter_present, "nonzero_lora_B_tensors": nonzero_b,
                                            "passed": adapter_passed},
            "probes": probes, "samples": results,
            "notes": ["Cache generation can differ near an argmax tie; numeric agreement and generation health are both checked.",
                      "A health mismatch needs investigation and does not by itself prove a cache implementation bug.",
                      "New rare Han characters are reported without automatically treating valid Unicode as corruption.",
                      "A reference longer than the configured budget should be reviewed before attributing truncation to weights.",
                      "This check does not establish answer accuracy or reference-data quality."]}


def save_report(output, report):
    directory = new_run_directory(output)
    samples = report.get("samples", [])
    with (directory / "samples.jsonl").open("w", encoding="utf-8", errors="backslashreplace") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {key: value for key, value in report.items() if key != "samples"}
    (directory / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8", errors="backslashreplace")
    print(f"[dev-check] passed={report['passed']} full={report.get('full_audit')} report={directory / 'report.json'}", flush=True)
    return directory


def saved_checkpoint_check(model, template, dev, task, output):
    dev_path = Path(dev).resolve() if dev else None
    try:
        policy = DevPolicy.from_env()
        if dev_path is None or task not in {"csc", "cgec", "ancient", "instruction"}:
            raise ValueError("checkpoint Dev check needs CC_LLM_SFT_DEV and CC_LLM_SFT_TASK")
        samples = fixed_samples(dev_path, task, policy.save_samples)
        report = audit(model, template, samples, policy, full=False)
    except Exception as error:
        traceback.print_exc()
        report = {"completed": False, "passed": False, "error": str(error), "full_audit": False}
    report.update(task=task, dev=str(dev_path) if dev_path else None)
    save_report(output, report)


def relocated_dev_path(value):
    if not value:
        return None
    path = Path(value)
    if path.is_file():
        return path.resolve()
    for index in range(len(path.parts) - 1):
        if path.parts[index:index + 2] in {("data", "finetune"), ("data", "downstream")}:
            relocated = ROOT / "data/downstream" / Path(*path.parts[index + 2:])
            if relocated.is_file():
                return relocated.resolve()
    return None


def validation_source(model_path, adapter_path, contract, explicit):
    if explicit is not None:
        return relocated_dev_path(explicit) or explicit.resolve()
    checkpoint = adapter_path or model_path
    # Existing checkpoints retain their original reports; new runs write sft_report.json.
    for filename in ("sft_report.json", "finetune_report.json"):
        selection = checkpoint.parent / filename
        if selection.is_file():
            path = relocated_dev_path(json.loads(selection.read_text()).get("dev_file"))
            if path is not None:
                return path
    args = json.loads(Path(contract["args_path"]).read_text())
    candidates = args.get("val_dataset", [])
    candidates = [candidates] if isinstance(candidates, str) else candidates
    if len(candidates) == 1 and isinstance(candidates[0], str):
        path = relocated_dev_path(candidates[0])
        if path is not None:
            return path
    raise ValueError("original SFT Dev file is unavailable; set DEV_CHECK_DEV or pass --dev explicitly")


def run(args):
    from peft import PeftModel
    from swift.model import get_model_processor
    from swift.utils import import_external_file
    from scripts.evaluation.inference import training_contract, training_template

    policy = DevPolicy.from_env()
    contract = training_contract(args.model, args.adapter)
    for plugin in contract["external_plugins"]:
        import_external_file(plugin)
    dev = validation_source(args.model, args.adapter, contract, args.dev)
    samples = fixed_samples(dev, args.task, policy.samples)
    model, processor = get_model_processor(str(args.model), model_type=contract["model_type"],
                                          device_map={"": args.device}, attn_impl="sdpa", download_model=False)
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    if getattr(model.config, "model_type", None) == "qwen3_pgca":
        model.config.pgca_enabled = True
    training_args = json.loads(Path(contract["args_path"]).read_text())
    template = training_template(processor, contract, training_args.get("max_length") or model.config.max_position_embeddings,
                                  {"delete": "raise"}.get(training_args.get("truncation_strategy"),
                                                         training_args.get("truncation_strategy") or "raise"))
    report = audit(model, template, samples, policy, full=True)
    report.update(task=args.task, dev=str(dev), model=str(args.model),
                  adapter_path=str(args.adapter) if args.adapter else None,
                  template=contract["template"], mode=args.mode,
                  training_settings={key: training_args.get(key) for key in
                      ("loss_scale", "is_binary_loss_scale", "max_length", "truncation_strategy", "disable_ignore_empty_think")})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--task", choices=("csc", "cgec", "ancient", "instruction"), required=True)
    parser.add_argument("--dev", type=Path, default=os.environ.get("DEV_CHECK_DEV"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=os.environ.get("DEV_CHECK_DEVICE", "cuda:0"))
    parser.add_argument("--mode", choices=("required", "report"), default=os.environ.get("DEV_CHECK_MODE", "required"))
    args = parser.parse_args()
    if args.mode not in {"required", "report"}:
        parser.error("DEV_CHECK_MODE must be required or report")
    args.model = args.model.resolve()
    if args.adapter is not None:
        args.adapter = args.adapter.resolve()
    try:
        report = run(args)
    except Exception as error:
        traceback.print_exc()
        report = {"completed": False, "full_audit": True, "passed": False,
                  "integrity_passed": False, "quality_passed": False, "error": str(error), "task": args.task,
                  "model": str(args.model), "adapter_path": str(args.adapter) if args.adapter else None,
                  "mode": args.mode}
    save_report(args.output, report)
    if not report["completed"] or not report["integrity_passed"]:
        raise SystemExit(1)
    if args.mode == "required" and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
