"""Offline P0 diagnostics against a fixed, self-contained PGCA checkpoint."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from .pgca_observers import LayerObserver, gradient_probe
from .probe_metrics import (file_digest, json_digest, paired_summary, sample_metrics,
                            tensor_digest, write_json)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "outputs/train/Qwen3-1.7B-Base-PGCA-CPT/joint/v0-20261020-023826/checkpoint-11500"
DEFAULT_CACHE = ROOT / "data/cc_llm_multitask/phase2_4096"
FEATURE_KEYS = ("is_hanzi", "pinyin_ids", "shengmu_ids", "yunmu_ids", "tone_ids",
                "pinyin_mask", "stroke_count_ids", "radical_stroke_ids", "structure_ids")


def tokenizer_digest(root):
    # Identical byte-level convention to cpt/prepare_data.py, without importing Swift.
    import hashlib
    digest = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        path = root / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    if not (root / "tokenizer.json").is_file():
        raise FileNotFoundError(root / "tokenizer.json")
    return digest.hexdigest()


def encoded_row(row):
    """Keep the exact prepared supervision, including A/B boundary masks."""
    result = {}
    for name in ("input_ids", "labels", "attention_mask", "position_ids"):
        if name in row and row[name] is not None:
            value = row[name]
            if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
                raise ValueError(f"{name} must be a flat integer sequence")
            result[name] = list(value)
    if not {"input_ids", "labels"}.issubset(result):
        raise ValueError("expected encoded input_ids and labels, not raw text")
    length = len(result["input_ids"])
    if length < 2 or any(len(value) != length for value in result.values()):
        raise ValueError("encoded lengths must match and contain next-token targets")
    if row.get("is_padding_dummy") or not any(value != -100 for value in result["labels"][1:]):
        raise ValueError("zero-target/dummy row; choose a real supervised row")
    mask = result.setdefault("attention_mask", [1] * length)
    if any(value not in (0, 1) for value in mask):
        raise ValueError("only unpacked 0/1 attention masks are supported")
    if any(not mask[i] and result["labels"][i] != -100 for i in range(1, length)):
        raise ValueError("padding must not contribute supervised labels")
    scale = row.get("loss_scale")
    if scale is not None and (len(scale) != length or any(
            scale[i] != 1 for i in range(1, length) if result["labels"][i] != -100)):
        raise ValueError("probe expects uniform loss_scale=all")
    return result


def batch_from_row(row, device):
    tensors = {name: torch.tensor([value], dtype=torch.long, device=device) for name, value in row.items()}
    labels = tensors.pop("labels")
    return tensors, labels


def score(model, inputs, labels, *, enabled):
    original = model.config.pgca_enabled
    model.config.pgca_enabled = enabled
    try:
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False, return_aux_logits=True)
            sums = model.multitask_loss_sums(outputs, labels)
            targets = labels[:, 1:]
            sums["clm_correct"] = ((outputs.logits[:, :-1].argmax(-1) == targets) & targets.ne(-100)).sum()
            result = {}
            for name, value in sums.items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"non-finite on={enabled} metric: {name}")
                result[name] = float(value) if name.endswith("_sum") else int(value)
            return result
    finally:
        model.config.pgca_enabled = original


def evaluate_pairs(model, rows, indices, device, layers):
    samples = []
    with LayerObserver(model, layers) as observer:
        for ordinal, (index, row) in enumerate(zip(indices, rows), 1):
            inputs, labels = batch_from_row(row, device)
            observer.last_digest.clear()
            on = score(model, inputs, labels, enabled=True)
            if set(observer.last_digest) != set(layers):
                raise RuntimeError("PGCA hooks did not run at every requested layer")
            if ordinal == 1:
                reference_digest = dict(observer.last_digest)
            off = score(model, inputs, labels, enabled=False)
            on_metrics, off_metrics = sample_metrics(on), sample_metrics(off)
            samples.append({"kind": "validation", "row_index": index, "input_sha256": json_digest(row),
                            "on": on, "off": off, "on_metrics": on_metrics, "off_metrics": off_metrics,
                            "delta_clm_nll": off_metrics["clm_nll"] - on_metrics["clm_nll"]})
            if ordinal == 1 or ordinal % 16 == 0 or ordinal == len(rows):
                print(f"validation {ordinal}/{len(rows)}", flush=True)
        layer_summary = observer.summary()
    # One explicit repeat checks actual hooks and forward determinism, not just fixed seeds.
    inputs, labels = batch_from_row(rows[0], device)
    with LayerObserver(model, layers, aggregate=False) as repeated:
        on_repeat = score(model, inputs, labels, enabled=True)
    off_repeat = score(model, inputs, labels, enabled=False)
    if samples[0]["on"] != on_repeat or samples[0]["off"] != off_repeat or reference_digest != repeated.last_digest:
        raise RuntimeError("fixed-input repeat differs; preserve outputs and investigate deterministic kernels")
    return samples, layer_summary, {"passed": True, "checked_validation_rows": [indices[0]],
                                    "comparison": "exact loss sums/counts and layer-statistic/attention digests"}


def load_polyphones(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {"id", "left_context", "target", "pinyin", "reviewed"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"polyphone TSV must contain {sorted(required)}")
    seen = set()
    for row in rows:
        if row["id"] in seen or not row["id"]:
            raise ValueError("polyphone IDs must be unique and nonempty")
        seen.add(row["id"])
        if not row["left_context"] or len(row["target"]) != 1:
            raise ValueError(f"need left context and exactly one target character: {row['id']}")
        if row["reviewed"] not in ("true", "false"):
            raise ValueError("reviewed must be explicitly true or false")
        row["reviewed"] = row["reviewed"] == "true"
    return rows


def rank_readings(attention, candidate_ids, valid, class_names, gold):
    weights = {}
    for index, keep in enumerate(valid):
        if keep and candidate_ids[index] >= 3:
            reading = class_names[candidate_ids[index] - 3]
            weights[reading] = weights.get(reading, 0.) + float(attention[index])
    ranking = sorted(weights.items(), key=lambda pair: (-pair[1], pair[0]))
    if len(ranking) < 2:
        return {"eligible": False, "reason": "fewer_than_two_distinct_readings", "ranking": ranking}
    if gold not in weights:
        return {"eligible": False, "reason": "gold_not_in_actual_candidates", "ranking": ranking}
    maximum = ranking[0][1]
    tied = [reading for reading, value in ranking if value == maximum]
    return {"eligible": True, "ranking": ranking,
            "gold_rank": 1 + sum(value > weights[gold] for value in weights.values()),
            "top1_tie_count": len(tied),
            "top1_correct": len(tied) == 1 and tied[0] == gold}


def evaluate_polyphones(model, tokenizer, cases, device, layers):
    names = getattr(model.config, "pgca_auxiliary_class_names", {}).get("pinyin")
    if not names:
        raise ValueError("checkpoint lacks pgca_auxiliary_class_names.pinyin; cannot identify gold readings")
    builder = model.model.feature_memory_builder
    samples, counts = [], {str(layer): {"eligible_reviewed": 0, "correct": 0} for layer in layers}
    original = model.config.pgca_enabled
    model.config.pgca_enabled = True
    try:
        with LayerObserver(model, layers, aggregate=False) as observer, torch.no_grad():
            for case in cases:
                record = {"kind": "polyphone", **case}
                text = case["left_context"] + case["target"]
                encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                start = len(case["left_context"])
                positions = [i for i, offset in enumerate(encoded["offset_mapping"])
                             if tuple(offset) == (start, len(text))]
                if len(positions) != 1 or positions[0] != len(encoded["input_ids"]) - 1:
                    record.update(eligible=False, reason="target_not_a_single_final_token")
                    samples.append(record)
                    continue
                ids = torch.tensor([encoded["input_ids"]], device=device)
                position = positions[0]
                features = builder.lookup(ids)
                valid = (features["pinyin_mask"][0, position].bool()
                         & features["pinyin_ids"][0, position].gt(2)
                         & features["is_hanzi"][0, position].bool()).tolist()
                candidates = features["pinyin_ids"][0, position].tolist()
                # Do not allocate full vocabulary logits for every prefix position.
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
                per_layer = {}
                for layer in layers:
                    result = rank_readings(observer.slots[layer][position, :-1].tolist(),
                                           candidates, valid, names, case["pinyin"])
                    result["included_in_formal_metric"] = result["eligible"] and case["reviewed"]
                    per_layer[str(layer)] = result
                    if result["included_in_formal_metric"]:
                        counts[str(layer)]["eligible_reviewed"] += 1
                        counts[str(layer)]["correct"] += int(result["top1_correct"])
                record.update(input_ids=encoded["input_ids"], target_token_position=position, layers=per_layer)
                samples.append(record)
    finally:
        model.config.pgca_enabled = original
    for values in counts.values():
        values["polyphone_top1"] = (values["correct"] / values["eligible_reviewed"]
                                   if values["eligible_reviewed"] else None)
    reviewed = sum(case["reviewed"] for case in cases)
    return samples, {"cases": len(cases), "reviewed": reviewed, "pending_review": len(cases) - reviewed,
                     "status": "reviewed" if reviewed == len(cases) else "pending_review",
                     "head_reduction": "mean attention over heads, sum duplicate reading slots",
                     "ties": "only a unique gold maximum counts as correct",
                     "layers": counts}


def git_state():
    """Git is optional provenance; deployed source copies may not contain .git."""
    def query(*arguments):
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.PIPE, timeout=10,
        ).strip()

    try:
        return {"available": True, "commit": query("rev-parse", "HEAD"), "status": query("status", "--short")}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        reason = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        return {"available": False, "commit": None, "status": None, "reason": reason}


def checkpoint_files(root):
    """Fingerprint model artifacts only; never read source code for hashing."""
    files = set()
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors.index.json",
                 "pytorch_model.bin.index.json"):
        if (root / name).is_file():
            files.add(root / name)
    weights = list(root.glob("model*.safetensors")) + list(root.glob("pytorch_model*.bin"))
    if not weights or (root / "adapter_config.json").exists():
        raise ValueError("expected full model weights, not an adapter or tokenizer-only directory")
    files.update(weights)
    return {str(path.relative_to(root)): file_digest(path) for path in sorted(files)}


def validate_output_location(output, protected):
    output = output.resolve()
    for path in protected:
        path = path.resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError(f"output must be separate from checkpoint/data: {path}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic results: {output}")


def run(args):
    validate_output_location(args.output, (args.model, args.cache))
    if not args.model.is_dir():
        raise FileNotFoundError(f"fixed checkpoint is missing: {args.model}; do not substitute a moving pointer")
    if args.samples < 1 or args.bootstrap < 1:
        raise ValueError("samples/bootstrap must be positive")
    if args.gradient_row_index < 0:
        raise ValueError("gradient-row-index must be nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if "PGCA_ENABLED" in os.environ:
        raise ValueError("unset PGCA_ENABLED: the probe manages its isolated model's on/off switch")
    preparation_path = args.cache / "preparation_report.json"
    preparation = json.loads(preparation_path.read_text())
    tokenizer_sha = tokenizer_digest(args.model)
    if preparation.get("tokenizer_sha256") != tokenizer_sha:
        raise ValueError("checkpoint tokenizer differs from prepared cache; do not retokenize silently")
    validation, train = (load_from_disk(str(args.cache / name)) for name in ("val", "train_b"))
    if not isinstance(validation, Dataset) or not isinstance(train, Dataset):
        raise ValueError("expected prepared val and train_b Dataset splits")
    indices = random.Random(args.seed).sample(range(len(validation)), args.samples)
    rows = [encoded_row(validation[index]) for index in indices]
    gradient_row = encoded_row(train[args.gradient_row_index])
    cases = load_polyphones(args.polyphones)
    print("hashing fixed checkpoint weights and model metadata", flush=True)
    model_files = checkpoint_files(args.model)
    dtype = getattr(torch, args.dtype)
    model, loading = AutoModelForCausalLM.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True, torch_dtype=dtype,
        attn_implementation=args.attention, output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("unexpected_keys") or loading.get("mismatched_keys"):
        raise ValueError(f"checkpoint does not load exactly: {loading}")
    if not getattr(model.config, "pgca_context_enabled", False) or model.auxiliary_heads is None:
        raise ValueError("P0 requires the trained contextual PGCA model with auxiliary heads")
    layers = [int(value) for value in args.layers.split(",")]
    if not layers or len(layers) != len(set(layers)) or not set(layers).issubset(model.config.pgca_layers):
        raise ValueError(f"requested layers {layers} are not a unique subset of {model.config.pgca_layers}")
    model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError("polyphone target alignment requires the checkpoint's fast tokenizer")
    state_before = tensor_digest(model.state_dict().items())
    builder = model.model.feature_memory_builder
    feature_sha = tensor_digest((key, getattr(builder, key)) for key in FEATURE_KEYS)
    manifest = {
        "schema": "cc-llm-pgca-probe-v1", "git": git_state(), "checkpoint": str(args.model.resolve()),
        "checkpoint_files_sha256": model_files, "config": model.config.to_dict(),
        "tokenizer_sha256": tokenizer_sha, "feature_index_tensor_sha256": feature_sha,
        "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "numpy")},
        "device": str(args.device), "dtype": args.dtype, "attention": args.attention,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None,
        "seed": args.seed, "layers_zero_based": layers, "bootstrap_replicates": args.bootstrap,
        "cache": str(args.cache.resolve()), "preparation_report": preparation,
        "preparation_report_sha256": file_digest(preparation_path),
        "validation_fingerprint": validation._fingerprint, "train_fingerprint": train._fingerprint,
        "validation_rows": indices, "validation_inputs_sha256": json_digest(rows),
        "gradient_row_index": args.gradient_row_index, "gradient_input_sha256": json_digest(gradient_row),
        "polyphone_file": str(args.polyphones.resolve()), "polyphone_sha256": file_digest(args.polyphones),
        "state_before_sha256": state_before,
        "protocol": {"mode": "eval", "use_cache": False, "batch_size": 1,
                     "nll_reduction": "sum NLL / sum effective next-token targets",
                     "bootstrap_unit": "paired encoded sequence, not original document",
                     "attention_population": "tokens with at least one valid feature slot",
                     "polyphone_input": "left_context + target only; no suffix",
                     "gradient_batch": "one unchanged train_b encoded sequence"},
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "probe_manifest.json", manifest)
    with (args.output / "probe_inputs.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in zip(indices, rows):
            stream.write(json.dumps({"split": "val", "row_index": index, **row}) + "\n")
        stream.write(json.dumps({"split": "train_b", "row_index": args.gradient_row_index, **gradient_row}) + "\n")
    print("paired PGCA on/off validation", flush=True)
    samples, layer_summary, repeat = evaluate_pairs(model, rows, indices, args.device, layers)
    print("polyphone attention: unreviewed cases are excluded from formal accuracy", flush=True)
    polyphone_samples, polyphones = evaluate_polyphones(model, tokenizer, cases, args.device, layers)
    with (args.output / "probe_samples.jsonl").open("x", encoding="utf-8") as stream:
        for sample in samples + polyphone_samples:
            stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
    print("three gradient-only forwards; no optimizer or checkpoint writes", flush=True)
    inputs, labels = batch_from_row(gradient_row, args.device)
    gradients = gradient_probe(model, inputs, labels)
    gradients["checkpoint_auxiliary_weights"] = {
        "phonetic": model.config.pgca_aux_phonetic_weight, "glyph": model.config.pgca_aux_glyph_weight,
        "warmup_tokens": model.config.pgca_aux_warmup_tokens,
        "trained_tokens": int(model.auxiliary_trained_tokens.item()),
    }
    write_json(args.output / "gradient_alignment.json", gradients)
    state_after = tensor_digest(model.state_dict().items())
    if state_before != state_after:
        raise RuntimeError("diagnostic changed model state; do not accept the result")
    metrics = paired_summary(samples, seed=args.seed, bootstrap=args.bootstrap)
    pending = []
    if polyphones["pending_review"]:
        pending.append("polyphone_review")
    if any(item["eligible_reviewed"] < len(cases) for item in polyphones["layers"].values()):
        pending.append("polyphone_review_or_candidate_coverage")
    if gradients["missing_auxiliary_supervision"]:
        pending.append("gradient_batch_auxiliary_supervision")
    write_json(args.output / "probe_summary.json", {
        "status": "complete" if not pending else "partial",
        "pending_acceptance": pending,
        "validation_sequences": len(samples), "paired_metrics": metrics,
        "delta_clm_nll": metrics["clm_nll"]["delta"],
        "delta_clm_nll_ci95": metrics["clm_nll"]["paired_bootstrap_ci95"],
        "layers": layer_summary, "polyphones": polyphones, "repeat_check": repeat,
        "weights_unchanged": True, "state_after_sha256": state_after,
        "gradient_report": "gradient_alignment.json",
        "limitations": ["Single fixed checkpoint ablation, not a no-PGCA retraining comparison.",
                       "Sequence bootstrap does not model dependence between fragments of one document.",
                       "Phonetic auxiliary accuracy is legal-set membership, not contextual pronunciation accuracy.",
                       "Gradient probe uses one eval-mode batch and unweighted task losses, not an optimizer update."],
    })
    print(f"saved {args.output / 'probe_summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/rnd2/p0_pgca_probe")
    parser.add_argument("--polyphones", type=Path, default=Path(__file__).with_name("polyphones_review.tsv"))
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--gradient-row-index", type=int, default=0)
    parser.add_argument("--layers", default="11,12,13,14,15,16", help="zero-based checkpoint layer indices")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
