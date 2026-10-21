"""One additional Uniform-Attention forward pass over the saved P0 Hanzi subset."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from .pgca_probe import batch_from_row, encoded_row, score, validate_output_location
from .probe_metrics import json_digest, tensor_digest, write_json


ROOT = Path(__file__).resolve().parents[2]
CLM_KEYS = ("clm_sum", "token_count", "clm_correct")
CONDITIONS = ("normal", "uniform", "off")
COMPARISONS = (("uniform", "normal"), ("off", "normal"), ("off", "uniform"))


class UniformAttention:
    """Replace only the tensor consumed by attention @ V, using the real slot mask."""

    def __init__(self, model, layers):
        self.model, self.layers = model, list(layers)
        self.handles, self.masks = [], {}
        self.calls = {layer: 0 for layer in self.layers}

    def __enter__(self):
        if any(module.training for module in self.model.modules()):
            raise ValueError("Uniform-Attention is an eval-only intervention")
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("expected nonempty, unique PGCA layers")
        modules = [self.model.model.layers[index].pgca_attn for index in self.layers]
        if any(module is None for module in modules):
            raise ValueError("requested layer has no PGCA")
        try:
            for index, module in zip(self.layers, modules):
                self.handles.append(module.register_forward_pre_hook(self._capture_mask(index), with_kwargs=True))
                self.handles.append(module.dropout.register_forward_hook(self._replace(index)))
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.masks.clear()

    def _capture_mask(self, index):
        def hook(_module, args, kwargs):
            memory = kwargs["feature_memory"] if "feature_memory" in kwargs else args[1]
            mask = kwargs.get("feature_mask", args[2] if len(args) > 2 else None)
            if memory.ndim != 4:
                raise ValueError("expected token-aligned feature memory")
            if mask is None:
                mask = torch.ones(memory.shape[:3], device=memory.device, dtype=torch.bool)
            if mask.shape != memory.shape[:3]:
                raise ValueError("feature mask and memory shapes differ")
            self.masks[index] = mask
        return hook

    def _replace(self, index):
        def hook(_module, _inputs, attention):
            mask = self.masks.pop(index).to(device=attention.device, dtype=torch.bool)
            if attention.ndim != 4 or attention.shape[:2] != mask.shape[:2] or attention.shape[-1] != mask.shape[-1]:
                raise ValueError("expected PGCA attention shape [batch, sequence, heads, slots]")
            weights = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
            self.calls[index] += 1
            # Empty rows stay zero; invalid slots never receive mass. BF16 cast matches Normal.
            return weights.unsqueeze(2).expand_as(attention).to(dtype=attention.dtype)
        return hook


def clm_only(result):
    return {name: result[name] for name in CLM_KEYS}


def load_reference(directory, *, limit=0):
    """Select by P0 supervision, not by treatment outcome or a fresh sample draw."""
    manifest = json.loads((directory / "probe_manifest.json").read_text())
    summary = json.loads((directory / "probe_summary.json").read_text())
    if not summary["weights_unchanged"] or not summary["repeat_check"]["passed"]:
        raise ValueError("P0 did not pass state/repeat checks")
    if manifest["state_before_sha256"] != summary["state_after_sha256"]:
        raise ValueError("P0 before/after model states differ")
    saved = {}
    with (directory / "probe_inputs.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == "val":
                if row["row_index"] in saved:
                    raise ValueError("duplicate P0 input row")
                saved[row["row_index"]] = encoded_row(row)
    references = []
    with (directory / "probe_samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "validation":
                references.append(row)
    if [row["row_index"] for row in references] != manifest["validation_rows"]:
        raise ValueError("P0 manifest and sample row order differ")
    if set(saved) != set(manifest["validation_rows"]) or len(saved) != len(references):
        raise ValueError("P0 inputs and sample rows differ")
    selected = []
    for reference in references:
        row = saved[reference["row_index"]]
        if json_digest(row) != reference["input_sha256"]:
            raise ValueError(f"P0 input/labels changed: row {reference['row_index']}")
        count = sum(label != -100 for label in row["labels"][1:])
        if count != reference["on"]["token_count"] or count != reference["off"]["token_count"]:
            raise ValueError("P0 paired target counts differ from saved labels")
        if reference["on"]["phon_count"] > 0:
            selected.append((reference, row))
    if not selected or limit < 0:
        raise ValueError("need a nonempty P0 Hanzi subset and nonnegative limit")
    eligible = len(selected)
    if limit:
        selected = selected[:limit]
    return manifest, selected, eligible


def summarize(samples, *, seed, bootstrap):
    """Positive A-minus-B means condition B has lower token-weighted NLL."""
    if not samples or bootstrap < 1:
        raise ValueError("samples and bootstrap replicates must be positive")
    counts = np.array([row["normal"]["token_count"] for row in samples], dtype=np.int64)
    numerators = {}
    for condition in CONDITIONS:
        if any(row[condition]["token_count"] != int(count) for row, count in zip(samples, counts)):
            raise ValueError("three-condition target counts differ")
        numerators[condition] = np.array([row[condition]["clm_sum"] for row in samples], dtype=np.float64)
        if not np.isfinite(numerators[condition]).all():
            raise ValueError("non-finite CLM NLL")
    if np.any(counts <= 0):
        raise ValueError("all sequences must have effective targets")
    totals = {condition: float(values.sum() / counts.sum()) for condition, values in numerators.items()}
    comparisons = {}
    for first, second in COMPARISONS:
        difference = numerators[first] - numerators[second]
        rng, draws = np.random.default_rng(seed), []
        for _ in range(bootstrap):
            indices = rng.integers(len(samples), size=len(samples))
            draws.append(float(difference[indices].sum() / counts[indices].sum()))
        comparisons[f"{first}_minus_{second}"] = {
            "delta_nll": float(difference.sum() / counts.sum()),
            "definition": f"{first}_nll - {second}_nll; positive favors {second}",
            "paired_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist(),
            "sequences_favoring_second": int((difference > 0).sum()),
            "sequences_favoring_first": int((difference < 0).sum()),
            "sequences_equal": int((difference == 0).sum()),
        }
    return {"validation_sequences": len(samples), "effective_targets": int(counts.sum()),
            **{f"{condition}_nll": value for condition, value in totals.items()},
            "paired_comparisons": comparisons, "bootstrap_replicates": bootstrap}


def evaluate_uniform(model, selected, device, layers, stream):
    first_reference, first_row = selected[0]
    inputs, labels = batch_from_row(first_row, device)
    # A cheap anchor before reusing all saved Normal/Off results; no new baseline sweep.
    for enabled, key in ((True, "on"), (False, "off")):
        actual = clm_only(score(model, inputs, labels, enabled=enabled))
        if actual != clm_only(first_reference[key]):
            raise ValueError(f"P0 {key} anchor differs; check model/runtime/device before pairing old results")
    samples = []
    with UniformAttention(model, layers) as intervention:
        for ordinal, (reference, row) in enumerate(selected, 1):
            inputs, labels = batch_from_row(row, device)
            uniform = clm_only(score(model, inputs, labels, enabled=True))
            if any(count != ordinal for count in intervention.calls.values()):
                raise RuntimeError("uniform replacement did not run exactly once per layer and sequence")
            sample = {"row_index": reference["row_index"], "input_sha256": reference["input_sha256"],
                      "normal": clm_only(reference["on"]), "uniform": uniform, "off": clm_only(reference["off"])}
            count = uniform["token_count"]
            for condition in CONDITIONS:
                sample[f"{condition}_nll"] = sample[condition]["clm_sum"] / count
            for first, second in COMPARISONS:
                sample[f"{first}_minus_{second}"] = sample[f"{first}_nll"] - sample[f"{second}_nll"]
            stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            samples.append(sample)
            if ordinal == 1 or ordinal % 16 == 0 or ordinal == len(selected):
                print(f"Uniform-Attention {ordinal}/{len(selected)}", flush=True)
        inputs, labels = batch_from_row(first_row, device)
        repeated = clm_only(score(model, inputs, labels, enabled=True))
        if repeated != samples[0]["uniform"]:
            raise RuntimeError("fixed-input Uniform repeat differs")
    # Confirms that the replacement was removed, rather than leaking into later forwards.
    if clm_only(score(model, inputs, labels, enabled=True)) != clm_only(first_reference["on"]):
        raise RuntimeError("Normal did not recover after removing the Uniform hooks")
    return samples, {
        "passed": True, "row_index": first_reference["row_index"],
        "normal_off_anchor_matches_p0": True, "uniform_repeat_matches": True,
        "normal_restored_after_hooks": True, "uniform_hook_calls": intervention.calls,
    }


def run(args):
    source, selected, eligible = load_reference(args.p0, limit=args.limit)
    model_path = args.model or Path(source["checkpoint"])
    validate_output_location(args.output, (args.p0, model_path))
    seed, bootstrap = source["seed"], source["bootstrap_replicates"]
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if "PGCA_ENABLED" in os.environ:
        raise ValueError("unset PGCA_ENABLED: the isolated probe manages the branch switch")
    model, loading = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True,
        torch_dtype=getattr(torch, source["dtype"]), attn_implementation=source["attention"],
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise ValueError(f"checkpoint does not load exactly: {loading}")
    model.to(args.device).eval()
    layers = source["layers_zero_based"]
    if list(model.config.pgca_layers) != layers:
        raise ValueError("Uniform must replace every PGCA layer, matching the P0 layer set")
    before = tensor_digest(model.state_dict().items())
    if before != source["state_before_sha256"]:
        raise ValueError("model state differs from P0; do not pair another checkpoint with these baselines")
    manifest = {
        "schema": "cc-llm-uniform-pgca-v1", "p0_directory": str(args.p0.resolve()),
        "checkpoint": str(model_path.resolve()), "source_checkpoint": source["checkpoint"],
        "state_before_sha256": before, "validation_rows": [r["row_index"] for r, _ in selected],
        "eligible_sequences": eligible, "selection": "P0 validation rows where on.phon_count > 0",
        "limit": args.limit, "dtype": source["dtype"], "attention": source["attention"],
        "device": args.device, "layers_zero_based": layers, "seed": seed,
        "bootstrap_replicates": bootstrap,
        "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")},
        "gpu": torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None,
        "protocol": {"mode": "eval", "use_cache": False, "batch_size": 1,
                     "uniform_rule": "mask / number of valid slots, independently at every token and head",
                     "empty_mask": "all zero", "slot_population": "pinyin and glyph together",
                     "intervention": "replace PGCA dropout output; V/O/mask/feature memory untouched",
                     "baselines": "reuse saved P0 on/off sums for the exact selected subset",
                     "bootstrap_unit": "paired encoded sequence", "optimizer_step": False},
    }
    args.output.mkdir(parents=True)
    write_json(args.output / "uniform_manifest.json", manifest)
    with (args.output / "uniform_samples.jsonl").open("x", encoding="utf-8") as stream:
        samples, checks = evaluate_uniform(model, selected, args.device, layers, stream)
    after = tensor_digest(model.state_dict().items())
    if before != after:
        raise RuntimeError("Uniform experiment changed model state")
    summary = summarize(samples, seed=seed, bootstrap=bootstrap)
    summary.update(
        status="complete" if len(samples) == eligible else "preflight_subset",
        eligible_sequences=eligible, checks=checks, weights_unchanged=True, state_after_sha256=after,
        limitations=[
            "All-slot uniformity changes both modality mass and within-modality candidate allocation.",
            "This does not isolate polyphone disambiguation or the benefit of the context encoder.",
            "No retraining: conclusions concern interventions on this trained checkpoint.",
            "Sequence bootstrap does not model dependence between fragments of one document.",
            "A confidence interval including zero does not establish equivalence.",
        ],
    )
    write_json(args.output / "uniform_summary.json", summary)
    print(f"saved {args.output / 'uniform_summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", type=Path, default=ROOT / "outputs/experiments/rnd2/p0_pgca_probe")
    parser.add_argument("--model", type=Path, help="optional relocated copy of the identical P0 model")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/rnd2/p0_uniform_pgca")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="0 = all eligible rows; positive = preflight prefix only")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
