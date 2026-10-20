"""Read-only hooks and gradient comparisons; no training-plugin monkey patches."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from .probe_metrics import GLYPH_NAMES, distribution, tensor_digest


class LayerObserver:
    def __init__(self, model, layers, *, aggregate=True):
        self.model, self.layers, self.aggregate = model, layers, aggregate
        self.handles = []
        self.pending = defaultdict(dict)
        self.slots = {}
        self.values = {layer: defaultdict(list) for layer in layers}
        self.counts = {layer: defaultdict(int) for layer in layers}
        self.by_candidates = {layer: defaultdict(list) for layer in layers}
        self.last_digest = {}

    def __enter__(self):
        if any(self.model.model.layers[index].pgca_attn is None for index in self.layers):
            raise ValueError("requested layer has no PGCA")
        for index in self.layers:
            layer = self.model.model.layers[index]
            self.handles.extend([
                layer.self_attn.register_forward_hook(self._save_sa(index)),
                layer.pgca_attn.dropout.register_forward_hook(self._save_attention(index)),
                layer.pgca_attn.register_forward_hook(self._finish(index)),
            ])
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.pending.clear()

    def _save_sa(self, index):
        def hook(_module, _inputs, output):
            if self.model.config.pgca_enabled:
                self.pending[index]["sa"] = output[0].detach()
        return hook

    def _save_attention(self, index):
        def hook(_module, _inputs, output):
            self.pending[index]["attention"] = output.detach().float().mean(dim=2)
        return hook

    def _finish(self, index):
        def hook(_module, inputs, output):
            state = self.pending.pop(index)
            mask = inputs[2].bool()
            attention = state["attention"]
            if attention.shape != mask.shape:
                raise ValueError("unexpected PGCA slot layout")
            self.slots[index] = attention[0].cpu()
            sa, pgca = state["sa"].float(), output.detach().float()
            active = mask.any(-1)
            sa_norm, pgca_norm = sa.norm(dim=-1), pgca.norm(dim=-1)
            ratio_valid = active & sa_norm.gt(0)
            cosine_valid = ratio_valid & pgca_norm.gt(0)
            ratio = (pgca_norm[ratio_valid] / sa_norm[ratio_valid]).cpu()
            cosine = F.cosine_similarity(pgca[cosine_valid], sa[cosine_valid], dim=-1, eps=1e-30).cpu()
            # No suffix, padding, or all-invalid slot positions enter modality statistics.
            phonetic = attention[..., :-1].sum(-1)
            glyph = attention[..., -1]
            both = mask[..., :-1].any(-1) & mask[..., -1]
            self.last_digest[index] = tensor_digest([
                ("ratio", ratio), ("cosine", cosine), ("attention", attention),
            ])
            if not self.aggregate:
                return
            counts = self.counts[index]
            counts["active_tokens"] += int(active.sum())
            counts["zero_sa_norm"] += int((active & ~sa_norm.gt(0)).sum())
            counts["zero_pgca_norm"] += int((active & ~pgca_norm.gt(0)).sum())
            for key, value in (("pgca_sa_norm_ratio", ratio), ("pgca_residual_cosine", cosine),
                               ("phonetic_mass", phonetic[active]), ("glyph_mass", glyph[active]),
                               ("both_modalities_phonetic_mass", phonetic[both]),
                               ("both_modalities_glyph_mass", glyph[both])):
                self.values[index][key].append(value.cpu().numpy())
            candidates = mask[..., :-1].sum(-1)
            for number in candidates[both].unique().tolist():
                selected = both & candidates.eq(number)
                self.by_candidates[index][number].append(phonetic[selected].cpu().numpy())
        return hook

    def summary(self):
        return {str(index): {
            **dict(self.counts[index]),
            **{name: distribution(np.concatenate(values)) for name, values in self.values[index].items()},
            "phonetic_mass_by_valid_slot_count_both_modalities": {
                str(count): distribution(np.concatenate(values))
                for count, values in sorted(self.by_candidates[index].items())
            },
        } for index in self.layers}


def parameter_group(name):
    if name.startswith("model.feature_memory_builder.feature_embedding."):
        return "feature_embedding"
    if name.startswith("model.feature_memory_builder.context_encoder."):
        return "context_encoder"
    if ".pgca_attn." in name:
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if f".{projection}." in name:
                return f"pgca_{projection}"
        return "pgca_norms"
    if name.startswith("auxiliary_heads.phonetic."):
        return "auxiliary_phonetic"
    if name.startswith("auxiliary_heads."):
        return "auxiliary_glyph"
    return None


def gradient_comparison(reference, current, names):
    """Treat unused coordinates as zeros; a zero-norm vector has no direction."""
    first = second = dot = 0.0
    used = 0
    for name in names:
        a, b = reference.get(name), current.get(name)
        if a is not None:
            first += float(a.double().square().sum())
        if b is not None:
            used += 1
            second += float(b.double().square().sum())
        if a is not None and b is not None:
            dot += float((a.double() * b.double()).sum())
    cosine = max(-1., min(1., dot / (first * second) ** .5)) if first and second else None
    return {"gradient_norm": second ** .5, "clm_gradient_norm": first ** .5,
            "cosine_with_clm": cosine, "parameters_with_gradient": used,
            "parameters": len(names)}


def gradient_probe(model, inputs, labels):
    selected = {name: parameter for name, parameter in model.named_parameters()
                if parameter_group(name) is not None}
    if not selected or any(not parameter.requires_grad for parameter in selected.values()):
        raise ValueError("probe needs trainable feature/PGCA/auxiliary parameters in its isolated model")
    groups = defaultdict(list)
    for name in selected:
        groups[parameter_group(name)].append(name)
        if ".pgca_attn." in name:
            groups["shared_pgca"].append(name)
    original = model.config.pgca_enabled
    model.config.pgca_enabled = True
    result, reference = {}, {}
    try:
        for task in ("clm", "phonetic", "glyph"):
            outputs = model(**inputs, use_cache=False, return_aux_logits=True)
            sums = model.multitask_loss_sums(outputs, labels)
            denominator = sums["token_count"]
            if not denominator.item():
                raise ValueError("gradient batch has no effective targets")
            losses = {"clm": sums["clm_sum"], "phonetic": sums["phon_sum"],
                      "glyph": sum(sums[f"{name}_sum"] for name in GLYPH_NAMES) / 3}
            loss = losses[task] / denominator
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite {task} gradient loss")
            gradients = torch.autograd.grad(loss, tuple(selected.values()), allow_unused=True)
            current = {}
            for name, gradient in zip(selected, gradients):
                if gradient is not None:
                    if not torch.isfinite(gradient).all():
                        raise ValueError(f"non-finite gradient: {task}/{name}")
                    current[name] = gradient.detach().to(device="cpu", dtype=torch.float32, copy=True)
            if task == "clm":
                reference = current
            result[task] = {
                "loss": float(loss.detach()),
                "counts": {name: int(value.detach()) for name, value in sums.items() if name.endswith("count")},
                "groups": {group: gradient_comparison(reference, current, names)
                           for group, names in groups.items()},
            }
            del outputs, sums, losses, loss, gradients, current
    finally:
        model.config.pgca_enabled = original
    counts = result["clm"]["counts"]
    missing = [name for name in ("phon_count", *(f"{name}_count" for name in GLYPH_NAMES))
               if not counts.get(name)]
    return {"mode": "eval_with_grad", "optimizer_step": False,
            "normalization": "all losses / CLM effective targets; glyph = sum(three CEs) / 3",
            "auxiliary_weights_applied": False, "tasks": result,
            "missing_auxiliary_supervision": missing,
            "interpretation": "zero gradients without supervised targets do not diagnose a broken branch"}
