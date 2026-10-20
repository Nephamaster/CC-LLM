"""Paired, sequence-bootstrap statistics for the offline PGCA probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


GLYPH_NAMES = ("stroke_count", "radical_stroke", "structure")
METRICS = {
    "clm_nll": ("clm_sum", "token_count"),
    "token_accuracy": ("clm_correct", "token_count"),
    "phonetic_accuracy": ("phon_correct", "phon_count"),
    **{f"{name}_accuracy": (f"{name}_correct", f"{name}_count") for name in GLYPH_NAMES},
}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in items:
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}:".encode())
        # uint8 also supports BF16, which NumPy cannot represent directly.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("non-finite diagnostic statistic")
    return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "p50": float(np.quantile(values, .5)) if len(values) else None,
            "p95": float(np.quantile(values, .95)) if len(values) else None}


def paired_summary(samples, *, seed=42, bootstrap=2000):
    """Resample paired sequences, then recompute ratios of summed numerators/counts."""
    if not samples or bootstrap < 1:
        raise ValueError("samples and bootstrap replicates must be positive")
    result = {}
    for metric, (numerator, denominator) in METRICS.items():
        on = np.array([[row["on"][numerator], row["on"][denominator]] for row in samples], dtype=float)
        off = np.array([[row["off"][numerator], row["off"][denominator]] for row in samples], dtype=float)
        if not np.array_equal(on[:, 1], off[:, 1]):
            raise ValueError(f"on/off denominator mismatch: {metric}")
        total = float(on[:, 1].sum())
        sign = -1 if metric == "clm_nll" else 1
        delta = (sign * float(on[:, 0].sum() - off[:, 0].sum()) / total) if total else None
        draws = []
        rng = np.random.default_rng(seed)
        for _ in range(bootstrap):
            indices = rng.integers(len(samples), size=len(samples))
            count = on[indices, 1].sum()
            if count:
                draws.append(sign * float((on[indices, 0] - off[indices, 0]).sum()) / count)
        result[metric] = {
            "on": float(on[:, 0].sum() / total) if total else None,
            "off": float(off[:, 0].sum() / total) if total else None,
            "denominator": int(total), "delta": delta,
            "delta_definition": "off - on" if sign == -1 else "on - off",
            "paired_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist() if draws else None,
            "bootstrap_valid_replicates": len(draws),
        }
    return result


def sample_metrics(sums):
    return {name: sums[numerator] / sums[denominator] if sums[denominator] else None
            for name, (numerator, denominator) in METRICS.items()}
