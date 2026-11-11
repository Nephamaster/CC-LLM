"""SFT Dev acceptance policy and generation statistics, independent of a model."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
import math
import os
import re
import unicodedata


@dataclass(frozen=True)
class DevPolicy:
    # These are the single defaults for both training callbacks and evaluation.
    samples: int = 10                 # per task/channel before formal evaluation
    save_samples: int = 1             # per task/channel at every saved checkpoint
    max_new_tokens: int = 512
    min_stop_rate: float = 0.9
    max_bad_rate: float = 0.1         # each of length, repetition and abnormal output
    cache_steps: int = 16
    cache_atol: float | None = None   # automatic, based on the loaded model dtype
    cache_rtol: float | None = None

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for field in fields(cls):
            raw = os.environ.get("DEV_CHECK_" + field.name.upper())
            if raw is None:
                continue
            default = getattr(defaults, field.name)
            values[field.name] = (None if raw == "auto" else float(raw)) if default is None else type(default)(raw)
        policy = cls(**values)
        for name in ("samples", "save_samples", "max_new_tokens", "cache_steps"):
            if getattr(policy, name) < 1:
                raise ValueError(f"DEV_CHECK_{name.upper()} must be positive")
        for name in ("min_stop_rate", "max_bad_rate"):
            if not 0 <= getattr(policy, name) <= 1:
                raise ValueError(f"DEV_CHECK_{name.upper()} must be in [0, 1]")
        for name in ("cache_atol", "cache_rtol"):
            value = getattr(policy, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"DEV_CHECK_{name.upper()} must be finite and nonnegative")
        return policy


def generation_stats(ids, body, reference, source, budget, im_end, stop_ids):
    stopped = bool(ids and ids[-1] in stop_ids)
    tokens = ids[:-1] if stopped else ids
    grams = [tuple(tokens[i:i + 4]) for i in range(max(0, len(tokens) - 3))]
    repeated_fraction = 1 - len(set(grams)) / len(grams) if grams else 0.0
    compact = re.sub(r"\s+", "", body)
    tail_loop = False
    for width in range(1, min(64, len(compact) // 4) + 1):
        unit = compact[-width:]
        repeats = 0
        while compact.endswith(unit * (repeats + 1)):
            repeats += 1
        if (repeats >= (16 if width == 1 else 4)
                and width * repeats >= len(compact) / 2 and any(c.isalnum() for c in unit)):
            tail_loop = True
            break
    controls = sorted({c for c in body if unicodedata.category(c) in {"Cc", "Cs", "Co"} and c not in "\n\r\t"})
    content = re.sub(r"<think>\s*</think>", "", body).strip()
    special = bool(re.search(r"<\|(?:im_start|im_end|endoftext)\|>", body))
    abnormal = bool("\ufffd" in body or controls or special or (reference.strip() and not content))
    # New rare Han characters are evidence, not automatically classified as corrupt text.
    new_rare = sorted({c for c in body if 0x20000 <= ord(c) <= 0x323AF and c not in source + reference})
    return {"generated_tokens": len(ids), "budget": budget, "stopped": stopped,
            "im_end": bool(ids and ids[-1] == im_end), "hit_limit": len(ids) >= budget and not stopped,
            "repetition": tail_loop or (len(tokens) >= 32 and repeated_fraction > 0.5),
            "repeated_4gram_fraction": repeated_fraction, "abnormal": abnormal,
            "abnormal_controls": controls, "new_rare_hanzi": new_rare,
            "last_token_id": ids[-1] if ids else None}


def summarize(samples, policy):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["channel"]].append(sample)
    results = {}
    for name, rows in grouped.items():
        rates = {key + "_rate": sum(bool(row[key]) for row in rows) / len(rows)
                 for key in ("stopped", "im_end", "hit_limit", "repetition", "abnormal")}
        passed = (rates["stopped_rate"] >= policy.min_stop_rate
                  and rates["im_end_rate"] >= policy.min_stop_rate
                  and all(rates[key] <= policy.max_bad_rate
                          for key in ("hit_limit_rate", "repetition_rate", "abnormal_rate")))
        results[name] = {"count": len(rows), **rates, "passed": passed}
    return results
