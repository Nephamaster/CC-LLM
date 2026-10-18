"""Next-token attribute heads and summed losses, independent of a trainer backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


GLYPH_ATTRIBUTES = ("stroke_count", "radical_stroke", "structure")


@dataclass
class PGCAAuxiliaryOutput(CausalLMOutputWithPast):
    phonetic_logits: torch.Tensor | None = None
    glyph_logits: dict[str, torch.Tensor] | None = None


class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.pgca_aux_hidden_size
        self.phonetic = nn.Sequential(nn.Linear(config.hidden_size, width), nn.SiLU(),
                                      nn.Linear(width, config.pgca_feature_vocab_sizes["pinyin"] - 3))
        self.glyph_shared = nn.Sequential(nn.Linear(config.hidden_size, width), nn.SiLU())
        self.glyph_classifiers = nn.ModuleDict({
            name: nn.Linear(width, config.pgca_feature_vocab_sizes[name] - 3) for name in GLYPH_ATTRIBUTES
        })

    def forward(self, hidden):
        glyph_hidden = self.glyph_shared(hidden)
        return self.phonetic(hidden), {name: classifier(glyph_hidden)
                                      for name, classifier in self.glyph_classifiers.items()}


def summed_next_token_losses(outputs, labels, feature_index, *, per_example=False):
    """Use the same shift and CLM denominator for every task; 0/1/2 are not classes."""
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    logits = outputs.logits[:, :-1].float()
    clm = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                          ignore_index=-100, reduction="none").reshape(targets.shape).sum(-1)
    result = {"clm_sum": clm, "token_count": valid.sum(-1)}
    if outputs.phonetic_logits is None:
        return result if per_example else {key: value.sum() for key, value in result.items()}
    ids = targets.masked_fill(~valid, 0)
    features = feature_index.lookup(ids)
    hanzi = features["is_hanzi"].bool() & valid
    candidates = features["pinyin_ids"] - 3
    candidate_mask = features["pinyin_mask"].bool() & hanzi[..., None] & candidates.ge(0)
    # Defend against duplicate readings: summing the same class twice is not a set likelihood.
    ordered, _ = candidates.masked_fill(~candidate_mask, -1).sort(dim=-1)
    unique = ordered.ge(0)
    unique[..., 1:] &= ordered[..., 1:].ne(ordered[..., :-1])
    log_prob = outputs.phonetic_logits[:, :-1].float().log_softmax(-1)
    values = log_prob.gather(-1, ordered.clamp_min(0))
    has_phonetic = unique.any(-1)
    # Finite masked values avoid an all--inf logsumexp backward on empty rows.
    values = values.masked_fill(~unique, torch.finfo(values.dtype).min)
    phonetic_nll = -values.logsumexp(-1)
    result["phon_sum"] = phonetic_nll.masked_fill(~has_phonetic, 0).sum(-1)
    result["phon_count"] = has_phonetic.sum(-1)
    result["phon_correct"] = ((log_prob.argmax(-1)[..., None] == ordered) & unique).any(-1).sum(-1)
    for name in GLYPH_ATTRIBUTES:
        target = features[f"{name}_ids"] - 3
        mask = hanzi & target.ge(0)
        target = target.masked_fill(~mask, -100)
        scores = outputs.glyph_logits[name][:, :-1].float()
        result[f"{name}_sum"] = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1),
                                                ignore_index=-100, reduction="none").reshape(target.shape).sum(-1)
        result[f"{name}_count"] = mask.sum(-1)
        result[f"{name}_correct"] = ((scores.argmax(-1) == target) & mask).sum(-1)
    return result if per_example else {key: value.sum() for key, value in result.items()}
