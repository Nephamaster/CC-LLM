"""Document-local sliding-window NLL and corpus-weighted PPL metrics."""
from __future__ import annotations

import math


COUNTERS = (
    "nll", "token_count", "encoded_token_count", "char_count", "byte_count",
    "document_count", "scored_documents", "empty_documents", "no_target_documents",
)


def validate_window(model, tokenizer, window, stride):
    if not 0 < stride < window:
        raise ValueError("require 0 < stride < window (at least one overlapping context token)")
    limits = []
    for owner, name in ((model.config, "max_position_embeddings"), (model.config, "n_positions"),
                        (tokenizer, "model_max_length")):
        value = getattr(owner, name, None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            limits.append(value)
    if limits and window > min(limits):
        raise ValueError(f"window={window} exceeds context limit={min(limits)}; set a common window explicitly")


def target_windows(length, window, stride):
    """Yield (begin, end, first_target); targets 1..length-1 occur exactly once."""
    if not 0 < stride < window:
        raise ValueError("require 0 < stride < window")
    previous_end = 1  # No artificial BOS; the first token is context only.
    for begin in range(0, length, stride):
        end = min(begin + window, length)
        if end > previous_end:
            yield begin, end, previous_end
        previous_end = end
        if end == length:
            break


def score_document(model, tokenizer, text, window=2048, stride=1024, loss_chunk_size=128):
    import torch
    import torch.nn.functional as F

    validate_window(model, tokenizer, window, stride)
    if loss_chunk_size < 1:
        raise ValueError("loss_chunk_size must be positive")
    ids = tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"] if text else []
    if text and not ids:
        raise ValueError("tokenizer produced no tokens for nonempty text")
    result = dict.fromkeys(COUNTERS, 0)
    result.update(nll=0.0, encoded_token_count=len(ids), char_count=len(text),
                  byte_count=len(text.encode("utf-8")), document_count=1,
                  empty_documents=int(not text), no_target_documents=int(bool(text) and len(ids) < 2))
    if len(ids) < 2:
        return result
    device = model.get_input_embeddings().weight.device
    if device.type == "meta":
        raise ValueError("embedding is on meta device; CPU/disk offload is not supported")
    sums = []
    with torch.inference_mode():
        for begin, end, first_target in target_windows(len(ids), window, stride):
            inputs = torch.tensor([ids[begin:end]], dtype=torch.long, device=device)
            # Do not pass labels or read outputs.loss: score only the causal LM head.
            outputs = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            use_cache=False, return_dict=True)
            logits = outputs.logits
            if logits.ndim != 3 or logits.shape[:2] != inputs.shape:
                raise ValueError("model must return full-sequence causal LM logits")
            local_target = first_target - begin
            for lo in range(local_target, end - begin, loss_chunk_size):
                hi = min(lo + loss_chunk_size, end - begin)
                # Only cast the scored chunk, not a whole 2048 x vocabulary tensor.
                targets = inputs[0, lo:hi].to(logits.device)
                loss = F.cross_entropy(logits[0, lo - 1:hi - 1].float(), targets, reduction="none")
                value = loss.double().sum().item()
                if not math.isfinite(value):
                    raise ValueError("non-finite language-model NLL; check model weights/dtype")
                sums.append(value)
                result["token_count"] += hi - lo
            del outputs, logits, inputs
    result["nll"] = math.fsum(sums)
    result["scored_documents"] = 1
    if result["token_count"] != len(ids) - 1:
        raise AssertionError("sliding windows lost or duplicated targets")
    return result


def aggregate(records):
    records = list(records)
    totals = {key: sum(row[key] for row in records) for key in COUNTERS}
    totals["nll"] = math.fsum(row["nll"] for row in records)
    return with_metrics(totals)


def with_metrics(totals):
    result = dict(totals)
    nll, tokens = totals["nll"], totals["token_count"]
    if not math.isfinite(nll):
        raise ValueError("non-finite accumulated NLL")
    for name, count in (("token_nll", tokens), ("char_nll", totals["char_count"]),
                        ("byte_nll", totals["byte_count"])):
        result[name] = nll / count if tokens and count else None
    overflow = []
    for unit in ("token", "char", "byte"):
        value = result[f"{unit}_nll"]
        try:
            result[f"{unit}_ppl"] = math.exp(value) if value is not None else None
        except OverflowError:
            result[f"{unit}_ppl"] = None
            overflow.append(f"{unit}_ppl")
    result["ppl_overflow"] = overflow
    result["bpc"] = result["char_nll"] / math.log(2) if tokens else None
    result["bpb"] = result["byte_nll"] / math.log(2) if tokens else None
    # Tokenization density counts ALL encoded tokens, including each first token.
    chars = totals["char_count"]
    result["tokens_per_char"] = totals["encoded_token_count"] / chars if chars else None
    result["scored_tokens_per_char"] = tokens / chars if chars else None
    return result
