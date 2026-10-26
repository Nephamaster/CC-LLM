"""Offline SQuAD v2 task and scorer."""
from __future__ import annotations

import collections
import os
import re
import string
from functools import partial
from pathlib import Path

from lm_eval.tasks.squadv2.task import SQuAD2

from .datasets import load_local_parquet
from .protocol import BENCHMARKS, DEFAULT_DATA_DIR


def _normalize(text):
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _answer_scores(prediction, answers):
    golds = [answer for answer in answers if _normalize(answer)] or [""]
    exact = max(float(_normalize(prediction) == _normalize(gold)) for gold in golds)
    f1 = 0.0
    prediction_tokens = _normalize(prediction).split()
    for gold in golds:
        gold_tokens = _normalize(gold).split()
        common = collections.Counter(prediction_tokens) & collections.Counter(gold_tokens)
        shared = sum(common.values())
        if not prediction_tokens or not gold_tokens:
            score = float(prediction_tokens == gold_tokens)
        elif shared == 0:
            score = 0.0
        else:
            precision = shared / len(prediction_tokens)
            recall = shared / len(gold_tokens)
            score = 2 * precision * recall / (precision + recall)
        f1 = max(f1, score)
    return exact, f1


def _summary(exact, f1, ids):
    ids = list(ids)
    if not ids:
        return {"total": 0, "exact": 0.0, "f1": 0.0}
    return {
        "total": len(ids),
        "exact": 100.0 * sum(exact[qid] for qid in ids) / len(ids),
        "f1": 100.0 * sum(f1[qid] for qid in ids) / len(ids),
    }


def _best_score(raw, predictions, no_answer, has_answer):
    current = sum(1.0 for qid in raw if not has_answer[qid])
    best = current
    threshold = 0.0
    for qid in sorted(raw, key=lambda item: no_answer[item]):
        if has_answer[qid]:
            current += raw[qid]
        elif _normalize(predictions[qid]):
            current -= 1.0
        if current > best:
            best = current
            threshold = no_answer[qid]
    return 100.0 * best / len(raw), threshold


def squad_metric(predictions, references):
    references = {row["id"]: row for row in references}
    prediction_text = {row["id"]: row["prediction_text"] for row in predictions}
    no_answer = {row["id"]: float(row.get("no_answer_probability", 0.0)) for row in predictions}
    has_answer = {qid: bool(row["answers"]["text"]) for qid, row in references.items()}
    raw_exact, raw_f1 = {}, {}
    for qid, row in references.items():
        raw_exact[qid], raw_f1[qid] = _answer_scores(prediction_text.get(qid, ""), row["answers"]["text"])

    exact = dict(raw_exact)
    f1 = dict(raw_f1)
    for qid in references:
        if no_answer.get(qid, 0.0) > 1.0:
            exact[qid] = f1[qid] = float(not has_answer[qid])
    result = _summary(exact, f1, references)
    for prefix, ids in (("HasAns", (qid for qid in references if has_answer[qid])),
                        ("NoAns", (qid for qid in references if not has_answer[qid]))):
        group = _summary(exact, f1, ids)
        result.update({f"{prefix}_{key}": value for key, value in group.items()})
    result["best_exact"], result["best_exact_thresh"] = _best_score(
        raw_exact, prediction_text, no_answer, has_answer)
    result["best_f1"], result["best_f1_thresh"] = _best_score(
        raw_f1, prediction_text, no_answer, has_answer)
    return result


def aggregate_squad(key, items):
    predictions, references = zip(*items)
    return squad_metric(predictions, references).get(key, 0)


class LocalSQuAD2(SQuAD2):
    DATASET_PATH = None

    def download(self, dataset_kwargs=None, **kwargs):
        del kwargs
        options = dict(dataset_kwargs or {})
        options.setdefault("data_dir", str(Path(os.getenv("CC_LLM_LMEH_DATA_DIR", DEFAULT_DATA_DIR)) / "squadv2"))
        options.setdefault("name", "default")
        self.config.dataset_kwargs = options
        self.dataset = load_local_parquet(**options)

    def aggregation(self):
        return {key: partial(aggregate_squad, key) for key in super().aggregation()}

    def construct_requests(self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs):
        requests = super().construct_requests(
            doc, ctx, chat_template=chat_template, apply_chat_template=apply_chat_template, **kwargs)
        prompt, generation_kwargs = requests[0].arguments
        requests[0].arguments = (
            prompt,
            {**generation_kwargs, "max_gen_toks": BENCHMARKS["squadv2"]["max_gen_toks"]},
        )
        return requests
