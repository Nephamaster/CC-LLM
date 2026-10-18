"""Explicit C3 reconstruction and upstream-compatible Fuxi automatic metrics."""
import json
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from .prepare import unpunctuated
from .fuxi_choice import multiple_choice_acc


def fscore(tp, predicted, gold):
    p = tp / predicted if predicted else 0.0
    r = tp / gold if gold else 0.0
    return {"precision": p * 100, "recall": r * 100, "f1": 200 * p * r / (p + r) if p + r else 0,
            "tp": tp, "predicted": predicted, "gold": gold}


def entities(text):
    text = text.strip().strip("。")
    if text in {"", "无", "無"}:
        return set()
    return {value.strip() for value in re.split(r"[、,，;；\n]", text) if value.strip()}


def punctuation(text):
    tokens, position = Counter(), 0
    for char in text:
        if char.isspace():
            continue
        if unicodedata.category(char).startswith("P"):
            tokens[(position, char)] += 1
        else:
            position += 1
    return tokens


def title_levels(text):
    return [part.strip().strip("《》〈〉。 \n") for part in re.split(r"[·‧/]", text.strip())]


def c3_metrics(rows):
    import sacrebleu

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    result = {}
    for task, samples in grouped.items():
        if task == "classification":
            correct = sum(r["predict_clean"].strip().strip("。\"“”") == r["gold"].strip() for r in samples)
            score = {"accuracy": correct / len(samples) * 100, "correct": correct}
        elif task == "retrieval":
            levels = defaultdict(lambda: [0, 0])
            for row in samples:
                gold, predicted = title_levels(row["gold"]), title_levels(row["predict_clean"])
                for level in range(len(gold)):
                    levels[level + 1][1] += 1
                    levels[level + 1][0] += predicted[:level + 1] == gold[:level + 1]
            score = {"levels": {str(k): {"correct": c, "count": n, "accuracy": 100 * c / n} for k, (c, n) in levels.items()}}
        elif task in {"ner", "punctuation"}:
            tp = predicted_count = gold_count = invalid = 0
            for row in samples:
                if task == "ner":
                    gold, predicted = entities(row["gold"]), entities(row["predict_clean"])
                    common = len(gold & predicted)
                else:
                    gold, predicted = punctuation(row["gold"]), punctuation(row["predict_clean"])
                    preserved = unpunctuated(row["predict_clean"]) == row["src"]
                    invalid += not preserved
                    common = sum((gold & predicted).values()) if preserved else 0
                tp += common
                predicted_count += len(predicted) if task == "ner" else sum(predicted.values())
                gold_count += len(gold) if task == "ner" else sum(gold.values())
            score = fscore(tp, predicted_count, gold_count)
            score["invalid_character_sequences"] = invalid
        elif task == "translation":
            metric = sacrebleu.metrics.BLEU(tokenize="zh")
            score = {"bleu": metric.corpus_score([r["predict_clean"] for r in samples],
                                                [[r["gold"] for r in samples]]).score,
                     "signature": str(metric.get_signature())}
        else:
            raise ValueError(f"unknown C3 task: {task}")
        result[task] = {"count": len(samples), **score}
    return {"tasks": result, "protocol": "paper-based reconstruction; micro set NER, typed-position punctuation F1, hierarchical prefix retrieval, corpus SacreBLEU zh",
            "official_code_equivalence": False}


def couplet(prediction, gold):
    parts = lambda text: [p for p in re.split(r"[ \t\n，。？！、,.?!]", text.strip()) if p.strip()]
    return [len(p.strip()) for p in parts(prediction)] == [len(p.strip()) for p in parts(gold)]


@lru_cache(maxsize=1)
def ci_table():
    return json.loads(Path(__file__).with_name("cipai2info.json").read_text(encoding="utf-8"))


def ci_format(prediction, cipai):
    table = ci_table()
    if cipai not in table:
        return False
    template = table[cipai]["longyusheng_template"][0]["template"]
    lengths = [len(s.strip()) for s in re.split(r"[\t\n]", template) if s.strip()]

    def matches(text):
        parts = [p.strip() for p in re.split(r"[ \t\n，。？！、,.?!]", text.strip()) if p.strip()]
        gold = lengths
        if len(parts) != len(gold):
            if len(parts) in [len(gold) * n for n in (2, 4, 8)]:
                gold = gold * (len(parts) // len(gold))
            else:
                return False
        return [len(p) for p in parts] == gold

    no_title = prediction[prediction.find("\n") + 1:].strip()
    no_author = no_title[no_title.find("\n") + 1:].strip()
    return matches(prediction) or matches(no_title) or matches(no_author)


def fuxi_metrics(rows, judgments=None):
    import sacrebleu

    judgments = judgments or {}
    tasks = {}
    for row in rows:
        task = tasks.setdefault(row["task"], {"count": 0, "scored": 0, "pending": 0, "sum": 0.0, "metric": row["metric"],
                                            "demonstration_overlap": 0})
        task["count"] += 1
        task["demonstration_overlap"] += bool(row.get("demonstration_overlap"))
        metric, prediction, gold = row["metric"], row["predict_clean"], row["gold"]
        if metric == "judge":
            entry = judgments.get(row["id"])
            if not entry or entry.get("score") not in (0, 1):
                task["pending"] += 1
                continue
            value = entry["score"] * 100
        elif metric == "acc":
            value = 100 * multiple_choice_acc(prediction, gold)
        elif metric == "bleu":
            value = sacrebleu.sentence_bleu(prediction, [gold], tokenize="zh").score
        elif metric == "couplet":
            value = 100 * couplet(prediction, gold)
        elif metric == "ci":
            value = 100 * ci_format(prediction, row["cipai"])
        else:
            raise ValueError(metric)
        task["scored"] += 1
        task["sum"] += value
    for task in tasks.values():
        partial = task.pop("sum") / task["scored"] if task["scored"] else None
        task["complete"] = task["scored"] == task["count"]
        task["score"] = partial if task["complete"] else None
        if not task["complete"] and partial is not None:
            task["partial_score"] = partial
    return {"tasks": tasks, "complete": all(t["complete"] for t in tasks.values()),
            "note": "Fuxi task-specific metrics; no combined score for incomplete judge tasks. Ci/couplet measure format, not literary quality."}
