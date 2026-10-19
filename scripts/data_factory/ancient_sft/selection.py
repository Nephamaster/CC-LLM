"""Task-balanced sampling and encoded supervision budgets."""

import json
import math
from collections import Counter

from .common import TASKS, Normalizer
from .partition import BoundaryIndex


def stratified_rows(db, split, task, source=None):
    where, args = "e.split=? AND e.task=?", [split, task]
    if source:
        where += " AND e.source=?"
        args.append(source)
    return db.execute(
        "SELECT book,payload FROM (SELECT e.book,e.payload,e.priority,ROW_NUMBER() OVER ("
        "PARTITION BY e.book,e.doc,s.length_bin,s.script ORDER BY e.priority) AS strata_rank "
        "FROM eligible e JOIN sampling_strata s ON e.id=s.id WHERE " + where + ") "
        "ORDER BY strata_rank,priority", args,
    )


def select_records(db, config, encoder):
    selected = {split: {t: [] for t in TASKS} for split in ("train", "dev")}
    stats = Counter()
    boundary = None
    for split in ("dev", "train"):
        order = ("c2m", "m2c", "dictionary", "provenance", "punctuate")
        for task in order:
            target = (config["quotas"] if split == "train" else config["dev_quotas"])[task]
            book_cap = max(1, math.ceil(target * config["max_book_fraction" if split == "train" else "dev_max_book_fraction"]))
            passes = [(None, target)]
            if task == "c2m" and split == "train":
                passes = [("historytrans", min(config["history_target"], target)), ("gujilab", target)]
            token_limit = None
            maximum = target
            other = 0
            if split == "train" and task == "punctuate":
                other = sum(r["assistant_loss_tokens"] for t, rows in selected[split].items() if t != task for r in rows)
                upper = config["punctuation_loss_fraction"][1]
                token_limit = math.floor(other * upper / (1 - upper))
                maximum = config["punctuation_max_samples"]
                passes = [(None, maximum)]
            books, loss_tokens = Counter(), 0
            for source, allowance in passes:
                added = 0
                for book, payload in stratified_rows(db, split, task, source):
                    enough = len(selected[split][task]) >= target
                    if token_limit is not None:
                        enough = enough and loss_tokens >= other * config["punctuation_loss_fraction"][0] / (1 - config["punctuation_loss_fraction"][0])
                    if len(selected[split][task]) >= maximum or enough or added >= allowance:
                        break
                    if task != "dictionary" and books[book] >= book_cap:
                        stats[f"book_cap/{split}/{task}"] += 1
                        continue
                    record = json.loads(payload)
                    lengths = encoder(record["messages"])
                    if lengths["sequence_tokens"] > config["max_length"]:
                        stats[f"overlength/{split}/{task}"] += 1
                        continue
                    if lengths["assistant_loss_tokens"] <= 0:
                        raise RuntimeError("empty assistant loss")
                    if token_limit is not None and loss_tokens + lengths["assistant_loss_tokens"] > token_limit:
                        stats["punctuation_token_budget_rejected"] += 1
                        continue
                    if boundary is not None:
                        conflict = boundary.conflict(record)
                        if conflict:
                            raise RuntimeError(f"prepared Train/Dev boundary violation: {conflict}")
                    record.update(lengths)
                    selected[split][task].append(record)
                    books[book] += 1
                    loss_tokens += lengths["assistant_loss_tokens"]
                    added += 1
        if split == "dev":
            boundary = BoundaryIndex((r for rows in selected[split].values() for r in rows), Normalizer())
    return selected, stats


def acceptance(selected, config):
    dev_checks = {}
    for task in TASKS[:3]:
        rows = selected["dev"][task]
        books = Counter(r["book_key"] for r in rows)
        maximum = max(books.values(), default=0) / len(rows) if rows else 0
        check = {
            "target": config["dev_quotas"][task], "selected": len(rows),
            "books": len(books), "minimum_books": config["dev_min_books"],
            "groups": len({r["document_group"] for r in rows}), "minimum_groups": config["dev_min_groups"],
            "max_book_fraction": maximum,
        }
        check["passed"] = (check["selected"] >= check["target"] and check["books"] >= check["minimum_books"]
                           and check["groups"] >= check["minimum_groups"]
                           and maximum <= config["dev_max_book_fraction"] + 1e-12)
        dev_checks[task] = check
    tokens = Counter({t: sum(r["assistant_loss_tokens"] for r in rows) for t, rows in selected["train"].items()})
    fraction = tokens["punctuate"] / sum(tokens.values()) if sum(tokens.values()) else 0
    low, high = config["punctuation_loss_fraction"]
    budget = {"fraction": fraction, "minimum": low, "maximum": high,
              "passed": low <= fraction <= high and tokens["punctuate"] > 0}
    core = all(len(selected["train"][t]) >= config["quotas"][t] for t in TASKS[:3])
    return {"dev_checks": dev_checks, "dev_coverage_passed": all(c["passed"] for c in dev_checks.values()),
            "punctuation_budget": budget,
            "ready_for_training": core and all(c["passed"] for c in dev_checks.values()) and budget["passed"]}
