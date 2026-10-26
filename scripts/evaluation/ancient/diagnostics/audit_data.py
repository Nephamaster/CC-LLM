"""P1: reproducible per-task Train/Dev sample and source/structure checks, without rewriting data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from itertools import zip_longest
import json
from pathlib import Path
import random
import unicodedata

from .checks import TASKS, save
from ...runs import new_run_directory


def unpunctuated(value):
    return "".join(c for c in value if not c.isspace() and not unicodedata.category(c).startswith("P"))


def text_pair(row):
    if row["task"] == "m2c":
        return row["modern"], row["ancient"]
    return row["ancient"], row["modern"]


def sample_split(folder, split, count, seed):
    pools = {task: [] for task in TASKS}
    with (folder / f"{split}.jsonl").open(encoding="utf-8") as data, \
            (folder / f"{split}.metadata.jsonl").open(encoding="utf-8") as meta:
        for line, (content, metadata) in enumerate(zip_longest(data, meta), 1):
            if content is None or metadata is None:
                raise ValueError(f"{split}: data/metadata row counts differ")
            row = json.loads(metadata)
            if json.loads(content)["messages"] != row["messages"]:
                raise ValueError(f"{split} line {line}: data/metadata messages differ")
            if row["task"] in pools:
                pools[row["task"]].append({**row, "audit_split": split, "audit_line": line})
    result = []
    for task, pool in pools.items():
        if len(pool) < count:
            raise ValueError(f"{split}/{task}: only {len(pool)} rows, need {count}")
        # No length/quality prefilter: preserve a random per-task sample for a meaningful denominator.
        result.extend(random.Random(f"{seed}/{split}/{task}").sample(pool, count))
    return result, {task: len(pool) for task, pool in pools.items()}


def source_targets(rows, raw_dir):
    targets = defaultdict(dict)
    for index, row in enumerate(rows):
        location = row.get("metadata", {}).get("source_location", "")
        try:
            filename, number = location.rsplit(":", 1)
            number = int(number)
        except ValueError:
            continue
        name = Path(filename).name
        if name in {"translate.jsonl", "punctuate.jsonl"}:
            path = raw_dir / "chinese-classical-corpus" / name
        elif name == "HistoryTrans-train.jsonl":
            path = raw_dir / "HistoryTrans" / name
        else:
            continue
        targets[path].setdefault(number, []).append(index)
    return targets


def trace_sources(rows, raw_dir):
    from scripts.data_factory.ancient_sft.common import clean

    evidence = {}
    for path, targets in source_targets(rows, raw_dir).items():
        if not path.is_file():
            continue
        print(f"[P1] checking source rows: {path.name}", flush=True)
        last = max(targets)
        with path.open(encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, 1):
                if number in targets:
                    raw = json.loads(line)
                    source = clean(raw.get("input", raw.get("inputs", "")))
                    target = clean(raw.get("output", raw.get("truth", "")))
                    for index in targets[number]:
                        expected_source, expected_target = text_pair(rows[index])
                        compare = (lambda s: "".join(s.split())) if rows[index]["task"] == "punctuate" else clean
                        evidence[index] = {"path": str(path), "line": number,
                            "source": source, "target": target,
                            "pair_matches": compare(source) == compare(expected_source)
                                            and compare(target) == compare(expected_target)}
                if number >= last:
                    break
    return evidence


def structural_flags(row):
    source, target = text_pair(row)
    flags = []
    messages = row["messages"]
    if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
        flags.append("unexpected_message_roles")
    if not messages[0]["content"].endswith(source) or messages[-1]["content"] != target:
        flags.append("metadata_message_mismatch")
    if not source.strip() or not target.strip():
        flags.append("empty_pair")
    if row["task"] == "punctuate":
        if unpunctuated(source) != unpunctuated(target):
            flags.append("punctuation_changes_characters")
        if target and unicodedata.category(target[0]).startswith("P"):
            flags.append("leading_punctuation_review")
    else:
        if unpunctuated(source) == unpunctuated(target):
            flags.append("translation_copies_source_review")
        if len(target) / max(len(source), 1) > 3 or len(target) / max(len(source), 1) < .33:
            flags.append("length_ratio_review")
    return flags


def run(args):
    args.output = new_run_directory(args.output)
    print(f"[P1] output={args.output}", flush=True)
    rows, populations = [], {}
    for split in ("train", "dev"):
        sampled, populations[split] = sample_split(args.data, split, args.per_task, args.seed)
        rows.extend(sampled)
    evidence = trace_sources(rows, args.raw)
    counts = defaultdict(Counter)
    fields = ["id", "split", "task", "line", "book", "source_location", "source", "target", "flags",
              "source_pair_matches", "semantic_verdict", "review_notes"]
    with (args.output / "review.tsv").open("w", encoding="utf-8", newline="") as handle, \
            (args.output / "samples.jsonl").open("w", encoding="utf-8") as full:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for index, row in enumerate(rows):
            source, target = text_pair(row)
            flags = structural_flags(row)
            raw = evidence.get(index)
            group = f"{row['audit_split']}/{row['task']}"
            counts[group]["sampled"] += 1
            counts[group]["source_verified"] += int(raw is not None)
            counts[group]["source_pair_mismatch"] += int(raw is not None and not raw["pair_matches"])
            counts[group]["flagged"] += bool(flags)
            counts[group]["semantic_pending"] += 1
            writer.writerow({"id": row["id"], "split": row["audit_split"], "task": row["task"],
                             "line": row["audit_line"], "book": row.get("book"),
                             "source_location": row.get("metadata", {}).get("source_location"),
                             "source": source, "target": target, "flags": ",".join(flags),
                             "source_pair_matches": raw["pair_matches"] if raw else "unknown",
                             "semantic_verdict": "pending", "review_notes": ""})
            full.write(json.dumps({**row, "source_evidence": raw, "structural_flags": flags}, ensure_ascii=False) + "\n")
    report = {"schema": "act-p1-audit-v1", "data": str(args.data.resolve()), "seed": args.seed,
              "sampling": "uniform within each split/task; no length or quality filtering",
              "population": populations, "groups": {k: dict(v) for k, v in counts.items()},
              "semantic_review": "pending; source agreement is not translation correctness",
              "verdicts": ["ok", "mismatch", "uncertain", "pending"]}
    save(args.output / "report.json", report)
    print(f"[P1] {len(rows)} sampled rows; semantic review pending: {args.output / 'review.tsv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/downstream/ancient/train/final"))
    parser.add_argument("--raw", type=Path, default=Path("data/downstream/ancient/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_task < 1:
        parser.error("per-task must be positive")
    run(args)


if __name__ == "__main__":
    main()
