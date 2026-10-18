"""Isolated prepare/finalize stages with fail-closed input and release checks."""

from __future__ import annotations

import importlib.metadata
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from .common import (
    TASKS, Groups, Normalizer, digest, file_hash, input_inventory, read_jsonl,
    resolve, split_for, write_json, write_line,
)
from .overlap import BenchmarkIndex, CorpusLocator, NearIndex
from .sources import iter_candidates


def implementation_hash():
    return digest("".join(file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))))


def inspect_inputs(config):
    inventory = input_inventory(config)
    samples = {}
    for name in config["sources"]:
        if name not in inventory["files"]:
            continue
        path = Path(inventory["files"][name][0]["path"])
        _, row = next(read_jsonl(path), (None, {}))
        samples[name] = {"fields": sorted(row), "first_record_types": {k: type(v).__name__ for k, v in row.items()}}
    return {"ready": not inventory["missing"], "inventory": inventory, "schemas": samples,
            "evahan_included": False, "optional_tasks_require_review": True}


def read_reviews(config):
    result = {}
    if config.get("reviews"):
        for _, row in read_jsonl(resolve(config["reviews"])):
            if row.get("decision") not in {"allow", "reject"} or not row.get("reviewer") or not row.get("reason"):
                raise ValueError("review requires id, decision=allow/reject, reviewer, reason")
            if row["id"] in result:
                raise ValueError(f"duplicate review id: {row['id']}")
            result[row["id"]] = row
    return result


def prepare(config, *, limit=None, diagnostic=False):
    inventory = input_inventory(config)
    if inventory["missing"] and not diagnostic:
        raise FileNotFoundError("required inputs missing: " + "; ".join(inventory["missing"]))
    # Diagnostic mode may omit holdouts only; never silently omit a training source or benchmark.
    necessary = set(config["sources"]) - {"history_validation", "history_test"}
    necessary |= set(config["benchmarks"])
    absent = necessary - inventory["files"].keys()
    if absent:
        raise FileNotFoundError(f"missing required sources/benchmarks: {sorted(absent)}")
    root = resolve(config["output_dir"])
    destination = root / "prepared"
    destination.mkdir(parents=True, exist_ok=False)
    normalize, documents, parallels = Normalizer(), Groups(), Groups()
    index, reviews = BenchmarkIndex(inventory, normalize), read_reviews(config)
    locator = CorpusLocator(resolve(config["sources"]["corpus"]), normalize)
    stats, blocked = Counter(), set()
    db = sqlite3.connect(destination / "candidates.sqlite")
    db.executescript("""
        CREATE TABLE candidates (id TEXT PRIMARY KEY, task TEXT, source TEXT, node TEXT, doc TEXT,
                                 priority TEXT, payload TEXT);
        CREATE TABLE modern_groups (hash TEXT PRIMARY KEY, doc TEXT);
    """)
    near = NearIndex(db, config["near_threshold"])
    try:
        with (destination / "audit.jsonl").open("w", encoding="utf-8") as audit:
            for number, record in enumerate(iter_candidates(config, stats, limit), 1):
                canonical = normalize(record["ancient"])
                if not canonical:
                    stats["empty_comparison_text"] += 1
                    continue
                group = normalize(record["document_group"]) if record["document_group"] else ""
                node, matches = near.add(canonical, group)
                parallels.root(node)
                if not group:
                    known = sorted({doc for _, doc in matches if doc})
                    recovered = locator.find(canonical) if not known else []
                    if recovered:
                        known = sorted({normalize("book:" + book) for book, _ in recovered})
                        record["book"] = recovered[0][0]
                        record["metadata"]["era"] = recovered[0][1] if len(recovered) == 1 else "unknown"
                        record["metadata"]["recovered_books"] = [book for book, _ in recovered]
                    if not known:
                        stats["unresolved_history_group"] += 1
                        write_line(audit, {"id": record["id"], "reason": "unresolved_history_group", "record": record})
                        continue
                    group = known[0]
                    for other_group in known[1:]:
                        documents.union(group, other_group)
                    record["metadata"]["group_recovery"] = "corpus_containment" if recovered else "normalized_exact_or_verified_near"
                    db.execute("UPDATE near_texts SET doc=? WHERE id=? AND doc=''", (group, node))
                documents.root(group)
                for other, doc in matches:
                    parallels.union(node, other)
                    if doc:
                        documents.union(group, doc)
                # Shared modern translations also stay in one split, even with different ancient wording.
                if record["task"] in {"c2m", "m2c"}:
                    modern = normalize(record["modern"])
                    if modern == canonical:
                        stats["identical_normalized_translation"] += 1
                        continue
                    modern_key = digest(modern)
                    previous = db.execute("SELECT doc FROM modern_groups WHERE hash=?", (modern_key,)).fetchone()
                    if previous:
                        documents.union(group, previous[0])
                    else:
                        db.execute("INSERT INTO modern_groups VALUES (?,?)", (modern_key, group))
                record["canonical_ancient_sha256"] = digest(canonical)
                record["pair_group"] = digest(canonical + "\0" + normalize(record["modern"]))
                record["metadata"]["script"] = normalize.script(record["ancient"])
                status, evidence = index.check(record)
                stats[f"benchmark/{status}"] += 1
                review = reviews.get(record["id"])
                if status == "exact":
                    blocked.add(node)
                    write_line(audit, {"id": record["id"], "reason": "benchmark_exact", "evidence": evidence,
                                       "record": record})
                    continue
                if review and review["decision"] == "reject":
                    stats["review_rejected"] += 1
                    write_line(audit, {"id": record["id"], "reason": "review_rejected", "review": review})
                    continue
                if (status == "possible" or record["review_required"]) and not review:
                    if status == "possible":
                        blocked.add(node)
                    stats["pending_review"] += 1
                    write_line(audit, {"id": record["id"], "reason": "possible_overlap" if status == "possible"
                                       else "source_quality_review", "evidence": evidence, "record": record})
                    continue
                record["review"] = review
                record["document_group"] = group
                db.execute("INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?)", (
                    record["id"], record["task"], record["source_id"], node, group,
                    digest(f"{config['seed']}:{record['id']}"), json.dumps(record, ensure_ascii=False),
                ))
                if number % 5000 == 0:
                    db.commit()
                    print(f"prepared {number} candidate records", flush=True)
        blocked = {parallels.root(node) for node in blocked}
        db.executescript("""
            CREATE TABLE eligible (id TEXT PRIMARY KEY, task TEXT, source TEXT, split TEXT,
                                   book TEXT, doc TEXT, cluster TEXT, priority TEXT, payload TEXT,
                                   input_hash TEXT);
            CREATE UNIQUE INDEX unique_task_cluster ON eligible(task,cluster);
            CREATE UNIQUE INDEX unique_task_input ON eligible(task,input_hash);
        """)
        # Source preference is explicit: Gujilab first; HistoryTrans contributes only new clusters.
        cursor = db.execute("SELECT id,task,source,node,doc,priority,payload FROM candidates "
                            "ORDER BY CASE source WHEN 'gujilab' THEN 0 ELSE 1 END, priority")
        for key, task, source, node, doc, priority, payload in cursor:
            cluster = parallels.root(node)
            if cluster in blocked:
                stats["blocked_parallel_cluster"] += 1
                continue
            document = documents.root(doc)
            split = split_for(document, config)
            record = json.loads(payload)
            record.update({"document_group": document, "near_group": cluster, "split": split})
            inserted = db.execute("INSERT OR IGNORE INTO eligible VALUES (?,?,?,?,?,?,?,?,?,?)", (
                key, task, source, split, normalize(record["book"]), document, cluster, priority,
                json.dumps(record, ensure_ascii=False), digest(normalize(record["messages"][0]["content"])),
            )).rowcount
            stats[f"eligible/{split}/{task}/{source}" if inserted else "duplicate_task_cluster"] += 1
        db.commit()
        manifest = {
            "stage": "prepare", "config": config, "config_sha256": digest(json.dumps(config, sort_keys=True)),
            "implementation_sha256": implementation_hash(), "inventory": inventory,
            "diagnostic": bool(diagnostic or limit is not None), "limit_per_source": limit,
            "evahan_included": False, "benchmark_counts": index.counts,
            "statistics": dict(stats), "database_sha256": file_hash(destination / "candidates.sqlite"),
            "source_provenance": config.get("provenance", {}),
            "build_dependencies": {name: importlib.metadata.version(name) for name in
                                   ("opencc-python-reimplemented", "datasketch", "numpy", "PyYAML")},
            "python": sys.version, "sqlite": sqlite3.sqlite_version,
            "unknown_provenance": [name for name in config["sources"] if name not in config.get("provenance", {})],
            "grouping": "conservative whole-book; dictionary by headword; corpus containment recovery; unresolved HistoryTrans quarantined",
            "near_detection": "64-permutation 5-character MinHash, 16x4 LSH; verified Jaccard; approximate recall",
            "benchmark_detection": "normalized exact plus sampled-anchor shared >=24-character spans; approximate recall, not semantic equivalence",
            "cpt_contamination_audited": False,
        }
        write_json(destination / "manifest.json", manifest)
        return manifest
    finally:
        db.close()


def distribution(values):
    if not values:
        return {"mean": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {"mean": sum(values) / len(values), "p95": ordered[math.ceil(.95 * len(values)) - 1],
            "p99": ordered[math.ceil(.99 * len(values)) - 1], "max": ordered[-1]}


def finalize(config, encoder=None):
    from .provenance import verify_prepared_implementation

    root = resolve(config["output_dir"])
    prepared = root / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest["diagnostic"] or manifest["inventory"]["missing"]:
        raise RuntimeError("diagnostic/incomplete preparation cannot produce training data")
    if manifest["config"] != config:
        raise RuntimeError("configuration or implementation changed; prepare in a new output directory")
    implementation_check = verify_prepared_implementation(manifest["implementation_sha256"], implementation_hash())
    if manifest["inventory"] != input_inventory(config):
        raise RuntimeError("input files changed since preparation")
    if file_hash(prepared / "candidates.sqlite") != manifest["database_sha256"]:
        raise RuntimeError("prepared database changed")
    if encoder is None:
        from .encoding import SwiftEncoder

        encoder = SwiftEncoder(config)
    final = root / "final"
    final.mkdir(exist_ok=False)
    db = sqlite3.connect(f"file:{prepared / 'candidates.sqlite'}?mode=ro", uri=True)
    stats, metrics = Counter(), {}
    selected_groups, selected_ancient = {}, {}
    try:
        for split in ("train", "dev"):
            with (final / f"{split}.jsonl").open("w", encoding="utf-8") as output, (
                final / f"{split}.metadata.jsonl"
            ).open("w", encoding="utf-8") as metadata:
                position = 0
                for task in TASKS:
                    target = config["quotas"][task]
                    if split == "dev":
                        target = math.ceil(target * config["dev_fraction"] / (1 - config["dev_fraction"]))
                    selected, book_counts = [], Counter()
                    # c2m first reserves a complementary HistoryTrans budget, then fills with Gujilab.
                    passes = [(None, target)]
                    if task == "c2m" and split == "train":
                        passes = [("historytrans", min(config["history_target"], target)), ("gujilab", target)]
                    eligible_count = db.execute("SELECT COUNT(*) FROM eligible WHERE split=? AND task=?",
                                                (split, task)).fetchone()[0]
                    for source, capacity in passes:
                        query = "SELECT book,payload FROM eligible WHERE split=? AND task=?"
                        args = [split, task]
                        if source:
                            query += " AND source=?"
                            args.append(source)
                        query += " ORDER BY priority"
                        added = 0
                        for book, payload in db.execute(query, args):
                            if len(selected) >= target or added >= capacity:
                                break
                            if task not in {"dictionary"} and book_counts[book] >= max(1, math.ceil(target * config["max_book_fraction"])):
                                stats[f"book_cap/{split}/{task}"] += 1
                                continue
                            record = json.loads(payload)
                            lengths = encoder(record["messages"])
                            if lengths["sequence_tokens"] > config["max_length"]:
                                stats[f"overlength/{split}/{task}"] += 1
                                continue
                            if lengths["assistant_loss_tokens"] <= 0:
                                raise RuntimeError("empty assistant loss")
                            for mapping, group in ((selected_groups, record["document_group"]),
                                                   (selected_ancient, record["canonical_ancient_sha256"])):
                                if group in mapping and mapping[group] != split:
                                    raise RuntimeError("train/dev leakage detected")
                                mapping[group] = split
                            record.update(lengths)
                            selected.append(record)
                            book_counts[book] += 1
                            added += 1
                    metrics[f"{split}/{task}"] = task_report(selected, target, eligible_count)
                    for record in selected:
                        position += 1
                        write_line(output, {"messages": record["messages"]})
                        write_line(metadata, {**record, "row_number": position,
                                              "messages_sha256": digest(json.dumps(record["messages"], ensure_ascii=False))})
        hashes = {path.name: file_hash(path) for path in final.glob("*.jsonl")}
        report = {
            "stage": "finalize", "execution_passed": True, "tasks": metrics,
            "quotas_passed": all(metrics[f"train/{task}"]["shortfall"] == 0 for task in TASKS),
            "core_quotas_passed": all(metrics[f"train/{task}"]["shortfall"] == 0 for task in TASKS[:3]),
            "ready_for_training": all(sum(v["selected"] for k, v in metrics.items() if k.startswith(split + "/"))
                                      > 0 for split in ("train", "dev")),
            "train_dev_group_overlap": 0, "train_dev_ancient_overlap": 0, "evahan_included": False,
            "encoding": encoder.info, "statistics": dict(stats), "output_sha256": hashes,
            "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
            "prepared_implementation_check": implementation_check,
            "finalization_implementation_sha256": implementation_hash(),
            "cpt_contamination_audited": False,
        }
        for split in ("train", "dev"):
            total = sum(v["assistant_loss_tokens"] for k, v in metrics.items() if k.startswith(split + "/"))
            for key, value in metrics.items():
                if key.startswith(split + "/"):
                    value["assistant_loss_fraction"] = value["assistant_loss_tokens"] / total if total else 0
        write_json(final / "report.json", report)
        write_json(final / "manifest.json", {"preparation": manifest, "finalization": report})
        return report
    finally:
        db.close()


def task_report(records, target, eligible_count):
    return {
        "eligible_candidates": eligible_count, "target": target, "selected": len(records),
        "shortfall": max(0, target - len(records)),
        "independent_document_groups": len({r["document_group"] for r in records}),
        "source_chapter_labels": len({r["metadata"].get("source_label") for r in records
                                      if r["metadata"].get("source_label")}),
        "sources": dict(Counter(r["source_id"] for r in records)),
        "books": dict(Counter(r["book"] for r in records)),
        "eras": dict(Counter(r["metadata"].get("era", "unknown") for r in records)),
        "scripts": dict(Counter(r["metadata"].get("script", "unknown") for r in records)),
        "sequence_tokens": distribution([r["sequence_tokens"] for r in records]),
        "assistant_token_lengths": distribution([r["assistant_loss_tokens"] for r in records]),
        "assistant_loss_tokens": sum(r["assistant_loss_tokens"] for r in records),
    }
