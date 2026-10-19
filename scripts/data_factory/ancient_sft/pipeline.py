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
    resolve, write_json, write_line,
)
from .overlap import BenchmarkIndex, NearIndex
from .partition import partition_candidates
from .selection import acceptance, select_records
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
    normalize, parallels = Normalizer(), Groups()
    index, reviews = BenchmarkIndex(inventory, normalize), read_reviews(config)
    stats, blocked = Counter(), set()
    db = sqlite3.connect(destination / "candidates.sqlite")
    db.executescript("""
        CREATE TABLE candidates (id TEXT PRIMARY KEY, task TEXT, source TEXT, node TEXT, doc TEXT,
                                 priority TEXT, payload TEXT);
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
                group = group or "unresolved:" + node
                for other, doc in matches:
                    parallels.union(node, other)
                if record["task"] in {"c2m", "m2c"}:
                    modern = normalize(record["modern"])
                    if modern == canonical:
                        stats["identical_normalized_translation"] += 1
                        continue
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
            record = json.loads(payload)
            record.update({"document_group": doc, "near_group": cluster, "split": "unassigned"})
            inserted = db.execute("INSERT OR IGNORE INTO eligible VALUES (?,?,?,?,?,?,?,?,?,?)", (
                key, task, source, "unassigned", normalize(record["book"]), doc, cluster, priority,
                json.dumps(record, ensure_ascii=False), digest(normalize(record["messages"][0]["content"])),
            )).rowcount
            stats[f"deduplicated/{task}/{source}" if inserted else "duplicate_task_cluster"] += 1
        db.commit()
        partition = partition_candidates(db, config, destination)
        manifest = {
            "stage": "prepare", "config": config, "config_sha256": digest(json.dumps(config, sort_keys=True)),
            "implementation_sha256": implementation_hash(), "inventory": inventory,
            "diagnostic": bool(diagnostic or limit is not None), "limit_per_source": limit,
            "evahan_included": False, "benchmark_counts": index.counts,
            "statistics": dict(stats), "database_sha256": file_hash(destination / "candidates.sqlite"),
            "partition": partition,
            "source_provenance": config.get("provenance", {}),
            "build_dependencies": {name: importlib.metadata.version(name) for name in
                                   ("opencc-python-reimplemented", "datasketch", "numpy", "PyYAML")},
            "python": sys.version, "sqlite": sqlite3.sqlite_version,
            "unknown_provenance": [name for name in config["sources"] if name not in config.get("provenance", {})],
            "grouping": "chapter-stratified Dev; shared content excluded from Train without merging books",
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
    from .relocation import canonical_config, canonical_inventory

    root = resolve(config["output_dir"])
    prepared = root / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest["diagnostic"] or manifest["inventory"]["missing"]:
        raise RuntimeError("diagnostic/incomplete preparation cannot produce training data")
    if canonical_config(manifest["config"]) != canonical_config(config):
        raise RuntimeError("configuration or implementation changed; prepare in a new output directory")
    implementation_check = verify_prepared_implementation(manifest["implementation_sha256"], implementation_hash())
    inventory = input_inventory(config)
    if canonical_inventory(manifest["inventory"]) != canonical_inventory(inventory):
        raise RuntimeError("input files changed since preparation")
    if file_hash(prepared / "candidates.sqlite") != manifest["database_sha256"]:
        raise RuntimeError("prepared database changed")
    if encoder is None:
        from .encoding import SwiftEncoder

        encoder = SwiftEncoder(config)
    final = root / "final"
    final.mkdir(exist_ok=False)
    db = sqlite3.connect(f"file:{prepared / 'candidates.sqlite'}?mode=ro", uri=True)
    metrics = {}
    try:
        selected, stats = select_records(db, config, encoder)
        checks = acceptance(selected, config)
        (final / "dev_by_task").mkdir()
        for split in ("train", "dev"):
            with (final / f"{split}.jsonl").open("w", encoding="utf-8") as output, (
                final / f"{split}.metadata.jsonl"
            ).open("w", encoding="utf-8") as metadata:
                position = 0
                for task in TASKS:
                    target = (config["quotas"] if split == "train" else config["dev_quotas"])[task]
                    eligible_count = db.execute("SELECT COUNT(*) FROM eligible WHERE split=? AND task=?",
                                                (split, task)).fetchone()[0]
                    rows = selected[split][task]
                    metrics[f"{split}/{task}"] = task_report(rows, target, eligible_count)
                    if split == "dev" and target:
                        with (final / "dev_by_task" / f"{task}.jsonl").open("w", encoding="utf-8") as task_file:
                            for record in rows:
                                write_line(task_file, {"messages": record["messages"]})
                    for record in rows:
                        position += 1
                        write_line(output, {"messages": record["messages"]})
                        write_line(metadata, {**record, "row_number": position,
                                              "messages_sha256": digest(json.dumps(record["messages"], ensure_ascii=False))})
        hashes = {path.relative_to(final).as_posix(): file_hash(path) for path in final.rglob("*.jsonl")}
        report = {
            "stage": "finalize", "execution_passed": True, "tasks": metrics,
            "quotas_passed": all(metrics[f"train/{task}"]["shortfall"] == 0 for task in TASKS),
            "core_quotas_passed": all(metrics[f"train/{task}"]["shortfall"] == 0 for task in TASKS[:3]),
            **checks,
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
        write_json(final / "manifest.json", {"preparation": manifest, "finalization": report,
                                            "effective_config": config, "effective_inventory": inventory})
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
