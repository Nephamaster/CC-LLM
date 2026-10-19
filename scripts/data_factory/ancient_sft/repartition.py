"""Reuse only hash-verified, previously admitted candidates in a new data version."""

import ast
import importlib.metadata
import json
import sqlite3

from .common import digest, file_hash, input_inventory, resolve, write_json
from .partition import partition_candidates
from .provenance import MODULE_DIR, profile
from .relocation import canonical_config, canonical_inventory


def filter_fingerprint():
    selected = {}
    for filename, names in (("common.py", {"clean", "unpunctuated", "Normalizer", "PUNCTUATION"}),
                            ("overlap.py", {"BenchmarkIndex", "NearIndex"})):
        source = (MODULE_DIR / filename).read_text(encoding="utf-8")
        pieces = []
        for node in ast.parse(source).body:
            name = getattr(node, "name", None)
            if isinstance(node, ast.Assign):
                name = getattr(node.targets[0], "id", None)
            if name in names:
                pieces.append(ast.get_source_segment(source, node))
        selected[filename] = "\n".join(pieces)
    selected["sources.py"] = (MODULE_DIR / "sources.py").read_text(encoding="utf-8")
    return digest(json.dumps(selected, sort_keys=True))


def repartition(config, from_prepared):
    from .pipeline import implementation_hash

    parent = resolve(from_prepared)
    manifest_file = parent / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    trusted = profile()["candidate_reuse"]
    if manifest.get("diagnostic") or manifest["inventory"]["missing"]:
        raise RuntimeError("diagnostic/incomplete candidates cannot be reused")
    if manifest["implementation_sha256"] not in {*trusted["implementations"], implementation_hash()}:
        raise RuntimeError("unknown candidate implementation; run a fresh prepare")
    if filter_fingerprint() != trusted["filter_sha256"]:
        raise RuntimeError("candidate filtering contract changed; run a fresh prepare")
    selectable = {"version", "output_dir", "seed", "quotas", "history_target", "max_book_fraction",
                  "dev_fraction", "dev_quotas", "dev_min_books", "dev_min_groups", "dev_max_book_fraction",
                  "dev_pool_factor", "punctuation_loss_fraction", "punctuation_max_samples"}
    old_config, current_config = canonical_config(manifest["config"]), canonical_config(config)
    for key in set(old_config) | set(current_config):
        if key not in selectable and old_config.get(key) != current_config.get(key):
            raise RuntimeError(f"non-selection configuration changed: {key}")
    inventory = input_inventory(config)
    if canonical_inventory(inventory) != canonical_inventory(manifest["inventory"]):
        raise RuntimeError("inputs or exclusion sets changed; run a fresh prepare")
    if manifest["build_dependencies"]["opencc-python-reimplemented"] != importlib.metadata.version("opencc-python-reimplemented"):
        raise RuntimeError("OpenCC normalization version changed; run a fresh prepare")
    source_path = parent / "candidates.sqlite"
    if file_hash(source_path) != manifest["database_sha256"]:
        raise RuntimeError("prepared database changed")
    destination = resolve(config["output_dir"]) / "prepared"
    destination.mkdir(parents=True, exist_ok=False)
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    db = sqlite3.connect(destination / "candidates.sqlite")
    try:
        schema = source.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='eligible'").fetchone()
        if not schema:
            raise RuntimeError("missing admitted candidate table")
        columns = [r[1] for r in source.execute("PRAGMA table_info(eligible)")]
        if columns != ["id", "task", "source", "split", "book", "doc", "cluster", "priority", "payload", "input_hash"]:
            raise RuntimeError("unsupported admitted candidate schema")
        db.execute(schema[0])
        # Do not read the raw candidates table: it can still contain blocked duplicate clusters.
        db.executemany("INSERT INTO eligible VALUES (?,?,?,?,?,?,?,?,?,?)", source.execute("SELECT * FROM eligible"))
        db.commit()
        partition = partition_candidates(db, config, destination)
        result = {**manifest, "stage": "prepare", "config": config,
                  "inventory": inventory,
                  "config_sha256": digest(json.dumps(config, sort_keys=True)),
                  "implementation_sha256": implementation_hash(), "partition": partition,
                  "grouping": "chapter-stratified Dev; cross-boundary samples excluded",
                  "statistics": {}, "upstream_statistics": manifest["statistics"],
                  "database_sha256": file_hash(destination / "candidates.sqlite"),
                  "reused_from": {"prepared": str(parent), "manifest_sha256": file_hash(manifest_file),
                                  "database_sha256": manifest["database_sha256"],
                                  "table": "eligible", "restores_previously_removed_records": False}}
        write_json(destination / "manifest.json", result)
        return result
    finally:
        db.close()
        source.close()
