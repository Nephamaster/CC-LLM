"""Small, dependency-free checks for the source/data directory migration."""

import copy
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.data_factory.ancient_sft import common, pipeline, provenance, relocation
from scripts.data_factory.ancient_sft.repartition import filter_fingerprint, repartition


class RelocationTest(unittest.TestCase):
    def test_project_root_and_new_entrypoint(self):
        self.assertEqual(common.ROOT, Path(__file__).resolve().parents[1])
        result = subprocess.run([sys.executable, "-B", "-m", "scripts.data_factory.ancient_sft.run", "--help"],
                                cwd=common.ROOT, capture_output=True, text=True, check=True)
        self.assertIn("repartition", result.stdout)

    def test_only_documented_path_moves_are_normalized(self):
        old = "data/downstream/ancient/HistoryTrans/HistoryTrans-test.jsonl"
        new = "data/downstream/ancient/raw/HistoryTrans/HistoryTrans-test.jsonl"
        self.assertEqual(relocation.canonical_path(old), new)
        self.assertEqual(relocation.canonical_path(str(common.ROOT / old)), new)
        unrelated = "data/downstream/ancient/HistoryTrans-extra/test.jsonl"
        self.assertEqual(relocation.canonical_path(unrelated), unrelated)
        self.assertEqual(relocation.canonical_path("/external/HistoryTrans/test.jsonl"), "/external/HistoryTrans/test.jsonl")

    def test_inventory_relocation_keeps_content_and_file_identity(self):
        old = {"missing": [], "files": {"corpus": [{"path": "data/downstream/ancient/chinese-classical-corpus/corpus.jsonl",
                                                   "bytes": 7, "sha256": "same"}]}}
        new = copy.deepcopy(old)
        new["files"]["corpus"][0]["path"] = "data/downstream/ancient/raw/chinese-classical-corpus/corpus.jsonl"
        self.assertEqual(relocation.canonical_inventory(old), relocation.canonical_inventory(new))
        new["files"]["corpus"][0]["sha256"] = "changed"
        self.assertNotEqual(relocation.canonical_inventory(old), relocation.canonical_inventory(new))
        new["files"]["corpus"][0].update(sha256="same", bytes=8)
        self.assertNotEqual(relocation.canonical_inventory(old), relocation.canonical_inventory(new))

    def test_obsolete_environment_pin_is_not_a_data_configuration_change(self):
        config = {"sources": {}, "benchmarks": {}, "max_length": 2048}
        self.assertEqual(relocation.canonical_config(config),
                         relocation.canonical_config({**config, "swift_commit": "old"}))
        self.assertNotEqual(relocation.canonical_config(config),
                            relocation.canonical_config({**config, "max_length": 1024}))

    def fixture(self, root):
        output = root / "data/downstream/ancient/train"
        prepared = output / "prepared"
        prepared.mkdir(parents=True)
        config = {"sources": {"corpus": "data/downstream/ancient/raw/chinese-classical-corpus/corpus.jsonl"},
                  "benchmarks": {}, "output_dir": str(output), "tokenizer_path": "models/char", "reviews": None,
                  "quotas": {"c2m": 38000}}
        old = copy.deepcopy(config)
        old["sources"]["corpus"] = "data/downstream/ancient/chinese-classical-corpus/corpus.jsonl"
        old["output_dir"] = "data/downstream/ancient_sft_v2"
        old["swift_commit"] = "obsolete-pin"
        inventory = {"missing": [], "files": {"corpus": [{"path": str(root / old["sources"]["corpus"]),
                                                         "bytes": 7, "sha256": "same"}]}}
        current_inventory = copy.deepcopy(inventory)
        current_inventory["files"]["corpus"][0]["path"] = str(root / config["sources"]["corpus"])
        with sqlite3.connect(prepared / "candidates.sqlite") as db:
            db.execute("CREATE TABLE eligible(id TEXT PRIMARY KEY,task TEXT,source TEXT,split TEXT,book TEXT,doc TEXT,"
                       "cluster TEXT,priority TEXT,payload TEXT,input_hash TEXT)")
        db.close()
        manifest = {"diagnostic": False, "config": old, "inventory": inventory, "statistics": {},
                    "partition": {"strategy": "chapter_stratified_holdout_v2"},
                    "implementation_sha256": provenance.profile()["candidate_reuse"]["implementations"][-1],
                    "database_sha256": common.file_hash(prepared / "candidates.sqlite"),
                    "build_dependencies": {"opencc-python-reimplemented": "fixture"}}
        common.write_json(prepared / "manifest.json", manifest)
        return config, prepared, current_inventory

    def test_finalize_accepts_moved_prepared_without_rewriting_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, prepared, inventory = self.fixture(root)
            original = (prepared / "manifest.json").read_bytes()
            with patch.object(relocation, "ROOT", root), patch.object(pipeline, "input_inventory", return_value=inventory), \
                    patch.object(pipeline, "select_records", side_effect=RuntimeError("reached sampling")):
                with self.assertRaisesRegex(RuntimeError, "reached sampling"):
                    pipeline.finalize(config, encoder=object())
            self.assertEqual((prepared / "manifest.json").read_bytes(), original)

    def test_relocated_prepared_still_rejects_content_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, inventory = self.fixture(root)
            inventory["files"]["corpus"][0]["sha256"] = "changed"
            with patch.object(relocation, "ROOT", root), patch.object(pipeline, "input_inventory", return_value=inventory):
                with self.assertRaisesRegex(RuntimeError, "input files changed"):
                    pipeline.finalize(config, encoder=object())

    def test_old_partition_strategy_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, prepared, _ = self.fixture(root)
            path = prepared / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest.pop("partition")
            common.write_json(path, manifest)
            with self.assertRaisesRegex(RuntimeError, "requires v2 chapter partitions"):
                pipeline.finalize(config, encoder=object())

    def test_repartition_accepts_relocated_inputs(self):
        module = "scripts.data_factory.ancient_sft.repartition"
        self.assertEqual(filter_fingerprint(), provenance.profile()["candidate_reuse"]["filter_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, prepared, inventory = self.fixture(root)
            config["output_dir"] = str(root / "data/downstream/ancient/train/rebuild")
            with patch.object(relocation, "ROOT", root), patch(module + ".input_inventory", return_value=inventory), \
                    patch(module + ".importlib.metadata.version", return_value="fixture"), \
                    patch(module + ".partition_candidates", return_value={"strategy": "fixture"}):
                result = repartition(config, str(prepared))
            self.assertEqual(result["inventory"], inventory)
            self.assertEqual(json.loads((prepared / "manifest.json").read_text())["config"]["sources"]["corpus"],
                             "data/downstream/ancient/chinese-classical-corpus/corpus.jsonl")


if __name__ == "__main__":
    unittest.main()
