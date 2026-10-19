"""Small, dependency-free checks for the source/data directory migration."""

import copy
import json
import shutil
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

    def test_known_code_move_is_verified_but_logic_changes_are_not(self):
        expected = provenance.profile()["relocation"]
        self.assertEqual(relocation.preparation_contract(provenance.MODULE_DIR), expected["preparation_contract_sha256"])
        self.assertEqual(provenance.verify_prepared_implementation(expected["implementation_sha256"], "new"),
                         "verified_directory_relocation")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for path in provenance.MODULE_DIR.glob("*.py"):
                shutil.copy2(path, target / path.name)
            with (target / "partition.py").open("a") as handle:
                handle.write("\nUNRELATED_CHANGE = True\n")
            with self.assertRaisesRegex(RuntimeError, "implementation changed"):
                provenance.verify_prepared_implementation(expected["implementation_sha256"], "new", target)

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
        inventory = {"missing": [], "files": {"corpus": [{"path": str(root / old["sources"]["corpus"]),
                                                         "bytes": 7, "sha256": "same"}]}}
        current_inventory = copy.deepcopy(inventory)
        current_inventory["files"]["corpus"][0]["path"] = str(root / config["sources"]["corpus"])
        with sqlite3.connect(prepared / "candidates.sqlite") as db:
            db.execute("CREATE TABLE eligible(id TEXT PRIMARY KEY,task TEXT,source TEXT,split TEXT,book TEXT,doc TEXT,"
                       "cluster TEXT,priority TEXT,payload TEXT,input_hash TEXT)")
        db.close()
        manifest = {"diagnostic": False, "config": old, "inventory": inventory, "statistics": {},
                    "implementation_sha256": provenance.profile()["relocation"]["implementation_sha256"],
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
