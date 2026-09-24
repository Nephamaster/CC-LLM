"""Explicit acceptance preserves observed deficits and all available documents."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers

from scripts.data_factory.v2.config import BucketSpec, load_data_factory_config
from scripts.data_factory.v2.tokenization import tokenize_selected, finalize_dataset, _require_mixture


class AcceptExistingMixtureTest(unittest.TestCase):
    def prepare(self, root, target=1000):
        config = load_data_factory_config(Path("scripts/data_factory/configs/phase1.yaml"))
        token_map = root / "characters.json"
        token_map.write_text(json.dumps({"100": "罕"}))
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "罕": 1, "word": 2}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.save(str(root / "tokenizer.json"))
        (root / "config.json").write_text('{"eos_token_id": 0}')
        config = replace(config, corpus_root=root, tokenizer_path=root, target_tokens=target,
                         validation_tokens=10,
                         buckets=(BucketSpec("zh_general", 1, {"cci3_hq": .5, "wanjuan": .5}, "zh_general", {}),),
                         enhancement=replace(config.enhancement, token_ids_path=token_map))
        plan = {"plan_sha256": "a" * 64, "round_index": 1}
        rows = [{"id": str(i), "parent_doc_id": str(i), "text": "罕 " + "word " * 6,
                 "source": "cci3_hq", "language": "zh", "domain": "general", "tags": [],
                 "candidate_bucket": "zh_general", "estimated_tokens": 7, "sample_key": 2**64 - 1 - i}
                for i in range(4)]
        schema = pa.schema([
            ("id", pa.string()), ("parent_doc_id", pa.string()), ("text", pa.large_string()),
            ("source", pa.string()), ("language", pa.string()), ("domain", pa.string()),
            ("tags", pa.list_(pa.string())), ("candidate_bucket", pa.string()),
            ("estimated_tokens", pa.int64()), ("sample_key", pa.uint64()),
        ])
        selected = config.run_root / "selected" / ("a" * 16)
        selected.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), selected / "part.parquet")
        report = {"run_id": config.run_id, "plan_sha256": plan["plan_sha256"], "passed": False,
                  "enhancement": {"constraint_checks": {"single_source": True, "modern": True}},
                  "distribution": {"checks": {"zh_general/source/wanjuan": {"passed": False}}}}
        reports = config.run_root / "reports"
        reports.mkdir()
        (reports / "mixture_report.json").write_text(json.dumps(report))
        return config, plan

    def test_acceptance_keeps_all_tokens_and_reports_failed_quota(self):
        for target in (10, 1000):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                config, plan = self.prepare(Path(directory), target)
                with self.assertRaisesRegex(RuntimeError, "requires a passed mixture"):
                    tokenize_selected(config, plan)
                tokenize_selected(config, plan, accept_existing_mixture=True)
                final = finalize_dataset(config, plan, accept_existing_mixture=True)
                self.assertFalse(final["passed"])
                self.assertTrue(final["accepted_for_use"])
                self.assertTrue(final["execution_passed"])
                self.assertEqual(final["actual_train_tokens"], 14)
                self.assertEqual(final["actual_validation_tokens"], 14)
                self.assertEqual(final["train_documents"] + final["validation_documents"], 4)
                self.assertEqual(final["document_overlap"], 0)
                self.assertFalse(json.loads((config.run_root / "reports/mixture_report.json").read_text())["passed"])

    def test_acceptance_does_not_bypass_identity_or_nonquota_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            config, plan = self.prepare(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                _require_mixture(config, {"plan_sha256": "b" * 64}, True)
            path = config.run_root / "reports/mixture_report.json"
            report = json.loads(path.read_text())
            report["distribution"]["checks"]["attribute/classical_chinese"] = {"passed": False}
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "failed mixture constraint"):
                _require_mixture(config, plan, True)

    def test_acceptance_rejects_incomplete_tokenized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            config, plan = self.prepare(Path(directory))
            tokenize_selected(config, plan, accept_existing_mixture=True)
            path = config.run_root / "reports/tokenization_report.json"
            report = json.loads(path.read_text())
            report["total_tokens"] += 1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                finalize_dataset(config, plan, accept_existing_mixture=True)


if __name__ == "__main__":
    unittest.main()
