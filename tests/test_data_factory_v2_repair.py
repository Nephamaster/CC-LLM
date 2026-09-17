"""Regression checks for overlapping eligibility and the complete finalization path."""

import json
import tempfile
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers

from scripts.data_factory.v2.config import BucketSpec, load_data_factory_config
from scripts.data_factory.v2.sampling import eligible_buckets, build_plan
from scripts.data_factory.v2.mixture import build_mixture
from scripts.data_factory.v2.tokenization import tokenize_selected, finalize_dataset
from scripts.data_factory.v2.dedup import minhash_dimensions
from scripts.data_factory.v2.candidate import CandidateSelector
from scripts.data_factory.v2.cache import CanonicalSourceReader
from scripts.data_factory.v2.documents import RawRecord
from scripts.data_factory.v2.runtime import build_executor
from datatrove.data import Document


class PipelineRepairTest(unittest.TestCase):
    def test_scan_budget_includes_ordinary_enhancement_reservation(self):
        config = load_data_factory_config(Path("scripts/data_factory/configs/phase1.yaml"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                config, corpus_root=root, target_tokens=1000, validation_tokens=0,
                buckets=(BucketSpec("zh_general", .8, {"cci3_hq": 1}, "zh_general", {}),
                         BucketSpec("new_char_enhancement", .2, {"cci3_hq": 1}, "new_char_coverage", {})),
            )
            calibration = {"calibration_sha256": "a" * 64, "sources": {"cci3_hq": {
                "tokens_per_document": 10,
                "cache_files": [{"path": str(root / f"cache-{i}.parquet"), "rows": 100} for i in range(24)],
                "buckets": {"zh_general": {"token_rate": .1}, "new_char_enhancement": {"token_rate": 1}},
            }}}
            plan = build_plan(config, calibration)
        self.assertTrue(plan["passed"], plan["shortfalls"])
        self.assertEqual(plan["selected_estimated_tokens"], 12000)
        self.assertEqual(plan["sampling_rates"]["zh_general"]["cci3_hq"], 1)

    def test_phase2_redistribution_fits_constrained_capacity(self):
        config = load_data_factory_config(Path("scripts/data_factory/configs/phase2.yaml"))
        capacities = {
            "cci3_hq": 34960166584, "fineweb_edu_chinese": 42064872424,
            "wanjuan": 198828805843, "chinese_cosmopedia": 44780462147,
            "wikipedia_zh": 1262137126, "fineweb_zhtw": 49369247389,
            "wikisource": 1291301020, "ect_krp": 4944880,
            "fineweb_edu_english": 9730970812, "fineweb2_multilingual": 25024730294,
            "the_stack_v3": 4145386947, "openwebmath": 13388716984, "peS2o": 51749118464,
        }
        # Capacity stress scenario, not a substitute for server Calibration:
        # Wikipedia yield follows the reported sampling rate; other knowledge
        # yields use conservative scenario values with observed corpus sizes.
        rates = {
            ("wikipedia_zh", "zh_knowledge"): .5796155918214336,
            ("wikipedia_zh", "new_char_enhancement"): .835,
            ("fineweb_zhtw", "zh_knowledge"): .8,
            ("fineweb_zhtw", "new_char_enhancement"): .99,
            ("wikisource", "zh_knowledge"): .8,
            ("wikisource", "new_char_enhancement"): .89,
            ("chinese_cosmopedia", "zh_knowledge"): .8,
            ("chinese_cosmopedia", "new_char_enhancement"): .01648,
            ("cci3_hq", "zh_general"): .744,
            ("cci3_hq", "new_char_enhancement"): .12896,
            ("fineweb_edu_chinese", "zh_general"): .55,
            ("fineweb_edu_chinese", "english_multilingual_mixed"): .385,
            ("fineweb_edu_chinese", "new_char_enhancement"): .05827,
            ("wanjuan", "zh_general"): .779,
            ("wanjuan", "new_char_enhancement"): .0581,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(config, corpus_root=root)
            calibration = {"calibration_sha256": "a" * 64, "sources": {
                source: {"tokens_per_document": 1,
                         "cache_files": [{"path": str(root / f"{source}-{i}.parquet"), "rows": capacity // 20}
                                         for i in range(20)],
                         "buckets": {b.name: {"token_rate": rates.get((source, b.name), 1)}
                                     for b in config.buckets}}
                for source, capacity in capacities.items()
            }}
            plan = build_plan(config, calibration)
        self.assertTrue(plan["passed"], plan["shortfalls"])
        self.assertTrue(all(rate <= 1 for values in plan["sampling_rates"].values() for rate in values.values()))
        self.assertEqual(config.bucket_tokens["zh_knowledge"], 2500000000)
        self.assertEqual(plan["source_targets"]["zh_knowledge"]["ect_krp"], 750750)
        self.assertEqual(plan["source_targets"]["new_char_enhancement"]["ect_krp"], 500500)

    def test_candidate_executor_serializes_bucket_densities(self):
        config = load_data_factory_config(Path("scripts/data_factory/configs/phase1.yaml"))
        plan = {"plan_sha256": "a" * 64, "sampling_rates": {
            "zh_general": {"cci3_hq": 1}, "mixed_zh_en": {"cci3_hq": 1},
        }}
        calibration = {"sources": {"cci3_hq": {
            "tokens_per_character": 1,
            "buckets": {"zh_general": {"tokens_per_character": 2},
                        "mixed_zh_en": {"tokens_per_character": 0}},
        }}}
        with patch("scripts.data_factory.v2.candidate.load_new_characters", return_value=frozenset("罕")):
            selector = CandidateSelector(config, plan, calibration)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executor = build_executor(pipeline=[selector], logging_dir=root,
                                      job_name="candidate", executor="local", tasks=1, workers=1)
            executor.save_executor_as_json()
            saved = json.loads((root / "executor.json").read_text())
        self.assertEqual(saved["pipeline"][0]["bucket_densities"],
                         {"cci3_hq": {"zh_general": 2.0}})
        docs = [Document(id="ordinary", text="中文文本", metadata={
                    "source": "cci3_hq", "language": "zh", "domain": "general", "char_count": 4}),
                Document(id="mixed", text="中文ABC", metadata={
                    "source": "cci3_hq", "language": "zh_en_mixed", "domain": "general", "char_count": 5})]
        rows = list(selector.run(iter(docs)))
        self.assertEqual([row.metadata["estimated_tokens"] for row in rows], [8, 5])

    def test_resume_rejects_changed_task_sharding(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = dict(pipeline=[], logging_dir=Path(directory), job_name="test", executor="local", workers=1)
            build_executor(**arguments, tasks=2)
            with self.assertRaisesRegex(ValueError, "task sharding changed"):
                build_executor(**arguments, tasks=3)

    def test_stack_rejected_child_does_not_drop_remaining_repository(self):
        config = load_data_factory_config(Path("scripts/data_factory/configs/phase1.yaml"))
        source = config.source_registry.sources["the_stack_v3"]
        files = [{"content_id": str(i), "content": content, "file_path": f"{i}.py",
                  "license_type": "permissive", "detected_licenses": ["MIT"], "is_vendor": False}
                 for i, content in enumerate(["x", "print('a valid source file containing enough characters')"])]
        raw = RawRecord({"repo_id": 1, "files": files}, Path("stack.parquet"), 0)
        reader = CanonicalSourceReader(source, [Path("stack.parquet")])
        with patch("scripts.data_factory.v2.cache.iter_raw_records", return_value=iter([raw])):
            rows = list(reader.run())
        self.assertEqual(len(rows), 1)
        self.assertIn("valid source", rows[0].text)

    def test_candidate_union_emits_once_and_preserves_ordinary_bucket(self):
        config = load_data_factory_config(Path("scripts/data_factory/configs/phase1.yaml"))
        plan = {"plan_sha256": "a" * 64, "sampling_rates": {
            "zh_general": {"cci3_hq": 0}, "new_char_enhancement": {"cci3_hq": 1},
        }}
        calibration = {"sources": {"cci3_hq": {"tokens_per_character": 1}}}
        with patch("scripts.data_factory.v2.candidate.load_new_characters", return_value=frozenset("罕")):
            selector = CandidateSelector(config, plan, calibration)
        doc = Document(id="one", text="罕见文字", metadata={"source": "cci3_hq", "language": "zh",
                       "domain": "general", "char_count": 4})
        rows = list(selector.run(iter([doc])))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].metadata["candidate_bucket"], "zh_general")

    def test_minhash_threshold_changes_signature_dimensions(self):
        natural = minhash_dimensions({"num_buckets": 14, "threshold": .8})
        code = minhash_dimensions({"num_buckets": 14, "threshold": .85})
        self.assertGreater(code[1], natural[1])
        self.assertAlmostEqual((1 / natural[0]) ** (1 / natural[1]), .8, delta=.01)
        self.assertAlmostEqual((1 / code[0]) ** (1 / code[1]), .85, delta=.01)

    def test_phase2_knowledge_keeps_enhancement_eligibility(self):
        config = load_data_factory_config(Path("scripts/data_factory/configs/phase2.yaml"))
        row = {"source": "wikipedia_zh", "language": "zh", "domain": "knowledge"}
        self.assertEqual(eligible_buckets(config, "罕见", row, frozenset("罕")),
                         ["zh_knowledge", "new_char_enhancement"])
        row["source"] = "fineweb_edu_chinese"
        self.assertEqual(eligible_buckets(config, "罕见", row, frozenset("罕")),
                         ["zh_general", "new_char_enhancement"])

    def test_rejects_source_weight_above_cap(self):
        path = Path("scripts/data_factory/configs/phase1.yaml").resolve()
        raw = yaml.safe_load(path.read_text())
        raw["repo_root"] = str(Path.cwd())
        raw["source_registry"] = str(path.parent / "sources.yaml")
        raw["new_char_enhancement"]["constraints"]["single_source_max_fraction"] = 0.3
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "phase.yaml"
            config_path.write_text(yaml.safe_dump(raw))
            with self.assertRaisesRegex(ValueError, "source_weights exceed"):
                load_data_factory_config(config_path)

    def test_mixture_tokenize_finalize_with_uint64_and_soft_coverage(self):
        config = load_data_factory_config(Path("scripts/data_factory/configs/phase1.yaml"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_map = root / "new_hanzi_token_ids.json"
            token_map.write_text(json.dumps({"1": "罕", "2": "缺"}))
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "罕": 1, "word": 2}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            tokenizer.save(str(root / "tokenizer.json"))
            (root / "config.json").write_text('{"eos_token_id": 0}')
            config = replace(
                config, corpus_root=root, tokenizer_path=root, target_tokens=200,
                validation_tokens=20,
                buckets=(BucketSpec("zh_general", .5, {"cci3_hq": 1.0}, "zh_general", {}),
                         BucketSpec("new_char_enhancement", .5, {"cci3_hq": 1.0}, "new_char_coverage", {})),
                candidate_priority=("new_char_enhancement", "zh_general"),
                enhancement=replace(config.enhancement, token_ids_path=token_map,
                                    constraints={"single_source_max_fraction": 1.0}),
            )
            plan = {"plan_sha256": "a" * 64}
            input_dir = config.run_root / "decontaminated" / ("a" * 16)
            input_dir.mkdir(parents=True)
            rows = [{"id": str(i), "parent_doc_id": str(i), "text": "罕 " + "word " * 9,
                     "source": "cci3_hq", "language": "zh", "domain": "general", "tags": [],
                     "candidate_bucket": "zh_general", "estimated_tokens": 10,
                     "sample_key": 2**64 - 1 - i} for i in range(30)]
            schema = pa.schema([
                ("id", pa.string()), ("parent_doc_id", pa.string()), ("text", pa.large_string()),
                ("source", pa.string()), ("language", pa.string()), ("domain", pa.string()),
                ("tags", pa.list_(pa.string())), ("candidate_bucket", pa.string()),
                ("estimated_tokens", pa.int64()), ("sample_key", pa.uint64()),
            ])
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), input_dir / "input.parquet")
            with patch("scripts.data_factory.v2.mixture._feature_map", return_value={"罕": frozenset({"tone:1"}), "缺": frozenset()}):
                mixture = build_mixture(config, plan)
            self.assertTrue(mixture["passed"], mixture)
            self.assertFalse(mixture["enhancement"]["coverage_passed"])
            tokenized = tokenize_selected(config, plan)
            self.assertEqual(tokenized["total_tokens"], 220)
            with patch("scripts.data_factory.v2.tokenization.stable_fraction", return_value=0):
                final = finalize_dataset(config, plan)
            self.assertTrue(final["passed"], final)
            self.assertEqual(final["document_overlap"], 0)
            self.assertEqual(final["training_character_coverage"]["1"]["actual_characters"], 1)
            for bucket in ("zh_general", "new_char_enhancement"):
                self.assertEqual(final["train"][bucket]["actual_tokens"], 100)
            next_plan = {"plan_sha256": "b" * 64, "previous_plan_hashes": ["a" * 64]}
            shutil.copytree(config.run_root / "selected" / ("a" * 16), config.run_root / "selected" / ("b" * 16))
            mixture["plan_sha256"] = next_plan["plan_sha256"]
            (config.run_root / "reports" / "mixture_report.json").write_text(json.dumps(mixture))
            reused = tokenize_selected(config, next_plan)
            self.assertEqual(reused["reused_documents"], 22)
            self.assertEqual(reused["total_tokens"], 220)

    def test_incremental_plan_excludes_previous_cache_files(self):
        config = load_data_factory_config(Path("scripts/data_factory/configs/phase1.yaml"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(config, corpus_root=root, target_tokens=1000, validation_tokens=10,
                             buckets=(BucketSpec("new_char_enhancement", 1, {"cci3_hq": 1}, "new_char_coverage", {}),))
            calibration = {"calibration_sha256": "a" * 64, "sources": {"cci3_hq": {
                "tokens_per_document": 10,
                "cache_files": [{"path": str(root / f"cache-{i}.parquet"), "rows": 1000} for i in range(5)],
                "buckets": {"new_char_enhancement": {"token_rate": 1}},
            }}}
            first = build_plan(config, calibration)
            reports = config.run_root / "reports"
            reports.mkdir()
            (reports / "mixture_report.json").write_text(json.dumps({
                "plan_sha256": first["plan_sha256"],
                "source_shortfalls": {"new_char_enhancement": {"cci3_hq": 100}},
            }))
            second = build_plan(config, calibration, round_index=1)
            self.assertTrue(second["passed"])
            self.assertEqual(second["source_targets"]["new_char_enhancement"]["cci3_hq"], 100)
            self.assertFalse({x["path"] for x in first["selected_files"]} & {x["path"] for x in second["selected_files"]})


if __name__ == "__main__":
    unittest.main()
