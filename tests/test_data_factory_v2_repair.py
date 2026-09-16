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
