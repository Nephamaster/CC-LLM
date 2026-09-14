from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data_factory.v2.config import load_data_factory_config
from scripts.data_factory.v2.documents import source_cache_id
from scripts.data_factory.v2.sampling import (
    _bounded_sample,
    assign_bucket,
    build_plan,
    calibrate,
)


CONFIG = Path("scripts/data_factory/configs/phase1.yaml")


def metadata(*, source: str, language: str, domain: str, tags: list[str] | None = None):
    return {
        "source": source,
        "language": language,
        "domain": domain,
        "tags": tags or [],
        "char_count": 100,
        "hanzi_count": 80 if language.startswith("zh") else 0,
        "latin_count": 20 if language != "zh" else 0,
    }


class DataFactoryV2SamplingTest(unittest.TestCase):
    def test_bucket_assignment_uses_configured_priority(self) -> None:
        config = load_data_factory_config(CONFIG)
        new_chars = frozenset({"罕"})

        self.assertEqual(
            assign_bucket(
                config,
                "这是包含罕见字符的自然中文上下文。",
                metadata(source="cci3_hq", language="zh", domain="general"),
                new_chars,
            ),
            "new_char_enhancement",
        )
        self.assertEqual(
            assign_bucket(
                config,
                "这是 API client 的中文使用说明 https://example.com",
                metadata(source="cci3_hq", language="zh_en_mixed", domain="general", tags=["mixed"]),
                new_chars,
            ),
            "mixed_zh_en",
        )
        self.assertEqual(
            assign_bucket(
                config,
                "def main():\n    return 1",
                metadata(source="the_stack_v2", language="en", domain="code"),
                new_chars,
            ),
            "specialized",
        )

    def test_bounded_calibration_sample_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files: list[tuple[Path, int]] = []
            for file_index in range(2):
                rows = []
                for row_index in range(20):
                    rows.append(
                        {
                            "id": f"{file_index}-{row_index}",
                            "text": "这是一条用于校准的中文样本文本。",
                            "parent_doc_id": f"{file_index}-{row_index}",
                            "source": "cci3_hq",
                            "subset": None,
                            "source_path": "sample.jsonl",
                            "revision": None,
                            "license": "Apache-2.0",
                            "url": None,
                            "language": "zh",
                            "domain": "general",
                            "char_count": 16,
                            "hanzi_count": 15,
                            "latin_count": 0,
                            "digit_count": 0,
                            "quality_prior": None,
                            "tags": [],
                            "metadata_json": "{}",
                        }
                    )
                path = Path(directory) / f"part-{file_index}.parquet"
                pq.write_table(pa.Table.from_pylist(rows), path)
                files.append((path, len(rows)))

            first, stats = _bounded_sample(
                files,
                source_name="cci3_hq",
                seed=7,
                target=10,
                max_files=2,
                scan_multiplier=2,
            )
            second, _ = _bounded_sample(
                files,
                source_name="cci3_hq",
                seed=7,
                target=10,
                max_files=2,
                scan_multiplier=2,
            )

        self.assertEqual(stats["scanned_documents"], 20)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])

    def test_plan_selects_only_required_cache_files(self) -> None:
        config = load_data_factory_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            config = replace(config, corpus_root=Path(directory))
            active_sources = {
                source
                for bucket in config.buckets
                for source in bucket.source_weights
            }
            calibration: dict = {"calibration_sha256": "a" * 64, "sources": {}}
            for source in active_sources:
                calibration["sources"][source] = {
                    "tokens_per_document": 100.0,
                    "tokens_per_character": 1.0,
                    "cache_files": [
                        {
                            "path": str(Path(directory) / f"{source}-{index}.parquet"),
                            "rows": 10_000_000,
                        }
                        for index in range(5)
                    ],
                    "buckets": {
                        bucket.name: {"token_rate": 0.20}
                        for bucket in config.buckets
                    },
                }

            plan = build_plan(config, calibration)

        total_files = sum(
            len(value["cache_files"])
            for value in calibration["sources"].values()
        )
        self.assertTrue(plan["passed"])
        self.assertGreater(plan["selected_file_count"], 0)
        self.assertLess(plan["selected_file_count"], total_files)


    def test_calibration_uses_real_tokenizer_on_bounded_cache(self) -> None:
        config = load_data_factory_config(CONFIG, require_tokenizer=True)
        source = config.source_registry.sources["cci3_hq"]
        manifest_sha = "b" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                config,
                corpus_root=root,
                calibration=replace(
                    config.calibration,
                    documents_per_source=10,
                    max_files_per_source=1,
                    scan_multiplier=2,
                    batch_size=4,
                ),
                hashes=replace(config.hashes, source_manifest="c" * 64),
            )
            cache_dir = root / "cache" / source.name / source_cache_id(source, manifest_sha)
            cache_dir.mkdir(parents=True)
            rows = []
            for index in range(30):
                rows.append(
                    {
                        "id": f"doc-{index}",
                        "text": "这是用于真实分词校准的中文自然文本，内容长度足够。",
                        "parent_doc_id": f"doc-{index}",
                        "source": "cci3_hq",
                        "subset": None,
                        "source_path": "part.jsonl",
                        "revision": None,
                        "license": "Apache-2.0",
                        "url": None,
                        "language": "zh",
                        "domain": "general",
                        "char_count": 24,
                        "hanzi_count": 23,
                        "latin_count": 0,
                        "digit_count": 0,
                        "quality_prior": 3.0,
                        "tags": [],
                        "metadata_json": "{}",
                    }
                )
            pq.write_table(pa.Table.from_pylist(rows), cache_dir / "00000.parquet")

            report = calibrate(
                config,
                {source.name: {"sha256": manifest_sha}},
                overwrite=True,
            )

        stats = report["sources"]["cci3_hq"]
        self.assertEqual(stats["sampled_documents"], 10)
        self.assertEqual(stats["scanned_documents"], 20)
        self.assertGreater(stats["sampled_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
