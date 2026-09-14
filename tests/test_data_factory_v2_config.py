from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.data_factory.v2.config import load_data_factory_config


CONFIG_ROOT = Path("scripts/data_factory/configs")


class DataFactoryV2ConfigTest(unittest.TestCase):
    def test_phase1_contract(self) -> None:
        config = load_data_factory_config(CONFIG_ROOT / "phase1.yaml")

        self.assertEqual(config.target_tokens, 1_000_000_000)
        self.assertEqual(config.validation_tokens, 1_000_000)
        self.assertEqual(config.seed, 20260714)
        self.assertEqual(
            config.bucket_tokens,
            {
                "zh_general": 300_000_000,
                "zh_knowledge": 200_000_000,
                "english": 150_000_000,
                "mixed_zh_en": 100_000_000,
                "specialized": 100_000_000,
                "new_char_enhancement": 150_000_000,
            },
        )
        self.assertTrue(config.dedup.minhash_enabled)
        self.assertEqual(config.calibration.documents_per_source, 50_000)
        self.assertEqual(config.candidate_priority[0], "new_char_enhancement")
        self.assertTrue(config.enhancement.natural_only)
        self.assertFalse(config.enhancement.synthetic_enabled)
        self.assertEqual(config.enhancement.coverage_targets[1], 0.99)
        self.assertEqual(config.enhancement.constraints["single_source_max_fraction"], 0.30)
        self.assertFalse(any("bridge" in bucket.name for bucket in config.buckets))

    def test_phase2_contract(self) -> None:
        config = load_data_factory_config(CONFIG_ROOT / "phase2.yaml")

        self.assertEqual(config.target_tokens, 10_000_000_000)
        self.assertEqual(
            config.bucket_tokens,
            {
                "zh_general": 3_000_000_000,
                "zh_knowledge": 2_500_000_000,
                "english_multilingual_mixed": 2_000_000_000,
                "math_code_science": 1_500_000_000,
                "new_char_enhancement": 1_000_000_000,
            },
        )
        self.assertEqual(config.attributes["long_document"]["min_fraction"], 0.10)
        self.assertEqual(config.attributes["classical_chinese"]["min_fraction"], 0.08)
        self.assertEqual(config.attributes["classical_chinese"]["max_fraction"], 0.12)

    def test_exact_only_profile_disables_only_minhash(self) -> None:
        default = load_data_factory_config(CONFIG_ROOT / "phase1.yaml")
        exact_only = load_data_factory_config(
            CONFIG_ROOT / "phase1.yaml",
            profile="exact_only",
        )

        self.assertTrue(default.dedup.minhash_enabled)
        self.assertFalse(exact_only.dedup.minhash_enabled)
        self.assertEqual(exact_only.dedup.exact_algorithm, "sha256")
        self.assertNotEqual(default.run_id, exact_only.run_id)

    def test_run_id_depends_on_source_manifest(self) -> None:
        first = load_data_factory_config(
            CONFIG_ROOT / "phase1.yaml",
            source_manifest_sha256="a" * 64,
        )
        repeated = load_data_factory_config(
            CONFIG_ROOT / "phase1.yaml",
            source_manifest_sha256="a" * 64,
        )
        changed = load_data_factory_config(
            CONFIG_ROOT / "phase1.yaml",
            source_manifest_sha256="b" * 64,
        )

        self.assertEqual(first.run_id, repeated.run_id)
        self.assertNotEqual(first.run_id, changed.run_id)

    def test_snapshot_contains_resolved_identity(self) -> None:
        config = load_data_factory_config(CONFIG_ROOT / "phase1.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.snapshot.yaml"
            config.write_snapshot(path)
            snapshot = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(snapshot["run_id"], config.run_id)
        self.assertEqual(snapshot["target_tokens"], 1_000_000_000)
        self.assertEqual(snapshot["hashes"]["source_registry"], config.hashes.source_registry)


if __name__ == "__main__":
    unittest.main()
