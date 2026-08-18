from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_factory.config import (
    PHASE1_QUOTAS,
    PHASE1_TOTAL_TOKENS,
    SPECIALIZED_QUOTAS,
    load_config,
)


class Phase1ConfigTest(unittest.TestCase):
    def test_repository_config_uses_one_billion_token_strategy(self) -> None:
        config = load_config(Path("scripts/data_factory/phase1_config.json"))

        self.assertEqual(config.quotas, PHASE1_QUOTAS)
        self.assertEqual(sum(config.quotas.values()), PHASE1_TOTAL_TOKENS)
        self.assertEqual(config.specialized_quotas, SPECIALIZED_QUOTAS)
        self.assertEqual(config.windowing.min_tokens, 128)
        self.assertEqual(config.windowing.target_tokens, 512)
        self.assertEqual(config.windowing.max_tokens, 1024)
        self.assertEqual(config.validation.natural_tokens, 2_500_000)
        self.assertEqual(config.validation.alignment_tokens, 1_000_000)
        self.assertEqual(
            [path.name for path in config.vocab_alignment.priority_hanzi_paths],
            ["tghz2013.txt", "common_traditional.txt", "rare_high_freq.txt"],
        )
        self.assertEqual(config.tolerance, 0.01)
        self.assertEqual(config.preselection_buffer_ratio, 1.25)
        self.assertEqual(config.prepare_workers, 8)
        self.assertEqual(config.dedup_workers, 8)
        self.assertEqual(config.dedup["batch_size"], 128)
        self.assertEqual(config.dedup["batch_chars"], 1_000_000)
        self.assertEqual(config.dedup["sqlite_cache_mb"], 1024)

    def test_rejects_specialized_quota_mismatch(self) -> None:
        source = Path("scripts/data_factory/phase1_config.json")
        raw = json.loads(source.read_text(encoding="utf-8"))
        raw["repo_root"] = "."
        raw["specialized_quotas"]["structured"] = 10

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "phase1.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "specialized_quotas"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()