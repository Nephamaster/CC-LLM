from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.data_factory.build_phase1_validation import build_validation_set, scale_quotas
from scripts.data_factory.config import MIXED_QUOTAS, PHASE1_QUOTAS, SUPPLEMENTAL_QUOTAS
from scripts.data_factory.sample import CandidateIndex


class QuotaScalingTest(unittest.TestCase):
    def test_scaled_quotas_sum_to_target(self) -> None:
        scaled = scale_quotas(PHASE1_QUOTAS, 1_000_000)

        self.assertEqual(sum(scaled.values()), 1_000_000)
        self.assertEqual(scaled["chinese_general"], 400_000)
        self.assertEqual(scaled["chinese_high_quality"], 200_000)


class ValidationSetTest(unittest.TestCase):
    def test_builds_held_out_validation_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            database_path = root / "candidate_index.sqlite"
            index = CandidateIndex(database_path)
            rows: list[tuple[dict, int]] = []

            def add_pool(pool: str, count: int, quota_groups: list[str | None]) -> None:
                for item in range(count):
                    quota_group = quota_groups[item % len(quota_groups)]
                    rows.append(
                        (
                            {
                                "doc_id": f"{pool}-{item}",
                                "category": pool,
                                "quota_group": quota_group,
                                "source": f"{pool}-source",
                                "text": f"{pool} document {item}",
                            },
                            10,
                        )
                    )

            add_pool("chinese_general", 100, [None])
            add_pool("chinese_high_quality", 100, [None])
            add_pool("non_chinese", 100, [None])
            add_pool("mixed_zh_en", 100, list(MIXED_QUOTAS))
            add_pool("supplemental", 100, list(SUPPLEMENTAL_QUOTAS))
            try:
                index.add_many(rows, seed=7)
                index.connection.execute(
                    "UPDATE candidates SET selected_category = 'chinese_general' "
                    "WHERE doc_id = 'chinese_general-0'"
                )
                index.connection.execute(
                    "UPDATE candidates SET selected_category = 'mixed_zh_en' "
                    "WHERE pool = 'mixed_zh_en'"
                )
                index.connection.commit()
                index.create_sampling_indexes()
            finally:
                index.close()

            output_path = root / "validation" / "validation.jsonl"
            report_path = root / "reports" / "validation.json"
            config = SimpleNamespace(
                path=root / "phase1_config.json",
                quotas=PHASE1_QUOTAS,
                mixed_quotas=MIXED_QUOTAS,
                supplemental_quotas=SUPPLEMENTAL_QUOTAS,
            )
            report = build_validation_set(
                config=config,
                target_tokens=1_000,
                output_path=output_path,
                report_path=report_path,
                database_path=database_path,
            )

            output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(report["passed"])
            self.assertFalse(report["target_reached"])
            self.assertFalse(report["category_quotas_met"])
            self.assertEqual(report["category_shortfalls"]["mixed_zh_en"]["actual_tokens"], 0)
            self.assertGreater(report["actual_tokens"], 0)
            self.assertLess(report["actual_tokens"], 1_000)
            self.assertEqual(report["training_overlap_records"], 0)
            self.assertNotIn("chinese_general-0", {row["doc_id"] for row in output_rows})
            self.assertEqual(
                sum(report["category_checks"][key]["target_tokens"] for key in PHASE1_QUOTAS),
                1_000,
            )


if __name__ == "__main__":
    unittest.main()
