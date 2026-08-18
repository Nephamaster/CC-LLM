from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.data_factory.build_phase1_validation import build_validation_set
from scripts.data_factory.config import PHASE1_QUOTAS
from scripts.data_factory.sample import CandidateIndex
from scripts.data_factory.selection import Phase1Selector, parent_split_overlap, scale_quotas


class QuotaScalingTest(unittest.TestCase):
    def test_scaled_quotas_sum_to_target(self) -> None:
        scaled = scale_quotas({"a": 3, "b": 1}, 10)

        self.assertEqual(scaled, {"a": 8, "b": 2})


class ValidationSetTest(unittest.TestCase):
    def test_reserves_parent_exclusive_splits_and_exports_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            database_path = root / "candidate_index.sqlite"
            index = CandidateIndex(database_path)
            rows: list[tuple[dict, int]] = []

            def add_base(pool: str, count: int, quota_group: str | None = None) -> None:
                for item in range(count):
                    rows.append(
                        (
                            {
                                "doc_id": f"{pool}-{quota_group}-{item}",
                                "parent_doc_id": f"{pool}-{quota_group}-{item}",
                                "category": pool,
                                "candidate_pool": pool,
                                "candidate_roles": ["base"],
                                "quota_group": quota_group,
                                "source": f"{pool}-source",
                                "license": "MIT",
                                "license_status": "verified",
                                "text": f"{pool} document {item}",
                            },
                            10,
                        )
                    )

            add_base("chinese_natural", 10)
            add_base("non_chinese", 5)
            add_base("mixed_zh_en", 3)
            for group in ("code", "math_science", "structured"):
                add_base("specialized", 2, group)

            rows.extend(
                [
                    (
                        {
                            "doc_id": "feature-new",
                            "parent_doc_id": "feature-parent",
                            "category": "chinese_natural",
                            "candidate_pool": "chinese_natural",
                            "candidate_roles": ["base", "new_hanzi_coverage"],
                            "source": "feature-source",
                            "text": "new",
                            "eligible_new_hanzi": True,
                            "new_hanzi_hits": {"㐀": 1},
                        },
                        10,
                    ),
                    (
                        {
                            "doc_id": "feature-bridge",
                            "parent_doc_id": "feature-parent",
                            "category": "chinese_natural",
                            "candidate_pool": "chinese_natural",
                            "candidate_roles": ["multi_hanzi_bridge"],
                            "source": "feature-source",
                            "text": "bridge",
                            "eligible_bridge": True,
                            "bridge_hits": {"7": 1},
                        },
                        10,
                    ),
                    (
                        {
                            "doc_id": "bridge-only",
                            "parent_doc_id": "bridge-only",
                            "category": "chinese_natural",
                            "candidate_pool": "chinese_natural",
                            "candidate_roles": ["multi_hanzi_bridge"],
                            "source": "feature-source",
                            "text": "bridge only",
                            "eligible_bridge": True,
                            "bridge_hits": {"8": 1},
                        },
                        10,
                    ),
                ]
            )
            try:
                index.add_many(rows, seed=7)
                index.create_sampling_indexes()
                config = SimpleNamespace(
                    path=root / "phase1.json",
                    phase_root=root,
                    reports_dir=root / "reports",
                    quotas=PHASE1_QUOTAS,
                    validation=SimpleNamespace(natural_tokens=80, alignment_tokens=40),
                )
                selector = Phase1Selector(index.connection, config)
                selector.reset()
                reservation = selector.reserve_validation()
                self.assertEqual(parent_split_overlap(index.connection), 0)
                self.assertEqual(
                    index.connection.execute(
                        "SELECT COUNT(*) FROM parent_assignments "
                        "WHERE parent_doc_id = 'feature-parent'"
                    ).fetchone()[0],
                    1,
                )
            finally:
                index.close()

            report = build_validation_set(
                config,
                database_path,
                overwrite=True,
            )

            natural_path = root / "validation" / "validation_natural.jsonl"
            alignment_path = root / "validation" / "validation_alignment.jsonl"
            natural_rows = [
                json.loads(line)
                for line in natural_path.read_text(encoding="utf-8").splitlines()
            ]
            alignment_rows = [
                json.loads(line)
                for line in alignment_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(report["passed"])
            self.assertTrue(reservation["natural"]["target_reached"])
            self.assertTrue(reservation["alignment"]["target_reached"])
            self.assertGreater(len(natural_rows), 0)
            self.assertGreater(len(alignment_rows), 0)
            self.assertTrue(
                {row["parent_doc_id"] for row in natural_rows}.isdisjoint(
                    {row["parent_doc_id"] for row in alignment_rows}
                )
            )
            self.assertNotIn("bridge_hits", alignment_rows[0])


if __name__ == "__main__":
    unittest.main()
