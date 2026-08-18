from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.data_factory.config import PHASE1_QUOTAS
from scripts.data_factory.sample import CandidateIndex
from scripts.data_factory.selection import Phase1Selector, parent_split_overlap


class AlignmentSelectionTest(unittest.TestCase):
    def test_covers_new_hanzi_before_filling_and_tracks_top_bridge_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            new_hanzi_path = root / "new_hanzi.json"
            new_hanzi_path.write_text(
                json.dumps({"100": "㐀", "101": "㐁"}, ensure_ascii=False),
                encoding="utf-8",
            )
            priority_path = root / "priority.txt"
            priority_path.write_text("㐀\n㐁\n", encoding="utf-8")

            index = CandidateIndex(root / "candidate.sqlite")
            rows = [
                (
                    {
                        "doc_id": "new-a",
                        "parent_doc_id": "new-a",
                        "category": "chinese_natural",
                        "candidate_pool": "chinese_natural",
                        "candidate_roles": ["new_hanzi_coverage"],
                        "eligible_new_hanzi": True,
                        "new_hanzi_hits": {"㐀": 2},
                        "source": "source-a",
                        "text": "甲㐀㐀",
                    },
                    10,
                ),
                (
                    {
                        "doc_id": "new-b",
                        "parent_doc_id": "new-b",
                        "category": "chinese_natural",
                        "candidate_pool": "chinese_natural",
                        "candidate_roles": ["new_hanzi_coverage"],
                        "eligible_new_hanzi": True,
                        "new_hanzi_hits": {"㐁": 1},
                        "source": "source-b",
                        "text": "乙㐁",
                    },
                    10,
                ),
                (
                    {
                        "doc_id": "bridge-a",
                        "parent_doc_id": "bridge-a",
                        "category": "chinese_natural",
                        "candidate_pool": "chinese_natural",
                        "candidate_roles": ["multi_hanzi_bridge"],
                        "eligible_bridge": True,
                        "bridge_hits": {"7": 3},
                        "source": "source-a",
                        "text": "中国中国中国",
                    },
                    10,
                ),
                (
                    {
                        "doc_id": "bridge-b",
                        "parent_doc_id": "bridge-b",
                        "category": "chinese_natural",
                        "candidate_pool": "chinese_natural",
                        "candidate_roles": ["multi_hanzi_bridge"],
                        "eligible_bridge": True,
                        "bridge_hits": {"8": 1},
                        "source": "source-b",
                        "text": "语言模型",
                    },
                    10,
                ),
            ]
            try:
                index.add_many(rows, seed=7)
                index.create_sampling_indexes()
                index.set_metadata(
                    "bridge_top_tokens",
                    [{"old_token_id": 8}, {"old_token_id": 7}],
                )
                config = SimpleNamespace(
                    quotas=PHASE1_QUOTAS,
                    vocab_alignment=SimpleNamespace(
                        new_hanzi_token_ids_path=new_hanzi_path,
                        priority_hanzi_paths=(priority_path,),
                        priority_hanzi_min_documents=1,
                        priority_hanzi_coverage=1.0,
                        bridge_top_token_count=2,
                        bridge_min_contexts=1,
                    ),
                )
                selector = Phase1Selector(index.connection, config)
                selector.reset()
                new_result = selector._select_new_hanzi(20)
                bridge_result = selector._select_bridge(20)

                self.assertTrue(new_result["coverage"]["passed"])
                self.assertEqual(
                    new_result["coverage"]["covered_observed_chars"],
                    2,
                )
                self.assertTrue(bridge_result["coverage"]["passed"])
                self.assertEqual(
                    bridge_result["coverage"]["tracked_top_tokens"],
                    2,
                )
                self.assertEqual(
                    [
                        row["old_token_id"]
                        for row in bridge_result["coverage"]["token_stats"]
                    ],
                    [8, 7],
                )
                self.assertEqual(parent_split_overlap(index.connection), 0)
            finally:
                index.close()


if __name__ == "__main__":
    unittest.main()
