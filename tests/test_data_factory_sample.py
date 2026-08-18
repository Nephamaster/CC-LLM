from __future__ import annotations

import json
import re
import tempfile
import threading
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.data_factory.sample import (
    BatchTokenCounter,
    CandidateIndex,
    TokenCountResult,
    _filter_candidates_for_intent,
    _materialize_preselected_candidates,
)


class CandidateIndexTest(unittest.TestCase):
    def test_bulk_insert_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            index = CandidateIndex(root / "candidates.sqlite")
            rows = [
                ({"doc_id": "a", "category": "pool", "source": "source", "text": "\u4e2d"}, 1),
                ({"doc_id": "b", "category": "pool", "source": "source", "text": "text"}, 2),
            ]
            try:
                self.assertEqual(index.add_many(rows, seed=7), 2)
                self.assertEqual(index.add_many(rows, seed=7), 0)
                self.assertEqual(index.totals(), (2, 3))
                self.assertEqual(index.feature_inventory()["parent_records"], 2)
                self.assertEqual(index.existing_doc_ids(["a", "missing"]), {"a"})
                self.assertEqual(index.existing_parent_doc_ids(["a", "missing"]), {"a"})

                input_path = root / "part-00000.jsonl"
                input_path.write_text("{}\n", encoding="utf-8")
                report = {"processed_records": 2, "skipped_by_reason": {}}
                index.mark_input_complete(input_path, report)
                self.assertTrue(index.input_is_complete(input_path))
                self.assertEqual(index.progress_reports(), [report])

                index.drop_sampling_indexes()
                index.create_sampling_indexes()
            finally:
                index.close()


class BatchTokenCounterTest(unittest.TestCase):
    def test_unsupported_hanzi_and_tokenization_errors_are_skipped(self) -> None:
        supported = "\u4e2d"
        unsupported = "\U0002543b"
        counter = BatchTokenCounter.__new__(BatchTokenCounter)
        counter.normalize_text = lambda text: text
        counter.supported_hanzi = {supported}
        counter.hanzi_pattern = re.compile(f"[{supported}{unsupported}]")

        def encode_lengths(texts: list[str]) -> list[int]:
            if "bad" in texts:
                raise ValueError("bad text")
            return [len(text) for text in texts]

        counter._encode_lengths = encode_lengths
        result = counter.count_rows(
            [
                {"doc_id": "valid", "category": "pool", "source": "source", "text": supported + "A"},
                {"doc_id": "unsupported", "category": "pool", "source": "source", "text": unsupported},
                {"doc_id": "bad", "category": "pool", "source": "source", "text": "bad"},
                {"doc_id": "empty", "category": "pool", "source": "source", "text": ""},
            ]
        )

        self.assertEqual(result.rows, [({"doc_id": "valid", "category": "pool", "source": "source", "text": supported + "A"}, 2)])
        self.assertEqual(result.zero_token_records, 1)
        self.assertEqual(result.skipped_by_reason["unsupported_hanzi"], 1)
        self.assertEqual(result.skipped_by_reason["tokenization_error"], 1)

class ConcurrentMaterializationTest(unittest.TestCase):
    def test_parallel_batches_and_resume(self) -> None:
        class FakeTokenCounter:
            active = 0
            max_active = 0
            lock = threading.Lock()

            def __init__(self, _config: object, _workers: int) -> None:
                pass

        class FakeCandidateProcessor:
            def __init__(self, _config: object, _counter: FakeTokenCounter) -> None:
                pass

            def count_rows(self, rows: list[dict]) -> TokenCountResult:
                with FakeTokenCounter.lock:
                    FakeTokenCounter.active += 1
                    FakeTokenCounter.max_active = max(
                        FakeTokenCounter.max_active,
                        FakeTokenCounter.active,
                    )
                try:
                    time.sleep(0.03)
                    candidates = []
                    for row in rows:
                        value = dict(row)
                        value.update(
                            {
                                "doc_id": f"{row['doc_id']}#window-0",
                                "parent_doc_id": row["doc_id"],
                                "candidate_pool": "chinese_natural",
                                "candidate_roles": ["base"],
                            }
                        )
                        candidates.append((value, len(str(row["text"]))))
                    return TokenCountResult(candidates, 0, Counter(), [])
                finally:
                    with FakeTokenCounter.lock:
                        FakeTokenCounter.active -= 1

        class FakeDocumentIndex:
            def __init__(self, items: list[tuple[dict, str]]) -> None:
                self.items = items
                self.materialized: set[str] = set()

            def selected_count(self) -> int:
                return len(self.items)

            def materialized_count(self) -> int:
                return len(self.materialized)

            def reset_materialized(self) -> None:
                self.materialized.clear()

            def mark_materialized(self, doc_ids: list[str]) -> None:
                self.materialized.update(doc_ids)

        rows = [
            (
                {
                    "doc_id": f"doc-{index}",
                    "category": "chinese_general",
                    "source": "source",
                    "text": f"text-{index}",
                },
                "chinese_natural",
            )
            for index in range(8)
        ]
        document_index = FakeDocumentIndex(rows)

        def batches(index, *, max_records: int, max_chars: int):
            del max_chars
            pending = [
                item
                for item in index.items
                if item[0]["doc_id"] not in index.materialized
            ]
            for start in range(0, len(pending), max_records):
                yield pending[start : start + max_records]

        config = SimpleNamespace(
            sample_batch_size=1,
            sample_batch_chars=100,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            index = CandidateIndex(Path(temporary_dir) / "candidate.sqlite")
            try:
                with (
                    patch(
                        "scripts.data_factory.sample.BatchTokenCounter",
                        FakeTokenCounter,
                    ),
                    patch(
                        "scripts.data_factory.sample.Phase1CandidateProcessor",
                        FakeCandidateProcessor,
                    ),
                    patch(
                        "scripts.data_factory.sample.iter_selected_document_batches",
                        batches,
                    ),
                ):
                    report = _materialize_preselected_candidates(
                        config,
                        index,
                        document_index,
                        workers=4,
                    )
                    resumed = _materialize_preselected_candidates(
                        config,
                        index,
                        document_index,
                        workers=4,
                    )

                self.assertEqual(index.totals(), (8, 48))
                self.assertEqual(report["candidate_records_inserted"], 8)
                self.assertGreaterEqual(FakeTokenCounter.max_active, 2)
                self.assertEqual(resumed["processed_this_run"], 0)
            finally:
                index.close()

    def test_candidate_intent_isolation(self) -> None:
        row = {
            "doc_id": "doc#window-0",
            "parent_doc_id": "doc",
            "candidate_roles": ["base", "new_hanzi_coverage", "multi_hanzi_bridge"],
            "candidate_pool": "chinese_natural",
            "eligible_new_hanzi": True,
            "eligible_bridge": True,
            "new_hanzi_hits": {"㐀": 1},
            "bridge_hits": {"7": 1},
        }
        [(filtered, _tokens)] = _filter_candidates_for_intent(
            [(row, 10)],
            {"doc": "new_hanzi_coverage"},
        )
        self.assertTrue(filtered["eligible_new_hanzi"])
        self.assertFalse(filtered["eligible_bridge"])
        self.assertEqual(filtered["bridge_hits"], {})
        self.assertEqual(filtered["candidate_roles"], ["new_hanzi_coverage"])


if __name__ == "__main__":
    unittest.main()