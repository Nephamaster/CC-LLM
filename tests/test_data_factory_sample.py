from __future__ import annotations

import json
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.data_factory.sample import (
    BatchTokenCounter,
    CandidateIndex,
    TokenCountResult,
    _build_candidate_index,
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
                self.assertEqual(index.existing_doc_ids(["a", "missing"]), {"a"})

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

class ConcurrentIndexBuildTest(unittest.TestCase):
    def test_parallel_batches_and_resume(self) -> None:
        class FakeTokenCounter:
            active = 0
            max_active = 0
            lock = threading.Lock()

            def __init__(self, _config: object, _workers: int) -> None:
                pass

            def count_rows(self, rows: list[dict]) -> TokenCountResult:
                with self.lock:
                    type(self).active += 1
                    type(self).max_active = max(type(self).max_active, type(self).active)
                try:
                    time.sleep(0.03)
                    counted = [(row, len(str(row["text"]))) for row in rows]
                    return TokenCountResult(counted, 0, {}, [])
                finally:
                    with self.lock:
                        type(self).active -= 1

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            deduplicated = root / "deduplicated"
            deduplicated.mkdir()
            rows = [
                {
                    "doc_id": f"doc-{index}",
                    "category": "chinese_general",
                    "source": "source",
                    "text": f"text-{index}",
                }
                for index in range(8)
            ]
            input_path = deduplicated / "part-00000.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            config = SimpleNamespace(
                deduplicated_dir=deduplicated,
                sample_workers=4,
                sample_batch_size=1,
                sample_batch_chars=100,
                seed=7,
            )
            database_path = root / "candidate_index.sqlite"

            with patch("scripts.data_factory.sample.BatchTokenCounter", FakeTokenCounter):
                report = _build_candidate_index(config, database_path, workers=4)
                resumed = _build_candidate_index(config, database_path, resume=True, workers=4)

            self.assertEqual(report["records"], 8)
            self.assertGreaterEqual(FakeTokenCounter.max_active, 2)
            self.assertEqual(resumed["records"], 8)
            self.assertEqual(resumed["resumed_files"], 1)
            self.assertEqual(resumed["processed_this_run"], 0)

if __name__ == "__main__":
    unittest.main()