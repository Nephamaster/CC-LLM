from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from scripts.data_factory.v2.tokenization import PackedWriter, SwiftJsonlWriter


class DataFactoryV2TokenizationTest(unittest.TestCase):
    def test_swift_writer_emits_pretraining_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = SwiftJsonlWriter(root, "train", max_rows=1)
            writer.write("第一条文本")
            writer.write("second text")
            writer.close()
            files = sorted(root.glob("*.jsonl"))
            rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]

        self.assertEqual(len(files), 2)
        self.assertEqual(rows[0]["messages"][0]["role"], "assistant")
        self.assertEqual(rows[1]["messages"][0]["content"], "second text")

    def test_packer_inserts_eos_and_respects_sequence_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = PackedWriter(root, sequence_length=4, max_rows=2)
            writer.add("a", [1, 2, 3], eos_id=99)
            writer.add("b", [4, 5, 6, 7, 8], eos_id=99)
            writer.close()
            rows = [
                row
                for path in sorted(root.glob("*.parquet"))
                for row in pq.read_table(path).to_pylist()
            ]

        self.assertEqual(rows[0]["input_ids"], [1, 2, 3, 99])
        self.assertTrue(all(len(row["input_ids"]) <= 4 for row in rows))
        self.assertIn(99, [token for row in rows for token in row["input_ids"]])


if __name__ == "__main__":
    unittest.main()
