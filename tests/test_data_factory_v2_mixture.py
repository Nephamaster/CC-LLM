from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data_factory.v2.mixture import ShardedParquetWriter


class DataFactoryV2MixtureTest(unittest.TestCase):
    def test_writer_preserves_uint64_sample_key(self) -> None:
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("sample_key", pa.uint64()),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = ShardedParquetWriter(root, schema=schema, max_rows=1)
            writer.write("bucket", {"id": "sample", "sample_key": 2**64 - 1})
            writer.close()
            table = pq.read_table(root / "bucket" / "part-00000.parquet")

        self.assertEqual(table.schema.field("sample_key").type, pa.uint64())
        self.assertEqual(table.column("sample_key").to_pylist(), [2**64 - 1])


if __name__ == "__main__":
    unittest.main()
