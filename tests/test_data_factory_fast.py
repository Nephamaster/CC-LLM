from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data_factory.fast_common import TokenEstimator, stable_fraction
from scripts.data_factory.fast_sample import PackedWriter, _allocate_source_targets
from scripts.data_factory.token_calibration import (
    _sample_cache_rows,
    _select_calibration_files,
)


class FastPipelineHelpersTest(unittest.TestCase):
    def test_source_allocation_respects_cap_when_feasible(self) -> None:
        result = _allocate_source_targets(
            {"a": 900, "b": 50, "c": 50},
            target=100,
            cap_ratio=0.40,
        )
        self.assertEqual(sum(result.values()), 100)
        self.assertLessEqual(max(result.values()), 40)

    def test_source_allocation_disables_impossible_cap(self) -> None:
        result = _allocate_source_targets(
            {"a": 900, "b": 100},
            target=100,
            cap_ratio=0.40,
        )
        self.assertEqual(sum(result.values()), 100)
        self.assertGreater(result["a"], 40)

    def test_token_estimator_uses_linear_coefficients(self) -> None:
        estimator = TokenEstimator(
            source_coefficients={"zh": (1.0, 0.25, 0.5, 0.5, 0.0)},
            source_ratios={"zh": 1.0},
            global_ratio=1.0,
        )
        value = estimator.estimate(
            source="zh",
            char_count=130,
            hanzi_count=100,
            latin_count=20,
            digit_count=10,
        )
        self.assertEqual(value, 110)

    def test_stable_fraction_is_deterministic(self) -> None:
        self.assertEqual(stable_fraction(7, "x", "doc"), stable_fraction(7, "x", "doc"))
        self.assertNotEqual(stable_fraction(7, "x", "doc"), stable_fraction(8, "x", "doc"))

    def test_calibration_file_selection_is_bounded_and_deterministic(self) -> None:
        files = [(Path(f"cache-{index}.parquet"), 100) for index in range(10)]
        first = _select_calibration_files(files, seed=7, source="zh", limit=3)
        second = _select_calibration_files(files, seed=7, source="zh", limit=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_calibration_row_sampling_has_a_hard_scan_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files: list[tuple[Path, int]] = []
            for file_index in range(2):
                rows = [
                    {
                        "doc_id": f"{file_index}-{row_index}",
                        "text": "中国 test 123",
                        "source": "zh",
                        "category": "chinese_natural",
                        "quota_group": None,
                        "char_count": 11,
                        "hanzi_count": 2,
                        "latin_count": 4,
                        "digit_count": 3,
                    }
                    for row_index in range(20)
                ]
                path = Path(directory) / f"part-{file_index}.parquet"
                pq.write_table(pa.Table.from_pylist(rows), path)
                files.append((path, len(rows)))

            sample, stats = _sample_cache_rows(
                files,
                source="zh",
                seed=7,
                target=10,
                scan_multiplier=2,
            )
            repeated, _ = _sample_cache_rows(
                files,
                source="zh",
                seed=7,
                target=10,
                scan_multiplier=2,
            )

            self.assertEqual(len(sample), 10)
            self.assertEqual(stats["scanned_documents"], 20)
            self.assertEqual(
                [row["doc_id"] for row in sample],
                [row["doc_id"] for row in repeated],
            )

    def test_packed_writer_splits_only_after_exact_tokenization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = PackedWriter(
                Path(directory),
                "train",
                sequence_length=4,
                shard_tokens=8,
                eos_id=99,
            )
            writer.add_document([1, 2, 3])
            writer.add_document([4, 5, 6, 7, 8])
            writer.close()
            self.assertGreaterEqual(writer.total_sequences, 2)
            self.assertGreaterEqual(writer.total_tokens, 8)
            self.assertTrue(writer.files)


if __name__ == "__main__":
    unittest.main()
