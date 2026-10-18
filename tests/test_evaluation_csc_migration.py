"""Regression tests for ChERRANT CSC substitution reconstruction."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluation.csc.metrics import _parse_m2_operations, cherrant_equalize


class ParseM2OperationsTests(unittest.TestCase):
    def test_first_alignment_only_and_noop_sentence(self):
        content = (
            "S 甲 乙 丙\n"
            "T0-A0 新 乙 丙\n"
            "A 0 1|||S|||新|||REQUIRED|||-NONE-|||0\n"
            "T0-A1 甲 旧 丙\n"
            "A 1 2|||S|||旧|||REQUIRED|||-NONE-|||0\n\n"
            "S 无 错\n"
            "T0 没有错误\n"
            "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n\n\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.m2"
            path.write_text(content, encoding="utf-8")
            self.assertEqual(
                _parse_m2_operations(path),
                [[(0, 1, "S", ["新"])], [(-1, -1, "noop", ["-NONE-"])]],
            )

    def test_missing_trailing_blank_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.m2"
            path.write_text("S 甲\nT0-A0 乙\nA 0 1|||S|||乙|||REQUIRED|||-NONE-|||0", encoding="utf-8")
            self.assertEqual(_parse_m2_operations(path), [[(0, 1, "S", ["乙"])]])


class CherrantEqualizeTests(unittest.TestCase):
    def run_mocked(self, rows, output_text):
        with tempfile.TemporaryDirectory() as directory:
            def fake_subprocess(command, cwd, check):
                self.assertEqual(cwd, directory)
                self.assertTrue(check)
                self.assertEqual(command[command.index("--multi_cheapest_strategy") + 1], "first")
                self.assertIn("--no_simplified", command)
                Path(command[command.index("-o") + 1]).write_text(output_text, encoding="utf-8")
            with patch("scripts.evaluation.csc.metrics.subprocess.run", side_effect=fake_subprocess):
                return cherrant_equalize(rows, Path(directory))

    def test_fullwidth_space_before_target(self):
        src = "曲显海\u3000嘿龙江省电力有限公司"
        expected = "曲显海\u3000黑龙江省电力有限公司"
        normalized = "".join(src.split())
        m2 = (
            "S " + " ".join(normalized) + "\n"
            "T0-A0 " + " ".join("".join(expected.split())) + "\n"
            "A 3 4|||S|||黑|||REQUIRED|||-NONE-|||0\n\n"
        )
        result, stats = self.run_mocked([{"src": src, "predict_clean": expected}], m2)
        self.assertEqual(result, [expected])
        self.assertEqual(stats["substitution_applied"], 1)

    def test_ascii_and_fullwidth_spaces_inside_substitution(self):
        src = "甲 乙\u3000丙丁"
        expected = "新 旧\u3000丙丁"
        m2 = "S 甲 乙 丙 丁\nT0-A0 新 旧 丙 丁\nA 0 2|||S|||新 旧|||REQUIRED|||-NONE-|||0\n\n"
        result, _ = self.run_mocked([{"src": src, "predict_clean": expected}], m2)
        self.assertEqual(result, [expected])

    def test_only_first_candidate_applied(self):
        src = "甲乙丙"
        m2 = (
            "S 甲 乙 丙\n"
            "T0-A0 新 乙 丙\nA 0 1|||S|||新|||REQUIRED|||-NONE-|||0\n"
            "T0-A1 甲 旧 丙\nA 1 2|||S|||旧|||REQUIRED|||-NONE-|||0\n\n"
        )
        result, stats = self.run_mocked([{"src": src, "predict_clean": "新乙丙"}], m2)
        self.assertEqual(result, ["新乙丙"])
        self.assertEqual(stats["substitution_applied"], 1)

    def test_normal_substitution_and_non_substitution(self):
        m2 = (
            "S 错 误\nT0-A0 正 误\nA 0 1|||S|||正|||REQUIRED|||-NONE-|||0\n\n"
            "S 无 错\nT0 没有错误\nA -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n\n"
        )
        rows = [{"src": "错误", "predict_clean": "正误"}, {"src": "无错", "predict_clean": "无错"}]
        result, stats = self.run_mocked(rows, m2)
        self.assertEqual(result, ["正误", "无错"])
        self.assertEqual(stats["substitution_applied"], 1)
        self.assertEqual(stats["non_equal_length_or_non_substitution"], 1)

    def test_preserves_traditional_unchanged(self):
        m2 = "S 乾 清\nT0 没有错误\nA -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n\n"
        result, _ = self.run_mocked([{"src": "乾清", "predict_clean": "乾清"}], m2)
        self.assertEqual(result, ["乾清"])

    def test_invalid_substitution_position_rejected(self):
        m2 = "S 甲 乙\nT0-A0 丙 乙\nA 2 3|||S|||丙|||REQUIRED|||-NONE-|||0\n\n"
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.run_mocked([{"src": "甲 乙", "predict_clean": "丙 乙"}], m2)


if __name__ == "__main__":
    unittest.main()

