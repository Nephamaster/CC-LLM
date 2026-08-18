from __future__ import annotations

import unittest

from scripts.data_factory.config import WindowingConfig
from scripts.data_factory.windowing import (
    OversizedProtectedBlockError,
    build_natural_windows,
    build_natural_windows_batched,
    natural_boundary_units,
)


class NaturalWindowingTests(unittest.TestCase):
    def test_protected_blocks_are_atomic(self) -> None:
        text = (
            "第一句。第二句。\n\n"
            "```python\nprint('x')\n```\n\n"
            "$$\nx + y = z\n$$\n\n"
            "| 名称 | 值 |\n| --- | --- |\n| a | 1 |"
        )
        protected = [unit.text for unit in natural_boundary_units(text) if unit.protected]

        self.assertEqual(len(protected), 3)
        self.assertIn("```python\nprint('x')\n```", protected)
        self.assertIn("$$\nx + y = z\n$$", protected)
        self.assertIn("| 名称 | 值 |\n| --- | --- |\n| a | 1 |", protected)

    def test_builds_bounded_windows_without_splitting_fence(self) -> None:
        fence = "```python\nprint('x')\n```"
        text = f"甲乙丙丁。戊己庚辛。\n\n{fence}\n\n壬癸子丑。"
        windows = build_natural_windows(
            text,
            len,
            WindowingConfig(min_tokens=1, target_tokens=12, max_tokens=40),
        )

        self.assertTrue(windows)
        self.assertTrue(all(window.token_count <= 40 for window in windows))
        self.assertEqual(sum(fence in window.text for window in windows), 1)

    def test_preserves_sentence_and_paragraph_separators(self) -> None:
        text = "第一句。第二句。\n\n第三句。"
        windows = build_natural_windows(
            text,
            len,
            WindowingConfig(min_tokens=1, target_tokens=100, max_tokens=100),
        )

        self.assertEqual([window.text for window in windows], [text])

    def test_one_line_display_math_is_protected(self) -> None:
        units = natural_boundary_units("前文。\n\n$$x + y = z$$\n\n后文。")
        self.assertEqual([unit.text for unit in units if unit.protected], ["$$x + y = z$$"])

    def test_batched_builder_limits_tokenizer_calls(self) -> None:
        calls: list[int] = []

        def count_many(texts: list[str]) -> list[int]:
            calls.append(len(texts))
            return [len(text) for text in texts]

        text = "甲乙丙丁。" * 20
        windows = build_natural_windows_batched(
            text,
            count_many,
            WindowingConfig(min_tokens=1, target_tokens=20, max_tokens=30),
        )

        self.assertTrue(windows)
        self.assertTrue(all(window.token_count <= 30 for window in windows))
        self.assertLessEqual(len(calls), 3)
        self.assertGreater(calls[0], 1)

    def test_rejects_oversized_protected_block(self) -> None:
        with self.assertRaises(OversizedProtectedBlockError):
            build_natural_windows(
                "```\n0123456789\n```",
                len,
                WindowingConfig(min_tokens=1, target_tokens=5, max_tokens=8),
            )


if __name__ == "__main__":
    unittest.main()
