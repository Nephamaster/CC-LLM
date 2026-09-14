from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts.data_factory.candidate_features import (
    CandidateClassifier,
    CandidateSkip,
    Phase1CandidateBuilder,
)
from scripts.data_factory.config import WindowingConfig
from scripts.data_factory.text import mixed_language_stats, validate_mixed_text


QUALITY = {
    "mixed_min_hanzi": 4,
    "mixed_min_latin_words": 2,
    "mixed_min_hanzi_ratio": 0.20,
    "mixed_max_hanzi_ratio": 0.80,
    "mixed_min_latin_ratio": 0.05,
    "mixed_max_latin_ratio": 0.50,
    "mixed_min_boundaries": 2,
}


class FakeMatcher:
    priority_bridge_ids = frozenset({10})

    @staticmethod
    def bridge_hits(text: str) -> Counter[int]:
        return Counter({10: text.count("中国")}) if "中国" in text else Counter()

    @staticmethod
    def new_hanzi_hits(text: str) -> Counter[str]:
        return Counter(
            {
                char: text.count(char)
                for char in ("㐀", "㐁")
                if char in text
            }
        )


class CandidateClassifierTest(unittest.TestCase):
    def test_mixed_language_thresholds_and_boundaries(self) -> None:
        text = "中文内容 API test 中文继续"
        stats = mixed_language_stats(text)

        self.assertEqual(stats["boundary_count"], 2)
        self.assertIsNone(validate_mixed_text(text, QUALITY))
        self.assertEqual(
            CandidateClassifier(QUALITY).classify(
                {"category": "chinese_natural", "source": "cci3_hq"},
                text,
            )[:2],
            ("mixed_zh_en", None),
        )

    def test_maps_specialized_group_and_excludes_clue(self) -> None:
        classifier = CandidateClassifier(QUALITY)
        pool, group, _stats = classifier.classify(
            {"category": "supplemental", "quota_group": "code", "source": "github"},
            "def main(): return 1",
        )
        self.assertEqual((pool, group), ("specialized", "code"))

        with self.assertRaisesRegex(CandidateSkip, "diagnostic-only"):
            classifier.classify(
                {"category": "chinese_general", "source": "clue_benchmark"},
                "中文文本",
            )


class CandidateBuilderTest(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            seed=7,
            quality=QUALITY,
            windowing=WindowingConfig(min_tokens=4, target_tokens=12, max_tokens=20),
        )

    def test_builds_parent_locked_base_new_hanzi_and_bridge_windows(self) -> None:
        text = "普通内容普通内容。中国发展很好。罕见字㐀出现在这里。结尾内容。"
        row = {
            "doc_id": "doc-1",
            "category": "chinese_natural",
            "source": "cci3_hq",
            "text": text,
        }
        candidates = Phase1CandidateBuilder(
            self._config(),
            len,
            matcher=FakeMatcher(),
        ).build(row, text, len(text))

        output_rows = [candidate for candidate, _tokens in candidates]
        roles = {
            role
            for candidate in output_rows
            for role in candidate["candidate_roles"]
        }
        self.assertEqual(
            roles,
            {"base", "new_hanzi_coverage", "multi_hanzi_bridge"},
        )
        self.assertEqual({candidate["parent_doc_id"] for candidate in output_rows}, {"doc-1"})
        self.assertTrue(all(candidate["doc_id"].startswith("doc-1#window-") for candidate in output_rows))
        self.assertTrue(all(tokens <= 20 for _candidate, tokens in candidates))
        self.assertEqual(
            sum(bool(candidate["eligible_new_hanzi"]) for candidate in output_rows),
            1,
        )
        self.assertEqual(
            sum(bool(candidate["eligible_bridge"]) for candidate in output_rows),
            1,
        )

    def test_keeps_windows_needed_for_distinct_new_hanzi(self) -> None:
        text = "㐀甲乙丙丁戊己庚辛。普通内容普通内容。㐁壬癸子丑寅卯辰巳。"
        row = {
            "doc_id": "doc-coverage",
            "category": "chinese_natural",
            "source": "cci3_hq",
            "text": text,
        }
        candidates = Phase1CandidateBuilder(
            self._config(),
            len,
            matcher=FakeMatcher(),
        ).build(row, text, len(text))
        new_rows = [
            candidate
            for candidate, _tokens in candidates
            if candidate["eligible_new_hanzi"]
        ]

        covered = {
            char
            for candidate in new_rows
            for char in candidate["new_hanzi_hits"]
        }
        self.assertEqual(covered, {"㐀", "㐁"})

    def test_rejects_oversized_structured_document(self) -> None:
        row = {
            "doc_id": "structured-1",
            "category": "supplemental",
            "quota_group": "structured",
            "source": "github",
            "text": "{" + '"key":"value",' * 10 + '"end":1}',
        }
        builder = Phase1CandidateBuilder(self._config(), len, matcher=FakeMatcher())

        with self.assertRaisesRegex(CandidateSkip, "cannot be split safely"):
            builder.build(row, row["text"], len(row["text"]))


if __name__ == "__main__":
    unittest.main()
