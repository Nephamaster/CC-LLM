"""Phase 1 candidate classification and alignment-feature extraction."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from scripts.data_factory.config import PipelineConfig
from scripts.data_factory.text import mixed_language_stats, validate_mixed_text
from scripts.data_factory.windowing import (
    OversizedProtectedBlockError,
    TextWindow,
    build_natural_windows,
    build_natural_windows_batched,
)


SPECIALIZED_GROUPS = {
    "code": "code",
    "math": "math_science",
    "math_science": "math_science",
    "structured": "structured",
}
ALIGNMENT_POOLS = frozenset({"chinese_natural", "mixed_zh_en"})
LATEX_MATH_RE = re.compile(
    r"\$\$|\\\[|\\\(|\\(?:frac|sum|prod|int|sqrt|lim)\b"
    r"|\\begin\{(?:equation|align)\*?\}"
)
MATH_SYMBOL_RE = re.compile(r"[=≈≠≤≥∑∏√∞±×÷∫∂∇]")
SCIENCE_TERM_RE = re.compile(
    r"(?:定理|公式|方程|函数|概率|矩阵|向量|几何|物理|化学|统计|证明|推导)"
)


class CandidateSkip(ValueError):
    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True)
class CandidateFeatures:
    pool: str
    quota_group: str | None
    language_stats: dict[str, float | int | bool]
    bridge_hits: Counter[int]
    new_hanzi_hits: Counter[str]


class AlignmentFeatureMatcher:
    def __init__(self, config: PipelineConfig) -> None:
        try:
            import ahocorasick
        except ImportError as error:
            raise RuntimeError(
                "pyahocorasick is required for Phase 1 bridge-token indexing; "
                "install project requirements"
            ) from error

        removed = self._read_json(config.vocab_alignment.removed_multi_hanzi_tokens_path)
        new_hanzi = self._read_json(config.vocab_alignment.new_hanzi_token_ids_path)
        if not isinstance(removed, list) or not isinstance(new_hanzi, dict):
            raise ValueError("invalid vocabulary alignment metadata")

        automaton = ahocorasick.Automaton(ahocorasick.STORE_INTS)
        for row in removed:
            if not isinstance(row, dict):
                raise ValueError("removed_multi_hanzi_tokens.json must contain objects")
            token = row.get("token")
            old_token_id = row.get("old_token_id")
            if not isinstance(token, str) or len(token) < 2 or not isinstance(old_token_id, int):
                raise ValueError(f"invalid removed multi-Hanzi token entry: {row!r}")
            automaton.add_word(token, old_token_id)
        automaton.make_automaton()

        self.automaton = automaton
        self.new_hanzi_chars = frozenset(
            char
            for char in new_hanzi.values()
            if isinstance(char, str) and len(char) == 1
        )

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(f"vocabulary alignment metadata is missing: {path}")
        with path.open("rt", encoding="utf-8") as file:
            return json.load(file)

    def bridge_hits(self, text: str) -> Counter[int]:
        return Counter(int(old_token_id) for _end, old_token_id in self.automaton.iter(text))

    def new_hanzi_hits(self, text: str) -> Counter[str]:
        return Counter(char for char in text if char in self.new_hanzi_chars)


class CandidateClassifier:
    def __init__(self, quality: dict[str, Any]) -> None:
        self.quality = quality

    def classify(self, row: dict[str, Any], text: str) -> tuple[str, str | None, dict[str, Any]]:
        source = str(row.get("source", ""))
        source_category = str(row.get("category", ""))
        source_group = row.get("quota_group")

        if source == "clue_benchmark":
            raise CandidateSkip("excluded_clue", "CLUE is diagnostic-only in Phase 1")
        if row.get("synthetic"):
            raise CandidateSkip("synthetic_alignment_text", "synthetic Hanzi text is excluded")

        if source_category in {"supplemental", "specialized"}:
            group = SPECIALIZED_GROUPS.get(str(source_group))
            if group is None:
                raise CandidateSkip(
                    "obsolete_phase1_category",
                    f"unsupported Phase 1 supplemental group: {source_group!r}",
                )
            return "specialized", group, mixed_language_stats(text)

        if source_category == "non_chinese":
            return "non_chinese", None, mixed_language_stats(text)

        if self._is_math_science(text):
            return "specialized", "math_science", mixed_language_stats(text)

        stats = mixed_language_stats(text)
        if validate_mixed_text(text, self.quality, stats) is None:
            return "mixed_zh_en", None, stats
        if int(stats["hanzi"]) > 0:
            return "chinese_natural", None, stats
        return "non_chinese", None, stats

    @staticmethod
    def _is_math_science(text: str) -> bool:
        if LATEX_MATH_RE.search(text):
            return True
        return (
            len(MATH_SYMBOL_RE.findall(text)) >= 3
            and SCIENCE_TERM_RE.search(text) is not None
        )


class Phase1CandidateBuilder:
    def __init__(
        self,
        config: PipelineConfig,
        count_tokens: Callable[[str], int],
        matcher: AlignmentFeatureMatcher | None = None,
        count_many: Callable[[list[str]], list[int]] | None = None,
    ) -> None:
        self.config = config
        self.count_tokens = count_tokens
        self.count_many = count_many
        self.matcher = matcher or AlignmentFeatureMatcher(config)
        self.classifier = CandidateClassifier(config.quality)

    def build(
        self,
        row: dict[str, Any],
        normalized_text: str,
        full_token_count: int,
    ) -> list[tuple[dict[str, Any], int]]:
        parent_doc_id = str(row["doc_id"])
        self.classifier.classify(row, normalized_text)

        if full_token_count < self.config.windowing.min_tokens:
            raise CandidateSkip(
                "window_too_short",
                f"document has {full_token_count} tokens; minimum is "
                f"{self.config.windowing.min_tokens}",
            )

        source_group = str(row.get("quota_group", ""))
        if (
            str(row.get("category", "")) in {"supplemental", "specialized"}
            and SPECIALIZED_GROUPS.get(source_group) == "structured"
            and full_token_count > self.config.windowing.max_tokens
        ):
            raise CandidateSkip(
                "oversized_structured",
                "structured document exceeds the maximum window and cannot be split safely",
            )

        if full_token_count <= self.config.windowing.max_tokens:
            windows = [TextWindow(normalized_text, full_token_count)]
        else:
            try:
                if self.count_many is None:
                    windows = build_natural_windows(
                        normalized_text,
                        self.count_tokens,
                        self.config.windowing,
                    )
                else:
                    windows = build_natural_windows_batched(
                        normalized_text,
                        self.count_many,
                        self.config.windowing,
                    )
            except OversizedProtectedBlockError as error:
                raise CandidateSkip("oversized_protected_block", str(error)) from error
        if not windows:
            raise CandidateSkip("no_valid_window", "document produced no valid token window")

        feature_cache = {
            index: self._features(row, window.text)
            for index, window in enumerate(windows)
        }
        roles_by_window: dict[int, set[str]] = {
            self._stable_index(parent_doc_id, "base", len(windows)): {"base"}
        }

        new_candidates = [
            index
            for index, features in feature_cache.items()
            if features.new_hanzi_hits and self._alignment_eligible(features)
        ]
        if new_candidates:
            for selected in self._cover_new_hanzi_windows(
                parent_doc_id,
                new_candidates,
                feature_cache,
            ):
                roles_by_window.setdefault(selected, set()).add("new_hanzi_coverage")

        bridge_candidates = [
            index
            for index, features in feature_cache.items()
            if features.bridge_hits and self._alignment_eligible(features)
        ]
        if bridge_candidates:
            selected = self._best_bridge_window(parent_doc_id, bridge_candidates, feature_cache)
            roles_by_window.setdefault(selected, set()).add("multi_hanzi_bridge")

        candidates: list[tuple[dict[str, Any], int]] = []
        for window_index, roles in sorted(roles_by_window.items()):
            window = windows[window_index]
            features = feature_cache[window_index]
            value = dict(row)
            value.update(
                {
                    "source_category": row.get("category"),
                    "source_quota_group": row.get("quota_group"),
                    "doc_id": f"{parent_doc_id}#window-{window_index}",
                    "parent_doc_id": parent_doc_id,
                    "window_index": window_index,
                    "text": window.text,
                    "candidate_pool": features.pool,
                    "quota_group": features.quota_group,
                    "candidate_roles": sorted(roles),
                    "eligible_new_hanzi": "new_hanzi_coverage" in roles,
                    "eligible_bridge": "multi_hanzi_bridge" in roles,
                    "bridge_hits": {
                        str(token_id): count
                        for token_id, count in sorted(features.bridge_hits.items())
                    },
                    "new_hanzi_hits": dict(sorted(features.new_hanzi_hits.items())),
                    "language_stats": features.language_stats,
                }
            )
            candidates.append((value, window.token_count))
        return candidates

    def _features(self, row: dict[str, Any], text: str) -> CandidateFeatures:
        pool, quota_group, stats = self.classifier.classify(row, text)
        alignment_eligible = pool in ALIGNMENT_POOLS or (
            pool == "specialized" and quota_group == "math_science"
        )
        return CandidateFeatures(
            pool=pool,
            quota_group=quota_group,
            language_stats=stats,
            bridge_hits=self.matcher.bridge_hits(text) if alignment_eligible else Counter(),
            new_hanzi_hits=self.matcher.new_hanzi_hits(text) if alignment_eligible else Counter(),
        )

    @staticmethod
    def _alignment_eligible(features: CandidateFeatures) -> bool:
        return features.pool in ALIGNMENT_POOLS or (
            features.pool == "specialized" and features.quota_group == "math_science"
        )

    def _cover_new_hanzi_windows(
        self,
        parent_doc_id: str,
        candidates: list[int],
        features: dict[int, CandidateFeatures],
    ) -> list[int]:
        uncovered = {
            char
            for index in candidates
            for char in features[index].new_hanzi_hits
        }
        selected: list[int] = []
        remaining = set(candidates)
        while uncovered and remaining:
            index = min(
                remaining,
                key=lambda item: (
                    -len(uncovered & features[item].new_hanzi_hits.keys()),
                    -sum(
                        count
                        for char, count in features[item].new_hanzi_hits.items()
                        if char in uncovered
                    ),
                    self._stable_value(parent_doc_id, f"new:{item}"),
                ),
            )
            covered = uncovered & features[index].new_hanzi_hits.keys()
            if not covered:
                break
            selected.append(index)
            uncovered.difference_update(covered)
            remaining.remove(index)
        return selected
    def _best_bridge_window(
        self,
        parent_doc_id: str,
        candidates: list[int],
        features: dict[int, CandidateFeatures],
    ) -> int:
        def key(index: int) -> tuple[int, int, int]:
            hits = features[index].bridge_hits
            return (
                -len(hits),
                -sum(hits.values()),
                self._stable_value(parent_doc_id, f"bridge:{index}"),
            )

        return min(candidates, key=key)

    def _stable_index(self, doc_id: str, namespace: str, count: int) -> int:
        return self._stable_value(doc_id, namespace) % count

    def _stable_value(self, doc_id: str, namespace: str) -> int:
        payload = f"{self.config.seed}:{namespace}:{doc_id}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
