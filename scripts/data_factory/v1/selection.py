"""Parent-exclusive Phase 1 validation reservation and quota selection."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from scripts.data_factory.config import PipelineConfig


CHINESE_TRAIN_CATEGORIES = frozenset(
    {"chinese_natural", "multi_hanzi_bridge", "new_hanzi_coverage", "mixed_zh_en"}
)
NATURAL_VALIDATION_WEIGHTS = {
    "chinese_natural": 450,
    "non_chinese": 150,
    "mixed_zh_en": 100,
    "code": 40,
    "math_science": 40,
    "structured": 20,
}
ALIGNMENT_VALIDATION_GROUPS = (
    "new_hanzi_coverage",
    "multi_hanzi_bridge",
    "original_hanzi",
    "non_chinese",
)


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    parent_doc_id: str
    source: str
    token_count: int
    row: dict[str, Any]


def scale_quotas(weights: dict[str, int], target_tokens: int) -> dict[str, int]:
    """Scale integer weights to an exact token target."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("quota weights must sum to a positive value")

    scaled: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    for name, weight in weights.items():
        value, remainder = divmod(target_tokens * weight, total)
        scaled[name] = value
        remainders.append((remainder, name))
    missing = target_tokens - sum(scaled.values())
    for _remainder, name in sorted(remainders, key=lambda item: (-item[0], item[1]))[:missing]:
        scaled[name] += 1
    return scaled


def _read_hanzi_resource(path: Path) -> set[str]:
    chars: set[str] = set()
    with path.open("rt", encoding="utf-8") as file:
        for line in file:
            value = line.lstrip("﻿").strip()
            if not value or value.startswith("#"):
                continue
            first_field = value.split("	", 1)[0].strip()
            if first_field and first_field.lower() != "char":
                chars.add(first_field[0])
    return chars


def _json_counter(row: dict[str, Any], key: str, cast_key: type = str) -> Counter[Any]:
    value = row.get(key, {})
    if not isinstance(value, dict):
        return Counter()
    result: Counter[Any] = Counter()
    for item, count in value.items():
        try:
            result[cast_key(item)] += int(count)
        except (TypeError, ValueError):
            continue
    return result


class Phase1Selector:
    """Assign each parent document to exactly one split and one category."""

    def __init__(self, connection: sqlite3.Connection, config: PipelineConfig) -> None:
        self.connection = connection
        self.config = config
        chinese_tokens = sum(config.quotas[name] for name in CHINESE_TRAIN_CATEGORIES)
        self.source_token_cap = int(chinese_tokens * 0.40)
        self.chinese_tokens_by_source: Counter[str] = Counter()

    def reset(self) -> None:
        self.connection.execute("DELETE FROM selections")
        self.connection.execute("DELETE FROM parent_assignments")
        self.connection.execute("UPDATE candidates SET selected_category = NULL")
        self.connection.commit()
        self.chinese_tokens_by_source.clear()

    def _iter_candidates(
        self,
        *,
        pool: str | None = None,
        quota_group: str | None = None,
        eligible: str | None = None,
        base_only: bool = False,
        extra_clause: str | None = None,
        include_assigned: bool = False,
    ) -> Iterator[Candidate]:
        clauses: list[str] = []
        values: list[Any] = []
        if pool is not None:
            clauses.append("c.pool = ?")
            values.append(pool)
        if quota_group is not None:
            clauses.append("c.quota_group = ?")
            values.append(quota_group)
        if eligible is not None:
            if eligible not in {"eligible_bridge", "eligible_new_hanzi"}:
                raise ValueError(f"invalid eligibility column: {eligible}")
            clauses.append(f"c.{eligible} = 1")
        if base_only:
            clauses.append("(',' || c.candidate_role || ',') LIKE '%,base,%'")
        if extra_clause:
            clauses.append(extra_clause)
        if not include_assigned:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM parent_assignments p "
                "WHERE p.parent_doc_id = c.parent_doc_id)"
            )
        where = " AND ".join(clauses) if clauses else "1 = 1"
        cursor = self.connection.execute(
            f"""
            SELECT c.doc_id, c.parent_doc_id, c.source, c.token_count, c.row_json
            FROM candidates c
            WHERE {where}
            ORDER BY c.sample_key, c.doc_id
            """,
            values,
        )
        try:
            for doc_id, parent_doc_id, source, token_count, row_json in cursor:
                yield Candidate(
                    doc_id=str(doc_id),
                    parent_doc_id=str(parent_doc_id),
                    source=str(source),
                    token_count=int(token_count),
                    row=json.loads(row_json),
                )
        finally:
            cursor.close()

    def _feature_rows(self, parent_doc_id: str, eligible: str) -> list[Candidate]:
        if eligible not in {"eligible_bridge", "eligible_new_hanzi"}:
            raise ValueError(f"invalid eligibility column: {eligible}")
        rows = self.connection.execute(
            f"""
            SELECT doc_id, parent_doc_id, source, token_count, row_json
            FROM candidates
            WHERE parent_doc_id = ? AND {eligible} = 1
            ORDER BY sample_key, doc_id
            """,
            (parent_doc_id,),
        )
        return [
            Candidate(
                doc_id=str(doc_id),
                parent_doc_id=str(parent),
                source=str(source),
                token_count=int(token_count),
                row=json.loads(row_json),
            )
            for doc_id, parent, source, token_count, row_json in rows
        ]

    def _assign(
        self,
        rows: list[Candidate],
        split: str,
        category: str,
    ) -> tuple[int, int]:
        if not rows:
            return 0, 0
        parent_doc_id = rows[0].parent_doc_id
        source = rows[0].source
        token_count = sum(row.token_count for row in rows)
        if split == "train" and category in CHINESE_TRAIN_CATEGORIES:
            if self.chinese_tokens_by_source[source] + token_count > self.source_token_cap:
                return 0, 0

        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO parent_assignments(parent_doc_id, split, category, source)
            VALUES (?, ?, ?, ?)
            """,
            (parent_doc_id, split, category, source),
        )
        if cursor.rowcount == 0:
            return 0, 0

        self.connection.executemany(
            """
            INSERT INTO selections(doc_id, parent_doc_id, split, category, token_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (row.doc_id, row.parent_doc_id, split, category, row.token_count)
                for row in rows
            ],
        )
        if split == "train" and category in CHINESE_TRAIN_CATEGORIES:
            self.chinese_tokens_by_source[source] += token_count
        return len(rows), token_count

    def select_base(
        self,
        *,
        split: str,
        category: str,
        target_tokens: int,
        pool: str,
        quota_group: str | None = None,
        extra_clause: str | None = None,
    ) -> dict[str, int]:
        records = 0
        tokens = 0
        for candidate in self._iter_candidates(
            pool=pool,
            quota_group=quota_group,
            base_only=True,
            extra_clause=extra_clause,
        ):
            added_records, added_tokens = self._assign([candidate], split, category)
            records += added_records
            tokens += added_tokens
            if tokens >= target_tokens:
                break
        return {"target_tokens": target_tokens, "records": records, "tokens": tokens}

    def select_feature(
        self,
        *,
        split: str,
        category: str,
        target_tokens: int,
        eligible: str,
    ) -> dict[str, int]:
        records = 0
        tokens = 0
        for candidate in self._iter_candidates(eligible=eligible):
            rows = self._feature_rows(candidate.parent_doc_id, eligible)
            added_records, added_tokens = self._assign(rows, split, category)
            records += added_records
            tokens += added_tokens
            if tokens >= target_tokens:
                break
        return {"target_tokens": target_tokens, "records": records, "tokens": tokens}

    def reserve_validation(self) -> dict[str, Any]:
        natural_targets = scale_quotas(
            NATURAL_VALIDATION_WEIGHTS,
            self.config.validation.natural_tokens,
        )
        natural: dict[str, dict[str, int]] = {}
        non_feature_parent = (
            "NOT EXISTS (SELECT 1 FROM candidates f "
            "WHERE f.parent_doc_id = c.parent_doc_id "
            "AND (f.eligible_new_hanzi = 1 OR f.eligible_bridge = 1))"
        )
        natural["chinese_natural"] = self.select_base(
            split="validation_natural",
            category="chinese_natural",
            target_tokens=natural_targets["chinese_natural"],
            pool="chinese_natural",
            extra_clause=non_feature_parent,
        )
        natural["non_chinese"] = self.select_base(
            split="validation_natural",
            category="non_chinese",
            target_tokens=natural_targets["non_chinese"],
            pool="non_chinese",
        )
        natural["mixed_zh_en"] = self.select_base(
            split="validation_natural",
            category="mixed_zh_en",
            target_tokens=natural_targets["mixed_zh_en"],
            pool="mixed_zh_en",
            extra_clause=non_feature_parent,
        )
        for group in ("code", "math_science", "structured"):
            natural[group] = self.select_base(
                split="validation_natural",
                category=group,
                target_tokens=natural_targets[group],
                pool="specialized",
                quota_group=group,
            )

        alignment_targets = scale_quotas(
            {name: 1 for name in ALIGNMENT_VALIDATION_GROUPS},
            self.config.validation.alignment_tokens,
        )
        alignment = {
            "new_hanzi_coverage": self.select_feature(
                split="validation_alignment",
                category="new_hanzi_coverage",
                target_tokens=alignment_targets["new_hanzi_coverage"],
                eligible="eligible_new_hanzi",
            ),
            "multi_hanzi_bridge": self.select_feature(
                split="validation_alignment",
                category="multi_hanzi_bridge",
                target_tokens=alignment_targets["multi_hanzi_bridge"],
                eligible="eligible_bridge",
            ),
            "original_hanzi": self.select_base(
                split="validation_alignment",
                category="original_hanzi",
                target_tokens=alignment_targets["original_hanzi"],
                pool="chinese_natural",
                extra_clause=non_feature_parent,
            ),
            "non_chinese": self.select_base(
                split="validation_alignment",
                category="non_chinese",
                target_tokens=alignment_targets["non_chinese"],
                pool="non_chinese",
            ),
        }
        self.connection.commit()
        return {
            "natural": self._group_report(natural_targets, natural),
            "alignment": self._group_report(alignment_targets, alignment),
        }

    @staticmethod
    def _group_report(
        targets: dict[str, int],
        results: dict[str, dict[str, int]],
    ) -> dict[str, Any]:
        actual_tokens = sum(result["tokens"] for result in results.values())
        shortfalls = {
            name: {
                "target_tokens": targets[name],
                "actual_tokens": results[name]["tokens"],
            }
            for name in targets
            if results[name]["tokens"] < targets[name]
        }
        return {
            "target_tokens": sum(targets.values()),
            "actual_tokens": actual_tokens,
            "records": sum(result["records"] for result in results.values()),
            "target_reached": not shortfalls,
            "shortfalls": shortfalls,
            "categories": results,
        }

    def select_training(self) -> tuple[dict[str, Any], dict[str, Any]]:
        new_hanzi = self._select_new_hanzi(self.config.quotas["new_hanzi_coverage"])
        bridge = self._select_bridge(self.config.quotas["multi_hanzi_bridge"])

        selection: dict[str, Any] = {
            "new_hanzi_coverage": new_hanzi["selection"],
            "multi_hanzi_bridge": bridge["selection"],
        }
        selection["mixed_zh_en"] = self.select_base(
            split="train",
            category="mixed_zh_en",
            target_tokens=self.config.quotas["mixed_zh_en"],
            pool="mixed_zh_en",
        )

        specialized_parts: dict[str, dict[str, int]] = {}
        for group, target in self.config.specialized_quotas.items():
            specialized_parts[group] = self.select_base(
                split="train",
                category="specialized",
                target_tokens=target,
                pool="specialized",
                quota_group=group,
            )
        selection["specialized"] = {
            "target_tokens": self.config.quotas["specialized"],
            "records": sum(item["records"] for item in specialized_parts.values()),
            "tokens": sum(item["tokens"] for item in specialized_parts.values()),
            "parts": specialized_parts,
        }
        selection["chinese_natural"] = self.select_base(
            split="train",
            category="chinese_natural",
            target_tokens=self.config.quotas["chinese_natural"],
            pool="chinese_natural",
        )
        selection["non_chinese"] = self.select_base(
            split="train",
            category="non_chinese",
            target_tokens=self.config.quotas["non_chinese"],
            pool="non_chinese",
        )

        self.connection.execute(
            """
            UPDATE candidates
            SET selected_category = (
                SELECT s.category FROM selections s
                WHERE s.doc_id = candidates.doc_id AND s.split = 'train'
            )
            """
        )
        self.connection.commit()
        return selection, {
            "new_hanzi_coverage": new_hanzi["coverage"],
            "multi_hanzi_bridge": bridge["coverage"],
        }

    def _select_new_hanzi(self, target_tokens: int) -> dict[str, Any]:
        new_chars = self._load_new_hanzi()
        priority_chars = self._load_priority_hanzi() & new_chars
        observed_chars: set[str] = set()
        for candidate in self._iter_candidates(eligible="eligible_new_hanzi"):
            observed_chars.update(_json_counter(candidate.row, "new_hanzi_hits"))

        document_counts: Counter[str] = Counter()
        input_counts: Counter[str] = Counter()
        prediction_counts: Counter[str] = Counter()
        sources: dict[str, set[str]] = defaultdict(set)
        selected_records = 0
        selected_tokens = 0

        def record(rows: Iterable[Candidate]) -> None:
            seen_in_parent: set[str] = set()
            for row in rows:
                hits = _json_counter(row.row, "new_hanzi_hits")
                text = str(row.row.get("text", ""))
                for char, count in hits.items():
                    input_counts[char] += count
                    prediction_counts[char] += count - int(text.startswith(char))
                    sources[char].add(row.source)
                    seen_in_parent.add(char)
            document_counts.update(seen_in_parent)

        def select_useful(predicate: Callable[[Counter[str]], bool]) -> None:
            nonlocal selected_records, selected_tokens
            for candidate in self._iter_candidates(eligible="eligible_new_hanzi"):
                hits = _json_counter(candidate.row, "new_hanzi_hits")
                if not predicate(hits):
                    continue
                rows = self._feature_rows(candidate.parent_doc_id, "eligible_new_hanzi")
                added_records, added_tokens = self._assign(
                    rows,
                    "train",
                    "new_hanzi_coverage",
                )
                if added_records:
                    record(rows)
                    selected_records += added_records
                    selected_tokens += added_tokens
                if selected_tokens >= target_tokens:
                    break

        minimum = self.config.vocab_alignment.priority_hanzi_min_documents
        select_useful(lambda hits: any(document_counts[char] == 0 for char in hits))
        if selected_tokens < target_tokens:
            select_useful(
                lambda hits: any(
                    char in priority_chars and document_counts[char] < minimum
                    for char in hits
                )
            )
        if selected_tokens < target_tokens:
            select_useful(lambda _hits: True)

        character_stats = {
            char: {
                "input_occurrences": input_counts[char],
                "prediction_occurrences": prediction_counts[char],
                "documents": document_counts[char],
                "sources": len(sources[char]),
                "priority": char in priority_chars,
            }
            for char in sorted(new_chars, key=ord)
        }
        uncovered_observed = sorted(observed_chars - document_counts.keys(), key=ord)
        priority_met = sum(document_counts[char] >= minimum for char in priority_chars)
        priority_ratio = priority_met / len(priority_chars) if priority_chars else 1.0
        return {
            "selection": {
                "target_tokens": target_tokens,
                "records": selected_records,
                "tokens": selected_tokens,
            },
            "coverage": {
                "passed": (
                    not uncovered_observed
                    and priority_ratio >= self.config.vocab_alignment.priority_hanzi_coverage
                ),
                "target_chars": len(new_chars),
                "observed_candidate_chars": len(observed_chars),
                "covered_observed_chars": len(observed_chars) - len(uncovered_observed),
                "uncovered_observed_chars": uncovered_observed,
                "priority_chars": len(priority_chars),
                "priority_min_documents": minimum,
                "priority_chars_met": priority_met,
                "priority_coverage": priority_ratio,
                "required_priority_coverage": self.config.vocab_alignment.priority_hanzi_coverage,
                "character_stats": character_stats,
            },
        }

    def _select_bridge(self, target_tokens: int) -> dict[str, Any]:
        candidate_occurrences: Counter[int] = Counter()
        candidate_documents: Counter[int] = Counter()
        available_documents: Counter[int] = Counter()
        cursor = self.connection.execute(
            """
            SELECT c.row_json, p.parent_doc_id IS NULL
            FROM candidates c
            LEFT JOIN parent_assignments p
                ON p.parent_doc_id = c.parent_doc_id
            WHERE c.eligible_bridge = 1
            """
        )
        try:
            for row_json, is_available in cursor:
                hits = _json_counter(json.loads(row_json), "bridge_hits", int)
                candidate_occurrences.update(hits)
                candidate_documents.update(hits.keys())
                if is_available:
                    available_documents.update(hits.keys())
        finally:
            cursor.close()

        metadata = self.connection.execute(
            "SELECT value FROM candidate_metadata WHERE key = 'bridge_top_tokens'"
        ).fetchone()
        if metadata is None:
            top_ids = [
                token_id
                for token_id, _count in sorted(
                    candidate_occurrences.items(),
                    key=lambda item: (-item[1], item[0]),
                )[: self.config.vocab_alignment.bridge_top_token_count]
            ]
        else:
            global_top = json.loads(str(metadata[0]))
            top_ids = [int(row["old_token_id"]) for row in global_top]
        top_set = set(top_ids)
        minimum = self.config.vocab_alignment.bridge_min_contexts
        selected_documents: Counter[int] = Counter()
        selected_occurrences: Counter[int] = Counter()
        selected_records = 0
        selected_tokens = 0

        def record(rows: Iterable[Candidate]) -> None:
            seen_in_parent: set[int] = set()
            for row in rows:
                hits = _json_counter(row.row, "bridge_hits", int)
                selected_occurrences.update(
                    {token_id: count for token_id, count in hits.items() if token_id in top_set}
                )
                seen_in_parent.update(token_id for token_id in hits if token_id in top_set)
            selected_documents.update(seen_in_parent)

        for candidate in self._iter_candidates(eligible="eligible_bridge"):
            hits = _json_counter(candidate.row, "bridge_hits", int)
            if not any(
                token_id in top_set and selected_documents[token_id] < minimum
                for token_id in hits
            ):
                continue
            rows = self._feature_rows(candidate.parent_doc_id, "eligible_bridge")
            added_records, added_tokens = self._assign(
                rows,
                "train",
                "multi_hanzi_bridge",
            )
            if added_records:
                record(rows)
                selected_records += added_records
                selected_tokens += added_tokens
            if selected_tokens >= target_tokens:
                break

        if selected_tokens < target_tokens:
            for candidate in self._iter_candidates(eligible="eligible_bridge"):
                rows = self._feature_rows(candidate.parent_doc_id, "eligible_bridge")
                added_records, added_tokens = self._assign(
                    rows,
                    "train",
                    "multi_hanzi_bridge",
                )
                if added_records:
                    record(rows)
                    selected_records += added_records
                    selected_tokens += added_tokens
                if selected_tokens >= target_tokens:
                    break

        token_stats = []
        unmet = []
        for token_id in top_ids:
            required = min(minimum, available_documents[token_id])
            value = {
                "old_token_id": token_id,
                "candidate_occurrences": candidate_occurrences[token_id],
                "candidate_documents": candidate_documents[token_id],
                "available_training_documents": available_documents[token_id],
                "required_documents": required,
                "selected_documents": selected_documents[token_id],
                "selected_occurrences": selected_occurrences[token_id],
            }
            token_stats.append(value)
            if selected_documents[token_id] < required:
                unmet.append(value)

        return {
            "selection": {
                "target_tokens": target_tokens,
                "records": selected_records,
                "tokens": selected_tokens,
            },
            "coverage": {
                "passed": not unmet,
                "requested_top_tokens": self.config.vocab_alignment.bridge_top_token_count,
                "observed_tokens": len(candidate_occurrences),
                "tracked_top_tokens": len(top_ids),
                "minimum_context_documents": minimum,
                "unmet_count": len(unmet),
                "unmet_tokens": unmet,
                "token_stats": token_stats,
            },
        }

    def _load_new_hanzi(self) -> set[str]:
        path = self.config.vocab_alignment.new_hanzi_token_ids_path
        with path.open("rt", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise ValueError(f"new Hanzi metadata must be a JSON object: {path}")
        return {
            char
            for char in value.values()
            if isinstance(char, str) and len(char) == 1
        }

    def _load_priority_hanzi(self) -> set[str]:
        chars: set[str] = set()
        for path in self.config.vocab_alignment.priority_hanzi_paths:
            if not path.is_file():
                raise FileNotFoundError(f"priority Hanzi resource is missing: {path}")
            chars.update(_read_hanzi_resource(path))
        return chars


_INTERNAL_CANDIDATE_FIELDS = {
    "source_category",
    "source_quota_group",
    "candidate_pool",
    "candidate_roles",
    "eligible_bridge",
    "eligible_new_hanzi",
    "bridge_hits",
    "new_hanzi_hits",
    "language_stats",
}


def materialize_row(category: str, token_count: int, row_json: str) -> dict[str, Any]:
    row = json.loads(row_json)
    for key in _INTERNAL_CANDIDATE_FIELDS:
        row.pop(key, None)
    row["category"] = category
    row["token_count"] = int(token_count)
    return row


def selection_inventory(connection: sqlite3.Connection, split: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT s.category, c.source, c.quota_group, COUNT(*), SUM(s.token_count)
        FROM selections s
        JOIN candidates c ON c.doc_id = s.doc_id
        WHERE s.split = ?
        GROUP BY s.category, c.source, c.quota_group
        ORDER BY s.category, c.source, c.quota_group
        """,
        (split,),
    )
    return [
        {
            "category": str(category),
            "source": str(source),
            "quota_group": quota_group,
            "records": int(records),
            "tokens": int(tokens),
        }
        for category, source, quota_group, records, tokens in rows
    ]


def split_rows(
    connection: sqlite3.Connection,
    split: str,
) -> Iterator[tuple[str, int, str]]:
    return connection.execute(
        """
        SELECT s.category, s.token_count, c.row_json
        FROM selections s
        JOIN candidates c ON c.doc_id = s.doc_id
        WHERE s.split = ?
        ORDER BY c.output_key, c.doc_id
        """,
        (split,),
    )


def parent_split_overlap(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT parent_doc_id
            FROM selections
            GROUP BY parent_doc_id
            HAVING COUNT(DISTINCT split) > 1
        )
        """
    ).fetchone()
    return int(row[0])
