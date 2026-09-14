from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts.data_factory.config import load_config
from scripts.data_factory.prescan import (
    DocumentIndex,
    _preselection_targets,
    _select_simple_documents,
    iter_selected_document_batches,
    refill_documents,
)


class CandidateBudgetTest(unittest.TestCase):
    def test_preselection_targets_match_candidate_budget(self) -> None:
        config = load_config(Path("scripts/data_factory/phase1_config.json"))
        targets = _preselection_targets(config, config.candidate_tokens)

        self.assertEqual(sum(targets.values()), 1_100_000_000)
        self.assertTrue(all(tokens > 0 for tokens in targets.values()))

class DocumentIndexTest(unittest.TestCase):
    def test_merge_preselect_and_offset_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_path = root / "part-00000.jsonl"
            source_rows = [
                {
                    "doc_id": f"doc-{index}",
                    "source": source,
                    "category": "chinese_general",
                    "text": f"text-{index}",
                }
                for index, source in enumerate(("a", "a", "b", "c"))
            ]
            raw_lines = [
                (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                for row in source_rows
            ]
            input_path.write_bytes(b"".join(raw_lines))

            shard_path = root / "scan.sqlite"
            shard = sqlite3.connect(shard_path)
            shard.executescript(
                """
                CREATE TABLE documents (
                    doc_id TEXT PRIMARY KEY,
                    input_path TEXT NOT NULL,
                    byte_offset INTEGER NOT NULL,
                    byte_length INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    pool TEXT NOT NULL,
                    quota_group TEXT,
                    estimated_tokens INTEGER NOT NULL,
                    sample_key INTEGER NOT NULL,
                    output_key INTEGER NOT NULL,
                    new_hanzi_hits TEXT NOT NULL
                );
                CREATE TABLE bridge_frequency (
                    token_id INTEGER PRIMARY KEY,
                    occurrences INTEGER NOT NULL,
                    documents INTEGER NOT NULL
                );
                """
            )
            offset = 0
            for index, (row, raw_line) in enumerate(zip(source_rows, raw_lines)):
                shard.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["doc_id"],
                        str(input_path.resolve()),
                        offset,
                        len(raw_line),
                        row["source"],
                        "chinese_natural",
                        None,
                        10,
                        index,
                        index,
                        "{}",
                    ),
                )
                offset += len(raw_line)
            shard.execute("INSERT INTO bridge_frequency VALUES (7, 3, 2)")
            shard.commit()
            shard.close()

            index = DocumentIndex(root / "documents.sqlite")
            try:
                index.merge_scan_shard(input_path, shard_path, {"accepted_records": 4})
                self.assertEqual(index.top_bridge_tokens(1)[0]["old_token_id"], 7)

                source_tokens: Counter[str] = Counter()
                report = _select_simple_documents(
                    index,
                    intent="chinese_natural",
                    target_tokens=30,
                    pool="chinese_natural",
                    source_tokens=source_tokens,
                    source_token_cap=10,
                )
                self.assertEqual(report["estimated_tokens"], 30)
                self.assertEqual(source_tokens, {"a": 10, "b": 10, "c": 10})

                batches = list(
                    iter_selected_document_batches(
                        index,
                        max_records=2,
                        max_chars=100,
                    )
                )
                selected = [item for batch in batches for item in batch]
                self.assertEqual(len(selected), 3)
                self.assertTrue(
                    all(intent == "chinese_natural" for _row, intent in selected)
                )
                index.mark_materialized([selected[0][0]["doc_id"]])
                remaining = sum(
                    len(batch)
                    for batch in iter_selected_document_batches(
                        index,
                        max_records=10,
                        max_chars=100,
                    )
                )
                self.assertEqual(remaining, 2)
            finally:
                index.close()

    def test_refill_uses_only_unselected_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            index = DocumentIndex(root / "documents.sqlite")
            try:
                rows = [
                    (
                        f"doc-{item}",
                        str(root / "normalized.jsonl"),
                        item,
                        1,
                        "source",
                        "non_chinese",
                        None,
                        10,
                        item,
                        item,
                        "{}",
                        "non_chinese" if item == 0 else None,
                    )
                    for item in range(3)
                ]
                index.connection.executemany(
                    """
                    INSERT INTO documents (
                        doc_id, input_path, byte_offset, byte_length, source,
                        pool, quota_group, estimated_tokens, sample_key,
                        output_key, new_hanzi_hits, selected_intent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                index.connection.commit()

                report = refill_documents(
                    index,
                    SimpleNamespace(),
                    [],
                    {"non_chinese": 12},
                )

                self.assertEqual(report["added_documents"], 2)
                self.assertEqual(
                    report["selection"]["non_chinese"]["estimated_tokens"],
                    20,
                )
                self.assertEqual(index.selected_count(), 3)
            finally:
                index.close()

if __name__ == "__main__":
    unittest.main()