from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_factory.v2.config import SourceSpec
from scripts.data_factory.v2.documents import (
    RawRecord,
    SourceRecordError,
    adapt_record,
    clean_and_tag,
    inspect_source,
    normalize_text,
)


def source_spec(
    *,
    name: str = "test",
    reader: str = "jsonl",
    adapter: str = "generic_text",
    paths: tuple[str, ...] = ("unused.jsonl",),
    license_value: str = "Apache-2.0",
    default_domain: str = "general",
) -> SourceSpec:
    return SourceSpec(
        name=name,
        reader=reader,
        adapter=adapter,
        paths=paths,
        phases=frozenset({"phase1", "phase2"}),
        license=license_value,
        license_mode="fixed",
        quality_profile="curated_zh",
        default_domain=default_domain,
        metadata={},
    )


class DataFactoryV2DocumentsTest(unittest.TestCase):
    def test_normalization_preserves_character_semantics(self) -> None:
        self.assertEqual(normalize_text("繁體\r\n中文\u200b\x00"), "繁體\n中文")

    def test_cci_adapter_and_tagging(self) -> None:
        source = source_spec(name="cci3_hq", adapter="cci3_hq")
        raw = RawRecord(
            row={"id": "abc", "text": "这是用于测试的高质量中文自然文本，内容完整并且长度足够。"},
            path=Path("part_000001.jsonl"),
            row_index=0,
        )

        canonical = clean_and_tag(source, adapt_record(source, raw))

        self.assertEqual(canonical.doc_id, "cci3_hq:abc")
        self.assertEqual(canonical.metadata["language"], "zh")
        self.assertEqual(canonical.metadata["license"], "Apache-2.0")

    def test_wanjuan_exam_keeps_question_options_and_answer(self) -> None:
        source = source_spec(name="wanjuan", reader="tar_jsonl", adapter="wanjuan")
        raw = RawRecord(
            row={
                "id": "q1",
                "q_main": "下列哪项正确？",
                "option_a": "选项甲",
                "option_b": "选项乙",
                "std_ans": "A",
                "answer_detail": "因为甲符合条件。",
            },
            path=Path("Exam-cn/part-000001.jsonl.tar.gz"),
            row_index=0,
            member="part-000001.jsonl",
        )

        adapted = adapt_record(source, raw)

        self.assertIn("A. 选项甲", adapted.text)
        self.assertIn("答案：A", adapted.text)
        self.assertIn("因为甲符合条件。", adapted.text)

    def test_stack_ids_without_content_fail_explicitly(self) -> None:
        source = source_spec(
            name="the_stack_v2",
            reader="parquet",
            adapter="the_stack_v2",
            license_value="",
            default_domain="code",
        )
        raw = RawRecord(
            row={"swhid": "swh:1:cnt:abc", "license": "MIT"},
            path=Path("ids.parquet"),
            row_index=0,
        )

        with self.assertRaisesRegex(SourceRecordError, "no code content"):
            adapt_record(source, raw)

    def test_inspection_reports_real_jsonl_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text(
                json.dumps(
                    {"id": "1", "text": "这是一条长度足够的中文检查样本文本，用于验证检查报告。"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            source = source_spec(paths=(str(path),))

            report = inspect_source(source, max_files=1, max_rows=1)

        self.assertTrue(report["passed"])
        self.assertEqual(report["accepted_rows"], 1)
        self.assertIn("text", report["raw_fields"])


if __name__ == "__main__":
    unittest.main()
