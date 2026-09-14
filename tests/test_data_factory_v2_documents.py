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
    adapt_records,
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

    def test_stack_v3_expands_only_permissive_non_vendor_files(self) -> None:
        source = SourceSpec(
            name="the_stack_v3", reader="parquet", adapter="stack_v3_train",
            paths=("unused.parquet",), phases=frozenset({"phase1", "phase2"}),
            license="", license_mode="per_record", quality_profile="code",
            default_domain="code",
            metadata={"require_permissive_license": True},
        )
        raw = RawRecord(
            row={
                "repo_path": "owner/repo", "repo_id": 7, "commit_id": "abc123",
                "files": [
                    {"content_id": "one", "content": "print('ok')", "file_path": "main.py", "language": "Python", "is_vendor": False, "license_type": "permissive", "detected_licenses": ["MIT"]},
                    {"content_id": "two", "content": "vendor", "file_path": "vendor/x.py", "language": "Python", "is_vendor": True, "license_type": "permissive", "detected_licenses": ["MIT"]},
                    {"content_id": "three", "content": "unknown", "file_path": "x.py", "language": "Python", "is_vendor": False, "license_type": "no_license", "detected_licenses": []},
                ],
            },
            path=Path("stack.parquet"), row_index=0,
        )
        rejected: list[str] = []
        records = list(adapt_records(source, raw, rejected.append))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].text, "print('ok')")
        self.assertEqual(records[0].license, "MIT")
        self.assertEqual(records[0].revision, "abc123")
        self.assertEqual(records[0].source_path, "main.py")
        self.assertEqual(rejected, ["stack_vendor_file", "stack_non_permissive"])
    def test_pes2o_keeps_only_s2orc_full_text(self) -> None:
        source = SourceSpec(
            name="peS2o", reader="zstd_jsonl", adapter="pes2o",
            paths=("unused.zst",), phases=frozenset({"phase2"}),
            license="ODC-By-1.0", license_mode="fixed",
            quality_profile="scientific", default_domain="scientific",
            metadata={"required_source": "s2orc"},
        )
        accepted = RawRecord(
            row={"id": "1", "source": "s2orc", "text": "full paper text"},
            path=Path("sample.zst"), row_index=0,
        )
        rejected = RawRecord(
            row={"id": "2", "source": "s2ag", "text": "abstract"},
            path=Path("sample.zst"), row_index=1,
        )
        self.assertEqual(adapt_record(source, accepted).text, "full paper text")
        with self.assertRaisesRegex(SourceRecordError, "s2ag"):
            adapt_record(source, rejected)

    def test_plain_text_adapter_uses_whole_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classic.txt"
            path.write_text("學而時習之，不亦說乎。此為完整古文測試內容。", encoding="utf-8")
            source = source_spec(
                name="ect_krp", reader="text", adapter="plain_text",
                paths=(str(path),), license_value="CC-BY-SA-4.0",
            )
            report = inspect_source(source, max_files=1, max_rows=1)
        self.assertTrue(report["passed"])
        self.assertIn("學而時習之", report["samples"][0]["text_preview"])

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
