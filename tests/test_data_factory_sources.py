from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.data_factory.sources import iter_cci3_hq, iter_wanjuan


class SourceAdapterTests(unittest.TestCase):
    def test_cci3_hq_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            path = root / "part_000000.jsonl"
            path.write_text(
                json.dumps({"id": "doc-1", "text": "中文正文", "score": 3.5}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            rows = list(
                iter_cci3_hq(
                    SimpleNamespace(repo_root=root),
                    {"paths": [str(root / "part_*.jsonl")], "license": "Apache-2.0"},
                )
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "中文正文")
        self.assertEqual(rows[0]["quality_score"], 3.5)
        self.assertEqual(rows[0]["category"], "chinese_natural")
        self.assertEqual(rows[0]["doc_id"], "cci3_hq-doc_1")

    def test_wanjuan_archive_and_exam_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            exam_path = root / "Exam-cn" / "part-000000-000001.jsonl.tar.gz"
            law_path = root / "Law-cn" / "part-000000-000001.jsonl.tar.gz"
            self._write_archive(
                exam_path,
                [{"id": "exam-1", "q_main": "题目", "answer_detail": "解析"}],
            )
            self._write_archive(law_path, [{"id": "law-1", "content": "法律正文"}])

            rows = list(
                iter_wanjuan(
                    SimpleNamespace(repo_root=root),
                    {
                        "paths": [str(root / "*" / "*.jsonl.tar.gz")],
                        "license": "CC-BY-4.0",
                    },
                )
            )

        by_subset = {row["subset"]: row for row in rows}
        self.assertEqual(by_subset["Exam_cn"]["text"], "题目\n解析")
        self.assertEqual(by_subset["Law_cn"]["text"], "法律正文")
        self.assertEqual(by_subset["Exam_cn"]["category"], "chinese_natural")
        self.assertEqual(by_subset["Law_cn"]["license"], "CC-BY-4.0")

    @staticmethod
    def _write_archive(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        info = tarfile.TarInfo("data.jsonl")
        info.size = len(payload)
        with tarfile.open(path, "w:gz") as archive:
            archive.addfile(info, io.BytesIO(payload))


if __name__ == "__main__":
    unittest.main()
