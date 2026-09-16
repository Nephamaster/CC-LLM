from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_factory.v2.benchmarks import read_benchmark_rows


class BenchmarkReaderTest(unittest.TestCase):
    def read(self, path: Path, name: str, fields: tuple[str, ...]) -> list[dict]:
        return list(read_benchmark_rows(path, benchmark_name=name, text_fields=fields))

    def test_csv_with_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            path.write_text("Question,A,Answer\nquestion,choice,A\n", encoding="utf-8")
            rows = self.read(path, "cmmlu", ("Question", "A", "Answer"))
        self.assertEqual(rows, [{"Question": "question", "A": "choice", "Answer": "A"}])

    def test_csc_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.tsv"
            path.write_text("1\toriginal\tcorrected\n", encoding="utf-8")
            rows = self.read(path, "csc", ("source", "target"))
        self.assertEqual(rows[0]["source"], "original")
        self.assertEqual(rows[0]["target"], ["corrected"])

    def test_cgec_para_with_multiple_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ref.para"
            path.write_text("1\tsource\ttarget one\ttarget two\n", encoding="utf-8")
            rows = self.read(path, "cgec", ("source", "target"))
        self.assertEqual(rows[0]["target"], ["target one", "target two"])

    def test_plain_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lemon.txt"
            path.write_text("first pair\n\nsecond pair\n", encoding="utf-8")
            rows = self.read(path, "csc", ("text",))
        self.assertEqual(rows, [{"text": "first pair"}, {"text": "second pair"}])

    def test_json_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fcgec.json"
            path.write_text(json.dumps({"sample-id": {"sentence": "sample"}}), encoding="utf-8")
            rows = self.read(path, "cgec", ("sentence",))
        self.assertEqual(rows, [{"id": "sample-id", "sentence": "sample"}])

    def test_json_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fuxi.json"
            value = [{"instruction": "task", "input": "question", "output": "answer"}]
            path.write_text(json.dumps(value), encoding="utf-8")
            rows = self.read(path, "fuxi", ("instruction", "input", "output"))
        self.assertEqual(rows, value)


if __name__ == "__main__":
    unittest.main()
