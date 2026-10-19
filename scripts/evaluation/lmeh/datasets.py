"""Local CMMLU CSV loading without numeric coercion of answer choices."""
from __future__ import annotations

import csv
from pathlib import Path

from datasets import Dataset, DatasetDict

from ..common import sha256

CMMLU_COLUMNS = ("Question", "A", "B", "C", "D", "Answer")


def load_cmmlu(subject, data_dir, **kwargs):
    del kwargs
    if not subject or Path(subject).name != subject or subject in {".", ".."}:
        raise ValueError("invalid CMMLU subject")
    splits = {}
    for split in ("dev", "test"):
        path = Path(data_dir) / split / f"{subject}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing CMMLU {split}: {path}; set CMMLU_DIR to the directory containing dev/ and test/")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(CMMLU_COLUMNS).issubset(reader.fieldnames or []):
                raise ValueError(f"invalid CMMLU columns: {path}")
            rows = [{key: row[key] for key in CMMLU_COLUMNS} for row in reader]
        if not rows or any(any(value is None for value in row.values()) or
                           row["Answer"] not in {"A", "B", "C", "D"} for row in rows):
            raise ValueError(f"invalid CMMLU rows/answers: {path}")
        if split == "dev" and len(rows) < 5:
            raise ValueError(f"CMMLU 5-shot requires at least five dev examples: {path}")
        splits[split] = Dataset.from_list(rows)
    return DatasetDict(splits)


def cmmlu_files(data_dir):
    return {str(p.relative_to(data_dir)): sha256(p) for split in ("dev", "test")
            for p in sorted((Path(data_dir) / split).glob("*.csv"))}
