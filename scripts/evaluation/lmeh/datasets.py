"""Load every benchmark from the local data bundle."""
from __future__ import annotations

import csv
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

CMMLU_COLUMNS = ("Question", "A", "B", "C", "D", "Answer")


def _safe_name(value):
    if value in (None, "", "default"):
        return None
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"invalid local dataset name: {value!r}")
    return value


def load_local_parquet(data_dir, name=None, **kwargs):
    del kwargs
    root = Path(data_dir)
    name = _safe_name(name)
    if name is not None:
        root /= name
    if not root.is_dir():
        raise FileNotFoundError(f"local benchmark directory is missing: {root}")

    data_files = {}
    for split in ("train", "validation", "test", "dev", "auxiliary_train"):
        files = sorted((root / split).glob("*.parquet")) if (root / split).is_dir() else []
        files.extend(sorted(root.glob(f"{split}*.parquet")))
        if files:
            data_files[split] = [str(path) for path in files]
    if not data_files:
        raise FileNotFoundError(f"no local Parquet benchmark splits found under {root}")
    return load_dataset("parquet", data_files=data_files)


def load_cmmlu(subject, data_dir, **kwargs):
    del kwargs
    if not subject or Path(subject).name != subject or subject in {".", ".."}:
        raise ValueError("invalid CMMLU subject")
    splits = {}
    for split in ("dev", "test"):
        path = Path(data_dir) / split / f"{subject}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing CMMLU {split}: {path}; check data/evaluation/lmeh/cmmlu")
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
