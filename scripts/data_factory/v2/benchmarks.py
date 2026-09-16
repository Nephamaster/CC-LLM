"""Readers for benchmark datasets used by decontamination."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


def _iter_json_records(value: Any, text_fields: Sequence[str]) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_json_records(item, text_fields)
        return
    if not isinstance(value, dict):
        return
    if any(field in value for field in text_fields):
        yield value
        return
    for key, item in value.items():
        for row in _iter_json_records(item, text_fields):
            if "id" not in row:
                row = {"id": str(key), **row}
            yield row


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rt", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _iter_csv(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _iter_delimited(path: Path, benchmark_name: str) -> Iterator[dict[str, Any]]:
    with path.open("rt", encoding="utf-8-sig", newline="") as handle:
        for line_number, values in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not values or not any(value.strip() for value in values):
                continue
            if benchmark_name == "csc":
                if len(values) < 3:
                    raise ValueError(f"{path}:{line_number} must contain label, source, target")
                yield {"label": values[0], "source": values[1], "target": values[2:]}
            elif benchmark_name == "cgec":
                if len(values) < 3:
                    raise ValueError(f"{path}:{line_number} must contain id, source, target")
                yield {"id": values[0], "source": values[1], "target": values[2:]}
            else:
                raise ValueError(f"unsupported tabular benchmark {benchmark_name!r}: {path}")


def _iter_text(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rt", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield {"text": text}


def read_benchmark_rows(
    path: Path,
    *,
    benchmark_name: str,
    text_fields: Sequence[str],
) -> Iterator[dict[str, Any]]:
    """Yield normalized row dictionaries without loading every format as JSONL."""
    if not path.is_file():
        raise ValueError(f"benchmark path is not a file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        yield from pq.read_table(path).to_pylist()
    elif suffix == ".csv":
        yield from _iter_csv(path)
    elif suffix in {".tsv", ".para"}:
        yield from _iter_delimited(path, benchmark_name)
    elif suffix == ".txt":
        yield from _iter_text(path)
    elif suffix == ".jsonl":
        yield from _iter_jsonl(path)
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        yield from _iter_json_records(value, text_fields)
    else:
        raise ValueError(f"unsupported benchmark file format {suffix!r}: {path}")
