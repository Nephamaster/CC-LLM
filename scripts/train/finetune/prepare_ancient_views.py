"""Build minimal task-labelled runtime views without changing accepted ancient data."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from itertools import zip_longest
from pathlib import Path


TASKS = ("c2m", "m2c", "punctuate")


def _read_json(line: str, path: Path, line_number: int) -> dict:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}:{line_number}")
    return value


def build_view(messages_path: Path, metadata_path: Path, output_path: Path,
               expected: dict[str, int] | None = None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with messages_path.open(encoding="utf-8") as messages_file, \
                metadata_path.open(encoding="utf-8") as metadata_file, \
                temporary.open("w", encoding="utf-8") as output_file:
            pairs = zip_longest(messages_file, metadata_file)
            for line_number, pair in enumerate(pairs, 1):
                messages_line, metadata_line = pair
                if messages_line is None or metadata_line is None:
                    raise ValueError(f"row count mismatch: {messages_path} vs {metadata_path}")
                messages_row = _read_json(messages_line, messages_path, line_number)
                metadata_row = _read_json(metadata_line, metadata_path, line_number)
                if messages_row.get("messages") != metadata_row.get("messages"):
                    raise ValueError(f"messages mismatch at row {line_number}: {messages_path} vs {metadata_path}")
                task = metadata_row.get("task")
                if task not in TASKS:
                    raise ValueError(f"unsupported ancient task {task!r} in {metadata_path}:{line_number}")
                output_file.write(json.dumps({"messages": messages_row["messages"], "channel": task},
                                             ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[task] += 1
        if expected is not None and dict(counts) != expected:
            raise ValueError(f"task counts do not match the accepted report: {dict(counts)} != {expected}")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(counts)


def prepare(data_dir: Path, output_dir: Path) -> dict:
    report_path = data_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("ready_for_training"):
        raise ValueError(f"ancient dataset is not accepted for training: {report_path}")

    counts = {}
    for split in ("train", "dev"):
        expected = {
            task: report["tasks"][f"{split}/{task}"]["selected"]
            for task in TASKS
        }
        split_counts = build_view(data_dir / f"{split}.jsonl", data_dir / f"{split}.metadata.jsonl",
                                  output_dir / f"{split}.jsonl", expected)
        counts[split] = split_counts

    view_report = {
        "schema": "cc-llm-ancient-runtime-view-v1",
        "source": str(data_dir.resolve()),
        "channels": list(TASKS),
        "counts": counts,
    }
    report_output = output_dir / "view_report.json"
    temporary = report_output.with_name(f".{report_output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(view_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_output)
    return view_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
