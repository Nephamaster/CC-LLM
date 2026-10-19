"""Convert CGEC raw pairs to CSC-style messages JSONL, excluding evaluation sources."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from scripts.evaluation.cgec.prepare import PROMPT

ROOT = Path(__file__).resolve().parents[2]


def read_pairs(path, stats):
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            stats["input_rows"] += 1
            fields = [field.strip() for field in line.rstrip("\r\n").split("\t")]
            if path.name == "train_CGED.txt":
                # Some CGED rows pad both sentences with empty TAB columns.
                if any(not field for field in fields):
                    stats["blank_tab_rows_fixed"] += 1
                    fields = [field for field in fields if field]
                if (len(fields) == 4 and fields[0] == fields[2]
                        and re.fullmatch(r"[0-9][A-Za-z0-9_-]*", fields[0])):
                    stats["id_rows_fixed"] += 1
                    fields = [fields[1], fields[3]]
            targets = [field for field in fields[1:] if field]
            if not fields or not fields[0] or not targets:
                raise ValueError(f"missing source/target: {path}:{line_number}")
            yield fields[0], targets[0]


def benchmark_sources(root):
    result = {}
    for split in ("valid", "test"):
        path = root / "test/FCGEC" / f"FCGEC_{split}.json"
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        result[f"fcgec_{split}"] = {row["sentence"].strip() for row in data.values()}
    for split in ("dev", "test"):
        path = root / "test/NaCGEC/data" / f"nacgec.{split}.ref.para"
        with path.open(encoding="utf-8-sig") as handle:
            result[f"nacgec_{split}"] = {
                line.rstrip("\r\n").split("\t")[1].strip() for line in handle if line.strip()
            }
    return result


def messages(source, target):
    return {"messages": [{"role": "user", "content": PROMPT + source},
                         {"role": "assistant", "content": target}]}


def prepare(root):
    raw, output = root / "sft/raw", root / "sft/final"
    paths = [output / name for name in ("train.jsonl", "validation.jsonl", "prepare_report.json")]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    valid_stats = Counter()
    validation = {}
    for source, target in read_pairs(raw / "valid_MuCGEC.txt", valid_stats):
        if source in validation:
            valid_stats["duplicate_sources_dropped"] += 1
        else:
            validation[source] = target
    valid_stats["output_rows"] = len(validation)

    blocked = benchmark_sources(root)
    blocked["sft_validation"] = set(validation)
    train, seen, source_stats = {}, set(), {}
    for name in ("train_CGED.txt", "train_FCGEC.txt"):
        stats, overlaps = Counter(), Counter()
        for source, target in read_pairs(raw / name, stats):
            if source in seen:
                stats["duplicate_sources_dropped"] += 1
                continue
            seen.add(source)
            matches = [key for key, values in blocked.items() if source in values]
            if matches:
                stats["overlapping_sources_dropped"] += 1
                overlaps.update(matches)
                continue
            train[source] = target
        stats["output_rows"] = stats["input_rows"] - stats["duplicate_sources_dropped"] - stats["overlapping_sources_dropped"]
        source_stats[name] = {**stats, "overlap_by_split_nonadditive": dict(overlaps)}

    report = {"train_rows": len(train), "validation_rows": len(validation),
              "training_sources": source_stats, "validation_source": dict(valid_stats),
              "excluded_file": "nacgec_all_type.json",
              "reference_policy": "first nonempty target; first occurrence of each source",
              "source_matching": "exact text after stripping outer whitespace; no other normalization",
              "training_source_order": ["train_CGED.txt", "train_FCGEC.txt"],
              "prompt": PROMPT}
    output.mkdir(parents=True, exist_ok=True)
    for path, pairs in zip(paths[:2], (train, validation)):
        with path.open("x", encoding="utf-8") as handle:
            for source, target in pairs.items():
                handle.write(json.dumps(messages(source, target), ensure_ascii=False) + "\n")
    with paths[2].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/downstream/cgec")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
