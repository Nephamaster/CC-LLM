"""Small runtime inputs from real data; original files and archives stay unchanged."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice, zip_longest
import json
from pathlib import Path
import tarfile

from scripts.evaluation.common import iter_jsonl, prepare_file, write_json


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    if not count:
        raise ValueError(f"no smoke samples: {path}")
    return count


def pretrain_rows(directory):
    paths = sorted(directory.glob("*.jsonl"))
    if paths:
        for path in paths:
            yield from iter_jsonl(path)
        return
    for archive in sorted(directory.glob("*.tar.xz")):
        with tarfile.open(archive, "r|xz") as bundle:
            for member in bundle:
                if member.isfile() and member.name.endswith(".jsonl"):
                    with bundle.extractfile(member) as raw:
                        for line in raw:
                            if line.strip():
                                yield json.loads(line.decode("utf-8-sig"))


def text_samples(rows, count):
    kept = 0
    for row in rows:
        messages = row.get("messages")
        if messages is not None:
            if len(messages) != 1 or messages[0].get("role") != "assistant":
                raise ValueError("expected one assistant message in exported pretraining data")
            text = messages[0]["content"]
        else:
            text = row["text"]
        if not isinstance(text, str):
            raise ValueError("pretraining/PPL text must be a string")
        if len(text.strip()) < 2:
            continue
        yield ({"messages": [{"role": "assistant", "content": text[:512]}]}
               if messages is not None else {"text": text[:512]})
        kept += 1
        if kept == count:
            return


def ancient_samples(directory, output, split, per_task):
    counts = Counter()
    tasks = {"c2m", "m2c", "punctuate"}
    source = directory / f"{split}.jsonl"
    metadata = directory / f"{split}.metadata.jsonl"
    selected, selected_metadata = [], []
    for row, meta in zip_longest(iter_jsonl(source), iter_jsonl(metadata)):
        if row is None or meta is None or row["messages"] != meta["messages"]:
            raise ValueError(f"ancient data/metadata mismatch in {split}")
        task = meta["task"]
        if task not in tasks:
            raise ValueError(f"unknown ancient task: {task}")
        if counts[task] < per_task:
            selected.append(row)
            selected_metadata.append(meta)
            counts[task] += 1
        if all(counts[t] >= per_task for t in tasks):
            break
    if set(counts) != tasks:
        raise ValueError(f"ancient sample is missing tasks: {dict(counts)}")
    write_rows(output / f"{split}.jsonl", selected)
    write_rows(output / f"{split}.metadata.jsonl", selected_metadata)
    return dict(counts)


def prepare(kind, data, output, train_samples, eval_samples):
    if kind in {"phase1", "phase2"}:
        source = data / "pretrain" / kind
        return {split: write_rows(output / split / "samples.jsonl", text_samples(
            pretrain_rows(source / split), train_samples if split == "train" else eval_samples))
                for split in ("train", "validation")}
    if kind == "instruction_sft":
        pointer = data / "instruction/prepared/latest.txt"
        directory = Path(pointer.read_text().strip()) if pointer.is_file() else data / "instruction/final"
        return {split: write_rows(output / f"{split}.jsonl", islice(iter_jsonl(directory / f"{split}.jsonl"), count))
                for split, count in (("train", train_samples), ("dev", eval_samples))}
    if kind in {"csc_sft", "cgec_sft"}:
        task = kind.removesuffix("_sft")
        filenames = ("csc_mix_dedup.jsonl", "cscd_dev.jsonl") if task == "csc" else ("train.jsonl", "validation.jsonl")
        return {split: write_rows(output / f"{split}.jsonl", islice(
            iter_jsonl(data / "downstream" / task / name), count))
                for split, name, count in zip(("train", "dev"), filenames, (train_samples, eval_samples))}
    if kind == "ancient_sft":
        return {split: ancient_samples(data / "downstream/ancient", output, split, count)
                for split, count in (("train", max(1, train_samples // 3)), ("dev", eval_samples))}
    if kind == "ppl":
        return {path.stem: write_rows(output / path.name, text_samples(iter_jsonl(path), eval_samples))
                for path in sorted((data / "evaluation/ppl").glob("*.jsonl"))}

    from scripts.evaluation.csc.prepare import FILES, rows as csc_rows
    from scripts.evaluation.cgec.prepare import fcgec_rows, nacgec_rows
    from scripts.evaluation.ancient.prepare import C3_PROMPTS, FUXI_TASKS, c3_rows, fuxi_rows

    evaluation = data / "evaluation"
    if kind == "csc":
        paths = [evaluation / "csc" / f"{name}.jsonl" for name in FILES]
        rows = (row for path in paths for row in islice(csc_rows([path]), eval_samples))
    elif kind.startswith("fcgec_"):
        split = kind.removeprefix("fcgec_")
        paths = [evaluation / "cgec/FCGEC" / f"FCGEC_{split}.json"]
        rows = islice(fcgec_rows(paths[0], split), eval_samples)
    elif kind in {"nacgec", "nacgec_test"}:
        labeled = kind == "nacgec"
        paths = [evaluation / "cgec/NaCGEC/data" / ("nacgec.dev.ref.para" if labeled else "nacgec.test.input")]
        rows = islice(nacgec_rows(paths[0], "dev" if labeled else "test", labeled=labeled), eval_samples)
    elif kind == "c3":
        paths = [evaluation / "ancient/c3bench/C3_bench.json"]
        rows = islice(c3_rows(paths[0]), eval_samples * len(C3_PROMPTS))
    elif kind == "fuxi":
        directory = evaluation / "ancient/FuxiBench"
        paths = [directory / "test_data" / f"{name}.json" for name in FUXI_TASKS]
        counts = Counter()

        def selected():
            for row in fuxi_rows(directory):
                if counts[row["task"]] < eval_samples:
                    counts[row["task"]] += 1
                    yield row

        rows = selected()
    else:
        raise ValueError(f"unknown sample kind: {kind}")
    return prepare_file(rows, output / "input.jsonl", paths, f"smoke/{kind}; not a full benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=1)
    args = parser.parse_args()
    if min(args.train_samples, args.eval_samples) < 1:
        parser.error("sample counts must be positive")
    report = prepare(args.kind, args.data_root, args.output, args.train_samples, args.eval_samples)
    if not report:
        raise ValueError(f"no data for {args.kind}")
    write_json(args.output / "smoke_samples.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
