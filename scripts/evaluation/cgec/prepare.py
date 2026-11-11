"""Read official local FCGEC/NaCGEC files without changing their splits."""
import argparse
import itertools
import json
import re
from pathlib import Path

from ..common import ROOT, iter_jsonl, prepare_file


def fcgec_references(sentence, operation):
    operations = json.loads(operation) if isinstance(operation, str) else operation
    if not isinstance(operations, list) or not operations:
        raise ValueError("erroneous FCGEC sentence has no reference operations")
    references = []
    for ops in operations:
        if not isinstance(ops, dict) or set(ops) - {"Switch", "Delete", "Insert", "Modify"}:
            raise ValueError(f"unrecognized FCGEC operation: {ops}")
        permutation = ops.get("Switch", list(range(len(sentence))))
        if any(not isinstance(i, int) or not 0 <= i < len(sentence) for i in permutation):
            raise ValueError("FCGEC Switch index out of range")
        source = "".join(sentence[i] for i in permutation)
        choices = []
        for kind in ("Insert", "Modify"):
            for op in ops.get(kind, []):
                labels = op["label"] if isinstance(op["label"], list) else [op["label"]]
                if not labels or not all(isinstance(label, str) for label in labels):
                    raise ValueError("invalid FCGEC label alternatives")
                choices.append([(kind, {**op, "label": label}) for label in labels])
        combinations = 1
        for alternatives in choices:
            combinations *= len(alternatives)
        if combinations > 10000:
            raise ValueError("too many FCGEC reference combinations; do not silently truncate gold")
        for variant in itertools.product(*choices):
            deleted = set(ops.get("Delete", []))
            if any(not 0 <= i < len(source) for i in deleted):
                raise ValueError("FCGEC deletion index out of range")
            tags = [("D", "") if i in deleted else ("O", "") for i in range(len(source))]
            head = ""
            for kind, op in variant:
                pos = op["pos"]
                if kind == "Insert" and pos == -1:
                    head += op["label"]
                    continue
                if not 0 <= pos < len(source):
                    raise ValueError("FCGEC edit index out of range")
                if not re.fullmatch(r"INS_.+|MOD_\d+(?:\+(?:INS|DEL)_\d+)?", op["tag"]):
                    raise ValueError(f"invalid FCGEC tag: {op['tag']}")
                # Upstream precedence: Delete -> Insert -> Modify. Some official
                # records overlap, or put INS tags inside the Modify list.
                tags[pos] = (op["tag"], op["label"])
            parts, i = [head], 0
            while i < len(source):
                tag, label = tags[i]
                if tag.startswith("MOD"):
                    width = int(tag.split("+", 1)[0].split("_")[1])
                    if width < 1 or i + width > len(source):
                        raise ValueError("FCGEC replacement extends beyond source")
                    parts.append(label)
                    i += width
                else:
                    if tag == "O":
                        parts.append(source[i])
                    elif tag.startswith("INS"):
                        parts.extend([source[i], label])
                    i += 1
            result = "".join(parts)
            if result not in references:
                references.append(result)
    return references


def fcgec_rows(path, split):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    for uid, item in data.items():
        source = item["sentence"]
        labeled = "error_flag" in item and "operation" in item
        if split == "valid" and not labeled:
            raise ValueError(f"FCGEC valid is missing labels: {uid}")
        references = None
        if labeled:
            references = [source] if item["error_flag"] == 0 else fcgec_references(source, item["operation"])
        yield {"id": f"fcgec/{split}/{uid}", "original_id": uid, "dataset": "fcgec", "task": "cgec",
               "split": split, "src": source, "references": references, "error_flag": item.get("error_flag"),
               "messages": [{"role": "user", "content": source}]}


def nacgec_rows(path, split, labeled=True):
    with path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if not labeled:
                if len(fields) != 1 or not fields[0]:
                    raise ValueError(f"invalid NaCGEC test input at line {index}: {path}")
                uid, source, references = str(index), fields[0], None
            elif len(fields) < 3 or not fields[1]:
                raise ValueError(f"invalid NaCGEC parallel row: {path}")
            else:
                uid, source, *references = fields
            yield {"id": f"nacgec/{split}/{uid}", "original_id": uid, "dataset": "nacgec", "task": "cgec",
                   "split": split, "src": source, "references": references,
                   "reference_origin": "explicit_test_reference" if split == "test" and labeled else "public_dev" if labeled else None,
                   "messages": [{"role": "user", "content": source}]}


def validate_input(path, dataset, split, explicit_gold=False):
    labeled = (split != "test") or explicit_gold
    count = 0
    for row in iter_jsonl(path):
        count += 1
        if row.get("dataset") != dataset or row.get("split") != split:
            raise ValueError("CGEC dataset/split does not match the requested run; prepare a new run")
        if (row.get("references") is not None) != labeled:
            raise ValueError("CGEC gold policy differs from requested Dev/Test export; prepare a new run")
        if dataset == "nacgec" and split == "test" and labeled and row.get("reference_origin") != "explicit_test_reference":
            raise ValueError("old implicit NaCGEC test gold: prepare with --reference-file explicitly")
    if not count:
        raise ValueError("empty CGEC input")


def nacgec_test_rows(input_path, reference_path=None):
    inputs = nacgec_rows(input_path, "test", labeled=False)
    if reference_path is None:
        yield from inputs
        return
    references = nacgec_rows(reference_path, "test")
    for source, gold in itertools.zip_longest(inputs, references):
        if source is None or gold is None or source["src"] != gold["src"] or source["original_id"] != gold["original_id"]:
            raise ValueError("NaCGEC explicit gold must match official test input order, IDs and sources")
        yield {**source, "references": gold["references"], "reference_origin": "explicit_test_reference"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fcgec", "nacgec"), required=True)
    parser.add_argument("--split", choices=("dev", "valid", "test"), default="dev")
    parser.add_argument("--reference-file", type=Path, help="explicit published NaCGEC test references; otherwise export only")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/evaluation/cgec")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.reference_file is not None and (args.dataset != "nacgec" or args.split != "test"):
        parser.error("--reference-file is only for explicit NaCGEC test scoring")
    args.split = ("valid" if args.split == "dev" else args.split) if args.dataset == "fcgec" else ("dev" if args.split == "valid" else args.split)
    if args.output.exists():
        validate_input(args.output, args.dataset, args.split, args.reference_file is not None)
    if args.check_only:
        if not args.output.exists():
            raise FileNotFoundError(args.output)
        return
    if args.dataset == "fcgec":
        path = args.data_dir / "FCGEC" / f"FCGEC_{args.split}.json"
        rows = fcgec_rows(path, args.split)
        sources = [path]
    else:
        directory = args.data_dir / "NaCGEC/data"
        path = directory / ("nacgec.dev.ref.para" if args.split == "dev" else "nacgec.test.input")
        rows = nacgec_rows(path, "dev") if args.split == "dev" else nacgec_test_rows(path, args.reference_file)
        sources = [path] + ([args.reference_file] if args.reference_file is not None else [])
    print(prepare_file(rows, args.output, sources, f"{args.dataset}-{args.split}-explicit-gold-v3"))


if __name__ == "__main__":
    main()
