"""Read official local FCGEC/NaCGEC files without changing their splits."""
import argparse
import itertools
import json
import re
from pathlib import Path

from ..common import ROOT, prepare_file

PROMPT = "纠正下列句子的语法错误，保留原意；如果没有错误则原样返回。只输出纠正后的句子：\n"


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
               "messages": [{"role": "user", "content": PROMPT + source}]}


def nacgec_rows(path, split):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 3 or not fields[1]:
                raise ValueError(f"invalid NaCGEC parallel row: {path}")
            uid, source, *references = fields
            yield {"id": f"nacgec/{split}/{uid}", "original_id": uid, "dataset": "nacgec", "task": "cgec",
                   "split": split, "src": source, "references": references,
                   "messages": [{"role": "user", "content": PROMPT + source}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fcgec", "nacgec"), required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/downstream/cgec")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "fcgec":
        path = args.data_dir / "FCGEC" / f"FCGEC_{args.split}.json"
        rows = fcgec_rows(path, args.split)
    else:
        suffix = "dev" if args.split == "valid" else "test"
        path = args.data_dir / "NaCGEC/data" / f"nacgec.{suffix}.ref.para"
        rows = nacgec_rows(path, args.split)
    print(prepare_file(rows, args.output, [path], f"{args.dataset}-{args.split}-multireference-v1"))


if __name__ == "__main__":
    main()
