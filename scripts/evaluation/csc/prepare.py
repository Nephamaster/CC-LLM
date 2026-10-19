import argparse
from pathlib import Path

from ..common import ROOT, iter_jsonl, prepare_file

FILES = ("cscd_test", "car", "cot", "enc", "gam", "mec", "new", "nov")


def rows(paths):
    for path in paths:
        found = False
        for item in iter_jsonl(path):
            found = True
            if not all(isinstance(item.get(key), str) for key in ("id", "dataset", "subset", "src", "tgt")):
                raise ValueError(f"invalid CSC test row: {path}")
            if item["dataset"] not in {"cscd_ns", "lemon_v2"}:
                raise ValueError(f"unexpected default CSC dataset: {item['dataset']}")
            yield {**item, "task": "csc", "messages": [{"role": "user", "content": item["src"]}]}
        if not found:
            raise ValueError(f"empty CSC test file: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/downstream/csc/test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.data_dir / f"{name}.jsonl" for name in FILES]
    print(prepare_file(rows(paths), args.output, paths, "csc-source-only-cscd-lemon-v2"))


if __name__ == "__main__":
    main()
