"""Allocate evaluation versions without replacing existing results."""
from __future__ import annotations

import argparse
from pathlib import Path
import re


def versions(root):
    return [(int(match.group(1)), path) for path in Path(root).iterdir()
            if (match := re.fullmatch(r"v(\d+)", path.name))]


def new_run_directory(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    number = max((n for n, _ in versions(root)), default=-1) + 1
    while True:
        output = root / f"v{number}"
        try:
            output.mkdir()
            return output
        except FileExistsError:
            # Concurrent launches must never receive the same directory.
            number += 1


def latest_run_directory(root):
    candidates = [(n, path) for n, path in versions(root) if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no evaluation versions under {root}; run prepare/all first or set RUN_DIR")
    return max(candidates)[1].resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()
    print(latest_run_directory(args.root) if args.latest else new_run_directory(args.root))


if __name__ == "__main__":
    main()
