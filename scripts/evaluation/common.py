from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = Path(__file__).parent / "vendor"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected JSON object")
                yield value


def write_json(path, value):
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_journal(path):
    """Recover only an interrupted final write in our own append-only journals."""
    with Path(path).open("rb+") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not line.endswith(b"\n") and not handle.read(1):
                    handle.truncate(offset)
                    break
                raise ValueError(f"malformed journal record before EOF: {path}")
            if not line.endswith(b"\n"):
                handle.write(b"\n")
            yield row


def prediction_health(rows):
    from collections import Counter

    return {"samples": len(rows), "empty_outputs": sum(not r["predict_clean"] for r in rows),
            "finish_reasons": dict(Counter(r.get("finish_reason") or "unknown" for r in rows))}


def scoring_environment():
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for name in ("numpy", "pypinyin", "opencc-python-reimplemented", "python-Levenshtein", "thulac", "sacrebleu"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).parent
    hashes = {}
    for directory, children, files in os.walk(root):
        children[:] = sorted(name for name in children if name not in {"env", ".venv", "__pycache__", ".git"})
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_file() and path.suffix in {".py", ".txt", ".json"}:
                hashes[str(path.relative_to(root))] = sha256(path)
    return {"packages": versions, "implementation_and_assets_sha256": signature(hashes)}


def is_cjk_char(value):
    import regex

    return bool(regex.fullmatch(r"\p{Unified_Ideograph}", value))


def prepare_file(rows, output, sources, protocol):
    output = Path(output)
    if output.exists():
        manifest = json.loads(output.with_suffix(".manifest.json").read_text())
        current = {str(Path(p)): sha256(p) for p in sources}
        if (manifest["sources"] != current or manifest["protocol"] != protocol
                or manifest["sha256"] != sha256(output)):
            raise ValueError("prepared input changed; choose a new output directory")
        return manifest
    ensure_dir(output.parent)
    seen = set()
    counts = {}
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            if row["id"] in seen:
                raise ValueError(f"duplicate evaluation id: {row['id']}")
            seen.add(row["id"])
            key = f"{row['dataset']}/{row['task']}"
            counts[key] = counts.get(key, 0) + 1
            write_line(handle, row)
    if not seen:
        raise ValueError("empty evaluation dataset")
    report = {"count": len(seen), "counts": counts, "sources": {str(Path(p)): sha256(p) for p in sources},
              "protocol": protocol, "sha256": sha256(output)}
    write_json(output.with_suffix(".manifest.json"), report)
    return report


def aligned_predictions(input_path, prediction_path):
    originals = list(iter_jsonl(input_path))
    if not originals:
        raise ValueError("cannot score an empty evaluation input")
    predictions = list(iter_jsonl(prediction_path))
    if len(originals) != len(predictions):
        raise ValueError(f"incomplete predictions: {len(predictions)}/{len(originals)}")
    seen = set()
    for expected, actual in zip(originals, predictions):
        if actual.get("id") != expected["id"] or actual["id"] in seen:
            raise ValueError("prediction order/IDs do not match the input")
        seen.add(actual["id"])
        if actual.get("input_sha256") != signature(expected):
            raise ValueError(f"prediction belongs to a different input: {actual['id']}")
        if actual.get("error"):
            raise ValueError(f"inference error in {actual['id']}; retry inference before scoring")
        if not isinstance(actual.get("predict_clean"), str):
            raise ValueError("prediction must be text, including empty text for a genuine empty generation")
    return originals, predictions
