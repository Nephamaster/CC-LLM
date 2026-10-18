"""Configuration, provenance, and comparison-only text normalization."""

from __future__ import annotations

import glob
import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("c2m", "m2c", "punctuate", "dictionary", "provenance")
PUNCTUATION = frozenset('，。！？；：、,.!?;:「」『』“”‘’（）()《》〈〉【】〔〕[]—…·‧')


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            yield number, row


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_config(path: str | Path) -> dict:
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if set(config["sources"]) != {"corpus", "translate", "punctuate", "history_train", "history_validation", "history_test"}:
        raise ValueError("all Gujilab sources and all three HistoryTrans splits must be configured")
    if set(config["benchmarks"]) != {"c3", "fuxi", "fuxi_dev"}:
        raise ValueError("C3, Fuxi test, and Fuxi few-shot exclusions must all be configured")
    if set(config["quotas"]) != set(TASKS):
        raise ValueError(f"quotas must specify exactly {TASKS}")
    if any(not isinstance(n, int) or n < 0 for n in config["quotas"].values()):
        raise ValueError("quotas must be nonnegative integers")
    if not 0 < config["dev_fraction"] < 0.5:
        raise ValueError("dev_fraction must be between zero and 0.5")
    if config["max_length"] < 32 or not 0 < config["max_book_fraction"] <= 1:
        raise ValueError("invalid sequence length or book fraction")
    if not 0 <= config["history_target"] <= config["quotas"]["c2m"]:
        raise ValueError("history_target must fit the c2m quota")
    for bounds in config["lengths"].values():
        if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
            raise ValueError("invalid character-length bounds")
    if not 0.5 <= config["near_threshold"] <= 1:
        raise ValueError("near_threshold must be between 0.5 and 1")
    if config["template"] != "qwen3" or config["enable_thinking"] is not False:
        raise ValueError("v1 supports qwen3 with enable_thinking=false only")
    return config


def input_inventory(config: dict) -> dict:
    files, missing = {}, []
    for name, path in config["sources"].items():
        target = resolve(path)
        if not target.is_file():
            missing.append(str(target))
        else:
            files[name] = [target]
    for name, pattern in config["benchmarks"].items():
        paths = sorted(p for p in map(Path, glob.glob(str(resolve(pattern)))) if p.is_file())
        if not paths:
            missing.append(pattern)
        else:
            files[name] = paths
    tokenizer = resolve(config["tokenizer_path"])
    if not (tokenizer / "tokenizer.json").is_file():
        missing.append(str(tokenizer / "tokenizer.json"))
    for path in sorted(tokenizer.glob("*")):
        if path.is_file() and path.name in {
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "vocab.json", "merges.txt", "chat_template.jinja", "config.json",
        }:
            files[f"tokenizer/{path.name}"] = [path]
    if config.get("reviews"):
        path = resolve(config["reviews"])
        if path.is_file():
            files["reviews"] = [path]
        else:
            missing.append(str(path))
    return {"missing": missing, "files": {
        key: [{"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)} for path in paths]
        for key, paths in files.items()
    }}


def clean(text) -> str:
    if not isinstance(text, str):
        raise ValueError("text field must be a string")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not text or any(c in text for c in ("□", "\ufffd", "<|", "<think>", "</think>")):
        raise ValueError("empty, damaged, or template-control text")
    if any(unicodedata.category(c) in {"Cc", "Cs", "Cf"} and c not in "\n\t" for c in text):
        raise ValueError("invalid control character")
    return text


def unpunctuated(text: str) -> str:
    return "".join(c for c in text if c not in PUNCTUATION and not c.isspace())


class Normalizer:
    def __init__(self):
        from opencc import OpenCC

        self.simplify = OpenCC("t2s")
        self.traditionalize = OpenCC("s2t")

    def __call__(self, text: str) -> str:
        value = self.simplify.convert(unicodedata.normalize("NFC", text))
        return "".join(c for c in value if not c.isspace() and not unicodedata.category(c).startswith("P"))

    def script(self, text: str) -> str:
        has_traditional = self.simplify.convert(text) != text
        has_simplified = self.traditionalize.convert(text) != text
        if has_traditional and has_simplified:
            return "mixed_or_ambiguous"
        if has_traditional:
            return "traditional"
        return "simplified" if has_simplified else "shared_or_unknown"


class Groups:
    def __init__(self):
        self.parent = {}

    def root(self, value: str) -> str:
        parent = self.parent
        parent.setdefault(value, value)
        trail = []
        while parent[value] != value:
            trail.append(value)
            value = parent[value]
        for node in trail:
            parent[node] = value
        return value

    def union(self, left: str, right: str):
        left, right = self.root(left), self.root(right)
        self.parent[max(left, right)] = min(left, right)


def split_for(group: str, config: dict) -> str:
    fraction = int(digest(f"{config['seed']}:{group}")[:13], 16) / 16**13
    return "dev" if fraction < config["dev_fraction"] else "train"
