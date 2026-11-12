"""Build one shared COIG-CQIA/Tulu3 messages pool from local JSON/JSONL/Parquet."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import tempfile
import unicodedata

from scripts.evaluation.common import ROOT, write_json
from scripts.evaluation.runs import new_run_directory
from scripts.train.runtime import atomic_write
from .decontamination import BenchmarkIndex, benchmark_texts

# Subset terms listed in the Tulu mixture card; unknown sources stay unknown.
TULU_LICENSES = {
    "no_robots": "CC-BY-NC-4.0",
    "oasst": "Apache-2.0",
    "personas": "ODC-BY-1.0",
    "numina": "Apache-2.0",
    "wildguard": "Apache-2.0",
    "wildjailbreak": "ODC-BY-1.0",
    "hardcoded": "CC-BY-4.0",
    "hard-coded": "CC-BY-4.0",
    "aya": "Apache-2.0",
    "wildchat": "ODC-BY-1.0",
    "tablegpt": "MIT",
    "table-gpt": "MIT",
    "sciriff": "ODC-BY-1.0",
    "codealpaca": "Apache-2.0",
    "coconot": "ODC-BY-1.0",
}


def files(path):
    paths = (
        [path]
        if path.is_file()
        else sorted(
            p
            for p in path.rglob("*")
            if p.suffix in {".json", ".jsonl", ".parquet"}
            and not any(x.startswith(".") for x in p.relative_to(path).parts)
        )
    )
    if not paths:
        raise FileNotFoundError(f"no instruction data files: {path}")
    return paths


def records(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        from scripts.evaluation.common import iter_jsonl

        yield from iter_jsonl(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError(f"expected a JSON array: {path}")
        yield from data


def convert(row, family, path, index):
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("invalid_metadata")
    messages = row.get("messages")
    if messages is None and family == "coig":
        if any(
            row.get(key) is not None and not isinstance(row[key], str) for key in ("instruction", "input")
        ):
            raise ValueError("invalid_prompt_text")
        prompt = "\n".join(str(row.get(key) or "") for key in ("instruction", "input")).strip()
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": row.get("output")}]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("invalid_messages")
    cleaned, expected = [], "user"
    for i, message in enumerate(messages):
        role, content = message.get("role"), message.get("content")
        if role == "system" and i == 0:
            pass
        elif role == expected:
            expected = "assistant" if role == "user" else "user"
        else:
            raise ValueError("invalid_role_order")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty_content")
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        if re.search(r"<\|(?:im_start|im_end|endoftext)\|>|</?think>", content):
            raise ValueError("embedded_template_tokens")
        if (
            any(unicodedata.category(c) in {"Cc", "Cs"} and c not in "\n\t" for c in content)
            or "\ufffd" in content
        ):
            raise ValueError("corrupt_text")
        cleaned.append({"role": role, "content": content})
    if cleaned[-1]["role"] != "assistant":
        raise ValueError("missing_assistant_target")
    text = "\n".join(m["content"] for m in cleaned)
    if len(text) > 100_000:
        raise ValueError("oversized_conversation")
    source = str(row.get("source") or metadata.get("source_subset") or path.parent.name)
    original_task_type = row.get("task_type", metadata.get("task_type"))
    hint = (source + " " + json.dumps(original_task_type, ensure_ascii=False)).lower()
    task = (
        "code"
        if re.search(r"code|python|编程|程序", hint) or "```" in text
        else "math"
        if re.search(r"math|gsm|algebra|数学|算术", hint)
        else "general"
    )
    hanzi = sum("\u3400" <= c <= "\u9fff" or 0x20000 <= ord(c) <= 0x323AF for c in text)
    letters = sum(c.isalpha() for c in text)
    language = "zh" if hanzi / max(1, letters) >= 0.2 else "other"
    original_id = next(
        (
            value
            for value in (
                row.get("id"),
                row.get("uuid"),
                metadata.get("original_id"),
                metadata.get("source_row_id"),
            )
            if value is not None
        ),
        None,
    )
    sample_id = row.get("sample_id")
    id_origin = "source" if original_id is not None else "sample_id" if sample_id is not None else "file_row"
    if original_id is None:
        original_id = sample_id if sample_id is not None else f"{path.name}:{index}"
    license_info = row.get("license", metadata.get("license"))
    license_origin = "row" if license_info else "unspecified"
    if not license_info:
        license_info = (
            "unspecified in COIG-CQIA dataset card"
            if family == "coig"
            else "subset terms unspecified; see original source"
        )
        if family == "tulu":
            for key, value in TULU_LICENSES.items():
                if key in source.lower():
                    license_info, license_origin = value, "Tulu mixture dataset card"
                    break
    elif not isinstance(license_info, str):
        license_info = json.dumps(license_info, ensure_ascii=False)
    copyright_info = row.get("copyright", metadata.get("copyright"))
    if copyright_info is not None and not isinstance(copyright_info, str):
        copyright_info = json.dumps(copyright_info, ensure_ascii=False)
    human_verified = row.get("human_verified", metadata.get("human_verified"))
    return {
        "messages": cleaned,
        "source_family": family,
        "source": source,
        "original_id": str(original_id),
        "id_origin": id_origin,
        "sample_id": str(sample_id) if sample_id is not None else None,
        "source_file": str(path),
        # Keep varying source schemas as a scalar so Arrow can read mixed-source rows.
        "source_metadata": json.dumps(metadata, ensure_ascii=False),
        "task_type": task,
        "original_task_type": json.dumps(original_task_type, ensure_ascii=False),
        "language": language,
        "language_method": "Han share among letters >= 0.2",
        "license": license_info,
        "license_origin": license_origin,
        "collection_license": "ODC-BY-1.0" if family == "tulu" else None,
        "copyright": copyright_info,
        "human_verified": human_verified if isinstance(human_verified, bool) else None,
    }


def prepare(args):
    sources = [
        (family, path) for family, root in [("coig", args.coig), ("tulu", args.tulu)] for path in files(root)
    ]
    output = new_run_directory(args.output)
    if not args.benchmark_text:
        text_file = output / "benchmark_inputs.txt"
        subprocess.run(
            [
                str(args.evaluation_python),
                "-B",
                "-m",
                "scripts.train.instruction.decontamination",
                "--data",
                str(args.benchmarks),
                "--output",
                str(text_file),
            ],
            check=True,
        )
        args.benchmark_text = [text_file]
    benchmark = BenchmarkIndex(benchmark_texts(args.benchmarks, args.benchmark_text))
    counts, rng = Counter(), random.Random(args.seed)
    with tempfile.TemporaryDirectory(prefix="cc-instruction-pool-", dir=output) as folder:
        connection = sqlite3.connect(str(Path(folder) / "pool.sqlite"))
        connection.execute(
            "CREATE TABLE pool (prompt TEXT PRIMARY KEY, payload TEXT, language TEXT, task TEXT, family TEXT, shuffle REAL)"
        )
        for family, path in sources:
            print(f"[instruction/data] {family} {path}", flush=True)
            for index, row in enumerate(records(path)):
                counts["input"] += 1
                try:
                    item = convert(row, family, path, index)
                except (ValueError, AttributeError) as error:
                    counts["dropped_" + str(error)] += 1
                    continue
                if benchmark.overlaps(item["messages"]):
                    counts["dropped_benchmark_overlap"] += 1
                    continue
                # First-user prompt grouping prevents alternate answers leaking across Train/Dev.
                prompt = unicodedata.normalize(
                    "NFC", next(m["content"] for m in item["messages"] if m["role"] == "user")
                ).strip()
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO pool VALUES (?,?,?,?,?,?)",
                    (
                        prompt,
                        json.dumps(item, ensure_ascii=False),
                        item["language"],
                        item["task_type"],
                        family,
                        rng.random(),
                    ),
                )
                counts["kept_pool" if cursor.rowcount else "dropped_duplicate_prompt"] += 1
            connection.commit()
        available = dict(connection.execute("SELECT language,COUNT(*) FROM pool GROUP BY language"))
        coig_count = connection.execute("SELECT COUNT(*) FROM pool WHERE family='coig'").fetchone()[0]
        capacity = min(
            int(available.get(lang, 0) / share)
            for lang, share in [("zh", args.chinese_share), ("other", 1 - args.chinese_share)]
            if share
        )
        capacity = min(capacity, int(coig_count / (1 - args.max_tulu_share)))
        total = min(capacity, args.samples) if args.samples else capacity
        if total < 2:
            raise ValueError("insufficient clean data for requested mixture; inspect input and ratios")
        targets = {"zh": round(total * args.chinese_share)}
        targets["other"] = total - targets["zh"]
        picked, families = [], Counter()
        for lang, target in targets.items():
            quotas = (
                {"math": round(target * args.math_share), "code": round(target * args.code_share)}
                if args.math_share is not None
                else {}
            )
            if quotas:
                quotas["general"] = target - sum(quotas.values())
            for task, quota in quotas.items() if quotas else [(None, target)]:
                query = "SELECT payload,family FROM pool WHERE language=?"
                values = [lang]
                if task is not None:
                    query += " AND task=?"
                    values.append(task)
                query += " ORDER BY (family='coig') DESC,shuffle"
                kept = 0
                for payload, family in connection.execute(query, values):
                    if kept >= quota:
                        break
                    if family == "tulu" and families["tulu"] >= int(total * args.max_tulu_share):
                        continue
                    picked.append(payload)
                    families[family] += 1
                    kept += 1
        connection.close()
    if len(picked) < 2:
        raise ValueError(
            "mixture/task constraints leave fewer than two conversations; adjust shares after inspecting available data"
        )
    rng.shuffle(picked)
    dev_count = min(len(picked) - 1, max(1, round(len(picked) * args.dev_ratio)))
    distribution = {split: Counter() for split in ("train", "dev")}
    for split, samples in [("dev", picked[:dev_count]), ("train", picked[dev_count:])]:
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for payload in samples:
                item = json.loads(payload)
                for key in ("source_family", "source", "language", "task_type"):
                    distribution[split][f"{key}:{item[key]}"] += 1
                handle.write(payload + "\n")
    combined = distribution["train"] + distribution["dev"]
    report = {
        "counts": dict(counts),
        "rows": len(picked),
        "sample_limit": args.samples or None,
        "sample_shortfall": max(0, args.samples - len(picked)) if args.samples else 0,
        "requested_rows": total,
        "actual_chinese_share": combined["language:zh"] / len(picked),
        "actual_tulu_share": combined["source_family:tulu"] / len(picked),
        "actual_task_shares": {
            name: combined[f"task_type:{name}"] / len(picked) for name in ("general", "math", "code")
        },
        "dev_rows": dev_count,
        "train_rows": len(picked) - dev_count,
        "distribution": distribution,
        "benchmark_patterns": benchmark.count,
        "decontamination": "normalized full input substring, min 24 characters; not paraphrase or exhaustive n-gram detection",
        "requested_chinese_share": args.chinese_share,
        "max_tulu_share": args.max_tulu_share,
        "requested_math_share": args.math_share,
        "requested_code_share": args.code_share,
        "seed": args.seed,
        "sampling_unit": "conversation, not tokens",
        "note": "Tokenizer-specific supervised budgets are measured separately; language/task labels include heuristics.",
    }
    write_json(output / "preparation_report.json", report)
    atomic_write(args.output / "latest.txt", str(output.resolve()) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coig", type=Path, default=ROOT / "data/instruction/raw/COIG-CQIA")
    parser.add_argument("--tulu", type=Path, default=ROOT / "data/instruction/raw/tulu-3-sft-mixture")
    parser.add_argument("--output", type=Path, default=ROOT / "data/instruction/prepared")
    parser.add_argument("--benchmarks", type=Path, default=ROOT / "data/evaluation")
    parser.add_argument("--evaluation-python", type=Path, default=ROOT / ".venv/evaluation/bin/python")
    parser.add_argument(
        "--benchmark-text",
        type=Path,
        action="append",
        help="explicit evaluation input text files, replacing automatic collection",
    )
    parser.add_argument("--chinese-share", type=float, default=0.4)
    parser.add_argument("--max-tulu-share", type=float, default=0.6)
    parser.add_argument("--math-share", type=float)
    parser.add_argument("--code-share", type=float)
    parser.add_argument("--dev-ratio", type=float, default=0.02)
    parser.add_argument(
        "--samples",
        type=int,
        default=150_000,
        help="clean conversation pool limit, including Dev; 0 keeps all available",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        not 0 <= args.chinese_share <= 1
        or not 0 < args.max_tulu_share < 1
        or not 0 < args.dev_ratio < 1
        or args.samples < 0
    ):
        parser.error("invalid mixture, dev ratio or sample count")
    if (args.math_share is None) != (args.code_share is None):
        parser.error("set both --math-share and --code-share, or neither")
    if args.math_share is not None and not (
        args.math_share >= 0 and args.code_share >= 0 and args.math_share + args.code_share <= 1
    ):
        parser.error("task shares must be nonnegative and sum to at most one")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(prepare(args))


if __name__ == "__main__":
    main()
