"""Extract nine fixed cases from the latest formal evaluation snapshots."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODELS = {
    "base": ("Qwen3-1.7B-Base-INS", "checkpoint-1641-vllm"),
    "shared": ("Qwen3-1.7B-Base-PGCA-Shared-CPT-INS", "checkpoint-1000-hf"),
}
SELECTION = {
    "gsm8k": ("gsm8k", [0, 19, 40]),
    "cmrc2018": ("clue", [0, 1, 4]),
    "squadv2": ("squadv2", [0, 3, 6]),
}


def selected_rows(path, ids):
    if not path.exists():
        return {}
    rows = {}
    with path.open("rb") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if row["doc_id"] in ids and row["filter"] in ("none", "strict-match"):
                rows[row["doc_id"]] = row
            if len(rows) == len(ids):
                break
    return rows


def build():
    cases = []
    for benchmark, (family, ids) in SELECTION.items():
        snapshots = {}
        for alias, (model, run_name) in MODELS.items():
            folder = ROOT / "outputs/evaluation" / family / model / run_name
            versions = [p for p in folder.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
            latest = max(versions, key=lambda p: int(p.name[1:]))
            name = "clue_cmrc2018" if benchmark == "cmrc2018" else benchmark
            snapshots[alias] = (latest, selected_rows(latest / "samples" / f"{name}.jsonl", ids))
        for doc_id in ids:
            source = snapshots["base"][1][doc_id]
            doc = source["doc"]
            if benchmark == "gsm8k":
                ordinary = doc["question"]
            elif benchmark == "cmrc2018":
                ordinary = f"文章：{doc['context']}\n问题：{doc['question']}"
            else:
                ordinary = f"Title: {doc['title']}\n\nContext:\n{doc['context']}\n\nQuestion: {doc['question']}"
            history = {}
            for alias, (folder, rows) in snapshots.items():
                row = rows.get(doc_id)
                if row is not None and row["doc"] != doc:
                    raise ValueError(f"different benchmark documents: {benchmark}/{doc_id}")
                history[alias] = {
                    "source": str(folder.relative_to(ROOT)),
                    "available": row is not None,
                    "prompt": (row or source)["arguments"][0][0],
                    "generation": (row or source)["arguments"][0][1],
                    "response": row["resps"][0][0] if row else None,
                    "note": None if row else "Shared SQuAD snapshot unavailable; reuse the Base prompt TEXT and tokenize with Shared tokenizer.",
                }
            cases.append({"id": f"{benchmark}/{doc_id}", "benchmark": benchmark,
                          "doc_id": doc_id, "document": doc, "reference": source["target"],
                          "ordinary_messages": [{"role": "user", "content": ordinary}],
                          "historical": history})
    output = HERE / "cases.jsonl"
    output.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    print(f"Prepared {len(cases)} fixed cases: {output}")


if __name__ == "__main__":
    build()
