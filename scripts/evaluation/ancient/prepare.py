import argparse
import json
import unicodedata
from pathlib import Path

from ..common import ROOT, prepare_file

C3_PROMPTS = {
    "classification": "不需要解释，直接从“诗、史、儒、道、佛、农、法、艺、医、兵”中选择一个类别输出，判断下列文言文的类别：",
    "retrieval": "不需要解释，直接给出下列文言文出处书名：",
    "ner": "找出下列文言文中的命名实体。若无实体，输出“无”；若有实体，直接输出实体，多个实体之间用“、”分隔：",
    "punctuation": "不需要解释，直接输出下列文言文添加标点符号后的结果：",
    "translation": "不需要解释，直接输出下列文言文的白话文翻译：",
}
FUXI_TASKS = {
    "ac_rc": ("ac_rc", "acc"), "idiom_rc": ("idiom_rc", "acc"), "tcmsd_rc": ("tcm_sd", "acc"),
    "lc_qa": ("lc_qa", "judge"), "as_qa": ("as_qa", "judge"), "book_author": ("ask_author", "judge"),
    "book_dynasty": ("ask_dynasty", "judge"), "book_collection": ("ask_collection", "judge"),
    "poem_gen": ("poem_creation", "judge"), "poem_line_st": ("poem_source", "judge"),
    "famous_quote_st": ("quote_source", "judge"), "idiom_st": ("idiom_source", "judge"),
    "poem_nmt_inv": ("poem_nmt_inv", "judge"), "poem_nmt": ("poem_nmt", "bleu"),
    "ac_nmt": ("ac_nmt", "bleu"), "idiom_exp": ("idiom_exp", "bleu"),
    "poem_appre": ("poem_appre", "bleu"), "tcm_qa": ("tcm_qa", "bleu"),
    "prescription_exp": ("prescription_detail", "bleu"), "couplet_gen": ("couplet_gen", "couplet"),
    "ci_gen": ("ci_gen", "ci"),
}


def unpunctuated(text):
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def c3_rows(path):
    for row in json.loads(path.read_text(encoding="utf-8-sig")):
        text = row["classical_Chinese"]
        golds = dict(classification=row["category"], retrieval=row["source"], ner=row["entity"],
                     punctuation=text, translation=row["modern_Chinese"])
        for task, prompt in C3_PROMPTS.items():
            source = unpunctuated(text) if task == "punctuation" else text
            yield {"id": f"c3/{row['id']}/{task}", "dataset": "c3", "task": task,
                   "domain": row["category"], "src": source, "gold": golds[task],
                   "messages": [{"role": "user", "content": prompt + source}]}


def fuxi_rows(directory, fewshot=0):
    if not 0 <= fewshot <= 5:
        raise ValueError("Fuxi fewshot must be between zero and five")
    examples = json.loads((directory / "dev_fewshot.json").read_text()) if fewshot else {}
    for filename, (task, metric) in FUXI_TASKS.items():
        data = json.loads((directory / "test_data" / f"{filename}.json").read_text(encoding="utf-8-sig"))
        if not isinstance(data, list) or not data:
            raise ValueError(f"empty or invalid Fuxi task: {filename}")
        demonstrations = examples.get(task, [])[:fewshot]
        if len(demonstrations) != fewshot:
            raise ValueError(f"not enough official few-shot examples: {task}")
        for row in data:
            prefix = ""
            if demonstrations:
                # Match the upstream textual few-shot construction, never add current test answers.
                prefix = "请参考以下示例，完成题目。" + "\n\n".join(
                    f"{e['instruction']}\n{e['input']}\n{e['output']}" for e in demonstrations) + "\n\n"
            prompt = prefix + row["instruction"] + "\n" + row["input"]
            yield {"id": f"fuxi/{row['sample_id']}", "dataset": "fuxi", "task": task, "metric": metric,
                   "src": row["input"], "instruction": row["instruction"], "gold": row["output"],
                   "cipai": row.get("cipai"), "fewshot": fewshot,
                   "demonstration_overlap": any(e['input'] == row['input'] for e in demonstrations),
                   "messages": [{"role": "user", "content": prompt}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("c3", "fuxi"), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/downstream/ancient/benchmark")
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset == "c3":
        if args.fewshot:
            parser.error("C3 uses the paper's zero-shot protocol")
        path = args.data_dir / "c3bench/C3_bench.json"
        rows, paths = c3_rows(path), [path]
    else:
        directory = args.data_dir / "FuxiBench"
        rows = fuxi_rows(directory, args.fewshot)
        paths = [directory / "test_data" / f"{name}.json" for name in FUXI_TASKS]
        if args.fewshot:
            paths.append(directory / "dev_fewshot.json")
    print(prepare_file(rows, args.output, paths, f"{args.dataset}-paper-protocol-v1-shot{args.fewshot}"))


if __name__ == "__main__":
    main()
