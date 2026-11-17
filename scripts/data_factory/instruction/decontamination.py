"""Read evaluation inputs and detect normalized benchmark prompts inside SFT text."""

from __future__ import annotations

import re
import tempfile
import unicodedata
from pathlib import Path


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def local_inputs(data):
    from scripts.evaluation.csc.prepare import FILES, rows
    from scripts.evaluation.cgec.prepare import fcgec_rows, nacgec_rows
    from scripts.evaluation.ancient.prepare import c3_rows, fuxi_rows

    for row in rows([data / "csc" / f"{name}.jsonl" for name in FILES]):
        yield row["src"]
    for split in ("valid", "test"):
        for row in fcgec_rows(data / "cgec/FCGEC" / f"FCGEC_{split}.json", split):
            yield row["src"]
    for row in nacgec_rows(data / "cgec/NaCGEC/data/nacgec.dev.ref.para", "dev"):
        yield row["src"]
    for row in nacgec_rows(data / "cgec/NaCGEC/data/nacgec.test.input", "test", labeled=False):
        yield row["src"]
    for row in c3_rows(data / "ancient/c3bench/C3_bench.json"):
        if row["task"] == "translation":
            yield row["src"]
    for row in fuxi_rows(data / "ancient/FuxiBench"):
        yield row["src"]


def benchmark_texts(data, explicit):
    if explicit:
        for path in explicit:
            with path.open(encoding="utf-8-sig") as handle:
                yield from handle
        return
    yield from local_inputs(data)
    from scripts.evaluation.lmeh.protocol import BENCHMARKS, build_bundle
    from scripts.evaluation.lmeh.run import load_tasks

    with tempfile.TemporaryDirectory(prefix="cc-instruction-benchmarks-") as folder:
        for name in BENCHMARKS:
            print(f"[instruction/data] benchmark {name}", flush=True)
            _, loaded, _ = load_tasks(build_bundle(Path(folder) / name, name), name)
            for task in loaded["tasks"].values():
                for row in task.eval_docs:
                    # Keep raw questions as well as the rendered task input; do not index short labels/answers.
                    for key, value in row.items():
                        if key.lower() in {
                            "question",
                            "sentence",
                            "text",
                            "context",
                            "query",
                            "passage",
                        } and isinstance(value, str):
                            yield value
                    rendered = task.doc_to_text(row)
                    if isinstance(rendered, str):
                        yield rendered


class BenchmarkIndex:
    def __init__(self, texts, minimum=24):
        import ahocorasick

        self.automaton = ahocorasick.Automaton()
        for text in texts:
            text = normalized(text)
            if len(text) >= minimum:
                self.automaton.add_word(text, True)
        self.count = len(self.automaton)
        if not self.count:
            raise ValueError("benchmark index is empty; provide real evaluation inputs")
        self.automaton.make_automaton()

    def overlaps(self, messages):
        return any(
            next(self.automaton.iter(normalized(message["content"])), None) is not None
            for message in messages
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export benchmark input text with the evaluation environment"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for text in benchmark_texts(args.data, None):
            value = normalized(text)
            if len(value) >= 24:
                handle.write(value + "\n")
                count += 1
    print(f"[instruction/data] exported {count} benchmark inputs", flush=True)


if __name__ == "__main__":
    main()
