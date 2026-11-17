"""Select complete conversations by actual assistant labels, preserving every EOS."""

import argparse
import json
from pathlib import Path

from scripts.evaluation.common import iter_jsonl, write_json


def prepare(model, source, output, budget, max_length):
    from swift.model import get_model_processor
    from swift.template import get_template
    from scripts.train.downstream import plugin  # noqa: F401 -- registers the shared SFT templates
    from scripts.train.downstream.dev_check import encoded_case

    _, processor = get_model_processor(str(model), model_type="qwen3", load_model=False, download_model=False)
    template = get_template(
        processor,
        template_type="cc_llm_instruction",
        max_length=max_length,
        enable_thinking=False,
        add_non_thinking_prefix=False,
        loss_scale="default",
    )
    template.set_mode("train")
    total, input_tokens, count, too_long, last_count = 0, 0, 0, 0, 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(source):
            try:
                encoded = template.encode({"messages": row["messages"]})
            except Exception as error:
                from swift.template import MaxLengthError

                if not isinstance(error, MaxLengthError):
                    raise
                too_long += 1
                continue
            _, _, _, _, supervised, aligned = encoded_case(template, row)
            if not supervised or not aligned:
                raise ValueError(
                    "instruction template must supervise every assistant EOS and match inference IDs"
                )
            labels = encoded["labels"][1:]
            last_count = sum(value != -100 for value in labels)
            if not last_count:
                raise ValueError("instruction sample has no supervised target")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += last_count
            input_tokens += len(encoded["input_ids"])
            count += 1
            if budget and total >= budget:
                break
    if not count:
        raise ValueError("no complete conversation fits MAX_LENGTH")
    report = {
        "model": str(model),
        "input": str(source),
        "rows": count,
        "input_tokens": input_tokens,
        "supervised_tokens": total,
        "token_scope": "per epoch; input includes Prompt, answers and template tokens, excludes padding",
        "requested_tokens": budget or None,
        "target_reached": total >= budget if budget else None,
        "overage_tokens": max(0, total - budget) if budget else 0,
        "last_conversation_tokens": last_count,
        "dropped_too_long": too_long,
        "max_length": max_length,
        "budget_policy": "per-epoch view; keep final complete conversation, no partial answer or EOS masking",
        "comparability": "same requested budget is not proof of equal actual targets; compare all three reports",
    }
    write_json(output.with_suffix(".report.json"), report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=0)
    p.add_argument("--max-length", type=int, default=4096)
    a = p.parse_args()
    if a.tokens < 0:
        p.error("--tokens must be nonnegative")
    print(json.dumps(prepare(a.model, a.input, a.output, a.tokens, a.max_length), ensure_ascii=False))


if __name__ == "__main__":
    main()
