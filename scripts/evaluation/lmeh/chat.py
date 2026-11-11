"""Use the saved Swift non-thinking template as harness's chat formatter."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType


def install_chat(backend, model_path, output):
    from swift.model import get_model_processor
    from swift.utils import import_external_file
    from scripts.evaluation.inference import training_contract, training_template

    contract = training_contract(model_path)
    if contract["settings"].get("enable_thinking") is not False:
        raise ValueError("Instruct evaluation requires the saved non-thinking SFT protocol")
    for path in contract["external_plugins"]:
        import_external_file(path)
    _, processor = get_model_processor(
        str(Path(model_path).resolve()),
        model_type=contract["model_type"],
        load_model=False,
        download_model=False,
    )
    template = training_template(processor, contract, backend.max_length)
    template.set_mode("transformers")
    state = {
        "checked_prompts": 0,
        "recorded_prompts": 0,
        "contract": contract,
        "formatter": "saved Swift training template",
        "fewshot_as_multiturn": True,
        "stop_strings": ["<|im_end|>", "<|endoftext|>"],
    }

    def apply(self, chat_history, add_generation_prompt=True):
        history = [dict(message) for message in chat_history]
        continuation = ""
        if not add_generation_prompt:
            if not history or history[-1]["role"] != "assistant":
                raise ValueError("assistant prefill requires a final assistant message")
            continuation = history.pop()["content"]
        encoded = template.encode({"messages": history})
        ids = encoded["input_ids"] + template.tokenizer.encode(continuation, add_special_tokens=False)
        text = template.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if self.tok_encode(text, add_special_tokens=False) != ids:
            raise ValueError("benchmark tokenizer IDs differ from the SFT template prompt IDs")
        state["checked_prompts"] += 1
        if state["recorded_prompts"] < 32:
            with (output / "chat_prompts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "messages": chat_history,
                            "prompt": text,
                            "prompt_token_ids": ids,
                            "add_generation_prompt": add_generation_prompt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            state["recorded_prompts"] += 1
        return text

    backend.apply_chat_template = MethodType(apply, backend)
    backend.instruct_protocol = state
    return state


def chat_stops(requests, enabled):
    if enabled:
        for request in requests:
            text, options = request.args
            until = options.get("until") or []
            if isinstance(until, str):
                until = [until]
            request.arguments = (
                text,
                {**options, "until": list(dict.fromkeys([*until, "<|im_end|>", "<|endoftext|>"]))},
            )
    return requests
