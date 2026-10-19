"""Use the user's actual ms-swift encoder; never estimate assistant loss from text length."""

from __future__ import annotations

import importlib.metadata

from .common import resolve


def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class SwiftEncoder:
    def __init__(self, config):
        import swift

        from swift.model import get_model_processor
        from swift.template import get_template
        _, processor = get_model_processor(
            str(resolve(config["tokenizer_path"])), model_type="qwen3", load_model=False,
            download_model=False,
        )
        # No truncation here: candidates exceeding the limit are rejected by selection.
        self.template = get_template(
            processor, template_type="qwen3", enable_thinking=False, max_length=10**9,
            truncation_strategy="raise", use_chat_template=True, loss_scale="default",
        )
        self.template.set_mode("train")
        self.info = {
            "backend": "ms-swift", "template": "qwen3",
            "package_version": getattr(swift, "__version__", None) or installed_version("ms-swift"),
            "enable_thinking": False, "loss_scale": "default", "template_backend": "swift",
            "add_non_thinking_prefix": True, "use_chat_template": True,
            "system": None,
            "max_length": config["max_length"], "truncation_strategy": "raise",
            "dependencies": {name: installed_version(name) for name in
                             ("transformers", "tokenizers", "torch")},
        }

    def __call__(self, messages):
        value = self.template.encode({"messages": messages})
        ids, labels = value["input_ids"], value["labels"]
        if len(ids) != len(labels) or labels[0] != -100:
            raise RuntimeError("unexpected causal-LM label contract")
        loss = sum(label != -100 for label in labels[1:])
        if not 0 < loss < len(ids):
            raise RuntimeError("missing answer loss or unmasked user prompt")
        return {"sequence_tokens": len(ids), "assistant_loss_tokens": loss}
