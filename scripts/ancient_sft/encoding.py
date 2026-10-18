"""Use the user's actual ms-swift encoder; never estimate assistant loss from text length."""

from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path

from .common import file_hash, resolve
from .provenance import verify_swift


class SwiftEncoder:
    def __init__(self, config):
        import swift

        root = Path(swift.__file__).resolve().parent.parent
        provenance = verify_swift(root, config["swift_commit"])
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
            "backend": "ms-swift", "commit": provenance["git_commit"], "template": "qwen3",
            "provenance": provenance, "package_version": getattr(swift, "__version__", None),
            "enable_thinking": False, "loss_scale": "default", "template_backend": "swift",
            "add_non_thinking_prefix": True, "use_chat_template": True,
            "system": None,
            "max_length": config["max_length"], "truncation_strategy": "raise",
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ("transformers", "tokenizers", "torch")},
            "template_source_sha256": file_hash(Path(inspect.getfile(type(self.template)))),
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
