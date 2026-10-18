"""Use the user's actual ms-swift encoder; never estimate assistant loss from text length."""

from __future__ import annotations

import importlib.metadata
import inspect
import subprocess
from pathlib import Path

from .common import file_hash, resolve


class SwiftEncoder:
    def __init__(self, config):
        import swift
        from swift.model import get_model_processor
        from swift.template import get_template

        root = Path(swift.__file__).resolve().parent.parent
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if commit != config["swift_commit"]:
            raise RuntimeError(f"ms-swift commit mismatch: {commit}; expected {config['swift_commit']}")
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "swift"], text=True)
        if dirty.strip():
            raise RuntimeError("ms-swift implementation has uncommitted changes; pin and audit it first")
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
            "backend": "ms-swift", "commit": commit, "template": "qwen3",
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
