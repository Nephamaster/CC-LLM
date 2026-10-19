"""Source-only correction SFT: native assistant loss and all-linear LoRA."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

from swift.callbacks import TrainerCallback, callbacks_map
from swift.template import register_template
from swift.template.register import TEMPLATE_MAPPING


BASE_META = TEMPLATE_MAPPING["qwen3_nothinking"]


def enable_pgca(model):
    while hasattr(model, "module"):
        model = model.module
    if getattr(model.config, "model_type", None) == "qwen3_pgca":
        # ALIGN checkpoints deliberately store pgca_enabled=false; SFT uses the full branch.
        model.config.pgca_enabled = True
        return True
    return False


class CorrectionTemplate(BASE_META.template_cls):
    def compute_sft_loss(self, model, inputs, num_items_in_batch=None, trainer=None):
        pgca = enable_pgca(model)
        inputs = dict(inputs, use_cache=False)
        if pgca:
            inputs["return_aux_logits"] = False
        return super().compute_sft_loss(model, inputs, num_items_in_batch=num_items_in_batch, trainer=trainer)

    def generate(self, model, *args, **kwargs):
        # The evaluator restores this template from args.json, also for base + adapter inference.
        if enable_pgca(model):
            kwargs["return_aux_logits"] = False
        return super().generate(model, *args, **kwargs)


class CorrectionCallback(TrainerCallback):
    def on_init_end(self, args, state, control, model=None, **kwargs):
        enable_pgca(model)
        # all-linear also wraps auxiliary-head linears. Freeze them before optimizer/DDP setup.
        for name, parameter in model.named_parameters():
            if "auxiliary_heads" in name.split("."):
                parameter.requires_grad_(False)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        model = self.trainer.accelerator.unwrap_model(model)
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        if not trainable or any("lora_" not in name or "auxiliary_heads" in name.split(".") for name in trainable):
            raise ValueError("correction SFT requires LoRA-only trainable parameters with auxiliary heads frozen")
        pgca = getattr(model.config, "model_type", None) == "qwen3_pgca"
        if pgca:
            if not model.config.pgca_enabled or not any(".pgca_attn." in name for name in trainable):
                raise ValueError("PGCA must be enabled and included in all-linear LoRA")
            if getattr(model.config, "pgca_context_enabled", False) and not any(".context_encoder." in name for name in trainable):
                raise ValueError("PGCA context encoder was not included in LoRA")
        if state.is_world_process_zero:
            report = {"loss": "assistant_only", "auxiliary_loss_enabled": False, "pgca_enabled": pgca,
                      "target_modules": "all-linear", "trainable_parameters": sum(p.numel() for p in trainable.values()),
                      "trainable_names": sorted(trainable)}
            (Path(args.output_dir) / "finetune_report.json").write_text(json.dumps(report, indent=2) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoints = list(Path(args.output_dir).glob("checkpoint-*"))
        last = max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[1])) if checkpoints else None
        selected = state.best_model_checkpoint or last
        if selected is None:
            raise RuntimeError("SFT ended without a saved LoRA checkpoint")
        value = str(Path(selected).resolve()) + "\n"
        (Path(args.output_dir) / "next_stage_model.txt").write_text(value)
        output_root = os.environ.get("CC_LLM_SFT_OUTPUT_ROOT")
        if output_root:
            pointer = Path(output_root) / "next_stage_model.txt"
            temporary = pointer.with_name(f".next_stage_model.{os.getpid()}.tmp")
            temporary.write_text(value)
            temporary.replace(pointer)


meta = deepcopy(BASE_META)
meta.template_type = "cc_llm_correction"
meta.template_cls = CorrectionTemplate
register_template(meta, exist_ok=True)
callbacks_map["cc_llm_correction"] = CorrectionCallback
