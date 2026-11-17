"""Downstream SFT templates, PGCA LoRA checks, and checkpoint publication."""
from __future__ import annotations

import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

from swift.callbacks import TrainerCallback, callbacks_map
from swift.template import register_template
from swift.template.register import TEMPLATE_MAPPING

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.train.runtime import TokenCountMetric, publish_checkpoint, report_validation


CORRECTION_META = TEMPLATE_MAPPING["qwen3_nothinking"]
ANCIENT_META = TEMPLATE_MAPPING["qwen3"]


def enable_pgca(model):
    while hasattr(model, "module"):
        model = model.module
    if getattr(model.config, "model_type", None) == "qwen3_pgca":
        # ALIGN checkpoints deliberately store pgca_enabled=false; SFT uses the full branch.
        model.config.pgca_enabled = True
        return True
    return False


class PGCASFTTemplateMixin:
    def compute_sft_loss(self, model, inputs, num_items_in_batch=None, trainer=None):
        pgca = enable_pgca(model)
        inputs = dict(inputs, use_cache=False)
        if pgca:
            inputs["return_aux_logits"] = False
        if trainer is not None and not model.training and inputs.get("labels") is not None:
            counts = inputs["labels"][:, 1:].ne(-100).sum(dim=-1)
            counts = trainer.accelerator.gather_for_metrics(counts)
            metric = trainer.custom_metrics["eval"].setdefault("supervised_tokens", TokenCountMetric())
            metric.update(counts.sum().item())
        if (trainer is not None and model.training and inputs.get("labels") is not None
                and os.environ.get("CC_LLM_SFT_TASK") == "instruction"):
            count = trainer.accelerator.reduce(inputs["labels"][:, 1:].ne(-100).sum(), reduction="sum")
            trainer.instruction_supervised_tokens = getattr(trainer, "instruction_supervised_tokens", 0) + count.item()
        return super().compute_sft_loss(model, inputs, num_items_in_batch=num_items_in_batch, trainer=trainer)

    def generate(self, model, *args, **kwargs):
        # The evaluator restores this template from args.json, also for base + adapter inference.
        if enable_pgca(model):
            kwargs["return_aux_logits"] = False
        return super().generate(model, *args, **kwargs)


class CorrectionTemplate(PGCASFTTemplateMixin, CORRECTION_META.template_cls):
    pass


class AncientTemplate(PGCASFTTemplateMixin, ANCIENT_META.template_cls):
    pass


class CorrectionCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        channels = tuple(filter(None, os.environ.get("CC_LLM_MACRO_CHANNELS", "").split(",")))
        if logs is None:
            return
        keys = [f"eval_loss_{channel}" for channel in channels]
        present = [key for key in keys if key in logs]
        if present:
            if len(present) != len(keys):
                raise RuntimeError(f"incomplete task validation losses: expected {keys}, found {present}")
            values = [float(logs[key]) for key in keys]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"non-finite task validation losses: {dict(zip(keys, values))}")
            macro = sum(values) / len(values)
            logs["eval_macro_task_loss"] = macro
            # Transformers appends log_history immediately before invoking callbacks.
            if state.log_history:
                state.log_history[-1]["eval_macro_task_loss"] = macro
        if state.is_world_process_zero and "eval_loss" in logs:
            report_validation(args, logs)
        if os.environ.get("CC_LLM_SFT_TASK") == "instruction" and ("loss" in logs or "train_loss" in logs):
            value = getattr(self.trainer, "instruction_supervised_tokens", 0)
            logs["train_supervised_tokens"] = value
            if state.log_history:
                state.log_history[-1]["train_supervised_tokens"] = value

    def on_init_end(self, args, state, control, model=None, **kwargs):
        enable_pgca(model)
        # all-linear also wraps auxiliary-head linears. Freeze them before optimizer/DDP setup.
        for name, parameter in model.named_parameters():
            if "auxiliary_heads" in name.split("."):
                parameter.requires_grad_(False)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        model = self.trainer.accelerator.unwrap_model(model)
        if os.environ.get("CC_LLM_SFT_TASK") == "instruction":
            self.trainer.instruction_supervised_tokens = max(
                (row.get("train_supervised_tokens", 0) for row in state.log_history), default=0)
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        if not trainable or any("lora_" not in name or "auxiliary_heads" in name.split(".") for name in trainable):
            raise ValueError("downstream SFT requires LoRA-only trainable parameters with auxiliary heads frozen")
        pgca = getattr(model.config, "model_type", None) == "qwen3_pgca"
        if pgca:
            if not model.config.pgca_enabled or not any(".pgca_attn." in name for name in trainable):
                raise ValueError("PGCA must be enabled and included in all-linear LoRA")
            if getattr(model.config, "pgca_context_enabled", False) and not any(".context_encoder." in name for name in trainable):
                raise ValueError("PGCA context encoder was not included in LoRA")
        if state.is_world_process_zero:
            report = {"loss": "assistant_only", "auxiliary_loss_enabled": False, "pgca_enabled": pgca,
                      "task": os.environ.get("CC_LLM_SFT_TASK"), "dev_file": os.environ.get("CC_LLM_SFT_DEV"),
                      "target_modules": "all-linear", "trainable_parameters": sum(p.numel() for p in trainable.values()),
                      "trainable_names": sorted(trainable),
                      "macro_channels": list(filter(None, os.environ.get("CC_LLM_MACRO_CHANNELS", "").split(","))),
                      "metric_for_best_model": args.metric_for_best_model}
            (Path(args.output_dir) / "sft_report.json").write_text(json.dumps(report, indent=2) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        publish_checkpoint(args, state, os.environ.get("CC_LLM_SFT_OUTPUT_ROOT"))

    def on_save(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        from scripts.train.downstream.dev_check import saved_checkpoint_check

        model = self.trainer.accelerator.unwrap_model(model)
        output = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "dev_check"
        saved_checkpoint_check(model, self.trainer.template, os.environ.get("CC_LLM_SFT_DEV"),
                               os.environ.get("CC_LLM_SFT_TASK"), output)


for base_meta, template_type, template_cls in (
        (CORRECTION_META, "cc_llm_correction", CorrectionTemplate),
        (ANCIENT_META, "cc_llm_ancient", AncientTemplate),
        (CORRECTION_META, "cc_llm_instruction", CorrectionTemplate)):
    meta = deepcopy(base_meta)
    meta.template_type = template_type
    meta.template_cls = template_cls
    register_template(meta, exist_ok=True)
callbacks_map["cc_llm_correction"] = CorrectionCallback
