"""Portable ms-swift 4.7 plugin; no imports from the CC-LLM checkout."""

from __future__ import annotations

import os
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import Trainer as HFTrainer

from swift.callbacks import TrainerCallback, callbacks_map
from swift.loss import BaseLoss, loss_map
from swift.optimizers import OptimizerCallback, optimizers_map
from swift.template import Template, register_template
from swift.template.register import TEMPLATE_MAPPING


GLYPH_NAMES = ("stroke_count", "radical_stroke", "structure")
BRANCH_GRAD_GROUPS = (
    "feature_embedding", "context_encoder", "pgca_q_proj", "pgca_k_proj", "pgca_v_proj", "pgca_o_proj",
    "pgca_norms", "auxiliary_phonetic", "auxiliary_glyph",
)


def stage():
    value = os.environ.get("CC_LLM_STAGE", "phase1")
    if value not in {"phase1", "phase2a", "phase2b"}:
        raise ValueError(f"invalid CC_LLM_STAGE: {value}")
    return value


def is_branch_parameter(name):
    return ".pgca_attn." in name or name.startswith(("model.feature_memory_builder.", "auxiliary_heads."))


def branch_gradient_group(name):
    if name.startswith("model.feature_memory_builder.feature_embedding."):
        return "feature_embedding"
    if name.startswith("model.feature_memory_builder.context_encoder."):
        return "context_encoder"
    if ".pgca_attn." in name:
        part = name.split(".pgca_attn.", 1)[1].split(".", 1)[0]
        return f"pgca_{part}" if part in ("q_proj", "k_proj", "v_proj", "o_proj") else "pgca_norms"
    if name.startswith("auxiliary_heads.phonetic."):
        return "auxiliary_phonetic"
    if name.startswith("auxiliary_heads."):
        return "auxiliary_glyph"
    return None


class RatioMetric:
    def __init__(self, *, synchronize=True, total=False):
        self.values = None
        self.device = None
        self.synchronize = synchronize
        self.total = total

    def update(self, numerator, denominator):
        values = torch.stack((numerator.detach().double(), torch.as_tensor(denominator, device=numerator.device).double()))
        self.device = values.device
        self.values = values if self.values is None else self.values + values

    def compute(self):
        values = self.values.clone() if self.values is not None else torch.zeros(2, device=self.device, dtype=torch.float64)
        if self.synchronize and dist.is_initialized():
            dist.all_reduce(values)
        if self.total:
            return values[0].item()
        return (values[0] / values[1]).item() if values[1] > 0 else None

    def reset(self):
        self.values = None


def record(trainer, mode, name, numerator, denominator):
    metrics = trainer.custom_metrics[mode]
    if name not in metrics:
        metrics[name] = RatioMetric(synchronize=mode == "train")
    metrics[name].update(numerator, denominator)


def record_count(trainer, mode, name, count):
    metrics = trainer.custom_metrics[mode]
    if name not in metrics:
        metrics[name] = RatioMetric(synchronize=mode == "train", total=True)
    metrics[name].update(count, 1)


class CCLLMTemplate(Template):
    def _encode_truncated(self, inputs):
        encoded = super()._encode_truncated(inputs)
        rows = encoded if isinstance(encoded, list) else [encoded]
        kept = []
        for row in rows:
            labels = row.get("labels")
            if labels is not None:
                labels = list(labels)
                if labels:
                    labels[0] = -100
                if not any(label != -100 for label in labels[1:]):
                    continue
                row["labels"] = labels
            kept.append(row)
        return kept if isinstance(encoded, list) or not kept else kept[0]

    def _data_collator(self, batch, *, padding_to=None):
        padding_dummy = torch.tensor([row.get("is_padding_dummy", False) for row in batch], dtype=torch.bool)
        batch = [{key: value for key, value in row.items() if key != "is_padding_dummy"} for row in batch]
        result = super()._data_collator(batch, padding_to=padding_to)
        labels = result.get("labels")
        if labels is not None:
            labels[:, 0] = -100
            has_targets = labels[:, 1:].ne(-100).any(dim=-1)
            if not (has_targets | padding_dummy.to(labels.device)).all():
                raise ValueError("cached dataset contains zero-target rows; run prepare_data.py first")
        return result

    def compute_sft_loss(self, model, inputs, num_items_in_batch=None, trainer=None):
        inputs = dict(inputs)
        inputs["return_aux_logits"] = stage() != "phase1"
        inputs["use_cache"] = False
        return model(**inputs)


class CCLLMLoss(BaseLoss):
    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        if loss_scale is not None and not torch.all(loss_scale[labels != -100] == 1):
            raise ValueError("CC-LLM pretraining requires uniform loss_scale=all")
        trainer = self.trainer
        model = trainer.accelerator.unwrap_model(trainer.model)
        mode = "train" if model.training else "eval"
        sums = model.multitask_loss_sums(outputs, labels, per_example=mode == "eval")
        if mode == "eval":
            keys = list(sums)
            # Remove Accelerate's repeated tail examples before accumulating token-level metrics.
            per_example = torch.stack([sums[key].detach().double() for key in keys], dim=-1)
            gathered = trainer.accelerator.gather_for_metrics(per_example)
            sums = dict(zip(keys, gathered.sum(0).unbind()))
        denominator = num_items_in_batch if num_items_in_batch is not None else sums["token_count"]
        denominator = torch.as_tensor(denominator, device=outputs.logits.device).clamp_min(1)
        count = sums["token_count"]
        total = sums["clm_sum"]
        record(trainer, mode, "clm_loss", total, count)
        record_count(trainer, mode, "clm_tokens", count)
        if stage() != "phase1":
            if outputs.phonetic_logits is None:
                raise RuntimeError("auxiliary forward was not executed")
            completed = int(model.auxiliary_trained_tokens.item())
            warmup = model.config.pgca_aux_warmup_tokens
            ramp = min(1.0, completed / warmup) if mode == "train" and warmup > 0 else 1.0
            phon_weight = model.config.pgca_aux_phonetic_weight
            glyph_weight = model.config.pgca_aux_glyph_weight
            phon = sums["phon_sum"]
            glyph = sum(sums[f"{name}_sum"] for name in GLYPH_NAMES) / 3
            weighted_aux = ramp * (phon_weight * phon + glyph_weight * glyph)
            total = total + weighted_aux
            record(trainer, mode, "phon_loss", phon, count)
            record(trainer, mode, "glyph_loss", glyph, count)
            record(trainer, mode, "phon_supervised_loss", phon, sums["phon_count"])
            record(trainer, mode, "phon_accuracy", sums["phon_correct"], sums["phon_count"])
            record_count(trainer, mode, "phon_tokens", sums["phon_count"])
            record(trainer, mode, "weighted_aux_to_clm_ratio", weighted_aux, sums["clm_sum"])
            record(trainer, mode, "weighted_phon_to_clm", ramp * phon_weight * phon, sums["clm_sum"])
            record(trainer, mode, "weighted_glyph_to_clm", ramp * glyph_weight * glyph, sums["clm_sum"])
            for name in GLYPH_NAMES:
                record(trainer, mode, f"{name}_supervised_loss", sums[f"{name}_sum"], sums[f"{name}_count"])
                record(trainer, mode, f"{name}_accuracy", sums[f"{name}_correct"], sums[f"{name}_count"])
                record_count(trainer, mode, f"{name}_tokens", sums[f"{name}_count"])
            if mode == "train":
                pending = getattr(trainer, "cc_llm_pending_tokens", None)
                trainer.cc_llm_pending_tokens = count.detach() if pending is None else pending + count.detach()
        record(trainer, mode, "total_loss", total, count)
        if mode == "eval":
            # Swift.log merges this token-weighted metric into eval_loss before logging/selection.
            record(trainer, mode, "loss", total, count)
        # Swift alone applies world_size; its Trainer supplies the accumulation-window denominator.
        return total / denominator


class CCLLMOptimizer(OptimizerCallback):
    def create_optimizer(self, model=None):
        model = model if model is not None else self.trainer.model
        branch_lr = self.args.learning_rate
        if stage() == "phase2b":
            branch_lr = float(os.environ.get("PHASE2_B_BRANCH_LR", "3e-5"))
            if not math.isfinite(branch_lr) or branch_lr < 0:
                raise ValueError("PHASE2_B_BRANCH_LR must be finite and non-negative")
        decay_names = set(self.trainer.get_decay_parameter_names(model))
        groups = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            lr = branch_lr if is_branch_parameter(name) else self.args.learning_rate
            decay = self.args.weight_decay if name in decay_names else 0.0
            groups.setdefault((lr, decay), []).append(parameter)
        optimizer_cls, kwargs = HFTrainer.get_optimizer_cls_and_kwargs(self.args, model)
        return optimizer_cls([{"params": parameters, "lr": lr, "weight_decay": decay}
                              for (lr, decay), parameters in groups.items()], **kwargs)


class CCLLMCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        model = self.trainer.accelerator.unwrap_model(model)
        if bool(model.config.pgca_enabled) != (stage() != "phase1"):
            raise ValueError("PGCA_ENABLED disagrees with CC_LLM_STAGE")
        unsupported = (args.use_liger_kernel or self.trainer.template.padding_free
                       or self.trainer.template.sequence_parallel_size != 1
                       or getattr(args, "packing", False))
        if unsupported or not args.average_tokens_across_devices:
            raise ValueError("requires unpacked full logits and average_tokens_across_devices=true")
        size = len(self.trainer.train_dataset)
        if size % (self.trainer.accelerator.num_processes * args.per_device_train_batch_size):
            raise ValueError("prepare cache with the actual world size and batch size to avoid duplicated training targets")
        if not hasattr(model, "auxiliary_trained_tokens"):
            raise ValueError("build a checkpoint with --auxiliary-heads before using this plugin")
        for name, parameter in model.named_parameters():
            expected = (not is_branch_parameter(name) if stage() == "phase1"
                        else is_branch_parameter(name) if stage() == "phase2a" else True)
            if parameter.requires_grad != expected:
                raise ValueError(f"incorrect trainable state for {name}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        pending = getattr(self.trainer, "cc_llm_pending_tokens", None)
        if pending is not None:
            if dist.is_initialized():
                dist.all_reduce(pending)
            model = self.trainer.accelerator.unwrap_model(model)
            model.auxiliary_trained_tokens.add_(pending)
            self.trainer.cc_llm_pending_tokens = None

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        model = self.trainer.accelerator.unwrap_model(model)
        gradients = {group: [] for group in ("backbone", "branch", *BRANCH_GRAD_GROUPS)}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            # Read each gradient once; aggregate and detailed groups share the scalar norm square.
            squared = parameter.grad.detach().float().square().sum()
            gradients["branch" if is_branch_parameter(name) else "backbone"].append(squared)
            group = branch_gradient_group(name)
            if group is not None:
                gradients[group].append(squared)
        zero = next(model.parameters()).new_zeros((), dtype=torch.float32)
        for group, values in gradients.items():
            # This callback sees accumulated, synchronized gradients after clipping, before the update.
            norm = torch.stack(values).sum().sqrt() if values else zero
            record(self.trainer, "train", f"{group}_grad_norm", norm, 1)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoints = list(Path(args.output_dir).glob("checkpoint-*"))
        last = max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[1])) if checkpoints else None
        selected = state.best_model_checkpoint if stage() != "phase2a" else last
        selected = selected or last
        if selected is None:
            raise RuntimeError("training ended without a saved checkpoint")
        checkpoint = str(Path(selected).resolve()) + "\n"
        (Path(args.output_dir) / "next_stage_model.txt").write_text(checkpoint)
        output_root = os.environ.get("CC_LLM_OUTPUT_ROOT")
        if output_root:
            # 只发布已完成训练的结果，不按最大版本号猜测可能尚未完成的 run。
            pointer = Path(output_root) / "next_stage_model.txt"
            temporary = pointer.with_name(f".next_stage_model.{os.getpid()}.tmp")
            temporary.write_text(checkpoint)
            temporary.replace(pointer)

    def on_save(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero:
            model = self.trainer.accelerator.unwrap_model(model)
            report = {"stage": stage(), "global_step": state.global_step,
                      "auxiliary_trained_tokens": int(model.auxiliary_trained_tokens.item()),
                      "feature_schema_version": getattr(model.config, "pgca_feature_schema_version", None)}
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "cc_llm_state.json"
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


meta = deepcopy(TEMPLATE_MAPPING["qwen3"])
meta.template_type = "cc_llm_multitask"
meta.template_cls = CCLLMTemplate
register_template(meta, exist_ok=True)
loss_map["cc_llm_multitask"] = CCLLMLoss
optimizers_map["cc_llm_multitask"] = CCLLMOptimizer
callbacks_map["cc_llm_multitask"] = CCLLMCallback
