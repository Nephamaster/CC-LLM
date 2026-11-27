"""Serialize actual generation calls and keep raw tokens before framework cleanup."""

from contextlib import contextmanager
import json
import math
from pathlib import Path


def serializable(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [serializable(v) for v in sorted(value, key=str)]
    if hasattr(value, "to_dict"):
        return serializable(value.to_dict())
    if hasattr(value, "tolist"):
        return serializable(value.tolist())
    return str(value)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serializable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serializable(value), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def stop_description(criterion):
    fields = ("sequence", "sequence_ids", "stop_words", "stop_strings", "eos_token_id",
              "max_length", "max_time", "initial_decoder_input_length")
    return {"class": type(criterion).__name__,
            **{k: serializable(getattr(criterion, k)) for k in fields if hasattr(criterion, k)}}


class HFTrace:
    def __init__(self, model):
        self.model = model

    @contextmanager
    def observe(self, record, expected_ids):
        """Observe one single-sequence call; restore patched methods even on failure."""
        import torch

        restorations = []
        stop_restorations = []

        def patch(name, replacement):
            previous = self.model.__dict__.get(name)
            present = name in self.model.__dict__
            restorations.append((name, previous, present))
            setattr(self.model, name, replacement)

        original_generate = self.model.generate

        def generate(*args, **kwargs):
            tokens = kwargs.get("input_ids", kwargs.get("inputs"))
            if tokens is None and args:
                tokens = args[0]
            actual = tokens.detach().cpu().tolist()
            if len(actual) != 1 or actual[0] != expected_ids:
                raise ValueError("actual model.generate prompt IDs differ from the fixed experiment input")
            record["actual_prompt_ids"] = actual[0]
            record["prompt_ids_match"] = True
            record["generation_call"] = {
                k: serializable(v) for k, v in kwargs.items()
                if k not in {"input_ids", "inputs", "attention_mask", "generation_config", "stopping_criteria"}
            }
            record["generation_config_argument"] = serializable(kwargs.get("generation_config"))
            record["model_generation_config_before"] = serializable(self.model.generation_config)
            record["active_model_settings"] = {k: serializable(getattr(self.model.config, k, None)) for k in
                                               ("model_type", "pgca_enabled", "pgca_context_mode", "use_cache", "eos_token_id")}
            record["provided_stopping_criteria"] = [stop_description(c) for c in (kwargs.get("stopping_criteria") or [])]
            with torch.inference_mode():
                output = original_generate(*args, **kwargs)
            sequence = output.sequences if hasattr(output, "sequences") else output
            record["generated_ids"] = sequence[0, len(expected_ids):].detach().cpu().tolist()
            return output

        patch("generate", generate)
        if hasattr(self.model, "_prepare_generation_config"):
            original_prepare = self.model._prepare_generation_config

            def prepare(*args, **kwargs):
                output = original_prepare(*args, **kwargs)
                config = output[0] if isinstance(output, tuple) else output
                record["effective_generation_config"] = serializable(config)
                record["effective_config_observed"] = True
                return output

            patch("_prepare_generation_config", prepare)
        if hasattr(self.model, "_get_stopping_criteria"):
            original_stops = self.model._get_stopping_criteria

            def observed_call(original):
                def call(criterion, input_ids, scores, **kwargs):
                    result = original(criterion, input_ids, scores, **kwargs)
                    active = bool(result.any().item()) if hasattr(result, "any") else bool(result)
                    if active:
                        record.setdefault("stop_events", []).append({
                            "criterion": type(criterion).__name__,
                            "generated_length": input_ids.shape[-1] - len(expected_ids),
                            "last_token_id": int(input_ids[0, -1]),
                        })
                    return result
                return call

            def stops(*args, **kwargs):
                criteria = original_stops(*args, **kwargs)
                record["effective_stopping_criteria"] = [stop_description(c) for c in criteria]
                # Preserve object types/attributes used by generate's EOS and length handling.
                for criterion in criteria:
                    cls = type(criterion)
                    if not any(previous[0] is cls for previous in stop_restorations):
                        stop_restorations.append((cls, cls.__call__))
                        cls.__call__ = observed_call(cls.__call__)
                return criteria

            patch("_get_stopping_criteria", stops)
        try:
            yield
        finally:
            for cls, original in reversed(stop_restorations):
                cls.__call__ = original
            for name, previous, present in reversed(restorations):
                if present:
                    setattr(self.model, name, previous)
                else:
                    delattr(self.model, name)


def sampling_description(params):
    fields = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
              "frequency_penalty", "max_tokens", "min_tokens", "stop", "stop_token_ids",
              "all_stop_token_ids", "ignore_eos", "include_stop_str_in_output", "skip_special_tokens", "seed")
    return {key: serializable(getattr(params, key)) for key in fields if hasattr(params, key)}


@contextmanager
def observe_vllm(backend, record, expected_ids):
    from unittest.mock import patch
    from vllm import SamplingParams

    original_generate = backend._model_generate
    original_update = SamplingParams.update_from_generation_config

    def update(params, *args, **kwargs):
        output = original_update(params, *args, **kwargs)
        record.setdefault("engine_sampling_updates", []).append({
            "generation_defaults": serializable(args[0] if args else kwargs.get("generation_config")),
            "effective_sampling": sampling_description(params),
        })
        return output

    def generate(requests, *args, **kwargs):
        if len(requests) != 1 or requests[0] != expected_ids:
            raise ValueError("vLLM Harness prompt IDs differ from the fixed experiment input")
        record["actual_prompt_ids"] = requests[0]
        record["prompt_ids_match"] = True
        params = kwargs.get("sampling_params")
        if isinstance(params, list):
            params = params[0]
        record["harness_sampling_parameters"] = sampling_description(params)
        outputs = original_generate(requests, *args, **kwargs)
        answer = outputs[0].outputs[0]
        record["generated_ids"] = list(answer.token_ids)
        record["framework_finish_reason"] = answer.finish_reason
        record["framework_stop_reason"] = serializable(answer.stop_reason)
        record["vllm_raw_text"] = answer.text
        record["finish_reason_source"] = "vllm RequestOutput"
        return outputs

    with patch.object(backend, "_model_generate", generate), patch.object(SamplingParams, "update_from_generation_config", update):
        yield
    record["effective_config_observed"] = bool(record.get("engine_sampling_updates"))


def complete_record(record, tokenizer):
    from scripts.train.downstream.dev_metrics import repeated_text

    ids = record.get("generated_ids", [])
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    record["raw_text"] = raw
    record["generated_length"] = len(ids)
    ends = {word: tokenizer.encode(word, add_special_tokens=False)
            for word in ("<|im_end|>", "<|endoftext|>")}
    record["special_token_ids"] = ends
    end_ids = {v[0] for v in ends.values() if len(v) == 1}
    record["end_token_positions"] = [{"position": i, "id": token} for i, token in enumerate(ids) if token in end_ids]
    grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
    fraction = 1 - len(set(grams))/len(grams) if grams else 0
    record["repetition"] = repeated_text(raw) or (len(ids) >= 32 and fraction > 0.5)
    record["repeated_token_4gram_fraction"] = fraction
    record["contains_magic_phrase"] = "魔龙令牌" in raw
    record["contains_rare_marker"] = "𫟦" in raw
    record["hit_budget"] = len(ids) >= record["budget"]
    if record.get("finish_reason_source") != "vllm RequestOutput":
        events = record.get("stop_events", [])
        reasons = [event["criterion"] for event in events]
        record["framework_finish_reason"] = None
        record["finish_reason_source"] = "observed Transformers stopping criteria"
        record["observed_finish_reason"] = reasons or ["unobserved"]
    return record
