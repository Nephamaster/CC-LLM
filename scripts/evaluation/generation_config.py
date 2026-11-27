"""Evaluation decoding identity and observations at the actual generation boundary."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
from uuid import uuid4

EVALUATION_SEED = 42
GENERATION_PROTOCOL = {
    "version": "greedy-seed42-v1", "do_sample": False, "temperature": 0.0,
    "num_return_sequences": 1, "seed": EVALUATION_SEED,
}
HF_FIELDS = (
    "do_sample", "temperature", "top_p", "top_k", "min_p", "typical_p",
    "num_beams", "num_return_sequences", "repetition_penalty", "no_repeat_ngram_size",
    "length_penalty", "max_length", "max_new_tokens", "min_length", "min_new_tokens",
    "eos_token_id", "bos_token_id", "pad_token_id", "forced_eos_token_id",
    "forced_bos_token_id", "bad_words_ids", "suppress_tokens", "begin_suppress_tokens",
    "sequence_bias", "use_cache", "cache_implementation", "stop_strings", "stop_words",
)
SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
    "frequency_penalty", "max_tokens", "min_tokens", "n", "seed", "stop",
    "stop_token_ids", "all_stop_token_ids", "ignore_eos", "include_stop_str_in_output",
    "skip_special_tokens",
)


def json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def config_values(config, fields):
    if isinstance(config, dict):
        return {key: json_value(config[key]) for key in fields if key in config}
    return {key: json_value(getattr(config, key)) for key in fields if hasattr(config, key)}


@contextmanager
def replace_method(obj, name, replacement):
    present, previous = name in vars(obj), vars(obj).get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if present:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


class GenerationConfigLog:
    """Append effective configs, without storing prompts, training args or credentials.

    Hooks are scoped to a synchronous generation dispatch and restored on failure.
    An unavailable hook is recorded as unobserved rather than inferred from a request.
    """

    def __init__(self, path, backend):
        self.path, self.backend = Path(path), backend
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.calls = self.observed_calls = 0

    def write(self, record):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_value(record), ensure_ascii=False, allow_nan=False) + "\n")

    @contextmanager
    def capture(self, *, model=None, engine=None, request=None):
        self.calls += 1
        base = {"call_id": uuid4().hex, "backend": self.backend, "request": request or {}}
        observed = False
        error = None

        def save(phase, config, **extra):
            nonlocal observed
            observed |= phase == "effective"
            self.write({**base, "phase": phase, "config": config, **extra})

        try:
            with ExitStack() as stack:
                if model is not None and hasattr(model, "_get_stopping_criteria"):
                    original_stops = model._get_stopping_criteria

                    def stops(*args, **kwargs):
                        criteria = original_stops(*args, **kwargs)
                        config = kwargs.get("generation_config", args[0] if args else None)
                        descriptions = [{"class": type(c).__name__, **config_values(c, (
                            "max_length", "eos_token_id", "stop_strings", "stop_words", "sequence"))}
                                        for c in criteria]
                        save("effective", config_values(config, HF_FIELDS),
                             source="model._get_stopping_criteria", stopping_criteria=descriptions)
                        return criteria

                    stack.enter_context(replace_method(model, "_get_stopping_criteria", stops))
                if engine is not None and hasattr(engine, "_prepare_generation_config"):
                    original_prepare = engine._prepare_generation_config

                    def prepare(*args, **kwargs):
                        config = original_prepare(*args, **kwargs)
                        save("prepared", config_values(config, SAMPLING_FIELDS if self.backend == "vllm" else HF_FIELDS),
                             source="Swift._prepare_generation_config")
                        return config

                    stack.enter_context(replace_method(engine, "_prepare_generation_config", prepare))
                if self.backend == "vllm":
                    from vllm import SamplingParams

                    if hasattr(SamplingParams, "update_from_generation_config"):
                        original_update = SamplingParams.update_from_generation_config

                        def update(params, *args, **kwargs):
                            result = original_update(params, *args, **kwargs)
                            defaults = args[0] if args else kwargs.get("generation_config", {})
                            save("effective", config_values(params, SAMPLING_FIELDS),
                                 source="SamplingParams.update_from_generation_config",
                                 engine_generation_defaults=config_values(defaults, HF_FIELDS))
                            return result

                        stack.enter_context(replace_method(SamplingParams, "update_from_generation_config", update))
                yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self.observed_calls += observed
            self.write({**base, "phase": "dispatch_end", "effective_config_observed": observed,
                        "error_type": error})

    @contextmanager
    def harness(self, backend):
        original = backend._model_generate

        def generate(*args, **kwargs):
            if self.backend == "vllm":
                enabled = kwargs.get("generate", args[1] if len(args) > 1 else False)
                if not enabled:  # Candidate loglikelihood is not free generation.
                    return original(*args, **kwargs)
                params = kwargs.get("sampling_params", args[2] if len(args) > 2 else None)
                params = params if isinstance(params, list) else [params]
                request = {"sampling_parameters": [config_values(p, SAMPLING_FIELDS) for p in params],
                           "engine_seed": backend.model_args.get("seed")}
                model = None
            else:
                request = {"generation_arguments": config_values(kwargs, HF_FIELDS),
                           "max_length": kwargs.get("max_length", args[1] if len(args) > 1 else None)}
                model = backend.model
            inputs = args[0] if args else kwargs.get("requests", kwargs.get("context"))
            request["batch_size"] = len(inputs) if inputs is not None else None
            with self.capture(model=model, request=request):
                return original(*args, **kwargs)

        with replace_method(backend, "_model_generate", generate):
            yield

    def summary(self):
        return {"path": str(self.path.resolve()), "scope": "current process dispatches; JSONL retains prior attempts",
                "generation_calls": self.calls, "observed_calls": self.observed_calls,
                "all_calls_observed": self.observed_calls == self.calls if self.calls else None}
