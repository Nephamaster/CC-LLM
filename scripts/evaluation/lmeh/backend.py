"""HF backend with explicit context-overflow errors, including generation requests."""
import json
from pathlib import Path

from lm_eval.models.huggingface import HFLM


class StrictHFLM(HFLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        generation_config = getattr(self.model, "generation_config", None)
        self.checkpoint_max_new_tokens = getattr(generation_config, "max_new_tokens", None)
        if generation_config is not None:
            # HFLM computes a task-specific total max_length for every request.
            # A checkpoint-level max_new_tokens takes precedence over that limit.
            generation_config.max_new_tokens = None

    def _loglikelihood_tokens(self, requests, **kwargs):
        for _, context, continuation in requests:
            if len(context) + len(continuation) - 1 > self.max_length:
                raise ValueError("benchmark context exceeds MAX_LENGTH; refusing to truncate question/few-shot examples")
        return super()._loglikelihood_tokens(requests, **kwargs)

    def generate_until(self, requests, **kwargs):
        for request in requests:
            text, options = request.args
            budget = options.get("max_gen_toks", self.max_gen_toks)
            if len(self.tok_encode(text)) + budget > self.max_length:
                raise ValueError("benchmark prompt + generation budget exceeds MAX_LENGTH; refusing to truncate")
        return super().generate_until(requests, **kwargs)


def build_vllm_backend(model_path, *, dtype, batch_size, max_length, tensor_parallel_size):
    from lm_eval.models.vllm_causallms import VLLM

    class StrictVLLM(VLLM):
        def _loglikelihood_tokens(self, requests, **kwargs):
            for _, context, continuation in requests:
                if len(context) + len(continuation) - 1 > self.max_length:
                    raise ValueError("benchmark context exceeds MAX_LENGTH; refusing to truncate")
            return super()._loglikelihood_tokens(requests, **kwargs)

        def generate_until(self, requests, **kwargs):
            for request in requests:
                text, options = request.args
                budget = options.get("max_gen_toks", self.max_gen_toks)
                if len(self.tok_encode(text)) + budget > self.max_length:
                    raise ValueError("benchmark prompt + generation budget exceeds MAX_LENGTH; refusing to truncate")
            return super().generate_until(requests, **kwargs)

    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    model_impl = "transformers" if config.get("model_type") == "qwen3_pgca" else "auto"
    return StrictVLLM(
        pretrained=str(Path(model_path).resolve()),
        dtype=dtype,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        batch_size=batch_size,
        max_model_len=max_length,
        add_bos_token=False,
        model_impl=model_impl,
    )
