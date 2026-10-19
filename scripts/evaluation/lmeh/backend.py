"""HF backend with explicit context-overflow errors, including generation requests."""
from lm_eval.models.huggingface import HFLM


class StrictHFLM(HFLM):
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
