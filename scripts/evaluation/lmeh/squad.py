"""SQuAD v2 task with an explicit generation budget."""
from lm_eval.tasks.squadv2.task import SQuAD2

from .protocol import BENCHMARKS


class BudgetedSQuAD2(SQuAD2):
    def construct_requests(self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs):
        requests = super().construct_requests(
            doc, ctx, chat_template=chat_template, apply_chat_template=apply_chat_template, **kwargs)
        prompt, generation_kwargs = requests[0].arguments
        requests[0].arguments = (
            prompt,
            {**generation_kwargs, "max_gen_toks": BENCHMARKS["squadv2"]["max_gen_toks"]},
        )
        return requests
