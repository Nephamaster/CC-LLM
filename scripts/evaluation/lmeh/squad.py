"""Pin SQuAD data/metric revisions without changing the harness request/scoring protocol."""
from functools import lru_cache, partial

import evaluate
from lm_eval.tasks.squadv2.task import SQuAD2

from .protocol import BENCHMARKS, DATA_REVISIONS


@lru_cache(maxsize=1)
def squad_metric():
    return evaluate.load("squad_v2", revision=DATA_REVISIONS["squad_metric_revision"])


def aggregate_squad(key, items):
    predictions, references = zip(*items)
    return squad_metric().compute(predictions=list(predictions), references=list(references)).get(key, 0)


class PinnedSQuAD2(SQuAD2):
    def download(self, dataset_kwargs=None, **kwargs):
        options = dict(dataset_kwargs or {}, revision=DATA_REVISIONS["datasets"][self.DATASET_PATH])
        self.config.dataset_kwargs = options
        super().download(options, **kwargs)

    def aggregation(self):
        return {key: partial(aggregate_squad, key) for key in super().aggregation()}

    def construct_requests(self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs):
        requests = super().construct_requests(
            doc, ctx, chat_template=chat_template, apply_chat_template=apply_chat_template, **kwargs)
        prompt, generation_kwargs = requests[0].arguments
        requests[0].arguments = (
            prompt,
            {**generation_kwargs, "max_gen_toks": BENCHMARKS["squadv2"]["max_gen_toks"]},
        )
        return requests
