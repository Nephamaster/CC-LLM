"""Pin SQuAD data/metric revisions without changing the harness request/scoring protocol."""
from functools import lru_cache, partial

import evaluate
from lm_eval.tasks.squadv2.task import SQuAD2

from .protocol import DATA_REVISIONS


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
