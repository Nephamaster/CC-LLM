import argparse
from pathlib import Path

from ..common import aligned_predictions, iter_jsonl, prediction_health, scoring_environment, sha256, signature, write_json
from .judge import prediction_signature
from .metrics import c3_metrics, fuxi_metrics


def score(input_path, predictions_path, output_path, judgments_path=None):
    _, rows = aligned_predictions(input_path, predictions_path)
    datasets = {r["dataset"] for r in rows}
    if len(datasets) != 1:
        raise ValueError("score C3 and Fuxi separately")
    judgments = {}
    judge_contracts = {}
    if judgments_path:
        by_id = {row["id"]: row for row in rows}
        for item in iter_jsonl(judgments_path):
            if item["id"] not in by_id or item["prediction_sha256"] != prediction_signature(by_id[item["id"]]):
                raise ValueError("judgment cache does not match predictions")
            judgments[item["id"]] = item
            judge_contracts[signature(item["judge"])] = item["judge"]
        if len(judge_contracts) > 1:
            raise ValueError("do not mix different judge models/configurations in one evaluation")
    dataset = next(iter(datasets))
    if dataset == "c3":
        report = c3_metrics(rows)
    elif dataset == "fuxi":
        report = fuxi_metrics(rows, judgments)
    else:
        raise ValueError(dataset)
    report.update(dataset=dataset, count=len(rows), input_sha256=sha256(input_path),
                  predictions_sha256=sha256(predictions_path),
                  judgments_sha256=sha256(judgments_path) if judgments_path else None,
                  judge=next(iter(judge_contracts.values()), None), inference=prediction_health(rows),
                  environment=scoring_environment())
    write_json(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judgments", type=Path)
    args = parser.parse_args()
    print(score(args.input, args.predictions, args.output, args.judgments))


if __name__ == "__main__":
    main()
