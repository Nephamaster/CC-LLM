import argparse
from pathlib import Path

from ..common import VENDOR, aligned_predictions, prediction_health, scoring_environment, write_json
from .metrics import evaluate_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, rows = aligned_predictions(args.input, args.predictions)
    report = evaluate_files([args.predictions], args.output, VENDOR / "cherrant",
                            args.output.with_name("predictions_cherrant.jsonl"))
    report.update(inference=prediction_health(rows), environment=scoring_environment())
    write_json(args.output, report)


if __name__ == "__main__":
    main()
