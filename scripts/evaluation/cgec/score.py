import argparse
import csv
from pathlib import Path

from ..common import aligned_predictions, ensure_dir, prediction_health, scoring_environment, sha256, write_json
from ..gec_metrics import cherrant_score, maxmatch_score, normalized


def score(input_path, predictions_path, output_path):
    _, rows = aligned_predictions(input_path, predictions_path)
    datasets = {row["dataset"] for row in rows}
    if len(datasets) != 1 or not datasets <= {"fcgec", "nacgec"}:
        raise ValueError("score one CGEC dataset/split at a time")
    dataset = next(iter(datasets))
    destination = ensure_dir(Path(output_path).parent)
    report = {"dataset": dataset, "split": rows[0]["split"], "count": len(rows),
              "input_sha256": sha256(input_path), "predictions_sha256": sha256(predictions_path),
              "inference": prediction_health(rows), "environment": scoring_environment()}
    if any(row["references"] is None for row in rows):
        if dataset != "fcgec" or any(row["references"] is not None for row in rows):
            raise ValueError("unexpected/mixed missing references")
        submission = destination / "fcgec_test_predictions.csv"
        with submission.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["UUID", "Sentence", "Correction"])
            writer.writerows((r["original_id"], r["src"], r["predict_clean"]) for r in rows)
        write_json(destination / "fcgec_test_predictions.json", {r["original_id"]: r["predict_clean"] for r in rows})
        report.update(status="exported_unlabeled_test", metrics=None, submission=str(submission),
                      note="No local test gold. UUID-indexed correction export; check current official submission packaging. No upload performed.")
    else:
        report["cherrant"] = cherrant_score(rows, destination / "cherrant", remove_punctuation=dataset == "fcgec")
        if dataset == "nacgec":
            report["maxmatch"] = maxmatch_score(rows, destination / "maxmatch")
            report["score"] = (report["cherrant"]["f0.5"] + report["maxmatch"]["f0.5"]) / 2
        else:
            wrong = [r for r in rows if r["error_flag"] == 1]
            correct = sum(normalized(r["predict_clean"], True) in {normalized(t, True) for t in r["references"]} for r in wrong)
            report["exact_match"] = {"correct": correct, "denominator_error_sentences": len(wrong),
                                     "percent": 100 * correct / len(wrong) if wrong else None}
        report["status"] = "scored"
    write_json(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(score(args.input, args.predictions, args.output))


if __name__ == "__main__":
    main()
