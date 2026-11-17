import argparse
import csv
from pathlib import Path

from ..common import aligned_predictions, ensure_dir, prediction_health, scoring_environment, write_json
from ..gec_metrics import cherrant_score, maxmatch_score, normalized


def score(input_path, predictions_path, output_path):
    originals, rows = aligned_predictions(input_path, predictions_path)
    if any(prediction.get("src") != original["src"] for original, prediction in zip(originals, rows)):
        raise ValueError("CGEC predictions were generated from different source sentences; do not reuse them")
    # Input owns the gold/split contract, including when old predictions are supplied.
    rows = [{**prediction, **original} for original, prediction in zip(originals, rows)]
    datasets = {row["dataset"] for row in rows}
    if len(datasets) != 1 or not datasets <= {"fcgec", "nacgec"}:
        raise ValueError("score one CGEC dataset/split at a time")
    dataset = next(iter(datasets))
    if len({r["split"] for r in rows}) != 1:
        raise ValueError("score one CGEC split at a time")
    if dataset == "nacgec" and rows[0]["split"] == "test" and any(
            r["references"] is not None and r.get("reference_origin") != "explicit_test_reference" for r in rows):
        raise ValueError("old NaCGEC test inputs do not declare an explicit gold file; prepare a new run")
    destination = ensure_dir(Path(output_path).parent)
    report = {"dataset": dataset, "split": rows[0]["split"], "count": len(rows),
              "inference": prediction_health(rows), "environment": scoring_environment()}
    if any(row["references"] is None for row in rows):
        if rows[0]["split"] != "test" or any(row["references"] is not None for row in rows):
            raise ValueError("unexpected/mixed missing references")
        submission = destination / f"{dataset}_test_predictions.csv"
        with submission.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["UUID", "Sentence", "Correction"])
            writer.writerows((r["original_id"], r["src"], r["predict_clean"]) for r in rows)
        write_json(destination / f"{dataset}_test_predictions.json", {r["original_id"]: r["predict_clean"] for r in rows})
        if dataset == "nacgec":
            submission = destination / "nacgec.test.output"
            submission.write_text("\n".join(r["predict_clean"].replace("\n", " ").replace("\r", " ") for r in rows) + "\n", encoding="utf-8")
        report.update(status="exported_unlabeled_test", metrics=None, submission=str(submission),
                      note="Export-only run: no gold was supplied. ID-indexed corrections preserve input order; check official submission packaging. No upload performed.")
    else:
        report["cherrant"] = cherrant_score(rows, destination / "cherrant", remove_punctuation=dataset == "fcgec")
        report["table_metrics"] = {key: report["cherrant"][key] for key in ("precision", "recall", "f0.5", "f1")}
        if dataset == "nacgec":
            report["maxmatch"] = maxmatch_score(rows, destination / "maxmatch")
            report["score"] = (report["cherrant"]["f0.5"] + report["maxmatch"]["f0.5"]) / 2
            report["primary_metric"] = "mean_cherrant_maxmatch_f0.5"
            report["table_metrics"]["score"] = report["score"]
        else:
            report["score"] = report["cherrant"]["f0.5"]
            report["primary_metric"] = "cherrant_f0.5"
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
