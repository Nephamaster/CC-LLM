from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from ..common import ensure_dir, is_cjk_char, iter_jsonl, write_json


def clean_prediction(raw: str, prompt: str) -> str:
    """CLLM's original cleanup; retain raw output separately."""
    value = raw.replace("\r\n", "\n").replace("\r", "\n")
    if value.startswith(prompt):
        value = value[len(prompt):]
    for prefix in ("纠错后的句子：", "纠错后的句子:", "修改后的句子：", "修改后的句子:", "输出：", "输出:"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    lines = [line.strip() for line in value.split("\n") if line.strip()]
    return lines[0] if lines else ""


def _safe_prf(correct: int, predicted: int, gold: int) -> Dict[str, float]:
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision * 100.0, "recall": recall * 100.0, "f1": f1 * 100.0}


def calculate_equal_length_metrics(rows: Sequence[Dict[str, Any]], prediction_key: str = "predict_clean") -> Dict[str, Any]:
    sentence_detection = [0, 0, 0]  # correct, predicted, gold
    sentence_correction = [0, 0, 0]  # correct edits, predicted edits, gold edits
    char_detection = [0, 0, 0]
    char_correction = [0, 0, 0]
    raw = Counter()
    errors: List[Dict[str, Any]] = []
    exact_count = 0
    for row in rows:
        source = row["src"]
        target = row["tgt"]
        prediction = row[prediction_key]
        raw["count"] += 1
        if len(source) != len(target):
            raw["invalid_reference_length"] += 1
            continue
        exact_count += target == prediction
        raw["correct_sentence"] += source == target
        gold_error = source != target
        pred_error = source != prediction
        if len(prediction) == len(source):
            raw["equal_length"] += 1
        elif len(prediction) == 0:
            raw["empty"] += 1
        elif len(prediction) > len(source):
            raw["overlong"] += 1
        else:
            raw["short"] += 1
        if gold_error:
            sentence_detection[2] += 1
            sentence_correction[2] += 1
        if pred_error:
            sentence_detection[1] += 1
            sentence_correction[1] += 1
        if gold_error and pred_error:
            sentence_detection[0] += 1
        if gold_error and pred_error and target == prediction:
            sentence_correction[0] += 1
        if gold_error and not pred_error:
            raw["unmodified_error"] += 1
            raw["false_negative"] += 1
        if not gold_error and pred_error:
            raw["over_correction"] += 1
            raw["false_positive"] += 1
        if len(prediction) != len(source):
            errors.append({"id": row["id"], "type": "unequal_prediction_length", "src": source, "tgt": target, "predict": prediction})
            continue
        gold_changed = {index for index, (src, tgt) in enumerate(zip(source, target)) if src != tgt}
        pred_changed = {index for index, (src, pred) in enumerate(zip(source, prediction)) if src != pred}
        char_detection[2] += len(gold_changed)
        char_detection[1] += len(pred_changed)
        char_detection[0] += len(gold_changed & pred_changed)
        char_correction[2] += len(gold_changed)
        char_correction[1] += len(pred_changed)
        char_correction[0] += sum(index in gold_changed and prediction[index] == target[index] for index in pred_changed)

    valid_count = raw["count"] - raw["invalid_reference_length"]
    correct_sentence_count = raw["correct_sentence"]
    detection = _safe_prf(*sentence_detection)
    detection["accuracy"] = (
        (sentence_detection[0] + correct_sentence_count - raw["false_positive"]) / valid_count * 100.0
        if valid_count else 0.0
    )
    correction = _safe_prf(*sentence_correction)
    correction["accuracy"] = exact_count / valid_count * 100.0 if valid_count else 0.0
    raw["false_positive_rate"] = raw["false_positive"] / correct_sentence_count if correct_sentence_count else 0.0
    result = {
        "raw": {
            **dict(raw),
            "equal_length_ratio": raw["equal_length"] / raw["count"] if raw["count"] else 0.0,
            "empty_ratio": raw["empty"] / raw["count"] if raw["count"] else 0.0,
            "overlong_ratio": raw["overlong"] / raw["count"] if raw["count"] else 0.0,
            "over_correction_ratio": raw["over_correction"] / raw["count"] if raw["count"] else 0.0,
        },
        "sentence_detection": detection,
        "sentence_correction": correction,
        "char_detection": _safe_prf(*char_detection),
        "char_correction": _safe_prf(*char_correction),
        "errors": errors,
    }
    result["table"] = {
        "detect-acc": detection["accuracy"],
        "detect-p": detection["precision"],
        "detect-r": detection["recall"],
        "detect-f1": detection["f1"],
        "correct-acc": correction["accuracy"],
        "correct-p": correction["precision"],
        "correct-r": correction["recall"],
        "correct-f1": correction["f1"],
        "FRP": raw["false_positive_rate"] * 100.0,
    }
    return result


def grouped_metrics(rows: Sequence[Dict[str, Any]], prediction_key: str = "predict_clean") -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        dataset = row.get("dataset")
        subset = row.get("subset")
        if not isinstance(dataset, str) or not isinstance(subset, str):
            raise ValueError("Every test row must contain dataset and subset metadata")
        grouped.setdefault((dataset, subset), []).append(row)

    by_dataset: Dict[str, Dict[str, Any]] = {}
    table_keys = ("detect-acc", "detect-p", "detect-r", "detect-f1", "correct-acc", "correct-p", "correct-r", "correct-f1", "FRP")
    for dataset in sorted({dataset for dataset, _subset in grouped}):
        subset_results = {}
        dataset_groups = sorted((subset, subset_rows) for (group_dataset, subset), subset_rows in grouped.items() if group_dataset == dataset)
        for subset, subset_rows in tqdm(dataset_groups, desc=f"metrics {dataset}", unit="subsets", leave=False):
            metrics = calculate_equal_length_metrics(subset_rows, prediction_key=prediction_key)
            subset_results[subset] = {
                "sample_count": len(subset_rows),
                "table": metrics["table"],
                "raw": metrics["raw"],
            }
        mean_table = {
            key: sum(result["table"][key] for result in subset_results.values()) / len(subset_results)
            for key in table_keys
        }
        by_dataset[dataset] = {
            "mean": {"subset_count": len(subset_results), "table": mean_table},
            "subsets": {} if dataset == "cscd_ns" else subset_results,
        }
    return by_dataset


def _parse_m2_operations(path: Path) -> List[List[Tuple[int, int, str, List[str]]]]:
    """Read one selected ChERRANT alignment per sentence (first T block).

    The M2 generator can output several candidate alignments as T0-A0,
    T0-A1, ... with edits for each. Applying them together is incorrect.
    """
    groups: List[List[Tuple[int, int, str, List[str]]]] = []
    current: List[Tuple[int, int, str, List[str]]] = []
    in_sentence = False
    has_candidate = False
    accept_edits = True
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith("S "):
                if in_sentence:
                    groups.append(current)
                current = []
                in_sentence = True
                has_candidate = False
                accept_edits = True
                continue
            if not line.strip():
                if in_sentence:
                    groups.append(current)
                    current = []
                    in_sentence = False
                continue
            if line.startswith("T"):
                if not in_sentence:
                    in_sentence = True
                accept_edits = not has_candidate
                has_candidate = True
                continue
            if not line.startswith("A ") or not accept_edits:
                continue
            if not in_sentence:
                in_sentence = True
            fields = line[2:].split("|||")
            if len(fields) < 3:
                continue
            position = fields[0].split()
            try:
                start, end = int(position[0]), int(position[1])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Invalid ChERRANT operation position: {line}") from exc
            operation = fields[1]
            replacement = fields[2].split() if fields[2] else []
            current.append((start, end, operation, replacement))
    if in_sentence:
        groups.append(current)
    return groups

def cherrant_equalize(rows: Sequence[Dict[str, Any]], cherrant_dir: Path) -> Tuple[List[str], Dict[str, Any]]:
    """Return predictions after retaining only length-preserving ChERRANT substitutions.

    ChERRANT strips all Unicode whitespace before computing M2 positions;
    the mapping back to original source positions must preserve that whitespace.
    """
    if not cherrant_dir.exists():
        raise FileNotFoundError(f"ChERRANT checkout not found: {cherrant_dir}")
    with tempfile.TemporaryDirectory(prefix="cllm-cherrant-") as temp_name:
        temp_dir = Path(temp_name)
        parallel = temp_dir / "parallel.txt"
        m2_path = temp_dir / "output.m2"
        with parallel.open("w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(rows, 1):
                source = row["src"]
                prediction = row["predict_clean"]
                handle.write(f"{index}\t{source}\t{prediction}\n")
        # Never simplify targets: doing so changes correctly predicted variants
        # (e.g. 乾 -> 干) even when src == predict_clean.  Only take the first
        # candidate alignment and keep the existing substitution-only policy.
        command = [
            sys.executable, "parallel_to_m2.py", "-f", str(parallel), "-o", str(m2_path),
            "-g", "char", "--multi_cheapest_strategy", "first", "--no_simplified",
        ]
        try:
            subprocess.run(command, cwd=str(cherrant_dir), check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "ChERRANT failed; run scripts/environment/run.sh and inspect its stderr"
            ) from exc
        groups = _parse_m2_operations(m2_path)
    if len(groups) != len(rows):
        raise ValueError(f"ChERRANT operation count {len(groups)} != sample count {len(rows)}")
    predictions: List[str] = []
    stats = Counter()
    for row, operations in zip(rows, groups):
        chars = list(row["src"])
        # Same normalization as ChERRANT's ''.join(source.strip().split()),
        # but retain original offsets for the character-wise substitutions.
        normalized_indices = [index for index, char in enumerate(chars) if not char.isspace()]
        for start, end, operation, replacement in operations:
            if operation == "S" and end > start and end - start == len(replacement) and all(len(char) == 1 for char in replacement):
                if start < 0 or end > len(normalized_indices):
                    raise ValueError(f"ChERRANT substitution span [{start}:{end}] out of range for source {row['src']!r}")
                for index, char in zip(normalized_indices[start:end], replacement):
                    chars[index] = char
                stats["substitution_applied"] += 1
            else:
                stats["non_equal_length_or_non_substitution"] += 1
        predictions.append("".join(chars))
    return predictions, dict(stats)

def phonetic_stats(rows: Sequence[Dict[str, Any]], prediction_key: str = "predict_clean") -> Dict[str, Any]:
    try:
        from pypinyin import Style, pinyin
    except ImportError:
        return {"available": False}

    def readings(value: str) -> List[str]:
        result = []
        for char in value:
            if not is_cjk_char(char):
                result.append("")
                continue
            result.append(pinyin(char, style=Style.NORMAL, heteronym=False)[0][0])
        return result

    predicted_replacements = 0
    predicted_non_homophone = 0
    error_replacements = 0
    error_non_homophone = 0
    reference_replacements = 0
    reference_non_homophone = 0
    for row in tqdm(rows, desc=f"phonetic {prediction_key}", unit="rows", leave=False):
        source = row["src"]
        target = row["tgt"]
        prediction = row[prediction_key]
        if len(source) != len(target) or len(source) != len(prediction):
            continue
        src_py, tgt_py, pred_py = readings(source), readings(target), readings(prediction)
        for src, tgt, pred, src_p, tgt_p, pred_p in zip(source, target, prediction, src_py, tgt_py, pred_py):
            if src != pred:
                predicted_replacements += 1
                predicted_non_homophone += src_p != pred_p
                if src != tgt:
                    error_replacements += 1
                    error_non_homophone += src_p != pred_p
            if src != tgt:
                reference_replacements += 1
                reference_non_homophone += src_p != tgt_p
    return {
        "available": True,
        "definition": "strict same pinyin without tone; pypinyin context-independent first reading",
        "non_homophone_predictions": predicted_non_homophone / predicted_replacements if predicted_replacements else 0.0,
        "non_homophone_among_errors": error_non_homophone / error_replacements if error_replacements else 0.0,
        "reference_non_homophone": reference_non_homophone / reference_replacements if reference_replacements else 0.0,
    }


def evaluate_files(
    input_paths: Sequence[Path],
    output_path: Path,
    cherrant_dir: Optional[Path] = None,
    predictions_output: Optional[Path] = None,
) -> Dict[str, Any]:
    if not input_paths:
        raise ValueError("At least one prediction JSONL input is required")
    rows: List[Dict[str, Any]] = []
    for input_path in input_paths:
        rows.extend(tqdm(iter_jsonl(input_path), desc=f"load {input_path.name}", unit="rows"))
    raw_result = calculate_equal_length_metrics(tqdm(rows, desc="metrics overall", unit="rows", leave=False))
    result: Dict[str, Any] = {
        "inputs": [str(path) for path in input_paths],
        "sample_count": len(rows),
        "raw": raw_result,
        "datasets": grouped_metrics(rows),
        "phonetic": phonetic_stats(rows),
    }
    if cherrant_dir is not None:
        predictions, cherrant_stats = cherrant_equalize(rows, cherrant_dir)
        adjusted = [dict(row, predict_cherrant=prediction) for row, prediction in zip(rows, predictions)]
        result["cherrant"] = calculate_equal_length_metrics(
            tqdm(adjusted, desc="metrics cherrant", unit="rows", leave=False), prediction_key="predict_cherrant"
        )
        result["datasets_cherrant"] = grouped_metrics(adjusted, prediction_key="predict_cherrant")
        result["cherrant_stats"] = cherrant_stats
        if predictions_output is not None:
            ensure_dir(predictions_output.parent)
            with predictions_output.open("w", encoding="utf-8", newline="\n") as handle:
                for row in adjusted:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_path, result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compute raw and ChERRANT CSC metrics")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cherrant-dir", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_files(args.input, args.output, args.cherrant_dir, args.predictions_output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
