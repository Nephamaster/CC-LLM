"""Semantic validation and artifact binding for phonetic/glyph features."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .feature_vocab import (
    FEATURE_SCHEMA_VERSION, NONE_TOKEN, PAD_TOKEN, UNK_TOKEN, POLYPHONE_EXPECTATIONS,
    normalize_pinyin_tone, sha256_file, split_pinyin,
)

PHONETIC = ("pinyin", "shengmu", "yunmu", "tone")
GLYPH = ("stroke_count", "radical_stroke", "structure")


def validate_char_features(vocabs: dict, features: dict) -> dict:
    errors, examples, counts = Counter(), [], Counter()

    def fail(reason, char):
        errors[reason] += 1
        if len(examples) < 30:
            examples.append({"reason": reason, "char": char})

    for name in (*PHONETIC, *GLYPH):
        vocab = vocabs[name]
        if sorted(vocab.values()) != list(range(len(vocab))) or any(
            vocab.get(token) != index for index, token in enumerate((PAD_TOKEN, NONE_TOKEN, UNK_TOKEN))
        ):
            fail("invalid_feature_vocab_ids", name)
    for char, feat in features.items():
        try:
            raw = feat["raw"]
            pinyins = raw["pinyins"]
            if not pinyins or len(set(pinyins)) != len(pinyins) or any(v in {PAD_TOKEN, NONE_TOKEN} for v in pinyins):
                fail("empty_or_duplicate_candidates", char)
            expected_mask = [value != UNK_TOKEN for value in pinyins]
            counts[str(sum(expected_mask))] += 1
            if feat.get("pinyin_mask") != expected_mask:
                fail("invalid_pinyin_mask", char)
            parts = [split_pinyin(value) for value in pinyins]
            if any(normalize_pinyin_tone(value) != value for value in pinyins):
                fail("noncanonical_pinyin", char)
            if (raw["shengmus"], raw["yunmus"], raw["tones"]) != (
                [p[0] for p in parts], [p[1] for p in parts], [p[2] for p in parts],
            ):
                fail("unaligned_or_incorrect_phonetic_parts", char)
            if not POLYPHONE_EXPECTATIONS.get(char, set()).issubset(pinyins):
                fail("missing_common_readings", char)
            for name, values in zip(PHONETIC, (pinyins, raw["shengmus"], raw["yunmus"], raw["tones"])):
                labels = [str(v) if v is not None else UNK_TOKEN for v in values]
                expected = [vocabs[name][v] for v in labels]
                if feat[f"{name}_ids"] != expected:
                    fail("raw_to_id_mismatch", char)
            for name in GLYPH:
                if feat[f"{name}_id"] != vocabs[name][raw[name]]:
                    fail("raw_to_id_mismatch", char)
            for value in pinyins:
                if value != UNK_TOKEN and not raw.get("pinyin_sources", {}).get(value):
                    fail("missing_pronunciation_provenance", char)
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"invalid_record:{type(exc).__name__}", char)
    return {"passed": not errors, "errors": dict(errors), "examples": examples,
            "characters": len(features), "candidate_distribution": dict(counts)}


def check_vocab_binding(features_dir: Path) -> dict:
    manifest = json.loads((features_dir / "feature_vocab_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("legacy feature schema: rebuild feature-vocab and feature-index; do not mix old IDs")
    expected = {f"feature_vocabs/{name}_vocab.json" for name in (*PHONETIC, *GLYPH)} | {"char_features.json"}
    hashes = manifest.get("artifact_sha256", {})
    if set(hashes) != expected:
        raise ValueError("incomplete feature-vocab artifact hashes")
    for relative in sorted(expected):
        if sha256_file(features_dir / relative) != hashes[relative]:
            raise ValueError(f"feature-vocab hash mismatch: {relative}")
    return manifest


def validate_feature_artifacts(features_dir: Path, vocab_size: int, tokenizer_dir: Path | None = None) -> dict:
    from .feature_index import load_feature_vocabs

    errors = Counter()
    try:
        feature_manifest = check_vocab_binding(features_dir)
        vocabs = load_feature_vocabs(features_dir)
        features = json.loads((features_dir / "char_features.json").read_text(encoding="utf-8"))
        semantics = validate_char_features(vocabs, features)
        manifest = json.loads((features_dir / "feature_index_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("feature index is not schema v2")
        if manifest["feature_vocab_manifest_sha256"] != sha256_file(features_dir / "feature_vocab_manifest.json"):
            raise ValueError("feature index belongs to a different feature vocab")
        if tokenizer_dir is not None and manifest["semantic_vocab_sha256"] != sha256_file(tokenizer_dir / "vocab.json"):
            raise ValueError("feature index belongs to a different semantic vocabulary")
        hashes = manifest["artifact_sha256"]
        if "char_feature_index.jsonl" not in hashes or set(hashes) - {"char_feature_index.jsonl", "feature_index.pt"}:
            raise ValueError("incomplete or unexpected feature-index hashes")
        for name, expected in hashes.items():
            if sha256_file(features_dir / name) != expected:
                raise ValueError(f"feature-index hash mismatch: {name}")
        pt = None
        if "feature_index.pt" in manifest["artifact_sha256"]:
            import torch

            pt = torch.load(features_dir / "feature_index.pt", map_location="cpu", weights_only=True)
        width = manifest["max_pinyin_per_char"]
        if width != feature_manifest["max_pinyin_per_char"]:
            raise ValueError("feature index width differs from feature vocabulary")
        semantic = None
        if tokenizer_dir is not None:
            from .bpe_state import bytes_to_unicode, decode_bpe_piece

            semantic = {value: key for key, value in json.loads((tokenizer_dir / "vocab.json").read_text()).items()}
            byte_decoder = {piece: byte for byte, piece in bytes_to_unicode().items()}
        count, hanzi = 0, 0
        with (features_dir / "char_feature_index.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                token_id = count
                count += 1
                if row["token_id"] != token_id:
                    errors["token_id_order"] += 1
                if semantic is not None:
                    token = semantic[token_id]
                    decoded = decode_bpe_piece(token, byte_decoder)
                    expected_char = decoded if decoded in features else None
                    if row["token"] != token or row.get("char") != expected_char or row["is_hanzi"] != (expected_char is not None):
                        errors["semantic_token_alignment"] += 1
                ref = features.get(row.get("char")) if row["is_hanzi"] else None
                if row["is_hanzi"] and ref is None:
                    errors["unknown_hanzi_row"] += 1
                hanzi += bool(row["is_hanzi"])
                for name in PHONETIC:
                    ids = list(ref[f"{name}_ids"]) if ref else []
                    expected_ids = ids + [vocabs[name][NONE_TOKEN]] * (width - len(ids))
                    expected_mask = ([v > 2 for v in ids] if ref else []) + [False] * (width - len(ids))
                    if row[f"{name}_ids"] != expected_ids or row["label_masks"][name] != expected_mask:
                        errors[f"{name}_alignment"] += 1
                    if name == "pinyin" and row["pinyin_mask"] != expected_mask:
                        errors["pinyin_mask"] += 1
                    if pt is not None:
                        if pt[f"{name}_ids"][token_id].tolist() != expected_ids:
                            errors["pt_ids"] += 1
                        if pt[f"{name}_mask"][token_id].tolist() != expected_mask:
                            errors["pt_masks"] += 1
                for name in GLYPH:
                    expected = ref[f"{name}_id"] if ref else vocabs[name][NONE_TOKEN]
                    valid = ref is not None and expected > 2
                    if row[f"{name}_id"] != expected or row["label_masks"][name] != valid:
                        errors[f"{name}_alignment"] += 1
                    if pt is not None and (pt[f"{name}_ids"][token_id].item() != expected or
                                           pt[f"{name}_mask"][token_id].item() != valid):
                        errors["pt_glyph"] += 1
                if pt is not None and bool(pt["is_hanzi"][token_id]) != bool(row["is_hanzi"]):
                    errors["pt_is_hanzi"] += 1
        if count != vocab_size or manifest["vocab_size"] != count:
            errors["vocab_size"] += 1
        if pt is not None and any(tensor.shape[0] != count for tensor in pt.values()):
            errors["pt_row_count"] += 1
        return {"passed": not errors and semantics["passed"], "rows": count, "hanzi_rows": hanzi,
                "errors": dict(errors), "semantics": semantics, "pt_checked": pt is not None}
    except (KeyError, ValueError, IndexError, OSError, TypeError) as exc:
        return {"passed": False, "reason": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    args = parser.parse_args()
    vocab_size = len(json.loads((args.tokenizer_dir / "vocab.json").read_text(encoding="utf-8")))
    report = validate_feature_artifacts(args.features_dir, vocab_size, args.tokenizer_dir)
    from .feature_vocab import write_json

    write_json(args.features_dir / "feature_validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
