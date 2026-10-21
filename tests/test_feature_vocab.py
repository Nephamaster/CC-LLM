from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.embedding.config import EmbeddingFeatureConfig
from src.embedding.feature_embedding import PhoneticGlyphFeatureEmbedding
from src.embedding.feature_memory import FeatureMemoryBuilder, auxiliary_feature_targets, load_feature_index
from src.vocab.bpe_state import bytes_to_unicode
from src.vocab.feature_index import FeatureIndexBuildConfig, FeatureIndexBuilder, pad_ids
from src.vocab.feature_validation import validate_char_features, validate_feature_artifacts
from src.vocab.feature_vocab import (
    FeatureVocabBuildConfig, FeatureVocabBuilder, UNK_TOKEN, collect_pinyin_candidates,
    normalize_pinyin_tone, parse_unihan, pinyin_candidates, sha256_file, split_pinyin,
)


@pytest.fixture
def artifacts(tmp_path):
    chars = "行重长乐还和得了㐂㕶呒呣嗯啊"
    hanzi = tmp_path / "hanzi_set.txt"
    hanzi.write_text("\n".join(chars) + "\n", encoding="utf-8")
    structure = tmp_path / "structure.tsv"
    structure.write_text("\n".join(f"{char}\tleft_right" for char in chars if char != "㐂"), encoding="utf-8")
    unihan = tmp_path / "Unihan"
    unihan.mkdir()
    rows = []
    for char in chars:
        rows += [f"U+{ord(char):04X}\tkTotalStrokes\t6", f"U+{ord(char):04X}\tkRSUnicode\t30.3"]
    rows += ["U+884C\tkHanyuPinyin\t10015.030:háng,xíng 10016.010:hàng",
             "U+884C\tkMandarin\txíng"]
    (unihan / "Unihan_fixture.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    root = tmp_path / "Char"
    root.mkdir()
    encoder = bytes_to_unicode()
    vocab = {"".join(encoder[b] for b in char.encode()): index for index, char in enumerate("A" + chars)}
    (root / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    config = FeatureVocabBuildConfig(hanzi_set_path=hanzi, structure_path=structure, unihan_path=unihan,
                                   output_dir=root / "features")
    built = FeatureVocabBuilder(config).build_and_write()
    index = FeatureIndexBuilder(FeatureIndexBuildConfig(
        tokenizer_dir=root, features_dir=config.output_dir,
        output_jsonl=config.output_dir / "char_feature_index.jsonl",
        output_pt=config.output_dir / "feature_index.pt",
    )).build_and_write()
    return root, config, built, index


@pytest.mark.parametrize("value,expected", [
    ("ň", "n3"), ("ḿ", "m2"), ("ǹg", "ng4"), ("m\u0301", "m2"), ("lüè", "lve4"),
    ("LU:4", "lv4"), ("ma5", "ma0"), ("ê", "ê0"), ("ế", "ê2"),
])
def test_canonical_special_syllables(value, expected):
    assert normalize_pinyin_tone(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("ju4", ("j", "v", 4)), ("jun1", ("j", "vn", 1)), ("yu2", ("", "v", 2)),
    ("wu3", ("", "u", 3)), ("liu2", ("l", "iou", 2)), ("gui1", ("g", "uei", 1)),
    ("lun2", ("l", "uen", 2)), ("ň", ("", "n", 3)), ("ḿ", ("", "m", 2)),
    ("ǹg", ("", "ng", 4)), ("<unk>", ("<unk>", "<unk>", None)),
])
def test_strict_candidate_aligned_split(value, expected):
    assert split_pinyin(value) == expected


def test_reject_original_hanzi_and_conflicting_tones():
    for text in ("㐂", "银行", "mǎ4", "ma8"):
        with pytest.raises(ValueError):
            normalize_pinyin_tone(text)


def test_dependency_and_execution_errors_do_not_fall_back():
    import sys
    import pypinyin

    with patch.dict(sys.modules, {"pypinyin": None}):
        with pytest.raises(RuntimeError, match="no silent fallback"):
            pinyin_candidates("行", {"kMandarin": "xing2"})
    with patch.object(pypinyin, "__version__", "wrong"):
        with pytest.raises(RuntimeError, match="expected pypinyin"):
            pinyin_candidates("行", {"kMandarin": "xing2"})
    with patch.object(pypinyin, "pinyin", side_effect=RuntimeError("library broken")):
        with pytest.raises(RuntimeError, match="library broken"):
            pinyin_candidates("行", {"kMandarin": "xing2"})


def test_sources_are_unioned_and_attributed():
    values = collect_pinyin_candidates("行", {"kHanyuPinyin": "10015.030,10016.010: háng,xíng", "kMandarin": "xíng"})
    assert {"xing2", "hang2"}.issubset(values)
    assert values["xing2"] == ["pypinyin", "Unihan.kHanyuPinyin", "Unihan.kMandarin"]
    with patch("pypinyin.pinyin", return_value=[]):
        assert pinyin_candidates("㕶", {"kHanyuPinyin": "00001.000:ň,ǹg", "kMandarin": "ḿ"}) == ["n3", "ng4", "m2"]
    assert pinyin_candidates("㐂", {}) == []


def test_no_silent_candidate_truncation(tmp_path):
    hanzi = tmp_path / "hanzi.txt"
    hanzi.write_text("欸\n", encoding="utf-8")
    config = FeatureVocabBuildConfig(hanzi_set_path=hanzi, max_pinyin_per_char=8,
                                   structure_path=tmp_path / "missing.tsv", unihan_path=tmp_path / "missing.zip",
                                   fallback_unihan_dir=tmp_path / "missing")
    with pytest.raises(ValueError, match="refusing silent truncation"):
        FeatureVocabBuilder(config).build()
    assert len(FeatureVocabBuilder(replace(config, max_pinyin_per_char=16)).build().char_features["欸"]["pinyin_ids"]) == 11
    with pytest.raises(ValueError, match="truncation"):
        pad_ids([3, 4], 1, 1)


def test_complete_jsonl_pt_semantic_validation(artifacts):
    root, config, built, index = artifacts
    assert built.manifest["schema_version"] == 2
    assert "kHanyuPinyin" in parse_unihan(config.unihan_path, Path("missing"))["行"]
    assert index.manifest["max_pinyin_per_char"] == 16
    report = validate_feature_artifacts(config.output_dir, len(index.jsonl_rows), root)
    assert report["passed"], report
    assert report["pt_checked"]
    assert {"xing2", "hang2"}.issubset(built.char_features["行"]["raw"]["pinyins"])
    unknown = built.char_features["㐂"]
    assert unknown["raw"]["shengmus"] == [UNK_TOKEN]
    assert unknown["raw"]["tones"] == [None]
    for name in ("pinyin", "shengmu", "yunmu", "tone"):
        assert unknown[f"{name}_ids"] == [2]
    assert unknown["pinyin_mask"] == [False]


def test_unknown_phonetics_keep_glyph_memory_and_ignore_auxiliary_labels(artifacts):
    root, _, _, index = artifacts
    config = EmbeddingFeatureConfig.from_artifacts(root, d_model=8, d_feat=4)
    tensors = load_feature_index(root / "features/feature_index.pt")
    builder = FeatureMemoryBuilder(tensors, PhoneticGlyphFeatureEmbedding(config))
    unknown = next(row["token_id"] for row in index.jsonl_rows if row["char"] == "㐂")
    memory, mask = builder(torch.tensor([[unknown, 0]]))
    assert not mask[0, 0, :-1].any() and mask[0, 0, -1]
    assert not mask[0, 1].any()
    assert torch.count_nonzero(memory[0, 0, :-1]) == 0
    targets = auxiliary_feature_targets(builder.lookup(torch.tensor([[unknown, 0]])))
    for name in ("pinyin", "shengmu", "yunmu", "tone"):
        assert (targets[name] == -100).all()
    assert targets["structure"][0, 0] == -100
    assert targets["stroke_count"][0, 0] >= 3
    assert targets["radical_stroke"][0, 0] >= 3
    # Glyph-only tables are valid; absence of pronunciation does not make the checkpoint unready.
    builder.pinyin_mask.zero_()
    builder.validate_ready()


def test_neutral_tone_and_zero_initial_are_valid_not_unknown(artifacts):
    root, _, _, index = artifacts
    tensors = load_feature_index(root / "features/feature_index.pt")
    config = EmbeddingFeatureConfig.from_artifacts(root, d_model=8, d_feat=4)
    builder = FeatureMemoryBuilder(tensors, PhoneticGlyphFeatureEmbedding(config))
    aid = next(row["token_id"] for row in index.jsonl_rows if row["char"] == "啊")
    targets = auxiliary_feature_targets(builder.lookup(torch.tensor([[aid]])))
    valid = tensors["pinyin_mask"][aid]
    assert (targets["shengmu"][0, 0, valid] >= 3).all()
    assert (targets["tone"][0, 0, valid] >= 3).all()


def test_semantics_reject_old_unknown_labels(artifacts):
    _, _, built, _ = artifacts
    built.char_features["㐂"]["raw"]["tones"] = [0]
    report = validate_char_features(built.feature_vocabs, built.char_features)
    assert not report["passed"]
    assert report["errors"]["unaligned_or_incorrect_phonetic_parts"] == 1


def test_hash_binding_rejects_same_size_id_permutation(artifacts):
    root, config, _, index = artifacts
    path = config.output_dir / "feature_vocabs/pinyin_vocab.json"
    vocab = json.loads(path.read_text())
    keys = [key for key in vocab if key not in {"<unk>", "<none>", "<pad>"}]
    vocab[keys[0]], vocab[keys[1]] = vocab[keys[1]], vocab[keys[0]]
    path.write_text(json.dumps(vocab))
    assert not validate_feature_artifacts(config.output_dir, len(index.jsonl_rows), root)["passed"]
    with pytest.raises(ValueError, match="hash mismatch"):
        EmbeddingFeatureConfig.from_artifacts(root)


def test_semantic_token_alignment_checked_beyond_hashes(artifacts):
    root, config, _, index = artifacts
    path = config.output_dir / "char_feature_index.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["token"] = "incorrect-token-piece"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest_path = config.output_dir / "feature_index_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"][path.name] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    report = validate_feature_artifacts(config.output_dir, len(index.jsonl_rows), root)
    assert not report["passed"]
    assert report["errors"]["semantic_token_alignment"] == 1


def test_stale_embedding_index_hash_rejected(artifacts, tmp_path):
    from src.pgca.build_pgca import update_config

    root, _, _, _ = artifacts
    config_path = root / "config.json"
    config_path.write_text(json.dumps({"vocab_size": 15, "hidden_size": 8,
                                       "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4}))
    embedding = EmbeddingFeatureConfig.from_artifacts(root, d_model=8, d_feat=4)
    embedding_path = root / "embedding_config.json"
    embedding.save_pretrained(embedding_path)
    with pytest.raises(ValueError, match="stale"):
        update_config(config_path, embedding_path, [0], "different-index-hash")


def test_pgca_rebuild_never_deletes_existing_output(tmp_path):
    from src.pgca.build_pgca import build_pgca_model

    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("old experiment")
    with pytest.raises(FileExistsError):
        build_pgca_model(tmp_path / "source", destination, None)
    assert marker.read_text() == "old experiment"


@pytest.mark.parametrize("contextual", [False, True])
@pytest.mark.parametrize("weights", [None, (0.07, 0.01)])
def test_v2_pgca_build_and_remote_reload(artifacts, tmp_path, contextual, weights):
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from src.pgca.build_pgca import build_pgca_model, save_pretrained_without_auto_copy

    root, _, _, index = artifacts
    base_config = Qwen3Config(vocab_size=len(index.jsonl_rows), hidden_size=8, intermediate_size=16,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=4, max_position_embeddings=32, use_cache=False)
    base = Qwen3ForCausalLM(base_config)
    base.save_pretrained(root)
    if contextual:
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast
        vocab = json.loads((root / "vocab.json").read_text())
        unknown_token = next(iter(vocab))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token=unknown_token)),
                                            unk_token=unknown_token)
        tokenizer.save_pretrained(root)
    EmbeddingFeatureConfig.from_artifacts(root, d_model=8, d_feat=4).save_pretrained(root / "embedding_config.json")
    destination = tmp_path / "new_pgca"
    options = {} if weights is None else dict(aux_phonetic_weight=weights[0], aux_glyph_weight=weights[1])
    source_config = (root / "config.json").read_bytes()
    report = build_pgca_model(root, destination, "0,1", context_encoder=contextual,
                             auxiliary_heads=contextual, zero_init_output=contextual, **options)
    assert report["feature_index"]["passed"]
    expected = weights or (0.03, 0.03)
    saved = json.loads((destination / "config.json").read_text())
    for name, value in zip(("pgca_aux_phonetic_weight", "pgca_aux_glyph_weight"), expected):
        assert saved[name] == value
        assert report[name] == value
    assert saved["pgca_aux_warmup_tokens"] == 10_000_000
    assert (root / "config.json").read_bytes() == source_config
    loaded = AutoModelForCausalLM.from_pretrained(destination, trust_remote_code=True)
    assert (loaded.config.pgca_aux_phonetic_weight, loaded.config.pgca_aux_glyph_weight) == expected
    assert loaded.config.pgca_feature_schema_version == 2
    assert loaded.config.pgca_context_enabled == contextual
    assert loaded.config.pgca_aux_enabled == contextual
    if contextual:
        assert loaded.auxiliary_trained_tokens.item() == 0
    assert loaded.config.pgca_feature_slots == 17
    assert loaded.config.pgca_feature_vocab_manifest_sha256
    assert loaded.config.pgca_feature_index_tensor_sha256 == report["feature_index_tensor_sha256"]
    torch.testing.assert_close(loaded.get_input_embeddings().weight, base.get_input_embeddings().weight)
    unknown = next(row["token_id"] for row in index.jsonl_rows if row["char"] == "㐂")
    mask = loaded.model.feature_memory_builder.pinyin_mask[unknown]
    assert not mask.any()
    ids = torch.tensor([[1, 2, unknown, 0]])
    output = loaded(ids, labels=ids)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    loaded.model.feature_memory_builder.pinyin_ids[0, 0] = 3
    save_pretrained_without_auto_copy(loaded, destination)
    with pytest.raises(RuntimeError, match="fingerprint"):
        AutoModelForCausalLM.from_pretrained(destination, trust_remote_code=True)


def test_pgca_build_rejects_invalid_auxiliary_weights_before_copying(tmp_path):
    from src.pgca.build_pgca import build_pgca_model

    for name in ("aux_phonetic_weight", "aux_glyph_weight"):
        for value in (-1, float("nan"), float("inf")):
            destination = tmp_path / "invalid"
            with pytest.raises(ValueError, match="auxiliary weight"):
                build_pgca_model(tmp_path / "source", destination, None, **{name: value})
            assert not destination.exists()


def test_pgca_build_cli_passes_explicit_auxiliary_weights(tmp_path):
    from src.pgca.build_pgca import main

    arguments = ["build", "--model-path", str(tmp_path / "char"), "--output-path", str(tmp_path / "pgca"),
                 "--auxiliary-heads", "--aux-phonetic-weight", "0.04", "--aux-glyph-weight", "0.06"]
    with patch("sys.argv", arguments), patch("src.pgca.build_pgca.build_pgca_model", return_value={}) as build:
        main()
    assert build.call_args.kwargs["aux_phonetic_weight"] == 0.04
    assert build.call_args.kwargs["aux_glyph_weight"] == 0.06


def test_pgca_validation_cli_returns_failure_status(tmp_path):
    from src.pgca.validate_pgca import main

    with patch("sys.argv", ["validate", "--model-path", str(tmp_path)]), patch(
        "src.pgca.validate_pgca.validate_pgca_model", return_value={"passed": False}
    ):
        with pytest.raises(SystemExit) as failure:
            main()
    assert failure.value.code == 1
    assert json.loads((tmp_path / "reports/pgca_validation_report.json").read_text())["passed"] is False


def test_embedding_validation_facade_does_not_ignore_failed_report(tmp_path):
    from src.embedding.build_embedding import validate

    with patch("src.embedding.build_embedding.EmbeddingValidator") as validator:
        validator.return_value.validate.return_value = {"passed": False}
        with pytest.raises(RuntimeError, match="embedding validation failed"):
            validate(SimpleNamespace(char_model_path=tmp_path))
