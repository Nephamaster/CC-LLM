from __future__ import annotations

import copy
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from scripts.ancient_sft.common import Groups, Normalizer, clean, load_config, write_json
from scripts.ancient_sft.overlap import BenchmarkIndex, CorpusLocator, NearIndex
from scripts.ancient_sft.pipeline import finalize, input_inventory, prepare
from scripts.ancient_sft.sources import candidate, iter_candidates


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def config(tmp_path):
    value = load_config("scripts/ancient_sft/config.yaml")
    value.update(output_dir=str(tmp_path / "output"), dev_fraction=.25, max_book_fraction=1.0,
                 history_target=2, quotas={"c2m": 8, "m2c": 3, "punctuate": 3, "dictionary": 1, "provenance": 1})
    value["sources"] = {name: str(tmp_path / f"{name}.jsonl") for name in value["sources"]}
    history_dir = tmp_path / "HistoryTrans"
    history_dir.mkdir()
    for split in ("train", "validation", "test"):
        value["sources"][f"history_{split}"] = str(history_dir / f"HistoryTrans-{split}.jsonl")
    for path in value["sources"].values():
        jsonl(Path(path), [])
    # Real official holdout schemas, with contents unlike the training fixtures.
    for name in ("history_validation", "history_test"):
        jsonl(Path(value["sources"][name]), [{"inputs": f"官方保留古文{name}", "truth": f"现代保留译文{name}"}])
    value["benchmarks"] = {name: str(tmp_path / f"{name}.json") for name in value["benchmarks"]}
    write_json(Path(value["benchmarks"]["c3"]), [{"classical_Chinese": "完全不同的测试古文原文", "modern_Chinese": "其他测试译文"}])
    for name in ("fuxi", "fuxi_dev"):
        write_json(Path(value["benchmarks"][name]), [{"input": "不相干的测试问题", "output": "李白"}])
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    value["tokenizer_path"] = str(tokenizer)
    return value


def translation(index, book=None):
    # Distinct CJK sequences avoid accidental near-cluster mergers between fixtures.
    ancient = "".join(chr(0x4e00 + index * 25 + k) for k in range(20))
    return {"id": str(index), "task": "c2m", "input": ancient,
            "output": "现代释义" + ancient[::-1], "source": book or f"典籍{index}", "category": "史"}


class FixtureEncoder:
    info = {"backend": "test_fixture_only"}

    def __call__(self, messages):
        return {"sequence_tokens": sum(len(m["content"]) for m in messages) + 12,
                "assistant_loss_tokens": len(messages[-1]["content"]) + 1}


def test_normalization_preserves_training_text():
    norm = Normalizer()
    assert norm("學而時習之，") == norm("学而时习之")
    assert clean("學而時習之，") == "學而時習之，"
    assert clean("𠡠令監造") == "𠡠令監造"
    for text in ["缺□字", "损\ufffd坏", "文本\x00", "<|im_end|>"]:
        with pytest.raises(ValueError):
            clean(text)


def test_history_uses_truth_not_generated_results(config):
    row = {"inputs": "臣聞國家興亡在乎賢人", "truth": "我听说国家的兴亡取决于贤能的人", "results": "错误译文", "BLEU": 1}
    jsonl(Path(config["sources"]["history_train"]), [row])
    output = list(iter_candidates(config, Counter()))
    assert output[0]["messages"][1]["content"] == row["truth"]
    assert "错误译文" not in str(output)


def test_punctuation_must_align_and_preserves_variants(config):
    ancient = "學而時習之不亦說乎有朋自遠方來不亦樂乎人不知而不慍不亦君子乎"
    good = {"task": "punctuate", "input": ancient,
            "output": ancient[:12] + "，" + ancient[12:] + "。", "source": "论语"}
    bad = {**good, "output": good["output"] + "错"}
    jsonl(Path(config["sources"]["punctuate"]), [good, bad])
    stats = Counter()
    result = list(iter_candidates(config, stats))
    assert len(result) == 1
    assert "學" in result[0]["ancient"]
    assert stats["filter/punctuate/punctuation_alignment"] == 1


def test_benchmark_real_c3_fields_and_short_answer(config):
    norm = Normalizer()
    index = BenchmarkIndex(input_inventory(config), norm)
    record = candidate("c2m", "x", "1", "完全不同的测试古文原文", "这是译文", "书", "书", {})
    assert index.check(record)[0] == "exact"
    record["ancient"], record["modern"] = "不同古文", ""
    record["messages"] = [{"content": "作者是谁"}, {"content": "李白"}]
    assert index.check(record)[0] == "short_answer_only"


def test_long_benchmark_substring_is_quarantined(config):
    text = "".join(chr(0x5000 + k) for k in range(90))
    write_json(Path(config["benchmarks"]["c3"]), [{"classical_Chinese": text}])
    index = BenchmarkIndex(input_inventory(config), Normalizer())
    record = candidate("m2c", "x", "1", text[7:55], "现代译文", "书", "书", {})
    assert index.check(record)[0] == "possible"


def test_empty_or_wrong_benchmark_rejected(config):
    write_json(Path(config["benchmarks"]["c3"]), [{"question": "wrong schema"}])
    with pytest.raises(ValueError, match="no recognized"):
        BenchmarkIndex(input_inventory(config), Normalizer())


def test_near_index_checks_actual_similarity():
    db = sqlite3.connect(":memory:")
    index = NearIndex(db, .80)
    text = "".join(chr(0x5000 + k) for k in range(120))
    first, _ = index.add(text, "book1")
    _, matches = index.add(text[:55] + "变" + text[56:], "book2")
    assert (first, "book1") in matches
    _, unrelated = index.add("春眠不觉晓处处闻啼鸟夜来风雨声花落知多少", "book3")
    assert not unrelated
    db.close()


def test_group_union_is_transitive_and_order_independent():
    group = Groups()
    group.union("c", "b")
    group.union("a", "c")
    assert group.root("b") == group.root("c") == "a"


def test_missing_holdouts_fail_closed(config):
    Path(config["sources"]["history_test"]).unlink()
    with pytest.raises(FileNotFoundError, match="required inputs"):
        prepare(config)
    result = prepare(config, diagnostic=True, limit=3)
    assert result["diagnostic"]
    with pytest.raises(RuntimeError, match="diagnostic"):
        finalize(config, FixtureEncoder())


def test_holdout_blocks_gujilab_both_directions(config):
    row = translation(1)
    reverse = {**row, "id": "reverse", "task": "m2c", "input": row["output"], "output": row["input"]}
    jsonl(Path(config["sources"]["translate"]), [row, reverse])
    jsonl(Path(config["sources"]["history_test"]), [{"inputs": row["input"], "truth": row["output"]}])
    prepare(config)
    db = sqlite3.connect(Path(config["output_dir"]) / "prepared/candidates.sqlite")
    assert db.execute("SELECT COUNT(*) FROM eligible").fetchone()[0] == 0
    db.close()


def test_review_cannot_allow_exact_benchmark_leak(config):
    row = translation(1)
    jsonl(Path(config["sources"]["translate"]), [row])
    write_json(Path(config["benchmarks"]["c3"]), [{"classical_Chinese": row["input"]}])
    record = next(iter_candidates(config, Counter()))
    reviews = Path(config["output_dir"]).parent / "review.jsonl"
    jsonl(reviews, [{"id": record["id"], "decision": "allow", "reviewer": "fixture", "reason": "invalid override"}])
    config["reviews"] = str(reviews)
    report = prepare(config)
    assert report["statistics"]["benchmark/exact"] == 1
    db = sqlite3.connect(Path(config["output_dir"]) / "prepared/candidates.sqlite")
    assert db.execute("SELECT COUNT(*) FROM eligible").fetchone()[0] == 0
    db.close()


def test_database_tampering_detected(config):
    prepare(config)
    db_path = Path(config["output_dir"]) / "prepared/candidates.sqlite"
    with db_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(RuntimeError, match="database changed"):
        finalize(config, FixtureEncoder())


def test_reject_incomplete_benchmark_config(config, tmp_path):
    import yaml

    config["benchmarks"].pop("c3")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="C3"):
        load_config(path)


def test_dedup_reverse_group_and_history_source_preference(config):
    row = translation(1)
    reverse = {**row, "id": "reverse", "task": "m2c", "input": row["output"], "output": row["input"]}
    jsonl(Path(config["sources"]["translate"]), [row, reverse, {**row, "id": "duplicate"}])
    jsonl(Path(config["sources"]["history_train"]), [{"inputs": row["input"], "truth": row["output"]}])
    prepare(config)
    db = sqlite3.connect(Path(config["output_dir"]) / "prepared/candidates.sqlite")
    rows = db.execute("SELECT source,split,doc FROM eligible").fetchall()
    assert len(rows) == 2
    assert len(set(rows)) == 1 and rows[0][0] == "gujilab"
    db.close()


def test_optional_dictionary_requires_review(config):
    row = {"id": "char1", "char": "一", "source": "说文解字", "radical": "一部", "content": "惟初太始道立於一"}
    jsonl(Path(config["sources"]["corpus"]), [row])
    result = prepare(config)
    assert result["statistics"]["pending_review"] == 1
    audit = list((Path(config["output_dir"]) / "prepared/audit.jsonl").open())[0]
    entry = json.loads(audit)
    review = Path(config["output_dir"]).parent / "review.jsonl"
    jsonl(review, [{"id": entry["id"], "decision": "allow", "reviewer": "fixture", "reason": "checked source"}])
    config.update(output_dir=str(Path(config["output_dir"]).with_name("reviewed")), reviews=str(review))
    result = prepare(config)
    assert result["statistics"].get("pending_review", 0) == 0


def test_end_to_end_report_and_aligned_metadata(config):
    jsonl(Path(config["sources"]["translate"]), [translation(i) for i in range(40)])
    prepare(config)
    result = finalize(config, FixtureEncoder())
    assert result["ready_for_training"]
    assert result["tasks"]["train/c2m"]["selected"] == 8
    assert not result["quotas_passed"]  # Optional and other task shortfalls stay visible.
    final = Path(config["output_dir"]) / "final"
    for split in ("train", "dev"):
        messages = [json.loads(line) for line in (final / f"{split}.jsonl").open()]
        metadata = [json.loads(line) for line in (final / f"{split}.metadata.jsonl").open()]
        assert len(messages) == len(metadata)
        assert all(set(row) == {"messages"} for row in messages)
        assert [row["row_number"] for row in metadata] == list(range(1, len(metadata) + 1))
        assert all(row["assistant_loss_tokens"] > 0 for row in metadata)
    with pytest.raises(FileExistsError):
        finalize(config, FixtureEncoder())


def test_inputs_and_config_cannot_change_between_stages(config):
    prepare(config)
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    with pytest.raises(RuntimeError, match="configuration"):
        finalize(changed, FixtureEncoder())
    Path(config["sources"]["translate"]).write_text("\n")
    with pytest.raises(RuntimeError, match="input files changed"):
        finalize(config, FixtureEncoder())


def test_verified_legacy_prepare_can_finalize_without_rewriting_it(config):
    from scripts.ancient_sft.common import file_hash
    from scripts.ancient_sft.provenance import profile

    jsonl(Path(config["sources"]["translate"]), [translation(i) for i in range(40)])
    manifest = prepare(config)
    path = Path(config["output_dir"]) / "prepared/manifest.json"
    manifest["implementation_sha256"] = profile()["legacy_prepare"]["implementation_sha256"]
    write_json(path, manifest)
    before = file_hash(path)
    result = finalize(config, FixtureEncoder())
    assert result["ready_for_training"]
    assert result["prepared_implementation_check"] == "verified_legacy_prepare_encoder_provenance_fix"
    assert file_hash(path) == before


@pytest.mark.parametrize("changed", ["input", "database"])
def test_legacy_prepare_keeps_data_integrity_guards(config, changed):
    from scripts.ancient_sft.provenance import profile

    manifest = prepare(config)
    root = Path(config["output_dir"]) / "prepared"
    manifest["implementation_sha256"] = profile()["legacy_prepare"]["implementation_sha256"]
    write_json(root / "manifest.json", manifest)
    target = Path(config["sources"]["translate"]) if changed == "input" else root / "candidates.sqlite"
    with target.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(RuntimeError, match="input files changed|database changed"):
        finalize(config, FixtureEncoder())


def test_overlength_rejected_without_splitting(config):
    jsonl(Path(config["sources"]["translate"]), [translation(i) for i in range(5)])
    config["max_length"] = 32
    prepare(config)
    result = finalize(config, FixtureEncoder())
    assert not result["ready_for_training"]
    assert sum(result["statistics"].values()) > 0


def test_unknown_history_is_not_randomly_split(config):
    jsonl(Path(config["sources"]["history_train"]), [{"inputs": "天下大事必作於細天下難事必作於易", "truth": "天下的大事需要从细节做起难事从容易处着手"}])
    result = prepare(config)
    assert result["statistics"]["unresolved_history_group"] == 1


def test_history_can_contribute_new_pairs_using_corpus_ownership(config):
    text = "".join(chr(0x5300 + i) for i in range(100))
    row = {"source": "某部史书", "content": text, "era": "唐"}
    jsonl(Path(config["sources"]["corpus"]), [row])
    ancient = text[9:80]
    jsonl(Path(config["sources"]["history_train"]), [{"inputs": ancient, "truth": "今译" + ancient[::-1]}])
    normalize = Normalizer()
    assert CorpusLocator(Path(config["sources"]["corpus"]), normalize).find(normalize(ancient)) == [("某部史书", "唐")]
    prepare(config)
    db = sqlite3.connect(Path(config["output_dir"]) / "prepared/candidates.sqlite")
    rows = db.execute("SELECT payload FROM eligible WHERE source='historytrans'").fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0])["metadata"]["group_recovery"] == "corpus_containment"
    db.close()


@pytest.mark.skipif(os.getenv("ANCIENT_SWIFT_INTEGRATION") != "1", reason="requires pinned ms-swift environment")
@pytest.mark.parametrize("legacy", [False, True])
def test_real_swift_encoding_and_finalize(config, legacy):
    from scripts.ancient_sft.encoding import SwiftEncoder

    config["tokenizer_path"] = "models/Qwen3-1.7B-Base-Char"
    encoder = SwiftEncoder(config)
    messages = [{"role": "user", "content": "请标点：學而時習之不亦說乎"},
                {"role": "assistant", "content": "學而時習之，不亦說乎？"}]
    short = encoder(messages)
    messages[0]["content"] *= 3
    long = encoder(messages)
    assert long["sequence_tokens"] > short["sequence_tokens"]
    assert long["assistant_loss_tokens"] == short["assistant_loss_tokens"]
    raw = encoder.template.encode({"messages": messages})
    assert "<think>\n\n</think>\n\n" in encoder.template.tokenizer.decode(raw["input_ids"])
    assert long["assistant_loss_tokens"] == sum(label != -100 for label in raw["labels"][1:])
    jsonl(Path(config["sources"]["translate"]), [translation(i) for i in range(40)])
    manifest = prepare(config)
    if legacy:
        from scripts.ancient_sft.provenance import profile

        manifest["implementation_sha256"] = profile()["legacy_prepare"]["implementation_sha256"]
        write_json(Path(config["output_dir"]) / "prepared/manifest.json", manifest)
    report = finalize(config, encoder)
    assert report["ready_for_training"]
    assert report["encoding"]["backend"] == "ms-swift"
    assert report["tasks"]["train/c2m"]["assistant_loss_fraction"] == 1
    assert report["prepared_implementation_check"] == (
        "verified_legacy_prepare_encoder_provenance_fix" if legacy else "exact"
    )
