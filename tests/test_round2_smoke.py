from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from test_pgca_context import context_model
from scripts.train.ms_swift.smoke_check import BRANCH_GROUPS, check_initial_model, summarize


class EnglishTokenizer:
    def encode(self, *_args, **_kwargs):
        return [3, 5, 7, 9]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_preflight_three_cases_and_empty_labels_preserve_weights():
    model = context_model(zero_output=True)
    model.model.feature_memory_builder.structure_ids[2] = 2
    before = {name: value.clone() for name, value in model.state_dict().items()}
    report = check_initial_model(model, EnglishTokenizer())
    assert report["passed"] and report["unknown_case_available"]
    assert report["cases"]["hanzi"]["counts"]["phon_count"] > 0
    assert report["cases"]["unknown_attributes"]["counts"]["structure_count"] == 0
    for name in ("english", "zero_targets"):
        assert report["cases"][name]["phon_loss"] == 0
        assert report["cases"][name]["glyph_loss"] == 0
    assert report["cases"]["zero_targets"]["total_loss"] == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("fault", ["old_weights", "phase2_progress", "nonzero_o", "align"])
def test_preflight_rejects_wrong_initial_model(fault):
    model = context_model(zero_output=True)
    if fault == "old_weights":
        model.config.pgca_aux_phonetic_weight = 0.05
    elif fault == "phase2_progress":
        model.auxiliary_trained_tokens.fill_(1)
    elif fault == "nonzero_o":
        with torch.no_grad():
            model.model.layers[0].pgca_attn.o_proj.weight[0, 0] = 1
    else:
        model.config.pgca_enabled = False
    with pytest.raises(ValueError):
        check_initial_model(model, EnglishTokenizer())


def test_preflight_reports_absent_unknown_cases_explicitly():
    report = check_initial_model(context_model(zero_output=True), EnglishTokenizer())
    assert not report["unknown_case_available"]
    assert report["unknown_case_note"]
    assert "unknown_attributes" not in report["cases"]


@pytest.fixture
def smoke_artifacts(tmp_path):
    save_json(tmp_path / "preflight_report.json", {"passed": True, "world_size": 2, "unknown_case_available": True})
    save_json(tmp_path / "cache_phase1/preparation_report.json", {"train_targets": 32, "validation_targets": 12})
    save_json(tmp_path / "cache_phase2/preparation_report.json", {
        "train_targets": 128, "validation_targets": 12,
        "splits": {"train_a": {"targets": 64, "range": [0, 64]}, "train_b": {"targets": 64, "range": [64, 128]}}})
    config = context_model(zero_output=True).config.to_dict()
    stages = (("phase1", "M-ALIGN", 0), ("phase2a", "M-CPT/adaptation", 64), ("phase2b", "M-CPT/joint", 128))
    paths = {}
    for stage, suffix, tokens in stages:
        folder = tmp_path / "training" / suffix
        checkpoint = folder / "v0/checkpoint-4"
        checkpoint.mkdir(parents=True)
        (folder / "next_stage_model.txt").write_text(str(checkpoint))
        save_json(checkpoint / "config.json", dict(config, pgca_enabled=stage != "phase1"))
        save_json(checkpoint / "cc_llm_state.json", {"stage": stage, "auxiliary_trained_tokens": tokens})
        evaluation = {"eval_loss": 2., "eval_total_loss": 2., "eval_clm_loss": 2., "eval_clm_tokens": 12, "step": 4}
        if stage != "phase1":
            evaluation.update(eval_loss=2.2, eval_total_loss=2.2, eval_weighted_aux_to_clm_ratio=.1,
                              eval_weighted_phon_to_clm=.06, eval_weighted_glyph_to_clm=.04)
        gradients = {"step": 4, "memory(GiB)": 8.}
        for group in ("backbone", "branch", *BRANCH_GROUPS):
            frozen = group != "backbone" if stage == "phase1" else group == "backbone" if stage == "phase2a" else False
            gradients[f"{group}_grad_norm"] = 0. if frozen else .01
        state = {"global_step": 4, "log_history": [gradients, evaluation]}
        save_json(checkpoint / "trainer_state.json", state)
        paths[stage] = checkpoint
    return tmp_path, paths


def test_summary_accepts_complete_smoke_and_preserves_warmup_boundary(smoke_artifacts):
    root, _ = smoke_artifacts
    report = summarize(root, "M")
    assert report["passed"] and report["ddp_exercised"]
    assert not report["full_auxiliary_warmup_exercised"]
    assert report["final_auxiliary_warmup_fraction"] == 128 / 10_000_000


@pytest.mark.parametrize("fault", ["nan", "frozen_gradient", "zero_trainable", "wrong_count", "scaled_eval",
                                  "missing_p1_logs", "wrong_progress", "wrong_weights", "only_initial_eval"])
def test_summary_does_not_mark_failed_checks_as_passed(smoke_artifacts, fault):
    root, paths = smoke_artifacts
    checkpoint = paths["phase2a"]
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    training, evaluation = state["log_history"]
    if fault == "nan":
        evaluation["eval_clm_loss"] = float("nan")
    elif fault == "frozen_gradient":
        training["backbone_grad_norm"] = .1
    elif fault == "zero_trainable":
        training["context_encoder_grad_norm"] = 0.
    elif fault == "wrong_count":
        evaluation["eval_clm_tokens"] = 24
    elif fault == "scaled_eval":
        evaluation["eval_loss"] *= 2
    elif fault == "missing_p1_logs":
        del training["pgca_q_proj_grad_norm"]
    elif fault == "wrong_progress":
        save_json(checkpoint / "cc_llm_state.json", {"stage": "phase2a", "auxiliary_trained_tokens": 32})
    elif fault == "wrong_weights":
        config = json.loads((checkpoint / "config.json").read_text())
        config["pgca_aux_glyph_weight"] = .02
        save_json(checkpoint / "config.json", config)
    else:
        evaluation["step"] = 0
    save_json(checkpoint / "trainer_state.json", state)
    with pytest.raises(ValueError):
        summarize(root, "M")


def test_summary_rejects_overlapping_ab_ranges(smoke_artifacts):
    root, _ = smoke_artifacts
    path = root / "cache_phase2/preparation_report.json"
    report = json.loads(path.read_text())
    report["splits"]["train_b"]["range"][0] = 63
    save_json(path, report)
    with pytest.raises(ValueError, match="contiguous partition"):
        summarize(root, "M")
