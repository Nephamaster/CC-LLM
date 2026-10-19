from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluation import common as evaluation_common
from scripts.validation.common import default_report_path


ROOT = Path(__file__).resolve().parents[1]
CPT = ROOT / "scripts/train/ms_swift/cpt"


@pytest.fixture
def recorder(tmp_path):
    program = tmp_path / "python"
    program.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["CAPTURE"], "a") as handle:
    handle.write(json.dumps(args) + "\\n")
if args[:4] == ["-B", "-m", "swift.cli.main", "pt"]:
    if os.environ.get("CAPTURE_ENV"):
        with open(os.environ["CAPTURE_ENV"], "a") as handle:
            handle.write(json.dumps({key: os.environ.get(key) for key in
                                    ("CC_LLM_STAGE", "PGCA_ENABLED", "PHASE2_B_BRANCH_LR")}) + "\\n")
    output = Path(args[args.index("--output_dir") + 1])
    checkpoint = output / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    (output / "next_stage_model.txt").write_text(str(checkpoint) + "\\n")
''', encoding="utf-8")
    program.chmod(0o755)
    capture = tmp_path / "calls.jsonl"
    return program, capture


def environment(**overrides):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for key in ("MODEL", "MODEL_NAME", "BASE_MODEL_NAME", "CC_ROOT", "PYTHON", "PYTHON_CMD", "OUT", "ALIGN_OUT",
                "TRAIN_ROOT", "DATA_DIR", "CACHE_DIR", "RUN_STAGE", "RUN_NAME", "LORA", "RESUME_A", "RESUME_B",
                "RESUME_FROM_CHECKPOINT", "DATASET", "SPLIT", "FEWSHOT", "STAGE",
                "CUDA_VISIBLE_DEVICES", "NPROC_PER_NODE", "TRAIN_BATCH_SIZE", "EVAL_BATCH_SIZE",
                "GRAD_ACCUMULATION", "DATASET_NUM_PROC", "TORCH_DTYPE", "GRADIENT_CHECKPOINTING",
                "OPTIM", "ADAM_BETA1", "ADAM_BETA2", "ADAM_EPSILON", "WEIGHT_DECAY",
                "LR_SCHEDULER_TYPE", "WARMUP_RATIO", "MAX_GRAD_NORM", "LOGGING_STEPS",
                "EVAL_STEPS", "SAVE_STEPS", "SAVE_TOTAL_LIMIT", "SEED", "DATA_SEED", "MAX_LENGTH",
                "PHASE1_TOKENS", "PHASE2_TOKENS", "ADAPTATION_TOKENS", "PHASE1_LR",
                "PHASE2_A_LR", "PHASE2_B_LR", "PHASE2_B_BRANCH_LR", "CAPTURE_ENV"):
        env.pop(key, None)
    env.update({name: str(value) for name, value in overrides.items()})
    return env


def captured(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def option(args, name):
    return args[args.index(name) + 1]


def test_training_root_and_env_are_resolved_without_cwd(tmp_path):
    result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s\\n" "$CC_ROOT" "$PYTHON_CMD" "$MODEL_NAME" "$TRAIN_ROOT"',
                             "_", str(CPT / "common.sh")], cwd=tmp_path, env=environment(), text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [str(ROOT), str(ROOT / "scripts/train/ms_swift/env/bin/python"),
                                        "Qwen3-1.7B-Base-PGCA", str(ROOT / "outputs/train")]


@pytest.mark.parametrize("overrides", [{}, {
    "TRAIN_BATCH_SIZE": "2", "EVAL_BATCH_SIZE": "3", "GRAD_ACCUMULATION": "4",
    "DATASET_NUM_PROC": "6", "TORCH_DTYPE": "float32", "GRADIENT_CHECKPOINTING": "false",
    "OPTIM": "adamw_torch", "ADAM_BETA1": "0.8", "ADAM_BETA2": "0.99", "ADAM_EPSILON": "1e-7",
    "WEIGHT_DECAY": "0.1", "LR_SCHEDULER_TYPE": "cosine", "WARMUP_RATIO": "0.05", "MAX_GRAD_NORM": "0.5",
    "LOGGING_STEPS": "2", "EVAL_STEPS": "4", "SAVE_STEPS": "8", "SAVE_TOTAL_LIMIT": "2",
    "SEED": "123", "DATA_SEED": "456", "MAX_LENGTH": "1024", "PHASE1_TOKENS": "1000",
    "PHASE2_TOKENS": "2000", "ADAPTATION_TOKENS": "200", "PHASE1_LR": "2e-5",
    "PHASE2_A_LR": "7e-5", "PHASE2_B_LR": "4e-6", "PHASE2_B_BRANCH_LR": "8e-5",
}])
def test_phase1_and_phase2_use_canonical_names_and_same_python(tmp_path, recorder, overrides):
    program, capture = recorder
    project = tmp_path / "project"
    model = project / "models" / "ExampleBase-PGCA"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    data = tmp_path / "data"
    for split in ("train", "validation"):
        (data / split).mkdir(parents=True)
        (data / split / "part.jsonl").write_text("{}\n")
    env = environment(CC_ROOT=project, BASE_MODEL_NAME="ExampleBase", PYTHON_CMD=program,
                      DATA_DIR=data, CAPTURE=capture, CUDA_VISIBLE_DEVICES="0,1",
                      CAPTURE_ENV=tmp_path / "env.jsonl", **overrides)
    for phase in ("phase1.sh", "phase2.sh"):
        subprocess.run(["bash", str(CPT / phase)], cwd=tmp_path, env=env, check=True)
    calls = captured(capture)
    training = [args for args in calls if args[:4] == ["-B", "-m", "swift.cli.main", "pt"]]
    assert len(training) == 3
    align = project / "outputs/train/ExampleBase-PGCA-ALIGN"
    cpt = project / "outputs/train/ExampleBase-PGCA-CPT"
    assert [option(args, "--output_dir") for args in training] == [str(align), str(cpt / "adaptation"), str(cpt / "joint")]
    assert option(training[1], "--model") == str(align / "checkpoint-1")
    assert option(training[2], "--model") == str(cpt / "adaptation/checkpoint-1")
    assert (cpt / "next_stage_model.txt").read_text().strip() == str(cpt / "joint/checkpoint-1")
    assert [option(args, "--learning_rate") for args in training] == [
        overrides.get(key, default) for key, default in
        (("PHASE1_LR", "1e-5"), ("PHASE2_A_LR", "5e-5"), ("PHASE2_B_LR", "1e-5"))]
    assert [option(args, "--max_length") for args in training] == [
        overrides.get("MAX_LENGTH", default) for default in ("2048", "4096", "4096")]
    shared_options = {
        "TRAIN_BATCH_SIZE": ("--per_device_train_batch_size", "1"),
        "EVAL_BATCH_SIZE": ("--per_device_eval_batch_size", "1"),
        "GRAD_ACCUMULATION": ("--gradient_accumulation_steps", "8"),
        "TORCH_DTYPE": ("--torch_dtype", "bfloat16"),
        "GRADIENT_CHECKPOINTING": ("--gradient_checkpointing", "true"),
        "OPTIM": ("--optim", "adamw_torch_fused"), "ADAM_BETA1": ("--adam_beta1", "0.9"),
        "ADAM_BETA2": ("--adam_beta2", "0.95"), "ADAM_EPSILON": ("--adam_epsilon", "1e-8"),
        "WEIGHT_DECAY": ("--weight_decay", "0.0"), "LR_SCHEDULER_TYPE": ("--lr_scheduler_type", "linear"),
        "WARMUP_RATIO": ("--warmup_ratio", "0.03"), "MAX_GRAD_NORM": ("--max_grad_norm", "1.0"),
        "LOGGING_STEPS": ("--logging_steps", "10"), "EVAL_STEPS": ("--eval_steps", "500"),
        "SAVE_STEPS": ("--save_steps", "500"), "SAVE_TOTAL_LIMIT": ("--save_total_limit", "3"),
        "SEED": ("--seed", "42"), "DATA_SEED": ("--data_seed", "42"),
    }
    for args in training:
        assert option(args, "--external_plugins") == str(CPT / "plugin.py")
        assert option(args, "--logging_dir") == str(Path(option(args, "--output_dir")) / "logs")
        for key, (flag, default) in shared_options.items():
            assert option(args, flag) == overrides.get(key, default)
        assert option(args, "--num_train_epochs") == "1"
        assert option(args, "--average_tokens_across_devices") == "true"
    exports = [args for args in calls if args[:4] == ["-B", "-m", "swift.cli.main", "export"]]
    for args in exports:
        assert option(args, "--dataset_num_proc") == overrides.get("DATASET_NUM_PROC", "8")
        assert option(args, "--data_seed") == overrides.get("DATA_SEED", "42")
    prepared = [args for args in calls if args[1] == str(CPT / "prepare_data.py")]
    for i, args in enumerate(prepared, start=1):
        assert option(args, "--batch-size") == overrides.get("TRAIN_BATCH_SIZE", "1")
        assert option(args, "--world-size") == "2"
        assert option(args, "--seed") == overrides.get("DATA_SEED", "42")
        assert option(args, "--total-tokens") == overrides.get(f"PHASE{i}_TOKENS", "1000000000" if i == 1 else "10000000000")
        assert option(args, "--adaptation-tokens") == ("0" if i == 1 else overrides.get("ADAPTATION_TOKENS", "100000000"))
        length = overrides.get("MAX_LENGTH", "2048" if i == 1 else "4096")
        assert Path(option(args, "--output")).name == f"phase{i}_{length}"
    stages = captured(tmp_path / "env.jsonl")
    assert [item["CC_LLM_STAGE"] for item in stages] == ["phase1", "phase2a", "phase2b"]
    assert [item["PGCA_ENABLED"] for item in stages] == ["false", "true", "true"]
    assert all(item["PHASE2_B_BRANCH_LR"] == overrides.get("PHASE2_B_BRANCH_LR", "3e-5") for item in stages[1:])
    assert all(args[:2] == ["-B", "-m"] or args[1] == str(CPT / "prepare_data.py") for args in calls)


@pytest.mark.parametrize("task,variables,relative", [
    ("csc", {}, "csc/ExampleBase-PGCA-CPT-CSC"),
    ("cgec", {"DATASET": "fcgec", "SPLIT": "valid"}, "fcgec/ExampleBase-PGCA-CPT-CSC/valid"),
    ("ancient", {"DATASET": "fuxi", "FEWSHOT": "5"}, "fuxi/ExampleBase-PGCA-CPT-CSC/shot5"),
])
def test_evaluation_defaults_isolate_model_and_protocol(tmp_path, recorder, task, variables, relative):
    program, capture = recorder
    env = environment(STAGE="prepare", MODEL_NAME="ExampleBase-PGCA-CPT-CSC", PYTHON=program, CAPTURE=capture, **variables)
    subprocess.run(["bash", str(ROOT / "scripts/evaluation" / task / "run.sh")], cwd=tmp_path, env=env, check=True)
    assert option(captured(capture)[0], "--output") == str(ROOT / "outputs/evaluation" / relative / "input.jsonl")


def test_evaluation_env_is_task_local(tmp_path):
    task = ROOT / "scripts/evaluation/csc"
    command = 'source "$1"; configure_evaluation "$2" csc; printf "%s\\n" "$PYTHON" "$OUT"'
    result = subprocess.run(["bash", "-c", command, "_", str(task.parent / "common.sh"), str(task)],
                            cwd=tmp_path, env=environment(MODEL_NAME="ExampleBase-CSC"), check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == [str(task / "env/bin/python"), str(ROOT / "outputs/evaluation/csc/ExampleBase-CSC")]


def test_evaluation_requires_a_model_name_for_default_outputs(tmp_path, recorder):
    program, capture = recorder
    result = subprocess.run(["bash", str(ROOT / "scripts/evaluation/csc/run.sh")], cwd=tmp_path,
                            env=environment(STAGE="prepare", PYTHON=program, CAPTURE=capture), capture_output=True, text=True)
    assert result.returncode != 0 and "MODEL_NAME" in result.stderr
    assert not capture.exists()


def test_scoring_fingerprint_prunes_env_and_bytecode(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_common, "__file__", str(tmp_path / "common.py"))
    source = tmp_path / "metric.py"
    source.write_text("original")
    before = evaluation_common.scoring_environment()["implementation_and_assets_sha256"]
    for directory in ("env/lib", "csc/env/lib", ".venv/lib", "__pycache__"):
        path = tmp_path / directory
        path.mkdir(parents=True)
        (path / "dependency.py").write_text("not project scoring code")
    assert evaluation_common.scoring_environment()["implementation_and_assets_sha256"] == before
    source.write_text("changed")
    assert evaluation_common.scoring_environment()["implementation_and_assets_sha256"] != before


def test_validation_reports_are_model_scoped():
    assert default_report_path(Path("models/ExampleBase-PGCA"), "phase0_forward") == (
        ROOT / "outputs/evaluation/phase0_forward/ExampleBase-PGCA/report.json")
    with pytest.raises(ValueError, match="model-name"):
        default_report_path(Path("output/checkpoint-1"), "phase0_forward")
    assert default_report_path(Path("output/checkpoint-1"), "phase0_forward", "ExampleBase-PGCA-ALIGN").parent.name == "ExampleBase-PGCA-ALIGN"


def test_generated_outputs_and_envs_are_git_ignored():
    paths = ["outputs/train/ExampleBase-CPT/checkpoint-1/model.bin", "scripts/train/ms_swift/env/bin/python",
             "scripts/evaluation/lmeh/env/bin/python"]
    result = subprocess.run(["git", "check-ignore", "--no-index", *paths], cwd=ROOT, check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == paths
