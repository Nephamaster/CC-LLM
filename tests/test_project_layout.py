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
                "RESUME_FROM_CHECKPOINT", "DATASET", "SPLIT", "FEWSHOT", "STAGE"):
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


def test_phase1_and_phase2_use_canonical_names_and_same_python(tmp_path, recorder):
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
                      DATA_DIR=data, CAPTURE=capture, CUDA_VISIBLE_DEVICES="0,1")
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
    for args in training:
        assert option(args, "--external_plugins") == str(CPT / "plugin.py")
        assert option(args, "--logging_dir") == str(Path(option(args, "--output_dir")) / "logs")
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
