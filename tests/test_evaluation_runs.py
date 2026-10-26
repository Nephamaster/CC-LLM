from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from scripts.evaluation.runs import latest_run_directory, new_run_directory


def test_versions_preserve_legacy_results_and_reserve_concurrently(tmp_path):
    (tmp_path / "results.json").write_text("old result")
    assert new_run_directory(tmp_path).name == "v0"
    (tmp_path / "v0/manifest.json").write_text("incomplete")
    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(lambda _: new_run_directory(tmp_path), range(16)))
    assert {p.name for p in outputs} == {f"v{i}" for i in range(1, 17)}
    assert latest_run_directory(tmp_path).name == "v16"
    assert (tmp_path / "results.json").read_text() == "old result"
    assert (tmp_path / "v0/manifest.json").read_text() == "incomplete"


def test_downstream_stages_continue_selected_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    common = root / "scripts/evaluation/common.sh"
    task = root / "scripts/evaluation/csc"
    command = 'source "$1"; configure_evaluation "$2" csc; printf "%s\\n" "$OUT"'
    env = {k: v for k, v in os.environ.items() if k not in {"OUT", "RUN_DIR", "STAGE"}}
    env.update(OUT=str(tmp_path), PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")

    def resolve(stage, **extra):
        result = subprocess.run(["bash", "-e", "-c", command, "_", str(common), str(task)],
                                env={**env, "STAGE": stage, **extra}, capture_output=True, text=True, check=True)
        return Path(result.stdout.strip())

    first = resolve("prepare")
    assert first == tmp_path / "v0"
    assert resolve("infer") == first
    assert resolve("score") == first
    second = resolve("all")
    assert second == tmp_path / "v1"
    assert resolve("judge") == second
    assert resolve("all", RUN_DIR=str(first)) == first
