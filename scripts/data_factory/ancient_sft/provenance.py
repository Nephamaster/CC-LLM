"""Verify exported Swift sources and the narrowly supported legacy preparation."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from .common import digest, file_hash

MODULE_DIR = Path(__file__).resolve().parent


def profile():
    return json.loads((MODULE_DIR / "provenance_profile.json").read_text(encoding="utf-8"))


def source_fingerprint(root: Path) -> dict:
    hashes = {path.relative_to(root).as_posix(): file_hash(path) for path in sorted((root / "swift").rglob("*.py"))}
    return {"source_sha256": digest(json.dumps(hashes, sort_keys=True)), "python_files": len(hashes)}


def git_output(root: Path, *args):
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def verify_swift(root: Path, expected_commit: str) -> dict:
    root = root.resolve()
    expected = profile()["swift"]
    if expected_commit != expected["commit"]:
        raise RuntimeError("no audited Swift source fingerprint for the configured swift_commit")
    # A package inside another checkout must not inherit the enclosing project's Git identity.
    top = git_output(root, "rev-parse", "--show-toplevel")
    is_checkout = top is not None and Path(top).resolve() == root
    commit = None
    if is_checkout:
        commit = git_output(root, "rev-parse", "HEAD")
        if commit != expected_commit:
            raise RuntimeError(f"ms-swift commit mismatch: {commit}; expected {expected_commit}")
        dirty = git_output(root, "status", "--porcelain", "--", "swift")
        if dirty is None or dirty:
            raise RuntimeError("ms-swift implementation is dirty or Git status could not be verified")
    actual = source_fingerprint(root)
    if any(actual[key] != expected[key] for key in actual):
        raise RuntimeError(
            f"ms-swift source fingerprint mismatch at {root}: {actual['source_sha256']}; "
            f"expected {expected['source_sha256']}. Restore the swift/ Python sources from "
            f"{expected_commit}; Git metadata is not required."
        )
    return {**actual, "git_commit": commit, "verified_reference_commit": expected_commit,
            "verification": "git_and_source_sha256" if is_checkout else "source_sha256_without_git"}


def preparation_fingerprint(root: Path) -> str:
    # Keep all original preparation code/imports, but omit the finalization-only functions.
    source = (root / "pipeline.py").read_text(encoding="utf-8")
    segments = [ast.get_source_segment(source, node) for node in ast.parse(source).body
                if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name in {"finalize", "distribution", "task_report"})]
    payload = {
        "files": {name: file_hash(root / name) for name in
                  ("__init__.py", "common.py", "overlap.py", "sources.py", "run.py")},
        "pipeline_prepare_source": "\n".join(segments),
    }
    return digest(json.dumps(payload, sort_keys=True))


def verify_prepared_implementation(recorded: str, current: str, root: Path = MODULE_DIR) -> str:
    if recorded == current:
        return "exact"
    from .relocation import is_verified_relocation

    if is_verified_relocation(recorded, root, profile()["relocation"]):
        return "verified_directory_relocation"
    legacy = profile()["legacy_prepare"]
    allowed_files = {"__init__.py", "common.py", "encoding.py", "overlap.py", "pipeline.py",
                     "run.py", "sources.py", "provenance.py"}
    if (recorded == legacy["implementation_sha256"]
            and {path.name for path in root.glob("*.py")} == allowed_files
            and preparation_fingerprint(root) == legacy["preparation_sha256"]):
        return "verified_legacy_prepare_encoder_provenance_fix"
    raise RuntimeError("configuration or implementation changed; prepare in a new output directory")
