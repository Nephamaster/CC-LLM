"""Version checks do not depend on .git being shipped with Swift sources."""

import shutil

import pytest

from scripts.ancient_sft import provenance


@pytest.fixture
def swift_tree(tmp_path, monkeypatch):
    root = tmp_path / "export"
    (root / "swift").mkdir(parents=True)
    (root / "swift/__init__.py").write_text("__version__ = 'fixture'\n")
    (root / "swift/template.py").write_text("VALUE = 1\n")
    value = {"swift": {"commit": "pinned", **provenance.source_fingerprint(root)}}
    monkeypatch.setattr(provenance, "profile", lambda: value)
    return root


def test_source_export_without_git_passes(swift_tree, capsys):
    result = provenance.verify_swift(swift_tree, "pinned")
    assert result["verification"] == "source_sha256_without_git"
    assert result["git_commit"] is None
    assert result["verified_reference_commit"] == "pinned"
    assert "fatal:" not in capsys.readouterr().err


def test_unrelated_parent_checkout_does_not_supply_git_identity(swift_tree, monkeypatch):
    monkeypatch.setattr(provenance, "git_output", lambda root, *args: str(root.parent))
    assert provenance.verify_swift(swift_tree, "pinned")["git_commit"] is None


def test_git_binary_unavailable_falls_back_to_sources(swift_tree, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git unavailable")
    monkeypatch.setattr(provenance.subprocess, "run", unavailable)
    assert provenance.verify_swift(swift_tree, "pinned")["verification"] == "source_sha256_without_git"


@pytest.mark.parametrize("change", ["edit", "delete", "add"])
def test_source_drift_fails_closed(swift_tree, change):
    path = swift_tree / "swift/template.py"
    if change == "edit":
        path.write_text("VALUE = 2\n")
    elif change == "delete":
        path.unlink()
    else:
        (swift_tree / "swift/extra.py").write_text("VALUE = 3\n")
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        provenance.verify_swift(swift_tree, "pinned")


def test_unknown_reference_is_not_silently_accepted(swift_tree):
    with pytest.raises(RuntimeError, match="no audited"):
        provenance.verify_swift(swift_tree, "other")


@pytest.mark.parametrize("commit,dirty,error", [("other", "", "commit mismatch"),
                                               ("pinned", "M swift/template.py", "dirty")])
def test_git_guards_retained(swift_tree, monkeypatch, commit, dirty, error):
    def output(root, *args):
        if args == ("rev-parse", "--show-toplevel"):
            return str(root)
        if args == ("rev-parse", "HEAD"):
            return commit
        return dirty
    monkeypatch.setattr(provenance, "git_output", output)
    with pytest.raises(RuntimeError, match=error):
        provenance.verify_swift(swift_tree, "pinned")


def test_git_and_hash_both_verified(swift_tree, monkeypatch):
    def output(root, *args):
        return str(root) if args[-1] == "--show-toplevel" else "pinned" if args[-1] == "HEAD" else ""
    monkeypatch.setattr(provenance, "git_output", output)
    assert provenance.verify_swift(swift_tree, "pinned")["verification"] == "git_and_source_sha256"


def test_v2_does_not_finalize_legacy_splits_without_repartition():
    legacy = provenance.profile()["legacy_prepare"]
    assert provenance.preparation_fingerprint(provenance.MODULE_DIR) != legacy["preparation_sha256"]
    with pytest.raises(RuntimeError, match="implementation changed"):
        provenance.verify_prepared_implementation(legacy["implementation_sha256"], "new")


def test_unknown_legacy_implementation_rejected():
    with pytest.raises(RuntimeError, match="implementation changed"):
        provenance.verify_prepared_implementation("unknown", "new")


@pytest.mark.parametrize("filename", ["sources.py", "pipeline.py", "new_module.py"])
def test_legacy_exception_does_not_hide_preparation_changes(tmp_path, filename):
    root = tmp_path / "module"
    shutil.copytree(provenance.MODULE_DIR, root)
    path = root / filename
    # A new top-level statement changes the preparation contract even outside a function.
    with path.open("a") as handle:
        handle.write("\nUNREVIEWED_CHANGE = True\n")
    with pytest.raises(RuntimeError, match="implementation changed"):
        provenance.verify_prepared_implementation(provenance.profile()["legacy_prepare"]["implementation_sha256"], "new", root)
