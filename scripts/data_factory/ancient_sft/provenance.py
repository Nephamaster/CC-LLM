"""Reference fingerprints for reusing previously admitted data candidates."""

import json
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent


def profile():
    return json.loads((MODULE_DIR / "provenance_profile.json").read_text(encoding="utf-8"))
