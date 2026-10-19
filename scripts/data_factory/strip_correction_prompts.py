"""Remove known CSC/CGEC user-message wrappers, retaining targets and chat roles."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSC_PREFIX = "任务: 纠错文本\n输入: "
CSC_SUFFIX = "\n输出: "
CGEC_PREFIX = "纠正下列句子的语法错误，保留原意；如果没有错误则原样返回。只输出纠正后的句子：\n"


def unwrap(text, task):
    if task == "csc" and text.startswith(CSC_PREFIX):
        if not text.endswith(CSC_SUFFIX):
            raise ValueError("CSC instruction prefix without its expected suffix")
        source = text[len(CSC_PREFIX):-len(CSC_SUFFIX)]
    elif task == "cgec" and text.startswith(CGEC_PREFIX):
        source = text[len(CGEC_PREFIX):]
    else:
        return text
    if not source:
        raise ValueError("removing the wrapper would leave an empty sentence")
    return source


def convert_file(path, task, apply=False):
    path = Path(path)
    original_stat = path.stat()
    backup = path.with_name(path.name + ".with-prompts.bak")
    report = {"path": str(path), "task": task, "rows": 0, "changed": 0}
    temporary = None
    writer = None
    try:
        if apply:
            writer = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                                 prefix=f".{path.name}.", suffix=".tmp", delete=False)
            temporary = Path(writer.name)
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                messages = row.get("messages", [])
                if (len(messages) != 2 or [item.get("role") for item in messages] != ["user", "assistant"]
                        or any(not isinstance(item.get("content"), str) for item in messages)):
                    raise ValueError(f"expected a user/assistant text pair: {path}:{line_number}")
                text = messages[0]["content"]
                source = unwrap(text, task)
                report["rows"] += 1
                report["changed"] += source != text
                messages[0]["content"] = source
                if writer:
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        if writer:
            writer.close()
        if apply and report["changed"]:
            if backup.exists():
                raise FileExistsError(f"backup already exists; refusing a second rewrite: {backup}")
            current_stat = path.stat()
            if ((current_stat.st_ino, current_stat.st_size, current_stat.st_mtime_ns)
                    != (original_stat.st_ino, original_stat.st_size, original_stat.st_mtime_ns)):
                raise RuntimeError(f"file changed during conversion: {path}")
            with path.open("rb") as source, backup.open("xb") as target:
                shutil.copyfileobj(source, target)
            shutil.copymode(path, temporary)
            temporary.replace(path)
            report["backup"] = str(backup)
        report["applied"] = apply and bool(report["changed"])
        return report
    finally:
        if writer:
            writer.close()
        if temporary and temporary.exists():
            temporary.unlink()  # Only the temporary file created by this invocation.


def target_files(data_dir):
    for task in ("csc", "cgec"):
        root = Path(data_dir) / task / "sft"
        # CSC's current final train/dev files live directly in sft/.
        paths = sorted((root / "final").rglob("*.jsonl"))
        if task == "csc":
            paths = sorted(set(paths) | set(root.glob("*.jsonl")))
        if not paths:
            raise FileNotFoundError(f"no {task} final JSONL files found under {root}")
        for path in paths:
            yield task, path


def update_cgec_report(data_dir):
    path = Path(data_dir) / "cgec/sft/final/prepare_report.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    report["prompt"] = ""
    report["input_format"] = "source_only_messages"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default="/share/project/wuhaiming/spaces/CC-LLM/data/downstream")
    parser.add_argument("--apply", action="store_true", help="back up then atomically replace JSONL files; otherwise preview")
    args = parser.parse_args()
    targets = list(target_files(args.data_dir))
    # Check all four datasets before publishing any changes.
    preview = [convert_file(path, task) for task, path in targets]
    if args.apply:
        for item in preview:
            backup = Path(item["path"] + ".with-prompts.bak")
            if item["changed"] and backup.exists():
                raise FileExistsError(f"backup already exists: {backup}")
        reports = [convert_file(path, task, apply=True) for task, path in targets]
        update_cgec_report(args.data_dir)
    else:
        reports = preview
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
