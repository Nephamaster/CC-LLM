"""Extract licensed Stack Exchange, Wikimedia, and OpenAlex dumps to standard JSONL."""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import re
import sqlite3
import xml.etree.ElementTree as element_tree
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, TextIO

from scripts.data_factory.collect_external import MainTextParser, _chunks, _doc_id


@contextmanager
def open_text(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            yield file
    elif path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8") as file:
            yield file
    else:
        with path.open("rt", encoding="utf-8") as file:
            yield file


def html_text(value: str) -> str:
    parser = MainTextParser()
    parser.feed(value)
    return parser.text()


def stackexchange_license(created_at: str) -> str:
    date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
    if date < datetime(2011, 4, 8).date():
        return "CC-BY-SA-2.5"
    if date < datetime(2018, 5, 2).date():
        return "CC-BY-SA-3.0"
    return "CC-BY-SA-4.0"


def _stack_rows(path: Path) -> Iterator[dict[str, str]]:
    source = bz2.open(path, "rb") if path.suffix == ".bz2" else path.open("rb")
    try:
        for _, element in element_tree.iterparse(source, events=("end",)):
            if element.tag == "row":
                yield dict(element.attrib)
                element.clear()
    finally:
        source.close()


def extract_stackexchange(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted_db = args.output.with_suffix(".accepted.sqlite")
    accepted_db.unlink(missing_ok=True)
    connection = sqlite3.connect(accepted_db)
    connection.execute("CREATE TABLE accepted(id TEXT PRIMARY KEY)")
    records = 0
    with args.output.open("wt", encoding="utf-8", newline="\n") as output:
        for row in _stack_rows(args.input):
            if row.get("PostTypeId") != "1":
                continue
            accepted = row.get("AcceptedAnswerId")
            if accepted:
                connection.execute("INSERT OR IGNORE INTO accepted VALUES (?)", (accepted,))
            if int(row.get("Score", "0")) < args.min_question_score and not accepted:
                continue
            text = "\n\n".join(filter(None, (row.get("Title", "").strip(), html_text(row.get("Body", "")))))
            records += _write_stack_row(output, row, text, args, "question")
        connection.commit()

        for row in _stack_rows(args.input):
            if row.get("PostTypeId") != "2":
                continue
            accepted = connection.execute("SELECT 1 FROM accepted WHERE id = ?", (row.get("Id"),)).fetchone()
            if accepted is None and int(row.get("Score", "0")) < args.min_answer_score:
                continue
            records += _write_stack_row(output, row, html_text(row.get("Body", "")), args, "answer")
    connection.close()
    accepted_db.unlink(missing_ok=True)
    return records


def _write_stack_row(output: TextIO, row: dict[str, str], text: str, args: argparse.Namespace, post_type: str) -> int:
    count = 0
    created_at = row.get("CreationDate", "2018-05-02T00:00:00")
    category = "supplemental"
    for chunk_index, chunk in enumerate(_chunks(text, args.max_chars)):
        value = {
            "text": chunk,
            "source": "stackexchange",
            "repo": args.site,
            "doc_id": _doc_id("stackexchange", args.site, args.revision, row.get("Id", ""), chunk_index),
            "license": stackexchange_license(created_at),
            "url": f"https://{args.site}/q/{row.get('ParentId') or row.get('Id')}",
            "path": row.get("Id"),
            "revision": args.revision,
            "category": category,
            "quota_group": args.quota_group,
            "post_type": post_type,
            "score": int(row.get("Score", "0")),
            "created_at": created_at,
            "owner_user_id": row.get("OwnerUserId"),
        }
        output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        count += 1
    return count


WIKI_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
WIKI_REF_RE = re.compile(r"<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/>", re.IGNORECASE | re.DOTALL)
WIKI_LINK_RE = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")
WIKI_EXTERNAL_RE = re.compile(r"\[https?://\S+\s+([^\]]+)\]")


def clean_wikitext(text: str) -> str:
    text = WIKI_REF_RE.sub("", text)
    for _ in range(3):
        text = WIKI_TEMPLATE_RE.sub("", text)
    text = WIKI_LINK_RE.sub(r"\1", text)
    text = WIKI_EXTERNAL_RE.sub(r"\1", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"'{2,5}", "", text)
    return text.strip()


def extract_wikimedia(args: argparse.Namespace) -> int:
    source = bz2.open(args.input, "rb") if args.input.suffix == ".bz2" else args.input.open("rb")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with source, args.output.open("wt", encoding="utf-8", newline="\n") as output:
        for _, page in element_tree.iterparse(source, events=("end",)):
            if not page.tag.endswith("page"):
                continue
            title = page.findtext("{*}title") or ""
            namespace = page.findtext("{*}ns")
            redirect = page.find("{*}redirect")
            text = page.findtext("{*}revision/{*}text") or ""
            if namespace != "0" or redirect is not None or not text:
                page.clear()
                continue
            cleaned = clean_wikitext(text)
            for chunk_index, chunk in enumerate(_chunks(f"{title}\n\n{cleaned}", args.max_chars)):
                row = {
                    "text": chunk,
                    "source": "wikimedia",
                    "repo": args.project,
                    "doc_id": _doc_id("wikimedia", args.project, args.revision, title, chunk_index),
                    "license": "CC-BY-SA-4.0",
                    "url": f"https://{args.project}/wiki/{title.replace(' ', '_')}",
                    "path": title,
                    "revision": args.revision,
                    "category": args.category,
                    "quota_group": args.quota_group,
                }
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                records += 1
            page.clear()
    return records


def openalex_abstract(index: Any) -> str:
    if not isinstance(index, dict) or not index:
        return ""
    largest = max((position for positions in index.values() for position in positions), default=-1)
    words = [""] * (largest + 1)
    for word, positions in index.items():
        for position in positions:
            if 0 <= int(position) < len(words):
                words[int(position)] = str(word)
    return " ".join(word for word in words if word)


def extract_openalex(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with open_text(args.input) as source, args.output.open("wt", encoding="utf-8", newline="\n") as output:
        for line in source:
            if not line.strip():
                continue
            work = json.loads(line)
            title = str(work.get("title") or work.get("display_name") or "").strip()
            abstract = openalex_abstract(work.get("abstract_inverted_index"))
            text = "\n\n".join(filter(None, (title, abstract)))
            if not text:
                continue
            work_id = str(work.get("id", ""))
            row = {
                "text": text,
                "source": "openalex",
                "repo": "openalex",
                "doc_id": _doc_id("openalex", "openalex", args.revision, work_id, 0),
                "license": "CC0-1.0",
                "url": work_id,
                "path": work_id.rsplit("/", 1)[-1],
                "revision": args.revision,
                "category": "mixed_zh_en",
                "quota_group": "wikimedia_openalex",
            }
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            records += 1
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    stack = subparsers.add_parser("stackexchange")
    stack.add_argument("--input", type=Path, required=True)
    stack.add_argument("--output", type=Path, required=True)
    stack.add_argument("--site", required=True)
    stack.add_argument("--revision", required=True)
    stack.add_argument("--quota-group", choices=("code", "structured", "math"), required=True)
    stack.add_argument("--min-question-score", type=int, default=0)
    stack.add_argument("--min-answer-score", type=int, default=5)
    stack.add_argument("--max-chars", type=int, default=16_000)

    wiki = subparsers.add_parser("wikimedia")
    wiki.add_argument("--input", type=Path, required=True)
    wiki.add_argument("--output", type=Path, required=True)
    wiki.add_argument("--project", default="zh.wikipedia.org")
    wiki.add_argument("--revision", required=True)
    wiki.add_argument("--category", choices=("mixed_zh_en", "supplemental"), required=True)
    wiki.add_argument("--quota-group", choices=("wikimedia_openalex", "math", "hanzi"), required=True)
    wiki.add_argument("--max-chars", type=int, default=16_000)

    openalex = subparsers.add_parser("openalex")
    openalex.add_argument("--input", type=Path, required=True)
    openalex.add_argument("--output", type=Path, required=True)
    openalex.add_argument("--revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind == "stackexchange":
        records = extract_stackexchange(args)
    elif args.kind == "wikimedia":
        records = extract_wikimedia(args)
    else:
        records = extract_openalex(args)
    print(json.dumps({"output": str(args.output), "records": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

