"""Adapters preserve source text; no generated answers or inferred author labels."""

from __future__ import annotations

import itertools
import json
from collections import Counter

from .common import clean, digest, read_jsonl, resolve, unpunctuated

PROMPTS = {
    "c2m": "将下列古文翻译成现代汉语，只输出译文：\n",
    "m2c": "将下列现代汉语译成古文，只输出译文：\n",
    "punctuate": "为下列古文添加标点，不增删或改写原有文字，只输出标点后的文本：\n",
}


def book_name(source: str) -> str:
    return source.replace("‧", "·").split("·", 1)[0].strip()


def corpus_catalog(config: dict):
    catalog = {}
    for _, row in read_jsonl(resolve(config["sources"]["corpus"])):
        book = book_name(str(row.get("source", "")))
        entry = catalog.setdefault(book, {"era": set(), "author": set(), "category": set()})
        for key in entry:
            if row.get(key):
                entry[key].add(str(row[key]))
    return catalog


def candidate(task, source_id, raw_id, ancient, modern, book, group, metadata, user=None, answer=None):
    if task in PROMPTS:
        source_text = modern if task == "m2c" else ancient
        user = PROMPTS[task] + source_text
        answer = ancient if task == "m2c" else modern
    record = {
        "task": task, "source_id": source_id, "source_record_id": str(raw_id),
        "ancient": ancient, "modern": modern, "book": book,
        "document_group": group, "metadata": metadata,
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
        "review_required": task in {"dictionary", "provenance"},
    }
    record["id"] = digest(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return record


def iter_candidates(config: dict, rejected: Counter, limit: int | None = None):
    catalog = corpus_catalog(config)
    for name in ("translate", "punctuate", "history_train", "corpus"):
        rows = read_jsonl(resolve(config["sources"][name]))
        if limit is not None:
            rows = itertools.islice(rows, limit)
        for number, raw in rows:
            rejected[f"read/{name}"] += 1
            try:
                if raw.get("_has_box"):
                    raise ValueError("source_reports_missing_character")
                if name == "corpus":
                    yield from corpus_candidates(raw, number)
                    continue
                if name == "history_train":
                    task, ancient, modern = "c2m", clean(raw["inputs"]), clean(raw["truth"])
                    source_id, book, group = "historytrans", "unknown", ""
                else:
                    task = raw["task"]
                    if (name == "translate" and task not in {"c2m", "m2c"}) or (
                        name == "punctuate" and task != "punctuate"
                    ):
                        raise ValueError("unexpected_task")
                    text, target = clean(raw["input"]), clean(raw["output"])
                    ancient, modern = (target, text) if task == "m2c" else (text, target)
                    source_id = "gujilab"
                    book = book_name(clean(raw["source"]))
                    # A book-level fallback prevents incompatible chapter labels leaking across splits.
                    group = "book:" + book
                if task == "punctuate":
                    if any(c in ancient for c in "，。！？；：、"):
                        raise ValueError("punctuation_input_not_raw")
                    if unpunctuated(ancient) != unpunctuated(modern):
                        raise ValueError("punctuation_alignment")
                    if unpunctuated(modern) == modern:
                        raise ValueError("no_punctuation_label")
                    # Whitespace is layout, not a supervision target. Preserve every other character.
                    ancient = "".join(ancient.split())
                    modern = "".join(modern.split())
                    bounds = config["lengths"]["punctuation"]
                else:
                    bounds = config["lengths"]["translation"]
                    if ancient == modern or not 0.2 <= len(modern) / max(len(ancient), 1) <= 5:
                        raise ValueError("translation_alignment_heuristic")
                if not bounds[0] <= len(unpunctuated(ancient)) <= bounds[1]:
                    raise ValueError("character_length")
                entry = catalog.get(book, {})
                metadata = {
                    "source_label": raw.get("source"), "category": raw.get("category", "unknown"),
                    "era": next(iter(entry["era"])) if len(entry.get("era", [])) == 1 else "unknown",
                    "quality": "rule_filtered_not_human_verified",
                    "source_location": f"{config['sources'][name]}:{number}",
                }
                yield candidate(task, source_id, raw.get("id", number), ancient, modern, book, group, metadata)
            except (KeyError, ValueError, TypeError) as exc:
                rejected[f"filter/{name}/{exc}"] += 1


def corpus_candidates(raw: dict, number: int):
    book, text = clean(raw["source"]), clean(raw["content"])
    metadata = {k: raw.get(k, "unknown") for k in ("era", "author", "category", "chapter", "volume")}
    metadata.update({"quality": "requires_source_review", "source_record": raw})
    if raw.get("char"):
        head, radical = clean(raw["char"]), clean(raw["radical"])
        if len(head) != 1 or len(text) > 600:
            raise ValueError("dictionary_entry")
        answer = f"部首：{radical}\n古代原释：{text}"
        yield candidate(
            "dictionary", "gujilab", raw.get("id", number), text, "", book,
            f"entry:{book}:{head}", metadata,
            user=f"摘录《{book}》中“{head}”字的部首与古代原释，不作现代释义改写。", answer=answer,
        )
    elif 30 <= len(text) <= 600:
        # Only full short passages are used. Never invent excerpt-to-chapter attribution.
        yield candidate(
            "provenance", "gujilab", raw.get("id", number), text, "", book,
            "book:" + book_name(book), metadata,
            user="指出下列古文出自哪部典籍，只输出书名：\n" + text, answer=book,
        )
