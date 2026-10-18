"""Benchmark isolation and disk-backed approximate near-duplicate grouping."""

from __future__ import annotations

import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from .common import digest, read_jsonl


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, dict)):
        for item in value.values() if isinstance(value, dict) else value:
            yield from strings(item)


class BenchmarkIndex:
    """Exact matches block; verified shared spans quarantine pending review.

    Short answer-only overlaps are counted, not treated as proof of leakage.
    Spaced benchmark anchors are queried with all candidate windows. This is
    approximate recall: short offset spans and semantic paraphrases can be missed.
    """

    def __init__(self, inventory: dict, normalize):
        self.normalize = normalize
        self.exact = defaultdict(set)
        self.texts, self.refs = [], []
        self.anchors = defaultdict(set)
        self.short_answers = set()
        self.qa = set()
        self.counts = {}
        for name in ("c3", "fuxi", "fuxi_dev", "history_validation", "history_test"):
            count = 0
            for item in inventory["files"].get(name, []):
                before = count
                path = Path(item["path"])
                if path.suffix == ".jsonl":
                    rows = (row for _, row in read_jsonl(path))
                else:
                    value = json.loads(path.read_text(encoding="utf-8-sig"))
                    rows = self._rows(value)
                for number, row in enumerate(rows):
                    count += 1
                    reference = f"{name}:{path.name}:{number}"
                    if name == "c3":
                        if not isinstance(row.get("classical_Chinese"), str) or not row["classical_Chinese"].strip():
                            raise ValueError(f"{reference}: missing classical_Chinese")
                        fields = [row["classical_Chinese"], row.get("modern_Chinese", "")]
                    elif name.startswith("history_"):
                        if not isinstance(row.get("inputs"), str) or not isinstance(row.get("truth"), str):
                            raise ValueError(f"{reference}: missing inputs/truth")
                        fields = [row["inputs"], row["truth"]]
                    else:
                        if "input" not in row or "output" not in row:
                            raise ValueError(f"{reference}: missing input/output")
                        questions = list(strings(row["input"]))
                        answers = list(strings(row["output"]))
                        for q in questions:
                            for a in answers:
                                self.qa.add((normalize(q), normalize(a)))
                        fields = questions + [a for a in answers if len(normalize(a)) >= 8]
                        self.short_answers.update(normalize(a) for a in answers if 0 < len(normalize(a)) < 8)
                    if not any(isinstance(text, str) and normalize(text) for text in fields):
                        raise ValueError(f"{reference}: no usable exclusion text")
                    for text in fields:
                        self.add(text, reference)
                if count == before:
                    raise ValueError(f"{path}: no recognized benchmark records")
            if not count and name in inventory["files"]:
                raise ValueError(f"{name}: no recognized benchmark records")
            self.counts[name] = count

    @staticmethod
    def _rows(value):
        if isinstance(value, list):
            for item in value:
                yield from BenchmarkIndex._rows(item)
        elif isinstance(value, dict):
            if "input" in value or "classical_Chinese" in value:
                yield value
            else:
                for item in value.values():
                    yield from BenchmarkIndex._rows(item)

    def add(self, text, reference):
        if not isinstance(text, str):
            raise ValueError(f"{reference}: benchmark text must be a string")
        norm = self.normalize(text)
        if not norm:
            return
        self.exact[norm].add(reference)
        if len(norm) >= 24:
            index = len(self.texts)
            self.texts.append(norm)
            self.refs.append(reference)
            for offset in set(range(0, len(norm) - 23, 12)) | {len(norm) - 24}:
                self.anchors[norm[offset:offset + 24]].add(index)

    def check(self, record):
        evidence = set()
        question, answer = (self.normalize(m["content"]) for m in record["messages"])
        if (question, answer) in self.qa:
            return "exact", ["fuxi:question_answer"]
        for text in {record["ancient"], record["modern"], record["messages"][0]["content"],
                     record["messages"][1]["content"]}:
            norm = self.normalize(text)
            if not norm:
                continue
            # Short prose inputs are still protected; generic short answer labels are not.
            if norm in self.exact and (text == record["ancient"] or len(norm) >= 8):
                return "exact", sorted(self.exact[norm])
            matches = set()
            for offset in range(len(norm) - 23):
                matches.update(self.anchors.get(norm[offset:offset + 24], ()))
            for index in matches:
                target = self.texts[index]
                match = SequenceMatcher(None, norm, target, autojunk=False).find_longest_match()
                if match.size >= 24:
                    evidence.add(self.refs[index])
        if evidence:
            return "possible", sorted(evidence)
        if answer in self.short_answers:
            return "short_answer_only", []
        return "clear", []


class NearIndex:
    """LSH candidates are verified by character-shingle Jaccard, not assumed duplicates."""

    def __init__(self, db, threshold):
        self.db, self.threshold = db, threshold
        db.executescript("""
            CREATE TABLE IF NOT EXISTS near_texts (id TEXT PRIMARY KEY, text TEXT, doc TEXT);
            CREATE TABLE IF NOT EXISTS near_bands (band TEXT, id TEXT);
            CREATE INDEX IF NOT EXISTS near_lookup ON near_bands(band);
        """)

    def add(self, text, document):
        from datasketch import MinHash

        key = digest(text)
        found = self.db.execute("SELECT doc FROM near_texts WHERE id=?", (key,)).fetchone()
        if found:
            return key, [(key, found[0])]
        matches, bands = [], []
        if len(text) >= 16:
            shingles = {text[i:i + 5] for i in range(len(text) - 4)}
            mh = MinHash(num_perm=64, seed=42)
            mh.update_batch([s.encode() for s in sorted(shingles)])
            bands = [f"{i}:" + mh.hashvalues[i:i + 4].tobytes().hex() for i in range(0, 64, 4)]
            marks = ",".join("?" for _ in bands)
            candidates = self.db.execute(
                f"SELECT DISTINCT t.id,t.text,t.doc FROM near_texts t JOIN near_bands b ON t.id=b.id "
                f"WHERE b.band IN ({marks})", bands,
            ).fetchall()
            for other_id, other, doc in candidates:
                other_shingles = {other[i:i + 5] for i in range(len(other) - 4)}
                score = len(shingles & other_shingles) / len(shingles | other_shingles)
                if score >= self.threshold:
                    matches.append((other_id, doc))
        self.db.execute("INSERT INTO near_texts VALUES (?,?,?)", (key, text, document))
        self.db.executemany("INSERT INTO near_bands VALUES (?,?)", ((band, key) for band in bands))
        return key, matches


class CorpusLocator:
    """Recover book ownership by verified containment in a traceable corpus passage.

    Spaced 24-character anchors avoid indexing every substring of entire books.
    Unmatched short sentences remain unknown instead of guessing their source.
    """

    def __init__(self, path, normalize):
        from .sources import book_name

        self.passages, self.anchors = [], defaultdict(set)
        for _, row in read_jsonl(path):
            if row.get("char") or not isinstance(row.get("content"), str) or not row.get("source"):
                continue
            text = normalize(row["content"])
            book = book_name(str(row["source"]))
            index = len(self.passages)
            self.passages.append((text, book, row.get("era", "unknown")))
            for offset in range(0, len(text) - 23, 24):
                self.anchors[text[offset:offset + 24]].add(index)

    def find(self, text):
        candidates = set()
        for offset in range(len(text) - 23):
            candidates.update(self.anchors.get(text[offset:offset + 24], ()))
        return sorted({(book, era) for index in candidates for passage, book, era in [self.passages[index]]
                       if text in passage})
