"""Chapter-aware holdouts without transitively merging books via shared quotations."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from functools import lru_cache

from .common import TASKS, Normalizer, digest, read_jsonl, resolve, write_line
from .sources import book_name


class ChapterResolver:
    def __init__(self, db, corpus, normalize):
        self.db, self.normalize = db, normalize
        self.origins = {}
        db.execute("CREATE VIRTUAL TABLE chapter_search USING fts5(body, tokenize='trigram')")
        for number, row in read_jsonl(corpus):
            if row.get("char") or not isinstance(row.get("content"), str) or not row.get("source"):
                continue
            book = book_name(row["source"])
            unit = row.get("volume") or row.get("chapter") or row.get("id")
            group = "chapter:" + digest(normalize(book) + "\0" + str(unit)) if unit is not None else "book:" + normalize(book)
            self.origins[number] = {"group": group, "book": book, "book_key": normalize(book),
                                    "era": row.get("era", "unknown"), "corpus_id": row.get("id"),
                                    "resolution": "corpus_chapter" if unit is not None else "book_fallback"}
            db.execute("INSERT INTO chapter_search(rowid,body) VALUES (?,?)", (number, normalize(row["content"])))
        db.commit()
        self.corpus_books = {o["book_key"] for o in self.origins.values()}

    @lru_cache(maxsize=16384)
    def locate(self, text):
        if len(text) < 3:
            return ()
        # A short exact anchor avoids expensive FTS positional joins for 200-character passages.
        # instr still verifies containment of the entire normalized input.
        query = '"' + text[:16].replace('"', '""') + '"'
        rows = self.db.execute("SELECT rowid FROM chapter_search WHERE chapter_search MATCH ? "
                               "AND instr(body,?)>0", (query, text))
        return tuple(sorted((self.origins[row[0]] for row in rows), key=lambda o: (o["group"], str(o["corpus_id"]))))

    def assign(self, record):
        normalized = self.normalize(record["ancient"])
        book = record["book"]
        original = record["metadata"].get("source_label") or ""
        label = self.normalize(original)
        key = self.normalize(book)
        origins = self.locate(normalized) if record["task"] != "dictionary" else ()
        if len({o["group"] for o in origins}) > 16:
            return False  # Highly ambiguous source ownership stays out of both partitions.
        if record["task"] == "dictionary":
            groups = [self.normalize(record["document_group"])]
            resolution = "dictionary_entry"
        elif origins:
            if record["source_id"] == "historytrans" and len({o["book_key"] for o in origins}) != 1:
                return False
            groups = sorted({o["group"] for o in origins})
            preferred = [o for o in origins if o["book_key"] == key]
            owner = (preferred or list(origins))[0]
            groups.remove(owner["group"])
            groups.insert(0, owner["group"])
            if book == "unknown":
                book, key = owner["book"], owner["book_key"]
                record["metadata"]["era"] = owner["era"]
            resolution = owner["resolution"]
            record["metadata"]["group_recovery"] = "corpus_containment"
            if original and label != self.normalize(book):
                declared = "label:" + digest(label)
                groups.append(declared)
                if not preferred:
                    groups.remove(declared)
                    groups.insert(0, declared)
                    resolution = "source_chapter"
            elif book != "unknown" and not preferred and record["source_id"] != "historytrans":
                groups.insert(0, "book:" + key)
                resolution = "book_fallback"
        elif label and label != key:
            groups = ["label:" + digest(label)]
            resolution = "source_chapter"
        elif book != "unknown":
            groups = ["book:" + key]
            resolution = "book_fallback"
        else:
            return False
        record.update(book=book, document_group=groups[0], document_groups=groups, book_key=key,
                      group_resolution=resolution,
                      origin_books=sorted({key} | {o["book_key"] for o in origins}),
                      comparison_length=len(normalized))
        # Unmatched source labels in a corpus book are not proven to use the same chapter scheme.
        record["uncertain_chapter_book"] = resolution == "source_chapter" and key in self.corpus_books
        record["document_groups"] = list(dict.fromkeys(record["document_groups"]))
        return True


class BoundaryIndex:
    """Protect all held-out candidates, not just the final sampled Dev rows."""
    def __init__(self, records, normalize):
        import ahocorasick

        self.normalize = normalize
        self.automaton = ahocorasick.Automaton()
        self.prefixes, self.spans = defaultdict(list), set()
        self.groups, self.clusters, self.modern = set(), set(), set()
        self.books, self.fallback_books, self.uncertain_books = set(), set(), set()
        texts = set()
        for r in records:
            self.groups.update(r["document_groups"])
            self.clusters.add(r["near_group"])
            self.books.update(r["origin_books"])
            if r["group_resolution"] == "book_fallback":
                self.fallback_books.update(r["origin_books"])
            if r["uncertain_chapter_book"]:
                self.uncertain_books.update(r["origin_books"])
            if r["task"] in {"c2m", "m2c"}:
                self.modern.add(normalize(r["modern"]))
            texts.add(normalize(r["ancient"]))
            if r["task"] in {"c2m", "m2c"}:
                texts.add(normalize(r["modern"]))
        for text in sorted(texts - {""}):
            self.automaton.add_word(text, True)
            for prefix in {text[i:i + 8] for i in range(len(text) - 7)}:
                self.prefixes[prefix].append(text)
            self.spans.update(text[i:i + 24] for i in range(len(text) - 23))
        self.has_text = bool(texts - {""})
        if self.has_text:
            self.automaton.make_automaton()

    def conflict(self, record):
        if self.groups.intersection(record["document_groups"]):
            return "chapter"
        books = set(record["origin_books"])
        if (books & self.fallback_books or books & self.uncertain_books
                or (record["group_resolution"] == "book_fallback" or record["uncertain_chapter_book"])
                and books & self.books):
            return "uncertain_book_boundary"
        if record["near_group"] in self.clusters:
            return "near_group"
        modern = self.normalize(record["modern"])
        if record["task"] in {"c2m", "m2c"} and modern in self.modern:
            return "parallel_translation"
        values = [self.normalize(record["ancient"])]
        if record["task"] in {"c2m", "m2c"}:
            values.append(modern)
        for text in values:
            if not text:
                continue
            if self.has_text and next(self.automaton.iter(text), None) is not None:
                return "dev_text_in_train"
            if len(text) >= 8 and any(text in target for target in self.prefixes.get(text[:8], ())):
                return "train_text_in_dev"
            if any(text[i:i + 24] in self.spans for i in range(len(text) - 23)):
                return "shared_24_char_span"
        return None


def choose_dev_groups(db, config):
    """Reserve small groups across books until capped task capacities cover Dev targets."""
    groups = {}
    total = Counter()
    for group, book, task, n, resolution in db.execute(
        "SELECT doc,book,task,COUNT(*),resolution FROM pool WHERE dev_eligible=1 GROUP BY doc,book,task,resolution"
    ):
        info = groups.setdefault(group, {"book": book, "counts": Counter(), "resolution": resolution})
        info["counts"][task] += n
        total[task] += n
    held, used, held_total = set(), defaultdict(Counter), Counter()
    caps = {t: max(1, math.ceil(config["dev_quotas"][t] * config["dev_max_book_fraction"])) for t in TASKS}
    targets = {t: math.ceil(config["dev_quotas"][t] * config["dev_pool_factor"]) for t in TASKS}
    tasks = sorted((t for t in TASKS if targets[t]), key=lambda t: sum(t in g["counts"] for g in groups.values()))
    for task in tasks:
        choices = defaultdict(list)
        for group, info in groups.items():
            if info["counts"][task]:
                choices[info["book"]].append(group)
        for options in choices.values():
            options.sort(key=lambda g: (groups[g]["resolution"] == "book_fallback",
                                       sum(groups[g]["counts"].values()), digest(f"{config['seed']}:{g}")))
        def sufficient():
            return (sum(min(n, caps[task]) for n in used[task].values()) >= targets[task]
                    and len([g for g in held if groups[g]["counts"][task]]) >= config["dev_min_groups"]
                    and len([b for b, n in used[task].items() if n]) >= config["dev_min_books"])
        while not sufficient():
            progress = False
            for book in sorted(choices, key=lambda b: (used[task][b], digest(f"{config['seed']}:{b}"))):
                if used[task][book] >= caps[task] and len([g for g in held if groups[g]["counts"][task]]) >= config["dev_min_groups"]:
                    continue
                options = choices[book]
                while options:
                    group = options.pop(0)
                    if group in held:
                        continue
                    counts = groups[group]["counts"]
                    if any(counts[t] and total[t] - held_total[t] - counts[t] < min(
                        config["quotas"][t], math.floor(total[t] * .8)
                    ) for t in TASKS[:3]):
                        continue
                    held.add(group)
                    held_total.update(counts)
                    for t, n in counts.items():
                        used[t][book] += n
                    progress = True
                    break
                if sufficient():
                    break
            if not progress:
                break
    return held, {"reserved_groups": sorted(held), "reserved_candidate_counts": dict(held_total),
                  "capped_capacity": {t: sum(min(n, caps[t]) for n in used[t].values()) for t in tasks}}


def partition_candidates(db, config, destination):
    normalize = Normalizer()
    resolver = ChapterResolver(db, resolve(config["sources"]["corpus"]), normalize)
    db.execute("CREATE TABLE pool (id TEXT PRIMARY KEY,task TEXT,source TEXT,book TEXT,doc TEXT,"
               "cluster TEXT,priority TEXT,payload TEXT,resolution TEXT,length_bin INTEGER,script TEXT,dev_eligible INTEGER)")
    stats = Counter()
    with (destination / "partition_audit.jsonl").open("w", encoding="utf-8") as audit:
        for number, (key, task, source, _, book, doc, cluster, priority, payload, _) in enumerate(db.execute("SELECT * FROM eligible"), 1):
            record = json.loads(payload)
            if not resolver.assign(record):
                stats["unresolved_or_ambiguous_chapter"] += 1
                write_line(audit, {"id": key, "reason": "unresolved_or_ambiguous_chapter"})
                continue
            db.execute("INSERT INTO pool VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                key, task, source, record["book_key"], record["document_group"], cluster,
                digest(f"{config['seed']}:{key}"), json.dumps(record, ensure_ascii=False),
                record["group_resolution"], record["comparison_length"] // 64,
                record["metadata"].get("script", "unknown"),
                int(not record["uncertain_chapter_book"]),
            ))
            stats["group_resolution/" + record["group_resolution"]] += 1
            if number % 10000 == 0:
                print(f"resolved chapters for {number} candidates", flush=True)
        db.commit()
    return partition_pool(db, config, destination, stats)


def partition_pool(db, config, destination, stats=None):
    stats = Counter() if stats is None else stats
    normalize = Normalizer()
    with (destination / "partition_audit.jsonl").open("a", encoding="utf-8") as audit:
        held, allocation = choose_dev_groups(db, config)
        db.execute("CREATE TABLE held_groups (doc TEXT PRIMARY KEY)")
        db.executemany("INSERT INTO held_groups VALUES (?)", ((g,) for g in sorted(held)))
        dev_records = (json.loads(row[0]) for row in db.execute("SELECT payload FROM pool WHERE doc IN (SELECT doc FROM held_groups)"))
        boundary = BoundaryIndex(dev_records, normalize)
        db.execute("DELETE FROM eligible")
        db.execute("CREATE TABLE sampling_strata(id TEXT PRIMARY KEY,length_bin INTEGER,script TEXT)")
        for key, task, source, book, doc, cluster, priority, payload, _, length_bin, script, _ in db.execute("SELECT * FROM pool"):
            record = json.loads(payload)
            split = "dev" if doc in held else "train"
            conflict = boundary.conflict(record) if split == "train" else None
            if conflict:
                stats["train_dev_excluded/" + conflict] += 1
                write_line(audit, {"id": key, "reason": conflict})
                continue
            record["split"] = split
            db.execute("INSERT INTO eligible VALUES (?,?,?,?,?,?,?,?,?,?)", (
                key, task, source, split, book, doc, cluster, priority, json.dumps(record, ensure_ascii=False),
                digest(normalize(record["messages"][0]["content"])),
            ))
            db.execute("INSERT INTO sampling_strata VALUES (?,?,?)", (key, length_bin, script))
            stats[f"eligible/{split}/{task}"] += 1
    db.commit()
    return {"allocation": allocation, "statistics": dict(stats), "strategy": "chapter_stratified_holdout_v2"}
