"""Official ChERRANT and NUS MaxMatch, with documented gold edit construction."""
import json
import re
import string
import subprocess
import sys
import unicodedata
from pathlib import Path

from .common import VENDOR, ensure_dir


def normalized(text, remove_punctuation=False):
    return "".join(c for c in text if not c.isspace() and not (
        remove_punctuation and (unicodedata.category(c).startswith("P") or c in string.punctuation)))


def cherrant_score(rows, output_dir, *, remove_punctuation=False):
    output_dir = ensure_dir(output_dir)
    reference, hypothesis = output_dir / "reference.para", output_dir / "hypothesis.para"
    for path, gold in ((reference, True), (hypothesis, False)):
        with path.open("w", encoding="utf-8") as handle:
            for i, row in enumerate(rows):
                source = normalized(row["src"], remove_punctuation)
                if not source:
                    raise ValueError("empty normalized source; cannot score without changing the sample")
                targets = row["references"] if gold else [row["predict_clean"]]
                values = [str(i), source] + [normalized(t, remove_punctuation) for t in targets]
                handle.write("\t".join(values) + "\n")
    scorer = VENDOR / "cherrant"
    for path in (reference, hypothesis):
        subprocess.run([sys.executable, str(scorer / "parallel_to_m2.py"), "-f", str(path.resolve()),
                        "-o", str(path.with_suffix(".m2").resolve()), "-g", "char", "-s", "all"],
                       cwd=scorer, check=True)
    result = subprocess.run([sys.executable, str(scorer / "compare_m2_for_evaluation.py"),
                             "-hyp", str(hypothesis.with_suffix(".m2").resolve()),
                             "-ref", str(reference.with_suffix(".m2").resolve())],
                            cwd=scorer, check=True, capture_output=True, text=True)
    (output_dir / "cherrant.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    matches = re.findall(r"(?m)^(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", result.stdout)
    if not matches:
        raise RuntimeError("cannot parse official ChERRANT report")
    tp, fp, fn, p, r, f = matches[-1]
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": float(p) * 100,
            "recall": float(r) * 100, "f0.5": float(f) * 100,
            "normalization": "remove whitespace; simplify targets (upstream default)",
            "punctuation_removed": remove_punctuation}


def reference_edits(source, target):
    """The pinned official edit_creator algorithm, applied separately to every reference."""
    from .vendor.m2scorer import levenshtein as m2

    graphs = [m2.edit_graph(*m2.levenshtein_matrix(source.split(), target.split(), 1, 1, cost)) for cost in (1, 2)]
    v1, e1, d1, a1 = graphs[0]
    v2, e2, d2, a2 = graphs[1]
    v, e, distance, edits = m2.merge_graph(v1, v2, e1, e2, d1, d2, a1, a2)
    v, e, distance, edits = m2.transitive_arcs(v, e, distance, edits, 0, False)
    weights = m2.set_weights(e, distance, edits, [], False, False)
    return [(start, end, old, [new]) for start, end, old, new in m2.best_edit_seq_bf(v, e, weights, edits, False)
            if old != new]


def maxmatch_score(rows, output_dir):
    import thulac
    from .vendor.m2scorer import levenshtein as m2

    segmenter = thulac.thulac(seg_only=True)
    cache = {}

    def segment(text):
        text = normalized(text)
        if text not in cache:
            cache[text] = segmenter.cut(text, text=True) if text else ""
        return cache[text]

    sources, hypotheses, all_edits = [], [], []
    output_dir = ensure_dir(output_dir)
    with (output_dir / "word.reference.m2").open("w", encoding="utf-8") as gold_file, (
        output_dir / "word.hypothesis.txt"
    ).open("w", encoding="utf-8") as hyp_file:
        for row in rows:
            source, hyp = segment(row["src"]), segment(row["predict_clean"])
            sources.append(source)
            hypotheses.append(hyp)
            edits = {i: reference_edits(source, segment(target)) for i, target in enumerate(row["references"])}
            all_edits.append(edits)
            gold_file.write("S " + source + "\n")
            for annotation, items in edits.items():
                if not items:
                    gold_file.write(f"A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||{annotation}\n")
                for start, end, _, targets in items:
                    gold_file.write(f"A {start} {end}|||UNK|||{targets[0]}|||REQUIRED|||-NONE-|||{annotation}\n")
            gold_file.write("\n")
            hyp_file.write(hyp + "\n")
    p, r, f = m2.batch_multi_pre_rec_f1(hypotheses, sources, all_edits, max_unchanged_words=2, beta=.5)
    return {"precision": p * 100, "recall": r * 100, "f0.5": f * 100,
            "segmenter": "THULAC seg_only=True", "gold_alignment": "NUS edit_creator, max_unchanged_words=0",
            "system_max_unchanged_words": 2}
