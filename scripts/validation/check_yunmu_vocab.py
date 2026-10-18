import json
import re
import unicodedata
from pathlib import Path

root = Path(
    "/share/project/wuhaiming/spaces/CC-LLM/models/"
    "Qwen3-1.7B-Base-Char/features/feature_vocabs"
)
specials = {"<pad>", "<none>", "<unk>"}

def contains_hanzi(text):
    return any(
        ch == "〇"
        or unicodedata.name(ch, "").startswith(
            ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
        )
        for ch in text
    )

for name, pattern in [
    ("pinyin", r"[a-zv]+[0-4]"),
    ("yunmu", r"[a-zv]*"),
]:
    path = root / f"{name}_vocab.json"
    vocab = json.loads(path.read_text(encoding="utf-8"))
    ordinary = [s for s in vocab if s not in specials]

    hanzi_entries = [s for s in ordinary if contains_hanzi(s)]
    review_entries = [
        s for s in ordinary
        if not contains_hanzi(s) and re.fullmatch(pattern, s) is None
    ]

    print(f"\n{name}: {path}")
    print("词表总大小:", len(vocab))
    print("特殊 ID:", {s: vocab.get(s) for s in sorted(specials)})
    print("ID 连续且无重复:",
          sorted(vocab.values()) == list(range(len(vocab))))
    print("是否包含空串:", "" in vocab)
    print("含汉字词条数:", len(hanzi_entries))
    print("含汉字示例:", [(s, vocab[s]) for s in hanzi_entries[:30]])
    print("其他待复核词条数:", len(review_entries))
    print("待复核示例:", [(s, vocab[s]) for s in review_entries[:30]])