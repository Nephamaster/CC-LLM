"""Task generation ceilings shared by HF and vLLM downstream inference."""

import os

C3_BUDGETS = {"classification": 32, "retrieval": 128, "ner": 256, "punctuation": 512, "translation": 1024}


def token_budget(row, ceiling):
    if row["dataset"] == "c3":
        name = row["task"]
        default = C3_BUDGETS[name]
    elif row["dataset"] == "fuxi":
        name = "choice" if row.get("metric") == "acc" else "generation"
        default = 32 if name == "choice" else 1024
    else:
        return ceiling
    budget = int(os.getenv(name.upper() + "_MAX_NEW_TOKENS", default))
    if budget < 1:
        raise ValueError(f"{name} generation budget must be positive")
    return min(budget, ceiling)
