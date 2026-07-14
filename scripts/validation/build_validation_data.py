"""Build the 500-example Phase 0 diagnostic dataset."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from scripts.validation.common import (
    category_counts,
    file_sha256,
    is_cjk_hanzi,
    read_json,
    read_jsonl,
    utc_now_iso,
    write_json,
    write_jsonl,
)


TOKENIZER_QUOTAS = {
    "modern_simplified": 30,
    "multi_hanzi": 30,
    "traditional": 20,
    "rare_hanzi": 20,
    "mixed_zh_en": 30,
    "number_symbol_math": 20,
    "code_json_markdown": 30,
    "special_chat": 20,
}

FORWARD_QUOTAS = {
    "natural_zh": 120,
    "natural_en": 105,
    "repo_markdown": 25,
    "repo_code": 25,
    "repo_json": 15,
    "math_latex": 10,
}

MATH_TEXTS = [
    r"欧拉公式写作 $e^{i\pi}+1=0$，它联系了五个基本常数。",
    r"若 $f(x)=x^2+2x+1$，则 $f'(x)=2x+2$。",
    r"矩阵乘法满足 $C_{ij}=\sum_k A_{ik}B_{kj}$。",
    r"交叉熵为 $L=-\sum_i y_i\log p_i$。",
    r"注意力可表示为 $\operatorname{softmax}(QK^T/\sqrt{d})V$。",
    r"For $x\in\mathbb{R}$, the inequality $e^x\geq 1+x$ holds.",
    r"The gradient is $\nabla_\theta L(\theta)$ and the update is $\theta_{t+1}=\theta_t-\eta g_t$.",
    r"A Gaussian density is $p(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/(2\sigma^2)}$.",
    r"设序列满足 $a_{n+1}=2a_n+1$，并且 $a_0=0$。",
    r"代码复杂度从 $O(n^2)$ 优化为 $O(n\log n)$，memory 为 $O(n)$。",
]

TOKENIZER_MATH_TEXTS = [
    "1 + 1 = 2，2 × 3 = 6。",
    "x∈[0,1]，且 x²≤x。",
    "A∪B、A∩B、A⊆B、A≠B。",
    "角度 θ=45°，sin(θ)=√2/2。",
    "概率 P(A|B)=P(A∩B)/P(B)。",
    "向量 v=(1,2,3)，范数 ||v||=√14。",
    "极限 lim(x→0) sin(x)/x = 1。",
    "积分 ∫₀¹ x²dx = 1/3。",
    "序列 aₙ=2ⁿ，n∈ℕ。",
    "损失 L=-Σᵢ yᵢlog(pᵢ)。",
]

CHAT_CASES = [
    ([{"role": "user", "content": "请解释什么是字符级语言模型。"}], True),
    (
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "用中文回答：What is PGCA?"},
        ],
        True,
    ),
    ([{"role": "user", "content": "计算 17 + 25。"}, {"role": "assistant", "content": "17 + 25 = 42。"}], False),
    ([{"role": "user", "content": "Return JSON with keys name and value."}], True),
    ([{"role": "user", "content": "代码如下：\n```python\nprint('你好')\n```"}], True),
    ([{"role": "system", "content": "回答必须准确。"}, {"role": "user", "content": "繁體字『學習』如何转换？"}], True),
]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(char for char in text if char in "\n\t" or ord(char) >= 0x20)
    return text.strip("\n")


def candidate(text: str, source_kind: str, source_path: str | Path, **metadata: Any) -> dict[str, Any] | None:
    text = normalize_text(text)
    if not text.strip():
        return None
    value: dict[str, Any] = {
        "text": text,
        "source": {"kind": source_kind, "path": str(source_path)},
    }
    value.update(metadata)
    return value


def read_hanzi(path: Path) -> list[str]:
    chars: list[str] = []
    seen: set[str] = set()
    with path.open("rt", encoding="utf-8") as file:
        for line in file:
            for char in line.strip():
                if is_cjk_hanzi(char) and char not in seen:
                    chars.append(char)
                    seen.add(char)
    return chars


def char_chunk_candidates(
    chars: list[str],
    count: int,
    width: int,
    path: Path,
    source_kind: str,
) -> list[dict[str, Any]]:
    required = count * width
    if len(chars) < required:
        raise ValueError(f"{path} provides {len(chars)} unique Hanzi; {required} are required")
    return [candidate("".join(chars[index * width : (index + 1) * width]), source_kind, path) for index in range(count)]


def choose_unique(
    candidates: Iterable[dict[str, Any] | None],
    count: int,
    rng: random.Random,
    category: str,
    excluded_texts: set[str],
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if item is not None and item["text"] not in excluded_texts:
            unique.setdefault(item["text"], item)
    values = list(unique.values())
    rng.shuffle(values)
    if len(values) < count:
        raise ValueError(f"category {category!r} has {len(values)} unique candidates; {count} are required")
    return values[:count]


def select_quotas(
    pools: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    rng: random.Random,
    excluded_texts: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    used_texts = set(excluded_texts or ())
    for category, count in quotas.items():
        values = choose_unique(pools[category], count, rng, category, used_texts)
        selected[category] = values
        used_texts.update(item["text"] for item in values)
    return selected


def _excluded(path: Path) -> bool:
    parts = path.parts
    excluded_roots = {("data", "validation"), ("resources", "raw")}
    return any(part in {".git", "__pycache__"} for part in parts) or parts[:2] in excluded_roots


def collect_repository_candidates(root: Path) -> dict[str, list[dict[str, Any]]]:
    pools = {"markdown": [], "code": [], "json": [], "mixed": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _excluded(path.relative_to(root)) or path.stat().st_size > 1_000_000:
            continue
        suffix = path.suffix.lower()
        if suffix not in {".md", ".py", ".json"}:
            continue
        relative = path.relative_to(root)
        if "models" in relative.parts and (
            suffix != ".json" or path.stat().st_size > 100_000 or "feature_vocabs" in relative.parts
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if suffix == ".md":
            blocks = re.split(r"\n\s*\n", text)
            for block in blocks:
                item = candidate(block, "repository_markdown", relative)
                if item is not None and 20 <= len(item["text"]) <= 800:
                    pools["markdown"].append(item)
                    if any(is_cjk_hanzi(char) for char in item["text"]) and re.search(r"[A-Za-z]", item["text"]):
                        pools["mixed"].append(item)
        elif suffix == ".py":
            lines = text.splitlines()
            for start in range(0, len(lines), 8):
                block = "\n".join(lines[start : start + 12]).strip("\n")
                item = candidate(block, "repository_code", relative, line_start=start + 1)
                if item is not None and 30 <= len(item["text"]) <= 1000:
                    pools["code"].append(item)
                    if any(is_cjk_hanzi(char) for char in item["text"]) and re.search(r"[A-Za-z]", item["text"]):
                        pools["mixed"].append(item)
        else:
            lines = text.splitlines()
            for start in range(0, len(lines), 10):
                block = "\n".join(lines[start : start + 15]).strip("\n")
                item = candidate(block, "repository_json", relative, line_start=start + 1)
                if item is not None and 20 <= len(item["text"]) <= 1000:
                    pools["json"].append(item)
    return pools


def load_manual_candidates(path: Path) -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        item = candidate(row["text"], "manual_boundary", path, source_id=row.get("id"), source_type=row.get("type"))
        pools.setdefault(str(row.get("type", "unknown")), []).append(item)
    return pools


def tokenizer_special_candidates(model_path: Path) -> list[dict[str, Any]]:
    config_path = model_path / "tokenizer_config.json"
    config = read_json(config_path)
    tokens: list[str] = []
    for key in ("bos_token", "eos_token", "pad_token"):
        value = config.get(key)
        if isinstance(value, str):
            tokens.append(value)
    tokens.extend(token for token in config.get("extra_special_tokens", []) if isinstance(token, str))
    tokens = list(dict.fromkeys(tokens))
    values = [candidate(token, "tokenizer_config", config_path, special_token=token) for token in tokens]
    for index, (messages, add_generation_prompt) in enumerate(CHAT_CASES):
        values.append(
            candidate(
                "\n".join(str(message["content"]) for message in messages),
                "manual_chat",
                "built_in",
                messages=messages,
                add_generation_prompt=add_generation_prompt,
                chat_case=index,
            )
        )
    return values


def build_tokenizer_rows(
    args: argparse.Namespace,
    rng: random.Random,
    repo: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    manual = load_manual_candidates(args.manual_boundary)
    simplified = read_hanzi(args.tghz)
    traditional = read_hanzi(args.traditional)
    all_hanzi = read_hanzi(args.hanzi_set)
    common = set(simplified) | set(traditional)
    rare = [char for char in all_hanzi if char not in common]

    multi_types = ("multi_hanzi_word", "idiom", "homophone", "glyph_similar", "polyphone")
    multi = [item for name in multi_types for item in manual.get(name, [])]
    multi.extend(
        candidate(text, "manual_extension", "built_in")
        for text in (
            "中华人民共和国中央人民政府",
            "人工智能推动自然语言处理发展",
            "大语言模型需要字符级语义表示",
            "北京大学和清华大学开展联合研究",
            "长江经济带高质量发展战略",
        )
    )
    mixed = manual.get("mixed", []) + manual.get("unicode", []) + repo["mixed"]
    number_symbol = manual.get("number_symbol", []) + [
        candidate(text, "manual_math_boundary", "built_in") for text in TOKENIZER_MATH_TEXTS
    ]
    number_symbol.extend(
        candidate(text, "manual_symbol", "built_in")
        for text in (
            "价格为 ¥123.45，折扣 8.5%。",
            "集合 A∩B≠∅，且 A⊆B。",
            "版本 v2.1.0+cuda12.4",
            "坐标为 (31.2304°N, 121.4737°E)。",
            "emoji: 😀🚀✅；日文：こんにちは；韩文：안녕하세요。",
        )
    )
    code = manual.get("code", []) + manual.get("json", []) + manual.get("markdown", [])
    code += repo["code"] + repo["json"] + repo["markdown"]

    pools = {
        "modern_simplified": char_chunk_candidates(simplified, 30, 24, args.tghz, "hanzi_resource"),
        "multi_hanzi": multi,
        "traditional": char_chunk_candidates(traditional, 20, 16, args.traditional, "hanzi_resource"),
        "rare_hanzi": manual.get("rare_hanzi", [])
        + char_chunk_candidates(rare, 20, 8, args.hanzi_set, "hanzi_resource"),
        "mixed_zh_en": mixed,
        "number_symbol_math": number_symbol,
        "code_json_markdown": code,
        "special_chat": tokenizer_special_candidates(args.char_model_path),
    }
    selected = select_quotas(pools, TOKENIZER_QUOTAS, rng)
    return materialize_rows("tokenizer", selected)


def split_passages(text: str, min_chars: int, max_chars: int) -> list[str]:
    passages: list[str] = []
    for paragraph in re.split(r"\n\s*\n", normalize_text(text)):
        paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
        while len(paragraph) > max_chars:
            cut = max(
                paragraph.rfind(marker, min_chars, max_chars + 1)
                for marker in ("。", "！", "？", ". ", "! ", "? ", "; ", "；")
            )
            if cut < min_chars:
                cut = max_chars
            else:
                cut += 1
            passages.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if len(paragraph) >= min_chars:
            passages.append(paragraph)
    return passages


def collect_cci(path: Path, max_records: int, min_chars: int, max_chars: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"CCI source does not exist: {path}")
    values: list[dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index >= max_records:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for passage_index, passage in enumerate(split_passages(str(row.get("text", "")), min_chars, max_chars)[:2]):
                values.append(candidate(passage, "cci3_hq", path, source_id=row.get("id"), passage_index=passage_index))
    return values


def collect_fineweb(
    path: Path,
    max_records: int,
    min_chars: int,
    max_chars: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"FineWeb-Edu source does not exist: {path}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required to sample FineWeb-Edu parquet data") from error

    parquet_file = parquet.ParquetFile(path)
    row_groups = list(range(parquet_file.num_row_groups))
    rng.shuffle(row_groups)
    values: list[dict[str, Any]] = []
    records = 0
    for row_group in row_groups:
        table = parquet_file.read_row_group(row_group, columns=["text", "id"])
        columns = table.to_pydict()
        ids = columns.get("id", [None] * len(columns["text"]))
        for row_index, (text, source_id) in enumerate(zip(columns["text"], ids, strict=True)):
            if records >= max_records:
                return values
            records += 1
            for passage_index, passage in enumerate(split_passages(str(text or ""), min_chars, max_chars)[:2]):
                values.append(
                    candidate(
                        passage,
                        "fineweb_edu",
                        path,
                        source_id=source_id,
                        row_group=row_group,
                        row_index=row_index,
                        passage_index=passage_index,
                    )
                )
    return values


def build_forward_rows(
    args: argparse.Namespace,
    rng: random.Random,
    repo: dict[str, list[dict[str, Any]]],
    excluded_texts: set[str],
) -> list[dict[str, Any]]:
    pools = {
        "natural_zh": collect_cci(args.cci_path, args.max_source_records, args.min_chars, args.max_chars),
        "natural_en": collect_fineweb(
            args.fineweb_path,
            args.max_source_records,
            args.min_chars,
            args.max_chars,
            rng,
        ),
        "repo_markdown": [item for item in repo["markdown"] if len(item["text"]) >= args.min_chars],
        "repo_code": [item for item in repo["code"] if len(item["text"]) >= args.min_chars],
        "repo_json": [item for item in repo["json"] if len(item["text"]) >= args.min_chars],
        "math_latex": [candidate(text, "manual_math", "built_in") for text in MATH_TEXTS],
    }
    selected = select_quotas(pools, FORWARD_QUOTAS, rng, excluded_texts)
    return materialize_rows("forward", selected)


def materialize_rows(split: str, selected: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, values in selected.items():
        for index, value in enumerate(values):
            rows.append({"id": f"{split}-{category}-{index:03d}", "split": split, "category": category, **value})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation"))
    parser.add_argument("--char-model-path", type=Path, default=Path("models/Qwen3-1.7B-Base-Char"))
    parser.add_argument("--manual-boundary", type=Path, default=Path("data/validation/manual_boundary.jsonl"))
    parser.add_argument("--hanzi-set", type=Path, default=Path("resources/hanzi/hanzi_set.txt"))
    parser.add_argument("--tghz", type=Path, default=Path("resources/hanzi/tghz2013.txt"))
    parser.add_argument("--traditional", type=Path, default=Path("resources/hanzi/common_traditional.txt"))
    parser.add_argument(
        "--cci-path",
        type=Path,
        default=Path("/share/project/wuhaiming/data/dataset/CCI3-HQ/data/part_000000.jsonl"),
    )
    parser.add_argument(
        "--fineweb-path",
        type=Path,
        default=Path("/share/project/wuhaiming/data/dataset/fineweb-edu/sample/10BT/000_00000.parquet"),
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-source-records", type=int, default=20_000)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    repo = collect_repository_candidates(args.repo_root.resolve())
    tokenizer_rows = build_tokenizer_rows(args, rng, repo)
    forward_rows = build_forward_rows(args, rng, repo, {row["text"] for row in tokenizer_rows})

    tokenizer_path = args.output_dir / "tokenizer_validation.jsonl"
    forward_path = args.output_dir / "forward_validation.jsonl"
    write_jsonl(tokenizer_path, tokenizer_rows)
    write_jsonl(forward_path, forward_rows)
    manifest = {
        "generated_at": utc_now_iso(),
        "seed": args.seed,
        "total_records": len(tokenizer_rows) + len(forward_rows),
        "unique_texts": len({row["text"] for row in tokenizer_rows + forward_rows}),
        "source_paths": {"cci3_hq": str(args.cci_path), "fineweb_edu": str(args.fineweb_path)},
        "source_kinds": dict(
            sorted(Counter(row["source"]["kind"] for row in tokenizer_rows + forward_rows).items())
        ),
        "files": {
            str(tokenizer_path): {
                "records": len(tokenizer_rows),
                "categories": category_counts(tokenizer_rows),
                "sha256": file_sha256(tokenizer_path),
            },
            str(forward_path): {
                "records": len(forward_rows),
                "categories": category_counts(forward_rows),
                "sha256": file_sha256(forward_path),
            },
        },
    }
    write_json(args.output_dir / "validation_data_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
