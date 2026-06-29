import argparse
import json
import os

from typing import Any

from transformers import AutoTokenizer


def bytes_to_unicode() -> dict[int, str]:
    """
    Byte-level unicode map used by GPT-style BPE tokenizers.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(161, 172 + 1))
        + list(range(174, 255 + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_byte_decoder(tokenizer) -> dict[str, int]:
    if hasattr(tokenizer, "byte_decoder"):
        return tokenizer.byte_decoder
    return {v: k for k, v in bytes_to_unicode().items()}


def get_byte_encoder(tokenizer) -> dict[int, str]:
    return {v: k for k, v in get_byte_decoder(tokenizer).items()}


def decode_bpe_piece(piece: str, byte_decoder: dict[str, int]) -> str:
    try:
        return bytearray([byte_decoder[c] for c in piece]).decode("utf-8", errors="replace")
    except Exception:
        # Fallback for non-bytelevel or already-decoded tokens.
        return piece


def is_chinese_char(c: str) -> bool:
    if len(c) != 1:
        return False
    cp = ord(c)
    return (
        0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0x2CEB0 <= cp <= 0x2EBEF
        or 0x30000 <= cp <= 0x3134F
    )


def is_chinese_string(s: str) -> bool:
    return bool(s) and all(is_chinese_char(c) for c in s)


def normalize_vocab(vocab_like: Any) -> dict[str, int]:
    if isinstance(vocab_like, dict):
        return {str(k): int(v) for k, v in vocab_like.items()}
    if isinstance(vocab_like, list):
        normalized: dict[str, int] = {}
        for i, item in enumerate(vocab_like):
            if isinstance(item, list):
                token = item[0]
            else:
                token = item
            normalized[str(token)] = i
        return normalized
    raise ValueError("Unsupported vocab format")


def normalize_merges(merges_like: Any) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    for item in merges_like or []:
        if isinstance(item, str):
            a, b = item.split(" ", 1)
            merges.append((a, b))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            merges.append((str(item[0]), str(item[1])))
        else:
            raise ValueError(f"Unsupported merge item: {item}")
    return merges


def extract_bpe_state(tokenizer):
    """
    Return (tokenizer_json, vocab, merges), where tokenizer_json can be None on legacy tokenizers.
    """
    if hasattr(tokenizer, "backend_tokenizer"):
        tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())
        model = tokenizer_json.get("model", {})
        if model.get("type") != "BPE":
            raise ValueError(f"Only BPE tokenizers are supported, got: {model.get('type')}")
        vocab = normalize_vocab(model.get("vocab", {}))
        merges = normalize_merges(model.get("merges", []))
        return tokenizer_json, vocab, merges

    if hasattr(tokenizer, "encoder") and hasattr(tokenizer, "bpe_ranks"):
        vocab = {str(k): int(v) for k, v in tokenizer.encoder.items()}
        merges_with_rank = sorted(tokenizer.bpe_ranks.items(), key=lambda x: x[1])
        merges = [(str(a), str(b)) for (a, b), _rank in merges_with_rank]
        return None, vocab, merges

    raise ValueError("Cannot extract BPE state from tokenizer")


def build_reindexed_vocab_and_mapping(vocab: dict[str, int], removed_token_ids: set[int]) -> tuple[dict[str, int], dict[int, int]]:
    """
    Reindex retained tokens to contiguous ids [0, new_vocab_size), preserving old id order.
    Returns:
      - new_vocab: token -> new_id
      - new2old: new_id -> old_id
    """
    kept = sorted(
        ((token, old_id) for token, old_id in vocab.items() if old_id not in removed_token_ids),
        key=lambda x: x[1],
    )
    new_vocab: dict[str, int] = {}
    new2old: dict[int, int] = {}
    for new_id, (token, old_id) in enumerate(kept):
        new_vocab[token] = new_id
        new2old[new_id] = old_id
    return new_vocab, new2old


def collect_single_chinese_dependencies(
    vocab: dict[str, int], merges: list[tuple[str, str]], byte_decoder: dict[str, int]
) -> tuple[set[str], set[int]]:
    """
    Protect every merge needed to build single Han-character tokens.
    Byte-level BPE often needs incomplete UTF-8 byte fragments as intermediate
    pieces before it can form one complete Han character.
    """
    merge_by_result = {a + b: (a, b, merge_idx) for merge_idx, (a, b) in enumerate(merges)}
    protected_tokens: set[str] = set()
    protected_merge_idx: set[int] = set()

    def protect(token: str):
        if token in protected_tokens:
            return
        protected_tokens.add(token)

        merge = merge_by_result.get(token)
        if merge is None:
            return

        a, b, merge_idx = merge
        protected_merge_idx.add(merge_idx)
        protect(a)
        protect(b)

    for token in vocab:
        decoded = decode_bpe_piece(token, byte_decoder)
        if len(decoded) == 1 and is_chinese_string(decoded):
            protect(token)

    return protected_tokens, protected_merge_idx


def filter_merges_by_vocab(merges: list[tuple[str, str]], vocab: dict[str, int]) -> list[tuple[str, str]]:
    """
    Hugging Face tokenizers validates that every token referenced by a merge is
    present in the BPE vocab. After pruning vocab entries, remove dangling merges.
    """
    vocab_tokens = set(vocab)
    filtered_merges: list[tuple[str, str]] = []
    dangling_count = 0
    for a, b in merges:
        if a in vocab_tokens and b in vocab_tokens and (a + b) in vocab_tokens:
            filtered_merges.append((a, b))
        else:
            dangling_count += 1
    print(dangling_count, "dangling merges removed")
    return filtered_merges


def encode_text_as_bpe_piece(text: str, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[b] for b in text.encode("utf-8"))


def ensure_single_chinese_bpe_paths(
    old_tokenizer,
    old_vocab: dict[str, int],
    pruned_vocab: dict[str, int],
    new2old: dict[int, int],
    pruned_merges: list[tuple[str, str]],
    candidate_chars: set[str],
    byte_encoder: dict[int, str],
) -> tuple[list[tuple[str, str]], dict[int, list[int]]]:
    """
    Make each candidate Han character reachable as one byte-level BPE token.
    Required character-completion merges are promoted to the front so they beat
    cross-character byte merges such as "æµ" + "ĭè¯ķ".
    """
    existing_merges = set(pruned_merges)
    promoted_merges: list[tuple[str, str]] = []
    promoted_merge_set: set[tuple[str, str]] = set()
    init_token_ids: dict[int, list[int]] = {}
    added_token_count = 0

    def add_vocab_token(token: str, old_ids: list[int]) -> None:
        nonlocal added_token_count
        if token in pruned_vocab:
            return
        new_id = len(pruned_vocab)
        pruned_vocab[token] = new_id
        added_token_count += 1
        if token in old_vocab:
            new2old[new_id] = int(old_vocab[token])
        else:
            init_token_ids[new_id] = [int(old_id) for old_id in old_ids]

    for char in sorted(candidate_chars):
        piece = encode_text_as_bpe_piece(char, byte_encoder)
        if len(piece) <= 1:
            continue

        old_char_ids = [int(old_id) for old_id in old_tokenizer.encode(char, add_special_tokens=False)]
        for byte_piece in piece:
            if byte_piece not in pruned_vocab:
                add_vocab_token(byte_piece, [int(old_vocab[byte_piece])])

        acc = piece[0]
        for part in piece[1:]:
            merged = acc + part
            add_vocab_token(merged, old_char_ids)
            merge = (acc, part)
            if merge not in promoted_merge_set:
                promoted_merges.append(merge)
                promoted_merge_set.add(merge)
            acc = merged

    pruned_merges = promoted_merges + [
        merge for merge in pruned_merges if merge not in promoted_merge_set and merge in existing_merges
    ]
    print(added_token_count, "single chinese bpe tokens added")
    print(len(promoted_merges), "single chinese bpe merges promoted")
    return pruned_merges, init_token_ids


def write_init_token_ids(new_model_path: str, init_token_ids: dict[int, list[int]]) -> None:
    if not init_token_ids:
        return
    with open(os.path.join(new_model_path, "new_token_init_token_ids.json"), "wt", encoding="utf-8") as f:
        json.dump(init_token_ids, f, ensure_ascii=False, indent=2)
    print(len(init_token_ids), "new token embedding init entries written")


def prune(old_model_path: str, new_model_path: str, prune_vocab_tokens: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(old_model_path)
    tokenizer_json, vocab, merges = extract_bpe_state(tokenizer)
    byte_decoder = get_byte_decoder(tokenizer)
    byte_encoder = get_byte_encoder(tokenizer)
    protected_tokens, protected_merge_idx = collect_single_chinese_dependencies(vocab, merges, byte_decoder)

    need_to_delete_words = set()
    need_to_delete_word_ids = set()
    for token, token_id in vocab.items():
        decoded = decode_bpe_piece(token, byte_decoder)
        if len(decoded) > 1 and is_chinese_string(decoded) and token not in protected_tokens:
            need_to_delete_words.add(decoded)
            need_to_delete_word_ids.add(token_id)
            print(f"DELETE WORD {decoded}")
    print(len(need_to_delete_word_ids), "chinese words to delete")

    need_to_delete_merge_idx = set()
    for merge_idx, (a, b) in enumerate(merges):
        if merge_idx in protected_merge_idx:
            continue

        merged = a + b
        dab = decode_bpe_piece(merged, byte_decoder)
        if len(dab) > 1 and is_chinese_string(dab):
            need_to_delete_merge_idx.add(merge_idx)
            print(f"DELETE MERGE {dab}")
    print(len(need_to_delete_merge_idx), "merges to delete")

    if prune_vocab_tokens:
        pruned_vocab, new2old = build_reindexed_vocab_and_mapping(vocab, need_to_delete_word_ids)
    else:
        pruned_vocab = vocab
        new2old = {old_id: old_id for old_id in sorted(vocab.values())}
    pruned_merges = [merge for i, merge in enumerate(merges) if i not in need_to_delete_merge_idx]
    pruned_merges = filter_merges_by_vocab(pruned_merges, pruned_vocab)

    init_token_ids: dict[int, list[int]] = {}
    candidate_chars = {char for word in need_to_delete_words for char in word if is_chinese_char(char)}
    if prune_vocab_tokens:
        pruned_merges, init_token_ids = ensure_single_chinese_bpe_paths(
            tokenizer,
            vocab,
            pruned_vocab,
            new2old,
            pruned_merges,
            candidate_chars,
            byte_encoder,
        )
        pruned_merges = filter_merges_by_vocab(pruned_merges, pruned_vocab)

    os.makedirs(new_model_path, exist_ok=True)
    tokenizer.save_pretrained(new_model_path)

    with open(os.path.join(new_model_path, "vocab.json"), "wt", encoding="utf-8") as f:
        json.dump(pruned_vocab, f, ensure_ascii=False)
    with open(os.path.join(new_model_path, "merges.txt"), "wt", encoding="utf-8") as f:
        f.write("\n".join(f"{a} {b}" for a, b in pruned_merges))
        f.write("\n")
    with open(os.path.join(new_model_path, "new2old_token_id.json"), "wt", encoding="utf-8") as f:
        json.dump(new2old, f, ensure_ascii=False, indent=2)
    write_init_token_ids(new_model_path, init_token_ids)

    # New Transformers loads tokenizer.json first when present, so patch it too.
    if tokenizer_json is not None:
        tokenizer_json["model"]["vocab"] = pruned_vocab
        tokenizer_json["model"]["merges"] = [f"{a} {b}" for a, b in pruned_merges]
        with open(os.path.join(new_model_path, "tokenizer.json"), "wt", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, ensure_ascii=False)


def main(old_model_path: str, new_model_path: str):
    text = "\u8fd9\u662f\u4e00\u4e2a\u4e2d\u6587\u5206\u8bcd\u6d4b\u8bd5"
    tokenizer_old = AutoTokenizer.from_pretrained(old_model_path)
    old_ids = tokenizer_old.encode(text, add_special_tokens=False)
    print(tokenizer_old.convert_ids_to_tokens(old_ids))
    print([tokenizer_old.decode([_id]) for _id in old_ids])
    print(tokenizer_old.decode(old_ids))
    print(len(tokenizer_old))

    tokenizer_new = AutoTokenizer.from_pretrained(new_model_path)
    new_ids = tokenizer_new.encode(text, add_special_tokens=False)
    print(tokenizer_new.convert_ids_to_tokens(new_ids))
    print([tokenizer_new.decode([_id]) for _id in new_ids])
    print(tokenizer_new.decode(new_ids))
    print(len(tokenizer_new))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_model_path", type=str, required=True)
    parser.add_argument("--new_model_path", type=str, required=True)

    args = parser.parse_args()
    prune(args.old_model_path, args.new_model_path, prune_vocab_tokens=True)
    main(args.old_model_path, args.new_model_path)
