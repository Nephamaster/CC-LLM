"""BPE tokenizer state helpers used by semantic vocabulary construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def bytes_to_unicode() -> dict[int, str]:
    """Byte-level unicode map used by GPT-style BPE tokenizers."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(161, 172 + 1))
        + list(range(174, 255 + 1))
    )
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(codepoint) for codepoint in cs)))


def normalize_vocab(vocab_like: Any) -> dict[str, int]:
    if isinstance(vocab_like, dict):
        return {str(token): int(token_id) for token, token_id in vocab_like.items()}
    if isinstance(vocab_like, list):
        vocab: dict[str, int] = {}
        for token_id, item in enumerate(vocab_like):
            token = item[0] if isinstance(item, list) else item
            vocab[str(token)] = token_id
        return vocab
    raise ValueError(f"Unsupported vocab format: {type(vocab_like)!r}")


def normalize_merges(merges_like: Any) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    for item in merges_like or []:
        if isinstance(item, str):
            left, right = item.split(" ", 1)
            merges.append((left, right))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            merges.append((str(item[0]), str(item[1])))
        else:
            raise ValueError(f"Unsupported merge item: {item!r}")
    return merges


def get_byte_decoder(tokenizer: Any) -> dict[str, int]:
    if hasattr(tokenizer, "byte_decoder"):
        return {str(token): int(byte) for token, byte in tokenizer.byte_decoder.items()}
    return {piece: byte for byte, piece in bytes_to_unicode().items()}


def get_byte_encoder(tokenizer: Any) -> dict[int, str]:
    return {byte: piece for piece, byte in get_byte_decoder(tokenizer).items()}


def decode_bpe_piece(piece: str, byte_decoder: dict[str, int]) -> str:
    """Decode one byte-level BPE token or merged piece to Unicode text."""
    try:
        return bytearray(byte_decoder[char] for char in piece).decode("utf-8", errors="replace")
    except Exception:
        return piece


def encode_text_as_bpe_piece(text: str, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[byte] for byte in text.encode("utf-8"))


def extract_bpe_state(tokenizer: Any) -> tuple[dict | None, dict[str, int], list[tuple[str, str]]]:
    """Extract tokenizer JSON, vocab and merges from a Hugging Face BPE tokenizer."""
    if hasattr(tokenizer, "backend_tokenizer"):
        tokenizer_json = json.loads(tokenizer.backend_tokenizer.to_str())
        model = tokenizer_json.get("model", {})
        if model.get("type") != "BPE":
            raise ValueError(f"Only BPE tokenizers are supported, got: {model.get('type')}")
        return tokenizer_json, normalize_vocab(model.get("vocab", {})), normalize_merges(model.get("merges", []))

    if hasattr(tokenizer, "encoder") and hasattr(tokenizer, "bpe_ranks"):
        vocab = {str(token): int(token_id) for token, token_id in tokenizer.encoder.items()}
        ranked_merges = sorted(tokenizer.bpe_ranks.items(), key=lambda item: item[1])
        merges = [(str(left), str(right)) for (left, right), _rank in ranked_merges]
        return None, vocab, merges

    raise ValueError(f"Cannot extract BPE state from tokenizer type: {type(tokenizer)!r}")


def get_special_token_strings(tokenizer: Any) -> set[str]:
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])
    special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    for value in special_tokens_map.values():
        if isinstance(value, str):
            special_tokens.add(value)
        elif isinstance(value, (list, tuple)):
            special_tokens.update(str(item) for item in value)
    return special_tokens


@dataclass
class BpeState:
    tokenizer_json: dict | None
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    byte_decoder: dict[str, int]
    byte_encoder: dict[int, str]
    special_tokens: set[str]

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "BpeState":
        tokenizer_json, vocab, merges = extract_bpe_state(tokenizer)
        return cls(
            tokenizer_json=tokenizer_json,
            vocab=vocab,
            merges=merges,
            byte_decoder=get_byte_decoder(tokenizer),
            byte_encoder=get_byte_encoder(tokenizer),
            special_tokens=get_special_token_strings(tokenizer),
        )

    def decode_piece(self, piece: str) -> str:
        return decode_bpe_piece(piece, self.byte_decoder)

    def encode_text_piece(self, text: str) -> str:
        return encode_text_as_bpe_piece(text, self.byte_encoder)

    def patched_tokenizer_json(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> dict | None:
        if self.tokenizer_json is None:
            return None
        tokenizer_json = json.loads(json.dumps(self.tokenizer_json, ensure_ascii=False))
        tokenizer_json["model"]["vocab"] = vocab
        tokenizer_json["model"]["merges"] = [f"{left} {right}" for left, right in merges]
        return tokenizer_json


def write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def write_merges(path: Path, merges: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="\n") as f:
        for left, right in merges:
            f.write(f"{left} {right}\n")
