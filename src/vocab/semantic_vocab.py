"""Semantic vocabulary construction for the character-level Chinese tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bpe_state import BpeState, write_json, write_merges
from .hanzi_set import read_hanzi_file
from .unicode_ranges import count_hanzi


@dataclass(frozen=True)
class SemanticVocabBuildConfig:
    base_tokenizer_path: str | Path = Path("/share/project/wuhaiming/data/models/Qwen3-1.7B-Base")
    output_dir: Path = Path("models/Qwen3-1.7B-Base-Char")
    hanzi_set_path: Path = Path("resources/hanzi/hanzi_set.txt")
    trust_remote_code: bool = True
    use_fast: bool = True
    remove_mixed_hanzi_tokens: bool = True
    write_tokenizer_files: bool = True


@dataclass
class SemanticVocabResult:
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    new2old_token_id: dict[int, int]
    new_token_init_token_ids: dict[int, list[int]]
    removed_multi_hanzi_tokens: list[dict[str, Any]]
    new_hanzi_token_ids: dict[int, str]
    manifest: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hanzi_set(path: Path) -> set[str]:
    chars, invalid = read_hanzi_file(path)
    if invalid:
        raise ValueError(f"Invalid non-Hanzi entries in {path}: {invalid}")
    return set(chars)


def is_kept_hanzi_token(decoded: str, hanzi_set: set[str]) -> bool:
    return len(decoded) == 1 and decoded in hanzi_set


def should_remove_token(
    token: str,
    decoded: str,
    state: BpeState,
    hanzi_set: set[str],
    protected_tokens: set[str],
    remove_mixed_hanzi_tokens: bool,
) -> bool:
    if token in state.special_tokens:
        return False
    hanzi_count = count_hanzi(decoded)
    if hanzi_count == 0:
        return False
    if is_kept_hanzi_token(decoded, hanzi_set):
        return False
    if token in protected_tokens:
        return False
    if remove_mixed_hanzi_tokens:
        return True
    return hanzi_count > 1 and len(decoded) == hanzi_count


def collect_single_hanzi_dependencies(
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
    state: BpeState,
    hanzi_set: set[str],
) -> tuple[set[str], set[int]]:
    """Protect BPE ancestors required to form existing target single-Hanzi tokens."""
    merge_by_result = {left + right: (left, right, index) for index, (left, right) in enumerate(merges)}
    protected_tokens: set[str] = set()
    protected_merge_ids: set[int] = set()

    def protect(piece: str) -> None:
        if piece in protected_tokens:
            return
        protected_tokens.add(piece)
        merge = merge_by_result.get(piece)
        if merge is None:
            return
        left, right, merge_id = merge
        protected_merge_ids.add(merge_id)
        protect(left)
        protect(right)

    for token in vocab:
        decoded = state.decode_piece(token)
        if is_kept_hanzi_token(decoded, hanzi_set):
            protect(token)

    return protected_tokens, protected_merge_ids


def build_reindexed_vocab_and_mapping(
    vocab: dict[str, int],
    removed_token_ids: set[int],
) -> tuple[dict[str, int], dict[int, int]]:
    kept = sorted(
        ((token, old_id) for token, old_id in vocab.items() if old_id not in removed_token_ids),
        key=lambda item: item[1],
    )
    new_vocab: dict[str, int] = {}
    new2old: dict[int, int] = {}
    for new_id, (token, old_id) in enumerate(kept):
        new_vocab[token] = new_id
        new2old[new_id] = old_id
    return new_vocab, new2old


def should_remove_merge(
    merged_decoded: str,
    protected: bool,
    hanzi_set: set[str],
    remove_mixed_hanzi_tokens: bool,
) -> bool:
    if protected:
        return False
    hanzi_count = count_hanzi(merged_decoded)
    if hanzi_count == 0:
        return False
    if is_kept_hanzi_token(merged_decoded, hanzi_set):
        return False
    if remove_mixed_hanzi_tokens:
        return True
    return hanzi_count > 1 and len(merged_decoded) == hanzi_count


def filter_merges_by_vocab(merges: list[tuple[str, str]], vocab: dict[str, int]) -> tuple[list[tuple[str, str]], int]:
    vocab_tokens = set(vocab)
    filtered: list[tuple[str, str]] = []
    dangling_count = 0
    for left, right in merges:
        if left in vocab_tokens and right in vocab_tokens and (left + right) in vocab_tokens:
            filtered.append((left, right))
        else:
            dangling_count += 1
    return filtered, dangling_count


def add_single_hanzi_bpe_paths(
    tokenizer: Any,
    state: BpeState,
    old_vocab: dict[str, int],
    new_vocab: dict[str, int],
    new2old_token_id: dict[int, int],
    merges: list[tuple[str, str]],
    hanzi_set: set[str],
) -> tuple[
    list[tuple[str, str]],
    dict[int, list[int]],
    dict[int, str],
    int,
    int,
]:
    existing_merges = set(merges)
    promoted_merges: list[tuple[str, str]] = []
    promoted_merge_set: set[tuple[str, str]] = set()
    init_token_ids: dict[int, list[int]] = {}
    new_hanzi_token_ids: dict[int, str] = {}
    added_token_count = 0

    def old_ids_for_text(text: str) -> list[int]:
        return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]

    def add_vocab_token(piece: str, init_old_ids: list[int]) -> None:
        nonlocal added_token_count
        if piece in new_vocab:
            return
        new_id = len(new_vocab)
        new_vocab[piece] = new_id
        added_token_count += 1
        if piece in old_vocab:
            new2old_token_id[new_id] = int(old_vocab[piece])
        else:
            init_token_ids[new_id] = [int(token_id) for token_id in init_old_ids]

    for char in sorted(hanzi_set, key=ord):
        piece = state.encode_text_piece(char)
        old_char_ids = old_ids_for_text(char)
        if not old_char_ids:
            raise ValueError(f"Base tokenizer produced no ids for Hanzi: {char}")

        for byte_piece in piece:
            if byte_piece not in new_vocab:
                init_old_ids = (
                    [int(old_vocab[byte_piece])]
                    if byte_piece in old_vocab
                    else old_char_ids
                )
                add_vocab_token(byte_piece, init_old_ids)

        acc = piece[0]
        for part in piece[1:]:
            merged = acc + part
            add_vocab_token(merged, old_char_ids)
            merge = (acc, part)
            if merge not in promoted_merge_set:
                promoted_merges.append(merge)
                promoted_merge_set.add(merge)
            acc = merged

        token_id = new_vocab[acc]
        if token_id in init_token_ids:
            new_hanzi_token_ids[token_id] = char

    patched_merges = promoted_merges + [
        merge for merge in merges if merge not in promoted_merge_set and merge in existing_merges
    ]
    return (
        patched_merges,
        init_token_ids,
        new_hanzi_token_ids,
        added_token_count,
        len(promoted_merges),
    )

def patch_added_tokens(tokenizer_json: dict, vocab: dict[str, int], new2old_token_id: dict[int, int]) -> int:
    """Keep added/special tokens contiguous with the new vocab and mapped to old ids."""
    added_count = 0
    added_tokens = tokenizer_json.get("added_tokens") or []
    for item in added_tokens:
        content = item.get("content")
        old_id = item.get("id")
        if not isinstance(content, str) or old_id is None:
            continue
        if content not in vocab:
            new_id = len(vocab)
            vocab[content] = new_id
            new2old_token_id[new_id] = int(old_id)
            added_count += 1
        item["id"] = int(vocab[content])
    tokenizer_json["model"]["vocab"] = vocab
    return added_count


class SemanticVocabBuilder:
    def __init__(self, config: SemanticVocabBuildConfig):
        self.config = config

    def build(self) -> SemanticVocabResult:
        tokenizer = self._load_tokenizer()
        state = BpeState.from_tokenizer(tokenizer)
        hanzi_set = load_hanzi_set(self.config.hanzi_set_path)
        protected_tokens, protected_merge_ids = collect_single_hanzi_dependencies(
            state.vocab, state.merges, state, hanzi_set
        )

        removed_token_ids: set[int] = set()
        removed_multi_hanzi_tokens: list[dict[str, Any]] = []
        removed_token_count_by_kind = {"mixed_hanzi": 0, "multi_hanzi": 0, "other_hanzi": 0}
        for token, token_id in state.vocab.items():
            decoded = state.decode_piece(token)
            if not should_remove_token(
                token=token,
                decoded=decoded,
                state=state,
                hanzi_set=hanzi_set,
                protected_tokens=protected_tokens,
                remove_mixed_hanzi_tokens=self.config.remove_mixed_hanzi_tokens,
            ):
                continue
            removed_token_ids.add(token_id)
            hanzi_count = count_hanzi(decoded)
            if hanzi_count > 0 and len(decoded) != hanzi_count:
                removed_token_count_by_kind["mixed_hanzi"] += 1
            elif hanzi_count > 1:
                removed_token_count_by_kind["multi_hanzi"] += 1
                removed_multi_hanzi_tokens.append(
                    {"token": decoded, "old_token_id": int(token_id)}
                )
            else:
                removed_token_count_by_kind["other_hanzi"] += 1

        removed_multi_hanzi_tokens.sort(
            key=lambda item: int(item["old_token_id"])
        )

        new_vocab, new2old_token_id = build_reindexed_vocab_and_mapping(state.vocab, removed_token_ids)

        removed_merge_ids: set[int] = set()
        for merge_id, (left, right) in enumerate(state.merges):
            decoded = state.decode_piece(left + right)
            if should_remove_merge(
                merged_decoded=decoded,
                protected=merge_id in protected_merge_ids,
                hanzi_set=hanzi_set,
                remove_mixed_hanzi_tokens=self.config.remove_mixed_hanzi_tokens,
            ):
                removed_merge_ids.add(merge_id)

        merges = [merge for index, merge in enumerate(state.merges) if index not in removed_merge_ids]
        merges, dangling_before_add = filter_merges_by_vocab(merges, new_vocab)
        (
            merges,
            init_token_ids,
            new_hanzi_token_ids,
            added_token_count,
            promoted_merge_count,
        ) = add_single_hanzi_bpe_paths(
            tokenizer=tokenizer,
            state=state,
            old_vocab=state.vocab,
            new_vocab=new_vocab,
            new2old_token_id=new2old_token_id,
            merges=merges,
            hanzi_set=hanzi_set,
        )
        merges, dangling_after_add = filter_merges_by_vocab(merges, new_vocab)

        manifest = {
            "generated_at": _utc_now_iso(),
            "base_tokenizer_path": str(self.config.base_tokenizer_path),
            "hanzi_set_path": str(self.config.hanzi_set_path),
            "hanzi_set_size": len(hanzi_set),
            "old_vocab_size": len(state.vocab),
            "new_vocab_size": len(new_vocab),
            "removed_token_count": len(removed_token_ids),
            "removed_token_count_by_kind": removed_token_count_by_kind,
            "removed_multi_hanzi_token_count": len(removed_multi_hanzi_tokens),
            "removed_merge_count": len(removed_merge_ids),
            "dangling_merge_count_before_single_hanzi_add": dangling_before_add,
            "dangling_merge_count_after_single_hanzi_add": dangling_after_add,
            "added_token_count": added_token_count,
            "new_token_init_count": len(init_token_ids),
            "new_hanzi_token_count": len(new_hanzi_token_ids),
            "promoted_merge_count": promoted_merge_count,
            "protected_token_count": len(protected_tokens),
            "protected_merge_count": len(protected_merge_ids),
            "remove_mixed_hanzi_tokens": self.config.remove_mixed_hanzi_tokens,
        }
        return SemanticVocabResult(
            vocab=new_vocab,
            merges=merges,
            new2old_token_id=new2old_token_id,
            new_token_init_token_ids=init_token_ids,
            removed_multi_hanzi_tokens=removed_multi_hanzi_tokens,
            new_hanzi_token_ids=new_hanzi_token_ids,
            manifest=manifest,
        )

    def build_and_write(self) -> SemanticVocabResult:
        tokenizer = self._load_tokenizer()
        result = self.build()
        if self.config.write_tokenizer_files:
            self._write_tokenizer_assets(tokenizer, result)
        return result

    def build_metadata_only(self) -> SemanticVocabResult:
        result = self.build()
        manifest_path = self.config.output_dir / "semantic_vocab_manifest.json"
        existing_manifest: dict[str, Any] | None = None
        if manifest_path.exists():
            with manifest_path.open("rt", encoding="utf-8") as file:
                existing_manifest = json.load(file)
            for key in ("old_vocab_size", "new_vocab_size", "removed_token_count"):
                if existing_manifest.get(key) != result.manifest.get(key):
                    raise ValueError(f"existing semantic vocab manifest does not match rebuilt {key}")

        self._write_alignment_metadata(result)
        if existing_manifest is not None:
            existing_manifest.update(
                {
                    key: value
                    for key, value in result.manifest.items()
                    if key.startswith("removed_multi_hanzi_")
                    or key.startswith("new_hanzi_token_")
                    or key.startswith("alignment_metadata_")
                }
            )
            result.manifest = existing_manifest

        write_json(manifest_path, result.manifest)
        return result

    def _load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.config.base_tokenizer_path,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=self.config.use_fast,
        )

    def _write_alignment_metadata(self, result: SemanticVocabResult) -> None:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        removed_path = output_dir / "removed_multi_hanzi_tokens.json"
        new_hanzi_path = output_dir / "new_hanzi_token_ids.json"

        write_json(removed_path, result.removed_multi_hanzi_tokens)
        write_json(new_hanzi_path, result.new_hanzi_token_ids)
        result.manifest.update(
            {
                "alignment_metadata_generated_at": _utc_now_iso(),
                "removed_multi_hanzi_tokens_file": removed_path.name,
                "removed_multi_hanzi_tokens_sha256": _file_sha256(removed_path),
                "new_hanzi_token_ids_file": new_hanzi_path.name,
                "new_hanzi_token_ids_sha256": _file_sha256(new_hanzi_path),
            }
        )

    def _write_tokenizer_assets(self, tokenizer: Any, result: SemanticVocabResult) -> None:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir)

        state = BpeState.from_tokenizer(tokenizer)
        write_json(output_dir / "vocab.json", result.vocab, indent=None)
        write_merges(output_dir / "merges.txt", result.merges)
        write_json(output_dir / "new2old_token_id.json", result.new2old_token_id)
        write_json(output_dir / "new_token_init_token_ids.json", result.new_token_init_token_ids)
        self._write_alignment_metadata(result)
        write_json(output_dir / "semantic_vocab_manifest.json", result.manifest)

        patched_tokenizer_json = state.patched_tokenizer_json(result.vocab, result.merges)
        if patched_tokenizer_json is not None:
            added_token_count = patch_added_tokens(
                patched_tokenizer_json,
                result.vocab,
                result.new2old_token_id,
            )
            result.manifest["added_token_count_after_patch"] = added_token_count
            result.manifest["new_vocab_size_after_added_tokens"] = len(result.vocab)
            write_json(output_dir / "tokenizer.json", patched_tokenizer_json, indent=None)
            write_json(output_dir / "vocab.json", result.vocab, indent=None)
            write_json(output_dir / "new2old_token_id.json", result.new2old_token_id)
            write_json(output_dir / "semantic_vocab_manifest.json", result.manifest)

        for filename in ("tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
            source = Path(self.config.base_tokenizer_path) / filename
            target = output_dir / filename
            if source.exists() and not target.exists():
                shutil.copy2(source, target)


def build_semantic_vocab(config: SemanticVocabBuildConfig) -> SemanticVocabResult:
    return SemanticVocabBuilder(config).build()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tokenizer-path", default="/share/project/wuhaiming/data/models/Qwen3-1.7B-Base/")
    parser.add_argument("--output-dir", type=Path, default=Path("models/Qwen3-1.7B-Base-Char"))
    parser.add_argument("--hanzi-set-path", type=Path, default=Path("resources/hanzi/hanzi_set.txt"))
    parser.add_argument("--allow-slow", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write alignment metadata without replacing tokenizer assets",
    )
    args = parser.parse_args()

    config = SemanticVocabBuildConfig(
        base_tokenizer_path=args.base_tokenizer_path,
        output_dir=args.output_dir,
        hanzi_set_path=args.hanzi_set_path,
        use_fast=not args.allow_slow,
    )
    builder = SemanticVocabBuilder(config)
    result = builder.build_metadata_only() if args.metadata_only else builder.build_and_write()
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
