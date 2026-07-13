"""Semantic embedding and lm_head migration for the character vocabulary."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class EmbeddingMigrationConfig:
    base_model_path: str | Path = Path("/share/project/wuhaiming/data/models/Qwen3-1.7B-Base")
    char_model_path: Path = Path("models/Qwen3-1.7B-Base-Char")
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_int_key_json(path: Path) -> dict[int, int]:
    with path.open("rt", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(key): int(value) for key, value in raw.items()}


def read_init_json(path: Path) -> dict[int, list[int]]:
    if not path.exists():
        return {}
    with path.open("rt", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(key): [int(item) for item in value] for key, value in raw.items()}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_id_coverage(
    new_vocab_size: int,
    old_vocab_size: int,
    new2old: dict[int, int],
    init_ids: dict[int, list[int]],
) -> None:
    covered = set(new2old) | set(init_ids)
    missing = [token_id for token_id in range(new_vocab_size) if token_id not in covered]
    if missing:
        raise ValueError(f"New vocab id coverage is incomplete, first missing ids: {missing[:20]}")

    bad_old = [old_id for old_id in new2old.values() if old_id < 0 or old_id >= old_vocab_size]
    if bad_old:
        raise ValueError(f"new2old contains out-of-range old ids: {bad_old[:20]}")

    bad_init = [
        old_id
        for old_ids in init_ids.values()
        for old_id in old_ids
        if old_id < 0 or old_id >= old_vocab_size
    ]
    if bad_init:
        raise ValueError(f"new_token_init_token_ids contains out-of-range old ids: {bad_init[:20]}")


class SemanticEmbeddingMigrator:
    def __init__(self, config: EmbeddingMigrationConfig):
        self.config = config

    def migrate(self) -> dict:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.char_model_path,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
        )

        input_weight = model.get_input_embeddings().weight.detach()
        output_embedding = model.get_output_embeddings()
        output_weight = output_embedding.weight.detach() if output_embedding is not None else input_weight
        old_vocab_size, hidden_size = input_weight.shape
        new_vocab_size = len(tokenizer)

        new2old = read_int_key_json(self.config.char_model_path / "new2old_token_id.json")
        init_ids = read_init_json(self.config.char_model_path / "new_token_init_token_ids.json")
        validate_id_coverage(new_vocab_size, old_vocab_size, new2old, init_ids)

        new_input, new_output = self._build_new_weights(
            input_weight=input_weight,
            output_weight=output_weight,
            new_vocab_size=new_vocab_size,
            hidden_size=hidden_size,
            new2old=new2old,
            init_ids=init_ids,
        )

        model.set_input_embeddings(new_input)
        model.set_output_embeddings(new_output)
        model.config.vocab_size = new_vocab_size
        self._sync_special_token_ids(model, tokenizer)
        if getattr(model.config, "tie_word_embeddings", False):
            model.tie_weights()

        self.config.char_model_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.config.char_model_path)
        tokenizer.save_pretrained(self.config.char_model_path)

        tied_after = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
        report = {
            "generated_at": _utc_now_iso(),
            "base_model_path": str(self.config.base_model_path),
            "char_model_path": str(self.config.char_model_path),
            "old_vocab_size": int(old_vocab_size),
            "new_vocab_size": int(new_vocab_size),
            "hidden_size": int(hidden_size),
            "copied_token_count": len(new2old),
            "mean_initialized_token_count": len(init_ids),
            "dtype": str(input_weight.dtype),
            "tie_word_embeddings_config": bool(getattr(model.config, "tie_word_embeddings", False)),
            "tied_after_migration": tied_after,
            "pad_token_id": model.config.pad_token_id,
            "eos_token_id": model.config.eos_token_id,
            "bos_token_id": model.config.bos_token_id,
        }
        write_json(self.config.char_model_path / "embedding_migration_report.json", report)
        return report

    @staticmethod
    def _build_new_weights(
        *,
        input_weight: torch.Tensor,
        output_weight: torch.Tensor,
        new_vocab_size: int,
        hidden_size: int,
        new2old: dict[int, int],
        init_ids: dict[int, list[int]],
    ) -> tuple[torch.nn.Embedding, torch.nn.Linear]:
        device = input_weight.device
        dtype = input_weight.dtype
        new_input = torch.nn.Embedding(new_vocab_size, hidden_size, dtype=dtype, device=device)
        new_output = torch.nn.Linear(hidden_size, new_vocab_size, bias=False, dtype=dtype, device=device)

        with torch.no_grad():
            for new_id, old_id in sorted(new2old.items()):
                new_input.weight[new_id].copy_(input_weight[old_id])
                new_output.weight[new_id].copy_(output_weight[old_id])

            for new_id, old_ids in sorted(init_ids.items()):
                old_tensor = torch.tensor(old_ids, dtype=torch.long, device=device)
                new_input.weight[new_id].copy_(input_weight.index_select(0, old_tensor).mean(dim=0))
                new_output.weight[new_id].copy_(output_weight.index_select(0, old_tensor).mean(dim=0))

        return new_input, new_output

    @staticmethod
    def _sync_special_token_ids(model, tokenizer) -> None:
        for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
            value = getattr(tokenizer, name, None)
            setattr(model.config, name, value)
            if getattr(model, "generation_config", None) is not None:
                setattr(model.generation_config, name, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("migrate",), nargs="?", default="migrate")
    parser.add_argument("--base-model-path", default="/share/project/wuhaiming/data/models/Qwen3-1.7B-Base/")
    parser.add_argument("--char-model-path", type=Path, default=Path("models/Qwen3-1.7B-Base-Char"))
    parser.add_argument("--torch-dtype", default="auto")
    args = parser.parse_args()
    report = SemanticEmbeddingMigrator(
        EmbeddingMigrationConfig(
            base_model_path=args.base_model_path,
            char_model_path=args.char_model_path,
            torch_dtype=args.torch_dtype,
        )
    ).migrate()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
