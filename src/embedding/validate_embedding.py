"""Validation utilities for migrated semantic embeddings and feature memory."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import EmbeddingFeatureConfig
from .feature_embedding import PhoneticGlyphFeatureEmbedding
from .feature_memory import FeatureMemoryBuilder, load_feature_index


@dataclass(frozen=True)
class EmbeddingValidationConfig:
    char_model_path: Path = Path("models/Qwen3-1.7B-Base-Char")
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    validate_model_weights: bool = True


class EmbeddingValidator:
    def __init__(self, config: EmbeddingValidationConfig):
        self.config = config

    def validate(self) -> dict:
        report = {
            "char_model_path": str(self.config.char_model_path),
            "semantic": self._validate_semantic() if self.config.validate_model_weights else None,
            "feature": self._validate_feature_memory(),
        }
        report["passed"] = (report["semantic"] is None or report["semantic"]["passed"]) and report["feature"]["passed"]
        output_path = self.config.char_model_path / "reports" / "embedding_validation_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wt", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return report

    def _validate_semantic(self) -> dict:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.char_model_path,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.char_model_path,
            torch_dtype=self.config.torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        checks = {
            "vocab_size_matches": input_weight.shape[0] == len(tokenizer) == int(model.config.vocab_size),
            "hidden_size_matches": input_weight.shape[1] == int(model.config.hidden_size),
            "output_shape_matches": output_weight.shape == input_weight.shape,
            "tied_embeddings": input_weight.data_ptr() == output_weight.data_ptr(),
            "bos_token_id": model.config.bos_token_id,
            "pad_token_id_matches": model.config.pad_token_id == tokenizer.pad_token_id,
            "eos_token_id_matches": model.config.eos_token_id == tokenizer.eos_token_id,
            "bos_token_id_matches": model.config.bos_token_id == tokenizer.bos_token_id,
        }
        bool_keys = [key for key in checks if key != "bos_token_id"]
        return {"passed": all(checks[key] is True for key in bool_keys), "checks": checks}

    def _validate_feature_memory(self) -> dict:
        config_path = self.config.char_model_path / "embedding_config.json"
        feature_index_path = self.config.char_model_path / "features" / "feature_index.pt"
        if config_path.exists():
            with config_path.open("rt", encoding="utf-8") as f:
                feature_config = EmbeddingFeatureConfig(**json.load(f))
        else:
            feature_config = EmbeddingFeatureConfig.from_artifacts(self.config.char_model_path)

        feature_index = load_feature_index(feature_index_path)
        module = PhoneticGlyphFeatureEmbedding(feature_config)
        builder = FeatureMemoryBuilder(feature_index, module)

        hanzi_positions = torch.nonzero(feature_index["is_hanzi"], as_tuple=False).flatten()
        non_hanzi_positions = torch.nonzero(~feature_index["is_hanzi"], as_tuple=False).flatten()
        if len(hanzi_positions) == 0:
            raise ValueError("feature_index contains no Hanzi token")
        sample_hanzi = hanzi_positions[:2]
        sample_non_hanzi = non_hanzi_positions[:2] if len(non_hanzi_positions) else hanzi_positions[:0]
        input_ids = torch.cat([sample_hanzi, sample_non_hanzi]).unsqueeze(0)
        memory, mask = builder(input_ids)

        expected_shape = (
            input_ids.shape[0],
            input_ids.shape[1],
            feature_config.num_feature_slots,
            feature_config.d_model,
        )
        checks = {
            "memory_shape": tuple(memory.shape),
            "mask_shape": tuple(mask.shape),
            "expected_memory_shape": expected_shape,
            "memory_shape_matches": tuple(memory.shape) == expected_shape,
            "mask_shape_matches": tuple(mask.shape) == expected_shape[:-1],
            "hanzi_has_feature": bool(mask[:, : len(sample_hanzi)].any().item()),
            "non_hanzi_empty": True
            if len(sample_non_hanzi) == 0
            else bool((~mask[:, len(sample_hanzi) :]).all().item()),
        }
        pass_keys = ("memory_shape_matches", "mask_shape_matches", "hanzi_has_feature", "non_hanzi_empty")
        return {"passed": all(checks[key] is True for key in pass_keys), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--char-model-path", type=Path, default=Path("models/Qwen3-1.7B-Base-Char"))
    parser.add_argument("--skip-model-weights", action="store_true")
    parser.add_argument("--torch-dtype", default="auto")
    args = parser.parse_args()
    report = EmbeddingValidator(
        EmbeddingValidationConfig(
            char_model_path=args.char_model_path,
            torch_dtype=args.torch_dtype,
            validate_model_weights=not args.skip_model_weights,
        )
    ).validate()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()