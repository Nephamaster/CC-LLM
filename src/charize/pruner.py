import argparse
import json
import os
from typing import Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class VocabularyPruner(object):

    @staticmethod
    def _load_mapping_from_tokenizer_dir(new_tokenizer_name_or_path: str) -> Dict[int, int] | None:
        mapping_path = os.path.join(new_tokenizer_name_or_path, "new2old_token_id.json")
        if not os.path.exists(mapping_path):
            return None
        with open(mapping_path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        mapping = {int(k): int(v) for k, v in raw.items()}
        print(f"Loaded mapping from: {mapping_path}, size={len(mapping)}")
        return mapping

    @staticmethod
    def _load_init_token_ids_from_tokenizer_dir(new_tokenizer_name_or_path: str) -> Dict[int, list[int]]:
        init_path = os.path.join(new_tokenizer_name_or_path, "new_token_init_token_ids.json")
        if not os.path.exists(init_path):
            return {}
        with open(init_path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        init_token_ids = {int(k): [int(v) for v in values] for k, values in raw.items()}
        print(f"Loaded new-token init ids from: {init_path}, size={len(init_token_ids)}")
        return init_token_ids

    @staticmethod
    def _build_mapping_from_vocab(old_vocab: Dict[str, int], new_vocab: Dict[str, int]) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for token, new_id in new_vocab.items():
            if token not in old_vocab:
                raise ValueError(f"Token not in old vocab: {token}")
            mapping[int(new_id)] = int(old_vocab[token])
        return mapping

    @staticmethod
    def _validate_mapping(
        mapping: Dict[int, int],
        init_token_ids: Dict[int, list[int]],
        old_vocab_size: int,
        new_vocab_size: int,
    ):
        covered_new_ids = set(mapping) | set(init_token_ids)
        if len(covered_new_ids) != new_vocab_size:
            raise ValueError(
                f"Mapping size mismatch: copied={len(mapping)} initialized={len(init_token_ids)} "
                f"new_vocab={new_vocab_size}"
            )
        missing_new_ids = [i for i in range(new_vocab_size) if i not in covered_new_ids]
        if missing_new_ids:
            raise ValueError(f"Mapping missing new ids, head={missing_new_ids[:20]}")
        bad_old_ids = [old_id for old_id in mapping.values() if old_id < 0 or old_id >= old_vocab_size]
        if bad_old_ids:
            raise ValueError(f"Mapping has out-of-range old ids, head={bad_old_ids[:20]}")
        bad_init_old_ids = [
            old_id
            for old_ids in init_token_ids.values()
            for old_id in old_ids
            if old_id < 0 or old_id >= old_vocab_size
        ]
        if bad_init_old_ids:
            raise ValueError(f"Init mapping has out-of-range old ids, head={bad_init_old_ids[:20]}")

    @staticmethod
    def _augment_mapping_with_vocab_alignment(
        mapping: Dict[int, int],
        init_token_ids: Dict[int, list[int]],
        old_vocab: Dict[str, int],
        new_vocab: Dict[str, int],
    ) -> Dict[int, int]:
        """
        Fill missing new_id entries (typically added/special tokens not present in base BPE vocab mapping)
        by token-string alignment against old_vocab.
        """
        completed = dict(mapping)
        missing_new_ids = [
            new_id
            for new_id in range(len(new_vocab))
            if new_id not in completed and new_id not in init_token_ids
        ]
        if not missing_new_ids:
            return completed

        id2token = {int(v): k for k, v in new_vocab.items()}
        for new_id in missing_new_ids:
            token = id2token.get(new_id)
            if token is None:
                raise ValueError(f"Cannot find token for missing new_id={new_id}")
            old_id = old_vocab.get(token)
            if old_id is None:
                raise ValueError(f"Missing token in old vocab while completing mapping: token={token} new_id={new_id}")
            completed[new_id] = int(old_id)
        print(f"Completed mapping with {len(missing_new_ids)} extra token ids from vocab alignment")
        return completed

    def check(self, old_model_name_or_path, new_model_name_or_path, text):
        max_length = 10

        old_model = AutoModelForCausalLM.from_pretrained(old_model_name_or_path, trust_remote_code=True)
        old_tokenizer = AutoTokenizer.from_pretrained(old_model_name_or_path, trust_remote_code=True, use_fast=False)
        old_input_ids = old_tokenizer(text, return_tensors="pt").input_ids
        old_output = old_model.generate(old_input_ids, max_new_tokens=max_length, do_sample=False, num_beams=1)
        old_output_text = old_tokenizer.batch_decode(old_output)
        print(f"old_output:{old_output_text}")

        new_model = AutoModelForCausalLM.from_pretrained(new_model_name_or_path, trust_remote_code=True)
        new_tokenizer = AutoTokenizer.from_pretrained(new_model_name_or_path, trust_remote_code=True, use_fast=False)
        new_input_ids = new_tokenizer(text, return_tensors="pt").input_ids
        new_output = new_model.generate(new_input_ids, max_new_tokens=max_length, do_sample=False, num_beams=1)
        new_output_text = new_tokenizer.batch_decode(new_output)
        print(f"new_output:{new_output_text}")

        if old_output_text == new_output_text:
            print("output is same, succeed to prune.")
        else:
            print("output is not same, fail to prune.")

    def check_embedding(self, old_model_name_or_path, new_model_name_or_path, text):
        old_model = AutoModelForCausalLM.from_pretrained(old_model_name_or_path, trust_remote_code=True)
        old_tokenizer = AutoTokenizer.from_pretrained(old_model_name_or_path, trust_remote_code=True, use_fast=False)
        old_input_ids = old_tokenizer(text, return_tensors="pt").input_ids
        old_in = old_model.get_input_embeddings().weight.data[old_input_ids]
        old_out = old_model.get_output_embeddings().weight.data[old_input_ids]

        new_model = AutoModelForCausalLM.from_pretrained(new_model_name_or_path, trust_remote_code=True)
        new_tokenizer = AutoTokenizer.from_pretrained(new_model_name_or_path, trust_remote_code=True, use_fast=False)
        new_input_ids = new_tokenizer(text, return_tensors="pt").input_ids
        new_in = new_model.get_input_embeddings().weight.data[new_input_ids]
        new_out = new_model.get_output_embeddings().weight.data[new_input_ids]

        print("old_tokens:", old_tokenizer.tokenize(text))
        print("new_tokens:", new_tokenizer.tokenize(text))
        print("input_embed_equal:", torch.equal(old_in, new_in))
        print("output_embed_equal:", torch.equal(old_out, new_out))

    def update_embeddings(self, model, new2old_token_id, init_token_ids, new_embeds, new_lm_head):
        raise NotImplementedError

    def prune(self, model_name_or_path, new_tokenizer_name_or_path, save_path, new_name_or_path=None):
        os.makedirs(save_path, exist_ok=True)

        new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_name_or_path, trust_remote_code=True, use_fast=False)
        old_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)

        print("new_tokenizer:", len(new_tokenizer), "old_tokenizer:", len(old_tokenizer))

        old_vocab = old_tokenizer.get_vocab()
        new_vocab = new_tokenizer.get_vocab()

        mapping = self._load_mapping_from_tokenizer_dir(new_tokenizer_name_or_path)
        init_token_ids = self._load_init_token_ids_from_tokenizer_dir(new_tokenizer_name_or_path)
        if mapping is None:
            print("new2old_token_id.json not found; fallback to token-string alignment mapping")
            mapping = self._build_mapping_from_vocab(old_vocab, new_vocab)
        else:
            mapping = self._augment_mapping_with_vocab_alignment(mapping, init_token_ids, old_vocab, new_vocab)

        self._validate_mapping(
            mapping,
            init_token_ids,
            old_vocab_size=len(old_vocab),
            new_vocab_size=len(new_vocab),
        )

        for token, new_id in tqdm(new_vocab.items(), desc="validate_token_mapping"):
            if int(new_id) in init_token_ids:
                continue
            old_id_by_token = old_vocab.get(token)
            if old_id_by_token is None:
                raise ValueError(f"token missing in old vocab: {token}")
            old_id_by_mapping = mapping[int(new_id)]
            if int(old_id_by_token) != int(old_id_by_mapping):
                raise ValueError(
                    f"mapping mismatch token={token} new_id={new_id} "
                    f"mapped_old_id={old_id_by_mapping} expected_old_id={old_id_by_token}"
                )

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto", trust_remote_code=True)
        print("Loaded model config vocab_size:", model.config.vocab_size)
        print("Embedding weight shape:", tuple(model.get_input_embeddings().weight.shape))

        old_params = sum(p.numel() for p in model.parameters())
        print("Total params of original model: %.2fM" % (old_params / 1e6))

        vocab_size = len(new_vocab)
        hidden_size = model.config.hidden_size

        new_embeds = torch.nn.Embedding(vocab_size, hidden_size, dtype=model.dtype)
        new_lm_head = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=False, dtype=model.dtype)

        self.update_embeddings(model, mapping, init_token_ids, new_embeds, new_lm_head)

        model.config.vocab_size = vocab_size
        if new_name_or_path is not None:
            model.config._name_or_path = new_name_or_path

        if new_tokenizer.pad_token_id is not None:
            model.config.pad_token_id = new_tokenizer.pad_token_id
        if new_tokenizer.eos_token_id is not None:
            model.config.eos_token_id = new_tokenizer.eos_token_id
        if new_tokenizer.bos_token_id is not None:
            model.config.bos_token_id = new_tokenizer.bos_token_id

        if hasattr(model, "generation_config") and model.generation_config is not None:
            if model.config.pad_token_id is not None:
                model.generation_config.pad_token_id = model.config.pad_token_id
            if model.config.eos_token_id is not None:
                model.generation_config.eos_token_id = model.config.eos_token_id
            if model.config.bos_token_id is not None:
                model.generation_config.bos_token_id = model.config.bos_token_id

        new_params = sum(p.numel() for p in model.parameters())
        print("Total params of new model: %.2fM" % (new_params / 1e6))
        print("vocab keep ratio: {}%".format(round(len(new_tokenizer) / len(old_tokenizer), 4) * 100))
        print("param keep ratio: {}%".format(round(new_params / old_params, 4) * 100))

        model.save_pretrained(save_path)
        new_tokenizer.save_pretrained(save_path)

        mapping_path = os.path.join(save_path, "new2old_token_id.json")
        with open(mapping_path, "wt", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"Saved mapping to: {mapping_path}")


class ModelVocabularyPruner(VocabularyPruner):

    def update_embeddings(self, model, new2old_token_id: Dict[int, int], init_token_ids: Dict[int, list[int]], new_embeds, new_lm_head):
        input_weight = model.get_input_embeddings().weight.data
        output_weight = model.get_output_embeddings().weight.data
        for new_id, old_id in tqdm(sorted(new2old_token_id.items()), desc="copy_embeddings"):
            new_embeds.weight.data[new_id] = input_weight[old_id]
            new_lm_head.weight.data[new_id] = output_weight[old_id]

        for new_id, old_ids in tqdm(init_token_ids.items(), desc="init_new_embeddings"):
            old_id_tensor = torch.tensor(old_ids, dtype=torch.long, device=input_weight.device)
            new_embeds.weight.data[new_id] = input_weight.index_select(0, old_id_tensor).mean(dim=0)
            new_lm_head.weight.data[new_id] = output_weight.index_select(0, old_id_tensor).mean(dim=0)

        model.set_input_embeddings(new_embeds)
        model.set_output_embeddings(new_lm_head)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_model_path", type=str, required=True)
    parser.add_argument("--new_model_path", type=str, required=True)
    args = parser.parse_args()

    # Example:
    # 1) Run tokenizer_prune_qwen.py first to generate pruned tokenizer + new2old_token_id.json
    # 2) Then run this file to shrink model embeddings/lm_head accordingly.

    pruner = ModelVocabularyPruner()
    pruner.prune(args.old_model_path, args.new_model_path, args.new_model_path)

    # Optional sanity checks after pruning:
    # pruner.check(model_name_or_path, save_path, text="这是一个中文分词测试")
    # pruner.check_embedding(model_name_or_path, save_path, text="中文")
