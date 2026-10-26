"""Evaluate local HF Base/Char/ALIGN/CPT checkpoints on unmodified JSONL texts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

from ..common import ROOT, iter_jsonl, sha256
from .metrics import aggregate, score_document, validate_window


def inspect_inputs(input_dir):
    paths = sorted(Path(input_dir).resolve().glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no *.jsonl evaluation data in {input_dir}")
    sources = []
    domains = set()
    for path in paths:
        domain = path.stem.removeprefix("zh_")
        if domain in domains:
            raise ValueError(f"duplicate domain name: {domain}")
        domains.add(domain)
        chars = byte_count = documents = empty = 0
        for row in iter_jsonl(path):
            documents += 1
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{path}: record {documents} must contain a string text")
            if row.get("domain", domain) != domain:
                raise ValueError(f"{path}: record {documents} domain disagrees with filename")
            chars += len(text)
            byte_count += len(text.encode("utf-8"))
            empty += not text
        if not documents or empty == documents:
            raise ValueError(f"no nonempty documents in {path}")
        sources.append({"path": str(path), "domain": domain, "sha256": sha256(path),
                        "documents": documents, "empty_documents": empty,
                        "char_count": chars, "byte_count": byte_count})
    return sources


def checkpoint_identity(path):
    path = Path(path).resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(f"not a local full HF checkpoint (config.json required): {path}")
    if (path / "adapter_config.json").exists():
        raise ValueError("PPL requires full/merged weights, not a LoRA adapter directory")
    # Exclude args.json: training arguments may include credentials.
    metadata = {p.name: sha256(p) for p in path.iterdir() if p.is_file() and
                (p.name in {"config.json", "generation_config.json", "merges.txt", "vocab.json", "vocab.txt",
                            "special_tokens_map.json", "added_tokens.json", "tokenizer.model", "spiece.model",
                            "model.safetensors.index.json", "pytorch_model.bin.index.json"}
                 or p.name.startswith("tokenizer") or p.suffix == ".jinja")}
    runtime = {str(p.relative_to(path)): sha256(p) for p in path.rglob("*.py")}
    weights = {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
               for p in path.iterdir() if p.is_file() and
               (p.suffix == ".safetensors" or p.name.startswith("pytorch_model") and p.suffix == ".bin")}
    if not weights:
        raise ValueError(f"no model weights in {path}; tokenizer/build metadata alone cannot be evaluated")
    return {"path": str(path), "metadata_sha256": metadata, "runtime_code_sha256": runtime,
            "weights_stat": weights, "weight_identity_method": "path/size/mtime_ns, not full content hash"}


def load_model(model_path, dtype="auto", device="auto", attention="sdpa"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    if device != "auto":
        torch.device(device)  # Reject unsupported device strings early.
    options = {"dtype": dtype, "trust_remote_code": True, "local_files_only": True,
               "attn_implementation": attention, "output_loading_info": True,
               "device_map": "auto" if device == "auto" else {"": device}}
    model, loading = AutoModelForCausalLM.from_pretrained(model_path, **options)
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"checkpoint did not load completely: {loading}")
    mapping = getattr(model, "hf_device_map", {})
    devices = {str(value) for value in mapping.values()}
    if "disk" in devices or ("cpu" in devices and len(devices) > 1):
        raise ValueError("CPU/disk offload is not supported; use a fitting model or an explicit CPU run")
    model.eval()
    vocab_size = model.get_input_embeddings().num_embeddings
    if max(tokenizer.get_vocab().values()) >= vocab_size:
        raise ValueError("tokenizer IDs exceed model embedding vocabulary")
    return model, tokenizer, loading


def runtime_info(model, tokenizer, loading):
    import torch

    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "tokenizers", "accelerate", "tqdm")}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    return {"python": platform.python_version(), "packages": versions, "cuda": torch.version.cuda,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "model_class": type(model).__name__, "model_dtype": str(model.dtype),
            "device_map": {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()},
            "attention": getattr(model.config, "_attn_implementation", None),
            "model_settings": {key: getattr(model.config, key, None) for key in
                               ("model_type", "max_position_embeddings", "use_pgca", "pgca_enabled",
                                "pgca_context_enabled", "pgca_aux_enabled", "pgca_feature_index_sha256")},
            "pgca_enabled_env": os.environ.get("PGCA_ENABLED"),
            "unexpected_checkpoint_keys": sorted(loading.get("unexpected_keys", [])),
            "tokenizer_class": type(tokenizer).__name__, "tokenizer_length": len(tokenizer),
            "tokenizer_backend_sha256": hashlib.sha256(backend.to_str().encode()).hexdigest() if backend else None,
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id}


def write_report(path, value):
    # A failed run must never publish a truncated final metrics JSON.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_ppl(model_path, input_dir, output_path, window=2048, stride=1024,
                 dtype="auto", device="auto", attention="sdpa", loss_chunk_size=128):
    from tqdm import tqdm
    from ..runs import new_run_directory

    if not 0 < stride < window or loss_chunk_size < 1:
        raise ValueError("require 0 < stride < window and loss_chunk_size > 0")
    output = Path(output_path).resolve()
    documents_path = output.with_suffix(".documents.jsonl")
    manifest_path = output.with_suffix(".manifest.json")
    if output in {documents_path, manifest_path}:
        raise ValueError("output must be a distinct metrics JSON filename")
    for protected in (Path(input_dir).resolve(), Path(model_path).resolve()):
        if output.is_relative_to(protected):
            raise ValueError("evaluation outputs must be outside source data and model directories")
    output = new_run_directory(output.parent) / output.name
    documents_path = output.with_suffix(".documents.jsonl")
    manifest_path = output.with_suffix(".manifest.json")
    print(f"[evaluation][ppl] output={output.parent}", flush=True)
    sources = inspect_inputs(input_dir)
    identity = checkpoint_identity(model_path)
    model, tokenizer, loading = load_model(str(Path(model_path).resolve()), dtype, device, attention)
    validate_window(model, tokenizer, window, stride)
    protocol = {"window": window, "stride": stride, "loss_chunk_size": loss_chunk_size,
                "add_special_tokens": False, "prepend_bos": False, "append_eos": False,
                "chat_template": False, "truncation": False, "cross_document_context": False,
                "use_cache": False, "targets": "all encoded tokens except the first token of each document, once",
                "loss": "FP32 causal LM cross entropy, FP64 sum; no auxiliary loss",
                "character_unit": "all original Unicode code points, including whitespace/punctuation",
                "byte_unit": "UTF-8 bytes of original text (not JSON serialization)",
                "normalization": "full raw text; first token conditions NLL but is not scored",
                "no_target_documents": "retain raw counts; zero NLL; counts disclosed",
                "aggregation": "sum NLL / sum denominators, then exp; no mean of document/domain PPL"}
    manifest = {"complete": False, "model": identity, "sources": sources, "protocol": protocol,
                "runtime": runtime_info(model, tokenizer, loading),
                "code_sha256": {name: sha256(Path(__file__).with_name(name))
                                for name in ("evaluate.py", "metrics.py", "requirements.txt")},
                "common_sha256": sha256(Path(__file__).parents[1] / "common.py")}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents two evaluations sharing the same OUT.
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    started = time.monotonic()
    domains = {}
    with documents_path.open("x", encoding="utf-8") as handle:
        for source in sources:
            records = []
            for index, row in enumerate(tqdm(iter_jsonl(source["path"]), total=source["documents"],
                                              desc=f"PPL {source['domain']}", unit="docs"), 1):
                result = score_document(model, tokenizer, row["text"], window, stride, loss_chunk_size)
                records.append(result)
                document = {"domain": source["domain"], "record_index": index, "id": row.get("id"),
                            "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(), **result}
                handle.write(json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            domains[source["domain"]] = aggregate(records)
            if sha256(source["path"]) != source["sha256"]:
                raise RuntimeError(f"input changed during evaluation: {source['path']}")
    overall = aggregate(domains.values())
    if any(not value["token_count"] for value in domains.values()):
        raise ValueError("a domain has no scored targets; final metrics not published")
    if checkpoint_identity(model_path) != identity:
        raise RuntimeError("checkpoint changed during evaluation; final metrics not published")
    warnings = ["Token PPL is not directly comparable between Base and Char tokenizers.",
                "Char/byte metrics normalize conditional NLL by full text; first-token likelihood is excluded.",
                "Equal token windows do not imply equal raw-text context across tokenizers."]
    if overall["no_target_documents"]:
        warnings.append("Some nonempty documents have fewer than two tokens; inspect no_target_documents before comparison.")
    result = {"complete": True, "output": str(output), "model": identity["path"], "protocol": protocol, "domains": domains,
              "overall": overall, "warnings": warnings, "elapsed_seconds": time.monotonic() - started,
              "manifest": str(manifest_path), "documents": str(documents_path),
              "documents_sha256": sha256(documents_path)}
    write_report(output, result)
    manifest.update(complete=True, metrics_sha256=sha256(output), documents_sha256=result["documents_sha256"])
    write_report(manifest_path, manifest)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/ppl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect", action="store_true", help="validate all source rows without loading a model")
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--device", default="auto", help="auto (HF dispatch), cpu, cuda:0, ...")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    args = parser.parse_args(argv)
    if args.inspect:
        result = {"sources": inspect_inputs(args.input_dir)}
    else:
        if args.model is None or args.output is None:
            parser.error("--model and --output are required unless --inspect is set")
        result = evaluate_ppl(args.model, args.input_dir, args.output, args.window, args.stride,
                              args.dtype, args.device, args.attention, args.loss_chunk_size)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
