"""Local HF or vLLM inference using the saved ms-swift training template."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import time
from pathlib import Path

from tqdm.auto import tqdm

from .common import ROOT, ensure_dir, iter_jsonl, read_journal, write_json, write_line
from .generation import token_budget

TEMPLATE_KEYS = (
    "system", "use_chat_template", "response_prefix", "enable_thinking", "preserve_thinking",
    "add_non_thinking_prefix", "template_backend", "agent_template", "norm_bbox",
)


def enable_generation_cache(model):
    model.config.use_cache = True
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = True


def relocated_training_plugins(plugins):
    paths = [plugins] if isinstance(plugins, str) else list(plugins or [])
    for index, value in enumerate(paths):
        path = Path(value)
        for stage in ("cpt", "finetune", "downstream"):
            if (path.parts[-5:] == ("scripts", "train", "ms_swift", stage, "plugin.py")
                    or path.parts[-4:] == ("scripts", "train", stage, "plugin.py")):
                if stage != "finetune" and path.is_file():
                    break
                target = "downstream" if stage == "finetune" else stage
                relocated = ROOT / "scripts/train" / target / "plugin.py"
                if relocated.is_file():
                    paths[index] = str(relocated)
                break
    return paths


def training_contract(model_path, adapter_path=None):
    checkpoint = Path(adapter_path or model_path).resolve()
    choices = [checkpoint / "args.json", checkpoint.parent / "args.json"]
    path = next((p for p in choices if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("training args.json is required; template guessing/raw prompt mode is disabled")
    args = json.loads(path.read_text(encoding="utf-8"))
    if not args.get("template") or "use_chat_template" not in args:
        raise ValueError("args.json does not identify the resolved training template")
    if args["use_chat_template"] is not True:
        raise ValueError("this evaluator requires a chat-template SFT checkpoint, not a CPT/raw-text checkpoint")
    # Never persist the full args.json: training arguments can contain Hub/API credentials.
    selected = {key: args[key] for key in TEMPLATE_KEYS if key in args}
    plugins = relocated_training_plugins(args.get("external_plugins"))
    return {"args_path": str(path), "template": args["template"],
            "settings": selected, "model_type": args.get("model_type"),
            "external_plugins": plugins}


def training_template(processor, contract, max_length, truncation_strategy="raise"):
    """Restore one saved Swift template for labels, generation and benchmark chat."""
    from swift.template import get_template
    settings = dict(contract["settings"])
    args = json.loads(Path(contract["args_path"]).read_text(encoding="utf-8"))
    for key in ("loss_scale", "is_binary_loss_scale", "padding_side"):
        if key in args:
            settings[key] = args[key]
    settings["default_system"] = settings.pop("system", None)
    if isinstance(settings["default_system"], str) and settings["default_system"].endswith(".txt"):
        settings["default_system"] = Path(settings["default_system"]).read_text(encoding="utf-8")
    return get_template(processor, template_type=contract["template"], max_length=max_length,
                        truncation_strategy=truncation_strategy, **settings)


def checkpoint_identity(path):
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return {"path": str(path)}


def resolve_backend(model_path, requested):
    if requested != "auto":
        return requested
    config = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    return "hf" if config.get("model_type") == "qwen3_pgca" else "vllm"


class TrainingTemplatePredictor:
    def __init__(self, model_path, adapter_path, contract, batch_size, max_new_tokens,
                 backend="auto", tensor_parallel_size=4):
        from swift import InferRequest, RequestConfig, TransformersEngine
        from swift.model import get_model_processor
        from swift.utils import import_external_file
        from transformers import AutoConfig

        plugins = contract["external_plugins"]
        for path in [plugins] if isinstance(plugins, str) else plugins:
            if not Path(path).is_file():
                raise FileNotFoundError(f"training plugin no longer exists: {path}")
            import_external_file(path)
        backend = resolve_backend(model_path, backend)
        load_model = backend == "hf"
        model, processor = get_model_processor(
            str(Path(model_path).resolve()), model_type=contract["model_type"], load_model=load_model)
        if model is not None:
            enable_generation_cache(model)
        config = model.config if model is not None else AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        context_length = int(config.max_position_embeddings)
        if max_new_tokens >= context_length:
            raise ValueError("generation budget must leave room for input within the model context")
        template = training_template(processor, contract, context_length - max_new_tokens)
        adapters = [str(Path(adapter_path).resolve())] if adapter_path else None
        if backend == "vllm":
            from swift import VllmEngine
            self.engine = VllmEngine(
                str(Path(model_path).resolve()), template=template, adapters=adapters,
                tensor_parallel_size=tensor_parallel_size, max_model_len=context_length,
                max_num_seqs=batch_size,
                engine_kwargs={"model_impl": "transformers", "trust_remote_code": True},
            )
        else:
            self.engine = TransformersEngine(model, template=template, adapters=adapters, max_batch_size=batch_size)
            enable_generation_cache(self.engine.model)
            self.engine.generation_config.use_cache = True
        self.request_cls = InferRequest
        self.config_cls = RequestConfig
        self.max_new_tokens = max_new_tokens
        self.info = {"backend": backend, "model_class": type(model).__name__ if model is not None else None,
                     "template": contract, "use_cache": True,
                     "tensor_parallel_size": tensor_parallel_size}

    def __call__(self, rows):
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[token_budget(row, self.max_new_tokens)].append((index, row))
        output = [None] * len(rows)
        for budget, group in groups.items():
            requests = [self.request_cls(messages=row["messages"]) for _, row in group]
            results = self.engine.infer(requests, self.config_cls(max_tokens=budget, temperature=0, stream=False), use_tqdm=False)
            if len(results) != len(group):
                raise RuntimeError("generation returned a different number of results")
            for (index, _), result in zip(group, results):
                if isinstance(result, Exception):
                    raise result
                text = result.choices[0].message.content
                if not isinstance(text, str):
                    raise RuntimeError("model response has no text content")
                output[index] = {"text": text, "finish_reason": result.choices[0].finish_reason,
                                 "generation_budget": budget,
                                 "generation_tokens": getattr(result.usage, "completion_tokens", None)}
        return output


def infer(input_path, output_path, model_path, adapter_path=None, batch_size=32, max_new_tokens=1024,
          predictor=None, backend="auto", tensor_parallel_size=4):
    from .csc.metrics import clean_prediction

    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch size and max-new-tokens must be positive")
    contract = training_contract(model_path, adapter_path)
    rows = list(iter_jsonl(input_path))
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError("evaluation input must be nonempty with unique IDs")
    output = Path(output_path)
    manifest_path = output.with_suffix(".manifest.json")
    identity = {"input": str(Path(input_path).resolve()), "model": checkpoint_identity(model_path),
                "adapter": checkpoint_identity(adapter_path) if adapter_path else None,
                "training_contract": contract, "max_new_tokens": max_new_tokens,
                "batch_size": batch_size, "decoder": "greedy", "use_cache": True,
                "backend": resolve_backend(model_path, backend),
                "tensor_parallel_size": tensor_parallel_size,
                "pgca_enabled_env": os.getenv("PGCA_ENABLED")}
    identity["dev_check"] = {"mode": os.getenv("DEV_CHECK_MODE", "not_run"),
                             "reports": os.getenv("DEV_CHECK_REPORT_ROOT")}
    previous = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    elif output.exists() or output.with_suffix(".retry.jsonl").exists():
        raise ValueError("predictions/journal exist without their identity manifest")
    else:
        ensure_dir(output.parent)
        manifest = {"identity": identity, "complete": False}
        write_json(manifest_path, manifest)
    if output.exists():
        previous = list(iter_jsonl(output))
        if len(previous) > len(rows):
            raise ValueError("prediction file has extra rows")
        for expected, actual in zip(rows, previous):
            if actual.get("id") != expected["id"]:
                raise ValueError("cannot resume altered or reordered inputs")
    # Error rows are retried; successful rows retain their exact original outputs.
    pending_ids = [i for i in range(len(rows)) if i >= len(previous) or previous[i].get("error")]
    if not pending_ids:
        print(f"[evaluation] inference already complete: {len(previous)}/{len(rows)} samples reused", flush=True)
        return {"complete": True, "count": len(previous), "reused": len(previous)}
    merged = {i: record for i, record in enumerate(previous) if not record.get("error")}
    # A separate journal makes retries safe without overwriting the successful prefix.
    journal = output.with_suffix(".retry.jsonl")
    if journal.exists():
        for record in read_journal(journal):
            i = record["row_number"]
            if i >= len(rows) or record.get("id") != rows[i]["id"]:
                raise ValueError("retry journal does not match inputs")
            if not record.get("error"):
                merged[i] = record
        pending_ids = [i for i in pending_ids if i not in merged]
    reused = len(rows) - len(pending_ids)
    print(f"[evaluation] inference samples: total={len(rows)} reused={reused} pending={len(pending_ids)} "
          f"batch_size={batch_size}", flush=True)
    if predictor is None and pending_ids:
        print("[evaluation] loading model and adapter", flush=True)
        predictor = TrainingTemplatePredictor(
            model_path, adapter_path, contract, batch_size, max_new_tokens,
            backend=backend, tensor_parallel_size=tensor_parallel_size)
        print("[evaluation] model and adapter loaded", flush=True)
    predictor_info = getattr(predictor, "info", manifest.get("predictor", {"backend": "test_fixture"}))
    write_json(manifest_path, {**manifest, "complete": False, "predictor": predictor_info})
    progress = tqdm(total=len(rows), initial=reused, desc="inference", unit="sample", dynamic_ncols=True)
    errors = 0
    try:
        with journal.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending_ids), batch_size):
                positions = pending_ids[start:start + batch_size]
                batch = [rows[i] for i in positions]
                began = time.perf_counter()
                try:
                    answers = predictor(batch)
                    if len(answers) != len(batch):
                        raise ValueError("wrong prediction count")
                except Exception as exc:
                    answers = [{"text": "", "error": f"{type(exc).__name__}: inference failed; inspect stderr"}] * len(batch)
                    import traceback
                    traceback.print_exc()
                for i, answer in zip(positions, answers):
                    raw = answer["text"]
                    cleaned = clean_prediction(raw, rows[i]["messages"][-1]["content"]) if rows[i]["task"] == "csc" else raw.strip()
                    record = {**rows[i], "row_number": i,
                              "predict_raw": raw, "predict_clean": cleaned, "error": answer.get("error"),
                              "generation_tokens": answer.get("generation_tokens"),
                              "generation_budget": answer.get("generation_budget", token_budget(rows[i], max_new_tokens)),
                              "finish_reason": answer.get("finish_reason"),
                              "elapsed_seconds": (time.perf_counter() - began) / len(batch)}
                    write_line(handle, record)
                    merged[i] = record
                    errors += bool(record["error"])
                handle.flush()
                progress.update(len(batch))
                progress.set_postfix(errors=errors)
    finally:
        progress.close()
    temporary = output.with_suffix(".tmp.jsonl")
    with temporary.open("w", encoding="utf-8") as handle:
        for i in range(len(rows)):
            write_line(handle, merged[i])
    temporary.replace(output)
    failures = sum(bool(r.get("error")) for r in merged.values())
    report = {"identity": identity, "count": len(rows), "errors": failures, "complete": not failures,
              "predictor": predictor_info}
    write_json(manifest_path, report)
    print(f"[evaluation] inference finished: count={len(rows)} errors={failures}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--backend", choices=("auto", "hf", "vllm"), default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    args = parser.parse_args()
    report = infer(args.input, args.output, args.model, args.adapter, args.batch_size, args.max_new_tokens,
                   backend=args.backend, tensor_parallel_size=args.tensor_parallel_size)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
