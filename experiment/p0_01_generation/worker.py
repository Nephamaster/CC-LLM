"""Run one model/framework cell in an isolated process on a single visible GPU."""

import argparse
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import time
import traceback

from .records import HFTrace, append_jsonl, complete_record, observe_vllm, write_json


def read_cases(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def runtime():
    import os
    import torch

    packages = {}
    for package in ("torch", "transformers", "ms-swift", "lm_eval", "vllm", "tokenizers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "executable": sys.executable, "packages": packages,
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_cuda": torch.version.cuda}


def template_for(model_path):
    from swift.model import get_model_processor
    from swift.utils import import_external_file
    from scripts.evaluation.inference import training_contract, training_template

    contract = training_contract(model_path)
    if contract["settings"].get("enable_thinking") is not False:
        raise ValueError("this experiment requires the actual saved non-thinking INS contract")
    for path in contract["external_plugins"]:
        import_external_file(path)
    _, processor = get_model_processor(str(model_path), model_type=contract["model_type"],
                                       load_model=False, download_model=False)
    template = training_template(processor, contract, 8192)
    template.set_mode("transformers")
    return template, contract


def prepare(args):
    template, contract = template_for(args.model)
    tokenizer = template.tokenizer
    config = json.loads((args.model / "config.json").read_text())
    model_keys = ("model_type", "architectures", "vocab_size", "eos_token_id", "pad_token_id",
                  "pgca_enabled", "pgca_context_mode", "pgca_context_enabled", "pgca_layers")
    generation = args.model / "generation_config.json"
    write_json(args.output / "model.json", {
        "model_path": str(args.model.resolve()), "runtime": runtime(), "training_contract": contract,
        "config": {k: config.get(k) for k in model_keys},
        "saved_generation_config": json.loads(generation.read_text()) if generation.exists() else None,
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_eos_id": tokenizer.eos_token_id,
        "tokenizer_pad_id": tokenizer.pad_token_id,
    })
    prepared = []
    for case in read_cases(args.cases):
        snapshot = case["historical"][args.alias]
        prompt = snapshot["prompt"]
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        ordinary = template.encode({"messages": deepcopy(case["ordinary_messages"])})["input_ids"]
        generation = deepcopy(snapshot["generation"])
        budget = int(generation["max_gen_toks"])
        if len(ids) + budget > 8192 or len(ordinary) + budget > 8192:
            raise ValueError(f"fixed case exceeds context: {case['id']}")
        prepared.append({
            "id": case["id"], "benchmark": case["benchmark"], "reference": case["reference"],
            "document": case["document"], "historical_available": snapshot["available"],
            "historical_source": snapshot["source"], "historical_response": snapshot["response"],
            "prompt": prompt, "prompt_ids": ids, "generation": generation, "budget": budget,
            "ordinary_messages": case["ordinary_messages"], "ordinary_prompt_ids": ordinary,
            "ordinary_prompt": tokenizer.decode(ordinary, skip_special_tokens=False),
        })
    temporary = args.output / "cases.jsonl.tmp"
    temporary.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in prepared), encoding="utf-8")
    temporary.replace(args.output / "cases.jsonl")


def new_record(args, case):
    ordinary = args.group == "D"
    return {"case_id": case["id"], "benchmark": case["benchmark"], "model": args.alias,
            "model_path": str(args.model.resolve()), "group": args.group,
            "prompt": case["ordinary_prompt"] if ordinary else case["prompt"],
            "expected_prompt_ids": case["ordinary_prompt_ids"] if ordinary else case["prompt_ids"],
            "budget": case["budget"], "requested_stop_strings": case["generation"].get("until", []),
            "historical_available": case["historical_available"], "batch_size": 1,
            "prepared_cases": str(args.prepared / "cases.jsonl"),
            "effective_config_observed": False, "completed": False}


def run_cases(args, tokenizer, action):
    for case in read_cases(args.prepared / "cases.jsonl"):
        record = new_record(args, case)
        print(f"[P0-01] {args.alias}/{args.group} {case['id']} budget={case['budget']}", flush=True)
        start = time.monotonic()
        try:
            action(case, record)
            complete_record(record, tokenizer)
            record["completed"] = True
        except Exception as error:
            traceback.print_exc()
            record["error"] = f"{type(error).__name__}: {error}"
        record["elapsed_seconds"] = time.monotonic() - start
        append_jsonl(args.output / "generations.jsonl", record)
    records = read_cases(args.output / "generations.jsonl")
    if not all(record["completed"] for record in records):
        raise RuntimeError("one or more cases failed; inspect generations.jsonl and worker.log")


def harness(args):
    from lm_eval.api.instance import Instance
    from scripts.evaluation.lmeh.backend import StrictHFLM, build_vllm_backend
    from scripts.evaluation.ppl.evaluate import load_model

    if args.alias == "base":
        backend = build_vllm_backend(args.model, dtype="bfloat16", batch_size=1,
                                     max_length=8192, tensor_parallel_size=1)
    else:
        model, tokenizer, _ = load_model(str(args.model), "bfloat16", args.device, "sdpa")
        backend = StrictHFLM(pretrained=model, tokenizer=tokenizer, backend="causal",
                             max_length=8192, batch_size=1, device=args.device,
                             add_bos_token=False, softmax_dtype="float32")
    # Prompts are already rendered historical benchmark text; keep the same generation wrapper.
    backend.instruct_protocol = {"formatter": "fixed historical benchmark prompt"}

    def action(case, record):
        request = Instance(request_type="generate_until", doc=case["document"],
                           arguments=(case["prompt"], deepcopy(case["generation"])), idx=0)
        observer = (observe_vllm(backend, record, case["prompt_ids"]) if args.alias == "base"
                    else HFTrace(backend.model).observe(record, case["prompt_ids"]))
        with observer:
            record["processed_text"] = backend.generate_until([request])[0]
        if args.alias == "base":
            record["vllm_model_parameters"] = {k: backend.model_args.get(k) for k in
                                                ("dtype", "seed", "tensor_parallel_size", "max_model_len")}

    run_cases(args, backend.tokenizer, action)


def native(args):
    import torch
    from transformers import GenerationConfig, StoppingCriteriaList, StopStringCriteria
    from scripts.evaluation.ppl.evaluate import load_model

    model, tokenizer, _ = load_model(str(args.model), "bfloat16", args.device, "sdpa")
    tracer = HFTrace(model)

    def action(case, record):
        config = GenerationConfig.from_dict(model.generation_config.to_dict())
        config.max_new_tokens = case["budget"]
        config.max_length = 8192
        config.do_sample = False
        config.use_cache = True
        config.return_dict_in_generate = True
        config.output_scores = False
        config.output_logits = False
        stop = case["generation"].get("until", [])
        criteria = StoppingCriteriaList([StopStringCriteria(tokenizer, stop)]) if stop else StoppingCriteriaList()
        ids = torch.tensor([case["prompt_ids"]], dtype=torch.long, device=model.device)
        with tracer.observe(record, case["prompt_ids"]):
            model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                           generation_config=config, stopping_criteria=criteria)
        record["processed_text"] = tokenizer.decode(record["generated_ids"], skip_special_tokens=True)

    run_cases(args, tokenizer, action)


def swift(args):
    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine
    from swift.model import get_model_processor

    template, contract = template_for(args.model)
    model, _ = get_model_processor(str(args.model), model_type=contract["model_type"],
                                   torch_dtype=torch.bfloat16, device_map={"": args.device},
                                   attn_impl="sdpa", download_model=False)
    engine = TransformersEngine(model, template=template, max_batch_size=1)
    tracer = HFTrace(model)

    def action(case, record):
        request = InferRequest(messages=deepcopy(case["ordinary_messages"]))
        config = RequestConfig(max_tokens=case["budget"], temperature=0, stream=False,
                               stop=case["generation"].get("until", []), return_details=True)

        def exact_input(kwargs):
            # Keep Swift's real collation/config/generation path; replace only the input tensor.
            ids = torch.tensor([case["prompt_ids"]], dtype=torch.long, device=model.device)
            kwargs["inputs"] = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            record["input_injection"] = "TransformersEngine pre_infer_hook; no second Chat Template"
            return kwargs

        with tracer.observe(record, record["expected_prompt_ids"]):
            if args.group == "C":
                result = engine._infer([request], config, pre_infer_hook=exact_input)[0]
            else:
                result = engine.infer([request], config, use_tqdm=False)[0]
        if isinstance(result, Exception):
            raise result
        record["processed_text"] = result.choices[0].message.content
        record["swift_finish_reason"] = result.choices[0].finish_reason

    run_cases(args, template.tokenizer, action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--alias", choices=("base", "shared"), required=True)
    parser.add_argument("--group", choices=("prepare", "A", "B", "C", "D"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "status.json", {"state": "running", "model": args.alias, "group": args.group})
    try:
        import random
        import torch

        random.seed(42)
        torch.manual_seed(42)
        write_json(args.output / "runtime.json", runtime())
        {"prepare": prepare, "A": harness, "B": native, "C": swift, "D": swift}[args.group](args)
    except Exception as error:
        write_json(args.output / "status.json", {"state": "failed", "error": str(error),
                                                 "model": args.alias, "group": args.group})
        raise
    write_json(args.output / "status.json", {"state": "complete", "model": args.alias, "group": args.group})


if __name__ == "__main__":
    main()
