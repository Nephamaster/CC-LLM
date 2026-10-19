"""python -m scripts.data_factory.ancient_sft.run inspect|prepare|repartition|finalize"""

import argparse
import json

from .common import load_config
from .pipeline import finalize, inspect_inputs, prepare
from .repartition import repartition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "prepare", "repartition", "finalize"))
    parser.add_argument("--config", default="scripts/data_factory/ancient_sft/config.yaml")
    parser.add_argument("--output-dir", help="use a new output directory; existing stages are never overwritten")
    parser.add_argument("--limit", type=int, help="per-source diagnostic row limit; cannot finalize")
    parser.add_argument("--diagnostic", action="store_true", help="allow missing HistoryTrans holdouts for diagnosis only")
    parser.add_argument("--from-prepared", help="verified old prepared directory, required for repartition")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.stage != "prepare" and (args.limit is not None or args.diagnostic):
        parser.error("--limit/--diagnostic apply to prepare only")
    if (args.stage == "repartition") != bool(args.from_prepared):
        parser.error("--from-prepared is required only for repartition")
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.stage == "inspect":
        result = inspect_inputs(config)
    elif args.stage == "prepare":
        result = prepare(config, limit=args.limit, diagnostic=args.diagnostic)
    elif args.stage == "repartition":
        result = repartition(config, args.from_prepared)
    else:
        result = finalize(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
