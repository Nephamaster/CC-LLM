import os
from modelscope.msdatasets import MsDataset

CACHE_DIR = "/share/project/wuhaiming/data/dataset/.cache/clue"
OUT_DIR = "/share/project/wuhaiming/data/dataset/clue"

subsets = [
    "afqmc",
    "tnews",
    "iflytek",
    "cmnli",
    "cluewsc2020",
    "csl",
    "cmrc2018",
    "drcd",
    "chid",
    "c3",
    "ocnli",
    "diagnostics",
]

for subset in subsets:
    print(f"\n===== {subset} =====")

    ds = MsDataset.load(
        "modelscope/clue",
        subset_name=subset,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    os.makedirs(os.path.join(OUT_DIR, subset), exist_ok=True)

    # 多 split
    if isinstance(ds, dict):
        for split, split_ds in ds.items():
            hf_ds = split_ds.to_hf_dataset()
            path = os.path.join(
                OUT_DIR, subset, f"{split}.parquet"
            )
            hf_ds.to_parquet(path)
            print(f"{subset}/{split}: {len(hf_ds)} -> {path}")

    # 只有单个 split
    else:
        hf_ds = ds.to_hf_dataset()
        path = os.path.join(
            OUT_DIR, subset, "data.parquet"
        )
        hf_ds.to_parquet(path)
        print(f"{subset}: {len(hf_ds)} -> {path}")