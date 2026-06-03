"""
Compute per-channel mean and std for each embedding modality on the training split.

Reads data/geo_folds.json and excludes the validation fold (default fold 0)
so stats are computed on training tiles only.

Writes: data/norm_stats.json
  {
    "alpha_earth":   {"mean": [...64...], "std": [...64...]},
    "tessera":       {"mean": [...128..], "std": [...128..]},
    "terramind_s1":  {"mean": [...768..], "std": [...768..]},
    "terramind_s2":  {"mean": [...768..], "std": [...768..]},
  }
"""

import os
import re
import sys
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
import config

GEO_FOLDS_PATH = os.path.join(REPO_ROOT, "data", "geo_folds.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "norm_stats.json")


def _data_root():
    head = "/mnt/head/users/bassam/data/geofmdata/embed2heights/data"
    if os.path.exists(config.ALPHA_EARTH_DIR):
        return os.path.dirname(os.path.dirname(config.ALPHA_EARTH_DIR))
    if os.path.exists(head):
        return head
    raise RuntimeError("Cannot find data root.")


def _normalize_id(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    for prefix in ("gee_emb_", "tessera_emb_", "s2_", "s1_", "label_", "emb_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    for suf in ("_embeddings", "_embedding", "_merged", "_quantized"):
        if base.endswith(suf):
            base = base[:-len(suf)]
    base = re.sub(r"_\d{4}$", "", base)
    return base


def compute_stats(files):
    """
    Vectorised two-pass: accumulate sum and sum-of-squares in float64,
    then compute mean and corrected std. Much faster than per-pixel Welford.
    """
    import rasterio
    total_sum = None
    total_sum_sq = None
    total_n = 0

    for f in tqdm(files, leave=False):
        with rasterio.open(f) as src:
            data = src.read().astype(np.float32)
        # Sanitize: drop NaN/Inf and corrupt sentinels (>1e6 magnitude). At least
        # one tessera tile carries a ~6.6e36 value that poisons global stats.
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data[np.abs(data) > 1e6] = 0.0
        data = data.astype(np.float64)
        C = data.shape[0]
        flat = data.reshape(C, -1)   # (C, N)
        if total_sum is None:
            total_sum    = np.zeros(C, dtype=np.float64)
            total_sum_sq = np.zeros(C, dtype=np.float64)
        total_sum    += flat.sum(axis=1)
        total_sum_sq += (flat ** 2).sum(axis=1)
        total_n      += flat.shape[1]

    mean = total_sum / total_n
    var  = total_sum_sq / total_n - mean ** 2
    std  = np.sqrt(np.maximum(var, 1e-12))
    return mean.tolist(), std.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fold", type=int, default=0,
                        help="Fold to exclude (used as validation)")
    parser.add_argument("--geo-folds", type=str, default=GEO_FOLDS_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated subset of modalities to (re)compute "
                             "(e.g. 'thor_s1,thor_s2'). Other existing keys in the "
                             "output file are preserved (merge, not overwrite).")
    args = parser.parse_args()

    with open(args.geo_folds) as f:
        folds = json.load(f)

    train_ids = {tid for tid, fold in folds.items() if fold != args.val_fold}
    print(f"Training tiles (fold != {args.val_fold}): {len(train_ids)}")

    data_root = _data_root()
    train_dir = os.path.join(data_root, "train")

    modalities = {
        "alpha_earth":  "alphaearth_emb",
        "tessera":      "tessera_emb",
        "terramind_s1": "terramind_s1_emb",
        "terramind_s2": "terramind_s2_emb",
        "thor_s1":      "thor_s1_emb",
        "thor_s2":      "thor_s2_emb",
    }

    if args.only:
        wanted = [m.strip() for m in args.only.split(",") if m.strip()]
        for m in wanted:
            if m not in modalities:
                raise ValueError(f"Unknown modality '{m}'. Valid: {list(modalities)}")
        modalities = {k: v for k, v in modalities.items() if k in wanted}
        print(f"Subset mode (--only): computing {list(modalities)}; "
              f"preserving other existing keys.")

    # Merge into existing stats rather than overwrite, so a subset recompute
    # (e.g. THOR) never disturbs already-validated terramind/alpha/tessera stats.
    stats = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            stats = json.load(f)

    for name, subdir in modalities.items():
        all_files = sorted(glob.glob(os.path.join(train_dir, subdir, "*.tif")))
        files = [f for f in all_files if _normalize_id(f) in train_ids]
        print(f"\nComputing stats for {name}: {len(files)} tiles ...")
        mean, std = compute_stats(files)
        stats[name] = {"mean": mean, "std": std}
        print(f"  mean range [{min(mean):.4f}, {max(mean):.4f}]  "
              f"std range [{min(std):.4f}, {max(std):.4f}]")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWritten: {args.output}  (keys: {sorted(stats)})")


if __name__ == "__main__":
    main()
