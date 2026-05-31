"""
Build geographic CV folds for 7A training.

Tile filenames encode a 2-letter grid code (e.g. "BE", "KE") that positions
each tile on a roughly geographic grid over France. We map each letter to an
ordinal (A=0 … Z=25), treat the pair as a 2D coordinate, and KMeans-cluster
into K folds. This gives geographically separated splits without real lat/lon.

Writes: data/geo_folds.json  {core_id: fold_int, ...}
"""

import os
import re
import sys
import json
import glob
import argparse
import numpy as np
from sklearn.cluster import KMeans

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
import config

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "geo_folds.json")


def _data_root():
    head = "/mnt/head/users/bassam/data/geofmdata/embed2heights/data"
    if os.path.exists(config.ALPHA_EARTH_DIR):
        return os.path.dirname(os.path.dirname(config.ALPHA_EARTH_DIR))
    if os.path.exists(head):
        return head
    raise RuntimeError("Cannot find data root. Check config.py or HEAD_DATA_ROOT.")


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


def region_to_coord(code):
    row = ord(code[0].upper()) - ord("A")
    col = ord(code[1].upper()) - ord("A")
    return (float(row), float(col))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    data_root = _data_root()
    alpha_dir = os.path.join(data_root, "train", "alphaearth_emb")
    files = sorted(glob.glob(os.path.join(alpha_dir, "*.tif")))
    if not files:
        raise RuntimeError(f"No tiles found in {alpha_dir}")

    tile_ids = [_normalize_id(f) for f in files]
    coords = []
    for tid in tile_ids:
        parts = tid.split("_")
        region = parts[-1]
        if len(region) == 2 and region.isalpha():
            coords.append(region_to_coord(region))
        else:
            coords.append((0.0, 0.0))

    coords_arr = np.array(coords, dtype=np.float32)

    kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    labels = kmeans.fit_predict(coords_arr)

    folds = {tid: int(fold) for tid, fold in zip(tile_ids, labels)}

    print(f"Total tiles: {len(tile_ids)}")
    for k in range(args.k):
        fold_tiles = [tid for tid, fold in folds.items() if fold == k]
        regions = sorted(set(tid.split("_")[-1] for tid in fold_tiles))
        c = kmeans.cluster_centers_[k]
        print(f"  Fold {k}: {len(fold_tiles):4d} tiles  regions={regions[:6]}{'...' if len(regions)>6 else ''}  centroid=({c[0]:.1f},{c[1]:.1f})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(folds, f, indent=2)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
