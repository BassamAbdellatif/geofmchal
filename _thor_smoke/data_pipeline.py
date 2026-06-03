"""End-to-end dataset->model smoke WITHOUT the cache (reads a few NFS tiles).
Validates THOR dataset plumbing + model routing before the cache is rebuilt."""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dataset import find_multimodal_train_tiles, GeoFMDataset7A
from core.model import build_model
from torch.utils.data import DataLoader

DATA_ROOT = "/mnt/head/users/bassam/data/geofmdata/embed2heights/data"


def run(tag, patch_inputs, stem, is_train):
    tiles = find_multimodal_train_tiles(DATA_ROOT, use_thor=any("thor" in p for p in patch_inputs))
    tiles = tiles[:60]  # keep coverage-score scan + IO small
    ds = GeoFMDataset7A(tiles, is_train=is_train, cv_fold=0, cache_dir=None,
                        patch_inputs=patch_inputs)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    keys = sorted(batch.keys())
    shapes = {k: tuple(batch[k].shape) for k in patch_inputs}
    model, _ = build_model("dual_enc_dec_fusion", 64, 4,
                           patch_inputs=patch_inputs, patch_stem_version=stem, xattn_heads=4)
    model.eval()
    with torch.no_grad():
        out = model({k: v for k, v in batch.items()})
    finite = all(torch.isfinite(out[k]).all().item() for k in out)
    print(f"[{tag:14s}] is_train={is_train} n={len(ds)} keys={keys}")
    print(f"                patch_shapes={shapes} -> out_finite={finite} "
          f"frac={tuple(out['fraction'].shape)}")


def main():
    run("baseline",  ["terramind_s1", "terramind_s2"], "v1", is_train=False)
    run("thor_full", ["terramind_s1", "terramind_s2", "thor_s1", "thor_s2"], "v2", is_train=False)
    run("thor_full_tr", ["terramind_s1", "terramind_s2", "thor_s1", "thor_s2"], "v2", is_train=True)
    run("thor_s2",   ["terramind_s1", "terramind_s2", "thor_s2"], "v2", is_train=True)
    print("OK")


if __name__ == "__main__":
    main()
