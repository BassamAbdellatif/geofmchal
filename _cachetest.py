import json, time, numpy as np, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from core.dataset import find_multimodal_train_tiles, GeoFMDataset7A

DATA_ROOT = "/mnt/head/users/bassam/data/geofmdata/embed2heights/data"
CACHE_DIR = "/mnt/head/users/bassam/data/geofmdata/cache7a"


def main():
    r = {}
    tiles = find_multimodal_train_tiles(DATA_ROOT)

    # --- correctness: val (no aug, deterministic) no-cache vs cache ---
    ds_nocache = GeoFMDataset7A(tiles, is_train=False, cv_fold=0)
    ds_cache = GeoFMDataset7A(tiles, is_train=False, cv_fold=0,
                              cache_dir=CACHE_DIR, rebuild_cache=True)

    max_abs = {}
    for k in ("alpha_earth", "tessera", "terramind_s1", "terramind_s2", "target"):
        a = ds_nocache[0][k].numpy()
        b = ds_cache[0][k].numpy()
        max_abs[k] = float(np.max(np.abs(a - b)))
    r["val_idx0_maxabs_diff"] = {k: round(v, 5) for k, v in max_abs.items()}
    r["val_shapes_ok"] = all(tuple(ds_cache[0][k].shape) == tuple(ds_nocache[0][k].shape)
                             for k in max_abs)

    # --- train cache builds + a batch loads finite ---
    ds_tr = GeoFMDataset7A(tiles, is_train=True, cv_fold=0,
                           cache_dir=CACHE_DIR, rebuild_cache=True)
    w = ds_tr.sampler_weights()
    loader = DataLoader(ds_tr, batch_size=32, num_workers=4,
                        sampler=WeightedRandomSampler(w, len(w), replacement=True),
                        drop_last=True)
    t0 = time.time()
    b = next(iter(loader))
    r["train_batch_shapes"] = {k: list(v.shape) for k, v in b.items()}
    r["train_batch_finite"] = bool(all(torch.isfinite(v).all() for v in b.values()))
    r["first_batch_sec"] = round(time.time() - t0, 2)

    json.dump(r, open("/tmp/cachetest.json", "w"), indent=2)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
