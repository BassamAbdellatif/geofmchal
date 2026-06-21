"""Phase 18 — generate pseudo-labels for target tiles from a trained model (self-training).
Saves 4-band tifs (building%, veg%, water% in [0,1]; height in METERS) matching the real-label
format, so a pseudo-aware dataset can load them exactly like real labels.

  ./run_env.sh make_pseudo_labels.py --source-exp 17_unetpp_ema_f1 --folds 1 \
      --out-dir /home/bassam/nvme_cache/pseudo/17unetpp_f1      # screening: pseudo-label fold 1
  ./run_env.sh make_pseudo_labels.py --source-exp <exp> --test --out-dir <dir>   # real: pseudo-label test
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, rasterio
from core.model import build_model
from core.dataset import (GeoFMDataset7A, find_multimodal_train_tiles,
                          find_multimodal_test_tiles, HEIGHT_NORM_CONSTANT)

DATA_ROOT = "/mnt/head/users/bassam/data/geofmdata/embed2heights/data"
RUNS = "/mnt/head/users/bassam/data/geofmdata/runs"


def read_params(path):
    p = {}
    with open(path) as f:
        for line in f:
            if line.startswith("Epoch"):
                continue
            k, sep, v = line.partition(":")
            if sep:
                p[k.strip().upper()] = v.strip()
    return p


def build_source(exp, device):
    d = os.path.join(RUNS, exp)
    p = read_params(os.path.join(d, "training_params.txt"))
    pix = [x.strip() for x in p.get("PIXEL_INPUTS", "alpha_earth,tessera").split(",") if x.strip()]
    patch = [x.strip() for x in p.get("PATCH_INPUTS", "terramind_s1").split(",") if x.strip()]
    ew = tuple(int(x) for x in p.get("ENC_WIDTHS", "96,160,256,384,512").split(","))
    ebs = p.get("ENC_BLOCKS", "1").strip(); eb = [int(x) for x in ebs.split(",")] if "," in ebs else int(ebs)
    eds = p.get("ENC_DILATIONS", "1").strip(); ed = [int(x) for x in eds.split(",")] if "," in eds else int(eds)
    model, _ = build_model(
        p.get("MODEL_TYPE", "flexnet").strip(), 64, 4,
        pixel_inputs=pix, patch_inputs=patch, patch_fusion=p.get("PATCH_FUSION", "none").strip(),
        enc_widths=ew, enc_blocks=eb, enc_block=p.get("ENC_BLOCK", "double").strip(),
        enc_dilations=ed, decoder=p.get("DECODER", "unet").strip(), dec_blocks=int(p.get("DEC_BLOCKS", "1")),
        height_mode=p.get("HEIGHT_MODE", "shared").strip(), fraction_head=p.get("FRACTION_HEAD", "softmax4").strip(),
        height_bins=int(p.get("HEIGHT_BINS", "0")), patch_stem_version=p.get("PATCH_STEM_VERSION", "v2").strip())
    ck = os.path.join(d, "model_best.pth")
    model.load_state_dict(torch.load(ck, map_location="cpu"), strict=True)
    return model.to(device).eval(), patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-exp", required=True)
    ap.add_argument("--folds", default=None, help="comma geo-folds to pseudo-label (train tiles)")
    ap.add_argument("--test", action="store_true", help="pseudo-label the test set instead")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default="/home/bassam/nvme_cache/cache7a")
    ap.add_argument("--limit", type=int, default=0, help="process only first N tiles (smoke)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, patch = build_source(a.source_exp, dev)
    use_thor = any(x.startswith("thor") for x in patch)
    if a.test:
        tiles = find_multimodal_test_tiles(DATA_ROOT, use_thor=use_thor); inc = None
    else:
        tiles = find_multimodal_train_tiles(DATA_ROOT, use_thor=use_thor)
        inc = [int(x) for x in a.folds.split(",")]
    ds = GeoFMDataset7A(tiles, is_train=False, include_folds=inc, cache_dir=a.cache_dir, patch_inputs=patch)
    N = len(ds) if not a.limit else min(a.limit, len(ds))
    print(f"pseudo-labeling {N}/{len(ds)} tiles -> {a.out_dir}")
    for i in range(N):
        item = ds[i]
        batch = {k: v.unsqueeze(0).to(dev) for k, v in item.items() if k != "target"}
        with torch.no_grad():
            pred = model.predict(batch)[0].float().cpu().numpy()      # (4,256,256): fracs[0,1], height norm
        pred[3] = pred[3] * HEIGHT_NORM_CONSTANT                       # -> meters (match real labels)
        with rasterio.open(ds.tiles[i]["label_path"]) as src:
            prof = src.profile
        prof.update(count=4, dtype="float32")
        out = os.path.join(a.out_dir, os.path.basename(ds.tiles[i]["label_path"]))
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(pred.astype(np.float32))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N}")
    print(f"done: {N} pseudo-labels in {a.out_dir}")


if __name__ == "__main__":
    main()
