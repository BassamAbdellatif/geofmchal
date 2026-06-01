"""
7A — isolate the effect of each inference trick (TTA / binary-blend / per-channel
thresholds) on the geographic-CV validation set, where we have ground truth.

For each val tile we run the model TWICE (raw forward + D4 TTA forward) and cache
the 5-channel logits [fracB,fracV,fracW, binary, height]. Then we evaluate the
proxy (C=4.0) under every config combination from those cached logits — cheap,
no extra forward passes. This tells us, on val, whether TTA / blend / thresholds
each add or subtract proxy, before spending a platform submission slot.

    ./run_env.sh eval_inference.py --experiment 7A_geocv_baseline
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from core.dataset import find_multimodal_train_tiles, GeoFMDataset7A, HEIGHT_NORM_CONSTANT
from core.model import build_model
from predict import _apply_threshold_remap

DATA_ROOT = "/mnt/head/users/bassam/data/geofmdata/embed2heights/data"


def _resolve_runs_dir():
    head = "/mnt/head/users/bassam/data/geofmdata/runs"
    return head if os.path.isdir(head) else __import__("config").SHARED_RUNS_DIR


def _read_cv_fold(exp_dir):
    p = os.path.join(exp_dir, "training_params.txt")
    if os.path.exists(p):
        for line in open(p):
            if line.upper().startswith("CV_FOLD:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    return 0


def _tta_logits(model, batch):
    """D4 8-fold TTA -> averaged 5-channel raw logits (matches predict._tta_logits_7a)."""
    acc = None
    for k in range(4):
        for flip in (False, True):
            b = {}
            for key in ("alpha_earth", "tessera"):
                x = torch.rot90(batch[key], k, dims=[-2, -1])
                if flip:
                    x = torch.flip(x, dims=[-1])
                b[key] = x
            for key in ("terramind_s1", "terramind_s2"):
                grid = batch[key].reshape(batch[key].shape[0], 16, 16, -1)
                grid = torch.rot90(grid, k, dims=[1, 2])
                if flip:
                    grid = torch.flip(grid, dims=[2])
                b[key] = grid.reshape(batch[key].shape[0], 256, -1)
            out = model(b)
            logits = torch.cat([out["fraction"], out["binary"], out["height"]], dim=1)
            if flip:
                logits = torch.flip(logits, dims=[-1])
            if k > 0:
                logits = torch.rot90(logits, -k, dims=[-2, -1])
            acc = logits if acc is None else acc + logits
    return acc / 8.0


def _proxy_from_preds(frac_prob, height_norm, target, C=4.0):
    """frac_prob (N,3,H,W) probs; height_norm (N,H,W) normalised; target (N,4,H,W)."""
    inter = np.zeros(3); union = np.zeros(3)
    for c in range(3):
        p = frac_prob[:, c] > 0.5
        t = target[:, c] > 0.5
        inter[c] = np.logical_and(p, t).sum()
        union[c] = np.logical_or(p, t).sum()
    iou = inter / np.maximum(union, 1.0)
    ph = height_norm * HEIGHT_NORM_CONSTANT
    th = target[:, 3] * HEIGHT_NORM_CONSTANT
    sq = (ph - th) ** 2
    mb = target[:, 0] > 0
    mv = target[:, 1] > 0
    rmse_b = float(np.sqrt((sq * mb).sum() / max(mb.sum(), 1)))
    rmse_v = float(np.sqrt((sq * mv).sum() / max(mv.sum(), 1)))
    proxy = (0.25 * iou[0] + 0.15 * iou[1] + 0.15 * iou[2]
             + 0.25 * max(0.0, 1 - rmse_b / C) + 0.20 * max(0.0, 1 - rmse_v / C))
    return dict(iou_b=iou[0], iou_v=iou[1], iou_w=iou[2],
                rmse_b=rmse_b, rmse_v=rmse_v, proxy=proxy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--thresholds", default="thresholds.json",
                    help="Threshold JSON inside the experiment dir.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(_resolve_runs_dir(), args.experiment)
    cv_fold = _read_cv_fold(exp_dir)

    thr_path = os.path.join(exp_dir, args.thresholds)
    thresholds = json.load(open(thr_path)) if os.path.exists(thr_path) else None
    print(f"Experiment {args.experiment} | cv_fold={cv_fold} | thresholds={thresholds}")

    tiles = find_multimodal_train_tiles(DATA_ROOT)
    val_ds = GeoFMDataset7A(tiles, is_train=False, cv_fold=cv_fold)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"Val tiles: {len(val_ds)}")

    mp = os.path.join(exp_dir, "model_best.pth")
    if not os.path.exists(mp):
        mp = os.path.join(exp_dir, "model_last.pth")
    model, _ = build_model("dual_enc_dec_fusion", 64, 4)
    model = model.to(device); model.load_state_dict(torch.load(mp, map_location=device)); model.eval()
    print(f"Loaded {mp}")

    # Cache raw + TTA fraction/binary probs and height, plus targets (CPU numpy).
    raw_frac, raw_bin, raw_h = [], [], []
    tta_frac, tta_bin, tta_h = [], [], []
    tgts = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val forward (raw+TTA)"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            tgts.append(batch["target"].cpu().numpy())

            out = model(batch)
            raw_frac.append(torch.sigmoid(out["fraction"]).cpu().numpy())
            raw_bin.append(torch.sigmoid(out["binary"][:, 0]).cpu().numpy())
            raw_h.append(out["height"][:, 0].cpu().numpy())

            tl = _tta_logits(model, batch)
            tta_frac.append(torch.sigmoid(tl[:, :3]).cpu().numpy())
            tta_bin.append(torch.sigmoid(tl[:, 3]).cpu().numpy())
            tta_h.append(tl[:, 4].cpu().numpy())

    cat = lambda xs: np.concatenate(xs, axis=0)
    target = cat(tgts)
    cache = {
        "raw": (cat(raw_frac), cat(raw_bin), cat(raw_h)),
        "tta": (cat(tta_frac), cat(tta_bin), cat(tta_h)),
    }

    def build(frac, binr, blend, thr):
        f = frac.copy()
        if blend:
            m = binr > 0.5
            f[:, 0] = np.where(m, np.maximum(f[:, 0], binr), f[:, 0])
        if thr and thresholds:
            for c, key in [(0, "B"), (1, "V"), (2, "W")]:
                t = thresholds.get(key)
                if t is not None:
                    f[:, c] = _apply_threshold_remap(f[:, c], float(t))
        return f

    configs = [
        ("raw",                 "raw", False, False),
        ("+threshold",          "raw", False, True),
        ("+blend",              "raw", True,  False),
        ("+blend+threshold",    "raw", True,  True),
        ("TTA",                 "tta", False, False),
        ("TTA+blend+threshold", "tta", True,  True),
    ]
    print(f"\n{'config':<22} {'proxy':>7} {'IoU_B':>6} {'IoU_V':>6} {'IoU_W':>6} {'RMSE_B':>7} {'RMSE_V':>7}")
    results = {}
    for name, src, blend, thr in configs:
        frac, binr, h = cache[src]
        f = build(frac, binr, blend, thr)
        m = _proxy_from_preds(f, h, target)
        results[name] = m
        print(f"{name:<22} {m['proxy']:>7.4f} {m['iou_b']:>6.3f} {m['iou_v']:>6.3f} "
              f"{m['iou_w']:>6.3f} {m['rmse_b']:>7.3f} {m['rmse_v']:>7.3f}")

    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
              open(os.path.join(exp_dir, "eval_inference.json"), "w"), indent=2)
    print(f"\nWritten: {os.path.join(exp_dir, 'eval_inference.json')}")


if __name__ == "__main__":
    main()
