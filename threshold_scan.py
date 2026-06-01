"""
7A — per-channel threshold calibration for hard-IoU.

The platform thresholds abundance at 0.5 for IoU. Our sigmoid outputs are not
necessarily calibrated to that cut, so we scan candidate thresholds per fraction
channel (B, V, W) on the geographic-CV validation set and pick, for each channel,
the threshold that maximises hard binary IoU against (target > 0.5).

Single pass over the val set: for every candidate threshold we accumulate
intersection and union counts, so memory stays O(channels x grid), not O(tiles).

Writes <run_dir>/thresholds.json  ->  {"B": t_b, "V": t_v, "W": t_w}

Run AFTER training completes (loads model_best.pth; uses the GPU):
    ./run_env.sh threshold_scan.py --experiment 7A_geocv_baseline
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

from core.dataset import find_multimodal_train_tiles, GeoFMDataset7A
from core.model import build_model

DATA_ROOT = "/mnt/head/users/bassam/data/geofmdata/embed2heights/data"
CHANNELS = {0: "B", 1: "V", 2: "W"}


def _resolve_runs_dir():
    head = "/mnt/head/users/bassam/data/geofmdata/runs"
    if os.path.isdir(head):
        return head
    import config
    return config.SHARED_RUNS_DIR


def _read_cv_fold(exp_dir):
    """Read CV_FOLD from training_params.txt (default 0)."""
    params_path = os.path.join(exp_dir, "training_params.txt")
    fold = 0
    if os.path.exists(params_path):
        with open(params_path) as f:
            for line in f:
                if line.upper().startswith("CV_FOLD:"):
                    try:
                        fold = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
    return fold


def main():
    parser = argparse.ArgumentParser(description="7A per-channel threshold calibration")
    parser.add_argument("--experiment", required=True, help="Experiment name under the runs dir.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--grid-min", type=float, default=0.05)
    parser.add_argument("--grid-max", type=float, default=0.95)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--output", type=str, default="thresholds.json",
                        help="Output filename (written inside the experiment dir).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(_resolve_runs_dir(), args.experiment)
    if not os.path.isdir(exp_dir):
        raise SystemExit(f"Experiment dir not found: {exp_dir}")

    cv_fold = _read_cv_fold(exp_dir)
    print(f"Experiment: {args.experiment} | cv_fold={cv_fold} | device={device}")

    # Validation split = the held-out geographic fold.
    tiles = find_multimodal_train_tiles(DATA_ROOT)
    val_ds = GeoFMDataset7A(tiles, is_train=False, cv_fold=cv_fold)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"Val tiles: {len(val_ds)}")

    model_path = os.path.join(exp_dir, "model_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(exp_dir, "model_last.pth")
    model, _ = build_model("dual_enc_dec_fusion", n_channels=64, n_classes=4)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded {model_path}")

    grid = np.arange(args.grid_min, args.grid_max + 1e-9, args.grid_step)
    n_t = len(grid)
    grid_t = torch.tensor(grid, device=device).view(n_t, 1)  # (T,1)

    # Accumulators: (3 channels, T thresholds)
    inter = torch.zeros(3, n_t, device=device)
    union = torch.zeros(3, n_t, device=device)
    # Also track IoU at the default 0.5 for a baseline comparison.
    inter05 = torch.zeros(3, device=device)
    union05 = torch.zeros(3, device=device)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Threshold scan"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch["target"]
            out = model(batch)
            prob = torch.sigmoid(out["fraction"])           # (B,3,H,W)

            for c in range(3):
                p = prob[:, c].reshape(-1)                   # (N,)
                t = (target[:, c].reshape(-1) > 0.5)         # bool (N,)
                # Vectorised over the threshold grid.
                pred_t = p.unsqueeze(0) > grid_t             # (T, N) bool
                tt = t.unsqueeze(0)                          # (1, N)
                inter[c] += (pred_t & tt).sum(dim=1).float()
                union[c] += (pred_t | tt).sum(dim=1).float()
                p05 = p > 0.5
                inter05[c] += (p05 & t).sum().float()
                union05[c] += (p05 | t).sum().float()

    iou = (inter / union.clamp_min(1.0)).cpu().numpy()       # (3, T)
    iou05 = (inter05 / union05.clamp_min(1.0)).cpu().numpy() # (3,)

    thresholds = {}
    print("\nPer-channel scan (IoU at best vs at 0.5):")
    for c in range(3):
        best_i = int(np.argmax(iou[c]))
        best_t = float(grid[best_i])
        thresholds[CHANNELS[c]] = round(best_t, 4)
        print(f"  {CHANNELS[c]}: best_t={best_t:.2f} IoU={iou[c, best_i]:.4f}  "
              f"(IoU@0.5={iou05[c]:.4f}, gain={iou[c, best_i]-iou05[c]:+.4f})")

    out_path = os.path.join(exp_dir, args.output)
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nWritten: {out_path}\n  {thresholds}")


if __name__ == "__main__":
    main()
