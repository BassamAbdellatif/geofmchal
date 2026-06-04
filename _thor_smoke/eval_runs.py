"""Evaluate finished 7A runs on the fold-0 val set (param-aware reconstruction),
report the standard hard-0.5 metrics + proxy, and sweep the WATER threshold to
test whether IoU_W=0 is a thresholding artifact or a genuine no-signal.
Usage: eval_runs.py <exp1> [exp2 ...]
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import DATA_ROOT_7A, evaluate_7a
from core.dataset import find_multimodal_train_tiles, GeoFMDataset7A, HEIGHT_NORM_CONSTANT
from core.losses import DualPathLoss
from core.model import build_model
import predict as P

CACHE = "/home/bassam/nvme_cache/cache7a"
RUNS = "/mnt/head/users/bassam/data/geofmdata/runs"
WT = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def load_model(exp, device):
    params = P.load_experiment_params(os.path.join(RUNS, exp))
    patch = [p.strip() for p in params.get("PATCH_INPUTS", "terramind_s1,terramind_s2").split(",") if p.strip()]
    stem = params.get("PATCH_STEM_VERSION", "v1").strip()
    heads = int(params.get("XATTN_HEADS", "4"))
    bridge = params.get("NO_HEIGHT_BRIDGE", "False").strip().lower() != "true"
    routing = params.get("PATCH_ROUTING", "sensor").strip()
    fbridge = params.get("USE_FRACTION_BRIDGE", "False").strip().lower() == "true"
    m, _ = build_model("dual_enc_dec_fusion", 64, 4, use_height_bridge=bridge,
                       patch_inputs=patch, patch_stem_version=stem, xattn_heads=heads,
                       patch_routing=routing, use_fraction_bridge=fbridge)
    sd = torch.load(os.path.join(RUNS, exp, "model_best.pth"), map_location=device)
    m.load_state_dict(P._remap_legacy_7a_state_dict(sd, m))
    return m.to(device).eval(), patch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    crit = DualPathLoss(veg_height_boost=0.0, use_binary=True).to(device)
    for exp in sys.argv[1:]:
        m, patch = load_model(exp, device)
        tiles = find_multimodal_train_tiles(DATA_ROOT_7A, use_thor=any("thor" in p for p in patch))
        val = GeoFMDataset7A(tiles, is_train=False, cv_fold=0, cache_dir=CACHE, patch_inputs=patch)
        loader = DataLoader(val, batch_size=24, shuffle=False, num_workers=4)
        r = evaluate_7a(m, loader, crit, device)
        # water-threshold sweep
        wi = torch.zeros(len(WT)); wu = torch.zeros(len(WT))
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}
                w = torch.sigmoid(m(b)["fraction"])[:, 2]
                gt = b["target"][:, 2] > 0.5
                for i, t in enumerate(WT):
                    p = w > t
                    wi[i] += (p & gt).sum().cpu(); wu[i] += (p | gt).sum().cpu()
        sweep = "  ".join(f"{t:.2f}:{(wi[i]/wu[i]).item():.3f}" for i, t in enumerate(WT))
        print(f"\n[{exp}] patch={patch}")
        print(f"  proxy={r['proxy']:.4f}  IoU B/V/W={r['iou_b']:.3f}/{r['iou_v']:.3f}/{r['iou_w']:.3f}  "
              f"RMSE B/V={r['rmse_b']:.2f}/{r['rmse_v']:.2f}")
        print(f"  IoU_W vs water-threshold:  {sweep}")


if __name__ == "__main__":
    main()
