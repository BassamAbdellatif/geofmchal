"""Precision / Recall diagnostic for a trained 7A model (no training).

Runs a trained dual_enc_dec_fusion model over its geo-CV validation fold and
decomposes the BUILDING error into TP/FP/FN -> precision, recall, IoU_B. The
point is to learn *why* IoU_B is low: recall-limited (we miss buildings, FN) ->
Tversky/P1 is the fix; precision-limited (we over-paint, FP) -> softmax+'other'/P2.
Also reports veg/water and a building threshold sweep for context.

Usage:
  ./run_env.sh pr_diagnostic.py --experiment-name 7A_v1_base
  ./run_env.sh pr_diagnostic.py --experiment-name 7A_v1_base --cv-fold 1 --amp
"""
import os
import argparse
from contextlib import nullcontext
import torch
from torch.utils.data import DataLoader

from core.model import build_model
from core.dataset import find_multimodal_train_tiles, GeoFMDataset7A
import train as T  # reuse DATA_ROOT_7A, _runs_dir_7a


def load_params(exp_dir):
    p = {}
    with open(os.path.join(exp_dir, "training_params.txt")) as f:
        for line in f:
            if ":" in line and not line.startswith("Epoch"):
                k, _, v = line.partition(":")
                p[k.strip()] = v.strip()
    return p


def main():
    ap = argparse.ArgumentParser(description="Building precision/recall diagnostic for a 7A run.")
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--cv-fold", type=int, default=None, help="Override; default = the run's CV_FOLD.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = os.path.join(T._runs_dir_7a(), args.experiment_name)
    p = load_params(exp_dir)

    patch_names = [x.strip() for x in p.get("PATCH_INPUTS", "terramind_s1,terramind_s2").split(",") if x.strip()]
    fraction_head = p.get("FRACTION_HEAD", "sigmoid3")
    cv_fold = args.cv_fold if args.cv_fold is not None else int(p.get("CV_FOLD", "0"))
    cache_dir = p.get("CACHE_DIR", None)
    if cache_dir in ("None", ""):
        cache_dir = None
    use_thor = any(n.startswith("thor") for n in patch_names)

    model, _ = build_model(
        "dual_enc_dec_fusion", n_channels=64, n_classes=4,
        use_height_bridge=(p.get("NO_HEIGHT_BRIDGE", "False").lower() != "true"),
        patch_inputs=patch_names,
        patch_stem_version=p.get("PATCH_STEM_VERSION", "v1"),
        xattn_heads=int(p.get("XATTN_HEADS", "4")),
        patch_routing=p.get("PATCH_ROUTING", "sensor"),
        use_fraction_bridge=(p.get("USE_FRACTION_BRIDGE", "False").lower() == "true"),
        fraction_bridge_alpha=float(p.get("FRACTION_BRIDGE_ALPHA", "0.2")),
        fraction_head=fraction_head,
    )
    ckpt = os.path.join(exp_dir, "model_best.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(exp_dir, "model_last.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()
    print(f"loaded {ckpt}\n  patch={patch_names} head={fraction_head} cv_fold={cv_fold}")

    tiles = find_multimodal_train_tiles(T.DATA_ROOT_7A, use_thor=use_thor)
    val_ds = GeoFMDataset7A(tiles, is_train=False, cv_fold=cv_fold,
                            cache_dir=cache_dir, patch_inputs=patch_names)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"  val tiles (fold {cv_fold}): {len(val_ds)}")

    thr = [float(x) for x in args.thresholds.split(",")]
    names = {0: "BUILDING", 1: "veg", 2: "water"}
    # per-class TP/FP/FN at 0.5
    half = {c: torch.zeros(3, device=device) for c in range(3)}  # [TP,FP,FN]
    # building only: TP/FP/FN per threshold
    bsweep = {t: torch.zeros(3, device=device) for t in thr}

    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
        if (args.amp and device.type == "cuda") else nullcontext
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch["target"]
            with amp_ctx():
                pred = model.predict(batch).float()        # (B,4) = [B,V,W,height], activated
            for c in range(3):
                t = target[:, c] > 0.5
                pmap = pred[:, c] > 0.5
                half[c][0] += (pmap & t).sum()
                half[c][1] += (pmap & ~t).sum()
                half[c][2] += (~pmap & t).sum()
            tb = target[:, 0] > 0.5
            for t_ in thr:
                pmap = pred[:, 0] > t_
                bsweep[t_][0] += (pmap & tb).sum()
                bsweep[t_][1] += (pmap & ~tb).sum()
                bsweep[t_][2] += (~pmap & tb).sum()

    def prf(tp, fp, fn):
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        iou = tp / max(1.0, tp + fp + fn)
        return prec, rec, iou

    print(f"\n=== per-class @0.5 (fold {cv_fold}) ===")
    print(f"{'class':10s} {'precision':>9s} {'recall':>7s} {'IoU':>6s}   (TP / FP / FN)")
    for c in range(3):
        tp, fp, fn = [float(x) for x in half[c]]
        pr, rc, io = prf(tp, fp, fn)
        print(f"{names[c]:10s} {pr:9.3f} {rc:7.3f} {io:6.3f}   ({tp:.0f} / {fp:.0f} / {fn:.0f})")

    print(f"\n=== BUILDING threshold sweep ===")
    print(f"{'thr':>5s} {'precision':>9s} {'recall':>7s} {'IoU':>6s}")
    for t_ in thr:
        tp, fp, fn = [float(x) for x in bsweep[t_]]
        pr, rc, io = prf(tp, fp, fn)
        print(f"{t_:5.2f} {pr:9.3f} {rc:7.3f} {io:6.3f}")

    tp, fp, fn = [float(x) for x in half[0]]
    pr, rc, _ = prf(tp, fp, fn)
    verdict = ("RECALL-limited (FN-dominated) -> P1/Tversky(beta>alpha) is the lever; P2/softmax would hurt"
               if rc < pr else
               "PRECISION-limited (FP-dominated) -> P2/softmax+'other' is the lever; Tversky-beta would hurt")
    print(f"\n>> building precision={pr:.3f} recall={rc:.3f}  =>  {verdict}")


if __name__ == "__main__":
    main()
