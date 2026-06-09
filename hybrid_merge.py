"""Hybrid merge: take building/veg/water (channels 0-2) from one model's test
predictions and the height channel (3) from another, writing a merged prediction
set ready for package.py. Used to combine 7A/sm4tv's strong fractions with 2A's
sub-3.9 m height (which clears the platform RMSE_V cliff).

Deterministic: a fixed channel swap over two seeded models' raw predictions — no
val-fitting / no calibration. Fully reproducible given both checkpoints + inputs.

Usage:
  ./run_env.sh hybrid_merge.py --fraction-exp 9_sm4tv_final          # height from 2A_vegboost
  ./run_env.sh package.py --experiment-name 9_sm4tv_final_hybrid     # then package + submit
"""
import os
import glob
import shutil
import argparse
import numpy as np
import train as T  # reuse _runs_dir_7a


def main():
    ap = argparse.ArgumentParser(description="Hybrid merge of fraction-model B/V/W + height-model height.")
    ap.add_argument("--fraction-exp", required=True,
                    help="Experiment providing channels 0-2 (B/V/W), e.g. 9_sm4tv_final.")
    ap.add_argument("--height-exp", default="2A_vegboost",
                    help="Experiment providing channel 3 (height). Default 2A_vegboost (sub-3.9m).")
    ap.add_argument("--out-exp", default=None,
                    help="Output experiment name. Default <fraction-exp>_hybrid.")
    ap.add_argument("--tta", action="store_true", help="Use predictions_tta/ instead of predictions/.")
    args = ap.parse_args()

    runs = T._runs_dir_7a()
    sub = "predictions_tta" if args.tta else "predictions"
    fdir = os.path.join(runs, args.fraction_exp, sub)
    hdir = os.path.join(runs, args.height_exp, sub)
    out_exp = args.out_exp or f"{args.fraction_exp}_hybrid"
    odir = os.path.join(runs, out_exp, sub)

    if not os.path.isdir(fdir):
        raise SystemExit(f"fraction predictions not found: {fdir} (run predict.py on {args.fraction_exp} first)")
    if not os.path.isdir(hdir):
        raise SystemExit(f"height predictions not found: {hdir} (run predict.py on {args.height_exp} first)")
    os.makedirs(odir, exist_ok=True)

    ffiles = sorted(glob.glob(os.path.join(fdir, "*.npy")))
    if not ffiles:
        raise SystemExit(f"no .npy in {fdir}")

    n_ok = n_miss = 0
    for fp in ffiles:
        name = os.path.basename(fp)
        frac = np.load(fp)
        hp = os.path.join(hdir, name)
        if not os.path.exists(hp):
            n_miss += 1
            print(f"  WARN: no height match for {name}; keeping fraction-model height")
            np.save(os.path.join(odir, name), frac.astype(np.float32))
            continue
        height = np.load(hp)
        if frac.shape != height.shape:
            raise SystemExit(f"shape mismatch {name}: frac {frac.shape} vs height {height.shape}")
        merged = frac.copy()
        merged[3] = height[3]            # graft the height channel only
        np.save(os.path.join(odir, name), merged.astype(np.float32))
        n_ok += 1

    # carry the fraction model's params so predict/package metadata exists
    src_params = os.path.join(runs, args.fraction_exp, "training_params.txt")
    if os.path.exists(src_params):
        shutil.copy2(src_params, os.path.join(runs, out_exp, "training_params.txt"))

    print(f"\nmerged {n_ok} tiles ({n_miss} without a height match) -> {odir}")
    print(f"  B/V/W from: {args.fraction_exp}")
    print(f"  height   from: {args.height_exp}")
    print(f"next: ./run_env.sh package.py --experiment-name {out_exp}{' --tta' if args.tta else ''}")


if __name__ == "__main__":
    main()
