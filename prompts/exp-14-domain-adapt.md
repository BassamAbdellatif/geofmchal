# Phase 14 — Domain Adaptation for Region/Year Transfer (Spec)

**Status:** design, ready to build. **Target:** the **train→test region/year gap** — the binding constraint
behind every "internal≠platform" surprise (f40/f43) and the explicit limitation of AlphaEarth height
mapping (arXiv 2602.17250: "distribution shifts… need to address bias for regional transferability").
Read `docs/literature_domain_adaptation.md`, `CLAUDE.md`, and `results.md` f40/f43 first.

**Base recipe:** `ce40` = `13_bins256_ce40_b24_f1` (fresh_extract, dual deep encoders + sym fusion,
softmax4+tversky, decoupled height wb1/wv3, **adaptive-bin height N=256 ce0.4**, batch24, AMP, no-gradnorm,
num-workers 4, `expandable_segments`). Phase-14 changes the **embedding augmentation strength** + the
**evaluation protocol**; everything else identical so deltas are clean.

## 1. The lever already half-exists
`core/dataset.py::augment_domain_shift` (line ~314) runs on **every training tile** (called ~line 645,
seeded per f25): per-channel **gain U[0.9,1.1]**, **offset U[−0.1,0.1]**, **channel dropout p=0.1** on the
pixel streams (gain+dropout on patch tokens). **But:** magnitudes are fixed/untuned, there's **no flag**,
it's **never been ablated**, and we've **never measured a no-aug baseline**. So `ce40` already includes aug
at "scale 1" — Phase 14 makes it *controllable and calibrated*, and finally validates it the right way.

## 2. Code changes (additive; `--domain-aug-scale 1.0` reproduces today's behavior, `0` = no-aug)
| file | change |
|------|--------|
| `core/dataset.py` | `augment_domain_shift(emb_dict, norm_stats, rng, scale=1.0)`: multiply the gain *deviation* (|g−1|), offset range, and dropout prob by `scale`; `scale=0` → return input unchanged (true no-aug). Add `domain_aug_scale` to `GeoFMDataset7A.__init__` (default 1.0), store it, pass at the call site. **Keep RNG seeded.** |
| `train.py` | flag `--domain-aug-scale` (float, default 1.0); thread into both train/val dataset builds (val gets scale 0 — never augment val); write `DOMAIN_AUG_SCALE` to params. |
| `predict.py` | none (aug is train-only). |
| (2nd wave) | optional `--mixstyle` (per-batch channel mean/std mixing) and `--domain-consistency W` (MSE between predictions on clean vs augmented embeddings); optional `--test-norm` in predict (per-tile instance-norm, *ablated*, off by default). |
- **Byte-identical at `--domain-aug-scale 1.0`** (matches all current runs). `scale 0` is the never-measured no-aug baseline.

## 3. The evaluation protocol IS the experiment — MULTI-FOLD
Single-fold val **cannot see** DA's benefit (it's cross-region by definition — this is why f40/f43 fooled
us). So judge by **held-out-region transfer across multiple geo-folds**: train on K−1 folds, hold out fold
N (`--cv-fold N`), and compare across N. **DA wins** = higher *mean held-out* score **and/or** smaller
*train→held-out gap* — a real platform predictor, unlike one fold. Do NOT screen DA on a single fold.

## 4. Screening plan
1. **Smoke:** `scale 0` leaves embeddings unchanged (assert max|Δ|=0); `scale 3` ≈ 3× jitter; finite loss;
   seeded → identical tiles across reruns. bs24 memory OK. **STOP & report.**
2. **Night-1 controlled 2×2 (4 nodes, full 40 ep):** scale ∈ {0, 3} × held-out fold ∈ {0, 2} — a clean
   A/B of no-aug vs strong-aug on **two independent held-out regions**. Reference points already in hand:
   `ce40` (scale 1, fold 1).
   - node1: `--cv-fold 0 --domain-aug-scale 0`   → `14_da0_f0`
   - node2: `--cv-fold 0 --domain-aug-scale 3`   → `14_da3_f0`
   - node3: `--cv-fold 2 --domain-aug-scale 0`   → `14_da0_f2`
   - head : `--cv-fold 2 --domain-aug-scale 3`   → `14_da3_f2`  *(head only if coolgpus/guardian up)*
   **Read:** does scale 3 beat scale 0 on **both** held-out folds (proxy + RMSE_V + IoU_B, and the
   train→val gap)? If yes on both → DA transfers → go to step 3. If mixed → try scale 2 / add MixStyle.
3. **Confirm:** best scale across **3 folds** (0,1,2); pick the setting with the best mean-held-out + gap;
   **submit to platform** (the real DA test — the gain may be platform-only, like binning's IoU_B).
4. **2nd wave (if step 2 positive):** MixStyle + consistency loss; U-Net++ skips (transferability, 2602.17250).

## 5. Success / fallback
- **Win:** scale>1 lifts mean held-out / shrinks the gap across folds → submit; expect platform gain where
  internal single-fold was flat.
- **Null (scale curve flat across folds):** the existing aug is already sufficient and the residual gap is
  embedding-information-limited → DA is not the lever; fall back to **multi-seed ensemble of `ce40`**
  (reliable small gain) and consolidate at best-honest-rank.

## 6. Constraints
- Additive; `--domain-aug-scale 1.0` byte-identical to current; **seeded** (don't regress f25); val never
  augmented; **no test-label peeking** (excludes threshold calibration — per-instance norm only if ablated).
- batch24, num-workers 4, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `--cache-dir` (each new
  `--cv-fold` builds its own fold cache on first epoch). Smoke before the campaign → STOP & report.
- Per-machine `guardian --start` (cooling + 82°C failsafe) before any node run — see `src/gpu_management.md`.

## 7. Night-1 launch (after the code change + smoke)
Common (ce40 recipe): `--model-type fresh_extract --patch-inputs terramind_s1 --patch-stem-version v2
--no-use-terramind --no-use-gradnorm --amp --fraction-head softmax4 --building-overlap tversky
--decouple-height --build-height-weight 1.0 --veg-height-weight 3.0 --batch-size 24 --epochs 40
--cache-dir /home/bassam/nvme_cache/cache7a --num-workers 4 --height-bins 256 --height-ce-weight 0.40`
plus per-node `--cv-fold {0|2} --domain-aug-scale {0|3} --experiment-name 14_da{0|3}_f{0|2}
--scratch-dir /home/bassam/nvme_cache/runs_local/<name>`, prefixed with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, piped to `tee train_<name>.log`.
