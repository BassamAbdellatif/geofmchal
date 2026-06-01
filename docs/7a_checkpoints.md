# 7A — Checkpoint Reports (branch `exp-7-clean-slate`)

Per-phase checkpoint reports for the 7A decoupled dual-encoder / dual-decoder
build. Each phase stops here for human review before the next begins. Spec:
`prompts/exp-7-clean-slate.md`. Data paths: real root is
`/mnt/head/users/bassam/data/geofmdata/embed2heights/data` (config.py paths do
not exist on this node; scripts self-resolve).

---

## Checkpoint 1 — Phase 1: Data Infrastructure ✅ (committed `77e28f8`)

### Embedding counts
2024 tiles each for alpha_earth, tessera, terramind_s1, terramind_s2, thor_s1,
thor_s2 + labels. **2024 fully-matched multimodal tiles** → train **1510** /
val **514** (geographic fold 0 held out).

### Geographic CV folds (`data/geo_folds.json`)
No real lat/lon exists (CRS=None, catalog bboxes all zero). KMeans K=5 runs over
the 2-letter region grid codes in filenames (France grid; letters → ordinals as
2D coords).

| Fold | Tiles | Centroid (row,col) |
|------|-------|--------------------|
| 0 (val) | 514 | (7.9, 14.1) |
| 1 | 396 | (7.4, 7.2) |
| 2 | 442 | (14.9, 13.9) |
| 3 | 378 | (9.2, 2.7) |
| 4 | 294 | (15.9, 5.6) |

### Normalisation stats (`data/norm_stats.json`, training folds only)
| Modality | ch | mean range | std range |
|----------|-----|-----------|-----------|
| alpha_earth | 64 | [-55.2, 47.9] | [16.0, 43.3] |
| tessera | 128 | [-4.2, 5.0] | [1.3, 3.6] |
| terramind_s1 | 768 | [-9.0, 4.9] | [0.09, 1.66] |
| terramind_s2 | 768 | [-10.3, 9.2] | [0.11, 7.36] |

Post-normalisation a real batch reads alpha mean≈0/std≈1, tm_s1 mean≈0/std≈0.9.

### Sample batch (shapes match spec)
`alpha_earth (8,64,256,256)`, `tessera (8,128,256,256)`,
`terramind_s1 (8,256,768)`, `terramind_s2 (8,256,768)`, `target (8,4,256,256)`.
Target frac ∈ [0,1], height ∈ [0,1.5]. 30 batches iterated with no errors; all
tensors finite; train aug differs across reads, val deterministic.

### Four data bugs found & fixed by verification
1. **Corrupt tile** `tessera_emb_1753_QE.tif` carries a `6.59e36` sentinel (not
   inf/nan, so `nan_to_num` misses it) that poisoned global stats →
   added `_sanitize` (zeros `|x|>1e6`) in dataset and stats script.
2. **Patch-token norm broadcast** — tokens are `(256,768)` feature-last; fixed
   `_apply_norm` with a `channel_axis` parameter.
3. **Stratified strata imbalance** — switched to tertiles of positive coverage
   → balanced 19 / 497 / 497 / 497.
4. **Size mismatch** — 258/2024 pixel+label tiles are 255 px in a dim → added
   reflect `_pad_to(256)`.

### Coverage-metric decision (human-chosen: continuous + retuned)
Coverage = `mean(building_frac) + mean(water_frac)` (unthresholded). Strata =
zero-coverage + tertiles of the positive part; weights [1.0, 1.5, 2.0, 3.0].
Sampler enriches mean coverage ~1.3–1.6×.

### Files
New: `build_geo_split.py`, `compute_norm_stats.py`, `data/geo_folds.json`,
`data/norm_stats.json`. Modified: `core/dataset.py` (GeoFMDataset7A + helpers,
additive), `train.py` (flags + worker_init_fn + import fix).

---

## Checkpoint 2 — Phase 2: Model ✅ (staged, awaiting review)

`DualEncDualDecFusion` registered as `dual_enc_dec_fusion` in `core/model.py`;
existing classes untouched (additive only).

### Parameter count: **18.66M total**
| Block | Params (M) |
|-------|-----------|
| alpha_stem | 0.14 |
| alpha_encoder | 4.06 |
| tessera_encoder | 4.06 |
| fraction_decoder | 4.57 |
| height_decoder | 4.86 |
| s1_stem / s2_stem | 0.39 each |

(No 2A ratio available — `YNetAttentionFusedDecoder` is not on this branch.)

### Forward pass (batch=2)
Output shapes match spec: `fraction (2,3,256,256)`, `height (2,1,256,256)`,
`binary (2,1,256,256)`. All finite. `predict()` → `(B,4,256,256)`, fraction
channels in sigmoid range.

### Backward pass — both encoders receive gradient
| Parameter | grad norm |
|-----------|-----------|
| alpha_stem first conv | 0.099 |
| tessera_stem first conv | 0.113 |
| alpha_encoder down1 | 0.360 |
| tessera_encoder down1 | 0.294 |

None near zero — dual-encoder paths and the GradScale(0.2) height bridge all
flow. Confirmed on real dataloader batches: loss finite, predict frac ∈ [0.19,
0.83].

### L0 skip wiring (verified by request, static + runtime hooks)
The full-resolution alpha-stem output (96ch @ 256×256) is **not** discarded:
- `UNetEncoderHalf.forward` returns `skips = [l0, l1, l2, l3]`, `l0 = stem_out`.
- `FractionDecoder` consumes it: `x = self.up4(x, skips[0])`.
- Runtime hook on `up4`: skip in = `[1,96,256,256]`, concat fuse in =
  `[1,192,256,256]`, out = `[1,96,256,256]` → feeds frac + binary heads.

### Flags for review
1. **GPU memory:** peak at batch=2 is 2.67 GB → linear extrapolation to batch=32
   ≈ **42.7 GB** (fits 48 GB, but tight; real usage typically below the linear
   estimate). Watch in the Phase 3 smoke test; spec fallbacks apply if it OOMs.
2. **Design deviation from the spec diagram:** the spec said stems downsample to
   96ch @ 32×32. Implemented instead as **full-resolution 96ch @ 256** stems,
   letting the 5-level encoder do all downsampling. Rationale (in the model.py
   banner): keeps encoder + 16×16 bottleneck internally consistent and preserves
   a high-res L0 skip for building detail (6A lesson). **Human-confirmed to keep
   the full-res stem.**

### Files
Modified: `core/model.py` (DualEncDualDecFusion + submodules, additive; build_model
dispatch).

---

## Checkpoint 3 — Phase 3: Loss + GradNorm + Training/Inference ✅ (awaiting review)

### `DualPathLoss` + `GradNormBalancer` (core/losses.py, additive)
- **Fractions:** MAE (calibration) + soft-Dice at `sigmoid(5·(logit−0.5))` vs
  `(target>0.5)` — continuous relaxation of hard IoU at the 0.5 cut. No SSIM/GDL.
- **Height:** Huber (δ=0.5) on mask `(B>0)|(V>0)` + 0.1× on the complement.
- **Binary:** BCE + soft-Dice on `(building_frac>0.5)`.
- **GradNorm** (α=1.5, separate AdamW lr=0.025): per-task grad-norm measured
  (detached) through a reference encoder param; `step()` is called *before* the
  model `backward()` (retain_graph, no second-order graph); weights renormalised
  to sum=n. Standalone test: losses finite, both encoders receive gradient,
  weights evolve (height down-weighted as it converges fastest), sum stays 3.0.

### `train.py` — `train_7a()` + `evaluate_7a()`
Self-contained dispatch (`if MODEL_TYPE=="dual_enc_dec_fusion": return train_7a`).
Geo-CV split, stratified sampler, GradNorm (flag-toggled), Cosine LR, grad-clip
1.0, best-checkpoint by proxy, writes `training_params.txt` + `loss_curve.png`.
`evaluate_7a`: hard-IoU@0.5, masked RMSE (m), proxy (C=4.0), per-task losses.

### `predict.py` — `predict_7a()` + helpers
Dict output; `--tta` (D4 8-fold, pre-sigmoid logit averaging incl. patch grid
rot/flip); `--blend-binary` (`B = max(fracB, binaryB)` where binary>0.5);
`--threshold-config` (piecewise-linear remap so a calibrated threshold → 0.5).
Added `_resolve_runs_dir()` so predict reads the same head-node runs dir train uses.

### Runtime verification (real, reproduced)
| Run | Result |
|-----|--------|
| **smoke** (b4, 1 ep, 2 batches) | completes exit 0; pipeline OK end-to-end |
| **mini** (b32, 3 ep, geo-CV fold 0) | proxy **0.209 → 0.247 → 0.272**, monotonic; ≫ spec bar 0.18 |
| predict plain | 946 test tiles, (4,256,256), frac [0.002, 0.996], height [0, 18.3 m], finite |
| predict TTA+blend+thresholds | path verified (valid finite output); full 946 run ≈110 min is a Phase-4 step |

Mini per-epoch (reproduced run):
| Ep | proxy | IoU B/V/W | RMSE B/V (m) | GradNorm w[f/h/b] |
|----|-------|-----------|--------------|-------------------|
| 1 | 0.209 | 0.069/0.661/0.301 | 5.91/5.72 | 1.07/0.79/1.14 |
| 2 | 0.247 | 0.102/0.701/0.346 | 4.93/5.18 | 1.16/0.62/1.22 |
| 3 | 0.272 | 0.127/0.726/0.379 | 4.41/4.79 | 1.31/0.49/1.20 |

RMSE is still above the 3.9 m scoring floor at epoch 3 — expected, height
converges slowly and only 3 epochs ran. The 60-epoch baseline (Phase 4) is where
height has time to matter. GradNorm steadily up-weights fractions (1.07→1.31) and
down-weights height (0.79→0.49) as height's loss falls fastest.

### ⚠️ GPU memory — real finding (corrects the Phase-2 estimate)
Batch 32 needs **~40 GB**. The first mini attempt **OOM'd** because a 6.48 GB
co-resident process was sharing the 47.4 GB card. On a free GPU the rerun used
40.1 GB and trained fine (run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
So batch 32 fits a 48 GB GPU **only with the card to itself**. Phase 4 must use an
exclusive GPU, or apply the spec OOM fallback (drop the 32×32 patch injection,
then reduce batch).

### Files
Modified (additive / new branches; legacy paths untouched):
`core/losses.py`, `train.py`, `predict.py`.

---

## Checkpoint 4 — Phase 4: Baseline + Inference + first platform submission

### Training (`7A_geocv_baseline`, 60 epochs, geo-CV fold 0, batch 32)
- **Best proxy 0.3946 at epoch 29** (`model_best.pth`); plateaued ~0.39 from
  ~epoch 25 onward, epoch 60 proxy 0.3912 — stable, not a spike.
- Best-epoch internal val: IoU_B 0.200, IoU_V 0.812, IoU_W 0.687, RMSE_B 2.09 m,
  RMSE_V 4.00 m. Total train time 8.7 h on one 48 GB GPU (~40 GB used).
- GradNorm behaved as designed: height weight high early (drives RMSE down), then
  capacity shifted to the binary head late (w_b 0.91→1.45) to fight buildings.

### Threshold calibration (`threshold_scan.py`, geo-CV val)
| Channel | best t | IoU gain vs 0.5 |
|---------|--------|-----------------|
| B | 0.20 | +0.037 (0.200 → 0.237) |
| V | 0.60 | +0.000 |
| W | 0.55 | +0.000 |

Buildings are systematically under-predicted (sigmoid mass below 0.5), so a 0.20
cut recovers real IoU_B. V and W are already well-calibrated.

### Inference ablation (`eval_inference.py`, geo-CV val, proxy C=4.0)
| config | proxy | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V |
|--------|-------|-------|-------|-------|--------|--------|
| raw | 0.3947 | 0.200 | 0.812 | 0.687 | 2.087 | 3.996 |
| +threshold | 0.4040 | 0.237 | 0.812 | 0.688 | 2.087 | 3.996 |
| +blend | 0.4048 | 0.241 | 0.812 | 0.687 | 2.087 | 3.996 |
| +blend+threshold | 0.4042 | 0.238 | 0.812 | 0.688 | 2.087 | 3.996 |
| TTA | 0.3973 | 0.200 | 0.812 | 0.689 | 2.067 | 3.975 |
| **TTA+blend+threshold** | **0.4072** | 0.240 | 0.813 | 0.689 | 2.067 | 3.975 |

The submitted config (TTA+blend+threshold) is the best on val (+0.0125 over raw).
Two notes: **blend and threshold don't stack** (both lower the B bar; together
overshoot slightly); **TTA is small but free** (RMSE ticks down).

### Platform submission: `7A_geocv_baseline_tta` = **0.3575** (vs 2A_vegboost 0.3721)
| Metric | 2A_vegboost | 7A_tta | Δ | weight |
|--------|-------------|--------|-----|--------|
| IoU_B | 0.3394 | 0.3403 | +0.001 | 0.25 |
| IoU_V | 0.7649 | **0.7981** | **+0.033** | 0.15 |
| IoU_W | 0.3695 | **0.4138** | **+0.044** | 0.15 |
| RMSE_B | 2.27 m | 2.35 m | −0.08 | 0.25 |
| RMSE_V | 3.74 m | **4.09 m** | **−0.35** | 0.20 |
| **score** | **0.3721** | **0.3575** | **−0.0146** | |

**Verdict: 7A genuinely improves segmentation (IoU_V +0.033, IoU_W +0.044, IoU_B
tied) but loses the aggregate on one cliff.** RMSE_V = 4.09 m is **above the 3.9 m
scoring floor**, so the vegetation-height term contributes **exactly 0**
(`0.20·max(0,1−4.09/3.9)=0`); 2A's 3.74 m squeaked under and earned ~0.008. That
term swing plus slightly-worse RMSE_B is the whole −0.015 gap.

### Diagnosis (drives Phase 5)
1. **RMSE_V over the cliff** is the #1 problem and is **training-side, not
   inference**. Smoking gun: `DualPathLoss` has **no vegetation-height boost**,
   whereas 2A's loss did (its `veg_height_boost`, +0.006 platform, kept RMSE_V
   under the floor). Dropped in the clean rebuild — prime suspect.
2. **IoU_W collapses train→test** (val 0.689 → platform 0.414): water overfits to
   the training regions. Geo-CV is exposing real domain-shift on water.
3. The inference stack is already optimal — do not change it; do not re-submit
   the same model.

### Submission filename bug (fixed)
First submission rejected: "additional files not required". Cause: `predict_7a`
named outputs from `tile['core_id']`, which strips the year → `3001_BE.npy`. The
platform requires `NNNN_AA_YYYY` (`3001_BE_2023`). Fixed `predict.py` to derive
the stem via `extract_core_id_from_filename(tile['alpha_path'])` (keeps year);
verified 946/946 match, unique. Existing predictions were renamed in place
(content byte-identical) to avoid a 110-min re-predict.

### Files
- New: `threshold_scan.py`, `eval_inference.py`.
- Fixed: `predict.py` (output naming), `package.py` + `uploader/submit.py`
  (`_resolve_runs_dir()` for the head-node runs path).
- `.gitignore`: added `uploader/cookies.json`, `*.png`, `nohup.out`.

---

## Phase 5 — Plan (post-baseline ablations)

Priority is set by the platform breakdown: recover the **RMSE_V cliff** first
(biggest score lever), then **water generalization**, then the structural claims.

### P5.1 — Vegetation-height boost in the loss  ⟵ HIGHEST VALUE
RMSE_V 4.09 m → under 3.9 m would flip a 0.0 term into ~+0.01–0.02 platform.
Add a weighted height term where vegetation is present, mirroring 2A's win:
```
L_height += w_veg · Huber(pred,target,δ=0.5) on (veg_frac > 0.1)
```
Start `w_veg ≈ 2–3`. Also consider de-weighting GradNorm's freedom to starve
height late (epoch-29 best had height weight already declining). Re-train fold 0,
re-evaluate; this is a `DualPathLoss` change only (additive — keep the old class
path intact for A/B). Expected: RMSE_V back under floor, small RMSE_B help.

### P5.2 — Water generalization (IoU_W 0.69 val → 0.41 test)
The largest train→test collapse. Options, cheapest first:
- Stronger domain-shift augmentation on the channels water relies on.
- Check whether water tiles cluster in particular geo folds (fold imbalance).
- A dedicated water-aware term or light decoder branch (only if the above
  doesn't recover it).

### P5.3 — Structural ablations (the 6 spec claims), each vs the P5.1 baseline
1. THOR_s1→height, THOR_s2→fraction (added alongside TerraMind).
2. Disable encoder modality split (concat α+τ, single encoder).
3. Disable decoder task split (single shared decoder).
4. Both S1 and S2 to both decoders.
5. Patches at encoder bottleneck instead of decoder skips.
6. Drop the auxiliary binary B head.

### P5.4 — Robustness / cheap wins
- Stem deviation: spec's downsample-to-32 stem vs the full-res 96ch@256 we built.
- Coverage metric: thresholded (spec) vs continuous (current) for the sampler.
- 3-seed ensemble of the best config for the final submission.
- Quiet the cosmetic `config.py` "Local NVMe not found" warning on the head node.

### Submission discipline
12 h limit; human decides each one. Next submission should be a model that
clears RMSE_V < 3.9 m on val (P5.1), not a re-run of the current one.
