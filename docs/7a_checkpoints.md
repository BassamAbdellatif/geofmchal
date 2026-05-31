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

## Pending

- **Phase 3** — `DualPathLoss` (MAE + soft-Dice k=5 fractions; Huber δ=0.5 masked
  + 0.1 tail height; BCE + soft-Dice binary), GradNorm (α=1.5), train.py 7A
  training branch (proxy C=4.0, best-ckpt by proxy on geo-CV val), predict.py
  dict-output + D4 TTA + `--blend-binary` + `--threshold-config`. Smoke + mini
  runs.
- **Phase 4** — 60-epoch baseline, threshold scan, TTA predict.
- **Phase 5** — ablations (incl. revisit coverage metric; THOR; stem
  downsample-to-32 vs full-res).
