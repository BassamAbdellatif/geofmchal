# GeoFM Challenge — Experiment Results & Findings

## Platform Submissions (Ground Truth)

| # | Experiment | Platform Score | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Notes |
|---|-----------|---------------|-------|-------|-------|--------|--------|-------|
| 1 | `2A_alpha_ts1_ts2_nologits` | **0.366039** | 0.3394 | 0.7649 | 0.3695 | 2.27m | 3.74m | Best before vegboost |
| 2 | `2A_vegboost` | **0.372136** | — | — | — | — | — | +0.006 gain; **current best** |

> Individual metric breakdown for submission 2 not yet retrieved from platform.
> 6A family produced nothing submission-worthy — proxy 0.24 max vs 2A's 0.37.

**Top team reference** (as of 2026-05-26): IoU_B=0.5269, IoU_V=0.8221, IoU_W=0.5194, RMSE_B=1.76m, RMSE_V=3.06m → score ~0.51

---

## All Experiments — Internal Validation Metrics

> **Proxy note**: Experiments 2A early (nologits/dynamic) used the *old* proxy (Tversky+2×MAE). Experiments 4A used old proxy with C=30. All 5A, 2A_mse/vegboost, and 6A variants use the *correct* proxy (C=4.0). Old and new proxy scores are **not comparable**.

### 2A Family — Attention Fusion, alpha_earth + patch embeddings

| Experiment | Patch inputs | Loss | Best epoch | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Proxy (C=4.0 equiv.) |
|-----------|-------------|------|-----------|-------|-------|-------|--------|--------|----------------------|
| `2A_alpha_ts1_ts2_nologits` | terramind_s1+s2 | Tversky+2×MAE | 46 | 0.1863 | 0.7668 | 0.4994 | 3.116m | 3.677m | — (old proxy) |
| `2A_alpha_ts1_ts2_dynamic` | terramind_s1+s2 | Tversky+2×MAE (dynamic) | 60 | 0.1808 | 0.7216 | 0.3786 | 2.813m | 3.713m | — (old proxy) |
| `2A_alpha_ts1_nologits` | terramind_s1 only | Tversky+2×MAE | 50 | 0.1873 | 0.7790 | 0.4287 | 3.134m | 3.873m | — (old proxy) |
| `2A_alpha_thors1_nologits` | thor_s1 only | Tversky+2×MAE | 56 | 0.1899 | 0.7596 | 0.3953 | 3.131m | 3.829m | — (old proxy) |
| `2A_alpha_thor1_thor2_nologits` | thor_s1+s2 | Tversky+2×MAE | 55 | 0.1874 | 0.7668 | 0.4776 | 3.162m | 3.726m | — (old proxy) |
| `2A_mse_sq` | terramind_s1+s2 | Pure MSE | 60 | 0.0566 | 0.7497 | 0.2606 | 2.785m | 3.690m | 0.2572 |
| `2A_mse_sigma` | terramind_s1+s2 | MSE+sigma weight | 60 | 0.0290 | 0.7395 | 0.2373 | 2.981m | 3.644m | 0.2353 |
| `2A_mse_vegboost` | terramind_s1+s2 | MSE+height boosts | 60 | 0.0225 | 0.7232 | 0.1598 | 2.943m | 3.682m | 0.2201 |
| **`2A_vegboost`** | terramind_s1+s2 | Composite+veg boost | 60 | **0.1887** | 0.7680 | **0.5725** | **3.011m** | **3.680m** | **0.3261** ✅ submitted |

### 3A Family — Y-Net (decoupled decoders, no GradScale)

| Experiment | Config | Best epoch | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V |
|-----------|--------|-----------|-------|-------|-------|--------|--------|
| `3A_ynet_@-ts1-2` | no aug, no dyn | 33 | 0.1788 | 0.7859 | 0.4265 | 3.570m | 3.845m |
| `3A_ynet_@ts1-2_dynloss` | dynamic loss | 33 | 0.1760 | 0.7817 | 0.3855 | 3.401m | 3.807m |
| `3A_ynet_@ts1-2_aug_dynloss` | aug + dynamic | 37 | 0.1642 | 0.7697 | 0.3920 | 3.711m | 3.860m |

### 4A Family — Y-Net + GradScale hook (α=0.1)

| Experiment | Config | Best epoch | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Old proxy |
|-----------|--------|-----------|-------|-------|-------|--------|--------|-----------|
| **`4A_hook`** | no aug, no dyn | 59 | 0.1895 | 0.7870 | 0.5681 | 3.462m | 3.799m | **0.6465** |
| `4A_hook_dyn` | dynamic loss | 59 | 0.1851 | 0.7867 | 0.5654 | 3.435m | 3.751m | 0.6454 |
| `4A_hook_aug` | augmentation | 56 | 0.1834 | 0.7836 | 0.5301 | 3.659m | 3.938m | 0.6362 |
| `4A_hook_aug_dyn` | aug + dynamic | 55 | 0.1809 | 0.7810 | 0.5230 | 3.621m | 3.882m | 0.6348 |

### 5A Family — Y-Net + GradScale + HeightBoost curriculum (1x→5x)

| Experiment | Seed | Best epoch | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Proxy (C=4.0) |
|-----------|------|-----------|-------|-------|-------|--------|--------|---------------|
| `5A_vegboost` | 0 | 60 | 0.1916 | 0.7787 | 0.5622 | 3.412m | 3.807m | 0.2954 |
| `5A_vegboost_s1` | 1 | 60 | 0.1874 | 0.7851 | 0.5322 | 3.413m | 3.723m | 0.2950 |
| `5A_vegboost_s2` | 2 | 60 | 0.1925 | 0.7867 | 0.5704 | 3.526m | 3.750m | 0.2938 |

### 6A Family — TESSERA pixel stream + cross-attention bottleneck fusion

Branch: `exp-6-tessera-xattn`. Detailed post-mortem: `prompts/exp-6-tessera-xattn-postmortem.md`.

| Experiment | Pixel inputs | Patch inputs | Dyn. loss | Best epoch | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Proxy (C=4.0) |
|-----------|-------------|--------------|-----------|-----------|-------|-------|-------|--------|--------|---------------|
| `6A_alpha_only_xattn` | alpha_earth | ts1+ts2 | No | 57 | 0.1683 | 0.7550 | 0.4075 | 3.57m | 4.03m | **0.2436** |
| `6A_alpha_only_xattn_dloss` | alpha_earth | ts1+ts2 | Yes | 55 | 0.1397 | 0.7410 | 0.2464 | 3.55m | 3.96m | 0.2132 |
| `6A_tessera_xattn` | alpha+tessera | ts1+ts2 | No | 44 | 0.0171 | 0.5668 | 0.4294 | 3.69m | 4.19m | 0.1731 |
| `6A_tessera_xattn_ts1only` | alpha+tessera | ts1 only | No | 49 | 0.0078 | 0.5759 | 0.4516 | 4.26m | 4.24m | 0.1561 |

**Conclusion**: 6A architecture is not competitive with 2A. Concluded without platform submission. Lessons drive the 7A design.

---

## Key Findings & Insights

### 1. Patch embedding choice matters for IoU_W
Comparing 2A variants with the same loss:
- terramind_s1 only: IoU_W = 0.429
- thor_s1 only: IoU_W = 0.395
- thor_s1+s2: IoU_W = 0.478
- **terramind_s1+s2: IoU_W = 0.499** ← best

Using both SAR and optical patch embeddings consistently outperforms single-modality, and terramind beats thor for water detection.

### 2. Dynamic loss scheduling hurts, not helps
In both 2A and 4A families:
- Dynamic loss (curriculum): IoU_W 0.379, RMSE_B 2.81m
- Fixed loss: IoU_W 0.499, RMSE_B 3.12m ← better IoU, similar RMSE

Dynamic loss may destabilise training by shifting the loss landscape mid-run.

### 3. Data augmentation consistently hurts (in 2A/4A regime)
In the 4A family:
- No augmentation: IoU_W 0.568, RMSE_B 3.46m
- With augmentation: IoU_W 0.530, RMSE_B 3.66m

Likely cause: crop diversity was reduced (augmentation ran before the numpy worker_init_fn seed fix), or the model's patch embeddings are already augmentation-invariant. 7A's domain-shift augmentation is qualitatively different (calibrated channel jitter, dropout, noise on embeddings) and should not be confused with this finding.

### 4. GradScale hook (4A) helps IoU_W but not RMSE_B vs 2A
| | 2A nologits | 4A hook |
|--|--|--|
| IoU_B | 0.186 | 0.190 |
| IoU_W | 0.499 | **0.568** |
| RMSE_B | **3.116m** | 3.462m |
| RMSE_V | **3.677m** | 3.799m |

The hook improves IoU_W (+0.07) but RMSE_B is significantly worse.

### 5. 4A had misleading internal proxy (old C=30)
- 4A_hook internal proxy = **0.6465** (C=30) → appeared to be a huge win
- Submitted 2A_nologits instead because RMSE_B was 3.12m vs 4A's 3.46m
- The old C=30 proxy was blind to RMSE contribution
- **Lesson**: the proxy formula with C=30 was useless. Fixed to C=4.0.

### 6. MSE loss destroys IoU metrics
Pure MSE and its variants (sigma-weighted, with height boosts) all collapsed IoU_B:

| Loss | IoU_B | IoU_W |
|------|-------|-------|
| Composite (Tversky+MAE+SSIM+GDL) | 0.189 | 0.573 |
| Pure MSE | 0.057 | 0.261 |
| MSE + sigma | 0.029 | 0.237 |
| MSE + vegboost | 0.023 | 0.160 |

MSE minimises average pixel error without pressure at the 0.5 decision boundary needed for hard-IoU. **Tversky/Dice components are essential.**

### 7. Vegboost: helpful but less than expected
- Expected: +0.038 platform score (RMSE_V 3.74m → ~3.0m)
- Actual: +0.006 platform score
- Internal validation improvement: IoU_W +0.073, RMSE_B -0.10m, RMSE_V -0.003m
- The RMSE_V was already close to the ceiling on the test set; the validation set gain did not generalise fully

### 8. HeightBoost curriculum (5A) backfires
5A trained with height weight ramping from 1x → 5x over 60 epochs:
- 5A RMSE_B: 3.41–3.53m (much worse than 2A's 3.01m)
- 5A proxy: 0.295 (vs 2A_vegboost 0.326)

The late-epoch over-emphasis on height (5x weight) shifted gradients away from the fraction channels, degrading RMSE_B and IoU_W. **Conclusion: a fixed weight is better; no curriculum needed.**

### 9. 2A_vegboost is the best model overall (internal + platform)
Direct comparison at best epoch, same architecture:

| Metric | 2A_nologits | 2A_vegboost | Δ |
|--------|-------------|-------------|---|
| IoU_B | 0.1863 | 0.1887 | +0.002 |
| IoU_V | 0.7668 | 0.7680 | +0.001 |
| IoU_W | 0.4994 | **0.5725** | **+0.073** |
| RMSE_B | 3.116m | **3.011m** | **−0.105m** |
| RMSE_V | **3.677m** | 3.680m | +0.003m |
| Proxy (C=4.0) | ~0.299* | **0.326** | +0.027 |

*Estimated by applying C=4.0 formula to nologits internal metrics.

### 10. Internal validation metrics underestimate platform IoU
Platform IoU_B = **0.3394** vs internal validation IoU_B = **0.186** for the same model (2A_nologits).
Platform IoU_W = **0.3695** vs internal = **0.499**.

The test set tiles appear to have higher-contrast (more distinctly building/non-building) regions than the validation split. The validation metrics can only be used for **relative** comparisons, not absolute prediction of platform scores.

> **Caveat (added 2026-05-30)**: This finding rests on a *random* validation split that shares spatial context with the training set. The platform test set is in different regions and years. Some of the "underestimation" may actually be optimism from the random split. 7A uses geographic CV; we expect validation scores to drop and align more closely with platform scores.

### 11. Concatenating TESSERA with AlphaEarth at the pixel stem destroys IoU_B (6A finding)

`6A_tessera_xattn` collapsed IoU_B from 0.168 → 0.017 — a 10× regression — vs `6A_alpha_only_xattn` (same model, no TESSERA). The 192→64ch projection at the encoder input forces high-frequency spatial features (buildings, edges) and low-frequency temporal features (phenology) into a single bottleneck. Temporal features dominate in variance and drown out building discrimination.

The conclusion is not "TESSERA is bad" but "TESSERA cannot be concatenated with AlphaEarth into a single encoder pipeline". The 7A design uses separate encoders per modality to avoid this competition.

### 12. Cross-attention with a shared decoder underperforms 2A's two-decoder split (6A finding)

`6A_alpha_only_xattn` (cross-attention bottleneck fusion, shared decoder + split heads) reached proxy 0.24 vs 2A_vegboost's 0.37 — same pixel input, same patch inputs, much worse result. The most likely cause is the shared decoder: 2A's Y-Net split into separate classification and height decoders was doing more work than credited. Cross-attention may or may not be a real improvement over broadcast fusion; the 6A experiment cannot tell us, because the shared decoder dominates the regression.

The 7A design keeps full decoder decoupling and tests the patch-fusion mechanism separately in ablations.

### 13. The two TerraMind patch streams (s1 + s2) are both needed

Removing `terramind_s2` (in `6A_tessera_xattn_ts1only`) worsens every metric, particularly RMSE_B (+0.7m). S1 (SAR) and S2 (optical) patch tokens carry complementary information. The 7A design routes each by sensor type — S1 to the height decoder, S2 to the fraction decoder — to exploit this complementarity rather than averaging it.

### 14. Dynamic loss continues to hurt in 6A

`6A_alpha_only_xattn_dloss` worsened proxy 0.244 → 0.213 vs the static version. IoU_W dropped 0.408 → 0.246. The dynamic ramp-up of height weight aggressively shifts the gradient balance late in training and the abundance channels stagnate. **Consistent with the 5A and 4A_dyn findings.** Dynamic curriculum on loss weights is not a useful technique for this problem; use static weights or learned weights (GradNorm).

---

## Current Best Platform Position

| Metric | Us (best) | Top team | Gap | Score weight |
|--------|-----------|----------|-----|-------------|
| IoU_B | 0.34 | 0.53 | −0.19 | 0.25 |
| IoU_V | 0.76 | 0.82 | −0.06 | 0.15 |
| IoU_W | 0.37 | 0.52 | −0.15 | 0.15 |
| RMSE_B | 2.27m | 1.76m | −0.51m | 0.25 |
| RMSE_V | 3.74m | 3.06m | −0.68m | 0.20 |
| **Score** | **0.3721** | **~0.51** | **−0.14** | |

> The largest score gap is IoU_B. This is structural — we cannot extract clean building signal — not a calibration issue. Architecture-side work (7A's decoupled encoders + sensor-routed patches + auxiliary binary B head) targets this directly.

---

## Pending Experiments & Expected Gains

| Action | Type | Est. gain | Status |
|--------|------|-----------|--------|
| 7A: dual-encoder/dual-decoder + sensor-routed patches | Retrain (new arch) | +0.05–0.10 | 📋 spec in `prompts/exp-7-clean-slate.md` |
| Geographic CV split | Data | +0.02–0.04 | 📋 part of 7A |
| Stratified tile sampling | Data | included in above | 📋 part of 7A |
| Auxiliary binary building head | Arch | +0.02–0.04 | 📋 part of 7A |
| Per-channel threshold calibration | Inference | +0.02–0.04 | 📋 part of 7A |
| TTA (8-fold D4) | Inference | +0.01–0.03 | 📋 part of 7A |
| 3-seed ensemble | Inference | +0.005–0.01 | 📋 part of 7A Phase 5 |
| THOR foundation model integration | Ablation | unknown | 📋 part of 7A Phase 5 |
| Guided filter on height output | Post-processing | small | ❌ deferred |