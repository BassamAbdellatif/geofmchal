# GeoFM Challenge — Experiment Results & Findings

## Platform Submissions (Ground Truth)

| Date | Experiment | Platform Score | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Inference | Notes |
|------|-----------|---------------|-------|-------|-------|--------|--------|-----------|-------|
| 5/23 | `2A_alpha_ts1_ts2_nologits` | 0.3660 | 0.3394 | 0.7649 | 0.3695 | 2.274 | 3.736 | raw | best before vegboost; under cliff |
| 5/25 | `4A_hook` | 0.3647 | 0.3420 | 0.7896 | 0.4126 | 2.394 | 3.789 | raw | Y-Net+GradScale; under cliff; best transfer ratio |
| 5/26 | `2A_alpha…_tta` | 0.3369 | 0.3197 | 0.7539 | 0.3878 | 2.409 | 4.087 | **TTA** | TTA *hurt*: RMSE_V 3.74→4.09 crossed cliff (−0.029) |
| 5/28 | **`2A_vegboost`** | **0.3721** | 0.3331 | 0.7642 | 0.4154 | 2.279 | 3.704 | raw | **current best**; under cliff |
| 6/1  | `7A_geocv_baseline_tta` | 0.3575 | 0.3403 | 0.7981 | 0.4138 | 2.348 | 4.093 | TTA+blend+thresh | over cliff |
| 6/8  | `7A_v1_base` | 0.3392 | 0.3263 | 0.7915 | 0.3602 | 2.401 | 4.128 | raw | water collapsed on transfer (f29) |
| 6/9  | **`9_sm4tv_final_hybrid`** | **0.3871** | 0.3511 | 0.7949 | 0.4544 | 2.279 | 3.704 | hybrid (sm4tv B/V/W + 2A height) | **new best**; rank 54 |

> **Current best: 0.3871 (`9_sm4tv_final_hybrid`, 6/9) — but still rank 54** (field is at 0.48–0.54).
> The hybrid worked: sm4tv's building lift *transferred* (IoU_B 0.326→0.351, above 2A's 0.333) and 2A's
> height gave RMSE_V 3.70.
>
> **Scoring formula (reverse-engineered from the hybrid-vs-2A delta, matches to ~0.001):**
> `final ≈ 0.25·IoU_B + 0.15·IoU_V + 0.15·IoU_W + 0.25·(1−RMSE_B/4) + 0.20·(1−RMSE_V/4)` — i.e. ≈ our
> internal C=4 proxy. **CORRECTION to the earlier "3.9 m cliff" framing (f21/science.md): there is NO
> hard cliff** — RMSE credit is ~linear, so height is lost *continuously*. Top teams at RMSE_V ~2.8–3.0
> bank far more height credit than us at 3.70 — a graded deficit, not a binary threshold.
>
> **Where we lose vs top-10 (Cisco 0.478; our 0.387 → ~0.09 gap):** IoU_B −0.13 (≈ −0.034 score),
> **RMSE_B + RMSE_V combined ≈ −0.044 (our *largest*, most under-invested deficit)**, IoU_W −0.055
> (≈ −0.008), IoU_V ~flat. Top teams beat us on *every* metric → a model-capability gap, not a tweak.
> **Lessons that stand:** TTA is unsafe (smears height → +RMSE); 7A's internal IoU leads are largely
> fold-0 overfit (f29); 6A family produced nothing submission-worthy (proxy 0.24 max).

**Leaderboard snapshot (as of 2026-06-01) — top 8 teams:**

| Rank | Team | Score | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V |
|------|------|-------|-------|-------|-------|--------|--------|
| 1 | Beddings | 0.5137 | 0.5269 | 0.8221 | 0.5194 | 1.760m | 3.065m |
| 2 | tOfMemory | 0.5020 | 0.5077 | 0.8218 | 0.5125 | 1.780m | 3.168m |
| 3 | DTM | 0.4925 | 0.4813 | 0.8129 | 0.4792 | 1.745m | 3.156m |
| 4 | HTEQ | 0.4907 | 0.4948 | 0.8093 | 0.4853 | 1.774m | 3.235m |
| 5 | nk | 0.4862 | 0.5128 | 0.8110 | 0.4747 | 1.832m | 3.303m |
| 6 | ention_Plzzz | 0.4827 | 0.4689 | 0.8158 | 0.5192 | 1.883m | 3.196m |
| 7 | altedLAB | 0.4789 | 0.4994 | 0.8048 | 0.5128 | 1.897m | 3.388m |
| 8 | (partial name) | 0.4745 | 0.4883 | 0.8136 | 0.4814 | 1.848m | 3.445m |

**Per-column best across visible leaderboard:**

| | Score | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V |
|--|-------|-------|-------|-------|--------|--------|
| **Field best** | **0.5137** | **0.5269** | **0.8221** | **0.5194** | **1.745m** | **3.065m** |

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

> Updated 2026-06-01 after 7A submission and leaderboard review.

| Metric | Us (2A best) | Us (7A submitted) | Field best | Gap to field | Score weight |
|--------|-------------|-------------------|-----------|-------------|-------------|
| IoU_B | 0.3394 | 0.3403 | 0.5269 | −0.187 | 0.25 |
| IoU_V | 0.7649 | **0.7981** | 0.8221 | −0.024 | 0.15 |
| IoU_W | 0.3695 | **0.4138** | 0.5194 | −0.106 | 0.15 |
| RMSE_B | 2.27m | 2.35m | 1.745m | −0.60m | 0.25 |
| RMSE_V | 3.74m | 4.09m ⚠️ | 3.065m | above cliff | 0.20 |
| **Score** | **0.3721** | **0.3575** | **0.5137** | **−0.156** | |

**Gap analysis by metric (7A vs field best, weighted score impact):**

| Metric | Gap | Weighted score gap | Status |
|--------|-----|-------------------|--------|
| IoU_B | −0.187 | −0.047 | 🔴 Primary problem. Every top-8 team above 0.468. Architecture claim not yet confirmed on test set. |
| RMSE_V | above cliff | −0.043 | 🔴 Entire term contributes zero. Fix: add veg_height_boost to DualPathLoss. Target val RMSE_V ≤ 3.8m. |
| RMSE_B | −0.60m | −0.038 | 🟡 Improving (2A: 2.27m → 7A internal: 2.09m) but regressed on platform (2.35m). |
| IoU_W | −0.106 | −0.016 | 🟡 7A made real gains (+0.044 vs 2A). Partially solved. |
| IoU_V | −0.024 | −0.004 | 🟢 Essentially solved. TESSERA τ-encoder working. |

**Score projection if RMSE_V fixed (gets under 3.9m on test):**
```
Current 7A:          0.358
+ RMSE_V fix:       +0.043   (cliff term activates at 3.7m internal → ~3.8m test)
+ inference tricks: +0.010   (TTA+blend+threshold, now safe once cliff cleared)
─────────────────────────
Estimated:          ~0.411   → rank ~7–8 on current leaderboard
```

**Score projection if IoU_B also closes to 0.45:**
```
Above:               0.411
+ IoU_B 0.34→0.45:  +0.028
─────────────────────────
Estimated:          ~0.439   → rank ~4–5 on current leaderboard
```

---

### 7A — Internal Validation Ablation (inference config scan, geo-CV val set)

Run after epoch 60 on `7A_geocv_baseline`. 514 val tiles, geo-CV fold 0.

| Config | Proxy | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V |
|--------|-------|-------|-------|-------|--------|--------|
| raw | 0.3947 | 0.200 | 0.812 | 0.687 | 2.087m | 3.996m |
| +threshold | 0.4040 | 0.237 | 0.812 | 0.688 | 2.087m | 3.996m |
| +blend | 0.4048 | 0.241 | 0.812 | 0.687 | 2.087m | 3.996m |
| +blend+threshold | 0.4042 | 0.238 | 0.812 | 0.688 | 2.087m | 3.996m |
| TTA | 0.3973 | 0.200 | 0.812 | 0.689 | 2.067m | 3.975m |
| TTA+blend+threshold | **0.4072** | 0.240 | 0.813 | 0.689 | 2.067m | 3.975m |

**Key observations from ablation:**
- Raw RMSE_V = 3.996m — only 6mm below the cliff. TTA reduces it to 3.975m on val but test-set domain shift adds ~94mm → 4.09m on platform.
- +threshold alone gives the largest IoU_B gain (0.200 → 0.237, +0.037).
- +blend alone gives similar IoU_B gain (0.200 → 0.241, +0.041).
- +blend+threshold is slightly worse than +blend alone (0.238 vs 0.241) — threshold calibration was optimised on raw predictions; after blending shifts channel 0 upward, the calibrated threshold is no longer optimal.
- TTA is net-positive on val (proxy +0.003) but the domain shift on RMSE_V is the platform problem — not TTA itself.
- **The domain shift gap on RMSE_V is ~94mm** (val 3.975m → platform 4.09m). Must get val RMSE_V ≤ 3.806m to safely clear the 3.9m cliff on the platform.

---

### 7A Family — Internal Validation Metrics (geo-CV fold 0)

| Experiment | Epoch | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Proxy (C=4.0) | Notes |
|-----------|-------|-------|-------|-------|--------|--------|---------------|-------|
| `7A_geocv_baseline` | 1 | 0.150 | 0.763 | 0.563 | 2.45m | 4.90m | 0.333 | First epoch, RMSE_V above cliff |
| `7A_geocv_baseline` | 60 | 0.182 | 0.813 | 0.686 | 2.09m | 3.97m | **0.391** | Best checkpoint |

GradNorm weights at epoch 60: w[f/h/b] = 0.39 / 1.31 / 1.30. Height task consistently needed amplification throughout training.

**7A vs 2A_vegboost (internal val):**

| Metric | 2A_vegboost | 7A ep60 | Δ |
|--------|-------------|---------|---|
| IoU_B | 0.189 | 0.182 | −0.007 |
| IoU_V | 0.768 | 0.813 | **+0.045** |
| IoU_W | 0.573 | 0.686 | **+0.113** |
| RMSE_B | 3.011m | 2.087m | **−0.924m** |
| RMSE_V | 3.680m | 3.996m | −0.316m |
| Proxy | 0.326 | **0.391** | **+0.065** |

RMSE_B improvement of −0.924m on val confirms the dual-encoder design is working for building height. RMSE_V regression on val (and worse on platform) is the veg_height_boost omission.

---

## Key Findings & Insights

### 1. Patch embedding choice matters for IoU_W

### 15. 7A architecture improvements are real but veg_height_boost omission costs the submission

7A platform submission (TTA+blend+threshold) scored 0.358 — below 2A_vegboost's 0.372 — despite genuine architectural improvements. Root cause: RMSE_V = 4.09m (above 3.9m cliff) contributing zero to score vs 2A's 3.74m contributing ~0.008. The cliff term alone explains the regression.

Internal val ablation confirmed RMSE_V = 3.996m on val (6mm below cliff), but ~94mm domain shift on the test set pushed it above. The `DualPathLoss` in 7A omitted the `veg_height_boost` term that 2A used to pull RMSE_V down. **Fix: add cliff-aware veg_height_boost to DualPathLoss and retrain as `7A_vegboost`.**

The inference ablation also revealed: +blend alone outperforms +blend+threshold for IoU_B (0.241 vs 0.238) because threshold calibration is optimised on raw predictions and becomes mis-calibrated after blending shifts channel 0's distribution upward. Calibrate thresholds after blending, not before.

---

## Pending Experiments & Expected Gains

| Action | Type | Est. gain | Status |
|--------|------|-----------|--------|
| 7A_vegboost: add cliff-aware veg_height_boost to DualPathLoss | Retrain | +0.04–0.05 | 🔥 highest priority — fixes RMSE_V cliff problem |
| Fix TTA to fraction channels only (exclude height ch) | Inference fix | +0.005 | 📋 one-line fix in predict.py |
| Re-calibrate thresholds after blending | Inference fix | +0.003 | 📋 run threshold scan on blended predictions |
| 7A Phase 5: structural ablations (encoder split, decoder split, patch routing) | Ablation | diagnostic | 📋 prompts/exp-7-clean-slate.md Phase 5 |
| THOR foundation model integration | Ablation | unknown | 📋 7A Phase 5 |
| 3-seed ensemble | Inference | +0.005–0.01 | 📋 7A Phase 5 |
| Full-dataset training (no val split) for final ensemble | Training | +0.005–0.01 | 📋 final submission only |
| Guided filter on height output | Post-processing | small | ❌ deferred |

---

## 7A Phase 5 — Ablation Campaign (5B + 5C)

Two ablation rounds on the 7A `DualEncDualDecFusion` model, geographic CV fold 0,
NVMe-cached tiles, one run per cluster node. 5B varied data/loss knobs (90 epochs,
GradNorm, vegboost 1.0); 5C varied training strategy + two architecture components
via additive **default-off CLI flags** (60 epochs, seed 0). The 5C flags
(`--static-weights`, `--no-height-bridge`, `--no-binary-head`, `--seed`) are
default-preserving — a forward pass without them is byte-identical to the pre-flag
model (verified max_abs_diff = 0 on all output channels).

### Phase 5B — data / loss ablations (90 epochs, GradNorm, stratified on)

| Run | Proxy | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Best ep |
|-----|-------|-------|-------|-------|--------|--------|---------|
| 7A_base_e90 (all on) | 0.396 | 0.206 | 0.808 | 0.686 | 2.14 | 3.92 | 59 |
| 7A_no_vegboost_e90 | 0.399 | 0.204 | 0.809 | 0.684 | 2.09 | 3.91 | 40 |
| 7A_no_gradnorm_e90 | 0.393 | 0.206 | 0.810 | 0.680 | 2.15 | 3.97 | 40 |
| 7A_no_stratified_e90 | 0.287 | 0.200 | 0.809 | **0.000** | 2.15 | 4.03 | 65 |

Isolated contribution (baseline − ablation): **stratified +0.108**, GradNorm +0.003,
vegboost −0.004 (removing vegboost slightly *helped* the val proxy).

### Phase 5C — architecture / training ablations (60 epochs, seed 0, static weights, stratified on)

| Run | Changed vs `7A_simple` | Proxy | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Best ep | mins |
|-----|------------------------|-------|-------|-------|-------|--------|--------|---------|------|
| **7A_simple** (reference) | — | **0.399** | **0.215** | 0.810 | 0.684 | 2.06 | 4.03 | 59 | 142 |
| 7A_abl_no_bridge | drop α→height bridge | 0.294 | 0.213 | 0.808 | **0.000** | 2.08 | 4.12 | 34 | 141 |
| 7A_abl_no_binary | drop aux binary head | 0.348 | **0.000** | 0.815 | 0.694 | 2.06 | 4.06 | 51 | 142 |
| 7A_vegboost_high | vegboost 5.0 + GradNorm | 0.392 | 0.207 | 0.809 | 0.685 | 2.14 | 4.05 | 59 | 170 |

`7A_simple` = static weights `[0.65,0.64,1.70]`, bridge on, binary on, **no GradNorm, no vegboost** — the recommended 7A baseline going forward.

### 16. Rare classes collapse to exactly 0 on hard IoU without targeted signal

The hard-IoU@0.5 metric punishes any rare class whose predictions sit below 0.5.
Two components keep them above the line, and removing either drives the matching IoU
to **0 for the entire run**: the **stratified sampler** (water → IoU_W=0 without it,
5B) and the **auxiliary binary head** (buildings → IoU_B=0 without it, 5C). This is
the central 7A lesson — buildings and water both sit right at the 0.5 boundary.

### 17. veg_height_boost is structurally ineffective in 7A — drop it

Inert at 1.0 (5B) and at 5.0 (5C): RMSE_V floors at **4.017m with boost vs 4.007m
without** — no movement (marginally worse), and proxy lower (0.392 vs 0.399). RMSE_V
plateaus ~4.0m regardless of boost strength. The 2A boost mechanism does not transfer
to the 7A height decoder. Remove vegboost from 7A.

### 18. GradNorm ≈ static weights — drop GradNorm

`7A_simple` (static `[0.65,0.64,1.70]`) matches the GradNorm run exactly (proxy 0.399
vs 0.399), with slightly better IoU_B (0.215 vs 0.204) and **16% less wall-time**
(142 vs 170 min). Use the static converged weights; GradNorm earns nothing here.

### 19. The auxiliary binary head is essential for IoU_B

Dropping it zeros IoU_B for all 60 epochs (0.215 → 0.000) while IoU_V/W stay healthy.
The binary head's BCE+Dice supervision is what carries building predictions over the
0.5 threshold. As IoU_B is our #1 platform gap, keep the head and lean on post-blend
threshold calibration at inference.

### 20. The cross-encoder bridge does not improve RMSE_B (its stated purpose); ablation confounded

RMSE_B is unchanged with/without the bridge (2.06 vs 2.08, Δ0.018m, far under the
0.2m threshold) — the bridge is not measurably helping building height. The proxy
drop is entirely an unexplained **IoU_W → 0 collapse** (architecturally implausible:
the bridge feeds the height decoder, not water). Treat as a calibration/seed artifact
pending a confirmation re-run; the bridge is cheap and default-on, so keep it for now.

### 21. RMSE_V cliff is the binding constraint for a 7A submission

Best 7A config floors RMSE_V at ~4.0m, above the **3.9m platform cliff** → 0 credit
for the RMSE_V term (20% of the score), and vegboost (the intended fix) is now proven
dead. A 7A standalone submission today would likely land ~0.36 — the same territory
as the prior 7A submission (0.358) — and **tie/lose to 2A_vegboost (0.372)**. Do not
spend a 12h submission slot on it.

### Updated next steps (post-Phase-5C)

| Action | Rationale | Priority |
|--------|-----------|----------|
| **Hybrid: `7A_simple` fraction/building channels + `2A_vegboost` height** | combines 7A's IoU strength with 2A's sub-cliff RMSE_V (3.74m) | 🔥 highest EV |
| Inference threshold + blend-binary calibration on `7A_simple` (calibrate post-blend) | buildings borderline at 0.5; IoU_B is #1 gap | 🔥 cheap |
| Adopt simplified 7A baseline: static weights, no GradNorm, no vegboost | equal/better, simpler, 16% faster | ✅ adopt |
| Structurally attack RMSE_V (vegboost dead): why does the height decoder plateau at 4.0m? | RMSE_V is the only thing keeping 7A below 2A | 📋 research |
| Confirm `no_bridge` IoU_W collapse with a seed re-run before removing the bridge | rule out artifact | 📋 |
| ~~7A_vegboost retrain (was 🔥 highest priority)~~ | **DONE — vegboost confirmed ineffective (findings 17)** | ✅ closed (negative) |

---

## 7A Phase 5D — THOR Integration + Enhanced Patch Stems (2026-06-04) — NEGATIVE

Full detail and diagnostic trail: `docs/phase5d_thor.md`. Branch `exp-7-clean-slate`.
All metrics: geo-CV fold 0, hard-IoU@0.5, **raw prediction (no threshold/blend)**, 60 epochs,
static weights `[0.65,0.64,1.70]`, no GradNorm, no vegboost, **batch 32**.

### Final campaign (v1 stem, water-safe)
| Experiment | Patch inputs | Stem | Proxy | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V |
|------------|--------------|------|-------|-------|-------|-------|--------|--------|
| `7A_v1_base` (reference repro) | terramind_s1,s2 | v1 | **0.400** | 0.196 | 0.811 | 0.684 | 2.07m | 4.03m |
| `7A_v1_thor` (full THOR) | +thor_s1,thor_s2 | v1 | 0.294 | 0.189 | 0.808 | **0.000** | 2.07m | 3.98m |
| `7A_v1_thor_s1` | +thor_s1 | v1 | 0.397 | 0.191 | 0.811 | 0.688 | 2.05m | 4.00m |
| `7A_v1_thor_s2` | +thor_s2 | v1 | 0.395 | 0.192 | 0.807 | 0.687 | 2.06m | 4.00m |

(An earlier v2-stem @ batch-24 campaign was discarded — water bug, finding 23/24.)

### 22. THOR adds no value to 7A
Single-stream THOR (s1 *or* s2) ties the base proxy (~0.40) with **no IoU_B gain**; full THOR
(both streams) collapses water (IoU_W 0.68→0.00). THOR rejected — consistent with the 2A-era
finding that it was redundant; the per-modality-calibration hypothesis did not rescue it (f23).

### 23. The enhanced patch stem (v2/v2b) suppresses rare-class ignition
The Phase-5D "enhanced" stem (`PatchTokenStemV2`: input-LayerNorm + 2-layer MLP) zeroed IoU_W
even terramind-only. Isolation proved it is **not** the input LayerNorm — the `v2b` variant
(LayerNorm removed, output-norm instead) still collapses water. It is the **deeper 2-layer stem
itself**; the simple **v1** single-linear stem is required. v2 and v2b rejected.

### 24. Rare-class (water) ignition needs batch ≥ 32 and a simple stem
Water IoU ignites sharply (~0→0.5 by epoch 2) only under a narrow regime: **v1 stem + batch 32 +
≤3 patch streams**. Batch 24, the v2/v2b stem, or full-THOR each *independently* prevent ignition
— water stays exactly 0.000 for all 60 epochs, at every threshold down to 0.05 (so not a
calibration artifact; cache verified byte-identical, ruling out data). Building survives these
perturbations because of its dedicated binary head (weight 1.70); water has none and dies first.
Extends finding 16. Root cause = weak rare-class supervision → the productive lever is a dedicated
**water head / aggressive rare-class sampling**, not patch-token engineering.

### 25. Augmentation RNG is unseeded (reproducibility gap)
`GeoFMDataset7A.__getitem__` uses `np.random.default_rng()` (no seed), so augmentation differs
every run regardless of `--seed`. Fine ablations (±0.01 IoU) are noise-limited until seeded.

### Phase 5D verdict & forward plan
- **Best 7A model = `7A_v1_base` = the reference** (proxy ~0.40, IoU_B 0.20). Phase 5D produced no gain.
- THOR, enhanced stems: **closed-negative.**
- Inference threshold calibration / blend-binary **not pursued** — fit to fold-0 val, won't
  transfer to the different-region/year test set (generalization risk). Rely on raw 0.5 prediction.
- New line **`exp-8-decouple`** (targets IoU_B): seed augmentation; single-task ablations
  (is multi-task interference real?); aggressive rare-class oversampling; symmetric cross-encoder
  gradients (tessera→fraction bridge, GradScale sweep); SAR-safe (no-rotation) augmentation.

---

## 7A Phase 8 — Task/Encoder Decoupling + Rare-Class Robustness (2026-06-08) — NEGATIVE (P4, P3a)

Branch `exp-8-decouple` (from `exp-7-clean-slate` HEAD). Spec: `prompts/exp-8-decouple.md`.
Goal: raise **IoU_B** (worst class, #1 platform gap) without regressing IoU_W/IoU_V; raw-0.5
prediction only (no calibration). All runs: geo-CV fold 0, hard-IoU@0.5, v1 stem, batch 32,
static weights `[0.65,0.64,1.70]`, no GradNorm/vegboost, **seeded augmentation (finding 25 fixed)**,
**40 epochs** (cut from 60 for economy — all plateaued well before 40).

**Methodology note:** IoU_B oscillates ±0.02–0.03 epoch-to-epoch, so the proxy-best *peak* rewards
luck. Numbers below are the **last-10-epoch mean** (the run's own swing in parens). Between-run
differences are all *smaller* than the within-run swing → treat them as ties unless stated.

### Campaign
| Run | Config | IoU_B | IoU_W | IoU_V | RMSE_V |
|-----|--------|-------|-------|-------|--------|
| `7A_v1_base_seeded` (anchor) | joint, default sampler `1,1.5,2,3` | **0.178** (0.171–0.185) | 0.685 | 0.806 | 4.05m |
| `8_strata_1248` (P3a) | joint, sampler `1,2,4,8` | 0.190 (0.171–0.205) | 0.683 | 0.808 | 4.10m |
| `8_frac_only` (P4) | `--task fraction` (drop height) | 0.183 (0.164–0.206) | 0.674 | 0.799 | — |
| `8_height_only` (P4) | `--task height` (drop fraction+binary) | — | — | — | **3.91m ↓** |

### 26. Multi-task interference is NOT the building bottleneck (P4)
Fraction-only IoU_B (0.183) = joint anchor (0.178), inside noise. Isolating the fraction task
frees **zero** building capacity → the height task is not stealing from buildings. Only isolation
effect is a small height gain (height-only RMSE_V 3.91 vs 4.05 joint) — minor, and not our gap
metric. **The loss combination is not the problem.**

### 27. Aggressive rare-class oversampling buys nothing (P3a)
4× rare-stratum weight (`1,2,4,8`) lands IoU_B 0.190 / IoU_W 0.683 — both on the anchor (+0.012
IoU_B is within swing). **Water is saturated at ~0.68; buildings capped at ~0.18 regardless of
sampling frequency.** P3a closed-negative. (Refines finding 24: rare-class *frequency* is not the
water/building lever once ignition has occurred.)

### 28. The IoU_B ~0.18 ceiling is intrinsic
Invariant across task-separation (f26) AND 4× oversampling (f27): neither interference nor
frequency. The ceiling is structural — building objective formulation, label/resolution quality,
or input features. NB: seeded anchor IoU_B is **~0.18**, below the **0.20** quoted from the
pre-seeding run (that was a lucky peak; seeding removed the optimism — corrects finding-table value).

### Infra note — AMP fits bs32 on 48 GB Ada, and preserves water
`--amp` (bf16 autocast) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` fits the 7A model at
batch 32 on the RTX 6000 Ada nodes (n1/n2 OOM'd in fp32 at the ~46.7 GB margin). Critically, bf16
**preserves the fragile water ignition** (`8_frac_only` ignited 0.27→0.57→0.67 under AMP) — so the
gradient-checkpointing fallback was not needed. autocast alone is insufficient memory-wise (weights
stay fp32); the `expandable_segments` fragmentation reclaim is what closes the last ~1.8 GB.

### Phase 8 interim verdict & forward plan
- **P3a: closed-negative.** No more sampling-weight experiments.
- **P1 (cross-encoder gradients): DEMOTED** to at most one cheap probe — premise undercut by f26
  (if total task isolation can't move IoU_B, re-routing gradient share won't either).
- **Reframe:** the honest IoU_B lever is the **building objective itself** (candidate: focal /
  boundary / Dice term on the binary head) or **input resolution/features** — not the multi-task
  plumbing. *Building-objective experiments are the leading next candidate, pending discussion.*
- P3b (SAR-safe aug) and P2 (patch ablate-out) still open but lower priority.

---

## Internal↔Platform Transfer Analysis (2026-06-08)

### 29. Single-fold (fold-0) validation is a biased, over-optimistic estimator — for 7A specifically
Comparing each submitted model's **internal best-epoch** metrics (unified C=4 proxy from
`training_params.txt`) against its **platform** result. Δ = platform − internal:

| Experiment | Δ score | Δ IoU_B | Δ IoU_W | Δ RMSE_B | Δ RMSE_V | net |
|-----------|---------|---------|---------|----------|----------|-----|
| `2A_alpha_nologits` | **+0.043** | +0.147 | −0.169 | −0.72 | +0.04 | magnified |
| `4A_hook` | **+0.067** | +0.150 | −0.139 | −1.00 | +0.01 | magnified |
| `2A_vegboost` | **+0.046** | +0.144 | −0.157 | −0.73 | +0.02 | magnified |
| `7A_geocv_baseline` | **−0.037** | +0.140 | −0.274 | +0.26 | +0.10 | attenuated |
| `7A_v1_base` | **−0.061** | +0.097 | −0.320 | +0.31 | +0.09 | attenuated |

The magnify→attenuate flip is **architectural, not temporal** (early wins = 2A/4A; recent = 7A):

- **IoU_B magnifies for all (+0.10–0.15)** — platform test buildings are easier than fold-0 val. But
  **7A_v1_base magnifies the *least* (+0.097)** despite the highest internal IoU_B (0.229): its
  building lead transfers worst. On the platform 7A_v1_base IoU_B (0.326) is **below** 2A_vegboost
  (0.333).
- **IoU_W is the killer and hits 7A 2× harder**: 2A/4A drop ~−0.15, **7A drops −0.27 to −0.32**.
  7A's internal water (0.68) is inflated and collapses to 0.36–0.41 on the different-region test.
- **RMSE_B confirms direction**: 2A/4A *under*-fit fold-0 (platform better, Δ −0.7 to −1.0); 7A
  *over*-fit it (platform worse, Δ +0.26 to +0.31).

**Conclusion.** 2A/4A earn their proxy from modest, honest components that transfer or improve, so
platform ≥ internal. 7A earns a *higher* proxy by **overfitting the fold-0 holdout (esp. water, the
rare class that "ignites" to 0.68)** — exactly the components that collapse on a new region. The
larger 7A model (18.6 M, dual-encoder + cross-attn, selected on one geo-fold) memorises fold-0;
2A is too simple to overfit that hard.

**Implications (these reshape the plan):**
- **Internal IoU_B is not a trustworthy north star for 7A.** A single-fold IoU_B "win" may be fold-0
  overfit. → **Validate the Phase-9 campaign on multiple geo-folds** (an improvement must hold across
  folds to count), or confirm on the platform. No code change needed — train with `--cv-fold 0/1/2`.
- **Do NOT push 7A water via oversampling (P3a)** — it deepens the overfit already costing −0.32 on
  transfer. The water fix is **domain generalization** (embedding-space augmentation), not fold-0
  sampling.
- **7A's only honest platform edge is IoU_V**; its building/water leads are largely fold-0 mirages.
- **4A_hook transfers best** (+0.067, clears the cliff) and is a strong base to revisit.

### 30. Building is recall-limited (mildly) — P/R diagnostic on `7A_v1_base`, fold 0
`pr_diagnostic.py` decomposing the building channel @0.5:

| class | precision | recall | IoU | TP / FP / FN |
|-------|-----------|--------|-----|--------------|
| **building** | 0.393 | 0.354 | 0.229 | 91k / 141k / **167k** |
| veg | 0.886 | 0.903 | 0.809 | — |
| water | 0.884 | 0.746 | 0.680 | — |

Building threshold sweep: IoU 0.243 @0.30 → 0.238 @0.40 → **0.229 @0.50** → 0.213 @0.60.

Recall (0.354) < precision (0.393), FN (167k) > FP (141k), and IoU *rises* as the threshold drops
→ the model **under-calls buildings at 0.5**. **Verdict: recall-limited → P1/Tversky (β>α) is the
lever** (pushes building prob up so more pixels cross 0.5, no calibration). **P2/softmax4 pushes the
wrong way** (suppresses positives near 0.25 → worsens recall). Caveats: (a) both axes are weak
(~+0.014 IoU headroom from the operating point) → P1 is a *modest* lever, not a breakthrough; big
building gains likely need features/resolution (f28). (b) fold-0 only — direction is stable, but
buildings *magnify* on the platform (0.229 → 0.326), so the recall gap may be gentler on the real test.

### 31. Validation methodology decision — geo multi-fold for selection, train-final on all
**Decision:** select the Phase-9 recipe on **geo multi-fold (folds 0 + 1)**; train the **final
submission model on all 5 folds**. An arm counts as a win only if it beats the anchor on *both* folds.
**Why not random / stratified-random split:** the platform test is a *new region + year* (OOD). A
random or stratified-random val puts validation tiles next to their training neighbours → spatial
autocorrelation **leakage** → an optimistic *in-distribution* estimate that won't transfer. f29
already showed geo-CV is over-optimistic for 7A; a leaky split would mislead *more*. Geo multi-fold
measures extrapolation to an unseen region (mimics the test) and, averaged over folds, still "sees
all variants" as held-out. "Use all data" is a *training* concern, satisfied by training the final
model on all folds — not a reason to leak the validation. (Buffered/blocked CV is the only stronger
option but adds code; plain geo multi-fold is the faithful, simpler choice.)

---

## Phase 9 — Building-Objective Multi-Fold Campaign (2026-06-09) — POSITIVE (sm4tv transfers)

3 arms × 3 geo-folds, 40 epochs, seed 0, `--amp`, last-10-epoch-mean IoU. Arms vs `7A_v1_base`:
`tversky` = `--building-overlap tversky 0.3/0.7` (P1); `sm4tv` = `+ --fraction-head softmax4` (P1+P2).

### Full matrix (last-10-mean)
| run | IoU_B | IoU_V | IoU_W |
|-----|-------|-------|-------|
| anchor_f0 | 0.1870 | 0.8080 | 0.6839 |
| tversky_f0 | 0.2355 | 0.8043 | 0.6840 |
| sm4tv_f0 | **0.2425** | 0.8056 | 0.6936 |
| anchor_f1 | 0.2533 | 0.7499 | 0.6724 |
| tversky_f1 | 0.2558 | 0.7483 | 0.6006 |
| sm4tv_f1 | **0.2625** | 0.7497 | 0.6289 |
| anchor_f2 | 0.1950 | 0.7847 | 0.6843 |
| tversky_f2 | 0.2175 | 0.7849 | 0.6971 |
| sm4tv_f2 | **0.2267** | 0.7923 | 0.6927 |

### 32. The building-objective redesign (P1+P2) lifts IoU_B and TRANSFERS across regions
Δ IoU_B vs each fold's anchor — **tversky** +0.049 / +0.003 / +0.023 (mean +0.025); **sm4tv**
+0.056 / +0.009 / +0.032 (mean **+0.032**). Δ IoU_W — tversky +0.000 / −0.072 / +0.013;
**sm4tv** +0.010 / −0.044 / +0.008.

- **It's a recall mechanism, not fold-0 overfit.** The lift is large in *hard*-building regions
  (fold0/fold2, anchor IoU_B ~0.19 → +0.05/+0.03) and ~flat in the *easy* region (fold1, anchor
  0.25, no recall headroom). Confirmation on fold2 (a third independent geo-fold) is the cross-fold
  proof finding-29 demanded — this **generalizes** (contrast 7A's water, which did not).
- **`sm4tv` (P1+P2) > `tversky` (P1) on both axes** — bigger IoU_B every fold *and* half the water
  cost. The derived "others" softmax channel sharpens buildings and partially protects rare classes.
  (Retires the earlier "softmax robs water" worry — that was a transient early-epoch dip; at
  convergence sm4tv water = anchor or higher on fold0/fold2.)
- **Only downside: fold-1 IoU_W** (sm4tv −0.044, tversky −0.072) — region-specific, not systemic.
- **Honest size:** net proxy gain ~+0.007 mean (0.25·ΔB + 0.15·ΔW), clearly positive on the hard
  folds, ~0 on the easy one. A real, transferable IoU_B improvement that rarely hurts — **not** a
  top-3 breakthrough alone; it pairs with the height-cliff fix (hybrid 2A height) for a leaderboard
  move.

**Recipe to carry forward:** `--fraction-head softmax4 --building-overlap tversky --tversky-alpha
0.3 --tversky-beta 0.7`. **Watch-item:** confirm fold-1's water dip doesn't hurt IoU_W on the
platform region.

### Forward plan
1. Train the **sm4tv** recipe on **all 5 folds** (no holdout) → final building/fraction model.
2. **Hybrid**: graft 2A_vegboost's sub-3.9 m height onto sm4tv's fraction/building channels (clears
   the RMSE_V cliff that caps standalone 7A — f29/cliff note).
3. Submit the hybrid; compare platform IoU_B/IoU_W vs `2A_vegboost` 0.3721. Optional confirm arms:
   gentler tversky 0.4/0.6 if the fold-1 water cost looks load-bearing on the platform.

**Outcome:** done — `9_sm4tv_final` (all-folds no-holdout) + `hybrid_merge.py` (graft 2A height) →
`9_sm4tv_final_hybrid` = **0.3871, our best** (but rank 54; the field moved). New code: `--cv-fold -1`
no-holdout mode, `--scratch-dir` (NVMe I/O), `hybrid_merge.py`, `--height-bridge-alpha`.

---

## Phase 10 — Cross-encoder gradient / bridge sweep (2026-06-10) — NEGATIVE (null)

Motivation: height (RMSE_V) is our largest leaderboard deficit; the alpha(optical)→height bridge is
GradScale-throttled to α=0.2, so maybe the optical encoder is under-used for height. Swept the
height-bridge α and the tessera→fraction bridge on the sm4tv recipe, fold 0 (last-10-mean):

| arm | α (opt→height) | IoU_B | IoU_W | RMSE_B | RMSE_V |
|-----|----------------|-------|-------|--------|--------|
| `sm4tv_f0` (control) | 0.2 | 0.2425 | 0.6936 | 2.098 | 4.044 |
| `10_ha05_f0` | 0.5 | 0.2372 | 0.6907 | 2.097 | 4.035 |
| `10_ha10_f0` | 1.0 | 0.2427 | 0.6927 | 2.100 | 4.065 |
| `10_fbr_f0` | 0.2 + tessera→fraction bridge | 0.2375 | 0.6952 | 2.129 | 4.121 |

### 33. Cross-encoder gradient/bridge tuning does NOT move height or buildings — null
**RMSE_V is flat across α** (4.044 → 4.035 → 4.065): feeding the optical encoder more height-gradient
does nothing for height. **Hypothesis falsified** — our height ceiling is *not* a gradient-routing
problem. IoU_B also flat (~0.24); the tessera→fraction bridge slightly *hurt* (RMSE_V 4.12, IoU_B
0.238). Combining best-α + fraction-bridge would not give a step (two null/negative levers don't
synergise). Extends f26: the multi-task *plumbing* is not a lever. **Incremental tuning is exhausted**
— Phase 8 (interference/sampling) null, Phase 9 (objective) small-positive, Phase 10 (gradient/bridge)
null. The 7A architecture is capped at IoU_B ~0.24 / RMSE_V ~4.0 internally.

### 34. Decode-resolution audit — the decoder is NOT the bottleneck
Traced the 7A decode path: stem 96ch@256 → encoder skips [96@256, 128@128, 192@64, 256@32] →
bottleneck 384@16 → decoder upsamples 16→32→64→128→**256 with a skip at every level** (incl. the
256-res L0 skip into the final block). So the decoder is **already full-resolution** — we are *not*
bottlenecking through 16×16 or discarding fine detail. Therefore **"raise decode resolution" is not an
available lever**, and a *lighter* U-Net (fewer params, same/!lower res) would not help buildings.
The real limits: (a) **input embeddings are ~10 m-scale** (a building is ~1–3 px; the task is
fractional abundance per coarse pixel, so it's an *extraction/regression-quality* problem, not edge
sharpness); (b) **capacity/extraction vs the field** — top teams get IoU_B ~0.5 / RMSE_V ~3.0 from the
*same* embeddings, so the info is there and we under-extract it. Closing that is a "much better/bigger
model" gap, not a structural-resolution fix.

### Phase 10 verdict & fork
- **Incremental levers (objective/gradient/bridge/sampling) are exhausted.** No more α/bridge sweeps.
- **Resolution is not the lever** (decoder already full-res) → the light-U-Net rebuild is *not*
  justified on resolution grounds (and 6A's concat failure warns against it).
- Remaining step-change candidates are all **capacity/extraction-side and uncertain in ~20 days**:
  heavier decoder, multi-seed ensemble of sm4tv, or a dedicated better *height* sub-model (ours ~4.0
  lags even simple 2A's 3.70). 
- **Honest position:** top-3 (+0.12) is out of reach; the realistic choice is (a) consolidate the
  0.3871 best, or (b) a capacity/height swing knowing it's a long shot. Pending decision.

---

## Phase 11 — Fresh Extraction Architecture: PLATFORM ANCHOR (2026-06-11) — BREAKTHROUGH

We took the fork's option (b) — the capacity swing — and it **worked**. `fresh_extract` (deep dual
pixel encoders + symmetric multi-scale fusion + own height head; ~50M params; sm4tv objective; no
TerraMind), trained on geo-folds {0,2,3,4} / val fold 1 (`11_fresh_f1`), single self-contained model,
no graft. Submitted 2026-06-11.

### 35. The fresh architecture broke the "intrinsic IoU_B ceiling" — and isolated height as the one remaining lever
Platform result `11_fresh_f1` = **0.4125 (rank 43, up from 54)**, vs prior best
`9_sm4tv_final_hybrid` 0.3871 (rank 54): **+0.025**.

| metric | hybrid 0.3871 (platform) | **11_fresh_f1 (platform)** | top-1 ≈ 0.5448 | contribution (w) | **gap to max** |
|--------|--------------------------|----------------------------|----------------|------------------|----------------|
| IoU_build  | ~0.35 | **0.4297** | ~0.53 | 0.107 (0.25) | 0.143 |
| IoU_veg    | —     | **0.8062** | —     | 0.121 (0.15) | **0.029 ✅ near-max** |
| IoU_water  | —     | **0.4797** | —     | 0.072 (0.15) | 0.078 |
| RMSE_build | 2.28  | **2.2256** | ~1.85 | 0.111 (0.25) | 0.139 (already strong) |
| **RMSE_veg** | 3.70 | **3.8069** | ~3.0 | **0.010 (0.20)** | **0.190 🔴 the giant** |

Reverse-engineered proxy predicts 0.421 vs actual 0.4125 (~0.008 off — formula is approximate but the
*ranking* of levers is robust).

**What this overturns:**
- **f28/f34's "IoU_B ~0.18 ceiling is intrinsic / extraction gap unbeatable in 20 days" is FALSIFIED.**
  Platform IoU_B went 0.35 → **0.43** purely by adding capacity + symmetric fusion (Bets 1+2). The gap
  *was* extraction/capacity, and capacity *closed most of it*. RMSE_B is now near the top of the field.
- The "consolidate at 0.3871" option is dead — the swing paid off.

**What it isolates — RMSE_veg is now the entire game:**
- Veg *location* is essentially solved (IoU_V **0.806**, near max). The model finds vegetation; it
  predicts its **height** poorly (3.81 m → earns 0.010 of a possible 0.20). That single metric holds
  **0.190** of unrealized score — more than the other four *combined* room.
- Root cause is a self-inflicted under-investment, not a ceiling:
  1. `11_fresh_f1` ran with **`veg_height_boost = 0.0`** — the loss term *purpose-built to drive RMSE_V
     down* was OFF. We disabled it because Phase-11 fast-split screening called it "a net wash." That
     verdict was a **proxy artifact**: fast-split RMSE_V was underfit/noisy, and the *internal*
     composite under-weighted height vs the platform's real 0.20. The platform says we turned off the
     one knob that matters most.
  2. **Height task weight = 0.64** (`--static-weights 0.65,0.64,1.70`) — inherited verbatim from the
     *old* 7A GradNorm convergence (`7A_base_e90 ep59`), never re-tuned for the fresh model.
  3. Height loss masks **building OR veg jointly** (one shared Huber, δ=0.5 ⇒ ~15 m normalized ⇒
     effectively MSE). Buildings (RMSE_B 2.23 ✓) and veg (RMSE_V 3.81 ✗) share one head — veg height is
     intrinsically higher-variance and is being out-competed.
- Secondary signal: internal fold-1 RMSE_V 3.56 → platform 3.81 = a **generalization gap** (region/year
  shift) on height specifically → variance reduction (all-data final, light embedding domain-aug) helps.

**Verdict:** pivot to a **height-first campaign (Phase 12)**. Levers, cheapest-first: re-enable
`veg_height_boost` + raise the height task weight (judge on RMSE_V directly, multi-fold); then
separate/scale-aware veg-height treatment; then an AlphaEarth-GEDI/DEM → veg-height pathway (Bet 1
strengthening — alpha *encodes* canopy height). IoU_B (0.43→0.5+) and IoU_W (0.48) are the next two,
smaller, levers. RMSE_B is near-max — leave it. Detail: `docs/phase12_height.md`.

---

## Phase 12a — TerraMind fusion ablation on fresh_extract (2026-06-12) — NEGATIVE (drop TerraMind)

We dropped TerraMind for `11_fresh_f1` based on a fast-screen where the *additive* injection
(`pyr += tok`) hurt. Before fully abandoning it we tested whether **cross-attention** (the lit-review
Alt-1 recommendation; CLAIRE/MMFNet; the way TerraMind itself separates token=context vs pixel=detail)
would rescue it. Fast screen throughout: pixel-only deep dual-encoder, train fold 0 / val fold 1, 30 ep,
seed 0, sm4tv, GPUs power-capped 100 W.

### 36. TerraMind does not earn its place on fresh_extract — four fusion variants all tie pixel-only
| arm | fusion | proxy | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V |
|-----|--------|-------|-------|-------|-------|--------|--------|
| `notm` | none (pixel-only) | **0.4074** | 0.3117 | 0.7604 | 0.5896 | **2.108** | **3.824** |
| `tm_add` | residual add | 0.3993 | 0.3093 | 0.7647 | 0.5652 | 2.113 | 3.910 |
| `tm_xattn` | shared gated cross-attn | 0.4078 | 0.3168 | 0.7660 | 0.6211 | 2.191 | 3.851 |
| `12_frac_route_A` | cross-attn → fraction decoder only | 0.4079 | 0.3168 | 0.7661 | 0.6196 | 2.182 | 3.856 |
| `12_frac_loc_B` | frac-only + Gaussian locality bias | 0.4057 | 0.3170 | 0.7661 | 0.6174 | 2.218 | 3.848 |

Four sub-findings:
1. **Cross-attention >> additive** (0.3993 → 0.4078): the mechanism matters exactly as the literature
   predicts — gated attention lets each pixel *select* token context; additive forces the coarse signal
   on every pixel (the modality-dominance failure). Good science, but…
2. **…all xattn variants merely TIE pixel-only.** Whole family sits in a ~0.002 band around `notm`
   0.4074; best (frac-route, 0.4079) = +0.0005 = noise. The IoU bumps (IoU_W +0.03, IoU_B +0.005) are
   exactly cancelled by RMSE wiggles. **TerraMind adds nothing net on this architecture.**
3. **Fraction-only routing did NOT recover height** (A height 2.18/3.86 ≈ shared 2.19/3.85, both > notm
   2.11/3.82) — even though the height decoder is *bit-isolated* from tokens at inference (verified).
   So the height degradation was **never** token-injection into the height decoder; the only remaining
   coupling is **shared-encoder gradients**, or it is **fold noise** (effect is ~0.03 m, single fold).
   This **reframes the earlier "cross-attention hurts regression" question — it was mis-posed.**
4. **Locality bias slightly HURT** (B 0.4057 < A 0.4079). Tobler's-law worry is theoretically valid but
   **empirically did not bite**: TerraMind tokens are *already spatially contextualized* (mixed globally
   in TM pretraining), so a hard spatial prior discards context the tokens legitimately carry.

**Verdict:** TerraMind is **dropped** on fresh_extract (tested add / shared-xattn / frac-route /
frac-loc — none clears pixel-only). The `_TokenCrossAttn` block + `--terramind-fusion {xattn,xattn_frac,
xattn_frac_loc}` are kept in code (off by default) for reuse. The dominant lever was never the patch
tokens — it is **RMSE_V (0.190)**. Proceeding to the height-first run (f37, pending).

---

## Phase 12b — Height campaign: vboost wash on the full model → decoupled height (2026-06-12)

### 37. veg_height_boost lowers RMSE_V but starves building height → net wash on the well-fit model
**Fast screen** (train fold 0 / val fold 1, underfit baseline) made vboost look like a clear win — the
0/2/3/4/6 sweep was monotone-ish, vboost=3 → proxy 0.4156 vs notm 0.4074 (RMSE_V 3.824→3.680). But the
sweep was noisy (b4 dipped below b3; run-to-run ≈ dose differences) and, crucially, **measured against an
underfit 514-tile baseline.**

On the **full cv-fold-1 config** (`12_vb3_f1`, vboost=3, 40 ep, directly comparable to `11_fresh_f1`), it
**washes out** (internal fold-1 val, best-proxy checkpoints):

| | proxy | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V |
|--|-------|-------|-------|-------|--------|--------|
| `11_fresh_f1` (vb0) | 0.4516 | 0.3487 | 0.7805 | 0.6025 | **1.876** | 3.516 |
| `12_vb3_f1` (vb3) | 0.4515 | 0.3471 | 0.7819 | **0.6174** | 1.946 | **3.472** |

vboost=3 bought **RMSE_V −0.044** (✓) and **IoU_W +0.015** (✓) but cost **RMSE_B +0.070** (✗). Score
arithmetic (RMSE_B weight **0.25** > RMSE_V **0.20**): +0.0022 (V) − 0.0044 (B) + 0.0022 (W) ≈ **0** →
tied proxy. **Two lessons:** (a) the fast screen's positive verdict was a *proxy artifact* of an underfit
baseline (cf. f29); (b) **the shared single height head makes vboost a *reallocation*, not a mutual
boost** — emphasising veg pixels starves building-height gradient (buildings & canopy have different
height distributions, so the shared head trades one for the other). Do **not** submit `12_vb3_f1` (wash;
the RMSE_B/RMSE_V weight asymmetry tilts it slightly negative).

### Fix — decoupled height loss (launched 2026-06-12)
`DualPathLoss(decouple_height=True, build_height_weight, veg_height_weight)`: separate, independently-
normalised Huber terms for building vs veg pixels instead of one shared mask + veg-only boost. **Verified**
(disjoint building/veg layout): building-pixel height gradient is bit-identical across veg_w=1 vs 5
(building **not** starved) while veg gradient scales exactly 5× — so we can pull RMSE_V down *without* the
RMSE_B cost. Build/veg pixels are ~spatially disjoint, so one head fits both when neither is gradient-starved.

**Two runs in flight (cv-fold 1, 40 ep, each = `11_fresh_f1` recipe + one change):**
- `12_dech_f1` (node1) — **decoupled height**, build_w=1.0 / veg_w=3.0 (boost veg, hold building).
- `12_xmodal_f1` (node2) — **alpha↔tessera coarse cross-attention** (`--cross-modal coarse`; Phase 12c,
  the IoU_B/capacity play, orthogonal to height).
