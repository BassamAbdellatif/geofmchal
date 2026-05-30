# Experiment 6A — TESSERA Cross-Attention: Analysis & Lessons Learned

**Branch:** `exp-6-tessera-xattn`  
**Date:** 2026-05-30  
**Status:** Concluded — architecture not yet competitive with baseline. See [Next Steps](#next-steps).

---

## 1. Hypothesis

The 2A_vegboost submission (platform score 0.3721, current best) uses AlphaEarth (64ch annual composite) as the only pixel-level backbone. The hypothesis driving experiment 6A was:

1. **TESSERA** (128ch, S1/S2 time-series) captures phenological variation that a single annual composite cannot. Adding it as a second pixel stream should improve vegetation and water predictions.
2. **Cross-attention** at the U-Net bottleneck (pixel queries × patch key/values) should extract richer semantic guidance from the TerraMind patch embeddings than the existing spatial-broadcast fusion does.

---

## 2. Architecture (`YNetTesseraXAttn`)

```
Pixel stems
  alpha_earth (64ch) ──┐
  tessera (128ch)  ────┴──► 1×1 conv (192→64ch) ──► U-Net Encoder (5 down-blocks)
                                                              │
                                                     Bottleneck (16×16 @ 512d)
                                                              │
                                            ┌─────────────────┘
                                            │   Multi-Head Cross-Attention
                                            │   Q = pixel bottleneck tokens  (256 × 512d)
                            patch inputs ──►│   K,V = patch embeddings (256 × 1536d → 512d)
                                            └─────────────────┐
                                                              │
                                                     U-Net Decoder
                                                     (shared, 4 up-blocks)
                                                              │
                                        ┌─────────────────────┴─────────────────────┐
                               Classification head                          Height head
                               (ch 0–2: abundance)                  (ch 3: nDSM)
                                                                     GradScale α=0.1
```

**Key design choices:**
- Shared encoder/decoder, split only at the final head layer.
- GradScale hook (α=0.1) on the height decoder prevents its gradients from overwriting the classification encoder (inherited from `exp-4-ynet-gradhook`).
- Patch key/value projection: 1536ch → 512ch linear before attention.

---

## 3. Runs Executed

Four full 60-epoch runs, in chronological order:

| Run name | Pixel inputs | Patch inputs | Dyn. loss | Epoch time | Best Epoch | Best Proxy |
|---|---|---|---|---|---|---|
| `6A_alpha_only_xattn` | alpha_earth | ts1+ts2 | No | ~115s | 57 | **0.2436** |
| `6A_alpha_only_xattn_dloss` | alpha_earth | ts1+ts2 | Yes | ~115s | 55 | 0.2132 |
| `6A_tessera_broadcast` | alpha+tessera | ts1+ts2 | No | — | — | (aborted) |
| `6A_tessera_xattn` | alpha+tessera | ts1+ts2 | No | ~1000s | 44 | 0.1731 |
| `6A_tessera_xattn_ts1only` | alpha+tessera | ts1 only | No | ~750s | 49 | 0.1561 |

Proxy formula (C=4.0):
```
proxy = 0.25×IoU_B + 0.15×IoU_V + 0.15×IoU_W
      + 0.25×max(0, 1 − RMSE_B / 4.0)
      + 0.20×max(0, 1 − RMSE_V / 4.0)
```

---

## 4. Results at Best Checkpoint

| Run | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Proxy |
|---|---|---|---|---|---|---|
| **Platform baseline** (2A_vegboost) | **0.3394** | **0.7649** | **0.3695** | **2.27m** | **3.74m** | **~0.372** |
| `6A_alpha_only_xattn` | 0.1683 | 0.7550 | 0.4075 | 3.57m | 4.03m | 0.2436 |
| `6A_alpha_only_xattn_dloss` | 0.1397 | 0.7410 | 0.2464 | 3.55m | 3.96m | 0.2132 |
| `6A_tessera_xattn` | 0.0171 | 0.5668 | 0.4294 | 3.69m | 4.19m | 0.1731 |
| `6A_tessera_xattn_ts1only` | 0.0078 | 0.5759 | 0.4516 | 4.26m | 4.24m | 0.1561 |

---

## 5. Lessons Learned

### 5.1 TESSERA pixel input destroys building detection

Adding the TESSERA stream drops IoU_B from 0.168 → 0.017 — a 10× regression. The 192ch→64ch 1×1 projection appears to discard building-discriminative features when the two pixel stems are mixed. The model falls back to learning water/vegetation (spatially smooth, large-area classes) while completely failing on buildings (small, high-frequency).

The epoch time blowup (115s → 1000s per epoch) with no accuracy gain makes TESSERA in the pixel stem a non-starter in its current form.

**Conclusion:** do not concatenate TESSERA as a pixel input without a different integration strategy (e.g. a separate early-fusion branch that only feeds into certain skip levels).

### 5.2 Alpha-only + cross-attention converges cleanly but is 35% below baseline

`6A_alpha_only_xattn` trains stably: validation loss decreases monotonically, no spikes, checkpoint at epoch 57. Despite this, its proxy (0.24) is far below 2A_vegboost (~0.37). The gap lives almost entirely in two metrics:

- **IoU_B**: 0.168 (6A) vs 0.339 (2A platform) — buildings not being segmented well.
- **RMSE_B**: 3.57m (6A) vs 2.27m (2A platform) — building heights systematically underestimated.

This suggests the cross-attention bottleneck is not improving the encoder's representation of buildings. The 16×16 query grid is too coarse to localise building footprints (typically <5 pixels across at 10m resolution).

### 5.3 Dynamic loss hurts IoU_W significantly

With dynamic loss (`dloss`), the height-boost weight ramps from 1× to 5× over 60 epochs, causing train loss to increase monotonically past epoch 15 — the model is re-weighting height so aggressively that abundance learning stagnates. IoU_W drops from 0.408 → 0.246. RMSE_V improves only marginally (4.03 → 3.96m) and is not worth the trade-off.

**Conclusion:** dynamic loss is counter-productive with this architecture. The static height weight of 2.0 (loss lambdas) is sufficient.

### 5.4 RMSE_V is pinned just above the scoring cliff

The proxy formula scores zero for RMSE_V > 4.0. `6A_alpha_only_xattn` lands at 4.027m — just 27mm above the threshold that would earn 0.20 × (1 − 4.027/4.0) ≈ 0. Getting RMSE_V to 3.9m would add ~0.005 proxy points. It is not the bottleneck; IoU_B is.

### 5.5 Both patch embedding streams (ts1 + ts2) are needed

Removing `terramind_s2` (`_ts1only`) worsens every metric, particularly RMSE_B (+0.7m). The S1 and S2 embeddings carry complementary information.

### 5.6 Cross-attention adds no measurable benefit over 2A broadcast fusion

Comparing `6A_alpha_only_xattn` (proxy 0.24) to the existing `2A_vegboost` platform score (proxy ~0.37), the cross-attention does not recover the gap. The key architectural differences between 2A and 6A are:

| | 2A (YNetAttentionFusedDecoder) | 6A (YNetTesseraXAttn) |
|---|---|---|
| Fusion | Spatial broadcast of patch at bottleneck | MHA cross-attention at bottleneck |
| Decoders | Two separate (class + height) | One shared decoder + split heads |
| Pixel stem | alpha_earth only (64ch) | alpha_earth only (64ch) in best variant |

The two-decoder design in 2A likely helps the height decoder specialise without contaminating classification gradients even with the GradScale hook in place.

---

## 6. Next Steps

### Immediate — quick wins on 2A_vegboost (no retraining)

1. **Threshold calibration**: scan thresholds 0.05–0.95 per channel on the validation set; find the value that maximises hard binary IoU per channel; apply at inference time in `predict.py`. Expected: IoU_W 0.37→0.42+, IoU_B 0.34→0.38+. Implementable in ~1 hour.

2. **TTA (Test-Time Augmentation)**: 8-fold (4 rotations × 2 flips), average predictions. Already implemented in `predict.py` — just run with `--tta`. Expected: IoU +0.02–0.04, RMSE −0.2–0.4m.

### Medium-term — retrain 2A variants

3. **Replace Tversky/SSIM/GDL with pure MSE for channels 0–2**: Tversky is designed for binary segmentation; our targets are continuous abundances. MSE is better calibrated for fractional outputs and the hard-IoU-at-0.5 evaluation. Likely the approach used by top-scoring teams with few submissions.

4. **Ensemble of 2–3 seeds**: train 2A_vegboost with seeds 42, 123, 456 on the full dataset (no val split) and average predictions. Low risk, typically +0.01–0.02.

### Lower priority — revisit 6A if above is exhausted

5. **Separate height decoder in 6A**: restore the two-decoder design from 2A (separate `decoder_class` and `decoder_height`) while keeping the cross-attention bottleneck. The shared decoder is likely why buildings and height predictions interfere.

6. **TESSERA as a late-fusion modality**: rather than mixing tessera into the pixel stem, inject it only into specific decoder skip-connection levels where temporal signal is most useful (e.g. vegetation / water skip levels only).

---

## 7. Current Branch State

All experiment 6A code is on branch `exp-6-tessera-xattn`. Key files modified relative to `main`:

| File | Change |
|---|---|
| [core/model.py](../core/model.py) | Added `YNetTesseraXAttn` class and `ynet_tessera_xattn` dispatch |
| [core/dataset.py](../core/dataset.py) | Refactored to support multi-pixel-stream loading (tessera + alpha_earth) |
| [train.py](../train.py) | Wired new model type; updated training_params.txt logging |
| [predict.py](../predict.py) | Minor fixes for multi-stream inference |
| [prompts/exp-6-tessera-xattn.md](../prompts/exp-6-tessera-xattn.md) | Full experiment spec and acceptance criteria |
| [prompts/experiments.md](../prompts/experiments.md) | Updated experiment log |

The `runs/` symlink (→ `/mnt/head/users/bassam/data/geofmdata/runs/`) is gitignored; run artifacts live on the shared NFS volume.
