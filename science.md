# GeoFM Challenge — Science & Ideas Reference

## 1. Task Formulation

### What we are predicting
Each input is a 256×256 tile at 10m/pixel resolution. The model outputs a **4-channel map**:

| Channel | Target | Range | Platform metric |
|---------|--------|-------|-----------------|
| 0 | Building abundance fraction | 0–1 | IoU_B (hard threshold @ 0.5), RMSE_B |
| 1 | Vegetation abundance fraction | 0–1 | IoU_V (hard threshold @ 0.5), RMSE_V |
| 2 | Water abundance fraction | 0–1 | IoU_W (hard threshold @ 0.5) |
| 3 | Relative building height (nDSM) | 0–1.5 (clipped) | RMSE_B, RMSE_V (on masked pixels only) |

**Abundance fractions** are computed by aggregating 1m-resolution binary masks to 10m — so a pixel with 60% building cover has label 0.6. This is **regression**, not binary classification.

**Height normalisation**: `clip(nDSM / 30.0, 0, 1.5)`. HEIGHT_NORM_CONSTANT = 30.0.

**RMSE_B** measures height error only on pixels where building fraction > 0 (building pixels). **RMSE_V** measures height error only on pixels where vegetation fraction > 0. These are the hardest metrics because the model must jointly regress abundance AND height.

### Platform scoring formula (reverse-engineered, C=3.9, R²=0.996)
```
score = 0.25×IoU_B + 0.15×IoU_V + 0.15×IoU_W
      + 0.25×max(0, 1 − RMSE_B / 3.9)
      + 0.20×max(0, 1 − RMSE_V / 3.9)
```
RMSE above 3.9 m contributes zero. IoU is computed at **hard threshold 0.5** on predicted fractions.

### Why this is hard
- The loss should be designed for **continuous regression** (fractions), but the eval metric applies a **hard binary threshold** — these are in tension.
- Losses designed for binary segmentation (Tversky, Dice) do not directly optimise for the hard-IoU-at-0.5 metric.
- Height is a separate regression target but shares the encoder with the fraction channels — gradients can interfere.

---

## 2. Inputs (GeoFM Embeddings)

The model does **not** see raw satellite pixels. It receives pre-computed embeddings from foundation models.

| Embedding | Type | Shape | Source model |
|-----------|------|-------|--------------|
| `alpha_earth` | Pixel-level | 64 × 256 × 256 | AlphaEarth (optical) |
| `terramind_s1` | Patch-level | 768 × 16 × 16 | TerraMind (SAR/S1) |
| `terramind_s2` | Patch-level | 768 × 16 × 16 | TerraMind (optical/S2) |
| `thor_s1` | Patch-level | 768 × 16 × 16 | THOR (SAR/S1) |
| `thor_s2` | Patch-level | 768 × 16 × 16 | THOR (optical/S2) |

Terramind_s1 + terramind_s2 concatenated = 1536 channels. Thor variants are alternative foundation models.

**Best combination found**: `alpha_earth` (pixel) + `terramind_s1,terramind_s2` (patch).

---

## 3. Architecture Families

### Family 2A — Attention Fusion (U-Net + bottleneck patch fusion)

```
alpha_earth (64ch, 256×256)
       │
   [U-Net Encoder]
       │
   Bottleneck ←── concat(terramind_s1, terramind_s2) (1536ch, 16×16)
       │              [linear projection + spatial broadcast]
   [U-Net Decoder]
       │
   [Single head] → 4 channels (fractions + height)
```

- Single shared decoder for all 4 output channels
- Patch embeddings fused at bottleneck via cross-attention or concatenation
- Simpler, faster to train (batch size 16 fits easily)
- **Best platform score achieved: 0.3721** (2A_vegboost)

### Family 3A — Y-Net (decoupled decoders, no gradient control)

```
alpha_earth (64ch, 256×256)
       │
   [Shared Encoder]
       │ ↓ bottleneck fusion with patches
  ┌────┴────┐
  │         │
[Decoder   [Decoder
 Class]     Height]
  │         │
ch 0-2    ch 3
```

- Two separate decoders branching from the shared encoder bottleneck
- Motivation: height regression and fraction classification have different spatial characteristics
- **No gradient isolation** between decoders — height loss still flows back through encoder
- Result: worse than 2A on RMSE_B (3.4–3.7m vs 3.1m), marginally better IoU_V

### Family 4A — Y-Net + GradScale Hook

Same as 3A but with a **GradScale hook** (α=0.1) on the height decoder branch:

```
Forward:  height decoder sees full encoder features
Backward: height decoder gradients scaled to 10% before reaching encoder
```

- Prevents height regression gradients from disrupting the classification encoder
- **Principle**: let the height decoder specialise without poisoning the shared representation
- Result: best internal proxy (0.6465 with old C=30 proxy), but IoU_B didn't improve over 2A

### Family 5A — Y-Net + GradScale + HeightBoost Curriculum

Same as 4A but with a **dynamic loss weight** on height:

```
height_loss_weight = 1.0 + (epoch / max_epochs) × 4.0
                   → ramps from 1.0× at epoch 1 to 5.0× at epoch 60
```

- Motivation: let the model first learn fractions, then progressively emphasise height
- Result: **worse than 4A** — RMSE_B regressed to 3.4–3.5m (vs 3.1m in 2A). The ramp-up over-weighted height at the expense of building fraction by the end of training.

---

## 4. Loss Functions

### 4.1 Original composite (baseline)
```
loss = Tversky(pred, target) + 2 × MAE(pred, target)
```
Used in all 2A nologits experiments. Proxy tracked as `Tversky + 2×MAE`.

**Problem**: Tversky is a binary segmentation loss. For continuous abundance targets (0–1), it does not directly optimise for IoU-at-0.5.

### 4.2 ImprovedCompositeLoss
```
loss = λ₁×MAE + λ₂×SSIM + λ₃×GDL + λ₄×Tversky + building_height_boost
```
- **MAE**: pixel-wise L1, good for regression
- **SSIM**: structural similarity, preserves spatial coherence
- **GDL** (Gradient Difference Loss): penalises blurry edges
- **Tversky**: pushes binary predictions toward IoU-at-0.5
- **building_height_boost**: extra MAE loss computed only on pixels where building fraction > 0.1

Default lambdas: [1.0, 0.5, 0.5, 2.0].

### 4.3 ImprovedCompositeLoss + VegBoost (current best)
```
loss = ImprovedCompositeLoss + veg_height_boost
```
- **veg_height_boost**: extra MAE loss computed only on pixels where vegetation fraction > 0.1
- Symmetric counterpart to building_height_boost
- **Motivation**: RMSE_V was our worst metric (3.74m, nearly zero contribution to score)
- **Result**: +0.006 platform score (0.366 → 0.372). Less than expected (+0.038 predicted).

### 4.4 Pure MSE variants (FAILED)
Three variants tried: `mse`, `mse_sigma`, `mse_vegboost`.

```
loss = MSE(pred, target)           # mse
loss = MSE / σ²(target)            # mse_sigma (uncertainty-weighted)
loss = MSE + height_boosts         # mse_vegboost
```

**All failed catastrophically** for IoU metrics:
- IoU_B collapsed to 0.02–0.06 (vs 0.19 with composite)
- IoU_W collapsed to 0.16–0.26 (vs 0.57 with composite)

**Why**: Pure MSE minimises average pixel error but applies no pressure at the 0.5 threshold needed for hard-IoU. A model outputting 0.3 and one outputting 0.7 have the same MSE if target is 0.5, but have opposite IoU outcomes. Tversky/Dice losses specifically create gradient pressure around the decision boundary.

---

## 5. Training Strategies Tried

| Strategy | Tried | Result |
|----------|-------|--------|
| ReduceLROnPlateau (factor=0.5, patience=2) | ✅ | Used in 2A family |
| CosineAnnealingLR | ✅ | Used in 3A/4A/5A family |
| Dynamic (curriculum) loss weight | ✅ | No benefit over fixed in 4A |
| Data augmentation (flips/rotations) | ✅ | Hurt performance slightly (RMSE_B worse) |
| GradScale hook α=0.1 | ✅ | Helped Y-Net but not enough vs 2A overall |
| HeightBoost curriculum 1x→5x | ✅ | Hurt (over-weighted height late) |
| Building height boost | ✅ | Part of composite loss |
| Vegetation height boost | ✅ | +0.006 platform score |
| Batch size 16 | ✅ | Used in 2A early |
| Batch size 32 | ✅ | Used in 3A/4A/5A/2A-later |
| Proxy with C=30 (old) | ✅ | Blind to RMSE contribution — misleading |
| Proxy with C=4.0 (fixed) | ✅ | Current standard |
| Multi-node parallel training | ✅ | 3-4 nodes used for 5A seeds |

---

## 6. Inference Strategies

| Strategy | Tried | Notes |
|----------|-------|-------|
| Direct prediction (single forward pass) | ✅ | Current standard |
| TTA — 8-fold (4 rotations × 2 flips) | ❌ | Planned, expected +0.01–0.03 IoU |
| Threshold calibration per channel | ❌ | Planned, expected +0.02–0.04 IoU |
| Ensemble of multiple seeds | ❌ | 3×5A seeds available, easy to do |
| Guided filter on height output | ❌ | Post-processing to sharpen height edges |

---

## 7. Ideas Not Yet Tried (Ranked by Expected Value)

### High value — no retraining needed
1. **Per-channel threshold calibration**
   - Scan threshold 0.05–0.95 on validation set per channel
   - Pick threshold maximising hard binary IoU (not fixed 0.5)
   - Apply at predict time
   - Expected: +0.02–0.04 on IoU_B and IoU_W

2. **TTA (Test-Time Augmentation)**
   - 8 transforms (D4 symmetry group): 4 rotations × 2 flips
   - Average predictions in **logit space** before sigmoid (more stable than post-sigmoid averaging)
   - Expected: +0.01–0.03 IoU, −0.1–0.3m RMSE
   - Can be combined with threshold calibration

3. **Ensemble of 5A seeds**
   - Three models already trained: `5A_vegboost`, `5A_vegboost_s1`, `5A_vegboost_s2`
   - Average raw predictions → apply optimal thresholds
   - Low effort

### Medium value — requires retraining
4. **Dedicated binary classification head for buildings**
   - IoU_B is our biggest gap (0.34 vs top team 0.52)
   - Add an auxiliary sigmoid head trained with BCE at threshold 0.5 explicitly
   - Joint training with the fraction regression head

5. **Full-dataset training (no validation split)**
   - Currently 80/20 train/val split → 1619/405 samples
   - At end of competition, retrain best model on all 2024 samples
   - Typically +1–2% on test metrics

6. **Better loss for IoU_B**
   - Tversky/Dice losses operate in [0,1] but targets are fractions
   - Explore: focal loss on thresholded predictions, or a differentiable hard-IoU surrogate

### Low value / experimental
7. Guided filter post-processing on height channel
8. Higher resolution crops / multi-scale training
9. Learned threshold (trainable sigmoid temperature per channel)
