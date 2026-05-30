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
- The **test set is in different regions and years** than training. Domain shift is the dominant generalisation challenge. Random validation splits hide this.

---

## 2. Inputs (GeoFM Embeddings)

The model does **not** see raw satellite pixels. It receives pre-computed embeddings from foundation models.

| Embedding | Type | Shape | Source model | Best at |
|-----------|------|-------|--------------|---------|
| `alpha_earth` | Pixel-level | 64 × 256 × 256 | AlphaEarth (annual optical composite) | Spatial detail, urban structure |
| `tessera` | Pixel-level | 128 × 256 × 256 | TESSERA (S1/S2 time series) | Phenology, temporal cover variation |
| `terramind_s1` | Patch-level | 768 × 16 × 16 | TerraMind (SAR/S1) | Vertical structure, surface geometry |
| `terramind_s2` | Patch-level | 768 × 16 × 16 | TerraMind (optical/S2) | Surface composition, spectral signatures |
| `thor_s1` | Patch-level | 768 × 16 × 16 | THOR (SAR/S1) | Alternative SSL objective, ensembling material |
| `thor_s2` | Patch-level | 768 × 16 × 16 | THOR (optical/S2) | Alternative SSL objective, ensembling material |

**Modality-to-task priors** (driving 7A design):
- AlphaEarth: annual optical composite — captures static spatial structure (buildings, road networks). Building-relevant signal.
- TESSERA: time-series-derived — captures phenological variation. Vegetation- and water-relevant signal.
- S1 (SAR): backscatter and range — captures vertical structure. Height-relevant signal.
- S2 (optical): multi-spectral surface reflectance — captures cover composition. Fraction-relevant signal.

**Best combination found through 5A**: `alpha_earth` (pixel) + `terramind_s1,terramind_s2` (patch).

**6A finding**: TESSERA cannot be naively concatenated with AlphaEarth in a shared pixel encoder. The 192→64ch projection forces high-frequency spatial features and low-frequency temporal features to compete, and temporal features dominate in variance. This destroys building discrimination. TESSERA needs its own encoder pathway.

---

## 3. Architecture Families

### Family 2A — Attention Fusion (U-Net + bottleneck patch broadcast)

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

- Single shared decoder for all 4 output channels (later variants split into two-decoder Y-Net)
- Patch embeddings fused at bottleneck via linear projection + spatial broadcast
- Simpler, faster to train
- **Best platform score achieved: 0.3721** (2A_vegboost). Still our submitted best.

### Family 3A — Y-Net (decoupled decoders, no gradient control)

Two separate decoders branching from the shared encoder bottleneck. Motivation: height regression and fraction classification have different spatial characteristics. **No gradient isolation** between decoders — height loss still flows back through encoder.

Result: worse than 2A on RMSE_B (3.4–3.7m vs 3.1m), marginally better IoU_V.

### Family 4A — Y-Net + GradScale Hook

Same as 3A but with a **GradScale hook** (α=0.1) on the height decoder branch: forward pass unchanged, but backward pass scales height-decoder gradients to 10% before they reach the encoder.

Motivation: let the height decoder specialise without poisoning the shared encoder representation.

Result: best internal proxy (0.6465 with the misleading old C=30 proxy), but IoU_B did not improve over 2A. The shared encoder still mediated the modality tension.

### Family 5A — Y-Net + GradScale + HeightBoost Curriculum

Same as 4A but with `height_loss_weight = 1.0 + (epoch / max_epochs) × 4.0`.

Result: **worse than 4A**. The ramp-up over-weighted height at the expense of building fraction by the end of training. RMSE_B regressed to 3.4–3.5m (vs 3.1m in 2A).

### Family 6A — TESSERA pixel stream + cross-attention bottleneck

```
alpha_earth (64ch) ─┐
                    ├─► 1×1 conv (192→64ch) ─► [shared U-Net Encoder]
tessera (128ch) ────┘                                  │
                                              Bottleneck (16×16)
                                                       │
                              patch tokens ─► [MHA cross-attention]
                                                       │
                                              [shared U-Net Decoder]
                                                       │
                                       ┌───────────────┴───────────────┐
                                  [class head]                  [height head]
                                  ch 0,1,2                      ch 3 (GradScale α=0.1)
```

Two changes vs 2A: (1) add TESSERA at the pixel stem, (2) replace bottleneck broadcast with multi-head cross-attention.

Result: **not competitive** with 2A. Best proxy 0.244 vs 2A's 0.326.
- Concatenating TESSERA with AlphaEarth at the input destroyed IoU_B (0.168 → 0.017)
- Cross-attention with the new shared-decoder + split-heads structure (instead of 2A's two-decoder Y-Net) regressed proxy by ~0.08

Two findings, both used in 7A:
1. **Shared encoders force destructive modality competition.** Decouple by giving each modality its own encoder.
2. **Shared decoders force destructive task competition.** Keep the full Y-Net decoupling at the bottleneck.

### Family 7A — Decoupled dual-encoder / dual-decoder with sensor-routed patches (planned)

```
alpha_earth (64ch, 256×256)                    tessera (128ch, 256×256)
        │                                                │
   [α-stem, conv blocks]                         [τ-stem, conv blocks]
        │                                                │
   [α-encoder, U-Net, 5 levels]                [τ-encoder, U-Net, 5 levels]
        │                                                │
   α-bottleneck (16×16)                        τ-bottleneck (16×16)
        │                                                │
        │   ─ S2 patch tokens ─►[cross-attn]            │
        │      (TerraMind S2)        │                   │
        │                            ▼                   │
        ▼                                                │
[Fraction Decoder, α-skips]                              │
 + S2 patch injection at decoder levels 16×16 and 32×32  │
        │                                                │
 ┌──┬──┬──────────────────┐                              │
[B][V][W][Auxiliary binary B head]                       │
                                                         ▼
                                       [Height Decoder, τ-skips]
                                        + α-bottleneck side-input at first block
                                        + S1 patch injection at decoder
                                          levels 16×16 and 32×32
                                                         │
                                                  [Height head, 1ch]
                                                  GradScale α=0.2 at branch entry
```

**Key design principles:**
- **Two encoders, two decoders.** Each modality gets its own encoder. Each task gets its own decoder.
- **Modality-to-encoder routing.** AlphaEarth → "urban-spatial" path (feeds fraction decoder). TESSERA → "temporal-natural" path (feeds height decoder, which is mostly about vegetation height once buildings are accounted for).
- **Sensor-to-decoder routing for patch tokens.** S1 (SAR, structure-sensitive) → height decoder. S2 (optical, composition-sensitive) → fraction decoder.
- **Skip injection of patches, not bottleneck injection.** Patches enter via cross-attention at decoder levels matching the patch grid resolution (16×16) and its 2× upsample (32×32). The encoders remain unimodal pixel pathways.
- **Cross-encoder bridge for height only.** The height decoder receives the α-bottleneck as a side input at its first block — the only point where the two encoder pathways meet. GradScale α=0.2 prevents height loss from disrupting the α-encoder's fraction-prediction specialisation.
- **Auxiliary binary B head** on the fraction decoder, trained against `(building_fraction > 0.5)` with BCE + soft Dice. Directly targets IoU_B (our worst metric) without compromising the fraction regression.

**Six structural claims, each ablatable in Phase 5:**

| # | Claim | Ablation |
|---|-------|----------|
| 1 | Encoder modality split helps | Single encoder with α and τ concatenated at input |
| 2 | Decoder task split helps | Single decoder + heads (collapse Y-Net) |
| 3 | Sensor-routed patches help | Both S1 and S2 to both decoders |
| 4 | Skip injection > bottleneck injection | Patches at encoder bottleneck instead |
| 5 | Cross-encoder bridge for height helps | Height decoder uses τ-encoder only |
| 6 | Auxiliary binary B head helps IoU_B | Drop the binary head |

Combined with data-preparation changes (geographic CV, stratified sampling, channel standardisation, embedding-space domain-shift augmentation) and inference-time strategies (D4 TTA, per-channel threshold calibration, binary-head blending, 3-seed ensemble).

See `prompts/exp-7-clean-slate.md` for full specification.

---

## 4. Loss Functions

### 4.1 Original composite (2A early)
```
loss = Tversky(pred, target) + 2 × MAE(pred, target)
```

### 4.2 ImprovedCompositeLoss (2A late, 4A, 5A)
```
loss = λ₁×MAE + λ₂×SSIM + λ₃×GDL + λ₄×Tversky + building_height_boost
```
Default lambdas: [1.0, 0.5, 0.5, 2.0].

### 4.3 ImprovedCompositeLoss + VegBoost (best to date)
```
loss = ImprovedCompositeLoss + veg_height_boost
```
- **veg_height_boost**: extra MAE loss on pixels where vegetation fraction > 0.1
- Symmetric counterpart to building_height_boost
- **Result**: +0.006 platform score over the same loss without it.

### 4.4 Pure MSE variants (FAILED)
Three variants tried (`mse_sq`, `mse_sigma`, `mse_vegboost`). All failed catastrophically — IoU_B collapsed to 0.02–0.06. Pure MSE minimises average pixel error but applies no pressure at the 0.5 threshold needed for hard-IoU. **Tversky/Dice components are essential.**

### 4.5 Planned 7A loss

```
L_fraction = MAE + λ · SoftDice(sigmoid(k · (pred − 0.5)), (target > 0.5))   # k=5
L_height   = Huber(pred, target, δ=0.5) on (building_frac > 0 OR veg_frac > 0)
           + 0.1 · Huber on the complement
L_binary   = BCE + SoftDice on (building_frac > 0.5).float()
L = w_frac · L_fraction + w_height · L_height + w_binary · L_binary
```

- **Soft Dice on shifted sigmoid (k=5)** is a continuous relaxation of hard IoU at 0.5. Creates gradient pressure at the decision boundary while remaining differentiable.
- **Huber instead of MAE** for height: smoother gradients on large errors.
- **No SSIM, no GDL.** They regularise toward smoothness, which hurts the sharp boundaries that drive IoU_B.
- **Task weights `w_*` learned by GradNorm** (Chen et al. 2018, α=1.5) — equalises gradient norms through the relevant encoder for each task. Replaces hand-tuned weights and the dynamic curriculum that backfired in 5A.

---

## 5. Training Strategies Tried (and what we learned)

| Strategy | Tried | Result |
|----------|-------|--------|
| ReduceLROnPlateau (factor=0.5, patience=2) | ✅ | Used in 2A family |
| CosineAnnealingLR | ✅ | Used in 3A/4A/5A/6A family |
| Dynamic (curriculum) loss weight | ✅ | **Hurts.** No benefit in 4A, hurt in 5A, hurt in 6A. Don't use. |
| Geometric data augmentation (flips/rotations) | ✅ | Slightly hurt 4A — likely a crop-diversity bug. Re-investigate with the worker_init_fn fix. |
| Domain-shift augmentation on embeddings | ❌ | Planned in 7A. Channel gain/offset, dropout, noise. Targets train→test distribution shift. |
| GradScale hook α=0.1 | ✅ | Helped Y-Net but not sufficient on its own |
| HeightBoost curriculum 1x→5x | ✅ | **Hurts.** Don't use. |
| Building height boost (static) | ✅ | Helpful. Part of composite loss. |
| Vegetation height boost (static) | ✅ | Helpful (+0.006 platform). |
| Random validation split | ✅ | Leaks spatial context. **Replace with geographic CV (7A).** |
| Stratified tile sampling | ❌ | Planned in 7A. |
| GradNorm task weighting | ❌ | Planned in 7A. |
| Multi-node parallel training | ✅ | 3-4 nodes used for 5A seeds |

---

## 6. Inference Strategies

| Strategy | Tried | Notes |
|----------|-------|-------|
| Direct prediction (single forward pass) | ✅ | Current standard |
| TTA — 8-fold D4 (4 rotations × 2 flips) | ❌ | Planned in 7A. Average in pre-sigmoid logit space, not post-sigmoid. |
| Per-channel threshold calibration | ❌ | Planned in 7A. Scan 0.05–0.95 on val, pick threshold maximising hard binary IoU. |
| Auxiliary binary head + blending | ❌ | Planned in 7A. At inference, channel 0 = max(fraction_head_B, binary_head_B) where binary head confident. |
| Ensemble of multiple seeds | ❌ | 3×5A seeds available, easy to do. Reserved for 7A Phase 5. |
| Guided filter on height output | ❌ | Deferred — small expected gain, high engineering cost. |

---

## 7. Open Scientific Questions

### 7.1 How do we know whether cross-attention helps over broadcast fusion?
The 6A experiment cannot answer this because shared-decoder regression dominated. 7A's structural claim 4 (skip injection > bottleneck injection) is partly testable here, but a clean "cross-attention vs broadcast at the bottleneck" comparison remains a Phase 5 ablation.

### 7.2 How transferable is the modality-to-task prior?
The 7A design assumes AlphaEarth helps urban-spatial tasks and TESSERA helps temporal-natural tasks. The 6A evidence supports the *negative* version (mixing them hurts buildings). The *positive* version (TESSERA actively helps vegetation/water) remains to be tested. 7A's encoder-split ablation (claim 1) provides this.

### 7.3 How robust is the geographic validation split to choice of K?
KMeans K=5 over lat/lon centroids is arbitrary. Test sensitivity in Phase 5.

### 7.4 Is THOR additive or redundant with TerraMind?
Both are S1/S2 foundation models with different SSL objectives. Could be free ensembling material or could be redundant. Deferred to 7A Phase 5.

---

## 8. Ideas Not Yet Tried (Ranked by Expected Value)

### High value — incorporated in 7A
- Geographic CV split — addresses domain shift
- Stratified tile sampling — addresses zero-inflated labels
- Per-channel standardisation — necessary for attention stability
- Embedding-space domain-shift augmentation — addresses train/test distribution gap
- Decoupled encoders by modality — addresses 6A modality competition
- Sensor-routed patch tokens with skip injection — exploits sensor-task priors
- Auxiliary binary B head with BCE + soft Dice — directly targets IoU_B
- GradNorm task weighting — replaces failed dynamic-weight curriculum
- D4 TTA with pre-sigmoid logit averaging
- Per-channel threshold calibration on the geographic val set
- Binary-head blending at inference

### Medium value — deferred to 7A Phase 5
- THOR foundation model integration (sensor-routed alongside TerraMind)
- 3-seed ensemble for final submission
- Full-dataset training (no val split) for final ensemble members

### Low value / experimental
- Guided filter post-processing on height channel
- Higher resolution crops / multi-scale training
- Learned threshold (trainable sigmoid temperature per channel)
- Dedicated water-only decoder branching off τ-encoder (only if IoU_W is the limiting metric after 7A baseline)

### Explicitly rejected
- Transformer encoder on top of embeddings — reasoning already done in the foundation model
- Diffusion refinement — high engineering cost for ~0.01
- MoE across foundation models — training instability vs. timeline
- SSL pretraining on the training set — doubles up; embeddings are already SSL
- Heavy HP search — spatial CV + principled design dominates HP tuning
- Pure MSE losses — destroy IoU (6A and 2A_mse evidence)
- Dynamic / curriculum loss weights — consistently hurt across 4A, 5A, 6A