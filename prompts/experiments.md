# Emb2Heights Architecture Experiment Tree

**Goal:** Optimize multi-modal fusion of high-res spatial (Alpha/Tessera) and low-res semantic (TerraMind/THOR) embeddings.

## The Decision Tree

* **[ ] Branch 1: Early Fusion** (Upsample 16x16 -> 256x256, Concat at input)
    * *Status:* Skipped (Inefficient, potential semantic dilution).
* **[ ] Branch 3: Cross-Attention Fusion** (Query=Patch, Key/Value=Pixel)
    * *Status:* Pending (Fallback if Branch 2 plateaus).
* **[x] Branch 2: Two-Stream Bottleneck Injection** (Extract spatial, inject patch at bottleneck)
    * *Status:* **ACTIVE BRANCH**
    * **[x] Option 2A: Attention-Gated Skip Connections**
        * *Status:* Best platform score (0.3660). Git: `exp-2A-attention-gate`
        * *Hypothesis:* Filtering spatial skip connections using S1 context will preserve height gradients.
    * **[ ] Option 2B: Multi-Scale Feature Injection (FPN)**
        * *Status:* Pending (Use if Option 2A struggles to resolve large building footprints).
    * **[ ] Option 2C: Deep Supervision**
        * *Status:* Pending (Use if validation loss stalls early in training).
* **[x] Branch 4: Y-Net Decoupled + GradScale Hook**
    * Git: `exp-4-ynet-gradhook`
    * *Status:* Active. Dual decoder (class + height). GradScale α=0.1 on height decoder.
    * **[x] Option 4A: Hook baseline** — `4A_hook` and variants
    * **[x] Option 4B: + Vegetation height boost** — `4B_vegboost` and variants

---

## Naming Convention

| Folder prefix | Git branch | Architecture | Notes |
|:---|:---|:---|:---|
| `2A_*` | `exp-2A-attention-gate` | `attention_fusion` | Branch 2, Option A |
| `2A_*_dynamic` | `exp-2A-dynamic-loss` | `attention_fusion` | Branch 2A + curriculum loss |
| `3A_*` | `exp-3-ynet-decoupled` | `ynet_attention_fusion` | Branch 3, no gradhook |
| `4A_*` | `exp-4-ynet-gradhook` | `ynet_attention_fusion` | Branch 4, Option A |
| `4B_*` | `exp-4-ynet-gradhook` | `ynet_attention_fusion` | Branch 4, + veg height boost |

> **Note:** Folders named `5A_vegboost*` were created before this convention was established.
> They belong to `exp-4-ynet-gradhook` (branch 4 + veg boost = Option 4B).
> No folder rename needed — the table above is the canonical reference.

---

## Run Log

| Run Folder | Git Branch | Pixel | Patch | Best Ep. | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Proxy(C=3.9) | Platform | Notes |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `2A_alpha_ts1_ts2_nologits` | `exp-2A-attention-gate` | alpha_earth | ts1,ts2 | 46 | 0.186 | 0.767 | 0.499 | 3.12 | 3.68 | 0.298 | **0.3660** ★ | Current best. No sigmoid bug. |
| `2A_alpha_ts1_ts2_dynamic` | `exp-2A-dynamic-loss` | alpha_earth | ts1,ts2 | — | — | — | — | — | — | — | — | Dynamic loss variant |
| `4A_hook` | `exp-4-ynet-gradhook` | alpha_earth | ts1,ts2 | 59 | 0.190 | 0.787 | 0.568 | 3.46 | 3.80 | 0.284 | worse | Better IoU_W, worse RMSE_B |
| `4A_hook_aug` | `exp-4-ynet-gradhook` | alpha_earth | ts1,ts2 | — | — | — | — | — | — | — | — | + augmentation |
| `4A_hook_dyn` | `exp-4-ynet-gradhook` | alpha_earth | ts1,ts2 | — | — | — | — | — | — | — | — | + dynamic loss |
| `4B_vegboost` *(folder: 5A_vegboost)* | `exp-4-ynet-gradhook` | alpha_earth | ts1,ts2 | 42 | 0.192 | 0.779 | 0.562 | 3.41 | 3.81 | 0.285 | TBD | + veg height boost |
| `4B_vegboost_s1` *(folder: 5A_vegboost_s1)* | `exp-4-ynet-gradhook` | alpha_earth | ts1,ts2 | 45 | 0.187 | 0.785 | 0.532 | 3.41 | 3.72 | 0.285 | TBD | seed 2, best RMSE_V |
| `4B_vegboost_s2` *(folder: 5A_vegboost_s2)* | `exp-4-ynet-gradhook` | alpha_earth | ts1,ts2 | 51 | 0.193 | 0.787 | 0.570 | 3.53 | 3.75 | 0.283 | TBD | seed 3, best IoU_W |

### How to add a new row
Copy the template below, fill in the fields, append to the table above:
```
| `<folder>` | `<git-branch>` | <pixel> | <patch> | <ep> | <iou_b> | <iou_v> | <iou_w> | <rmse_b> | <rmse_v> | <proxy> | <platform or TBD> | <notes> |
```

---

## Platform Scoring Formula (reverse-engineered, C=3.9, R²=0.996)

```
score = 0.25×IoU_B + 0.15×IoU_V + 0.15×IoU_W
      + 0.25×max(0, 1 - RMSE_B / 3.9)
      + 0.20×max(0, 1 - RMSE_V / 3.9)
```

RMSE ≥ 3.9m → quality = 0 (zero contribution). Our RMSE_V=3.74m is nearly at zero.

## Pending Improvements (priority order)

| # | Idea | Status | Expected Gain |
|:---:|:---|:---|:---|
| 1 | Veg height boost in losses.py | ✅ Done (4B runs) | Disappointing — RMSE_V barely moved |
| 2 | Fix proxy C=4.0 in train.py | ✅ Done | Better checkpoint selection |
| 3 | **TTA on 2A_nologits** (8-fold) | 🔄 In progress | IoU +0.02, RMSE −0.2m |
| 4 | Replace Tversky+SSIM+GDL with pure MSE (ch 0-2) | Pending | Calibrate abundance fractions properly |
| 5 | Threshold calibration script | ✅ Done (`calibrate_threshold.py`) | Diagnostic only |
| 6 | Ensemble 3 seeds | Pending | Marginal |
