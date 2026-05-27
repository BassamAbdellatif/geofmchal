# Emb2Heights — Full Project Manual

> **Scope:** Science (task, scoring, architecture decisions) + Implementation (pipeline, commands, branch strategy).
> `readme.md` is the upstream baseline and is now superseded by this file.
> `experiments.md` tracks individual run results and is the companion to this document.

---

## Table of Contents

1. [Competition Overview](#1-competition-overview)
2. [Task Description](#2-task-description)
3. [Platform Scoring Formula](#3-platform-scoring-formula)
4. [Architecture Evolution](#4-architecture-evolution)
5. [Current Best Architecture](#5-current-best-architecture)
6. [Loss Function Design](#6-loss-function-design)
7. [Training Pipeline](#7-training-pipeline)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Submit & Upload Pipeline](#9-submit--upload-pipeline)
10. [Test-Time Augmentation (TTA)](#10-test-time-augmentation-tta)
11. [Cluster Setup](#11-cluster-setup)
12. [Branch & Naming Strategy](#12-branch--naming-strategy)
13. [Lessons Learned](#13-lessons-learned)

---

## 1. Competition Overview

| Field | Value |
|---|---|
| Challenge | ESA Φ-lab "Reaching new heights with GeoFM embeddings" |
| URL | https://platform-challenges.philab.esa.int/geoai/leaderboard |
| Team | Challenger |
| Submission limit | Once every 12 hours (max 2/day) |
| Deadline | 30 June 2026 |

---

## 2. Task Description

The model must predict **4 output channels** per 10m pixel from GeoFM embeddings:

| Ch | Target | Range | Meaning |
|---|---|---|---|
| 0 | Building abundance | 0–1 | Fraction of 1m pixels within the 10m cell that are buildings |
| 1 | Vegetation abundance | 0–1 | Same for vegetation |
| 2 | Water abundance | 0–1 | Same for water |
| 3 | Height (nDSM) | 0–∞ m | Normalised surface model height in metres |

**Critical:** channels 0–2 are **fractional abundances**, not binary masks. They are computed from 1m-resolution binary rasters aggregated to 10m. The platform thresholds them at 0.5 to compute IoU, but the regression target is continuous.

**Height normalisation:** `clip(nDSM / 30, 0, 1.5)`. `HEIGHT_NORM_CONSTANT = 30.0`.

### Input Embeddings

| Name | Type | Spatial res | Channels | Used as |
|---|---|---|---|---|
| `alpha_earth` | Pixel embedding | 256×256 | 64 | Spatial backbone input |
| `tessera` | Pixel embedding | 256×256 | 128 | Alternative to alpha_earth |
| `terramind_s1` | Patch embedding | 16×16 | 768 | Bottleneck injection |
| `terramind_s2` | Patch embedding | 16×16 | 768 | Bottleneck injection |
| `thor_s1/s2` | Patch embedding | 16×16 | varies | Alternatives to terramind |

---

## 3. Platform Scoring Formula

Reverse-engineered from all 48 leaderboard rows (R²=0.996, C=3.9):

```
score = 0.25 × IoU_B
      + 0.15 × IoU_V
      + 0.15 × IoU_W
      + 0.25 × max(0, 1 − RMSE_B / 3.9)
      + 0.20 × max(0, 1 − RMSE_V / 3.9)
```

- **IoU** is hard binary at threshold 0.5 on the abundance predictions.
- **RMSE** above 3.9 m scores exactly zero for that term.
- **Priority order** for improvement: RMSE_V > IoU_B > IoU_W > RMSE_B > IoU_V

### Current Challenger metrics (best submission)

| Metric | Us | Top team | Gap | Score gap |
|---|---|---|---|---|
| IoU_B | 0.3394 | 0.5269 | −0.188 | −0.047 |
| IoU_V | 0.7649 | 0.8221 | −0.057 | −0.009 |
| IoU_W | 0.3695 | 0.5194 | −0.150 | −0.022 |
| RMSE_B | 2.27 m | 1.76 m | −0.51 m | −0.033 |
| RMSE_V | 3.74 m | 3.06 m | −0.68 m | −0.034 |

> Note: internal validation metrics are systematically lower than platform metrics
> (platform test set has more building-dense tiles). Do not compare them directly.

---

## 4. Architecture Evolution

```
Branch 1: Early Fusion          → SKIPPED (semantic dilution risk)

Branch 2: Two-Stream Bottleneck Injection
  └─ Option 2A: Attention-Gated Skip Connections   (git: exp-2A-attention-gate)
       Model: attention_fusion
       Best platform score: 0.3660 ★
       Strength: lowest RMSE_B (2.27 m)

Branch 3: Y-Net Decoupled       (git: exp-3-ynet-decoupled)
       Model: ynet_attention_fusion (no hook)
       Separate class/height decoders, no gradient isolation

Branch 4: Y-Net + GradScale Hook (git: exp-4-ynet-gradhook)  ← CURRENT
       Model: ynet_attention_fusion
       GradScale α=0.1: height gradients scaled to 10% entering encoder
       Strength: best IoU_W (~0.57 internal)
  └─ Option 4A: Hook baseline
  └─ Option 4B: + Vegetation height boost in loss
```

---

## 5. Current Best Architecture

**Model:** `YNetAttentionFusedDecoder` (`ynet_attention_fusion`) in `core/model.py`

```
alpha_earth (256×256×64)
       │
  [Encoder — shared U-Net backbone]
       │
  [Bottleneck]  ←── terramind_s1 + terramind_s2 fused (16×16×1536)
       │
  ┌────┴─────┐
  │          │
[decoder_class]   [decoder_height]
  │          │         │
ch 0,1,2    GradScale hook (α=0.1)
(sigmoid)   ← height gradients scaled to 10%
             ch 3 (linear)
```

**GradScale hook:** custom autograd function. Forward = identity. Backward = multiply gradient by α=0.1. This prevents the height regression from disrupting the classification encoder — the two tasks share the encoder but their gradient contributions are decoupled.

---

## 6. Loss Function Design

Defined in `core/losses.py` — `ImprovedCompositeLoss`.

```
total_loss = 1.0 × MAE (fg/bg split, ch weights [3,1,3,1])
           + 0.5 × SSIM (abundance channels 0–2)
           + 0.5 × GDL  (gradient difference loss, channels 0–2)
           + 2.0 × Tversky (α=0.3, β=0.7, channels 0–2)
           + 1.0 × height_boost_buildings  (GT ch0 > 0.1)
           + 1.0 × height_boost_vegetation (GT ch1 > 0.1)
```

**Height boost:** masked MAE on height channel, applied only where the target has building or vegetation presence. Without this, height supervision is diluted by flat background areas where height = 0 trivially.

**Known limitation:** Tversky + SSIM + GDL are binary segmentation losses; our targets are continuous fractions. Pure MSE for channels 0–2 may be better calibrated (pending experiment).

**Proxy checkpoint criterion** (in `train.py`):
```python
PROXY_C = 4.0   # matches platform C=3.9
quality_B = max(0, 1 - rmse_b / PROXY_C)
quality_V = max(0, 1 - rmse_v / PROXY_C)
proxy = 0.25*iou_b + 0.15*iou_v + 0.15*iou_w + 0.25*quality_B + 0.20*quality_V
```

---

## 7. Training Pipeline

### Full command

```bash
cd /mnt/head/users/bassam/src/geofmchal
./run_env.sh train.py \
  --model-type ynet_attention_fusion \
  --pixel-inputs alpha_earth \
  --patch-inputs terramind_s1,terramind_s2 \
  --experiment-name <NAME> \
  --batch-size 32 \
  --epochs 60
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--model-type` | `auto` | Architecture: `attention_fusion`, `ynet_attention_fusion`, `lightunet`, `decoder_residual` |
| `--pixel-inputs` | `tessera` | Comma-separated pixel embedding names or paths |
| `--patch-inputs` | `terramind_s1` | Comma-separated patch embedding names or paths |
| `--experiment-name` | required | Folder name under `SHARED_RUNS_DIR` |
| `--batch-size` | 32 | Batch size |
| `--epochs` | 30 | Number of epochs |
| `--augment` | False | Enable spatial augmentation |
| `--dynamic-loss` | False | Ramp height boost 1×→5× over training |

### Outputs (in `runs/<NAME>/`)

```
training_params.txt     ← hyperparams + per-epoch metrics log
model_best_e1.pth       ← checkpoint at best proxy score
model_last.pth          ← final epoch checkpoint
loss_curve.png
visualizations/
```

---

## 8. Inference Pipeline

```bash
./run_env.sh predict.py --experiment-name <NAME>
# with TTA:
./run_env.sh predict.py --experiment-name <NAME> --tta
```

`predict.py` is self-contained: it reads `training_params.txt` to determine model type and inputs automatically. CLI flags override when needed.

| Argument | Default | Description |
|---|---|---|
| `--experiment-name` | required | Must match a folder in `SHARED_RUNS_DIR` |
| `--pixel-inputs` | from params.txt | Override pixel embedding source |
| `--patch-inputs` | from params.txt | Override patch embedding source |
| `--tta` | off | Enable 8-fold Test-Time Augmentation |

**Outputs:**
- Without `--tta` → `runs/<NAME>/predictions/*.npy`
- With `--tta` → `runs/<NAME>/predictions_tta/*.npy`

Each `.npy` is `float32`, shape `(4, 256, 256)`:
channels = `[building_fraction, veg_fraction, water_fraction, height_metres]`

---

## 9. Submit & Upload Pipeline

### Overview

```
predict.py ──→ predictions/        ──→ package.py ──→ submission_<NAME>.zip     ──→ submit.py ──→ ESA platform
predict.py --tta ──→ predictions_tta/ ──→ package.py --tta ──→ submission_<NAME>_tta.zip ──→ submit.py --tta
```

### Step-by-step flow

```
┌──────────────────────────────────────────────────────────────────┐
│  predict.py                                                      │
│  reads: model_best_e1.pth + test embeddings                      │
│  writes: predictions/       (base)                               │
│          predictions_tta/   (with --tta, 8× slower)              │
└──────────────────┬───────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────────┐
│  package.py                                                      │
│  reads: predictions/  or  predictions_tta/ (with --tta)          │
│  validates: shape == (4, 256, 256)                               │
│  writes: submission_<NAME>.zip  or  submission_<NAME>_tta.zip    │
│  internal zip structure: predictions/<core_id>.npy  (required    │
│  by the platform)                                                │
└──────────────────┬───────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────────┐
│  uploader/submit.py                                              │
│  reads: submission_<NAME>.zip  (or _tta.zip with --tta)          │
│         OR any explicit path via --zip-file                      │
│  auth:  uploader/cookies.json  (paste from EditThisCookie)       │
│         sameSite values normalised automatically                  │
│  uses:  Playwright headless Chromium                             │
│  steps: navigate → check session → click NEW → fill form         │
│         → attach zip → submit → await server response            │
└──────────────────────────────────────────────────────────────────┘
```

### Commands reference

```bash
# Base predictions (no TTA)
./run_env.sh predict.py  --experiment-name <NAME>
./run_env.sh package.py  --experiment-name <NAME>
./run_env.sh uploader/submit.py --experiment-name <NAME>

# TTA predictions
./run_env.sh predict.py  --experiment-name <NAME> --tta
./run_env.sh package.py  --experiment-name <NAME> --tta
./run_env.sh uploader/submit.py --experiment-name <NAME> --tta

# Submit an explicitly named zip (e.g. from a renamed experiment)
./run_env.sh uploader/submit.py --experiment-name <NAME> \
  --zip-file /mnt/head/users/bassam/data/geofmdata/runs/<NAME>/any_file.zip
```

### Cookie refresh (every ~24h)

1. Open ESA platform in browser, sign in
2. Open EditThisCookie extension → Export (copies JSON to clipboard)
3. Paste into `uploader/cookies.json` (overwrite entirely)
4. `sameSite` values are fixed automatically — no manual editing needed

---

## 10. Test-Time Augmentation (TTA)

### Concept

The model is not perfectly rotation/flip invariant despite training. TTA exploits this by running inference 8 times on transformed versions of each input and averaging the results.

### The 8 transforms (D4 symmetry group)

| # | Transform applied to input | Inverse applied to output |
|---|---|---|
| 1 | Identity | Identity |
| 2 | Rotate 90° | Rotate −90° |
| 3 | Rotate 180° | Rotate −180° |
| 4 | Rotate 270° | Rotate −270° |
| 5 | Flip horizontal | Flip horizontal |
| 6 | Rotate 90° + Flip H | Unflip + Rotate −90° |
| 7 | Rotate 180° + Flip H | Unflip + Rotate −180° |
| 8 | Rotate 270° + Flip H | Unflip + Rotate −270° |

### Why average in logit space

Sigmoid is nonlinear. Averaging raw model outputs (logits) before applying sigmoid gives more conservative, better-calibrated probabilities than averaging after sigmoid — especially when one augmentation is very confident and another is uncertain.

```
avg_logit = mean(out₁, out₂, ..., out₈)
final_pred_abundance = sigmoid(avg_logit[:3])
final_pred_height    = avg_logit[3]          # linear, no sigmoid
```

### Cost

8× slower inference (~25 min vs ~3 min for 946 test samples on a 48GB GPU). No retraining.

### Expected gains

| Metric | Typical improvement |
|---|---|
| IoU_B / IoU_W | +0.01 – 0.03 |
| RMSE_B / RMSE_V | −0.1 – 0.3 m |

---

## 11. Cluster Setup

| Node | GPU | Notes |
|---|---|---|
| head | 48 GB | Login node, runs all 4 pipeline steps |
| n1, n2, n3 | 48 GB each | Training nodes, SSH from head |

**Conda environment:**
```bash
/scratch/head/geofm_env
```

**Launch script:** `./run_env.sh <script.py> [args]` — activates the env and runs the script.

**Shared data root:** `/mnt/head/users/bassam/data/geofmdata/`

**Shared runs output:** `/mnt/head/users/bassam/data/geofmdata/runs/`

**Config file:** `config.py` — auto-routes data paths per node (n1/n2/n3/head).

### Running on multiple nodes in parallel

```bash
# Head
./run_env.sh train.py --experiment-name 4B_vegboost_h  [args]

# n1
ssh n1 "cd /mnt/head/users/bassam/src/geofmchal && ./run_env.sh train.py --experiment-name 4B_vegboost_n1 [args]"

# n2
ssh n2 "cd /mnt/head/users/bassam/src/geofmchal && ./run_env.sh train.py --experiment-name 4B_vegboost_n2 [args]"
```

---

## 12. Branch & Naming Strategy

### Git branches

| Branch | Purpose |
|---|---|
| `main` | Infrastructure only: predict.py, package.py, uploader/submit.py, docs |
| `exp-2A-attention-gate` | Branch 2, Option A — best platform score so far |
| `exp-2A-dynamic-loss` | Branch 2A + curriculum loss variant |
| `exp-3-ynet-decoupled` | Branch 3 — Y-Net without hook |
| `exp-4-ynet-gradhook` | Branch 4 — Y-Net + GradScale hook (active) |

**Rule:** architecture-specific files (`core/model.py`, `core/losses.py`, `train.py`) live on experiment branches. Utilities (`predict.py`, `package.py`, `uploader/submit.py`, docs) live on `main` and are cherry-picked to experiment branches.

### Experiment folder naming

```
<branch>_<variant>_<seed_or_note>

Examples:
  2A_alpha_ts1_ts2_nologits    → exp-2A-attention-gate, alpha_earth + ts1+ts2, no logit bug
  4A_hook                      → exp-4-ynet-gradhook, baseline hook
  4B_vegboost                  → exp-4-ynet-gradhook, + vegetation height boost
  4B_vegboost_s1               → same, seed 2
```

See `experiments.md` for the full run log with metrics and platform scores.

---

## 13. Lessons Learned

| # | Lesson |
|---|---|
| 1 | **Soft IoU ≠ Hard IoU.** Internal soft IoU 0.19 ≈ platform hard IoU 0.34. Never compare them directly. |
| 2 | **Proxy C matters.** Using C=HEIGHT_NORM_CONSTANT=30 for checkpoint selection made RMSE differences invisible. Fixed to C=4.0. |
| 3 | **Platform test set differs from val set.** IoU_B on platform is ~1.8× higher than internal; IoU_W is ~0.75× lower. |
| 4 | **4A_hook scored worse than 2A on platform** despite better internal IoU_W, because its RMSE_B (3.46m) was worse than 2A (3.12m). RMSE_B is worth 0.25 weight. |
| 5 | **Veg height boost (4B) did not dramatically improve RMSE_V** — still ~3.73m after 60 epochs. RMSE_V improvement requires more than just a loss mask. |
| 6 | **DataLoader workers need numpy seeding** — without `worker_init_fn`, all 8 workers generate the same crop sequence, reducing effective data diversity. |
| 7 | **Abundance targets need regression losses** — Tversky/SSIM/GDL are binary segmentation losses; targets are continuous fractions. Pure MSE likely better. |
| 8 | **Playwright screenshots can hang** on headless servers due to font loading. All screenshots must be wrapped in try/except. |
| 9 | **EditThisCookie exports `sameSite: "lax"` and `"unspecified"`** — Playwright requires capitalised `"Lax"`. Now normalised automatically in uploader/submit.py. |
