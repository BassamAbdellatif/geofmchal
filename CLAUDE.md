# GeoFM Challenge — Project Context

## Competition
- **Challenge**: ESA Φ-lab "Reaching new heights with GeoFM embeddings"
- **URL**: https://platform-challenges.philab.esa.int/geoai/leaderboard
- **Team**: Challenger (position 35 of 48, score 0.3660 as of 2026-05-26)
- **Deadline**: 30 June 2026 (~35 days remaining)
- **Submission limit**: once every 12 hours

## Platform Scoring Formula (reverse-engineered, C=3.9, R²=0.996)
```
score = 0.25×IoU_B + 0.15×IoU_V + 0.15×IoU_W
      + 0.25×max(0, 1 - RMSE_B / 3.9)
      + 0.20×max(0, 1 - RMSE_V / 3.9)
```
- RMSE above 3.9m scores zero. We are at RMSE_V=3.74m — almost zero!
- IoU is hard binary at threshold 0.5 on abundance predictions
- The proxy in train.py must use C=4.0 (not the old C=30)

## Our Current Platform Metrics (Challenger, best submission)
| Metric   | Us     | Top team | Gap (score) | Priority |
|----------|--------|----------|-------------|----------|
| IoU_B    | 0.3394 | 0.5269   | −0.047      | 2nd      |
| IoU_V    | 0.7649 | 0.8221   | −0.009      | low      |
| IoU_W    | 0.3695 | 0.5194   | −0.022      | 3rd      |
| RMSE_B   | 2.27m  | 1.76m    | −0.033      | 4th      |
| RMSE_V   | 3.74m  | 3.06m    | −0.034      | **1st**  |

## Task Description (critical)
- Targets are **abundance fractions** (0–1), computed from 1m-resolution binary masks
  aggregated to 10m. NOT binary classification. Channel 0=buildings, 1=vegetation,
  2=water (all fractions), 3=relative height (nDSM, continuous).
- Platform thresholds abundance at 0.5 for IoU computation.
- HEIGHT_NORM_CONSTANT = 30.0 (in dataset.py). Height normalized as clip(nDSM/30, 0, 1.5).

## Architecture — current best branch: `exp-4-ynet-gradhook`
- **Model**: `YNetAttentionFusedDecoder` (ynet_attention_fusion) in `core/model.py`
  - Shared encoder (U-Net pixel backbone, alpha_earth 64ch input)
  - Bottleneck fusion with patch embeddings (terramind_s1 + terramind_s2, 1536ch total)
  - Two separate decoders: `decoder_class` (ch 0-2) and `decoder_height` (ch 3)
  - **GradScale hook** (α=0.1): height decoder gradients scaled to 10% before reaching encoder
    → prevents height regression from disrupting classification gradients
- **Inputs**: `--pixel-inputs alpha_earth --patch-inputs terramind_s1,terramind_s2`
- **Best run**: `4A_hook` (proxy 0.6465 internal) but 2A_nologits scored better on platform
  due to better RMSE (old proxy used C=30, was blind to RMSE)

## Key Files
| File | Role |
|------|------|
| `train.py` | Training loop. Proxy checkpoint uses C=4.0 (updated). worker_init_fn seeds numpy per worker. |
| `core/losses.py` | `ImprovedCompositeLoss`: MAE + SSIM + GDL + Tversky + building_height_boost. **Missing: veg_height_boost** |
| `core/model.py` | YNetAttentionFusedDecoder with GradScale hook |
| `core/dataset.py` | `Emb2HeightsDataset` with aligned augmentation (pixel+patch+target). augment_triplet() is synchronized. |
| `predict.py` | Auto-reads training_params.txt. Use `--pixel-inputs`/`--patch-inputs` to override. |
| `package.py` | Packages predictions into zip for upload |
| `uploader/submit.py` | Playwright-based direct upload to ESA platform using uploader/cookies.json |
| `config.py` | Auto-routes data paths per cluster node (n1/n2/n3/head) |

## Cluster Setup
- 4 nodes with 48GB GPUs: n1, n2, n3, head
- Conda env: `/scratch/head/geofm_env`  Launch via: `./run_env.sh train.py [args]`
- Shared run output: `/mnt/head/users/bassam/data/geofmdata/runs/`
- Shared data root: `/mnt/head/users/bassam/data/geofmdata/`

## Full Pipeline (per experiment)
```bash
# 1. Train
cd /mnt/head/users/bassam/src/geofmchal
./run_env.sh train.py --model-type ynet_attention_fusion \
  --pixel-inputs alpha_earth --patch-inputs terramind_s1,terramind_s2 \
  --experiment-name <NAME> --batch-size 32 --epochs 60

# 2. Predict (reads config from training_params.txt automatically)
./run_env.sh predict.py --experiment-name <NAME>

# 3. Package
./run_env.sh package.py --experiment-name <NAME>

# 4. Submit (needs uploader/cookies.json with valid ESA session)
./run_env.sh uploader/submit.py --experiment-name <NAME>
```

## Pending Improvements (priority order)

### HIGH — implement now
1. **Add vegetation height boost to losses.py** (3-line change)
   - Currently: `loss_height_boost` only masks on building pixels (channel 0 > 0.1)
   - Missing: symmetric `loss_veg_height_boost` masking on vegetation pixels (channel 1 > 0.1)
   - Expected gain: RMSE_V 3.74m → ~3.0m → **+0.038 score**
   - Add with weight 1.0 to total_loss. Retrain on all 4 nodes.

2. **Fix proxy C in train.py** (already done in current code)
   - `rmse_b_quality = max(0.0, 1.0 - rmse_b / 4.0)` ← use 4.0 not 30.0
   - Ensures best checkpoint = best platform score

3. **Threshold calibration script** (no retraining needed)
   - Scan thresholds 0.05–0.95 per channel on validation set
   - Find threshold that maximises hard binary IoU per channel
   - Apply optimal threshold in predict.py at inference time
   - Expected: IoU_W 0.37 → 0.42+, IoU_B 0.34 → 0.38+

### MEDIUM — next wave
4. **Replace Tversky/SSIM/GDL with pure MSE for channels 0-2**
   - Tversky is a binary segmentation loss; targets are continuous abundances
   - Pure MSE better calibrates fractional outputs for hard-IoU-at-0.5 evaluation
   - Teams with few submissions and high scores (ExaltedLAB: 7 subs, rank 6) likely did this

5. **TTA (Test-Time Augmentation)**
   - 8-fold: 4 rotations × 2 flips, average predictions
   - No retraining, apply at predict.py inference stage
   - Expected: IoU +0.02-0.04, RMSE -0.2-0.4m

### LOW — if time allows
6. Ensemble of 3 seeds (full dataset, no val split)
7. Guided filter post-processing on height maps
8. Dedicated binary classification head for buildings

## Lessons Learned
- **Soft IoU ≠ Hard IoU**: internal soft IoU (0.19) corresponds to platform hard IoU (0.34)
  — don't trust internal metrics directly, use proxy score
- **Dynamic loss with C=3.9**: height matters from epoch 1, no need for curriculum ramp
- **Augmentation**: ran before numpy worker_init_fn fix (crop diversity reduced but not fatal)
- **GradScale α=0.1**: prevents height gradients from disrupting classification encoder
- **2A beat 4A on platform**: because RMSE_B 3.12 vs 3.46m, and old proxy (C=30) didn't catch this

## Experiment 6A — TESSERA Stream + Cross-Attention Fusion (`exp-6-tessera-xattn`)

- **Branch**: `exp-6-tessera-xattn` (forked from `exp-4-ynet-gradhook`)
- **Model class**: `YNetTesseraXAttn`  |  **dispatch name**: `ynet_tessera_xattn`
- **Folder prefix**: `6A_*` (`6A_smoke_xattn`, `6A_mini_xattn`, `6A_tessera_xattn`)
- **Inputs**: `--pixel-inputs alpha_earth,tessera --patch-inputs terramind_s1,terramind_s2`
- **Hypothesis**: TESSERA (128ch, S1/S2 time-series) adds phenological signal AlphaEarth (64ch annual) lacks; cross-attention extracts more from patch tokens than spatial-broadcast fusion does.
- **Architecture**: concat pixel stems (64+128=192ch → 64ch via 1×1) → shared U-Net encoder → MHA bottleneck (Q=pixel 16×16 @512d, K/V=patch 256 tokens @1536d→512d) → shared decoder → split heads (class 3ch, height 1ch with GradScale α=0.1).
- **Spec**: see `prompts/exp-6-tessera-xattn.md` for the full implementation plan, smoke/mini/full test sequence, and acceptance criteria.

> **Note on existing CLAUDE.md notes above:** the "Missing: veg_height_boost" annotation in the Key Files table is stale on this branch — `core/losses.py` does contain `veg_height_boost` (Section 0 pre-flight on 2026-05-28 confirmed it at line 208). Likewise the "best run: 4A_hook" line predates the `2A_vegboost` platform submission (0.3721) which is now the platform leader.
