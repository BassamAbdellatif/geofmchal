# GeoFM Challenge — Project Context

## Competition
- **Challenge**: ESA Φ-lab "Reaching new heights with GeoFM embeddings"
- **URL**: https://platform-challenges.philab.esa.int/geoai/leaderboard
- **Team**: Challenger (last submission: `2A_vegboost`, score 0.3721)
- **Deadline**: 30 June 2026 (~30 days remaining as of 2026-05-30)
- **Submission limit**: once every 12 hours

## Platform Scoring Formula (reverse-engineered, C=3.9, R²=0.996)
```
score = 0.25×IoU_B + 0.15×IoU_V + 0.15×IoU_W
      + 0.25×max(0, 1 - RMSE_B / 3.9)
      + 0.20×max(0, 1 - RMSE_V / 3.9)
```
- RMSE above 3.9m scores zero.
- IoU is hard binary at threshold 0.5 on abundance predictions.
- The proxy in train.py uses C=4.0 (close enough; do not change without rerunning all proxy comparisons).

## Our Current Platform Metrics (best submission: `2A_vegboost`)
| Metric   | Us     | Top team | Gap (score) | Priority |
|----------|--------|----------|-------------|----------|
| IoU_B    | 0.3394 | 0.5269   | −0.047      | **1st**  |
| IoU_V    | 0.7649 | 0.8221   | −0.009      | low      |
| IoU_W    | 0.3695 | 0.5194   | −0.022      | 2nd      |
| RMSE_B   | 2.27m  | 1.76m    | −0.033      | 3rd      |
| RMSE_V   | 3.74m  | 3.06m    | −0.034      | 4th      |

> Priority revised after the 6A post-mortem: IoU_B is the largest score gap and represents a *representation* problem (we cannot extract clean building signal), not a calibration problem. The 7A plan addresses this directly.

## Task Description (critical)
- Targets are **abundance fractions** (0–1), computed from 1m-resolution binary masks
  aggregated to 10m. NOT binary classification. Channel 0=buildings, 1=vegetation,
  2=water (all fractions), 3=relative height (nDSM, continuous).
- Platform thresholds abundance at 0.5 for IoU computation.
- HEIGHT_NORM_CONSTANT = 30.0 (in dataset.py). Height normalized as clip(nDSM/30, 0, 1.5).

## Architecture — Status

### Current platform-best: `2A_vegboost` (branch: `exp-2A-attention-gate`)
- `YNetAttentionFusedDecoder` (`ynet_attention_fusion`) in `core/model.py`
- AlphaEarth pixel input (64ch) + TerraMind S1+S2 patch tokens fused at bottleneck via spatial broadcast
- Two-decoder Y-Net split (classification + height)
- Loss: ImprovedCompositeLoss + veg_height_boost
- Platform score: **0.3721**

### Concluded: 6A family (`exp-6-tessera-xattn`)
- TESSERA pixel concatenation + cross-attention bottleneck fusion
- **Not competitive.** Mixing TESSERA into the pixel stem destroyed IoU_B (0.168 → 0.017)
- Cross-attention with shared decoder produced proxy 0.24 vs 2A's 0.37
- See `results.md` § 6A and prompts/exp-6-tessera-xattn.md for full analysis

### Planned: 7A — decoupled dual-encoder / dual-decoder (`exp-7-clean-slate`)
- Two modality-specialised encoders (AlphaEarth → urban-spatial, TESSERA → temporal-natural)
- Two task-specialised decoders (fractions, height)
- Patch tokens routed by sensor type: S1 → height decoder, S2 → fraction decoder
- Injected at decoder skip connections (16×16, 32×32), not at encoder bottleneck
- Geographic CV, stratified sampling, channel standardisation, auxiliary binary B head
- GradNorm task balancing; D4 TTA and threshold calibration at inference
- See `prompts/exp-7-clean-slate.md` for full spec

## Key Files
| File | Role |
|------|------|
| `train.py` | Training loop. Proxy uses C=4.0. worker_init_fn seeds numpy per worker. |
| `core/losses.py` | `ImprovedCompositeLoss`: MAE + SSIM + GDL + Tversky + building_height_boost + veg_height_boost. |
| `core/model.py` | Current models: `YNetAttentionFusedDecoder`, `YNetTesseraXAttn`. 7A will add `DualEncDualDecFusion`. |
| `core/dataset.py` | `Emb2HeightsDataset`. 7A will extend with multi-modal dict output, stratified sampling, geographic CV, domain-shift augmentation. |
| `predict.py` | Auto-reads training_params.txt. 7A will add `--tta`, `--blend-binary`, `--threshold-config`. |
| `package.py` | Packages predictions into zip for upload. |
| `uploader/submit.py` | Playwright-based upload to ESA platform via uploader/cookies.json. |
| `config.py` | Auto-routes data paths per cluster node (n1/n2/n3/head). |
| `prompts/` | Coding-agent specs per experiment branch. |

## Cluster Setup
- 4 nodes with 48GB GPUs: n1, n2, n3, head
- Conda env: `/scratch/head/geofm_env`  Launch via: `./run_env.sh train.py [args]`
- Shared run output: `/mnt/head/users/bassam/data/geofmdata/runs/`
- Shared data root: `/mnt/head/users/bassam/data/geofmdata/`

## Working with this repo
- **Task specs for the coding agent live in `prompts/`.** Each new experiment branch starts by committing its spec there as the first commit.
- **Active project context: `CLAUDE.md`, `results.md`, `science.md` at root.** Read these first when starting a session.
- **After each experiment, append findings to `results.md`. Do not rewrite previous entries.** The history of what was tried (and what didn't work) is the most valuable part of the document.
- **Do not auto-submit to the ESA platform.** 12-hour submission limit makes each one expensive; the human decides.
- **One model class per architecture, additive only.** Never delete or modify existing model classes when introducing a new one; the 6A vs 2A comparison only worked because both classes were preserved.

## Full Pipeline (per experiment)
```bash
# 1. Train
cd /mnt/head/users/bassam/src/geofmchal
./run_env.sh train.py --model-type <NAME> \
  --pixel-inputs <...> --patch-inputs <...> \
  --experiment-name <EXP> --batch-size 32 --epochs 60

# 2. Predict (reads config from training_params.txt automatically)
./run_env.sh predict.py --experiment-name <EXP>

# 3. Package
./run_env.sh package.py --experiment-name <EXP>

# 4. Submit (needs uploader/cookies.json with valid ESA session)
./run_env.sh uploader/submit.py --experiment-name <EXP>
```

## Lessons Learned (running list — see results.md for per-experiment detail)
- **Hard IoU ≠ soft IoU.** Internal soft metrics underestimate platform hard IoU systematically. Use proxy (C=4.0), not raw IoU.
- **Tversky/Dice components are essential** for fraction prediction at hard-threshold-0.5 evaluation. Pure MSE destroys IoU.
- **Dynamic loss weights backfire.** 5A's curriculum (1× → 5× height weight) regressed RMSE_B and IoU_W. Use static weights or learned (GradNorm).
- **Augmentation hurt slightly** in the 2A/4A regime (crop-diversity bug). 7A's domain-shift augmentation is different — calibrated channel jitter on embeddings, not pixel-space colour transforms.
- **Shared encoders force destructive modality competition** (6A finding). Concatenating TESSERA with AlphaEarth at input destroyed building detection.
- **Shared decoders force destructive task competition** (6A finding). The 2A two-decoder Y-Net split was doing more work than credited; collapsing to a shared decoder cost ~0.13 in proxy.
- **GradScale α=0.1** on the height decoder helps but is not sufficient on its own to prevent task interference.
- **Random validation splits leak spatial context.** 7A switches to geographic CV (KMeans over lat/lon) because the test set is in different regions and years.