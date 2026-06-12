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

## Our Current Platform Metrics (best submission: `12_dech_f1` = 0.4277, rank 39 — 2026-06-12)
| Metric   | Us (`12_dech_f1`) | prev (`11_fresh_f1`) | #1 leader | room-to-max (×weight) |
|----------|-------------------|----------------------|-----------|-----------------------|
| final    | **0.4277**        | 0.4125               | 0.5448    | —                     |
| IoU_B    | 0.4272            | 0.4297               | 0.5316    | **0.143** (stuck)     |
| IoU_V    | 0.8062            | 0.8062               | 0.8924    | 0.029 (near-max ✅)    |
| IoU_W    | 0.4844            | 0.4797               | 0.6141    | 0.078                 |
| RMSE_B   | 2.078m            | 2.226m               | 1.83m     | 0.120 (near-top ✅)    |
| RMSE_V   | **3.740m**        | 3.807m               | 2.78m     | **0.187 🔴 still the giant** |

> **State (2026-06-12):** best = `12_dech_f1` (`fresh_extract` + **decoupled height loss**: separate,
> independently-weighted Huber for building vs veg pixels, wb=1/wv=3; sm4tv, no-TerraMind, cv-fold 1) =
> **0.4277, rank 39** (up from 43/47). The height campaign delivered: **both height metrics dropped on the
> platform** (RMSE_B −0.148, RMSE_V −0.067), *more* than internal val predicted — decoupling fixed the
> veg↔building trade (raw `veg_height_boost` washed out, f37) AND helped the train→test gap. **Building
> height now protected → push veg weight higher** (decoupled wv sweep in flight). Remaining levers:
> **RMSE_V (0.187 room, tractable)** and **IoU_B (0.143, stuck across every fusion/objective experiment)**.
> Parked: TerraMind (f36) & alpha↔tessera cross-modal (f38) both net-null. No calibration/TTA.
> **Incremental tuning is exhausted** (Phase 8 null / Phase 9 small / Phase 10 gradient-bridge null, f33) and the **decoder is already full-resolution** (f34) — so the "raise-resolution / light-U-Net" lever is not available. Remaining step-change candidates (capacity/ensemble/better-height) are uncertain; **top-3 is out of reach in the remaining days.** Full detail: `docs/results.md` f29–f34, `docs/roadmap_final21d.md` status update.

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

### 7A — decoupled dual-encoder / dual-decoder (`exp-7-clean-slate`) — current 7A best: `7A_v1_base` (proxy ~0.40)
- Two modality-specialised encoders (AlphaEarth → urban-spatial, TESSERA → temporal-natural)
- Two task-specialised decoders (fractions, height)
- Patch tokens routed by sensor type: S1 → height decoder, S2 → fraction decoder
- Injected at decoder skip connections (16×16, 32×32), not at encoder bottleneck
- Geographic CV, stratified sampling, channel standardisation, auxiliary binary B head
- GradNorm task balancing; D4 TTA and threshold calibration at inference
- See `prompts/exp-7-clean-slate.md` for full spec

**Outcome (Phase 5B–5D, 2026-06-04):** recommended recipe = **`7A_v1_base`** — v1 patch stem, static weights `[0.65,0.64,1.70]`, no GradNorm, no vegboost, batch 32, bridge on, binary head on; internal proxy **~0.40** (IoU_B 0.20 / IoU_V 0.81 / IoU_W 0.68 / RMSE_V 4.0). **Rejected:** GradNorm≈static (f18), vegboost (f17), THOR (f22), enhanced v2/v2b stems (f23). **Not used going forward:** inference threshold calibration / blend-binary (fit to fold-0 val → won't transfer to the shifted test set; depend on the raw 0.5 prediction). RMSE_V 4.0m cliff (f21) is the binding constraint. Full detail: `docs/phase5d_thor.md`. Next line: `exp-8-decouple`.

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
- **THOR adds nothing to 7A** (Phase 5D, f22): single-stream ties the base, full-stream collapses water. Dropped.
- **Rare-class (water) ignition is fragile** (f23–24): it needs the simple **v1** patch stem + **batch ≥ 32** + ≤3 patch streams. The deeper v2/v2b stem, batch 24, or full-THOR each independently zero IoU_W for all 60 epochs (dead at every threshold down to 0.05 — not a calibration artifact). Building survives via its dedicated binary head; water has none → root cause is weak rare-class supervision.
- **Augmentation RNG is unseeded** (f25): `np.random.default_rng()` in the dataset ignores `--seed`; seed it before trusting fine ablations.
- **Don't tune inference thresholds / blend-binary to fold-0 val** — fit to that fold's distribution, won't transfer to the different-region/year test set. Depend on the raw prediction at 0.5.