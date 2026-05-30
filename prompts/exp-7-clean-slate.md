# 7A — Decoupled Dual-Encoder / Dual-Decoder with Sensor-Routed Patch Skips

**Branch:** `exp-7-clean-slate` (branches from `main`, not from 6A)
**Status:** Proposal — not yet implemented
**Supersedes:** earlier 7A draft (shared-encoder + cross-attention version)

## One-line summary
A `>-<` architecture: two modality-specialised encoders (AlphaEarth → urban-spatial path, TESSERA → temporal-natural path), two task-specialised decoders (fractions, height), with patch tokens routed by sensor type (S1 → height decoder, S2 → fraction decoder) and injected at decoder skip connections rather than forced through the encoder bottleneck. Trained on a geographic CV split with stratified sampling and an auxiliary binary head for buildings.

---

## Why this design (what the 6A post-mortem taught us)

The 6A experiments produced three findings that constrain the design space hard:

1. **Concatenating TESSERA into the pixel stem destroyed IoU_B** (0.168 → 0.017). The 192→64ch projection forces high-frequency spatial features (buildings) and low-frequency temporal features (phenology) to compete in a single bottleneck. Temporal features dominate in variance and drown out building discrimination.

2. **Cross-attention at a shared bottleneck did not recover the 2A baseline** (proxy 0.24 vs 0.37). The shared decoder is the likely culprit, not the attention mechanism. The 2A two-decoder split was doing more work than we credited.

3. **TESSERA + alpha mixed at input ran ~9× slower per epoch** with worse results — the integration strategy, not the modality, was wrong.

Conclusion: 6A isn't telling us TESSERA is useless. It's telling us that forcing modalities to compete in shared parameters causes destructive interference. Decouple them, route them to where their signal is genuinely useful, and the negative transfer disappears.

The same logic extends to patch tokens. S1 (SAR) measures vertical structure; S2 (multi-spectral optical) measures surface composition. Routing each to the decoder branch that needs its signal is principled, not aesthetic — it follows the physics of the sensors.

---

## Architecture

```
alpha_earth (64ch, 256×256)                 tessera (128ch, 256×256)
        │                                            │
   [α-stem, 3 conv blocks]                  [τ-stem, 3 conv blocks]
        │                                            │
   [α-encoder, U-Net, 5 levels]              [τ-encoder, U-Net, 5 levels]
        │                                            │
   α-bottleneck (16×16 @ 384d)              τ-bottleneck (16×16 @ 384d)
        │                                            │
        │                                            │
        │   ── S2 patch tokens ──► [cross-attn]      │
        │      (TerraMind S2,           │            │
        │       256 tok × 768d,                      │
        │       proj → 384d)            │            │
        │                               │            │
        ▼                               │            │
[Fraction Decoder, 4 up-blocks]         │            │
  skips from α-encoder                  │            │
  + S2 patch injection at decoder       │            │
    levels matching 16×16 and 32×32 ◄───┘            │
        │                                            │
   ┌────┼──────┬──────┬─────────────────┐            │
[B head] [V] [W]  [Aux Binary B head]                │
                                                     │
                                                     ▼
                                       [Height Decoder, 4 up-blocks]
                                         skips from τ-encoder
                                         + α-bottleneck side-input at first block
                                         + S1 patch injection at decoder
                                           levels matching 16×16 and 32×32
                                                     │
                                              [Height head, 1ch]
                                              GradScale α=0.2 at branch entry
                                              + masked loss
```

### Structural claims (each individually ablatable in Phase 5)

| # | Claim | Disable by |
|---|-------|------------|
| 1 | Encoder modality split helps | Concat α and τ at input, use a single encoder |
| 2 | Decoder task split helps | Single shared decoder + heads |
| 3 | Sensor-routed patches help | Send both S1 and S2 to both decoders |
| 4 | Skip injection > bottleneck injection | Inject patches at encoder bottleneck instead |
| 5 | Cross-encoder bridge for height helps | Height decoder uses τ-encoder only |
| 6 | Auxiliary binary B head helps IoU_B | Drop the binary head |

### Component details

**Stems (per modality):**
- 3 conv blocks, BN + GELU, stride-2 downsample per block → output 96ch at 32×32
- Per-modality batch normalisation calibrates statistics before any further mixing

**Encoders:**
- Standard U-Net encoder, 5 levels, channel progression 96 → 128 → 192 → 256 → 384
- Each encoder is ~half the parameter count of 2A's single encoder, so total parameters ≈ 2A
- Skips saved at each level for the matching decoder

**Patch token preparation:**
- S1 tokens: 256 tokens × 768d (TerraMind S1)
- S2 tokens: 256 tokens × 768d (TerraMind S2)
- THOR S1/S2: read but not used in baseline; Phase 5 ablation
- Per-stream MLP (768 → 384) + learned 16×16 positional encoding

**Skip injection of patches (the key novelty):**

Patches do NOT enter the encoder. They condition the decoder at levels where spatial resolution matches the patch grid or its 2× upsample. Pseudocode for one decoder block:

```python
def decoder_block_with_patches(prev, skip, patches):
    x = upsample(prev)
    x = combine_with_skip(x, skip)          # standard U-Net merge
    # Patch injection by cross-attention
    B, C, H, W = x.shape
    q = x.flatten(2).transpose(1, 2)        # [B, H*W, C]
    k = v = patches                          # [B, 256, 384]
    attended = cross_attn(q, k, v)           # [B, H*W, C]
    return x + attended.transpose(1, 2).reshape(B, C, H, W)
```

At 32×32, patch tokens are bilinearly upsampled to 32×32=1024 tokens before attention. At 16×16 they align directly. At lower resolutions (8×8, 4×4) patches don't enter; the decoder relies on its own skip path.

**Height decoder cross-encoder bridge:**
The height decoder takes skips primarily from τ-encoder, but its first block concatenates the α-bottleneck as a side input (then projects back to standard channel count). This is the *only* point where the two encoder paths meet in the forward pass. The GradScale α=0.2 hook at the bridge entry prevents height loss from dominating the α-encoder (which we want specialised for fraction prediction).

**Output assembly for metrics:**
- Channel 0 = fraction-decoder B head (or binary B head, see inference blending)
- Channel 1 = fraction-decoder V head
- Channel 2 = fraction-decoder W head
- Channel 3 = height-decoder output

---

## Data preparation

These changes are independent of architecture and are the highest-expected-value lever.

### 1. Per-channel standardisation
- Compute mean/std per channel on the training set only
- Cache to `data/norm_stats.json`
- Apply in `__getitem__` for all six embedding streams

### 2. Geographic CV split (replaces random split)
- KMeans-cluster training tiles by lat/lon, K=5 folds
- Default validation = fold 0
- Validation scores WILL drop vs 2A_vegboost's reported numbers — that is the point; the previous numbers were optimistic

### 3. Stratified tile sampling
- Per-tile coverage score = mean of `(building_frac > 0.5) + (water_frac > 0.5)`
- Bin into 4 strata: `[0%, 0–5%], (5–20%], (20–50%], (>50%]`
- WeightedRandomSampler weights `[1.0, 1.5, 2.0, 3.0]` over strata

### 4. Domain-shift augmentation
Applied to embeddings (not pixels — these are FM representations):
- Channel-wise gain `[0.9, 1.1]` and offset `[−0.1, 0.1]`, p=0.5 per modality
- Channel dropout p=0.1 per channel
- Gaussian noise σ = 0.05 × per-channel std, p=0.5
- Patch tokens get gain + dropout only

### 5. Synchronised D4 geometric augmentation
- 4 rotations × 2 flips on pixel streams and target
- Patch tokens get the corresponding spatial permutation of their 16×16 grid (reordering only; treat orientation-dependent token updates as out-of-scope for v1)

---

## Loss

### Fractions (channels 0–2, from fraction decoder)
```
L_fraction = MAE(pred, target) + λ · SoftDice(sigmoid(k · (pred − 0.5)), (target > 0.5))
```
- MAE keeps continuous fractions calibrated (for RMSE)
- Soft Dice at k=5 puts gradient pressure at the 0.5 decision boundary (for hard IoU)
- No SSIM, no GDL (these regularise toward smoothness, hurting building boundaries)

### Height (channel 3)
```
L_height = Huber(pred, target, δ=0.5) on mask
         + 0.1 · Huber(pred, target, δ=0.5) on (1 − mask)
mask = (building_frac > 0) OR (veg_frac > 0)
```
Huber gives smoother gradients on large errors than MAE.

### Auxiliary binary B head
```
L_binary = BCE(pred_bin, target_bin) + SoftDice(pred_bin, target_bin)
target_bin = (building_fraction > 0.5).float()
```

### Total + GradNorm
```
L = w_frac · L_fraction + w_height · L_height + w_binary · L_binary
```
GradNorm (Chen et al. 2018, α=1.5) learns weights to equalise per-task gradient norms through the relevant encoder.

---

## Test-time strategy

1. **TTA** — D4 8-fold, average pre-sigmoid logits
2. **Per-channel threshold calibration** — scan 0.05–0.95 on geo val set
3. **Binary head blending** — channel 0 = `max(fraction_head_B, binary_head_B)` where binary head confidence exceeds threshold
4. **3-seed ensemble** for final submission, pre-sigmoid averaging
5. *(Optional)* guided filter on height with predicted building mask as edge guidance

---

## Expected gains (revised after 6A evidence)

| Change | Expected gain | Confidence |
|--------|---------------|------------|
| Geographic CV + stratified sampling | +0.02 to +0.04 | High |
| Auxiliary binary head + threshold calibration | +0.02 to +0.04 | High |
| TTA + 3-seed ensemble | +0.01 to +0.03 | High |
| Decoupled encoders (avoid 6A pixel-fusion failure) | +0.01 to +0.03 | Medium |
| Decoupled decoders (recover 2A's two-decoder win) | +0.01 to +0.02 | Medium |
| Sensor-routed patches at decoder skips | +0.005 to +0.02 | Lower (novel here) |
| GradNorm task balancing | +0.005 to +0.01 | Medium |

If half land, expected platform score ≈ 0.45–0.48.

---

## Non-goals
- Transformer encoder on embeddings (reasoning already done in the FM)
- Diffusion refinement
- MoE across foundation models
- SSL pretraining
- Heavy HP search
- Adding THOR in baseline (Phase 5 ablation only)

---

# Coding Agent Prompt

## Task
Implement experiment family **7A** as specified above. Branch from `main`. The 6A line is a dead end for this architecture; we want a clean rebuild that reuses only the dataset basics.

## Branch and naming
- Branch: `exp-7-clean-slate`
- Model class: `DualEncDualDecFusion`
- Dispatch name: `dual_enc_dec_fusion`
- Experiments:
  - `7A_smoke` — 2 batches, sanity
  - `7A_mini` — 3 epochs full data, sanity + gradient flow
  - `7A_geocv_baseline` — full 60 epochs on geographic CV fold 0
  - Phase 5 ablations come from a follow-up prompt

## Setup
```bash
cd /mnt/head/users/bassam/src/geofmchal
git checkout main
git pull
git checkout -b exp-7-clean-slate
```

If `main` is not the right baseline, ask before proceeding.

Commit this spec immediately as `prompts/exp-7-clean-slate.md` as the first commit on the branch.

## Order of work — DO NOT parallelise

Execute phases in order. Checkpoint after each. Wait for human review before proceeding.

### Phase 1 — Data infrastructure

1. **Verify all six embeddings on disk**:
   ```bash
   for emb in alpha_earth tessera terramind_s1 terramind_s2 thor_s1 thor_s2; do
     count=$(ls /mnt/head/users/bassam/data/geofmdata/embeddings/$emb/ 2>/dev/null | wc -l)
     echo "$emb: $count tiles"
   done
   ```
   Counts should match training set size. Missing → STOP and report.

2. **Per-channel norm stats** — `scripts/compute_norm_stats.py`:
   - Iterate training split, compute per-channel mean and std per embedding type
   - Write `data/norm_stats.json`
   - Dataset reads at construction, applies in `__getitem__`

3. **Geographic CV** — `scripts/build_geo_split.py`:
   - Load tile lat/lon centroids
   - KMeans K=5
   - Write `data/geo_folds.json` (fold per tile)
   - Add `--cv-fold N` flag to `train.py`
   - Report fold sizes + centroids

4. **Stratified sampler** in `core/dataset.py`:
   - Per-tile coverage score, 4 strata, `WeightedRandomSampler` weights `[1.0, 1.5, 2.0, 3.0]`
   - `--use-stratified-sampler` flag, default True

5. **Domain-shift augmentation** — `augment_domain_shift(emb_dict)`:
   - Channel gain/offset, channel dropout, Gaussian noise
   - Applied before geometric augmentation
   - Patch tokens get gain + dropout only

6. **Dataset output schema**:
   ```python
   {
     "alpha_earth":  Tensor(64, H, W),
     "tessera":      Tensor(128, H, W),
     "terramind_s1": Tensor(256, 768),
     "terramind_s2": Tensor(256, 768),
     "target":       Tensor(4, H, W),
   }
   ```
   THOR: load but skip in baseline batch unless `--use-thor` flag set.

**Checkpoint 1 — report and wait**:
- Embedding counts per modality
- Normalisation stats summary
- Geographic fold sizes and centroids
- Sample tile from dataloader: shapes match spec
- **Stop. Wait for review before Phase 2.**

### Phase 2 — Model

Implement in `core/model.py` as a new class. Do not modify existing model classes.

7. Stems — `AlphaStem`, `TesseraStem` (3 conv blocks, BN+GELU, stride-2, output 96ch at 32×32)
8. Encoders — `UNetEncoderHalf`, 5 levels, channels `[96, 128, 192, 256, 384]`, GroupNorm
9. Patch token modules — `PatchTokenStem(in=768, out=384)` + learned 16×16 positional encoding (one per S1, one per S2)
10. Decoder cross-attention block — standard up-block + skip merge, then 4-head cross-attention to patch tokens, residual + LayerNorm
11. `FractionDecoder` — 4 up-blocks, α-skip, S2 patch injection at levels matching 16×16 and 32×32, output: 3 frac channels + 1 binary B channel
12. `HeightDecoder` — 4 up-blocks, τ-skip, first block concatenates α-bottleneck side input, S1 patch injection at 16×16 and 32×32, output 1 channel, GradScale α=0.2 at entry
13. Top-level `DualEncDualDecFusion`:
    - `forward(batch)` returns `{"fraction": B,3,H,W, "height": B,1,H,W, "binary": B,1,H,W}`
    - `predict()` assembles the 4-channel metric output
14. Register in `train.py` model dispatch as `dual_enc_dec_fusion`
15. Print parameter count at instantiation, compare to 2A_vegboost. Expected within 2×.

**Checkpoint 2 — report and wait**:
- Parameter count vs 2A_vegboost
- Forward pass on batch of 2: peak GPU memory, output shapes
- Backward pass: print `α_encoder.first_conv.weight.grad.norm()` and `τ_encoder.first_conv.weight.grad.norm()` — neither should be ~0
- **Stop. Wait for review before Phase 3.**

### Phase 3 — Loss and training

16. `DualPathLoss` in `core/losses.py` (new class, don't modify existing):
    - MAE + soft-Dice (k=5) for fractions
    - Huber (δ=0.5) for height, masked + 0.1 unmasked tail
    - BCE + soft Dice for binary head
    - Returns dict of per-task loss values

17. GradNorm:
    - Learnable `log_w_frac`, `log_w_height`, `log_w_binary`
    - Separate AdamW for these weights, lr ~ 0.025
    - Sample per-task gradient norms through relevant encoder
    - Update log-weights per GradNorm paper, α=1.5
    - `--use-gradnorm` flag, default True

18. `train.py` updates:
    - `--cv-fold` (default 0), `--use-stratified-sampler` (default True), `--use-gradnorm` (default True), `--use-thor` (default False)
    - Proxy formula unchanged, C=4.0
    - Best-checkpoint by proxy on geo-CV val
    - Log per-task losses + GradNorm weights every epoch

19. `predict.py` updates:
    - Read `training_params.txt`
    - Handle dict model output, assemble `[fraction_B, fraction_V, fraction_W, height]`
    - `--tta` flag, D4 8-fold, pre-sigmoid averaging
    - `--blend-binary` flag for inference blending
    - `--threshold-config FILE` to apply per-channel thresholds

**Checkpoint 3 — run smoke + mini, report and wait**:
```bash
./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --cv-fold 0 --experiment-name 7A_smoke \
  --batch-size 4 --epochs 1 --max-batches 2

./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --cv-fold 0 --experiment-name 7A_mini \
  --batch-size 32 --epochs 3
```
- Smoke: pass/fail, GPU memory estimate at batch 32
- Mini: per-task losses, GradNorm weights at epoch 3, per-encoder gradient norm, proxy at epoch 3 (expect ≥ 0.18; lower than 6A because geo-CV is harder)
- **Stop. Wait for review before Phase 4.**

### Phase 4 — Baseline run

20. Launch on single node, 60 epochs:
    ```bash
    ./run_env.sh train.py --model-type dual_enc_dec_fusion \
      --cv-fold 0 --experiment-name 7A_geocv_baseline \
      --batch-size 32 --epochs 60
    ```

21. Threshold scan on val set:
    ```bash
    ./run_env.sh scripts/threshold_scan.py --experiment 7A_geocv_baseline
    ```
    Save `7A_geocv_baseline/thresholds.json`.

22. Predict with TTA + thresholds + binary blending:
    ```bash
    ./run_env.sh predict.py --experiment-name 7A_geocv_baseline \
      --tta --blend-binary --threshold-config thresholds.json
    ```

**Checkpoint 4 — report**:
- Best epoch metrics: IoU_B/V/W, RMSE_B/V, proxy (geo-CV val)
- Per-channel optimal thresholds
- TTA-augmented metrics
- Comparison vs `2A_vegboost` (with caveat about CV change)
- Sample predictions: visualise tiles, save PNG
- **Do NOT submit without explicit approval.**

### Phase 5 — Ablations (later)

Reserved for follow-up prompt. Priority order:
1. Add THOR_s1 to height decoder + THOR_s2 to fraction decoder
2. Disable encoder modality split (claim 1)
3. Disable decoder task split (claim 2)
4. Both S1 and S2 to both decoders (claim 3)
5. Patches at encoder bottleneck instead of decoder skips (claim 4)
6. Random CV vs geographic CV (quantify validation-leak)
7. 3-seed ensemble for final submission

## Acceptance criteria per phase
- Phase 1: data structures correct, normalisation reasonable, fold sizes balanced
- Phase 2: forward/backward works, both encoders receive gradient, param count sensible
- Phase 3: training stable, GradNorm non-degenerate, proxy ≥ 0.18 at epoch 3
- Phase 4: full run completes, threshold scan converges, TTA improves val metrics

## Things to NOT do
- Do not modify existing model classes
- Do not modify existing loss classes
- Do not change the proxy formula (C=4.0)
- Do not run multi-node until single-node baseline validates
- Do not skip checkpoints
- Do not submit to platform without approval
- Do not modify `results.md` or `science.md`; only append to `CLAUDE.md` for new branch/model
- Do not include THOR in baseline (Phase 5 ablation)

## If you get stuck
- Embedding missing → STOP, report
- Modality shape unexpected → STOP, do not silently adapt
- GradNorm unstable → reduce α from 1.5 to 0.5, report
- Cross-attention OOM at batch 32 → 4→2 heads, then 1 layer, then reduce batch
- 32×32 patch injection OOM → drop the 32×32 injection, keep 16×16, report
- Mini-run proxy < 0.12 → STOP, dump predictions, target distributions, per-task losses
- Full run NaN → STOP, do not retry

Report any decision point with multiple reasonable paths rather than picking silently.