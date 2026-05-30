# Coding Task: TESSERA Stream + Cross-Attention Patch Fusion

## Context
You are working on the ESA Φ-lab GeoFM Challenge (see `CLAUDE.md`, `results.md`, `science.md` in project root). Current best **platform** model is `2A_vegboost` (`AttentionFusedDecoder`, 2A family, platform score 0.3721, branch `exp-2A-attention-gate`). We fork from `exp-4-ynet-gradhook` (`YNetAttentionFusedDecoder`, 4A family) because it carries the GradScale hook, `augment_triplet()`, and the multi-pixel dict dataset we need. The decision from the science session is to test two architectural changes together:

1. **Add TESSERA as a second pixel-level encoder stream** (128 channels, 256×256, pixel-wise, derived from S1/S2 time series — carries phenological/temporal signal that AlphaEarth's annual composite lacks).
2. **Replace bottleneck patch fusion with cross-attention** (query = pixel features, key/value = patch tokens from terramind_s1 + terramind_s2). The current "linear projection + spatial broadcast" is the weakest link in the fusion pipeline.

Hypothesis: TESSERA closes IoU_V and possibly IoU_W; cross-attention extracts more from the patch tokens than spatial broadcast does. Combined, expected gain: +0.03 to +0.05 platform score.

## Branch Setup

```bash
cd /mnt/head/users/bassam/src/geofmchal
git status                          # verify clean
git checkout exp-4-ynet-gradhook    # parent branch (has GradScale hook, C=4.0 proxy, worker_init_fn fix)
git pull
git checkout -b exp-6-tessera-xattn
```

If there are uncommitted changes on `exp-4-ynet-gradhook`, STOP and report — do not branch over dirty state.

**Naming convention for this task:**
- Branch: `exp-6-tessera-xattn`
- Model class: `YNetTesseraXAttn`
- Model dispatch name: `ynet_tessera_xattn`
- Experiment names: `6A_smoke_xattn` (smoke test), `6A_mini_xattn` (mini run), `6A_tessera_xattn` (full run)

The `6A_` prefix keeps results.md sortable by experiment family. This task is the founding experiment of the 6A family (TESSERA + cross-attention architectures).

## Implementation Scope

### 0. Pre-flight check (do this FIRST, before any implementation)

Branching from `exp-4-ynet-gradhook` instead of the branch where `2A_vegboost` was trained means we may be missing the vegetation height boost loss. Verify the following are present on the current branch:

```bash
grep -n "veg_height_boost\|veg.*boost" core/losses.py
grep -n "GradScale\|grad_scale\|register_hook" core/model.py
grep -n "rmse_b / 4\|/ 4.0\|C.*=.*4" train.py
grep -n "worker_init_fn\|np.random.seed" train.py
```

Required:
- `veg_height_boost` term in `ImprovedCompositeLoss` (or a `+ veg_height_boost` addition pattern)
- `GradScale` utility class or `register_hook` mechanism for scaling height-decoder gradients
- Proxy formula using C=4.0 (not C=30.0)
- `worker_init_fn` setting numpy seed per worker

If `veg_height_boost` is **missing**:
- Identify which branch `2A_vegboost` was trained on: `git log --all --oneline | grep -i vegboost` or check the commit history of `core/losses.py`.
- Cherry-pick the commit that added `veg_height_boost` onto `exp-6-tessera-xattn` before proceeding.
- Report which commit was cherry-picked and confirm tests still pass before moving to Section 1.

If `GradScale` mechanism is **missing or hard-coded** into the existing 4A decoder (i.e., not reusable as a utility):
- Report this and propose a clean extraction before continuing. The new model needs a reusable hook.

If proxy C or worker_init_fn fix is missing:
- STOP and report. These are listed in `CLAUDE.md` as already applied — their absence indicates branch confusion.

### 1. Verify TESSERA is available in the dataset

- Check `core/dataset.py` for TESSERA support. The `Emb2HeightsDataset` class loads embeddings by name from disk.
- Confirm TESSERA embeddings exist in the data root. Run:
```bash
ls /mnt/head/users/bassam/data/geofmdata/ | grep -i tessera
```
- TESSERA shape should be `(128, 256, 256)` — pixel-level, same spatial resolution as alpha_earth.
- If TESSERA is missing on disk or not wired into the dataset class, STOP and report before writing model code.

### 2. Extend dataset to support multiple pixel inputs

- The current code accepts `--pixel-inputs alpha_earth` (single string). Extend to comma-separated list: `--pixel-inputs alpha_earth,tessera`.
- The dataset should return a dict like `{"alpha_earth": tensor(64,H,W), "tessera": tensor(128,H,W), "patches": ...}` rather than a single pixel tensor, so the model can route streams independently.
- **Critical**: the synchronized `augment_triplet()` must apply identical geometric augmentations to all pixel inputs AND the target. Patch embeddings (16×16) should not receive geometric augmentation — they're already coarse semantic tokens.
- Add a quick sanity assertion that all pixel streams have the same spatial dimensions.

### 3. New model: `YNetTesseraXAttn`

Create in `core/model.py`. Architecture:

```
alpha_earth (64ch, 256×256) ──┐
                              ├─→ [concat or parallel CNN stems] ──→ [shared U-Net encoder]
tessera (128ch, 256×256) ─────┘                                              │
                                                                             │
                                                                  [bottleneck features, e.g. 512ch @ 16×16]
                                                                             │
                                                                             │ ←── cross-attention
                                                                             │       Q = bottleneck (16×16 tokens, 512d)
                                                                             │       K,V = patch tokens (terramind_s1+s2, 256 tokens, 1536d → projected to 512d)
                                                                             │
                                                                  [fused bottleneck features]
                                                                             │
                                                                  [shared U-Net decoder]
                                                                             │
                                                            ┌────────────────┴────────────────┐
                                                       [class head]                    [height head]
                                                       ch 0,1,2                          ch 3
                                                       (2 conv layers)                  (2 conv layers)
                                                       + GradScale α=0.1 hook on height head
```

**Design notes**:
- **Input fusion**: start with **simple concatenation** at the input (`torch.cat([alpha, tessera], dim=1)` → 192ch into first conv). Optionally add a 1×1 projection back to 64ch to keep encoder dimensions identical to current model. This is the cheap option — a separate stem per stream with mid-level fusion is the principled option but adds complexity. **Start cheap, validate, then iterate.**
- **Cross-attention block**: standard multi-head attention (4 or 8 heads). The patch tokens (16×16 = 256 tokens × 1536d) are projected to the bottleneck dim, then a single MHA layer with learned positional encodings on both query and key sides. Add a residual connection: `bottleneck = bottleneck + MHA(LN(bottleneck), LN(patches_proj))`. Follow with a small FFN block (standard transformer block layout).
- **Decoder**: keep the shared decoder body from current `YNetAttentionFusedDecoder` but split the *final* 1–2 blocks into two heads (class head outputs 3ch, height head outputs 1ch). This is the "shared decoder + task-specific heads" pattern from the science session — not full Y-Net decoupling.
- **GradScale hook**: keep α=0.1 on the height head branch as in current 4A architecture. Use the utility verified in Section 0.
- **Output**: concatenate `[class_head_output, height_head_output]` to get 4-channel tensor matching current target format. No sigmoid in forward (apply in loss / at inference).

**Implementation discipline**:
- Add the model class alongside the existing one — do not delete `YNetAttentionFusedDecoder`.
- Register the new model in whatever model-name dispatch dict `train.py` uses (search for `ynet_attention_fusion` to find it). New name: `ynet_tessera_xattn`.
- Print parameter count at instantiation. If it explodes >2× the current model, report before training.

### 4. Loss function

Use the **current best loss unchanged**: `ImprovedCompositeLoss + veg_height_boost` from `core/losses.py` (verified present in Section 0). Do not modify the loss in this experiment. Isolate the architecture change.

### 5. Training script changes

- `train.py` already supports `--pixel-inputs` and `--patch-inputs` as comma-separated. Verify it parses correctly and passes through to the dataset/model.
- Keep batch size 32, epochs 60, optimiser/scheduler unchanged.
- Worker init function: keep the numpy per-worker seeding fix.
- Proxy formula: C=4.0 (already fixed). Do not touch.

### 6. Predict + package + submit

- `predict.py` auto-reads `training_params.txt` — verify the new model loads correctly.
- Test prediction on a few validation tiles before running the full prediction pass.
- `package.py` and `uploader/submit.py` should work unchanged.

## Testing Plan (before launching full 60-epoch run)

Run these in order. Do not proceed to the next until the previous passes.

1. **Smoke test — single batch forward pass** (5 min)
```bash
./run_env.sh train.py --model-type ynet_tessera_xattn \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --experiment-name 6A_smoke_xattn \
  --batch-size 4 --epochs 1 --max-batches 2
```
If `--max-batches` doesn't exist, add it as a temporary CLI flag for this test. Verify: forward pass runs, loss is finite, backward pass updates parameters, GPU memory under 40GB.

2. **Mini training run — 3 epochs on full data** (~20 min)
```bash
./run_env.sh train.py --model-type ynet_tessera_xattn \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --experiment-name 6A_mini_xattn --batch-size 32 --epochs 3
```
Verify: proxy score is non-degenerate (should be ≥0.15 by epoch 3 if architecture is sound), no NaN losses, gradients flow to both encoder streams (log grad norms per stream — quick way: print `model.alpha_stem.weight.grad.norm()` vs `model.tessera_stem.weight.grad.norm()` at end of epoch 1; if either is ~0 the stream is dead).

3. **Full training run** (~6–8 hours, single node)
```bash
./run_env.sh train.py --model-type ynet_tessera_xattn \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --experiment-name 6A_tessera_xattn --batch-size 32 --epochs 60
```

## Acceptance Criteria

Report back with:
- Pre-flight check results (Section 0): which utilities were present, which were cherry-picked, from which commits.
- Parameter count of new model vs `2A_vegboost`.
- Smoke test result (pass/fail + GPU memory).
- Mini-run final proxy score (epoch 3) + per-stream gradient norms (confirms no dead stream).
- Full run: best epoch, internal IoU_B/V/W, RMSE_B/V, proxy (C=4.0). Compare against `2A_vegboost` baseline in `results.md`.
- Whether to submit to platform. **Do NOT submit automatically** — present results and wait for confirmation. We have 12-hour submission limits and want to choose carefully.

## Things to NOT do

- Do not change the loss function.
- Do not change the optimiser, scheduler, or training hyperparameters.
- Do not delete or modify existing model classes — only add the new one.
- Do not run on all 4 nodes in parallel until the single-node full run validates. One clean signal first.
- Do not submit to the ESA platform without explicit confirmation.
- Do not modify `results.md` or `science.md` — only `CLAUDE.md` if needed to document the new model name and branch. Append, do not rewrite.

## If You Get Stuck

- Pre-flight check finds missing `veg_height_boost` and the source branch is unclear → STOP, report `git log` output for `core/losses.py`, ask for guidance.
- TESSERA not on disk → STOP, report.
- TESSERA shape mismatch → STOP, report.
- Cross-attention OOM at batch 32 → reduce attention heads to 2, report, retry. If still OOM, reduce bottleneck projection dim before attempting batch size reduction.
- Mini-run proxy < 0.10 by epoch 3 → STOP, dump a few sample predictions and target distributions, report. Something is wired wrong.
- Full run NaN loss → STOP, do not auto-retry. Report training curve.

Report any decision point where multiple reasonable paths exist rather than picking silently.