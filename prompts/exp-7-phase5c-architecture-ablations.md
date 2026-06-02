# Coding Task: Phase 5C — Architecture Ablations + VegBoost Sensitivity (Revised)

## Context
Read `CLAUDE.md`, `docs/results.md`, `docs/science.md` first.

Phase 5B (data/loss ablations) is complete:

| Run | Proxy | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V | Best epoch |
|-----|-------|-------|-------|-------|--------|--------|-----------|
| 7A_base_e90 | 0.396 | 0.206 | 0.808 | 0.686 | 2.14 | 3.92 | 59 |
| 7A_no_vegboost_e90 | 0.399 | 0.204 | 0.809 | 0.684 | 2.09 | 3.91 | 40 |
| 7A_no_gradnorm_e90 | 0.393 | 0.206 | 0.810 | 0.680 | 2.15 | 3.97 | 40 |
| 7A_no_stratified_e90 | 0.287 | 0.200 | 0.809 | 0.000 | 2.15 | 4.03 | 65 |

**Conclusions from 5B:**
- Stratified sampler is irreplaceable (IoU_W collapses without it)
- vegboost @ 1.0 is doing essentially nothing
- GradNorm gives a marginal +0.003

**Open questions for 5C:**
1. Does the cross-encoder bridge (α-bottleneck side-input to height decoder) earn its complexity?
2. Does the auxiliary binary B head actually help internal IoU_B?
3. Is the static-weight simplification reproducible (drop GradNorm, fix weights)?
4. Was vegboost @ 1.0 just too low a dose — does 5.0 pull RMSE_V meaningfully below the cliff?

**Skipped from earlier plan:** `7A_abl_single_enc` (single encoder taking both modalities). The 6A evidence already shows that concatenating α+τ in a single encoder is destructive (IoU_B collapsed from 0.168 to 0.017). Re-testing with a properly-sized single encoder is interesting but not actionable — we would still ship the dual encoder either way. Saves ~3 hours of implementation.

---

## Branch and naming
Stay on `exp-7-clean-slate`. Commit this spec as `prompts/exp-7-phase5c-architecture-ablations.md` as the first commit.

| Node | Experiment | What changes |
|------|-----------|--------------|
| n1 | `7A_simple` | Static weights (no GradNorm), no vegboost — reference point |
| n2 | `7A_abl_no_bridge` | Drop α-bottleneck side input to height decoder |
| n3 | `7A_abl_no_binary` | Drop auxiliary binary B head |
| n4 | `7A_vegboost_high` | vegboost = 5.0 (sensitivity test) |

All three implementation changes are CLI flags only — no new model classes, no architectural rewrites. Total implementation: ~30 minutes.

---

## Pre-flight: training duration

All 5B runs peaked between epochs 40 and 65. **Train for 60 epochs**, not 90. If any run shows the proxy still climbing at epoch 50, extend to 80 on that run only.

---

## Implementation

### Change 1 — Static loss weights (for `7A_simple`)
Add `--static-weights "w_f,w_h,w_b"` flag. When `--use-gradnorm False` is set, parse the static weights and use them directly in the loss combination instead of GradNorm's learned weights.

Defaults: `--static-weights "0.65,0.64,1.70"` (the GradNorm-converged values from `7A_base_e90` at epoch 59).

### Change 2 — No cross-encoder bridge (for `7A_abl_no_bridge`)
Add `--no-height-bridge` flag. When set, the height decoder's first block does NOT receive the α-bottleneck as a side input. The bridge code path is bypassed (set to zero or skipped in the forward).

This is one branch in the forward pass — ~5–10 lines.

### Change 3 — No auxiliary binary head (for `7A_abl_no_binary`)
Add `--no-binary-head` flag. When set:
- The binary head module is not instantiated (or its output is not used)
- `DualPathLoss` skips the binary loss term entirely (`L_binary = 0`)
- The `binary` output is not returned from the forward (or returned as `None`)
- `predict.py` falls back to using the fraction head's channel 0 only when `--blend-binary` is set but no binary head is present (skip the blend step)

This requires guards in three places: model `__init__`, model `forward`, loss class. ~15 lines.

### Change 4 — High vegboost (for `7A_vegboost_high`)
No code change. Pass `--veg-height-boost 5.0` (existing flag).

---

## Smoke tests

Before launching training, run three smoke tests (one per new code path):

```
# Smoke 1 — static weights
./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --use-gradnorm False --static-weights "0.65,0.64,1.70" \
  --experiment-name smoke_static \
  --batch-size 4 --epochs 1 --max-batches 2

# Smoke 2 — no bridge
./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --no-height-bridge \
  --experiment-name smoke_no_bridge \
  --batch-size 4 --epochs 1 --max-batches 2

# Smoke 3 — no binary head
./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --no-binary-head \
  --experiment-name smoke_no_binary \
  --batch-size 4 --epochs 1 --max-batches 2
```

Vegboost @ 5.0 doesn't need a separate smoke — the loss path is identical to vegboost @ 1.0 which already works.

All three smokes must run, produce finite loss, and have similar GPU memory to baseline.

**Stop after smoke tests. Report results and wait for review before launching the four training runs.**

---

## Training runs (four nodes, 60 epochs each)

### Node n1 — 7A_simple (reference)
```
./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --cache-dir /path/to/cache --num-workers 8 \
  --use-gradnorm False --static-weights "0.65,0.64,1.70" \
  --veg-height-boost 0.0 \
  --cv-fold 0 --batch-size 32 --epochs 60 --seed 0 \
  --experiment-name 7A_simple
```

### Node n2 — 7A_abl_no_bridge
```
./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --no-height-bridge \
  --cache-dir /path/to/cache --num-workers 8 \
  --use-gradnorm False --static-weights "0.65,0.64,1.70" \
  --veg-height-boost 0.0 \
  --cv-fold 0 --batch-size 32 --epochs 60 --seed 0 \
  --experiment-name 7A_abl_no_bridge
```

### Node n3 — 7A_abl_no_binary
```
./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --no-binary-head \
  --cache-dir /path/to/cache --num-workers 8 \
  --use-gradnorm False --static-weights "0.65,0.64,1.70" \
  --veg-height-boost 0.0 \
  --cv-fold 0 --batch-size 32 --epochs 60 --seed 0 \
  --experiment-name 7A_abl_no_binary
```

### Node n4 — 7A_vegboost_high
```
./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --veg-height-boost 5.0 \
  --cache-dir /path/to/cache --num-workers 8 \
  --use-gradnorm True \
  --cv-fold 0 --batch-size 32 --epochs 60 --seed 0 \
  --experiment-name 7A_vegboost_high
```

**Rationale for asymmetric configs**: n1, n2, n3 all use static weights so the architecture comparisons share identical training dynamics. n4 uses GradNorm because vegboost interacts with task weighting and we want GradNorm to mediate.

For static-weights runs, `--static-weights "0.65,0.64,1.70"` means w_f=0.65, w_h=0.64, w_b=1.70. n3 (no binary head) doesn't use w_b — internally the loss should ignore that weight when the binary head is disabled.

---

## Acceptance criteria

Comparison table with `7A_no_vegboost_e90` as reference:

```
| Experiment        | Best ep | Proxy | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V |
| no_vegboost (ref) | 40      | 0.399 | 0.204 | 0.809 | 0.684 | 2.09   | 3.91   |
| 7A_simple         | ?       | ?     | ...                                       |
| 7A_abl_no_bridge  | ?       | ?     | ...                                       |
| 7A_abl_no_binary  | ?       | ?     | ...                                       |
| 7A_vegboost_high  | ?       | ?     | ...                                       |
```

Then answer four questions:

1. **Does the static-weight simplification reproduce GradNorm performance?** Compare `7A_simple` to `7A_no_vegboost_e90` (which used GradNorm). If proxy is within ±0.005, drop GradNorm permanently.

2. **Does the cross-encoder bridge matter?** Compare `7A_abl_no_bridge` to `7A_simple`. If proxy is within ±0.005 and RMSE_B doesn't regress, the bridge is removable. Watch RMSE_B specifically — the bridge's purpose was to give the height decoder access to α-encoder's building spatial features. If RMSE_B regresses by 0.2m or more, the bridge is doing real work for building height.

3. **Does the auxiliary binary head help?** Compare `7A_abl_no_binary` to `7A_simple` on IoU_B specifically. If `7A_abl_no_binary` IoU_B is within 0.005 of `7A_simple`, the binary head is not contributing on internal val (and was likely cancelled by the blend+threshold interaction on the platform). If `7A_abl_no_binary` IoU_B drops significantly, the binary head is doing work — investigate why it isn't translating to platform IoU_B gains.

4. **Does high vegboost help RMSE_V?** Compare `7A_vegboost_high` RMSE_V to `7A_no_vegboost_e90` (3.91m). Three scenarios:
   - RMSE_V drops to ~3.6–3.7m AND other metrics hold → useful safety knob against the 3.9m cliff
   - RMSE_V drops but IoU_V regresses → vegboost is trading IoU for RMSE, undesirable
   - RMSE_V doesn't change → vegboost is structurally ineffective in 7A; drop permanently

**Do NOT submit anything to the platform without explicit approval.**

---

## Things to NOT do
- Do not change the dataset, cache, or augmentation logic
- Do not modify any existing model class — only add CLI flag handling
- Do not run on multiple nodes before smoke tests pass
- Do not submit to platform without approval
- Do not modify `results.md` or `science.md`
- Do not run beyond 60 epochs unless the proxy curve at epoch 50 is still climbing

## If you get stuck
- `--no-binary-head` breaks loss class because `L_binary` is undefined → return `torch.tensor(0.0, device=...)` for the binary loss and skip the weight multiplication
- `--no-binary-head` breaks `predict.py --blend-binary` → log a warning, skip blending, continue with fraction head only
- `--no-height-bridge` causes shape mismatch in height decoder's first block → the projection layer that combines α and τ bottlenecks must handle the no-bridge case (skip the addition entirely, do not pass zeros through the projection)
- 7A_vegboost_high loss explodes at any point → STOP, do not retry, report
- Any NaN → STOP, do not retry

Report any decision point with multiple reasonable paths rather than picking silently.