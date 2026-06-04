# Coding Task: Phase 8 (`exp-8-decouple`) — Task/Encoder Decoupling + Rare-Class Robustness

## Context
Read `CLAUDE.md`, `docs/results.md`, and `docs/phase5d_thor.md` before starting. This branches
from `exp-7-clean-slate` HEAD; that branch is **frozen** as the record of the 7A+THOR
investigation (THOR rejected — findings 22–24).

**Base config = `7A_v1_base`** (the recommended 7A recipe):
`--model-type dual_enc_dec_fusion --pixel-inputs alpha_earth,tessera
--patch-inputs terramind_s1,terramind_s2 --patch-stem-version v1
--no-use-gradnorm --static-weights 0.65,0.64,1.70 --veg-height-boost 0.0
--cache-dir /home/bassam/nvme_cache/cache7a --num-workers 4 --cv-fold 0
--batch-size 32 --epochs 60 --seed 0`. Internal proxy ~0.40 (IoU_B 0.20 / IoU_V 0.81 /
IoU_W 0.68 / RMSE_V 4.0). **Do not use THOR, the v2/v2b stems, GradNorm, or vegboost.**

## Goal
Raise **IoU_B** (internal 0.20 — our worst class metric and the #1 platform gap), without
regressing IoU_W/IoU_V. **Reproducibility/generalisation rule: rely on the raw model prediction
thresholded at 0.5 — no inference threshold calibration or blend-binary** (they overfit fold-0 val
and won't transfer to the different-region/year test set). The internal proxy already measures raw
hard-0.5, so what we optimise is what the platform sees.

## Prerequisite (do first)
- **Seed the augmentation.** `GeoFMDataset7A.__getitem__` uses `np.random.default_rng()` (no seed),
  so augmentation ignores `--seed` and every run varies (finding 25). Make it deterministic per
  `(seed, epoch, index)` (e.g., derive a per-sample seed from the worker seed + index). Without
  this, the ±0.01 IoU effects below are noise-limited. Confirm two runs with the same `--seed`
  produce identical epoch-1 val metrics.

## Experiments (additive, default-off, byte-identical when off; smoke → STOP for review)
Each behind its own flag; with the flag at default the model/training must be **byte-identical** to
`7A_v1_base` (verify max_abs_diff = 0 vs a captured reference, as in `_thor_smoke/verify.py`).
Write every new flag to `training_params.txt` and confirm `predict.py` round-trips it.

### P4 — single-task ablation (do FIRST; it gates P1)
Add `--task {all,fraction,height}` (default `all`). `fraction` drops the height decoder + height
loss; `height` drops the fraction + binary losses. Train fraction-only and height-only, compare to
joint on each metric. *Question: is multi-task interference real?* If fraction-only IoU_B ≫ joint,
the height task is stealing capacity from building → favours more separation (and tells P1 to
protect alpha *more*, not less). If ≈ joint, the loss combination is not the problem.

### P3a — aggressive rare-class oversampling (direct IoU_B lever)
Make the stratified-sampler weights configurable (e.g. `--strata-weights 1,2,4,8`, default keeps
the current `1,1.5,2,3`). Optionally stratify **building separately from water** so buildings get
their own boost. *Question: does more rare-class signal per step lift IoU_B/IoU_W?* Also directly
addresses the rare-class ignition fragility (finding 24).

### P1 — symmetric cross-encoder gradients (direction set by P4)
Today only `alpha` gets fraction-loss gradient + a throttled (`GradScale α=0.2`) height-loss
gradient via the bridge; `tessera` gets only height-loss gradient. Two levers (both already partly
coded): `--use-fraction-bridge` (tessera→fraction bridge, mirror of the height bridge — TESSERA is
temporal/multispectral, plausibly useful for veg/water) and a sweep of the bridge `GradScale α`
(AlphaEarth encodes S1+S2 so it carries real height signal; 0.2 may underuse it). **Decide the α
direction from P4**: if alpha is being degraded by height, throttle *more*; if it has spare
capacity, share *more*.

### P3b — SAR-safe augmentation
SAR (S1) backscatter depends on look/incidence direction, so D4 rotation on `terramind_s1` teaches
false invariance. Add an option to **skip rotation for S1** (keep flips + embedding-space jitter),
or to disable geometric aug entirely. Note: CLAUDE.md records augmentation hurting slightly before,
so the honest hypothesis is "less/cleaner aug ≥ current aug."

### P2 — patch-token value check (lowest priority)
Before any injection enhancement, **ablate patch tokens out** (pixel encoders only) to test whether
the terramind patches earn their place at all. Only if they clearly contribute, consider an extra
injection level (64×64) — and watch rare-class stability (deeper/more injection destabilised rare
classes in Phase 5D).

## Sequence
1. Seed augmentation (prereq).
2. P4 (single-task) + P3a (oversampling) in parallel on separate nodes.
3. Read → P1 (correct α direction) + P3b (SAR-safe aug).
4. P2 only if a patch ablate-out shows terramind matters.

## Constraints / things to NOT do
- Additive only; never alter the default (`7A_v1_base`) path. Byte-identical-when-off, verified.
- No THOR, no v2/v2b stem, no GradNorm, no vegboost. Batch ≥ 32 (rare-class ignition needs it).
- No inference threshold calibration / blend-binary; raw 0.5 prediction only.
- No platform submission without explicit approval.
- Smoke each change (batch 4, finite loss, byte-identical-off, GPU mem); **stop and report before
  any 60-epoch training.** Be mindful of load on `head` when it is training.
- Report any decision point with multiple reasonable paths instead of picking silently.
