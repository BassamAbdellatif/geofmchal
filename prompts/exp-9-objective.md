# Coding Task: Phase 9 (`exp-9-objective`) — Building-Objective Redesign (P1 + P2)

## Context
Read `docs/roadmap_final21d.md`, `docs/results.md` (findings 1–28), and `docs/Literature_review.md`
(Alternative 2) first. Branch from **`exp-8-decouple` HEAD** (recommended name `exp-9-objective`;
adjustable). This is the lead bet to break the **intrinsic IoU_B ≈ 0.18 ceiling** (f28): the
ceiling is *not* multi-task interference (f26) or sampling frequency (f27), so we attack the
**building objective itself**.

**Base config = `7A_v1_base`** (unchanged): `--model-type dual_enc_dec_fusion --pixel-inputs
alpha_earth,tessera --patch-inputs terramind_s1,terramind_s2 --patch-stem-version v1
--no-use-gradnorm --static-weights 0.65,0.64,1.70 --veg-height-boost 0.0 --batch-size 32 --seed 0`.
Seeded augmentation is in place (f25). bf16 `--amp` is validated to preserve water ignition.

### What the objective is *today* (verified — do not assume)
- Building IoU is scored from **fraction channel 0**: `sigmoid(fraction)[:,0] > 0.5`
  (`train.py:574–577`). The fraction head is **3 independent sigmoids** (building, veg, water) — NOT
  a softmax simplex. There is **no "other"/background channel** (target ch3 is height).
- Fraction loss (`core/losses.py` `DualPathLoss`): `MAE(sigmoid, target)` + `dice_lambda *`
  **lumped** `_soft_dice` over the shifted sigmoid `σ(k·(logit−0.5))`, `k=5`, **class-blind across
  all 3 channels**.
- A separate **binary head** (BCE + Dice on `build_frac>0.5`, weight 1.70) shapes the encoder but
  **does not feed the metric** (metric reads fraction ch0). Leave it as-is.
- `TverskyLoss(alpha=0.3, beta=0.7)` already exists in `core/losses.py` — reuse it.

## Goal
Raise **IoU_B** (internal seeded anchor ≈ 0.178) without regressing IoU_W (≈0.68) / IoU_V (≈0.81).
**Reproducibility rule:** raw 0.5 prediction only — no threshold calibration / blend-binary.

## Prerequisite
Capture a byte-identical reference (fixed batch, all new flags default) of both the model forward
and the per-task loss dict, as in `_thor_smoke/verify.py`. Every experiment below must reproduce
`max_abs_diff = 0` against it when its flag is at default.

---

## P1 — Building-aware overlap loss (default-off, byte-identical when off)

Make the fraction overlap term **building-aware** instead of class-blind.

**Flags** (all defaults reproduce `7A_v1_base` exactly):
- `--building-overlap {dice,tversky,focal_tversky}`  (default `dice`)
- `--tversky-alpha FLOAT` (default `0.3`, FP weight) / `--tversky-beta FLOAT` (default `0.7`, FN
  weight). β>α favours recall — buildings are under-predicted.
- `--focal-tversky-gamma FLOAT` (default `1.333`; only for `focal_tversky`; `(1−Tversky)**(1/γ)`).
- `--building-overlap-weight FLOAT` (default `1.0`): multiplier on the **building channel's** overlap
  term relative to veg/water.
- `--dice-k FLOAT` (default `5.0`): expose the existing shifted-sigmoid sharpness.

**Implementation guard (critical for byte-identical):** when `building_overlap=='dice' and
building_overlap_weight==1.0 and dice_k==5.0`, call the **existing lumped `_soft_dice`** path
unchanged (lumped 3-channel Dice ≠ mean of per-channel Dice, so do **not** silently refactor the
default path). Only when a non-default P1 flag is set, switch to a **per-channel** decomposition:
- veg (ch1) + water (ch2): keep soft-Dice on `σ(k·(logit−0.5))`.
- building (ch0): apply the selected overlap (`tversky`/`focal_tversky` on the same shifted sigmoid,
  target `build_frac>0.5`), scaled by `building_overlap_weight`.

Write all five flags to `training_params.txt`; confirm `predict.py` round-trips them (loss-only
flags don't change inference, but the params file must parse cleanly).

## P2 — "Others" channel + softmax simplex (default-off, byte-identical when off)

**Flag:** `--fraction-head {sigmoid3,softmax4}` (default `sigmoid3` = current path, byte-identical).

When `softmax4`:
- **Model:** `FractionDecoder` final conv `out_channels 3→4`, channel order **[building, veg, water,
  other]**; apply **softmax over the 4** (replaces the 3 independent sigmoids). Build under the flag
  (conditional construction, like the existing optional heads). `build_model` + `predict.py` must
  reconstruct from `FRACTION_HEAD` in `training_params.txt`.
- **Target:** derive `other = clamp(1 − build − veg − water, 0, 1)`; stack to a 4-way target.
- **Loss:** `MAE(softmax_prob, target4)` over all 4 channels + the P1 overlap on the 3 supervised
  goal channels (building still gets the building-aware term). Do **not** supervise "other" with an
  overlap term (it's a derived sink); MAE on it is enough to enforce the simplex.
- **Metric (`evaluate_7a`) + `predict.py`:** branch on head type. `sigmoid3` → exact current path.
  `softmax4` → building/veg/water prob = `softmax(fraction)[:,0:3]`, threshold `>0.5` as before
  (height channel handling unchanged). Keep the `sigmoid3` branch byte-identical.

**Rare-class risk to watch (f24):** softmax makes water *compete* with "other"; rare-class ignition
may be fragile. Treat epoch-2–3 IoU_W as a gate exactly as in the AMP validation.

---

## Smoke (batch 4, ≤2 steps; then STOP and report — no 40-epoch run yet)
For each of: `dice` (default → assert byte-identical), `tversky`, `focal_tversky`, `softmax4`,
`softmax4 + tversky`:
1. finite forward + loss; 2. `evaluate_7a` runs and returns a valid IoU_B for that head;
3. one backward step; 4. GPU-mem sanity at bs32 with `--amp` (softmax4 adds one channel — expect
≈ base). Confirm the default-off byte-identical check passes.

## Campaign (only after smoke approval) — 40 epochs, seeded, `--amp`, vs the seeded anchor
- **A:** `--building-overlap tversky` (0.3/0.7), `sigmoid3`.
- **B:** `--fraction-head softmax4`, `dice`.
- **C:** `--fraction-head softmax4 --building-overlap tversky` (the combined bet).
Read **last-10-epoch-mean IoU_B** vs anchor 0.178 (swing ±0.02 — ties below that). Gate every run on
**IoU_W not collapsing** (epoch 2–3 ignition; finding 24). If a Tversky/softmax run over-predicts
buildings (IoU_B up but IoU_V/precision down), tune α/β toward 0.4/0.6.

## Constraints / NOT to do
- Additive only; default path byte-identical, verified. No THOR, no v2/v2b stem, no GradNorm, no
  vegboost. Batch ≥ 32 (rare-class ignition). `--amp` ok (validated).
- No threshold calibration / blend-binary; raw 0.5 only. No platform submission without approval.
- Leave the binary head and its weight (1.70) unchanged — it shapes the encoder, not the metric.
- Report any fork with multiple reasonable paths instead of picking silently; STOP at the smoke
  checkpoint.

---

## Validation protocol & campaign (post-smoke) — multi-fold, P/R-aimed

Smoke is **done & passing** (byte-identical-off verified; all 5 configs finite/grad-ok). The P/R
diagnostic (finding 30) shows building is **recall-limited**, so the campaign is aimed at P1
(Tversky), keeps the softmax4+tversky combo as the "does *other* help precision" test, and **drops
pure softmax4** (it worsens recall). Selection is **geo multi-fold** (finding 31): an arm wins only
if it beats the anchor on **both** folds; the final submission model is later trained on all 5 folds.

### Arms (3 configs × folds {0,1} = 6 runs), 40 epochs, seed 0, `--amp`
| name | extra flags vs `7A_v1_base` |
|------|-----------------------------|
| `9_anchor_f{0,1}` | — (seeded default = control) |
| `9_tversky_f{0,1}` | `--building-overlap tversky --tversky-alpha 0.3 --tversky-beta 0.7` |
| `9_sm4tv_f{0,1}` | `--fraction-head softmax4 --building-overlap tversky --tversky-alpha 0.3 --tversky-beta 0.7` |

### Waves (4 nodes)
- **Wave 1 (fold 0 + fold-1 control):** `9_anchor_f0`, `9_tversky_f0`, `9_sm4tv_f0`, `9_anchor_f1`.
- **Wave 2 (fold-1 arms):** `9_tversky_f1`, `9_sm4tv_f1` (2 nodes; 2 free, reserved for expansion).

### Read-out & gates
- **IoU_B**: last-10-epoch mean per run; Δ = arm − anchor *within each fold*. Win = Δ>0 on **both**
  folds, magnitude above the ±0.02–0.03 swing.
- **IoU_W gate**: watch epoch 2–3 ignition (finding 24); softmax4 makes water compete with *other* —
  if IoU_W stays ~0, the combo is killing rare classes.
- **Precision check**: for `9_sm4tv` vs `9_tversky`, compare building FP (the "other"-sink should cut
  false positives) — if it doesn't beat plain Tversky, *other* isn't earning its keep.

### Conditional expansion (only if Wave 1/2 shows signal)
Add a gentler `--tversky-alpha 0.4 --tversky-beta 0.6` arm and extend to **fold 2** (folds 0+1+2)
to confirm before committing to an all-folds final-model train.
