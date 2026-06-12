# Phase 12 — Height-First Campaign (2026-06-11)

**Standing:** rank **43**, `11_fresh_f1` = **0.4125**. **Top-1:** ~0.5448 (gap **0.132**).
**Deadline:** 2026-06-30 (~19 days). Read `docs/results.md` f35 and `docs/phase11_fresh_architecture.md` first.

## 1. The thesis (from the platform anchor, f35)
The fresh architecture closed the building/extraction gap (IoU_B 0.35→0.43, RMSE_B 2.23 near-top). The
remaining gap to #1 is **concentrated in one metric**: **RMSE_veg 3.81 m earns 0.010 of a possible
0.20 → 0.190 of unrealized score, more than every other metric's room combined.** Veg *location* is
solved (IoU_V 0.806). This is a **height-regression-quality** problem on vegetation, and it is
self-inflicted: we shipped with the veg-height knobs OFF/inherited.

**Budget of the 0.132 gap to #1 (by metric room × weight):**
| lever | room | priority | why |
|-------|------|----------|-----|
| **RMSE_veg** 3.81→ | **0.190** | **P0** | knobs were off; veg located but mis-heighted |
| IoU_build 0.43→0.5+ | 0.143 | P2 | fresh arch already moved it; push further |
| IoU_water 0.48→ | 0.078 | P3 | rare-class, historically sticky |
| RMSE_build 2.23 | 0.139 (≈maxed) | — | near top of field; **do not chase** |

## 2. P0 — Drive RMSE_veg down (the campaign)

### H1 — Turn the veg-height knobs back ON (cheap, highest expected value)
`11_fresh_f1` ran `veg_height_boost=0.0` and `--static-weights 0.65,0.64,1.70` (height weight **0.64**,
inherited from old-7A GradNorm, never re-tuned). Two one-line changes:
- **`--veg-height-boost {2,3}`** — extra masked-Huber on veg pixels (`veg_frac > 0.1`); the term exists
  *specifically* for RMSE_V. The "wash" verdict (Phase 11) was a fast-split proxy artifact (f35).
- **Raise the height task weight**, e.g. `--static-weights 0.65,1.0,1.70` (or higher). Height is
  under-weighted vs fraction; IoU_V is already maxed, so trading a little fraction for a lot of RMSE_V
  is a good bet.
- **Judge on RMSE_V *specifically*** (not the blended proxy that misled us), multi-fold, not fast-split.

### H2 — Stop buildings and veg sharing one height head
The Huber masks `(build OR veg)` jointly (`losses.py:344`). Buildings (low-variance, 2.23 ✓) and veg
(high-variance, 3.81 ✗) compete in one regression. Options, in order:
- **Separate the masked terms** with independent weights (a `building_height_weight` vs
  `veg_height_boost`) — keep one head, decouple the loss.
- **Scale-aware veg-height loss** — the `losses.py` docstring already flags "revisit a scale-aware
  height loss if RMSE stalls." Veg canopy height is heavy-tailed; per-pixel normalization or a
  log/relative term reduces tall-canopy domination of the L2.
- (heavier) **Dedicated veg-height head** off the shared pyramid.

### H3 — AlphaEarth-GEDI/DEM → veg-height pathway (Bet 1, strengthened)
AlphaEarth's inputs include **GEDI LiDAR canopy height + GLO-30 DEM** — it literally encodes height. The
fresh model fuses alpha symmetrically into both decoders, but if H1/H2 stall, add an explicit
alpha-derived height residual / give the height decoder a dedicated high-capacity alpha branch.

## 3. Close the generalization gap (orthogonal, do on the final)
Internal fold-1 RMSE_V 3.56 → platform 3.81 = region/year domain shift on height.
- **All-data final.** `11_fresh_f1` trained on 4/5 folds; the final winner trains on all 5 (`--cv-fold -1`).
- **Light embedding domain-aug** (P4 from the old roadmap, minus calibration): channel-wise gain/offset
  + noise on the 64+128 pixel embeddings → decoder robust to the test domain. Reproducible; may lift the
  *platform* even when internal doesn't move.

## 4. Secondary levers (after P0 lands)
- **P2 IoU_build 0.43→0.5+:** the substrate changed under the fresh arch, so re-test building-objective
  sharpness (Tversky bias, `building_overlap_weight`, sharper `dice_k`) — the "objective exhausted"
  verdict (f33) was on the *old* arch.
- **P3 IoU_water 0.48:** rare-class; needs bs≥32 + simple stem (f24). Lower priority.

## 5. Method & guardrails
- **Screen H1 on multi-fold, judging RMSE_V** — NOT the fast-split, NOT the blended proxy (both misled
  us on vegboost). One geo-fold + the held-out fold, ~30–40 ep.
- Additive flags only; seeded; smoke each change (finite loss, shapes, GPU mem) → STOP & report before a campaign.
- **No calibration / no TTA** (TTA smears height — it *raises* RMSE_V; calibration won't transfer).
  Single-model preferred; end-stage multi-seed ensemble allowed only as final polish.
- `--scratch-dir` (NVMe) for run outputs; run locally on head or a confirmed-up node (avoid ssh in the
  predict/submit path — it stalled `11_fresh_f1`'s first predict at 259/946).

## 6. Success criteria
- **Win** = RMSE_V drops vs `11_fresh_f1` (platform 3.81) by more than fold noise, **without** regressing
  IoU_B/IoU_V. RMSE_V 3.81→3.0 alone ≈ **+0.040** score (3.81→2.5 ≈ +0.065).
- If H1+H2+H3 cannot move RMSE_V below ~3.4, veg-canopy height is at the embeddings' information limit →
  consolidate at the best height-tuned single model + end-stage ensemble.
