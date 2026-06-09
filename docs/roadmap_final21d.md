# Final-21-Days Roadmap — Close the IoU_B Gap (2026-06-08)

**Deadline:** 2026-06-30 (~21 days). **Standing:** rank **53**. **Leaderboard best (ours):**
`2A_vegboost` = **0.3721** (composite). **Top team:** IoU_B **0.5269** (gap on IoU_B component:
our platform 0.3394 → 0.5269, i.e. **+0.19**). **Internal best:** `7A_v1_base` proxy **0.40**,
IoU_B 0.18 (seeded, f28). **Goal:** top-3 — see "Honest read" at the end.

Sources synthesised: `docs/Literature_review.md` (Alts 1–4), `docs/results.md` (findings 1–28),
this session's Phase-8 ablations, and the "no *others* class" observation.

## Three framing facts that reorder the literature review's own picks
1. **Our ablation overrides the review on gradient balancing.** The review's headline is Alt 1 +
   **Alt 4 (Rotograd)**. But **f26** showed multi-task interference is *not* the building
   bottleneck (full task isolation didn't move IoU_B). So Rotograd/PCGrad → low expected value.
2. **The review's other big IoU_B lever — threshold calibration (Alt 3) — is excluded** for
   reproducibility (overfits fold-0 val; won't transfer to the different-region/year test set).
   With both of the review's top IoU_B claims off the table, its third idea —
   **IoU-aligned losses (Alt 2)** — moves to the front, and **f28** independently named "objective
   formulation" as the prime suspect for the ceiling. Review and experiment point at the same door.
3. **We optimise an internal proxy that is not yet anchored to the platform.** The whole
   7A/Phase-5D/Phase-8 line is internal-only; the live leaderboard entry is still `2A_vegboost`.
   Note platform IoU_B (0.34) > internal IoU_B (0.18), so the mapping is non-trivial. One
   submission de-risks every downstream decision.

## Ranked plan (most → least significant for IoU_B / rank)

### P0 — Anchor to the platform (do first, ~0 compute)
Submit **`7A_v1_base` → `model_best.pth`** (complete `training_params.txt`; proxy-best checkpoint
mirrors the platform composite). Tells us (a) does 7A beat 0.3721, (b) the internal→platform
transfer factor, so we optimise the right number. *Highest-value action; needs a submission slot.*

### P1 — Building-aware overlap loss (lead experimental bet)
*(Alt 2; refs Calibrated-Dice [14], Auto-Seg-Loss [16], Soft-IoU [15]).* The fraction loss already
has a generic soft-Dice at the 0.5 boundary (`losses.py`, `dice_k=5`), applied **class-blind**
across the 3 channels. Buildings are rare and **under-predicted** → make the overlap term
**building-aware**: Tversky / focal-Tversky biased toward recall (FN penalty > FP), class-weighted,
optionally sharper `k`. This is the most likely lever to break the 0.18 ceiling; cheap (loss
change), byte-identical-when-off, fully reproducible (training-time, not inference calibration).
**Detailed spec:** `prompts/exp-9-objective.md`.

### P2 — Add the "others"/background class + softmax simplex (pairs with P1)
*(The "no-others" observation; complements Alt 2.)* Today the 3 fraction channels are **independent
sigmoids**; non-target pixels (bare soil, roads — what's confused with buildings) have no sink, so
their mass leaks into the goal classes and blurs the building 0.5 boundary. Derive an `other`
channel (`1−build−veg−water`) and switch the head to a 4-way **softmax simplex** so classes compete
and sum to 1 → sharper building precision. Changes the metric's building-extraction path, so it must
be byte-identical when off. **Run jointly with P1** as one objective-redesign campaign.

### P3 — TESSERA→fraction structural bridge + skip-level cross-attention
*(Alt 1 fusion.)* SAR/TESSERA carries building **structure**; today only AlphaEarth feeds the
fraction decoder. A downweighted TESSERA→fraction bridge (mirror of the alpha→height bridge; the
`--use-fraction-bridge` flag already exists) gives buildings a structural prior. Medium cost; run
**after** P1/P2 to avoid confounding the objective change.

### P4 — Embedding-space domain augmentation
*(Alt 3 minus calibration; refs DASAM [21], DAugNet [20], SS(DA)² [19].)* The real challenge is
train→test region/year shift. Channel-wise gain/offset + noise on embeddings (+ optional contrastive
consistency) makes the decoder domain-robust. May lift the **leaderboard** even when the internal
proxy doesn't move (value partly invisible internally — another reason P0 matters). Reproducible.

### P5 — Multi-seed / multi-fold ensemble (final-week polish)
*(Alt 3 ensemble.)* Train the winning recipe on 3 seeds / geo-folds, average. Reliable small,
reproducible gain. Freeze architecture first.

### Demoted / excluded (with reason)
- **Rotograd / PCGrad (Alt 4):** demoted by f26 (interference isn't the building problem).
- **Aggressive oversampling:** closed-negative, f27.
- **Threshold calibration / blend-binary (Alt 3):** excluded by the reproducibility rule.
- **Higher-resolution decoding / building hi-res branch:** the *other* f28 suspect
  (resolution/label quality). Higher cost — the **pivot** if P1+P2 fail to move IoU_B.

## 21-day sequence
1. **Now:** P0 submit + implement P1+P2 as one objective-redesign branch (`exp-9-objective`); smoke.
2. **Days 1–7:** P1+P2 campaign vs the seeded anchor — make-or-break on the 0.18 ceiling.
3. **Days 7–12:** branch on result — moved IoU_B → stack **P3** + **P4**; didn't → pivot to the
   resolution/feature hypothesis.
4. **Days 12–18:** integrate winners, tune, submit checkpoints to track real movement.
5. **Days 18–21:** P5 ensemble + final submission.

## Honest read on "top 3"
The gap is **+0.19 IoU_B** — not reachable by tuning; it requires the **P1+P2 objective redesign to
actually work** (the review's whole thesis is that leaders close this gap with IoU-aligned
objectives + domain robustness, not architecture). If P1+P2 lands even half the gap, top-10 is
plausible and top-3 is in play with P3/P4/P5 stacked. If it doesn't move, the ceiling is structural
(resolution/features) and top-3 is unlikely in 21 days — we'd optimise for best-honest-rank instead.
**P0 tells us early which world we're in.**

---

## STATUS UPDATE — 2026-06-10 (which world we're in)

We're in the harder world. Outcome of the plan above (detail: `results.md` f29–f34):
- **P0/anchor + P1+P2 done:** sm4tv (softmax4+tversky) **transferred** (+IoU_B across 3 geo-folds) →
  `9_sm4tv_final_hybrid` (sm4tv fractions + 2A height) = **0.3871, our best submission**.
- **But rank 54** — the field moved to 0.48–0.54; a +0.015 gain *lost* ground.
- **Incremental tuning is exhausted:** Phase 8 null, Phase 9 small, **Phase 10 (gradient/bridge) null**
  (f33). The 7A architecture is capped (IoU_B ~0.24 / RMSE_V ~4.0 internal).
- **Decode-resolution audit (f34): the decoder is already full-res** → the "raise resolution / light
  U-Net" lever is *not available*; the gap is input-embedding scale (~10 m) + extraction/capacity.
- **Scoring is ~linear, no hard cliff** (corrected): height (RMSE) is our biggest, most under-invested
  deficit, lost *continuously* vs the field's ~3.0 m.

**Fork (pending user decision):** (a) consolidate the 0.3871 best; or (b) a capacity/height swing
(heavier decoder / multi-seed ensemble / a dedicated better height model) — uncertain, top-3 out of
reach in the remaining days. P1 (gradient), P3a (sampling), P2/P3b: closed-negative. Calibration/TTA:
excluded (TTA hurts; calibration won't transfer).
