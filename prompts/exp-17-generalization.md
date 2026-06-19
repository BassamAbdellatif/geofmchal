# Phase 17 — Generalization / transfer (EMA · weight-decay · ensemble) (Spec)

**Status:** in progress (EMA implemented). **Target:** the **binding constraint of the whole campaign —
the train→test transfer gap.** Internal proxy has never predicted the platform (f40 anti-correlated, f43
binning IoU_B platform-only, f45 DA negative, f46 pyramid internal-win→platform-loss). The architecture
lever is now exhausted with proper controls (f47 data-source ablation, f48 FlexNet depth/decoder screen all
flat). So the remaining lever is **how we extract a *generalizing* solution from the same training**, not
*what* the model is. Read `CLAUDE.md`, `docs/results.md` f40/f43/f45/f46/f47/f48, `docs/science.md`.

## 0. What we already use (a real regularization stack — don't re-add)
D4 geometric aug (rot90+flip, train-only) · domain-shift embedding aug (scale 1) · stratified rare-class
sampling · weight decay 1e-4 (AdamW) · geographic CV (transfer-aware val) · channel standardization ·
best-epoch selection on geo-val · **bs16** (small-batch noise = implicit regularization, Phase-16).

## 1. What's MISSING — the levers, ranked by EV for *transfer*
1. **Weight averaging (EMA / SWA)** — the canonical flat-minimum → better-generalization tool; we had
   none. Directly targets our signature (internal proxy peaks at one sharp epoch, doesn't survive the
   domain shift). **Implemented:** `--ema` (`--ema-decay`, default 0.999) tracks an EMA of params+buffers,
   **evaluates + saves the EMA model** as `model_best` (raw weights restored for continued training).
   GroupNorm ⇒ no BN-stat recompute (clean SWA equivalent). Model-agnostic (state_dict ops) ⇒ works on
   `fresh_extract`/`flexnet` alike. `predict.py` unchanged (EMA checkpoint loads as normal weights).
2. **Weight-decay sweep** — `--weight-decay` now a flag (was a constant); 1e-4 is modest, try 3e-4/1e-3.
3. **Multi-seed / multi-fold ENSEMBLE** of the best recipe — the most *reliable* generalizer (averaging
   independent models); ~+0.005–0.01. The end-game once a single-model recipe is fixed.
4. *(lower)* dropout at the decoder/bottleneck (none currently); MixStyle / clean-vs-aug consistency (DA lit).

## 2. Experiment plan
1. **[primary] `ce40` + EMA** (`17_ce40_ema_f1`, fresh_extract recipe, cv-fold 1, bs24) — EMA on the
   standing-best recipe, same held-out fold as ce40. Read: internal fold-1 proxy vs ce40, **and the
   platform** (EMA targets transfer, so the platform is the real test — compare to ce40 = 0.4413).
2. **[optional] U-Net++ + EMA** (`17_unetpp_ema_f1`, flexnet unetpp `--grad-checkpoint`, cv-fold 1, bs16)
   — stacks the one consistent FlexNet signal (U-Net++ IoU_B tilt, f48) with EMA. Only if chasing IoU_B.
3. **[if EMA helps] weight-decay sweep** — cheap fast-fold (train 0 / val 1, finc0/finc1 cached): wd ∈
   {1e-4, 3e-4, 1e-3} on the winning recipe.
4. **[end-game] ensemble** — multi-seed EMA models of the best recipe (no-holdout for final members),
   average predictions, single platform submission.

## 3. Method
- EMA targets *transfer*, so **judge on the platform** (internal may stay flat — like binning's IoU_B).
  A held-out-fold proxy lift is a positive sign but not required. Multi-fold for any single-model claim.
- One change at a time; `--ema` default-off so all prior runs reproduce. Seeded; val never augmented;
  no test-label peeking. Guardian + failsafe2 up; bs16 standard for flexnet, bs24 for the ce40 recipe.

## 4. Success / fallback
- **Win:** EMA (and/or higher WD) lifts the platform vs ce40 → lock the recipe, build a **multi-seed EMA
  ensemble** for the final submission (best-honest-rank push). Expect the gain to be platform-side.
- **Null (EMA flat on platform):** the residual gap is embedding-information-limited, not a
  trainability/flat-minimum problem → consolidate at `ce40` best-honest-rank and bank the reliable
  multi-seed ensemble of ce40 as the final. Deadline 2026-06-30.
