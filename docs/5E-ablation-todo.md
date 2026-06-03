# Phase 5E — Architecture Ablations (candidate, NOT started)

**Status:** backlog / design notes. **Do not start until Phase 5D (THOR) results are in.**
**Branch:** TBD (e.g. `exp-7-phase5e-arch`). **Base config:** whatever 5D concludes is best
(`7A_simple_v2stem`, or `7A_simple_thor` if THOR proves additive). Freeze that as the reference.

These four came out of a design review of the current `DualEncDualDecFusion` (see
`docs/phase5d_thor.md` §3 for the architecture and schematic). They are **independent** — each
should be a separate, default-off flag so its effect is cleanly attributable, and so the model is
**byte-identical to the base when the flag is off** (the repo's iron rule: additive only, never
modify existing classes destructively).

Priority order (highest expected payoff first): **#4 → #1 → #3 → #2**.

---

## #4 — Route S1 (SAR) into the fraction branch  ⟵ highest priority

**Motivation.** S1 is SAR: double-bounce makes built-up areas bright, specular reflection makes
water unambiguously dark — i.e. SAR is a premier signal for exactly our two worst fraction metrics,
**IoU_B (priority #1) and IoU_W**. Yet today S1 is routed **only** to the height decoder, so the
building-fraction channel *and the auxiliary binary-building head never see SAR directly* — they
are built from optical (S2) + AlphaEarth alone. Likely a real miss for IoU_B.

**Change.** Make the sensor→branch routing configurable instead of hard `_s1→height / _s2→fraction`.
In `DualEncDualDecFusion.forward`, the fraction decoder would additionally receive the S1 token set
(concatenated with its S2 tokens along the token dim, exactly as multi-modal same-sensor tokens are
concatenated today). `FractionDecoder` already injects via `PatchCrossAttnBlock`, which already
handles arbitrary K/V token counts — so this is mostly a routing/plumbing change, not new modules.

- New flag, e.g. `--patch-routing {sensor,s1-both,all-both}` (default `sensor` = current behaviour).
  - `sensor`  : S1→height, S2→fraction (current, byte-identical).
  - `s1-both` : S1→height **and** fraction; S2→fraction. (Test this first — targets IoU_B/IoU_W.)
  - `all-both`: every stream into both decoders (most expressive, most memory).
- Keep `None`-skip behaviour when a side is empty.

**Watch:** IoU_B, IoU_W (primary), RMSE_B. **Risk:** more K/V at the fraction injection points →
more attention memory (see 5D memory notes; `--xattn-heads 2` is the lever). **Do NOT** do the full
swap (S1→fraction-only / S2→height-only) — that strips the strongest height cue (SAR structure) and
the strongest veg-fraction cue (optical); physically expected to regress.

---

## #1 — Symmetric `tessera→fraction` bridge

**Motivation.** Today the only cross-encoder link is one-way: alpha-bottleneck → height decoder
(via GradScale α=0.2). TESSERA (temporal / multispectral) plausibly helps **vegetation and water
fractions** (seasonality, spectral indices), but it is never offered to the fraction decoder.

**Change.** Add an optional mirror of the existing bridge: feed the **tessera** bottleneck into the
**fraction** decoder's first block, concatenated with the alpha bottleneck, fused by a 1×1 conv,
with its own `GradScale(α_frac)` so height-encoder protection is preserved (TESSERA is the height
encoder; we must not let fraction loss corrupt it).

- New flag `--use-fraction-bridge` (default off → byte-identical). Build the bridge module
  unconditionally (like the height bridge) so init/param order is stable; use it only when on.
- Reuse the `bridge_alpha` mechanism; consider a separate `--fraction-bridge-alpha` (default 0.2).

**Watch:** IoU_V, IoU_W. **Risk:** adding inputs to the priority (fraction/alpha) path is the thing
we most want to protect — keep the GradScale on the tessera side and start with a small α.

---

## #3 — Cross-attention at the 64×64 decoder level

**Motivation.** Patch tokens are injected only at the two coarsest decoder levels (16×16, 32×32);
64/128/256 rely on U-Net skips alone. A 64×64 injection gives finer-grained patch guidance.

**Change.** Add an `inject64` `PatchCrossAttnBlock` after `up2` in each decoder (`dim_q=_DEC[2]=192`
@ 64×64). `PatchCrossAttnBlock` already resamples each 16×16 grid to H×W per-grid, so no new logic —
just an extra block, gated by a flag.

- New flag `--patch-inject-levels 16,32` (default) → add `64` to enable.
- **Memory:** at 64×64, Q=4096 and K/V = G×4096; the attention score tensor is large. Measure peak
  before committing to batch 24/32; `--xattn-heads 2` likely required when combined with full THOR.

**Watch:** all metrics, but especially fine-structure (IoU_B edges). **Risk:** memory; possible
over-smoothing or marginal gain for the cost. Keep `before/after` byte-identical when level absent.

**Note:** this is the *defensible* version of "inject earlier/deeper." Do **not** move injection into
the encoder or down to 8×8 — encoder-side foreign-embedding mixing is what 6A showed destroys IoU_B,
and 8×8 would force downsampling the native 16×16 tokens (throwing away patch resolution).

---

## #2 — Learnable bridge gate (NOT a learnable GradScale)

**Motivation.** α=0.2 is a hand-tuned hyperparameter. A natural question is "can the network learn
the right amount of cross-encoder sharing?"

**Key subtlety.** The GradScale α **cannot** be learned by the task loss: `GradScale` is identity in
forward, so α never appears in the forward graph and `∂loss/∂α = 0`. A "learnable α" on that
mechanism is a no-op. What *is* learnable is a **forward gate**: `x = bridge(cat[tBN, g · aBN])`
with `g` a learned scalar (or per-channel vector, sigmoid-bounded). That learns *how much alpha to
mix in* — but it does **not** replace the gradient protection (full gradient still flows back into
the alpha encoder through `g`). So:

**Change.** Add `--bridge-gate {none,scalar,channel}` (default `none` = current GradScale-only).
When on, multiply the alpha side input by a learned `g` (init so it starts ≈ the current behaviour).
Decide explicitly whether to *keep* the GradScale (protection) alongside the gate (sharing control)
— recommended: keep both (gate controls forward mix, GradScale still protects the encoder).

**Watch:** whether learned `g` drifts (tells us if the hand-picked 0.2 was over/under-sharing).
**Risk:** lowest payoff of the four; mostly diagnostic. **Do this last.**

---

## Cross-cutting rules (all four)
- One default-off flag per ablation; **off ⇒ byte-identical to the 5E base** (verify with a fixed-seed
  forward, max_abs_diff = 0, same as the 5D `_thor_smoke/verify.py` pattern).
- Never modify the existing `FractionDecoder`/`HeightDecoder`/`DualEncDualDecFusion` behaviour for the
  default path; add branches, don't replace.
- Write every new flag to `training_params.txt` and confirm `predict.py` reconstructs it.
- Test each ablation **alone** first (clean attribution), then optionally the best combination.
- Smoke (batch 4, 2 batches, cache) → report GPU mem + finite loss → **stop for review before training.**
- Geographic CV fold 0, static weights `[0.65,0.64,1.70]`, no GradNorm, no vegboost (unchanged).
- Do not submit to the platform without explicit approval.

---

## Convenient prompt (paste to a coding agent when 5E is greenlit)

> # Coding Task: Phase 5E — Architecture Ablations on the 7A dual-enc/dual-dec model
>
> ## Context
> Read `CLAUDE.md`, `docs/results.md`, `docs/science.md`, and `docs/phase5d_thor.md` (§3 for the
> current architecture/schematic) before starting. Continuation of `exp-7-clean-slate`. Phase 5D
> (THOR + PatchTokenStemV2) is complete; **freeze its winning config as the base** — confirm from
> the 5D results table which of `7A_simple_v2stem` / `7A_simple_thor` is the reference, and match
> its `--patch-inputs` / `--patch-stem-version` exactly in every 5E run.
>
> Create a new branch `exp-7-phase5e-arch` and commit this spec as
> `prompts/exp-7-phase5e-arch.md` as the first commit.
>
> ## Scope — four INDEPENDENT, additive, default-off ablations
> Implement each behind its own flag; with the flag off the model must be **byte-identical** to the
> 5E base (prove it: fixed-seed forward, `max_abs_diff == 0`, like `_thor_smoke/verify.py`). Never
> modify the default behaviour of existing model classes — add branches, don't replace.
>
> 1. **`--patch-routing {sensor,s1-both,all-both}`** (default `sensor`). Route S1 (SAR) tokens into
>    the **fraction** decoder in addition to height (`s1-both`); `all-both` = every stream into both.
>    Reuse the existing `PatchCrossAttnBlock` token-dim concatenation; no new modules expected.
>    *Goal: does SAR in the fraction branch raise IoU_B / IoU_W (our worst metrics)?*
> 2. **`--use-fraction-bridge`** (+ `--fraction-bridge-alpha`, default 0.2; default off). Mirror of
>    the height bridge: tessera bottleneck → fraction decoder first block, 1×1-conv fused, with
>    GradScale on the tessera side. Build the module unconditionally; use only when on.
>    *Goal: does TESSERA help vegetation/water fractions?*
> 3. **`--patch-inject-levels 16,32`** (default; add `64` to enable). Extra `PatchCrossAttnBlock`
>    after `up2` (dim_q=192 @ 64×64) in each decoder. **Measure peak GPU memory** — likely needs
>    `--xattn-heads 2` with full THOR. Do NOT move injection into the encoder or to 8×8.
>    *Goal: does finer-grained patch guidance help fine structure?*
> 4. **`--bridge-gate {none,scalar,channel}`** (default `none`). Learned forward gate `g` on the
>    alpha side input of the height bridge (init ≈ current behaviour). Keep the existing GradScale.
>    *Note: a learnable GradScale-α is a no-op (α is forward-invisible); this is a forward gate.*
>    *Goal: diagnostic — does learned sharing differ from the hand-picked 0.2?*
>
> ## Requirements
> - Each new flag is written to `training_params.txt` and round-trips through `predict.py`.
> - Use the cache at `/home/bassam/nvme_cache/cache7a` (THOR-inclusive; rebuild per node if needed).
> - Static weights `0.65,0.64,1.70`, `--no-use-gradnorm`, `--veg-height-boost 0.0`, `--cv-fold 0`,
>   `--epochs 60`, `--seed 0`, batch 24 (full-THOR is memory-tight at 32 — see `docs/phase5d_thor.md` §6).
>   NOTE: disable GradNorm with `--no-use-gradnorm` (the token `--use-gradnorm False` is rejected by argparse).
>
> ## Testing
> - Per ablation: byte-identical-when-off check (max_abs_diff = 0), param count, finite-loss smoke
>   (batch 4, `--max-batches 2`), peak GPU memory (especially #3 and #4 with full THOR).
> - **Stop at the smoke checkpoint and report. Wait for review before launching training.**
>
> ## Training (after approval)
> One ablation per node, 60 epochs, vs the frozen 5E base:
> ```
> n1  base (reference re-run)
> n2  --patch-routing s1-both
> n3  --use-fraction-bridge
> n4  --patch-inject-levels 16,32,64   (watch memory; --xattn-heads 2 if needed)
> ```
> (#2 bridge-gate can take a later slot or share a node — lowest priority.)
>
> ## Report / questions
> Comparison table vs the 5E base. Answer: (1) does S1→fraction raise IoU_B/IoU_W? (2) does the
> tessera→fraction bridge raise IoU_V/IoU_W? (3) does 64² injection help, and at what memory cost?
> (4) does the learned bridge gate drift from 0.2?
>
> ## Things to NOT do
> - No platform submission without approval. No vegboost/GradNorm. No destructive edits to existing
>   model classes. No encoder-side or 8×8 patch injection. Report any decision point with multiple
>   reasonable paths instead of picking silently.
