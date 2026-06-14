# Literature Review — Domain Adaptation for Frozen GeoFM Embeddings (Phase 14)

**Date:** 2026-06-14. **Context:** best = `13_bins256_ce40_b24_f1` = 0.4413 (rank 30). Architecture +
loss levers are near their internal-fit ceiling (capacity swing f35; adaptive-bin height f41–44, now
saturating). This review identifies the next step-change lever: **closing the train→test region/year gap.**

## 1. Why domain adaptation is the binding constraint (evidence)
Our internal validation has *systematically failed to predict the platform* — the signature of a
distribution-shift (domain-generalization) problem, not an underfitting one:
- **f40:** `wb2wv5` internal proxy +0.0027 but platform **−0.0138** (anti-correlated); IoU_B held on
  fold-1, dropped on test.
- **f43:** binning's IoU_B was *tied* internally (0.344) yet *moved* on the platform (−0.015 then +0.0076).
- The data are **French regions, different cities/years; the test set is held-out regions/years**
  (`geo-CV` over the 2-letter region grid). The decoder overfits *training-region* embedding statistics.

This is corroborated directly by the most on-point paper:
- **"Inferring Height from Earth Embeddings" (arXiv 2602.17250)** — height/DSM from the *same* 64-dim
  AlphaEarth pixel embeddings. Decoder = U-Net/U-Net++, direct regression. **Headline limitation:
  "performance decreased due to distribution shifts in height frequency between training and testing
  areas… need to address bias for improved regional transferability."** U-Net++ was *more transferable*
  than U-Net (R² 0.84 vs 0.78). They offer **no mitigation** — i.e., this is open and unexploited.
- **AlphaEarth best-practices** (DeepMind/community): "match pretraining and downstream sensor
  distributions, apply normalization, and use transfer learning / **domain adaptation** strategies."
  Also: AlphaEarth is **pixel-level → suited to dense prediction** (validates our pixel-first design and
  the patch-token deprioritization, §4).

**Conclusion:** the residual gap to the leaders is most plausibly *generalization*, and DA is the one
lever we have **never executed** (roadmap P4 was planned, never run) that targets it.

## 2. DA techniques applicable to our setting
Constraints: **frozen, pre-computed embeddings** (cannot fine-tune the backbone, cannot pretrain, cannot
peek at test labels). That rules out most encoder-side / pretraining DA and leaves *decoder-side* methods:

| technique | idea | refs | fit for us |
|-----------|------|------|------------|
| **Embedding-space augmentation** | per-channel gain/offset/noise/dropout on the embeddings during training → decoder robust to embedding drift | DAugNet, DASAM | **★ core** — half-built (`augment_domain_shift`), untuned |
| **Feature-statistics perturbation (MixStyle)** | mix per-sample channel mean/std across the batch to synthesize novel "domains" | MixStyle (Zhou 2021) | ★ cheap add-on, decoder-side |
| **Consistency regularization** | enforce prediction consistency between clean and domain-augmented embeddings | SS(DA)², FixMatch-style | ★ add-on, semi-supervised flavor |
| **Test-time normalization (AdaBN / instance-norm)** | normalize each test tile/region by its own stats at inference (no labels) | AdaBN | △ principled but input-shifting; ablate carefully (NOT threshold calibration — that's excluded) |
| **Height-frequency / bias correction** | the 2602.17250 shift is in *height distribution*; reweight height-strata in training; debias the head | — | △ harder, second wave |
| **Transferable architecture** | U-Net++ nested-dense skips were *more transferable* than U-Net | 2602.17250 | △ stretch, pairs with DA |

Excluded (reproducibility rule): **threshold calibration / blend-binary fit to a val fold** (won't
transfer; CLAUDE.md lesson). Per-instance *normalization* is different (no label peeking) and is allowed
as an ablated option.

## 3. The methodological key: MULTI-FOLD evaluation
Single-fold internal val *cannot see* DA's benefit (the whole point — the gap only shows on held-out
regions). So Phase-14 changes the **evaluation protocol**: train on K−1 geo-folds, hold out each fold in
turn, and report (a) **mean held-out score** and (b) the **train→held-out gap**. DA wins = the *gap
shrinks* and/or the *mean held-out* rises across folds — a far better platform predictor than one fold.

## 4. What we are NOT doing, and why (answers to the standing questions)
- **Patch/token fusion — deprioritized.** AlphaEarth is itself a *fused multimodal pixel embedding*, so
  the coarse TerraMind tokens are largely redundant → our net-null (f36/f38/f39) is expected. Literature
  multimodal wins (Fus-MAE, MM-OVSeg) require *pretraining-time* cross-attention or *explicit cross-modal
  alignment* — unavailable on frozen embeddings. Low EV.
- **Multi-task / gradient balancing — done / null.** Decoupled loss, dual decoders, IoU-aligned loss,
  binning all in; GradNorm/Rotograd null (f26). Low remaining value.
- **Just deeper/bigger — risky.** More capacity without DA *worsens* the transfer gap (overfits training
  regions). Only valuable paired with §2.

## 5. Recommended Phase-14 plan
1. **[core] Calibrate + scale embedding-space augmentation** (the half-built `augment_domain_shift`):
   expose a strength flag, calibrate magnitudes to the *actual* fold-to-fold (region) channel-stat
   variation, sweep strength under **multi-fold** eval.
2. **[add-on] MixStyle + consistency loss** if (1) shows signal.
3. **[stretch] U-Net++ skips** for transferability; **[2nd wave] height-bias correction.**
4. Validate by **multi-fold gap-reduction**, then confirm on the platform (the real test of DA — like
   binning, the gain may be platform-only).

Detailed experiment spec: **`prompts/exp-14-domain-adapt.md`**.

## 6. Sources
- Inferring Height from AlphaEarth — arXiv 2602.17250
- AlphaEarth embeddings overview / best-practices — emergentmind.com/topics/alphaearth-foundation-embeddings
- PEFT for Geospatial Foundation Models — arXiv 2504.17397
- CHMv2 (DINOv3 canopy height) — arXiv 2603.06382 ; Depth Any Canopy — arXiv 2408.04523
- Fus-MAE (SAR-optical cross-attention fusion) — arXiv 2401.02764 ; MM-OVSeg — arXiv 2603.17528
- SegFormer building extraction — PLOS One 10.1371/journal.pone.0338104 ; Buildformer global/local tradeoff — MDPI RS 13/13/2524
- MixStyle (Domain Generalization with MixStyle) — Zhou et al., ICLR 2021
