# Phase 16 — FlexNet: configurable-depth, multi-modal, ablatable encoder/decoder (Spec)

**Status:** design, ready to build. **Target:** the **architecture/extraction gap to the 0.5+ leaders**.
The top-5 reach 0.51–0.58 from the *same provided embeddings*; our 0.4413 trails on IoU_B (+0.045 of the
gap to #1) and RMSE_V (+0.038) — together ~60%. So the ceiling is **method/capacity, not data** (f47 only
killed *more data sources*). Read `CLAUDE.md`, `docs/science.md`, `docs/results.md` f32/f34/f35/f44/f46/f47.

## 0. Confirmed facts this builds on (do NOT re-litigate)
- **Capacity/depth IS the lever (f35).** The `fresh_extract` swing lifted platform IoU_B 0.35→0.43 and
  falsified the "intrinsic ceiling"; f34 named the gap to leaders as *"a much better/bigger model."*
- **Output resolution is NOT the lever (f34).** Decoder already emits full-res via per-level skips (incl.
  the 256-res L0). This phase is about *richer feature extraction across scales*, not output resolution.
- **softmax4 + tversky confirmed (f32)** for IoU_B + transfer → KEEP; do NOT revert to sigmoid3 (tested, worse).
- **Binned height (f41–44)** is the only thing that ever moved RMSE_V → KEEP (`--height-bins 256`, soft-exp).
- **Patch FMs add no transferable value via add/pyramid (f47); alpha+tessera carry the signal** →
  patches default OFF, kept *selectable/ablatable* (per the full-input-flexibility requirement); spend
  capacity on the pixel pathway.
- **Leaderboard:** top-5 ≥0.51 IoU_B, ≤3.14 RMSE_V (same embeddings). Biggest gaps: **IoU_B, RMSE_V.**

## 1. The depth/downsampling instinct — validated + made ablatable
"Deeper branches / gentler downsampling so it learns features before losing resolution" — **goal correct**
(f35), realized the standard way (stride-2 halving, not 3-px steps). FlexNet exposes the real knobs so the
hypothesis is directly testable: **blocks-per-stage**, **bottleneck resolution**, **dilation** (grow RF
without downsampling), **block type** (residual/dense), **decoder topology** (U-Net++).

## 2. New class `FlexNet` (model-type `flexnet`) — additive
Keep `FreshExtract` / `FreshExtractFlex` frozen. FlexNet generalizes the dual-encoder → symmetric-fuse →
dual-decoder skeleton with everything configurable. **Default config ≈ `fresh_extract_flex`** (dual enc,
1 block/stage, →16 bottleneck, plain U-Net, softmax4, bins256) so each ablation moves ONE axis from a known
baseline. Reuse `_ModalityStem`, `_make_patch_stem`, `_TokenPyramid`; new code = configurable enc/dec.

### Inputs (flexible: `--pixel-inputs`, `--patch-inputs`)
- Per-modality stem (each its own — 6A lesson), stem depth `--stem-blocks` (default 2). alpha(64)/tessera(128)
  pixel; terramind/thor s1/s2 patch tokens (768→width), routed `*_s1→height`, `*_s2→fraction`.
- Any subset selectable; patches default empty (none).

### Encoder (the core knobs)
| flag | meaning | default |
|------|---------|---------|
| `--enc-widths "96,160,256,384,512"` | channels per stage (len = #stages) | 5 stages |
| `--enc-blocks "1,1,1,1,1"` (or int) | **conv blocks per stage = "deeper branches"** | 1 |
| `--enc-block {double,residual,dense}` | block type (deeper feature learning / grad flow) | double |
| `--bottleneck-res {16,32,64}` | **how far to downsample** (gentler = 32/64) | 16 |
| `--enc-dilations "1,1,1,1,1"` | dilated convs at deep stages: grow RF, keep resolution | off |

### Fusion
- Symmetric per-level 1×1 fuse of the K pixel pyramids. `--patch-fusion {none,add,pyramid}` (default none).

### Decoder (topology ablatable)
- `--decoder {unet, unetpp}` — plain vs **U-Net++ nested dense skips** (arXiv 2602.17250: U-Net++ more
  *transferable* for height from these embeddings, R² 0.84 vs 0.78). `--dec-blocks` per up-stage.

### Heads
- Fraction: `--fraction-head softmax4` (keep) + aux binary-building head.
- Height: binned (`--height-bins 256`) + `--height-mode {shared, class_cond}` — `class_cond` conditions the
  height head on the fraction logits (or predicts building/veg height fields composited by argmax class), so
  sharp-building vs smooth-canopy regimes stop sharing one field (attacks RMSE_V, the 28% lever).

## 3. Ablation plan — ONE axis at a time, MULTI-FOLD (capacity overfits → must transfer)
Held-out folds {0,2}, anchored to the FlexNet default baseline; judge by **mean held-out + train→held-out gap**:
1. **Depth** — `--enc-blocks` 1 vs 2 vs 3 (the instinct).
2. **Bottleneck res** — 16 vs 32 (gentler downsampling).
3. **Block type** — double vs residual.
4. **Decoder** — unet vs unetpp.
5. **Dilation** — off vs `"1,1,1,2,4"`.
6. **Height** — shared vs class_cond.
Then **combine the top 2–3 transferring axes**, re-confirm multi-fold, and spend ONE platform slot on the winner.

## 4. Smoke (STOP & report before any campaign)
1. Each config builds; default-config param count ≈ flex; forward shapes (frac B×4, height B×256, binary B×1)
   at 256²; finite grads; `predict()` 4-ch collapse.
2. `--bottleneck-res 32/64`, `--enc-blocks 3`, `--enc-block residual`, `--decoder unetpp`, `--enc-dilations`,
   `--height-mode class_cond` each build + run a 2-batch step with finite loss.
3. Deeper / higher-res / unetpp variants are heavier → record the largest bs that fits 48 GB per config;
   keep bs constant *within* each ablation axis for comparability (note any reduction).

## 5. Constraints
- Additive; `FreshExtract`/`FreshExtractFlex` untouched. Seeded aug; val never augmented; no test-label peeking.
- ce40 loss recipe: `--fraction-head softmax4 --building-overlap tversky --decouple-height
  --build-height-weight 1.0 --veg-height-weight 3.0 --height-bins 256 --height-ce-weight 0.40`,
  `--no-use-gradnorm --amp --num-workers 4 --cache-dir /home/bassam/nvme_cache/cache7a`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Guardian + failsafe2 up on each box first.

## 6. Success / fallback
- **Win:** a config raises mean held-out IoU_B and/or lowers RMSE_V with a non-worsening train→val gap →
  platform-confirm; expect to recover part of the 0.045 (IoU_B) / 0.038 (RMSE_V) gap. IoU_B is the more
  capacity-responsive target (f35 precedent); RMSE_V is the hardest (binning saturated 3.56, leaders 2.8) —
  class_cond height + unetpp + depth are the bets, genuinely uncertain.
- **Null (depth/topology flat across folds):** capacity is saturated at our extraction; then the honest
  end-game is the multi-seed/fold ensemble of ce40. But the leaderboard says headroom exists, so screen first.
