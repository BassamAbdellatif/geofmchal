# Phase 11 — Fresh Extraction Architecture (Design Doc)

Branch: `exp-11-fresh-arch` (from `exp-9-objective` HEAD). Status: **design, not yet built.**
Read `docs/results.md` f29–f34 and `docs/roadmap_final21d.md` status first.

## 1. Why a fresh architecture (the evidence)

Current best = `9_sm4tv_final_hybrid` = **0.3871, rank 54**. Two deficits vs the field (top-10 ≈ 0.478):
- **IoU_B 0.35 vs ~0.49** (−0.034 score)
- **Height: RMSE_B 2.28 / RMSE_V 3.70 vs ~1.85 / ~3.0** (−0.044 score, **our largest, most under-invested gap**)

Scoring ≈ our C=4 proxy (`0.25·IoU_B + 0.15·IoU_V + 0.15·IoU_W + 0.25·(1−RMSE_B/4) + 0.20·(1−RMSE_V/4)`),
**no hard cliff** — height is lost continuously. Top teams beat us on *every* metric from the *same*
embeddings → this is an **extraction/capacity** gap, confirmed by:
- f33: incremental tuning (objective/gradient/bridge/sampling) is **exhausted**.
- f34: the decoder is **already full-resolution** → "more resolution" is *not* the lever.

### Two legacy mistakes we are correcting
1. **The network is deliberately *light*** (`UNetEncoderHalf`, `EfficientDecoder256Fast`) — built for an
   **M2 Max's memory**. We now have 4× 48 GB GPUs barely used. We've been under-modeling rich data.
2. **Height is starved of AlphaEarth.** Per the science (below), AlphaEarth's inputs include **GEDI
   LiDAR + GLO-30 DEM + S1/PALSAR radar** — it *encodes elevation*. Yet 7A routes alpha mainly to
   *fractions* and feeds *height* only a GradScale-0.2, 16×16 **side-input**. The Phase-10 α-sweep was
   null because the problem isn't gradient magnitude — it's **architectural**: alpha enters the
   (tessera-primary) height head once, coarsely, throttled.

## 2. The data (facts, 2026-06-10)
All 10 m ground resolution; a 256×256 tile = 2.56 km (a building ≈ 1–3 px; targets are at the same
10 m, so we are **not** resolution-starved — we under-extract).

| input | shape | content |
|-------|-------|---------|
| `alpha_earth` | 64 × **256×256** | 10 m multi-sensor: S2/Landsat optical, **S1+PALSAR radar, GEDI LiDAR, GLO-30 DEM**, climate → composition **and height** |
| `tessera` | 128 × **256×256** | 10 m, a year of S1+S2 time-series → structure + phenology |
| `terramind_s1/s2` | 768 × **16×16** | ViT patch tokens, each = one 16×16-px patch (coarse) |
| target | 4 × 256×256 | building/veg/water fractions + normalised height (÷30) |

## 3. Goals
- **Single, reproducible model — DECISION LOCKED (2026-06-10):** the fresh model produces all 4
  channels itself, including height. **No 2A height graft / no 2-model hybrid.** (Ensembling the final
  winner across seeds is still allowed as an end-stage, orthogonal step.) Make the model's *own* height
  good via Bet 1 (AlphaEarth→height).
- Raise **IoU_B** and lower **RMSE_B/V**, without regressing IoU_W/V.
- **Keep the Phase-9 objective win** (`softmax4` fraction head + Tversky 0.3/0.7 = "sm4tv").

## 4. The three bets → one architecture

### Bet 1 — AlphaEarth first-class for height (fix the starvation)
**Symmetric, multi-scale cross-encoder fusion.** Both decoders consume *both* pixel encoders'
features at **every** level (256/128/64/32/16), not asymmetric throttled bridges. The height decoder
gets alpha's full multi-scale features (the LiDAR/DEM signal) as a **co-equal** input. **Drop the
GradScale throttle** (α-sweep proved it's not the lever); if task imbalance appears, fix it with loss
weights, not architectural starvation.

### Bet 2 — Capacity (drop the light net)
Deeper + wider encoders over the full-res 192-ch (64+128) pixel embeddings: ≥2 conv blocks/stage,
larger channel widths, a real decoder. Use `--amp` + gradient checkpointing if memory needs it. We
finally have the GPU budget to *fit the data*, not a toy.

### Bet 3 — TerraMind decoded right, or demoted
- **Default:** a proper **dense decode** of the 16×16 tokens (DPT/UPerNet-style — reassemble + progressive
  upsample 16→256, fused at multiple scales), instead of the current 2-level cross-attention injection.
- **Ablation:** **drop TerraMind** (pixel-only). The pixel embeddings are full-res *and* richer; the
  coarse tokens may not earn their place. Test it explicitly — we may have over-invested in patches.

## 5. Concrete v1 to build (`--model-type fresh_extract`)
```
alpha (64@256) ─stemA→ W@256 ┐
tessera(128@256)─stemT→ W@256 ┤  two DEEP U-Net encoders (5 levels, wider, ≥2 blocks/stage)
                              │  skips at 256/128/64/32/16
        ┌─────────────────────┴─────────────────────┐
        │  per-level SYMMETRIC fusion (concat→1×1 or  │   ← Bet 1
        │  light cross-attn) → shared feature pyramid │
        └──────┬───────────────────────────┬─────────┘
 terramind 16×16 ──token dense-decode (Bet3)│ (fused at coarse levels; toggle-able)
        │                                   │
   FRACTION decoder (softmax4 head)    HEIGHT decoder (regression)
   ← both read the SHARED pyramid, so HEIGHT sees alpha fully (Bet 1)
```
- Stems per-modality (avoid 6A's raw-concat collapse). W ≈ 128.
- Objective: keep **sm4tv** (softmax4 + Tversky 0.3/0.7) for fractions; height = Huber (revisit a
  scale-aware height loss later if RMSE stalls).
- New `--model-type` → lives **alongside** `dual_enc_dec_fusion` (don't break the working 7A path).

## 6. Ablation plan (on the fast split)
| arm | tests | isolates |
|-----|-------|----------|
| A0  | deep dual-encoder, current asymmetric bridges | Bet 2 (capacity) alone |
| A1  | A0 + symmetric multi-scale fusion | Bet 1 (alpha→height) |
| A2  | A1 + TerraMind DPT decode | Bet 3a |
| A3  | A1 − TerraMind (pixel-only) | Bet 3b (do patches earn their place?) |

Read **RMSE_B/V** (height) and **IoU_B/W**. Winner → multi-fold confirm (folds 0/1/2) → all-data final.

## 7. Faster iteration (the "20→40 days" lever)
Add `--train-folds` / `--val-folds`. **Screen on `--train-folds 0 --val-folds 1`** (514 train / 396
val tiles) → ~3× fewer tiles + fewer screening epochs ≈ **~5× faster dev loop**. Caveat: 1-fold-train
is a *screening proxy* (hard→easy region) — good for *ranking* ideas, not the final number; confirm
the winner on multi-fold + all-data.

## 8. Success criteria / fallback
- **Win** = beats `sm4tv` on the fast split's RMSE_V (and/or IoU_B) by more than noise, consistently.
- If capacity + fusion *don't* move RMSE_V/IoU_B → the ceiling is the embeddings' information content,
  and we **consolidate** at 0.3871 + a multi-seed ensemble.

## 9. Constraints
- Additive: new `--model-type`, do not break `dual_enc_dec_fusion`. Seeded.
- No calibration / no TTA (TTA hurts; calibration won't transfer). Single-model preferred (reproducibility).
- Smoke every change (finite loss, GPU mem, shapes) → **STOP and report before any campaign.**
- Be mindful of NFS: use `--scratch-dir` (NVMe) for run outputs; thermal caps on the GPUs as desired.
