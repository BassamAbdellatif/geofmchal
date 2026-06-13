# Phase 13a — Adaptive-Bin Height Head (Spec)

**Status:** design, ready to build. **Target:** RMSE_veg — **41% of the gap to #1** (us 3.740 m, #1 2.782 m;
GEDI footprint-level RMSE floor ≈ 2.7 m, so #1 is *at* the data limit and our ~1 m gap is *method*, not data).
Read `results.md` f37–f39 and `CLAUDE.md` first.

## 1. Why (the evidence)
Our height head is a single `Conv2d(c0,1)` + Huber — direct scalar regression, which regresses toward the
**mean** and blurs the heavy-tailed canopy distribution. The canopy-height SOTA reframes this as
**classification-into-bins + expectation** (DORN, AdaBins, SORD; "discrete-continuous" loss): predict a
probability over height bins, take the soft-expectation as the value. The classification signal gives
sharper gradients and consistently **beats direct regression on RMSE** for depth/height. This is a
**head-only** change — keep the working `fresh_extract` encoder/decoder and the decoupled per-class
weighting (f37); swap only the height head + its loss.

## 2. The head
Replace the 1-channel height head with an **N-bin classifier** over normalized height (target = m/30 ∈ [0,1.5]):
```python
N_BINS = 128                                   # 1.5/128 ≈ 0.0117 norm ≈ 0.35 m/bin
self.height_head = nn.Conv2d(c[0], N_BINS, 1)  # per-pixel bin logits
# fixed uniform bin centers (normalized), registered as a buffer:
centers = (torch.arange(N_BINS) + 0.5) / N_BINS * 1.5     # (N,) in [0,1.5]
self.register_buffer("height_centers", centers)
```
- `forward` returns `height` = **(B, N_BINS, H, W) logits** (for the loss).
- **Soft-expectation** (the value, used by predict + the train proxy eval):
  ```python
  def expected_height(self, logits):           # logits (B,N,H,W) -> (B,1,H,W) normalized
      p = torch.softmax(logits, dim=1)
      return (p * self.height_centers.view(1,-1,1,1)).sum(1, keepdim=True)
  ```
- **Bin spacing:** start **uniform**; canopy is low-skewed, so a **sqrt/quantile spacing** (finer bins at low
  height) is the first refinement if uniform underperforms. (Adaptive per-image bins = AdaBins-style
  transformer predictor — the v2 refinement, deferred; keep v1 a pure conv head.)

## 3. The discrete-continuous loss (slots into decoupled per-class weighting)
Per-pixel height loss becomes **CE (discrete) + Huber-on-expectation (continuous)**, then the **existing
decoupled build/veg/bg masking + independent weights (f37) wrap it unchanged**:
```python
# logits_h: (B,N,H,W); h_tgt: (B,H,W) normalized in [0,1.5]
p     = softmax(logits_h, 1)
E_h   = (p * centers).sum(1)                                  # (B,H,W) expected norm height
gt_b  = (h_tgt / 1.5 * N).long().clamp(0, N-1)                # target bin index
ce    = F.cross_entropy(logits_h, gt_b, reduction="none")     # (B,H,W) discrete term
reg   = F.huber_loss(E_h, h_tgt, delta=huber_delta, reduction="none")  # continuous term
per_pix = ce_weight * ce + reg_weight * reg                   # combined per-pixel
# --- decoupled masking, IDENTICAL structure to f37 ---
loss_height = build_height_weight * (per_pix * bmask).sum()/b_sum \
            + veg_height_weight   * (per_pix * vmask).sum()/v_sum \
            + height_bg_weight    * (per_pix * bgm ).sum()/bg_sum
```
- **CE↔reg balance:** CE ≈ ln(N) ≈ 4.85 at init, Huber ≈ O(0.01) on normalized height — very different
  scales. Start `ce_weight=0.1, reg_weight=1.0` and **check in smoke that neither term dominates**; tune.
- **Refinement (cheap, optional):** replace hard-bin CE with **SORD** (soft ordinal labels: target =
  softmax(−|center − h_tgt|/τ)) so far-bin errors are penalized more — respects height ordering within CE.
- **decoupled weighting unchanged** → we keep the wb=2/wv≈3 win; only the per-pixel term sharpens.

## 4. Code touch-points
| file | change |
|------|--------|
| `core/model.py` | `FreshExtract(height_bins=0)`: if >0, `height_head=Conv2d(c0,N)`, register `height_centers`, add `expected_height()`. `forward` returns N-ch height logits when on. `predict()` collapses via `expected_height` → 1-ch ×30. `build_model` passes `height_bins`. |
| `core/losses.py` | `DualPathLoss(height_bins=0, height_ce_weight=0.1)`: in the **decoupled** branch, if bins on, per-pixel = ce_weight·CE + Huber(E_h). Needs `centers` (pass in or recompute). |
| `train.py` | flags `--height-bins`, `--height-ce-weight`; write to params. **Proxy eval fix:** `train_7a` builds `pred=cat([frac, height])` for the RMSE proxy — when bins on, convert the N-ch height to `expected_height` (×30 handled as today) before the proxy/RMSE computation. |
| `predict.py` | read `HEIGHT_BINS`/`HEIGHT_CE_WEIGHT` from params, build model with them; height output already collapsed by `predict()`/`expected_height`. Submission stays 4-ch (B/V/W + height). |
- **Byte-identical when `--height-bins 0`** (default): 1-ch head + Huber, exactly today's path.

## 5. Screening plan
1. **Smoke:** shapes (height N-ch logits; expectation 1-ch), finite loss, CE/reg balance sane, bs32 memory.
2. **Fast screen** (train fold0/val fold1, 30 ep) vs the pixel-only/decoupled baseline — judge **RMSE_V**.
   Bin/weight knobs to try: `N∈{64,128,256}`, spacing {uniform, sqrt}, `ce_weight∈{0.05,0.1,0.3}`.
3. **Confirm winner** on **cv-fold 1, 40 ep** vs `12_dech_f1` (0.4599 / RMSE_V 3.500) with decoupled wb2/wv3.
   Keep the f37 decoupled weights (don't re-confound).
4. Judge on **RMSE_V** primarily; guard IoU/RMSE_B don't regress. Internal→platform amplified ~3.5× on
   height before, so a real internal RMSE_V drop is high-value.

## 6. Success / fallback
- **Win** = RMSE_V drops vs `12_dech_f1` (3.500) beyond fold noise, no IoU regression → confirm + submit.
- **Null** (fixed bins ≈ regression) → try **sqrt/quantile spacing** then **SORD**; if still flat → **AdaBins
  transformer bin-predictor** (v2, heavier) — the per-image adaptive bins are the part most credited for
  AdaBins' RMSE gains.
- If even adaptive bins don't move RMSE_V → height is at *our* embedding's information limit (not #1's), and
  the remaining gap is encoder-side (Phase 13b: transformer decode for IoU_B) or data we don't have.

## 7. Constraints
- Additive; `--height-bins 0` byte-identical; seeded; reuse `fresh_extract` encoder + f37 decoupled weights.
- No calibration/TTA. Smoke before any campaign → STOP & report. `--scratch-dir` (NVMe); power-capped + coolgpus.
