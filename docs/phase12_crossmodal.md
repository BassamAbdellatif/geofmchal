# Phase 12c — Coarse Alpha↔Tessera Bidirectional Cross-Attention (Spec)

**Status:** design, ready to wire. Sequenced *after* the height (vboost) gain is banked — this is an
**IoU_B / capacity** play (the #2 lever, 0.143 room), not a direct RMSE_V fix. Read `results.md` f36
(TerraMind cross-attn) and `docs/phase12_height.md` first.

## 1. Why
Today the two pixel encoders run **fully independently**; they only meet at a **static 1×1 concat**
(`fuse[l]`) at each level — a linear merge, no interactive exchange. The lit review's Alt-1 headline
("cross-attention fusion at 16×16/32×32 skips"; CMX / crossmodal-multiscale-fusion) is the richer fusion
we **skipped**. 6A showed alpha & tessera are genuinely **complementary** (naive concat *collapsed*
IoU_B 0.168→0.017) — so unlike the redundant TerraMind tokens (f36), interactive alpha↔tessera fusion
has a strong prior to actually add information.

## 2. Where it sits
Insert a bidirectional cross-modal exchange on the two encoder pyramids at the **coarse levels L3
(32×32) and L4 (16×16)**, **before** the existing `fuse`:
```
aP = alpha_enc(a0)            # [96@256,160@128,256@64,384@32,512@16]
tP = tessera_enc(t0)          # same shapes
# NEW — bidirectional cross-attention at coarse levels only:
for l in (3, 4):
    aP[l], tP[l] = cross_modal[l](aP[l], tP[l])
# unchanged — static merge into the shared pyramid:
pyr[l] = fuse[l](cat(aP[l], tP[l]))
```
Coarse-only because attention is quadratic: L4 = 256 tokens (256² trivial), L3 = 1024 tokens (~1M, cheap).
Fine levels (64–256 px) stay convolutional — they carry the fine detail the coarse exchange would smear.

## 3. Module — reuse `_TokenCrossAttn` (already in `core/model.py`)
Same-resolution cross-attention (alpha L4 16×16 ↔ tessera L4 16×16; q_ch == kv_ch per level), so the
existing block applies verbatim. Bidirectional = **two** instances per level:
```python
# in FreshExtract.__init__, when cross_modal == "coarse":
self.xmod_a3 = _TokenCrossAttn(c[3], c[3], heads=4, local_bias=cm_local)  # alpha←tessera @32
self.xmod_t3 = _TokenCrossAttn(c[3], c[3], heads=4, local_bias=cm_local)  # tessera←alpha @32
self.xmod_a4 = _TokenCrossAttn(c[4], c[4], heads=4, local_bias=cm_local)  # alpha←tessera @16
self.xmod_t4 = _TokenCrossAttn(c[4], c[4], heads=4, local_bias=cm_local)  # tessera←alpha @16
```
Forward (compute BOTH directions from the ORIGINAL features, then assign — no leakage):
```python
if self.cross_modal == "coarse":
    a3 = self.xmod_a3(aP[3], tP[3]); t3 = self.xmod_t3(tP[3], aP[3])
    a4 = self.xmod_a4(aP[4], tP[4]); t4 = self.xmod_t4(tP[4], aP[4])
    aP[3], tP[3], aP[4], tP[4] = a3, t3, a4, t4
```
Properties (inherited from `_TokenCrossAttn`):
- **Gated zero-init residual** → identity at start; can only add signal if it earns it (never degrades
  the working path — same safety that protected the TerraMind runs).
- 4 heads, scaled dot-product. Optional Gaussian **locality bias** via `cm_local` (default OFF at coarse:
  few tokens, large receptive fields, and the f36 locality bias was a wash — but kept as a toggle).

## 4. Plumbing (additive, off by default)
- `core/model.py`: `FreshExtract(..., cross_modal="off")`; `build_model(..., cross_modal="off")`.
- `train.py`: `--cross-modal {off,coarse}` (default off) + a `--cross-modal-local` toggle for `cm_local`;
  write `CROSS_MODAL` / `CROSS_MODAL_LOCAL` to `training_params.txt`.
- `predict.py`: read `CROSS_MODAL` (+ local) and pass to `build_model` (reconstruction parity).
- **Byte-identical when `off`** (no new modules built, forward path unchanged) — preserves all prior runs.

## 5. Cost
~3.2M params (4 blocks: 2×~1.0M @512 + 2×~0.6M @384) on the ~50M model. Negligible FLOPs (coarse grids).

## 6. Experiment plan (fast screen: train fold 0 / val fold 1, 30 ep, sm4tv, vboost=3, capped 100 W)
| arm | config | isolates |
|-----|--------|----------|
| **C0** | vboost3, cross-modal **off** | = our `12_vboost3` baseline (proxy 0.4156, IoU_B 0.3157) |
| **C1** | vboost3 + cross-modal **coarse** (bidir) | the cross-modal exchange |
| C2 (opt) | C1 + locality bias | does Tobler help *here* (vs the f36 null)? |
| C3 (opt) | C1 unidirectional (a←t only) | is one direction enough / which matters |

Run C1 vs C0 first. Judge **IoU_B** primarily (expected lever) + composite proxy; **guard RMSE_V/RMSE_B
do not regress**. Watch the learned gates: if they stay ≈0, the block found nothing (TerraMind-style null).

## 7. Success / fallback
- **Win** = IoU_B (and/or proxy) up vs C0 beyond fast-screen noise (~±0.005 proxy), **no** height regression
  → promote to multi-fold + full cv-fold-1 config (submittable).
- **Null** (gates ≈0 / ties C0) → coarse interaction adds nothing; fall back options, in order:
  1. **windowed cross-attention at a finer level** (L2 64×64, local windows) — if buildings need fine
     cross-modal detail the coarse exchange smears;
  2. **Mamba fusion** (MFMamba/CVMFusion-style, linear-cost global) — bigger rewrite, last resort.
- Keep blocks in code, off by default, regardless of outcome.

## 8. Open choices to confirm before wiring
- Levels: **L3+L4** (proposed) vs L4-only (cheaper, less reach).
- Direction: **bidirectional** (proposed) vs single.
- Locality bias: **off** by default (proposed) vs on.
