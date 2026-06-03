# Coding Task: Phase 5D — THOR Integration with Enhanced Patch Stems

## Context
Read `CLAUDE.md`, `docs/results.md`, `docs/science.md` before starting. This is a continuation of the `exp-7-clean-slate` branch. Phases 5B and 5C are complete.

**Critical update from Phase 5C — the baseline has changed.** The recommended 7A configuration is now `7A_simple`:
- Static loss weights `[0.65, 0.64, 1.70]` (`--use-gradnorm False`)
- No vegboost (`--veg-height-boost 0.0`)
- 60 epochs (the model peaks ~epoch 40–59; 90 epochs wastes time)
- Bridge on, binary head on, stratified sampler on
- `7A_simple` proxy = 0.399, IoU_B 0.215 — the strongest 7A config found

**Three findings from Phase 5C that shape this task:**
- GradNorm ≈ static weights (finding 18) → use static weights, not GradNorm
- vegboost is structurally ineffective at 1.0 and 5.0 (finding 17) → do not use vegboost
- The auxiliary binary head is essential for IoU_B (finding 19) → keep it on

This phase makes two additive changes to test the THOR hypothesis:

1. **Enhanced `PatchTokenStem`** (LayerNorm + 2-layer MLP) — per-modality calibration before patch token fusion.
2. **THOR patch token support** wired via the same sensor-routed skip injection as TerraMind. S1 → height decoder, S2 → fraction decoder.

---

## Science background (why this matters)

In 2A experiments, adding THOR alongside TerraMind at the bottleneck gave no gain or slight regression. We concluded THOR was redundant. The 7A finding changes this interpretation: TESSERA also looked harmful in 6A (IoU_B 0.168 → 0.017) when naively concatenated, but became beneficial in 7A once it got its own encoder with per-modality normalisation.

The mechanism: foundation model embeddings from different models have mismatched value ranges, channel covariance structure, and effective rank. A bare linear projection forces one matrix to simultaneously rescale two misaligned embedding spaces — the dominant modality (TerraMind, larger norm) hijacks the projection and the other is suppressed. Independent LayerNorm + 2-layer MLP per modality calibrates each space before fusion.

THOR and TerraMind are both S1/S2 foundation models trained with different SSL objectives (TerraMind: MAE-style; THOR: different self-supervised task). Their embedding spaces are not aligned. The enhanced `PatchTokenStem` with per-modality LayerNorm is the right way to bring them into a common cross-attention space before they are concatenated as cross-attention keys/values.

**Expected gain**: +0.005–0.015 if THOR was previously suppressed by TerraMind's larger embedding norm. The hypothesis is falsifiable via the `nostem` ablation below.

---

## Branch and naming

Stay on `exp-7-clean-slate`. Commit this spec as `prompts/exp-7-phase5d-thor.md` as the first commit.

Experiment names:
- `7A_simple` — reference (no THOR, enhanced stems, already run, proxy 0.399). Re-run only if not reproducible.
- `7A_simple_thor` — +THOR with enhanced (v2) stems, all four patch streams
- `7A_simple_thor_s1` — +THOR S1 only (height decoder path)
- `7A_simple_nostem` — THOR + TerraMind with OLD bare-linear (v1) stems — ablation: does the enhanced stem matter?

---

## Implementation scope

### Change 1 — Enhanced `PatchTokenStem`

In `core/model.py`, rename the existing patch token stem class to `PatchTokenStemV1` (keep it byte-identical, only the name changes). Add `PatchTokenStemV2` alongside it:

```python
class PatchTokenStemV2(nn.Module):
    """
    Per-modality calibration before patch token fusion.
    LayerNorm normalises embedding statistics independently per modality.
    2-layer MLP with GELU projects to common cross-attention dimension.
    Learned positional encoding added after projection.
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 384):
        super().__init__()
        self.norm   = nn.LayerNorm(in_dim)
        self.proj1  = nn.Linear(in_dim, out_dim)
        self.act    = nn.GELU()
        self.proj2  = nn.Linear(out_dim, out_dim)
        self.pos_embed = nn.Parameter(torch.randn(256, out_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 256, in_dim]
        x = self.norm(x)
        x = self.act(self.proj1(x))
        x = self.proj2(x)
        return x + self.pos_embed
```

Add `--patch-stem-version {v1,v2}` flag, default `v2`. The model selects which stem class to instantiate based on this flag.

**Backward compatibility**: with `--patch-stem-version v1`, the model must be byte-identical to the current `7A_simple` model. Confirm by checking that a forward pass with v1 and a fixed seed produces the same output as the pre-change model. Report max_abs_diff = 0.

---

### Change 2 — THOR support in `DualEncDualDecFusion`

THOR tokens follow the same sensor-routed skip injection as TerraMind:
- `thor_s1` → height decoder (same path as `terramind_s1`)
- `thor_s2` → fraction decoder (same path as `terramind_s2`)

At each injection point, THOR and TerraMind tokens for the same sensor type are **concatenated along the token dimension** before cross-attention:

```python
# At height decoder skip injection, when both terramind_s1 and thor_s1 present:
s1_tokens = torch.cat([terramind_s1_proj, thor_s1_proj], dim=1)  # [B, 512, 384]
# cross-attention: Q = decoder features, K/V = s1_tokens
```

Each modality gets its own `PatchTokenStemV2` instance (separate LayerNorm + MLP per modality — this is the whole point, per-modality calibration). So with full THOR you have four stems: terramind_s1, terramind_s2, thor_s1, thor_s2.

K/V token count doubles (256 → 512) at injection points where both modalities are present. Attention score matrix grows in the K dimension: `[B, heads, Q, 512]` vs `[B, heads, Q, 256]`. ~2× attention memory at those levels. Watch for OOM in smoke test.

**Fallback if OOM**: add `--xattn-heads N` flag (default 4). Reduce to 2 when THOR is active if OOM.

**Graceful absence**: if `--patch-inputs` excludes THOR, no THOR stems are instantiated and the model is byte-identical to `7A_simple`. The routing is data-driven by the embedding-name suffix — no hardcoded THOR.

---

### Change 3 — Verify flexible `--patch-inputs`

`--patch-inputs` accepts any comma-separated subset of: `terramind_s1`, `terramind_s2`, `thor_s1`, `thor_s2`.

Routing rules (data-driven by name suffix):
- Any `*_s1` token → height decoder injection
- Any `*_s2` token → fraction decoder injection
- Multiple same-sensor tokens → concatenated along token dim before cross-attention
- No `*_s1` tokens → height decoder gets no patch injection (τ-encoder features only)
- No `*_s2` tokens → fraction decoder gets no patch injection

### Change 4 — Save new flags to `training_params.txt`

`predict.py` reconstructs the model from `training_params.txt`. The new flags `--patch-stem-version`, `--xattn-heads`, and the full `--patch-inputs` list affect architecture and must be written and read back correctly. Confirm predict.py round-trips them.

**Do NOT add vegboost or build-height-boost flags.** Phase 5C proved them ineffective. Keep the loss as `7A_simple` uses it. If the flags already exist from a prior phase, leave them but default to 0.0 and do not use them in this phase.

---

## Testing plan

### Smoke tests (batch 4, 2 batches, with cache for speed)

Cache dir: `/home/bassam/nvme_cache/cache7a` (adjust if the path differs — check where the Phase 5 cache was built).

```bash
# 1. Baseline reproduction — v1 stem, no THOR. Must reproduce 7A_simple behaviour.
./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --patch-stem-version v1 \
  --use-gradnorm False --static-weights "0.65,0.64,1.70" \
  --veg-height-boost 0.0 \
  --cache-dir /home/bassam/nvme_cache/cache7a \
  --experiment-name smoke_baseline_v1 \
  --batch-size 4 --epochs 1 --max-batches 2

# 2. Full THOR — v2 stem, all four patch streams
./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2,thor_s1,thor_s2 \
  --patch-stem-version v2 \
  --use-gradnorm False --static-weights "0.65,0.64,1.70" \
  --veg-height-boost 0.0 \
  --cache-dir /home/bassam/nvme_cache/cache7a \
  --experiment-name smoke_thor_full \
  --batch-size 4 --epochs 1 --max-batches 2

# 3. THOR S2 only — tests asymmetric routing (fraction decoder only)
./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2,thor_s2 \
  --patch-stem-version v2 \
  --use-gradnorm False --static-weights "0.65,0.64,1.70" \
  --veg-height-boost 0.0 \
  --cache-dir /home/bassam/nvme_cache/cache7a \
  --experiment-name smoke_thor_s2 \
  --batch-size 4 --epochs 1 --max-batches 2
```

All three must run with finite loss. Report:
- GPU memory at batch 4 for smoke_thor_full (the heaviest config)
- Parameter count: full-THOR vs baseline
- max_abs_diff between smoke_baseline_v1 output and the pre-change `7A_simple` model (should be 0)
- Confirmation that `training_params.txt` writes the new flags and `predict.py` reads them

**Stop here. Report and wait for review before launching training runs.**

---

## Training runs — launch after checkpoint approved

Four nodes, **60 epochs**, static weights, no vegboost, with cache.

### Node n1 — 7A_simple (reference, re-run for clean comparison)
```bash
ssh n1
nohup ./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2 \
  --patch-stem-version v2 \
  --use-gradnorm False --static-weights "0.65,0.64,1.70" \
  --veg-height-boost 0.0 \
  --cache-dir /home/bassam/nvme_cache/cache7a --num-workers 8 \
  --cv-fold 0 --batch-size 32 --epochs 60 --seed 0 \
  --experiment-name 7A_simple_v2stem \
  > /mnt/head/users/bassam/data/geofmdata/runs/7A_simple_v2stem/train.log 2>&1 &
```

Note: this uses the v2 stem (no THOR). It tests whether the enhanced stem alone — without THOR — changes anything vs the original `7A_simple` (which used v1). If v2-stem-no-THOR differs from `7A_simple`, the stem change itself has an effect independent of THOR.

### Node n2 — 7A_simple_thor (full THOR, v2 stem)
```bash
ssh n2
nohup ./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2,thor_s1,thor_s2 \
  --patch-stem-version v2 \
  --use-gradnorm False --static-weights "0.65,0.64,1.70" \
  --veg-height-boost 0.0 \
  --cache-dir /home/bassam/nvme_cache/cache7a --num-workers 8 \
  --cv-fold 0 --batch-size 32 --epochs 60 --seed 0 \
  --experiment-name 7A_simple_thor \
  > /mnt/head/users/bassam/data/geofmdata/runs/7A_simple_thor/train.log 2>&1 &
```

### Node n3 — 7A_simple_thor_s1 (THOR S1 only)
```bash
ssh n3
nohup ./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2,thor_s1 \
  --patch-stem-version v2 \
  --use-gradnorm False --static-weights "0.65,0.64,1.70" \
  --veg-height-boost 0.0 \
  --cache-dir /home/bassam/nvme_cache/cache7a --num-workers 8 \
  --cv-fold 0 --batch-size 32 --epochs 60 --seed 0 \
  --experiment-name 7A_simple_thor_s1 \
  > /mnt/head/users/bassam/data/geofmdata/runs/7A_simple_thor_s1/train.log 2>&1 &
```

### Node n4 — 7A_simple_nostem (full THOR, OLD v1 stem)
```bash
ssh n4
nohup ./run_env.sh train.py --model-type dual_enc_dec_fusion \
  --pixel-inputs alpha_earth,tessera \
  --patch-inputs terramind_s1,terramind_s2,thor_s1,thor_s2 \
  --patch-stem-version v1 \
  --use-gradnorm False --static-weights "0.65,0.64,1.70" \
  --veg-height-boost 0.0 \
  --cache-dir /home/bassam/nvme_cache/cache7a --num-workers 8 \
  --cv-fold 0 --batch-size 32 --epochs 60 --seed 0 \
  --experiment-name 7A_simple_nostem \
  > /mnt/head/users/bassam/data/geofmdata/runs/7A_simple_nostem/train.log 2>&1 &
```

Monitor after 15 minutes:
```bash
for exp in 7A_simple_v2stem 7A_simple_thor 7A_simple_thor_s1 7A_simple_nostem; do
  echo "=== $exp ===" && tail -3 \
  /mnt/head/users/bassam/data/geofmdata/runs/$exp/train.log 2>/dev/null || echo "NOT STARTED"
done
```

---

## What to read from the results

Comparison table with `7A_simple` (proxy 0.399, v1 stem, no THOR) as the reference:

```
| Experiment          | Best ep | Proxy | IoU_B | IoU_V | IoU_W | RMSE_B | RMSE_V |
| 7A_simple (ref)     | 59      | 0.399 | 0.215 | 0.810 | 0.684 | 2.06   | 4.03   |
| 7A_simple_v2stem    | ?       | ?     | ...                                       |
| 7A_simple_thor      | ?       | ?     | ...                                       |
| 7A_simple_thor_s1   | ?       | ?     | ...                                       |
| 7A_simple_nostem    | ?       | ?     | ...                                       |
```

Then answer four questions:

1. **Does the enhanced stem alone help (no THOR)?** Compare `7A_simple_v2stem` to `7A_simple` (v1). Isolates the stem change from THOR.

2. **Does THOR add value with enhanced stems?** Compare `7A_simple_thor` to `7A_simple_v2stem`. Both use v2 stem; only THOR differs. If proxy improves by more than 0.005, THOR is additive.

3. **Is THOR's value mostly on the height (S1) side?** Compare `7A_simple_thor_s1` to `7A_simple_v2stem`. THOR S1 feeds the height decoder; if RMSE_B or RMSE_V improves, THOR SAR is helping structure.

4. **Does the enhanced stem matter for THOR specifically?** Compare `7A_simple_thor` (v2) to `7A_simple_nostem` (v1), both with full THOR. If v2 > v1, the per-modality LayerNorm is doing the work the hypothesis predicts.

Note: RMSE_V is expected to stay ~4.0m across all runs (the cliff problem is unsolved and orthogonal to THOR). THOR is being tested for IoU and RMSE_B gains, not RMSE_V. Do not expect THOR to fix the cliff.

---

## Acceptance criteria

**After smoke tests**: three configs run, GPU memory reported, v1-stem reproduction confirmed (max_abs_diff = 0), training_params.txt round-trip confirmed.

**After training**: comparison table with the four questions answered. **Do NOT submit to the platform without explicit approval.**

---

## Things to NOT do
- Do not use vegboost or GradNorm — Phase 5C proved both ineffective; this phase uses the `7A_simple` config (static weights, no vegboost)
- Do not modify existing model classes destructively — rename the old stem to `PatchTokenStemV1` (name only) and add `PatchTokenStemV2` alongside
- Do not change default behaviour when THOR is absent — must be byte-identical to `7A_simple`
- Do not hardcode THOR — it must remain optional via `--patch-inputs`
- Do not run training before the smoke checkpoint is approved
- Do not run beyond 60 epochs unless proxy is still climbing at epoch 50
- Do not submit to platform without approval
- Do not modify `results.md` or `science.md` — append to `CLAUDE.md` only with new experiment/model names

## If you get stuck
- OOM with full THOR at batch 32 → `--xattn-heads 2` first, report memory before reducing batch
- THOR embeddings not in the cache → check if the cache build included thor_s1/thor_s2; if not, STOP and report (the cache may need rebuilding with THOR included)
- THOR embeddings not on disk at all → STOP, report
- v1-stem reproduction max_abs_diff ≠ 0 → the rename changed behaviour; STOP and find what differs
- training_params.txt not writing new flags → fix before training, not after
- Any NaN → STOP, do not retry, report per-task loss breakdown

Report any decision point with multiple reasonable paths rather than picking silently.