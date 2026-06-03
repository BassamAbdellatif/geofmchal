# Phase 5D — THOR Integration with Enhanced Patch Stems

**Branch:** `exp-7-clean-slate`  ·  **Date:** 2026-06-03  ·  **Status:** smoke checkpoint reached, **training not yet launched (awaiting review).**

Spec: `prompts/Exp 7 phase5d thor·MD` (to be renamed `prompts/exp-7-phase5d-thor.md`).
Reference config: `7A_simple` (static weights `[0.65,0.64,1.70]`, no GradNorm, no vegboost, bridge on, binary head on, stratified sampler on; proxy 0.399, IoU_B 0.215).

---

## 1. Goal

Test whether THOR (a second S1/S2 foundation model, different SSL objective from TerraMind)
adds value to the 7A architecture when given **per-modality calibration** before fusion, rather
than being naively mixed. Two additive, optional changes:

1. **`PatchTokenStemV2`** — LayerNorm + 2-layer MLP per modality (vs the original v1 = Linear+GELU+LayerNorm).
2. **THOR patch tokens** routed through the existing sensor-routed cross-attention injection
   (`thor_s1`→height decoder, `thor_s2`→fraction decoder), concatenated with TerraMind tokens
   along the token dimension before cross-attention.

Both are gated behind `--patch-inputs` / `--patch-stem-version`; with THOR absent and `v1` the
model is **byte-identical** to `7A_simple`.

---

## 2. Prerequisite finding — the cache did not contain THOR

The NVMe cache (`/home/bassam/nvme_cache/cache7a`) and its shared source held only the 4 base
streams (alpha_earth, tessera, terramind_s1, terramind_s2) + target. **No THOR.**

THOR *raw* embeddings do exist on disk (`embed2heights/data/{train,test}/thor_*_emb`, 2024 train
tiles each, shape `(768,16,16)` — identical to TerraMind). But the dataset pipeline itself had no
THOR support: `find_multimodal_train_tiles`, `_CACHE_SPECS`, `_preprocess_tile`, and
`norm_stats.json` all omitted it. So a cache "rebuild" required first extending the dataset code +
computing THOR norm stats.

**Resolution (agreed split):** code + norm stats implemented here; the human owns the cache
rebuild + node sync. THOR cache subsequently built at `/home/bassam/nvme_cache/cache7a` (train +
val `.done` present, all 7 streams). The rebuild command (run per node):

```bash
cd /mnt/head/users/bassam/src/geofmchal && ./run_env.sh -c "
from train import DATA_ROOT_7A
from core.dataset import find_multimodal_train_tiles, GeoFMDataset7A
pi = ['terramind_s1','terramind_s2','thor_s1','thor_s2']
tiles = find_multimodal_train_tiles(DATA_ROOT_7A, use_thor=True)
print('data_root:', DATA_ROOT_7A, '| matched tiles:', len(tiles))   # expect 2024
for is_train in (True, False):
    GeoFMDataset7A(tiles, is_train=is_train, cv_fold=0,
                   cache_dir='/home/bassam/nvme_cache/cache7a',
                   rebuild_cache=True, patch_inputs=pi)
print('THOR cache built')
"
```
The cache is a superset; non-THOR configs read their subset from it. `rebuild_cache=True`
overwrites in place (no duplication, ~+1.6 GB for THOR, ~51 GB total).

---

## 3. Architecture (current network)

### 3.1 What "full-THOR" is
Purely a `--patch-inputs` value — **no new script, nothing hardcoded.** The same `train.py` and the
same model class `DualEncDualDecFusion` serve every config:

| Config        | `--patch-inputs`                                  | `--patch-stem-version` |
|---------------|---------------------------------------------------|------------------------|
| baseline/ref  | `terramind_s1,terramind_s2`                       | v1                     |
| v2stem        | `terramind_s1,terramind_s2`                       | v2                     |
| **full-THOR** | `terramind_s1,terramind_s2,thor_s1,thor_s2`       | v2                     |
| thor_s1       | `terramind_s1,terramind_s2,thor_s1`               | v2                     |
| thor_s2       | `terramind_s1,terramind_s2,thor_s2`               | v2                     |
| nostem        | `terramind_s1,terramind_s2,thor_s1,thor_s2`       | v1                     |

The model instantiates one patch stem per name (`nn.ModuleDict`, built in list order), then routes
at runtime by name suffix (`_route_tokens`): any `*_s1` → height decoder, any `*_s2` → fraction
decoder. Omit the `thor_*` names and no THOR stems exist → byte-identical to `7A_simple`. THOR
never appears as a literal in the model.

### 3.2 Two encoders, two decoders
- **AlphaEarth encoder → Fraction decoder** (urban-spatial; outputs 3 fraction channels + 1 aux binary-building channel).
- **TESSERA encoder → Height decoder** (temporal-natural; outputs 1 nDSM height channel).

Each encoder is a `UNetEncoderHalf`: channels `[96,128,192,256,384]` at resolutions
`[256,128,64,32,16]`, producing skips `L0..L3` and a `384@16` bottleneck.

### 3.3 The connections (three distinct mechanisms — do not conflate)
1. **U-Net skips (within-modality, concat).** Each decoder up-block concatenates *its own*
   encoder's skip at matching resolution: Fraction decoder uses α-skips, Height decoder uses τ-skips.
   This is two parallel U-Nets, not a cross-wired skip grid.
2. **Bridge (the only cross-encoder link).** The Height decoder's first block concatenates the
   **alpha** bottleneck with the **tessera** bottleneck (1×1 conv fuse), with `GradScale(α=0.2)` on
   the alpha side so height loss can't dominate the AlphaEarth encoder. On by default
   (`--no-height-bridge` disables).
3. **Patch-token injection (cross-attention, not concat).** TerraMind/THOR tokens enter via
   cross-attention at the **16×16** (bottleneck) and **32×32** (after up1) decoder levels:
   decoder features = queries, patch tokens = keys/values, residual + LayerNorm. S1 tokens →
   height decoder, S2 tokens → fraction decoder. When multiple same-sensor streams are active
   (TerraMind + THOR), each is projected by its **own** `PatchTokenStemV2` and the outputs are
   **concatenated along the token dim** (256→512 K/V at 16×16; each 16×16 grid resampled
   independently to 32×32 → 2×1024 K/V). THOR therefore only adds keys/values — it never touches
   the encoders, the U-Net skips, or the bridge. Its only cost is attention memory.

### 3.4 Schematic
```
 alpha_earth (B,64,256²)              tessera (B,128,256²)
        │ AlphaStem (BN, →96)               │ TesseraStem (BN, →96)
        ▼                                   ▼
   alpha_encoder (UNetEncoderHalf)     tessera_encoder (UNetEncoderHalf)
   skips a0..a3 @ [256,128,64,32]      skips t0..t3 @ [256,128,64,32]
   bottleneck aBN = 384@16             bottleneck tBN = 384@16
        │                                   │
        │        ┌── aBN (GradScale α=0.2) ─► BRIDGE ─┐
        ▼        │                                    ▼
 ┌──────────────────────────┐        ┌────────────────────────────────┐
 │ FRACTION DECODER (α-path) │        │ HEIGHT DECODER (τ-path)         │
 │ 16² ⟵xATTN⟵ S2 tokens     │        │ 16² ⟵xATTN⟵ S1 tokens           │
 │ up1→32² (concat a3)       │        │ up1→32² (concat t3)             │
 │ 32² ⟵xATTN⟵ S2 tokens     │        │ 32² ⟵xATTN⟵ S1 tokens           │
 │ up2→64² (a2) up3→128² (a1)│        │ up2→64² (t2) up3→128² (t1)      │
 │ up4→256² (a0)             │        │ up4→256² (t0)                   │
 │ → frac_head(3) +binary(1) │        │ → height_head(1)                │
 └──────────────────────────┘        └────────────────────────────────┘
        │                                          │
   fraction (B,V,W) + binary(aux)             height (nDSM)

 Patch tokens: each (B,256,768) → PatchTokenStemV2 → (B,256,384)
   S1 K/V = cat[terramind_s1, thor_s1] → height decoder    (own stem each)
   S2 K/V = cat[terramind_s2, thor_s2] → fraction decoder
```

---

## 4. Implementation (additive, per file)

- **`core/model.py`**
  - `PatchTokenStem` → renamed **`PatchTokenStemV1`** (byte-identical, name only).
  - **`PatchTokenStemV2`** added (spec exact): `LayerNorm(768) → Linear → GELU → Linear → +pos_embed`.
  - `PatchCrossAttnBlock` made **multi-grid aware**: concatenated `G` 16×16 grids used directly as
    K/V at 16×16; each grid resampled *independently* at higher resolutions (merging into one fake
    grid would scramble modalities). G=1 path is bit-identical to before.
  - `xattn_heads` threaded through both decoders (default 4 → byte-identical).
  - `DualEncDualDecFusion` made data-driven: `patch_inputs`, `patch_stem_version`, `xattn_heads`;
    `ModuleDict` of stems built in input order (preserves init RNG order); suffix routing; `None`
    skips injection when a sensor side is absent.
  - `build_model(...)` passes the three new args through.
- **`core/dataset.py`** — `find_multimodal_train_tiles(use_thor=)`; `GeoFMDataset7A(patch_inputs=)`;
  per-instance cache specs (+thor); `_preprocess_tile` reads/normalises THOR; `_init_cache` uses
  instance specs; `__getitem__` + `augment_domain_shift` handle THOR. **Non-THOR path is
  byte-identical** (no extra RNG draws, same output keys).
- **`train.py`** — flags `--patch-stem-version {v1,v2}` (default v2), `--xattn-heads` (default 4);
  parses/validates `--patch-inputs`, infers `use_thor`, threads into tiles/datasets/model; writes
  `PATCH_INPUTS`, `PATCH_STEM_VERSION`, `XATTN_HEADS`, `USE_THOR` to `training_params.txt`.
- **`predict.py`** — `predict_7a` reconstructs the architecture from `training_params.txt`;
  THOR-aware test-tile loading + TTA; **legacy state-dict remap** so pre-5D `7A_simple` checkpoints
  (old `s1/s2_token_stem.*` keys) still load.
- **`compute_norm_stats.py`** — added THOR modalities, `--only` subset, and **merge** (no longer
  overwrites the whole file). THOR stats computed and merged; the pre-existing 4 keys are
  byte-identical to backup.

### State-dict key change
Stems moved from attributes `s1_token_stem`/`s2_token_stem` to `token_stems.terramind_s1`/`_s2`.
All four Phase 5D runs train fresh so this is internal; the predict.py legacy remap preserves
loading of the historical `7A_simple` checkpoint. (Downstream `eval_inference.py` /
`threshold_scan.py` build the model with defaults and would need the same param-aware
reconstruction if used on THOR checkpoints — not on the critical path for this phase.)

---

## 5. Verification

| Check | Result |
|---|---|
| **v1-stem byte-identical** vs pre-edit model | **max_abs_diff = 0.000e+00**, params 18,663,141 (identical) |
| Dataset→model pipeline (no cache, NFS) | finite for baseline / thor_full (val+train) / thor_s2 |
| THOR cache build + readback (temp dir) | 7 files written; cache-vs-nocache diff ~0.002 = pure f16 roundtrip (same as terramind) |
| predict.py param round-trip + legacy remap | all configs reconstruct, load, forward |
| THOR norm stats merged | 6 keys; pre-existing 4 byte-identical to backup |
| Compile + import (all 5 files) | OK |

### Parameter counts
| Config | Params |
|---|---|
| baseline / v1 | 18.66 M |
| v2stem (no THOR) | 18.96 M |
| **thor_full (v2)** | **20.05 M** |
| thor_s1 / thor_s2 (v2) | 19.50 M |
| nostem (full THOR, v1) | 19.45 M |

### Norm-stats finding (validates the hypothesis)
THOR's raw magnitudes dwarf TerraMind's — **`thor_s2` std up to 4425** vs terramind_s2's 7.4
(`thor_s1` std ≤21 vs terramind_s1 ≤1.7). A shared/bare-linear projection would be swamped by
THOR; per-modality norm-stats + the v2 LayerNorm are doing real calibration work. Prediction:
the v1 `nostem` ablation should underperform v2 with THOR.

---

## 6. Smoke tests (batch 4, 1 epoch, 2 train batches, cache)

All three ran with **finite loss** and completed (proxy/IoU from 2 batches are meaningless, omitted).

| Config | Params | Peak GPU @ batch 4 |
|---|---|---|
| smoke_baseline_v1 | 18.66 M | 5.43 GB |
| smoke_thor_full | 20.05 M | 5.67 GB |
| smoke_thor_s2 | 19.50 M | 5.56 GB |

`training_params.txt` for all three correctly recorded the new flags (write side confirmed on real
files; read side via predict.py round-trip).

### GPU memory at training batch sizes (full-THOR, measured/derived)
Measured: **batch 32 → 43.4 GB / 48 GB (~90%, tight)**. Linear fit (≈0.28 GB fixed + 1.35 GB/sample):

| Batch | Full-THOR peak (GB) | Headroom on 48 GB |
|---|---|---|
| 32 | 43.4 (measured) | ~4.6 GB (tight) |
| 24 | ~32.6 | comfortable |
| 16 | ~21.8 | very safe |

The two full-THOR configs (`7A_simple_thor`, `7A_simple_nostem`) drive this; v2stem and thor_s1 are lighter.

---

## 7. Decisions & discrepancies

1. **`--use-gradnorm False` is rejected by argparse** (it's a `BooleanOptionalAction`). Correct token
   to disable GradNorm is **`--no-use-gradnorm`**. The spec's command blocks need this substitution.
2. **Batch 32 is memory-tight for full-THOR.** Decision (approved): run training at **batch 24**
   (all four runs, for a clean cross-run comparison; ~32.6 GB, comfortable). Note this means
   `7A_simple_v2stem` differs from the historical `7A_simple` (batch 32) by batch size as well as
   stem — so Question 1 below is best read alongside an optional bs-matched `7A_simple` v1 re-run at
   batch 24 if perfectly clean stem isolation is wanted.
3. **Dataset = selective, not always-all.** THOR loaded only when in `--patch-inputs`, keeping the
   non-THOR data path byte-identical so v2stem-vs-baseline isolates only the stem.
4. **Cross-attn doubles K/V at 32×32 too** (2×1024), not only at 16×16 — slightly more memory than
   the spec note implied; reflected in the measurements above. `--xattn-heads 2` ≈ halves the
   attention score tensor if more margin is needed.

---

## 8. Planned training runs (corrected) — NOT yet launched

Four nodes, **60 epochs**, static weights, no GradNorm, no vegboost, with cache, **batch 24**,
`--no-use-gradnorm`. (`--xattn-heads 4` default; runs differ only in `--patch-inputs` /
`--patch-stem-version`.)

```
n1  7A_simple_v2stem  : --patch-inputs terramind_s1,terramind_s2               --patch-stem-version v2
n2  7A_simple_thor    : --patch-inputs terramind_s1,terramind_s2,thor_s1,thor_s2 --patch-stem-version v2
n3  7A_simple_thor_s1 : --patch-inputs terramind_s1,terramind_s2,thor_s1        --patch-stem-version v2
n4  7A_simple_nostem  : --patch-inputs terramind_s1,terramind_s2,thor_s1,thor_s2 --patch-stem-version v1
```
Common flags: `--model-type dual_enc_dec_fusion --pixel-inputs alpha_earth,tessera
--no-use-gradnorm --static-weights "0.65,0.64,1.70" --veg-height-boost 0.0
--cache-dir /home/bassam/nvme_cache/cache7a --num-workers 8 --cv-fold 0 --batch-size 24
--epochs 60 --seed 0`.

### Questions to answer from results (ref: `7A_simple`, proxy 0.399, v1, no THOR)
1. **Enhanced stem alone?** `7A_simple_v2stem` vs `7A_simple` (isolates v2 vs v1).
2. **THOR adds value with v2?** `7A_simple_thor` vs `7A_simple_v2stem` (>0.005 ⇒ additive).
3. **THOR mostly on the height/S1 side?** `7A_simple_thor_s1` vs `7A_simple_v2stem` (RMSE_B/V).
4. **Does v2 matter for THOR?** `7A_simple_thor` (v2) vs `7A_simple_nostem` (v1), both full-THOR.

RMSE_V expected ~4.0 m across all (cliff problem orthogonal to THOR). Do **not** submit to the
platform without explicit approval.

---

## 9. Not done (awaiting go-ahead)
- No training launched.
- Spec not renamed/committed (`prompts/Exp 7 phase5d thor·MD` → `prompts/exp-7-phase5d-thor.md`);
  nothing committed; `results.md` / `science.md` untouched.
- Artifacts in tree: `_thor_smoke/` (verification scripts incl. `ref.pt`), `data/norm_stats.json.bak`.
