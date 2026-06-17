# Phase 15 — Learned Multi-Scale Token Decoder + Parametrized 6-Embedding Ablation (Spec)

**Status:** design, ready to build. **Target:** the **RMSE_V (vegetation/canopy height) lever** — 0.040 of
our 0.103 gap to #1 (39%, the single largest), saturated under binning because binning re-extracts the
*same* alpha+tessera pixel features. Read `CLAUDE.md`, `docs/science.md`, `docs/results.md` f36/f38/f39/f41-45,
and `docs/literature_domain_adaptation.md` first.

## 0. Why this, and why now (the scientific basis)
Three verified facts reframe the long-standing "patches are null" verdict as **never actually tested**:
1. **ce40 ran with zero patch tokens.** `FreshExtract.forward` gates the whole patch branch on
   `if self.use_terramind and self.patch_inputs:` (`core/model.py:1113`); the winning recipe passed
   `--no-use-terramind`. Every binning gain came from **alpha+tessera pixels alone**. Binning (responsive
   height head) and *any* SAR/structure signal have **never been in the same model**.
2. **When patches were on, they only touched the 2 coarsest levels.** `add` mode injects tokens at L4 (16×16)
   and L3 (32×32) only (`model.py:1124-1127`); levels 64/128/256 are fed *only* by pixel skips. The structure
   signal is confined to the coarse end and dissolves before reaching 10 m output.
3. **The token channels provably encode sub-patch spatial detail.** TerraMind tokens are **FSQ-VAE** latents
   whose decoder reconstructs the full 16×16-pixel Sentinel patch — packing finer-than-160m structure into the
   768 channels *by construction* (the model generates sharp S1/S2). **THOR** (NR/UiT/**ESA Φ-lab** — the
   challenge organizers) is explicitly designed to *"preserve detailed spatial information"* (vs Tessera's
   annual summary), native 10 m → high recoverability ceiling, and **non-redundant with Tessera** (instantaneous
   structure vs annual phenology — the right axis for *height*).

**The hypothesis:** patches read null because (a) they were off, (b) injected only coarsely as a static
context, (c) into a frozen height head. The fix is a **learned multi-scale token decoder** (DPT-style: ViT
tokens → dense prediction via learned channel→space reassembly at every scale; cf. monocular depth) that
inverts the channel-packing and **conditions the height branch at all 5 levels** — pixel skips carry the fine
geometry, SAR tokens carry the vertical-structure signal alpha/tessera lack. **Bilinear upsampling adds nothing;
this is learned detokenization, not interpolation.**
**Honest ceiling:** we hold backbone tokens (not raw tokenizer latents) and cannot fine-tune the FM, so expect
*partial* recovery from ~2024 labeled tiles. The robust win is **cross-scale conditioning** (a new modality at
coarse res) even if sub-token super-resolution recovers little — `pyramid`-vs-`add` measures which.

## 1. Architecture — NEW class `FreshExtractFlex` (`fresh_extract_flex`), additive
**Do not modify `FreshExtract`** — keep ce40 byte-reproducible (CLAUDE.md: one class per architecture,
additive only; 2A-vs-6A only worked because both were preserved). `FreshExtractFlex` generalizes it:
- `CHANNELS=[96,160,256,384,512] @ [256,128,64,32,16]` (identical to `FreshExtract`).
- **Pixel side = parametrized.** Build one stem+`_DeepEncoder` per active `--pixel-inputs` entry
  (`alpha_earth`→stem(64), `tessera`→stem(128)) in a `ModuleDict`. Per-level fuse becomes
  `Conv2d(K·c[l], c[l])` with K = #active pixel encoders (K∈{1,2}). **Each modality keeps its own encoder**
  — never concat at input (6A lesson: concat destroys IoU_B). K=1 → fuse is `Conv2d(c,c)` (kept for a uniform
  pyramid interface).
- **Patch side = parametrized (already is).** `tok_stems` per `--patch-inputs` name; route by suffix
  `*_s1→height`, `*_s2→fraction` (`model.py:1083-1092` logic). Multiple same-sensor streams (e.g.
  `terramind_s1,thor_s1`) → per-modality stem then mean/concat before fusion.
- **Patch-fusion mode flag** `--patch-fusion {add,pyramid}` (default `add` = legacy behavior):
  - `add`: current 2-coarse-level injection (parity check / baseline).
  - **`pyramid` (the experiment):** learned token decoder. Token map (c4@16) → learned upsample blocks
    (`ConvTranspose2d` or `PixelShuffle`, **not** bilinear) producing token-feature maps at {16,32,64,128,256},
    each fused into the matching decoder level of the routed branch via a **gated zero-init residual**
    (identity at init → no destabilization, byte-clean default). s1-pyramid→height decoder, s2-pyramid→fraction.
- **Binned height head** carried over (`height_bins>0` → distribution + soft-expectation; `height_centers`
  buffer; identical to `FreshExtract`).
- Reuse existing submodules (`_ModalityStem`, `_DeepEncoder`, `_PyrDecoder`, `_make_patch_stem`,
  `_TokenCrossAttn`) — the new code is the pixel `ModuleDict`+variable fuse and the `pyramid` token decoder.

## 2. Code changes (additive)
| file | change |
|------|--------|
| `core/model.py` | Add `class FreshExtractFlex`. Add `"fresh_extract_flex"` to `build_model` choices, passing `pixel_inputs`, `patch_inputs`, `patch_fusion`, `height_bins`, `fraction_head`, `patch_stem_version`. |
| `train.py` | Make `FreshExtractFlex` honor `--pixel-inputs` (currently a no-op for fresh_extract). Add `--patch-fusion {add,pyramid}` (default `add`). Add `fresh_extract_flex` to `--model-type` choices. Write `PIXEL_INPUTS`/`PATCH_INPUTS`/`PATCH_FUSION` to params (already partly written). |
| `core/dataset.py` | none expected — `patch_inputs`/`pixel_inputs` plumbing exists; verify pixel-subset (alpha-only / tessera-only) loads correctly (dataset must not require both). |
| `predict.py` | none (reads params; soft-expectation collapse already handles binned head). |

## 3. The 6-embedding ablation (a flag sweep — no per-cell code edits)
All cells: `--model-type fresh_extract_flex --height-bins 256 --height-ce-weight 0.40` + ce40 commons.
- **A. Pixel modality** (patches off): `--pixel-inputs alpha_earth` / `tessera` / `alpha_earth,tessera`
  → answers the *skipped* 7A claim-1 (does dual pixel encoder earn its keep; tessera standalone value).
- **B. SAR-for-height** (pixels=`alpha_earth,tessera`): `--patch-inputs` none / `terramind_s1` /
  `terramind_s1,thor_s1` / all-four → marginal value of 1 vs 2 independent SAR FMs into height.
- **C. Fusion mode** (the central test): fix best data, compare `--patch-fusion add` vs `pyramid`
  → isolates *"is it the data, or the way we decode it?"* (= is sub-token detail recoverable / does
  multi-scale conditioning help).
- **Headline cell:** `--pixel-inputs alpha_earth,tessera --patch-inputs terramind_s1,thor_s1
  --patch-fusion pyramid --height-bins 256`.

## 4. Evaluation protocol — MULTI-FOLD (gain may be transfer-only, like binning/U-Net++)
Single-fold val misled us repeatedly (f40/f43). Judge by **held-out-region transfer**: hold out fold
N∈{0,1,2} in turn (`--cv-fold N`), report **mean held-out** (proxy + RMSE_V + IoU_B) **and the
train→held-out gap**. A win = higher mean held-out and/or smaller gap across folds. Confirm the winner on
the **platform** (the real test — Do not auto-submit; human decides). Reference: ce40 = fold-1, scale-1.

## 5. Smoke (STOP & report before any campaign)
1. `FreshExtractFlex` builds for `--pixel-inputs alpha_earth` (K=1), `tessera` (K=1), `alpha_earth,tessera`
   (K=2) — variable fuse wired correctly; param count sane.
2. THOR + TerraMind tokens load; `terramind_s1,thor_s1` routes to height, `*_s2` to fraction; token shapes
   (B,256,768)→(B,c4,16,16) line up; `pyramid` decoder emits 256×256 at every fused level; finite loss.
3. `pyramid` gated residual is **identity at init** (max|Δ vs add-off| ≈ 0 on first forward) — no
   destabilization. Seeded → reproducible tiles (don't regress f25).
4. bs24 + `expandable_segments` memory OK with the extra `pyramid` params on a 48 GB card.

## 6. Success / fallback
- **Win:** `pyramid` + dual-SAR lifts mean held-out RMSE_V (and/or shrinks the gap) vs `add`/no-patch across
  folds → confirm on platform; this is the first real test of SAR-for-height under a responsive (binned) head.
- **Null (`pyramid`≈`add`≈no-patch across folds):** sub-token height detail is not recoverable from frozen
  backbone tokens by a small decoder, and the coarse SAR prior is redundant with tessera → patches are
  genuinely exhausted; fall back to **multi-seed ensemble of ce40** and consolidate at best-honest-rank.

## 7. Constraints
- Additive; `FreshExtract` untouched (ce40 reproducible). `--patch-fusion add` = legacy parity.
- **Seeded** aug (f25); **val never augmented**; **no test-label peeking** (no threshold calibration).
- ce40 commons: `--patch-stem-version v2 --no-use-gradnorm --amp --fraction-head softmax4
  --building-overlap tversky --decouple-height --build-height-weight 1.0 --veg-height-weight 3.0
  --batch-size 24 --epochs 40 --num-workers 4 --cache-dir /home/bassam/nvme_cache/cache7a`,
  prefixed `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Each new `--cv-fold` builds its own cache.
- Per-machine `guardian --start` (cooling + 82 °C failsafe) before any node run; central thermal monitor up.

## 8. Launch (after code change + smoke STOP/report)
Common (ce40, §7) + per-node `--cv-fold {0|1|2}` + the ablation flags of §3, distinct
`--experiment-name 15_<cell>` and `--scratch-dir /home/bassam/nvme_cache/runs_local/<name>`, piped to
`tee train_<name>.log`. Start with the **C. fusion test** (`add` vs `pyramid` at the headline data) on two
held-out folds — the highest-information cell — before the full grid.
