# Option 2A: Attention-Gated Skip Connections

We will implement "Option 2A: Attention-Gated Skip Connections" to fuse pixel-level high-resolution optical embeddings (Tessera) with patch-level low-resolution semantic embeddings (TerraMind S1).

## User Review Required

> [!NOTE]
> We will add the AttentionFusedDecoder model under the `attention_fusion` model type option in `train.py`.
> The new dataset class `Emb2HeightsDataset` will handle loading Tessera, TerraMind S1, and label pairs together and matching them using their normalized core IDs.

## Open Questions

No open questions.

## Proposed Changes

### Configuration

#### [MODIFY] [config.py](file:///mnt/head/users/bassam/src/geofmchal/config.py)
Make sure config contains all directory constants (already done in previous steps).

---

### Dataset

#### [MODIFY] [dataset.py](file:///mnt/head/users/bassam/src/geofmchal/core/dataset.py)
* Add `find_triple_file_pairs(pixel_dir, patch_dir, label_dir)` helper function to match Tessera, TerraMind S1, and label images by their normalized core IDs.
* Add `Emb2HeightsDataset(Dataset)` which:
  - Takes pixel embeddings dir, patch embeddings dir, and labels dir.
  - Matches files.
  - In `__getitem__`, loads Tessera `.tif` (shape: `128, 256, 256`), TerraMind S1 `.tif` (shape: `768, 16, 16`), and Target `.tif` (shape: `4, 256, 256`).
  - Handles padding and random/center cropping with 16x scaling factor (so cropping 16x16 in patch-level matches 256x256 in pixel-level).
  - Returns a dictionary `{"pixel_emb": pixel_emb, "patch_emb": patch_emb, "target": target}`.

---

### Model Architecture

#### [MODIFY] [model.py](file:///mnt/head/users/bassam/src/geofmchal/core/model.py)
* Implement `AttentionGate(nn.Module)` executing $\sigma(W_{int}(ReLU(W_x(x) + W_g(g))))$.
* Implement `ResidualConvBlock(nn.Module)` for standard dense residual convolutions.
* Implement `AttentionFusedDecoder(nn.Module)` combining:
  - **Encoder**: Process pixel_emb (128 channels) down to a 16x16 bottleneck, saving skip connections.
  - **Bottleneck Fusion**: Concat encoder bottleneck (e.g. 512 channels) and patch_emb (768 channels), reduce channels using a 1x1 convolution block.
  - **Decoder**: Upsample bottleneck. Pass skip connection and decoder features through `AttentionGate`, concatenate them, and apply `ResidualConvBlock`.
  - **Output Head**: Final convolution to exactly 4 channels.
* Update `build_model` to register and instantiate the `attention_fusion` model.

---

### Training Script

#### [MODIFY] [train.py](file:///mnt/head/users/bassam/src/geofmchal/train.py)
* Add `attention_fusion` to choices of model type.
* In data setup, if model type is `attention_fusion`, instantiate `Emb2HeightsDataset` passing the Tessera and TerraMind S1 directories.
* Update training and validation loop to unpack the output dictionary if `attention_fusion` is used.
* Add `align_target_to_output` definition to avoid name errors in visualization.

---

## Verification Plan

### Automated Tests
* Run a smoke training test (1 epoch, batch size 2) for the `attention_fusion` model.
```bash
./run_env.sh train.py --model-type attention_fusion --epochs 1 --batch-size 2 --experiment-name test_attention_fusion
```
