# Walkthrough: Attention-Gated Skip Connections Fusion

We have implemented the Attention-Gated multi-modal fusion architecture (Option 2A) to combine pixel-level high-resolution optical embeddings (Tessera, AlphaEarth) with patch-level low-resolution semantic embeddings (TerraMind S1/S2, Thor S1/S2).

## Dynamic Multi-Input Selection & Concatenation

We added support to dynamically load and stack multiple friendly input names or paths via command-line arguments without having to edit the code.

### 1. Available Embeddings Map
Friendly names are mapped directly to directories in [config.py](file:///mnt/head/users/bassam/src/geofmchal/config.py):
* **Pixel-level (256x256)**:
  * `tessera` (128 channels)
  * `alpha_earth` (64 channels)
* **Patch-level (16x16)**:
  * `terramind_s1` (768 channels)
  * `terramind_s2` (768 channels)
  * `thor_s1` (768 channels)
  * `thor_s2` (768 channels)

### 2. Multi-Input Dataset Setup
* Refactored [dataset.py](file:///mnt/head/users/bassam/src/geofmchal/core/dataset.py):
  * `find_triple_file_pairs` finds matching sets across all selected pixel and patch folders by intersecting normalized core IDs.
  * `Emb2HeightsDataset` dynamically loads, pads, and concatenates all specified input embeddings along the channel dimension.
* Added new command line options in [train.py](file:///mnt/head/users/bassam/src/geofmchal/train.py):
  * `--pixel-inputs`: Comma-separated names (e.g. `tessera,alpha_earth`) or `all`.
  * `--patch-inputs`: Comma-separated names (e.g. `terramind_s1,thor_s2`) or `all`.

### 3. Model Adaptation
* Adapted `build_model` in [model.py](file:///mnt/head/users/bassam/src/geofmchal/core/model.py) to receive dynamic `pixel_channels` and `patch_channels` values computed directly from the dataset.

---

## 4. Configurable Runs Directory (`RUNS_DIR`) and Symlink
* Relocated all existing training folders and predictions from the local directory to:
  `/mnt/head/users/bassam/data/geofmdata/runs`
* Replaced the local `runs` directory with a symbolic link pointing to the new location.
* Updated [.gitignore](file:///mnt/head/users/bassam/src/geofmchal/.gitignore) (`runs` instead of `runs/`) to ensure the symlink remains untracked by Git.
* Modified [train.py](file:///mnt/head/users/bassam/src/geofmchal/train.py) and [predict.py](file:///mnt/head/users/bassam/src/geofmchal/predict.py) to read `config.RUNS_DIR` as their default directory for storing training checkpoints and predicted outputs, respectively.

---

## 5. Side-by-Side Prediction Viewer Script
Created a new script [view_predictions.py](file:///mnt/head/users/bassam/src/geofmchal/view_predictions.py) to visualize model outputs side-by-side with original test embeddings:
* Automatically pairs predictions (`.npy` files) in `--predictions-dir` with original test `.tif` embeddings in `--pixel-inputs` matching their `core_id`.
* Plots a 5-panel comparison including:
  1. Input false-color embedding visualization.
  2. Predicted % Building.
  3. Predicted % Vegetation.
  4. Predicted % Water.
  5. Predicted physical Height (m) with terrain colormap and colorbar scales.

---

## Verification Runs

### Run 1: Generating Predictions
```bash
./run_env.sh predict.py --model-type attention_fusion --experiment-name test_attention_fusion --pixel-inputs tessera --patch-inputs terramind_s1 --max-samples 3 --base-dir ./runs
```
* **Output**:
  ```
  🔍 Found 946 matched test pairs.
  Loaded model: attention_fusion from ./runs/test_attention_fusion/model_best_e1.pth (pixel channels=128, patch channels=768)
  Running inference on 3 samples...
  Predictions saved to: ./runs/test_attention_fusion/predictions
  ```

### Run 2: Visualizing Predictions
```bash
./run_env.sh view_predictions.py --predictions-dir ./runs/test_attention_fusion/predictions --pixel-inputs tessera --num-samples 3
```
* **Output**:
  ```
  🔍 Found 3 prediction files.
  📂 Searching for matching test files in: /mnt/head/users/bassam/data/geofmdata/embed2heights/data/test/tessera_test_emb
  ✅ Successfully paired 3 / 3 files.
  🎨 Generating 3 visualizations...
  🎉 Visualizations saved to: ./runs/test_attention_fusion/predictions/visualizations
  ```
Generated files:
* `./runs/test_attention_fusion/predictions/visualizations/viz_3001_BE.png`
* `./runs/test_attention_fusion/predictions/visualizations/viz_3002_BE.png`
* `./runs/test_attention_fusion/predictions/visualizations/viz_3003_BE.png`
