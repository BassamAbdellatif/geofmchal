#!/usr/bin/env python3
"""
Viewer script for comparing test embeddings with corresponding predicted targets.
Generates a side-by-side visualization showing the input embedding, predicted building %,
vegetation %, water %, and physical height (m).
Automatically detects if ground truth label files exist for the paired core IDs and,
if so, displays a 2x5 comparison layout showing (Input, True Targets, Predicted Targets).
"""

import os
import argparse
import glob
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm
import config

# Friendly mappings for test inputs
PIXEL_TEST_MAP = {
    "tessera": config.TESSERA_TEST_DIR,
    "alpha_earth": config.ALPHA_EARTH_TEST_DIR,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize test/train/val embeddings side-by-side with their predictions."
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Path to the directory containing predicted .npy files.",
    )
    parser.add_argument(
        "--pixel-inputs",
        type=str,
        default="tessera",
        help="Friendly name ('tessera', 'alpha_earth') or path to pixel embeddings directory.",
    )
    parser.add_argument(
        "--targets-dir",
        type=str,
        default=config.LABELS_DIR,
        help="Path to directory containing label_*.tif ground truth files. Defaults to path in config.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the visualization plots. Defaults to <predictions-dir>/visualizations.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize (randomly selected).",
    )
    return parser.parse_args()


def extract_core_id(filename):
    """Extracts the core ID pattern (e.g. 1234_AB_2021) from a filename."""
    basename = os.path.basename(filename)
    match = re.search(r"(\d+_[A-Z]{2}_\d+)", basename)
    if match:
        return match.group(1)
    return os.path.splitext(basename)[0]


def find_matching_pixel_file(core_id, pixel_dir):
    """Finds a .tif file matching the core_id in the pixel embedding directory."""
    patterns = [
        os.path.join(pixel_dir, f"*{core_id}*.tif"),
        os.path.join(pixel_dir, f"*{core_id}*.tiff"),
        os.path.join(pixel_dir, "**", f"*{core_id}*.tif"),
        os.path.join(pixel_dir, "**", f"*{core_id}*.tiff"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def find_matching_target_file(core_id, targets_dir):
    """Finds a label .tif file matching the core_id in the targets directory."""
    if not targets_dir or not os.path.isdir(targets_dir):
        return None
    patterns = [
        os.path.join(targets_dir, f"label_*{core_id}*.tif"),
        os.path.join(targets_dir, f"*{core_id}*.tif"),
        os.path.join(targets_dir, "**", f"label_*{core_id}*.tif"),
        os.path.join(targets_dir, "**", f"*{core_id}*.tif"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def load_experiment_params(exp_dir):
    """Loads training parameters from training_params.txt in the experiment directory."""
    params_path = os.path.join(exp_dir, "training_params.txt")
    params = {}
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    params[key.strip().upper()] = value.strip()
    return params


def main():
    args = parse_args()

    # Try to auto-detect parameters from training_params.txt if available
    # We look for training_params.txt in predictions_dir's parent or grandparent directory
    parent_dir = os.path.dirname(args.predictions_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    params = {}
    for d in [args.predictions_dir, parent_dir, grandparent_dir]:
        p_path = os.path.join(d, "training_params.txt")
        if os.path.exists(p_path):
            params = load_experiment_params(d)
            print(f"🔄 Found training_params.txt in {d}")
            break

    pixel_inputs = args.pixel_inputs
    # Auto-load pixel_inputs from training_params.txt if user didn't explicitly customize it (i.e. default "tessera")
    if "PIXEL_INPUTS" in params and args.pixel_inputs == "tessera":
        pixel_inputs = params["PIXEL_INPUTS"]
        print(f"🔄 Auto-detected pixel_inputs from training_params.txt: '{pixel_inputs}'")

    # If pixel_inputs is a comma-separated list or "all", use the first item to search for inputs to visualize
    first_pixel = pixel_inputs.split(",")[0].strip()
    if first_pixel.lower() == "all":
        first_pixel = "tessera"  # Default to tessera if 'all'

    # Resolve search directories for the pixel input
    pixel_dir = first_pixel
    if pixel_dir.lower() in PIXEL_TEST_MAP:
        pixel_dir = PIXEL_TEST_MAP[pixel_dir.lower()]

    if not os.path.isdir(pixel_dir):
        if pixel_dir.lower() == "alpha_earth":
            pixel_dir = config.ALPHA_EARTH_DIR
        elif pixel_dir.lower() == "tessera":
            pixel_dir = config.TESSERA_DIR

    # Build list of directories to search in
    search_dirs = [pixel_dir]
    if first_pixel.lower() == "tessera":
        search_dirs.extend([config.TESSERA_TEST_DIR, config.TESSERA_DIR])
    elif first_pixel.lower() == "alpha_earth":
        search_dirs.extend([config.ALPHA_EARTH_TEST_DIR, config.ALPHA_EARTH_DIR])

    # De-duplicate while preserving order
    search_dirs = list(dict.fromkeys([d for d in search_dirs if d and os.path.isdir(d)]))

    if not search_dirs:
        raise ValueError(f"Pixel embeddings directory not found for: '{first_pixel}'")

    if not os.path.isdir(args.predictions_dir):
        raise ValueError(f"Predictions directory not found: '{args.predictions_dir}'")

    # Resolve output directory
    output_dir = args.output_dir or os.path.join(args.predictions_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Find prediction .npy files
    npy_files = glob.glob(os.path.join(args.predictions_dir, "*.npy"))
    if not npy_files:
        raise RuntimeError(f"No predicted .npy files found in '{args.predictions_dir}'")

    print(f"🔍 Found {len(npy_files)} prediction files.")
    print(f"📂 Searching matching input files in: {search_dirs}")

    # Pair predictions with inputs
    valid_pairs = []
    for npy_path in npy_files:
        core_id = extract_core_id(npy_path)
        tif_path = None
        for s_dir in search_dirs:
            tif_path = find_matching_pixel_file(core_id, s_dir)
            if tif_path:
                break
        if tif_path:
            valid_pairs.append((core_id, npy_path, tif_path))

    if not valid_pairs:
        raise RuntimeError(
            "Could not match any prediction .npy files to .tif files in "
            f"any search directory: {search_dirs}. Verify that core IDs match."
        )

    print(f"✅ Successfully paired {len(valid_pairs)} / {len(npy_files)} files.")

    # Select samples to visualize
    import random
    num_samples = min(args.num_samples, len(valid_pairs))
    selected_pairs = random.sample(valid_pairs, num_samples)

    print(f"🎨 Generating {num_samples} visualizations...")

    for core_id, npy_path, tif_path in tqdm(selected_pairs, desc="Visualizing"):
        # Load prediction
        pred = np.load(npy_path)  # Shape [4, H, W]
        # Load input embedding
        with rasterio.open(tif_path) as src:
            input_emb = src.read().astype(np.float32)

        # Handle shapes (prediction is center cropped)
        H_pred, W_pred = pred.shape[1], pred.shape[2]
        H_in, W_in = input_emb.shape[1], input_emb.shape[2]

        # Extract representative input image (average over first few channels or channels 0-2 if available)
        input_emb = np.nan_to_num(input_emb)
        if input_emb.shape[0] >= 3:
            # Create a false-color composite of the first 3 channels
            input_vis = input_emb[:3].transpose(1, 2, 0)
            # Normalize to [0, 1] for visualization
            input_vis = (input_vis - input_vis.min()) / max(1e-5, (input_vis.max() - input_vis.min()))
        else:
            input_vis = input_emb[0]

        # Crop input vis to match prediction shape if they differ
        if input_vis.shape[0] != H_pred or input_vis.shape[1] != W_pred:
            top = (input_vis.shape[0] - H_pred) // 2
            left = (input_vis.shape[1] - W_pred) // 2
            input_vis = input_vis[top:top + H_pred, left:left + W_pred]

        # Check if ground truth is available
        target_path = find_matching_target_file(core_id, args.targets_dir)

        if target_path and os.path.exists(target_path):
            # Load target label
            with rasterio.open(target_path) as src:
                target = src.read().astype(np.float32)
            target = np.nan_to_num(target)

            # Crop target to match prediction shape if they differ
            if target.shape[1] != H_pred or target.shape[2] != W_pred:
                top = (target.shape[1] - H_pred) // 2
                left = (target.shape[2] - W_pred) // 2
                target = target[:, top:top + H_pred, left:left + W_pred]

            # Generate 2x5 visualization (Row 0: Input & Truth, Row 1: Pred)
            fig, axes = plt.subplots(2, 5, figsize=(25, 10))

            # Column 0: Input Image
            axes[0, 0].imshow(input_vis)
            axes[0, 0].set_title("Input Image")
            axes[0, 0].axis("off")

            axes[1, 0].axis("off")  # Leave bottom-left empty

            colormaps = ["magma", "viridis", "Blues", "terrain"]
            titles = ["Building %", "Vegetation %", "Water %", "Height (m)"]

            for c in range(4):
                vmin, vmax = (0, 1) if c < 3 else (None, None)
                col_idx = c + 1

                # True Target
                im_true = axes[0, col_idx].imshow(target[c], cmap=colormaps[c], vmin=vmin, vmax=vmax)
                axes[0, col_idx].set_title(f"True {titles[c]}")
                axes[0, col_idx].axis("off")
                fig.colorbar(im_true, ax=axes[0, col_idx], fraction=0.046, pad=0.04)

                # Predicted Target
                im_pred = axes[1, col_idx].imshow(pred[c], cmap=colormaps[c], vmin=vmin, vmax=vmax)
                axes[1, col_idx].set_title(f"Pred {titles[c]}")
                axes[1, col_idx].axis("off")
                fig.colorbar(im_pred, ax=axes[1, col_idx], fraction=0.046, pad=0.04)

            plt.suptitle(f"Core ID: {core_id} (Comparison with Ground Truth)", fontsize=18, weight="bold")

        else:
            # 1x5 layout showing Input + Predictions
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))

            # Column 0: Input Image
            axes[0].imshow(input_vis)
            axes[0].set_title(f"Input ({os.path.basename(tif_path)[:20]}...)")
            axes[0].axis("off")

            colormaps = ["magma", "viridis", "Blues", "terrain"]
            titles = ["Building %", "Vegetation %", "Water %", "Height (m)"]

            for c in range(4):
                vmin, vmax = (0, 1) if c < 3 else (None, None)
                im = axes[c + 1].imshow(pred[c], cmap=colormaps[c], vmin=vmin, vmax=vmax)
                axes[c + 1].set_title(f"Pred {titles[c]}")
                axes[c + 1].axis("off")
                fig.colorbar(im, ax=axes[c + 1], fraction=0.046, pad=0.04)

            plt.suptitle(f"Core ID: {core_id}", fontsize=16, weight="bold")

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(output_dir, f"viz_{core_id}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"🎉 Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
