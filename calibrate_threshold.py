"""
calibrate_threshold.py — Platform-equivalent validation metrics for a trained experiment.

Loads the best checkpoint for a given experiment, runs inference on the same
validation split used during training (20% hold-out, random_state=42), and reports:

  1. Hard binary IoU per abundance channel at threshold 0.5  (matches platform exactly)
  2. Threshold scan 0.05–0.95 per channel to find the IoU-maximising threshold
     (diagnostic only — the platform always thresholds submitted values at 0.5)
  3. Masked RMSE_B and RMSE_V in physical metres (matches platform formula)
  4. Proxy score using platform weights and C=3.9

Usage:
    ./run_env.sh calibrate_threshold.py --experiment-name 4A_hook
"""

import config
import os
import re
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from core.model import build_model
from core.dataset import (
    find_triple_file_pairs,
    find_file_pairs,
    Emb2HeightsDataset,
    PixelEmbeddingDataset,
    LatentTokenDataset,
    HEIGHT_NORM_CONSTANT,
)

# --- Constants matching the platform formula ---
PLATFORM_C = 3.9          # RMSE above this scores zero on the platform
VAL_SPLIT   = 0.2
RANDOM_SEED = 42

# --- Device ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# --- Directory maps (train-split inputs, same as train.py) ---
PIXEL_DIR_MAP = {
    "tessera":      config.TESSERA_DIR,
    "alpha_earth":  config.ALPHA_EARTH_DIR,
}

PATCH_DIR_MAP = {
    "terramind_s1": config.TERRAMIND_S1_DIR,
    "terramind_s2": config.TERRAMIND_S2_DIR,
    "thor_s1":      config.THOR_S1_DIR,
    "thor_s2":      config.THOR_S2_DIR,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute platform-equivalent validation metrics for a trained experiment."
    )
    parser.add_argument(
        "--experiment-name", type=str, required=True,
        help="Name of the experiment (sub-directory under SHARED_RUNS_DIR).",
    )
    parser.add_argument(
        "--pixel-inputs", type=str, default=None,
        help="Override PIXEL_INPUTS from training_params.txt (e.g. alpha_earth).",
    )
    parser.add_argument(
        "--patch-inputs", type=str, default=None,
        help="Override PATCH_INPUTS from training_params.txt (e.g. terramind_s1,terramind_s2).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes.",
    )
    return parser.parse_args()


def load_experiment_params(exp_dir):
    """Parse training_params.txt into a dict (keys uppercased)."""
    params_path = os.path.join(exp_dir, "training_params.txt")
    params = {}
    if not os.path.exists(params_path):
        raise RuntimeError(f"training_params.txt not found in {exp_dir}")
    with open(params_path) as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                params[key.strip().upper()] = value.strip()
    return params


def resolve_dirs(input_str, name_map):
    """Resolve comma-separated embedding names to directory paths."""
    input_str = input_str.strip().lower()
    if input_str == "all":
        return list(name_map.values())
    paths = []
    for item in input_str.split(","):
        item = item.strip()
        if item in name_map:
            paths.append(name_map[item])
        elif os.path.isdir(item):
            paths.append(item)
        else:
            raise ValueError(f"Unknown embedding name or path: '{item}'")
    return paths


def find_model_path(exp_dir):
    """Return best available checkpoint path (same logic as predict.py)."""
    p1 = os.path.join(exp_dir, "model_best_e1.pth")
    p2 = os.path.join(exp_dir, "model_best.pth")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    raise RuntimeError(f"No model checkpoint found in {exp_dir}")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def hard_iou(pred_probs, targets, threshold):
    """
    Compute hard binary IoU for a single channel.

    pred_probs : 1-D tensor of predicted probabilities (sigmoid output, 0-1)
    targets    : 1-D tensor of ground-truth fractions (0-1)
    threshold  : scalar float — applied to pred_probs
    Returns    : scalar float IoU
    """
    pred_bin   = (pred_probs > threshold).float()
    target_bin = (targets    > 0.5      ).float()
    intersection = (pred_bin * target_bin).sum()
    union        = (pred_bin + target_bin - pred_bin * target_bin).sum()
    return (intersection / (union + 1e-8)).item()


def masked_rmse_physical(pred_height_norm, target_height_norm, mask):
    """
    Compute RMSE in physical metres over pixels where mask is True.

    pred_height_norm / target_height_norm : tensors in [0, ~1.5] (divided by HEIGHT_NORM_CONSTANT)
    mask : bool tensor
    Returns scalar float (metres), or 0.0 if mask is empty.
    """
    if mask.sum() == 0:
        return 0.0
    p = pred_height_norm[mask] * HEIGHT_NORM_CONSTANT
    t = target_height_norm[mask] * HEIGHT_NORM_CONSTANT
    return torch.sqrt(((p - t) ** 2).mean()).item()


def proxy_score(iou_b, iou_v, iou_w, rmse_b, rmse_v, C=PLATFORM_C):
    """Platform proxy score (higher is better)."""
    q_b = max(0.0, 1.0 - rmse_b / C)
    q_v = max(0.0, 1.0 - rmse_v / C)
    return (0.25 * iou_b
          + 0.15 * iou_v
          + 0.15 * iou_w
          + 0.25 * q_b
          + 0.20 * q_v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    exp_dir = os.path.join(config.SHARED_RUNS_DIR, args.experiment_name)
    if not os.path.isdir(exp_dir):
        raise RuntimeError(f"Experiment directory not found: {exp_dir}")

    params = load_experiment_params(exp_dir)
    model_type   = params.get("MODEL_TYPE", "decoder_residual").lower()
    patch_size   = int(params.get("PATCH_SIZE", "256"))

    # Resolve pixel/patch input names (CLI takes priority over training_params.txt)
    pixel_inputs_str = args.pixel_inputs or params.get("PIXEL_INPUTS", "")
    patch_inputs_str = args.patch_inputs or params.get("PATCH_INPUTS", "")

    # Fallback: older runs stored "Pixel:X_Patch:Y" inside TRAIN_EMBEDDINGS_DIR
    if model_type in ("attention_fusion", "ynet_attention_fusion"):
        if not pixel_inputs_str or not patch_inputs_str:
            combined = params.get("TRAIN_EMBEDDINGS_DIR", "")
            m = re.match(r"Pixel:(.+?)_Patch:(.+)", combined)
            if m:
                pixel_inputs_str = m.group(1).strip()
                patch_inputs_str = m.group(2).strip()
                print(f"  (Fallback) Parsed from TRAIN_EMBEDDINGS_DIR: "
                      f"pixel='{pixel_inputs_str}' patch='{patch_inputs_str}'")
            else:
                raise RuntimeError(
                    f"Cannot determine pixel/patch inputs for model '{model_type}'. "
                    f"Expected PIXEL_INPUTS and PATCH_INPUTS in training_params.txt, "
                    f"or TRAIN_EMBEDDINGS_DIR in format 'Pixel:X_Patch:Y'. Got: '{combined}'"
                )

    print(f"\n=== Calibration: {args.experiment_name} ===")
    print(f"Model type : {model_type}")
    print(f"Patch size : {patch_size}")
    print(f"Device     : {DEVICE}")

    # ------------------------------------------------------------------
    # 1. Reconstruct the EXACT validation split from training
    # ------------------------------------------------------------------
    print("\n--- Building validation split (same seed as training) ---")

    if model_type in ("attention_fusion", "ynet_attention_fusion"):
        pixel_dirs  = resolve_dirs(pixel_inputs_str, PIXEL_DIR_MAP)
        patch_dirs  = resolve_dirs(patch_inputs_str, PATCH_DIR_MAP)
        targets_dir = config.LABELS_DIR

        all_triplets = find_triple_file_pairs(pixel_dirs, patch_dirs, targets_dir)
        if not all_triplets:
            raise RuntimeError(
                f"No training triplets found. "
                f"pixel_dirs={pixel_dirs}, patch_dirs={patch_dirs}, targets_dir={targets_dir}"
            )
        print(f"Total matched triplets : {len(all_triplets)}")

        _, val_triplets = train_test_split(
            all_triplets, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
        print(f"Validation triplets    : {len(val_triplets)}")

        val_ds = Emb2HeightsDataset(
            val_triplets,
            patch_size=patch_size,
            scale_factor=16,
            is_train=False,   # centre-crop, no augmentation
            cache_in_memory=False,
            augment=False,
        )

        # Sample first item to get channel counts
        sample = val_ds[0]
        pixel_channels = sample["pixel_emb"].shape[0]
        patch_channels = sample["patch_emb"].shape[0]
        n_channels     = pixel_channels

        val_loader = DataLoader(
            val_ds,
            batch_size=1,       # one sample at a time — safe on any GPU
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Load model
        model_path = find_model_path(exp_dir)
        model, selected_model = build_model(
            model_type,
            n_channels=n_channels,
            n_classes=4,
            pixel_channels=pixel_channels,
            patch_channels=patch_channels,
        )
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Loaded : {selected_model} from {model_path}")
        print(f"         pixel_channels={pixel_channels}, patch_channels={patch_channels}")

        def run_batch(batch):
            pixel_emb = batch["pixel_emb"].to(DEVICE)
            patch_emb = batch["patch_emb"].to(DEVICE)
            targets   = batch["target"].to(DEVICE)
            with torch.no_grad():
                outputs = model(pixel_emb, patch_emb)
            return outputs, targets

    else:
        # Simpler models (lightunet / decoder_residual)
        if model_type == "lightunet":
            emb_dir = config.TESSERA_DIR
        else:
            emb_dir = config.TERRAMIND_S1_DIR
        targets_dir = config.LABELS_DIR

        all_pairs = find_file_pairs(emb_dir, targets_dir)
        if not all_pairs:
            raise RuntimeError(
                f"No training pairs found. emb_dir={emb_dir}, targets_dir={targets_dir}"
            )
        print(f"Total matched pairs : {len(all_pairs)}")

        _, val_pairs = train_test_split(
            all_pairs, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
        print(f"Validation pairs    : {len(val_pairs)}")

        if model_type == "lightunet":
            val_ds = PixelEmbeddingDataset(val_pairs, patch_size=patch_size, is_train=False)
        else:
            val_ds = LatentTokenDataset(val_pairs, patch_size=patch_size, scale_factor=16, is_train=False)

        sample_img, _ = val_ds[0]
        n_channels = sample_img.shape[0]

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model_path = find_model_path(exp_dir)
        model, selected_model = build_model(model_type, n_channels=n_channels, n_classes=4)
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Loaded : {selected_model} from {model_path}")
        print(f"         n_channels={n_channels}")

        def run_batch(batch):
            imgs, targets = batch
            imgs    = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            with torch.no_grad():
                outputs = model(imgs)
            return outputs, targets

    # ------------------------------------------------------------------
    # 2. Run inference and accumulate per-pixel predictions + targets
    # ------------------------------------------------------------------
    print("\n--- Running inference on validation set ---")

    # We accumulate flattened per-pixel values to keep memory manageable.
    # Channels 0-2: abundance fractions; channel 3: normalised height.
    all_pred_abund = [[] for _ in range(3)]   # channels 0, 1, 2
    all_tgt_abund  = [[] for _ in range(3)]
    all_pred_height = []                        # channel 3 (normalised)
    all_tgt_height  = []

    for batch in tqdm(val_loader, desc="Validating"):
        outputs, targets = run_batch(batch)

        # Align spatial size (in case model output != target, same as train.py)
        if targets.shape[-2:] != outputs.shape[-2:]:
            targets = F.interpolate(
                targets, size=outputs.shape[-2:], mode='bilinear', align_corners=False
            )

        # Sigmoid abundance channels — the model outputs logits for ch 0-2
        pred_abund = torch.sigmoid(outputs[:, :3])  # [B, 3, H, W]
        tgt_abund  = targets[:, :3]                 # [B, 3, H, W]

        # Height channel — kept as linear normalised value
        pred_h = outputs[:, 3]   # [B, H, W]
        tgt_h  = targets[:, 3]   # [B, H, W]

        # Flatten and move to CPU
        for c in range(3):
            all_pred_abund[c].append(pred_abund[:, c].reshape(-1).cpu())
            all_tgt_abund[c].append(tgt_abund[:, c].reshape(-1).cpu())
        all_pred_height.append(pred_h.reshape(-1).cpu())
        all_tgt_height.append(tgt_h.reshape(-1).cpu())

    # Concatenate everything into single tensors
    pred_abund_flat = [torch.cat(all_pred_abund[c]) for c in range(3)]
    tgt_abund_flat  = [torch.cat(all_tgt_abund[c])  for c in range(3)]
    pred_h_flat     = torch.cat(all_pred_height)
    tgt_h_flat      = torch.cat(all_tgt_height)

    print(f"Total pixels accumulated: {pred_h_flat.numel():,}")

    # ------------------------------------------------------------------
    # 3. Platform-equivalent metrics at threshold 0.5
    # ------------------------------------------------------------------
    print("\n--- Platform-equivalent validation metrics (threshold = 0.5) ---")

    iou_b_05 = hard_iou(pred_abund_flat[0], tgt_abund_flat[0], 0.5)
    iou_v_05 = hard_iou(pred_abund_flat[1], tgt_abund_flat[1], 0.5)
    iou_w_05 = hard_iou(pred_abund_flat[2], tgt_abund_flat[2], 0.5)

    # RMSE masked by GT building / vegetation pixels (GT abundance > 0.5)
    mask_build = tgt_abund_flat[0] > 0.5
    mask_veg   = tgt_abund_flat[1] > 0.5
    rmse_b = masked_rmse_physical(pred_h_flat, tgt_h_flat, mask_build)
    rmse_v = masked_rmse_physical(pred_h_flat, tgt_h_flat, mask_veg)

    score = proxy_score(iou_b_05, iou_v_05, iou_w_05, rmse_b, rmse_v)

    print()
    print(f"  IoU_B  : {iou_b_05:.4f}   (platform weight 25%)")
    print(f"  IoU_V  : {iou_v_05:.4f}   (platform weight 15%)")
    print(f"  IoU_W  : {iou_w_05:.4f}   (platform weight 15%)")
    print(f"  RMSE_B : {rmse_b:.4f} m   (platform weight 25%, C={PLATFORM_C})")
    print(f"  RMSE_V : {rmse_v:.4f} m   (platform weight 20%, C={PLATFORM_C})")
    print(f"  --")
    print(f"  Proxy score : {score:.4f}")
    print()
    print("  Mask coverage:")
    print(f"    Building pixels (GT > 0.5): {mask_build.sum().item():,} / {mask_build.numel():,} "
          f"({100*mask_build.float().mean().item():.1f}%)")
    print(f"    Vegetation pixels (GT > 0.5): {mask_veg.sum().item():,} / {mask_veg.numel():,} "
          f"({100*mask_veg.float().mean().item():.1f}%)")

    # ------------------------------------------------------------------
    # 4. Threshold scan — find IoU-maximising threshold per channel
    #    (diagnostic: platform always uses 0.5 on our submitted floats)
    # ------------------------------------------------------------------
    print("\n--- Threshold scan (0.05 – 0.95, step 0.01) ---")
    print("    (Diagnostic only: the platform fixes threshold at 0.5 on submitted values)")

    thresholds   = np.arange(0.05, 0.96, 0.01)
    channel_names = ["Building", "Vegetation", "Water"]

    optimal_thresholds = {}
    optimal_ious       = {}

    for c, name in enumerate(channel_names):
        best_thresh = 0.5
        best_iou    = hard_iou(pred_abund_flat[c], tgt_abund_flat[c], 0.5)
        for th in thresholds:
            iou = hard_iou(pred_abund_flat[c], tgt_abund_flat[c], float(th))
            if iou > best_iou:
                best_iou    = iou
                best_thresh = float(th)

        default_iou = hard_iou(pred_abund_flat[c], tgt_abund_flat[c], 0.5)
        delta       = best_iou - default_iou
        sign        = "+" if delta >= 0 else ""
        print(f"  {name:10s}: optimal threshold = {best_thresh:.2f}  "
              f"IoU = {best_iou:.4f}  (default 0.5 -> {default_iou:.4f}, {sign}{delta:.4f})")

        optimal_thresholds[name.lower()] = round(best_thresh, 2)
        optimal_ious[name.lower()]       = round(best_iou, 4)

    # ------------------------------------------------------------------
    # 5. Save optimal thresholds
    # ------------------------------------------------------------------
    out_path = os.path.join(exp_dir, "optimal_thresholds.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "building":   optimal_thresholds["building"],
                "vegetation": optimal_thresholds["vegetation"],
                "water":      optimal_thresholds["water"],
            },
            f,
            indent=2,
        )
    print(f"\nOptimal thresholds saved to: {out_path}")

    # ------------------------------------------------------------------
    # 6. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PLATFORM-EQUIVALENT VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Experiment : {args.experiment_name}")
    print(f"  Model      : {selected_model}")
    print(f"  Val samples: {len(val_ds)}")
    print()
    print(f"  IoU_B  = {iou_b_05:.4f}")
    print(f"  IoU_V  = {iou_v_05:.4f}")
    print(f"  IoU_W  = {iou_w_05:.4f}")
    print(f"  RMSE_B = {rmse_b:.4f} m")
    print(f"  RMSE_V = {rmse_v:.4f} m")
    print()
    print(f"  Proxy score = {score:.4f}")
    print()
    print("  Optimal thresholds (diagnostic):")
    for name in channel_names:
        k = name.lower()
        print(f"    {name:10s}: {optimal_thresholds[k]:.2f}  "
              f"(best IoU {optimal_ious[k]:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
