import config
import os
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# --- IMPORT FROM CORE MODULES ---
from core.model import build_model
from core.dataset import (
    PixelEmbeddingDataset,
    LatentTokenDataset,
    Emb2HeightsDataset,
    find_file_pairs,
    find_triple_file_pairs,
    HEIGHT_NORM_CONSTANT
)
from core.losses import ImprovedCompositeLoss

# --- 1. EXPERIMENT TRACKING ---
EXPERIMENT_NAME = "terramid_run02/"
BASE_DIR = config.SHARED_RUNS_DIR
EXP_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)
VIZ_OUTPUT_DIR = os.path.join(EXP_DIR, "visualizations")

# Paths for saving models and plots
BEST_MODEL_PATH = os.path.join(EXP_DIR, "model_best.pth")
LAST_MODEL_PATH = os.path.join(EXP_DIR, "model_last.pth")
LOSS_CURVE_PATH = os.path.join(EXP_DIR, "loss_curve.png")
CONFIG_LOG_PATH = os.path.join(EXP_DIR, "training_params.txt")

# --- 2. CONFIGURATION ---

TRAIN_EMBEDDINGS_DIR = None
TRAIN_TARGETS_DIR = None

BATCH_SIZE = config.BATCH_SIZE
PATCH_SIZE = config.PATCH_SIZE
EPOCHS = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4  # L2 Regularization
VAL_SPLIT = 0.2
LAMBDAS = [1.0, 0.5, 0.5, 2.0]  # [MAE, SSIM, Gradient, Structure/Tversky]
RANDOM_SEED = 42
MODEL_TYPE = "auto"  # one of: auto, lightunet, decoder_residual, attention_fusion

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


PIXEL_DIR_MAP = {
    "tessera": config.TESSERA_DIR,
    "alpha_earth": config.ALPHA_EARTH_DIR,
}

PATCH_DIR_MAP = {
    "terramind_s1": config.TERRAMIND_S1_DIR,
    "terramind_s2": config.TERRAMIND_S2_DIR,
    "thor_s1": config.THOR_S1_DIR,
    "thor_s2": config.THOR_S2_DIR,
}


def resolve_dirs(input_str, name_map):
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
            raise ValueError(f"Unknown input embedding name or path: '{item}'")
    return paths


def save_experiment_config(pixel_inputs=None, patch_inputs=None):
    """Logs all hyperparameters to a text file in the experiment folder."""
    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

    with open(CONFIG_LOG_PATH, "w") as f:
        f.write(f"--- EXPERIMENT: {EXPERIMENT_NAME} ---\n")
        f.write(f"OUTPUT_DIR: {BASE_DIR}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"PATCH_SIZE: {PATCH_SIZE}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"WEIGHT_DECAY: {WEIGHT_DECAY}\n")
        f.write(f"LOSS LAMBDAS: {LAMBDAS}\n")
        f.write(f"MODEL_TYPE: {MODEL_TYPE}\n")
        if MODEL_TYPE == "attention_fusion":
            f.write(f"PIXEL_INPUTS: {pixel_inputs}\n")
            f.write(f"PATCH_INPUTS: {patch_inputs}\n")
        else:
            f.write(f"TRAIN_EMBEDDINGS_DIR: {TRAIN_EMBEDDINGS_DIR}\n")
        f.write(f"TRAIN_TARGETS_DIR: {TRAIN_TARGETS_DIR}\n")
        f.write(f"VAL_SPLIT: {VAL_SPLIT}\n")
        f.write(f"OPTIMIZER: AdamW\n")
        f.write(f"SCHEDULER: ReduceLROnPlateau (factor=0.5, patience=2)\n")
        f.write(f"GRADIENT CLIPPING: max_norm=1.0\n")
    print(f"📁 Created experiment folder: {EXP_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train emb2heights baseline models")
    parser.add_argument("--model-type", type=str, default=MODEL_TYPE, choices=["auto", "lightunet", "decoder_residual", "attention_fusion"])
    parser.add_argument("--output-dir", type=str, default=BASE_DIR)
    parser.add_argument("--train-embeddings-dir", type=str, default=None, help="Path to training embeddings. Defaults to path in config.py based on model-type.")
    parser.add_argument("--train-targets-dir", type=str, default=None, help="Path to training targets. Defaults to path in config.py.")
    parser.add_argument("--pixel-inputs", type=str, default="tessera", help="Comma-separated pixel embeddings to concatenate (e.g. tessera,alpha_earth, or 'all').")
    parser.add_argument("--patch-inputs", type=str, default="terramind_s1", help="Comma-separated patch embeddings to concatenate (e.g. terramind_s1,thor_s2, or 'all').")
    parser.add_argument("--experiment-name", type=str, default=EXPERIMENT_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes for DataLoader.")
    parser.add_argument("--cache-in-memory", action="store_true", help="Cache dataset samples in CPU RAM on first load to speed up subsequent epochs.")
    return parser.parse_args()


def calc_leaderboard_metrics(preds, targets):
    """Calculate continuous Soft IoU and Masked RMSE (in physical meters)."""
    # Bound Abundance logits between 0 and 1
    preds_abund = torch.sigmoid(preds[:, :3])
    
    def soft_iou(p, t):
        intersection = (p * t).sum()
        union = (p + t - (p * t)).sum()
        return (intersection / (union + 1e-8)).item()

    iou_build = soft_iou(preds_abund[:, 0], targets[:, 0])
    iou_veg = soft_iou(preds_abund[:, 1], targets[:, 1])
    iou_water = soft_iou(preds_abund[:, 2], targets[:, 2])

    def masked_rmse(p_height, t_height, mask):
        if mask.sum() == 0:
            return 0.0
        p_h = p_height[mask] * HEIGHT_NORM_CONSTANT
        t_h = t_height[mask] * HEIGHT_NORM_CONSTANT
        return torch.sqrt(((p_h - t_h) ** 2).mean()).item()

    mask_build = targets[:, 0] > 0.1
    # Channel 3 is kept linear/untouched for Masked RMSE
    rmse_h_build = masked_rmse(preds[:, 3], targets[:, 3], mask_build)

    mask_veg = targets[:, 1] > 0.1
    rmse_h_veg = masked_rmse(preds[:, 3], targets[:, 3], mask_veg)

    return iou_build, iou_veg, iou_water, rmse_h_build, rmse_h_veg


def align_target_to_output(target, output):
    if target.shape[-2:] != output.shape[-2:]:
        return F.interpolate(target, size=output.shape[-2:], mode='bilinear', align_corners=False)
    return target


def visualize_results(model, dataset, num_samples=3):
    """Generates sample visualizations from the dataset."""
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    target_names = ["% Building", "% Vegetation", "% Water", "nDSM Height (m)"]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            if isinstance(sample, dict):
                pixel_emb = sample["pixel_emb"].unsqueeze(0).to(DEVICE)
                patch_emb = sample["patch_emb"].unsqueeze(0).to(DEVICE)
                target_tensor = sample["target"]
                output_batch = model(pixel_emb, patch_emb)
            else:
                img_tensor, target_tensor = sample
                input_batch = img_tensor.unsqueeze(0).to(DEVICE)
                output_batch = model(input_batch)

            target_batch = align_target_to_output(target_tensor.unsqueeze(0).to(DEVICE), output_batch)

            pred = output_batch.squeeze().cpu().numpy()
            true = target_batch.squeeze().cpu().numpy()

            # UN-NORMALIZE HEIGHT FOR VISUALIZATION
            pred[3] = pred[3] * HEIGHT_NORM_CONSTANT
            true[3] = true[3] * HEIGHT_NORM_CONSTANT

            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            for c in range(4):
                vmin, vmax = (0, 1) if c < 3 else (0, HEIGHT_NORM_CONSTANT)
                axes[0, c].imshow(true[c], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[0, c].set_title(f"True {target_names[c]}")
                axes[0, c].axis('off')

                axes[1, c].imshow(pred[c], cmap='viridis', vmin=vmin, vmax=vmax)
                axes[1, c].set_title(f"Pred {target_names[c]}")
                axes[1, c].axis('off')

            plt.suptitle(f"{model.__class__.__name__} Prediction (Sample {i})")
            plt.tight_layout()
            plt.savefig(os.path.join(VIZ_OUTPUT_DIR, f"viz_{i}.png"))
            plt.close()


def main():
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    global BASE_DIR, EXPERIMENT_NAME, EXP_DIR, VIZ_OUTPUT_DIR
    global BEST_MODEL_PATH, LAST_MODEL_PATH, LOSS_CURVE_PATH, CONFIG_LOG_PATH
    global TRAIN_EMBEDDINGS_DIR, TRAIN_TARGETS_DIR, TEST_TARGETS_DIR
    global MODEL_TYPE, EPOCHS, BATCH_SIZE, PATCH_SIZE

    args = parse_args()
    MODEL_TYPE = args.model_type
    BASE_DIR = args.output_dir
    EXPERIMENT_NAME = args.experiment_name
    BATCH_SIZE = args.batch_size
    PATCH_SIZE = args.patch_size
    EPOCHS = args.epochs

    # Resolve directories using config.py
    if MODEL_TYPE == "attention_fusion":
        pixel_dirs = resolve_dirs(args.pixel_inputs, PIXEL_DIR_MAP)
        patch_dirs = resolve_dirs(args.patch_inputs, PATCH_DIR_MAP)
        targets_dir = args.train_targets_dir if args.train_targets_dir is not None else config.LABELS_DIR
        TRAIN_EMBEDDINGS_DIR = f"Pixel:{args.pixel_inputs}_Patch:{args.patch_inputs}"
        TRAIN_TARGETS_DIR = targets_dir
    else:
        if args.train_embeddings_dir is None:
            if MODEL_TYPE == "lightunet":
                TRAIN_EMBEDDINGS_DIR = config.TESSERA_DIR
            else:
                TRAIN_EMBEDDINGS_DIR = config.TERRAMIND_S1_DIR
        else:
            TRAIN_EMBEDDINGS_DIR = args.train_embeddings_dir

        if args.train_targets_dir is None:
            TRAIN_TARGETS_DIR = config.LABELS_DIR
        else:
            TRAIN_TARGETS_DIR = args.train_targets_dir

    EXP_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)
    VIZ_OUTPUT_DIR = os.path.join(EXP_DIR, "visualizations")
    BEST_MODEL_PATH = os.path.join(EXP_DIR, "model_best_e1.pth")
    LAST_MODEL_PATH = os.path.join(EXP_DIR, "model_last.pth")
    LOSS_CURVE_PATH = os.path.join(EXP_DIR, "loss_curve.png")
    CONFIG_LOG_PATH = os.path.join(EXP_DIR, "training_params.txt")

    if MODEL_TYPE == "attention_fusion":
        save_experiment_config(pixel_inputs=args.pixel_inputs, patch_inputs=args.patch_inputs)
    else:
        save_experiment_config()

    print("--- 1. Data Setup ---")
    if MODEL_TYPE == "attention_fusion":
        all_train_triplets = find_triple_file_pairs(pixel_dirs, patch_dirs, targets_dir)
        print(f"   >> Total matched triplets found: {len(all_train_triplets)}")
        if len(all_train_triplets) == 0:
            raise ValueError(
                "No training triplets found. "
                f"pixel_dirs='{pixel_dirs}', patch_dirs='{patch_dirs}', targets_dir='{targets_dir}'."
            )
        train_triplets, val_triplets = train_test_split(
            all_train_triplets, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
        train_ds = Emb2HeightsDataset(train_triplets, patch_size=PATCH_SIZE, scale_factor=16, is_train=True, cache_in_memory=args.cache_in_memory)
        val_ds = Emb2HeightsDataset(val_triplets, patch_size=PATCH_SIZE, scale_factor=16, is_train=False, cache_in_memory=args.cache_in_memory)
        print(f"   >> Train split triplets: {len(train_ds)}")
        print(f"   >> Val split triplets:   {len(val_ds)}")
        
        sample = train_ds[0]
        print(f"   >> Pixel Embedding shape: {sample['pixel_emb'].shape}")
        print(f"   >> Patch Embedding shape: {sample['patch_emb'].shape}")
        print(f"   >> Target Label shape:    {sample['target'].shape}")
        pixel_channels = sample['pixel_emb'].shape[0]
        patch_channels = sample['patch_emb'].shape[0]
        n_channels = pixel_channels
    else:
        all_train_pairs = find_file_pairs(TRAIN_EMBEDDINGS_DIR, TRAIN_TARGETS_DIR)
        print(f"   >> Total matched pairs found: {len(all_train_pairs)}")
        if len(all_train_pairs) == 0:
            raise ValueError(
                "No training (embedding, label) pairs found. "
                f"train_embeddings_dir='{TRAIN_EMBEDDINGS_DIR}', "
                f"train_targets_dir='{TRAIN_TARGETS_DIR}'."
            )
        train_pairs, val_pairs = train_test_split(
            all_train_pairs, test_size=VAL_SPLIT, random_state=RANDOM_SEED
        )
        if MODEL_TYPE == "lightunet":
            train_ds = PixelEmbeddingDataset(train_pairs, patch_size=PATCH_SIZE, is_train=True)
            val_ds = PixelEmbeddingDataset(val_pairs, patch_size=PATCH_SIZE, is_train=False)
        else:
            train_ds = LatentTokenDataset(train_pairs, patch_size=PATCH_SIZE, scale_factor=16, is_train=True)
            val_ds = LatentTokenDataset(val_pairs, patch_size=PATCH_SIZE, scale_factor=16, is_train=False)
        
        print(f"   >> Train split pairs: {len(train_ds)}")
        print(f"   >> Val split pairs:   {len(val_ds)}")
        
        sample_img, sample_tar = train_ds[0]
        print(f"   >> Embedding shape: {sample_img.shape}")
        print(f"   >> Target shape:    {sample_tar.shape}")
        n_channels = sample_img.shape[0]
        pixel_channels = 128
        patch_channels = 768

    n_classes = 4

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=False
    )

    print("--- 2. Model Init ---")
    model, selected_model = build_model(MODEL_TYPE, n_channels, n_classes, pixel_channels=pixel_channels, patch_channels=patch_channels)
    model = model.to(DEVICE)
    print(f"Using model: {selected_model} (pixel channels={pixel_channels}, patch channels={patch_channels})")

    # NEW: AdamW with Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # NEW: Aggressive Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = ImprovedCompositeLoss(lambdas=LAMBDAS).to(DEVICE)

    print(f"Starting training on {DEVICE}...")

    train_losses, val_losses = [], []
    best_val_score = float('inf')
    best_epoch = 0
    best_metrics = None
    epoch_times = []
    total_start_time = time.time()

    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        train_samples_seen = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [train]", leave=True)
        for batch in train_pbar:
            optimizer.zero_grad()
            if isinstance(batch, dict):
                pixel_emb = batch["pixel_emb"].to(DEVICE)
                patch_emb = batch["patch_emb"].to(DEVICE)
                targets = batch["target"].to(DEVICE)
                outputs = model(pixel_emb, patch_emb)
                batch_size = pixel_emb.size(0)
            else:
                imgs, targets = batch
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                outputs = model(imgs)
                batch_size = imgs.size(0)

            loss, _, _, _, _ = criterion(outputs, targets)
            loss.backward()

            # NEW: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item() * batch_size
            train_samples_seen += batch_size
            train_avg = running_loss / max(1, train_samples_seen)
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{train_avg:.4f}")

        epoch_loss = running_loss / len(train_ds)
        train_losses.append(epoch_loss)

        # --- VALIDATION LOOP ---
        model.eval()
        val_running_loss = 0.0
        val_components = torch.zeros(4).to(DEVICE)
        val_metrics = torch.zeros(5).to(DEVICE)
        val_samples_seen = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [val]", leave=True)
            for batch in val_pbar:
                if isinstance(batch, dict):
                    pixel_emb = batch["pixel_emb"].to(DEVICE)
                    patch_emb = batch["patch_emb"].to(DEVICE)
                    targets = batch["target"].to(DEVICE)
                    outputs = model(pixel_emb, patch_emb)
                    batch_size = pixel_emb.size(0)
                else:
                    imgs, targets = batch
                    imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(imgs)
                    batch_size = imgs.size(0)

                loss, l_mae, l_ssim, l_grad, l_tversky = criterion(outputs, targets)
                val_running_loss += loss.item() * batch_size

                val_components[0] += l_mae * batch_size
                val_components[1] += l_ssim * batch_size
                val_components[2] += l_grad * batch_size
                val_components[3] += l_tversky * batch_size
                
                # Leaderboard metrics
                m_iou_b, m_iou_v, m_iou_w, m_rmse_b, m_rmse_v = calc_leaderboard_metrics(outputs, targets)
                val_metrics[0] += m_iou_b * batch_size
                val_metrics[1] += m_iou_v * batch_size
                val_metrics[2] += m_iou_w * batch_size
                val_metrics[3] += m_rmse_b * batch_size
                val_metrics[4] += m_rmse_v * batch_size
                
                val_samples_seen += batch_size
                val_avg_live = val_running_loss / max(1, val_samples_seen)
                val_pbar.set_postfix(avg=f"{val_avg_live:.4f}")

        epoch_val_loss = val_running_loss / len(val_ds)
        epoch_comp = val_components / len(val_ds)
        epoch_metrics = val_metrics / len(val_ds)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        val_mae = epoch_comp[0].item()
        val_tversky = epoch_comp[3].item()
        val_score = val_mae + (val_tversky * 2.0)

        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch + 1
            best_metrics = epoch_metrics
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   >> [Checkpoint] New best model saved! (Score: {val_score:.4f} | MAE: {val_mae:.4f} | Tversky: {val_tversky:.4f})")

        epoch_elapsed = time.time() - epoch_start_time
        epoch_times.append(epoch_elapsed)
        epoch_time_msg = f"   >> Epoch Time: {epoch_elapsed:.2f} seconds ({epoch_elapsed/60:.2f} minutes)\n"
        
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train: {epoch_loss:.4f} | Val: {epoch_val_loss:.4f}")
        print(epoch_time_msg, end="")
        print(
            f"   >> Val Breakdown: MAE:{epoch_comp[0]:.3f} | SSIM:{epoch_comp[1]:.3f} | Grad:{epoch_comp[2]:.3f} | Tversky:{epoch_comp[3]:.3f}")
        print(f"   >> Leaderboard: IOU_B: {epoch_metrics[0]:.4f} | IOU_V: {epoch_metrics[1]:.4f} | IOU_W: {epoch_metrics[2]:.4f} | RMSE_H_B: {epoch_metrics[3]:.4f} | RMSE_H_V: {epoch_metrics[4]:.4f}")

        # Append epoch time to params log file
        current_lr = optimizer.param_groups[0]['lr']
        with open(CONFIG_LOG_PATH, "a") as f:
            f.write(f"Epoch {epoch + 1} finished in {epoch_elapsed:.2f}s | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | LR: {current_lr:.6f} | IOU_B: {epoch_metrics[0]:.4f} | IOU_V: {epoch_metrics[1]:.4f} | IOU_W: {epoch_metrics[2]:.4f} | RMSE_B: {epoch_metrics[3]:.4f} | RMSE_V: {epoch_metrics[4]:.4f}\n")

        # Clean memory to avoid leaks/fragmentation across epochs
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start_time
    
    # Add Best Model Summary block
    if best_metrics is not None:
        summary_msg = (
            f"\n=== BEST MODEL SUMMARY ===\n"
            f"Best Epoch: {best_epoch}\n"
            f"Best Val Score (Tversky + 2*MAE): {best_val_score:.4f}\n"
            f"Leaderboard Metrics at Best Epoch -> IOU_B: {best_metrics[0]:.4f} | IOU_V: {best_metrics[1]:.4f} | IOU_W: {best_metrics[2]:.4f} | RMSE_B: {best_metrics[3]:.4f} | RMSE_V: {best_metrics[4]:.4f}\n"
        )
        print(summary_msg, end="")
        with open(CONFIG_LOG_PATH, "a") as f:
            f.write(summary_msg)

    print("--- 3. Saving & Visualizing ---")
    torch.save(model.state_dict(), LAST_MODEL_PATH)

    try:
        visualize_results(model, val_ds, num_samples=3)
        print("📁 Visualizations saved to:", VIZ_OUTPUT_DIR)
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f"Training Loss Curve ({EXPERIMENT_NAME})")
    plt.legend()
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()

if __name__ == "__main__":
    main()