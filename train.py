import config
import os
import random
import time
import argparse
from contextlib import nullcontext
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# --- IMPORT FROM CORE MODULES ---
from core.model import build_model
from core.dataset import (
    PixelEmbeddingDataset,
    LatentTokenDataset,
    find_file_pairs,
    find_multimodal_train_tiles,
    GeoFMDataset7A,
    HEIGHT_NORM_CONSTANT
)
from core.losses import ImprovedCompositeLoss, DualPathLoss, GradNormBalancer

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
    parser.add_argument("--model-type", type=str, default=MODEL_TYPE, choices=["auto", "lightunet", "decoder_residual", "attention_fusion", "dual_enc_dec_fusion", "fresh_extract"])
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
    # --- 7A (dual_enc_dec_fusion) options ---
    parser.add_argument("--cv-fold", type=int, default=0, help="[7A] Geographic CV fold to hold out for validation (0-4). See data/geo_folds.json.")
    parser.add_argument("--use-stratified-sampler", action=argparse.BooleanOptionalAction, default=True, help="[7A] Use WeightedRandomSampler over coverage strata (default True).")
    parser.add_argument("--use-gradnorm", action=argparse.BooleanOptionalAction, default=True, help="[7A] Learn task loss weights with GradNorm (default True). Used in Phase 3.")
    parser.add_argument("--static-weights", type=str, default="0.65,0.64,1.70", help="[7A Phase 5C] Static task weights 'w_f,w_h,w_b' used when --no-use-gradnorm. Default = GradNorm-converged values from 7A_base_e90 ep59.")
    parser.add_argument("--use-thor", action=argparse.BooleanOptionalAction, default=False, help="[7A] Include THOR embeddings (default False; Phase 5 ablation). [Phase 5D] Inferred automatically from --patch-inputs; kept for back-compat.)")
    parser.add_argument("--patch-stem-version", type=str, default="v2", choices=["v1", "v2", "v2b"],
                        help="[7A Phase 5D/5E] Patch token stem: v1 = original (7A_simple); "
                             "v2 = enhanced input-LayerNorm + 2-layer MLP (NOTE: input LayerNorm "
                             "collapses water IoU); v2b = 2-layer MLP + output norm, no input "
                             "LayerNorm (water-safe). Default v2.")
    parser.add_argument("--xattn-heads", type=int, default=4,
                        help="[7A Phase 5D] Cross-attention heads at patch-token injection. "
                             "Default 4. Reduce to 2 if THOR (2x K/V tokens) OOMs.")
    parser.add_argument("--patch-routing", type=str, default="sensor",
                        choices=["sensor", "s1-both", "all-both"],
                        help="[7A Phase 5E #4] Sensor->branch routing. 'sensor' (default) = "
                             "s1->height, s2->fraction (byte-identical). 's1-both' also feeds "
                             "S1 (SAR) into the fraction branch (water/building). 'all-both' = both.")
    parser.add_argument("--use-fraction-bridge", action=argparse.BooleanOptionalAction, default=False,
                        help="[7A Phase 5E #1] Add a tessera-bottleneck -> fraction-decoder bridge "
                             "(GradScale-protected), mirror of the height bridge. Default off.")
    parser.add_argument("--fraction-bridge-alpha", type=float, default=0.2,
                        help="[7A Phase 5E #1] GradScale alpha for the fraction bridge. Default 0.2.")
    parser.add_argument("--train-folds", type=str, default=None,
                        help="[Phase 11 fast-dev] Comma-separated geo-folds to TRAIN on (e.g. '0'). "
                             "With --val-folds, overrides the cv_fold split for fast screening "
                             "(train fold 0 / val fold 1 ≈ 5× faster). Default None = normal cv_fold.")
    parser.add_argument("--val-folds", type=str, default=None,
                        help="[Phase 11 fast-dev] Comma-separated geo-folds to VALIDATE on (e.g. '1').")
    parser.add_argument("--use-terramind", action=argparse.BooleanOptionalAction, default=True,
                        help="[Phase 11] fresh_extract: inject TerraMind tokens (default True). "
                             "--no-use-terramind = pixel-only ablation (Bet 3b).")
    parser.add_argument("--terramind-fusion", type=str, default="add",
                        choices=["add", "xattn", "xattn_frac", "xattn_frac_loc"],
                        help="[Phase 12] fresh_extract: how TerraMind tokens enter the pyramid. "
                             "'add' = legacy residual injection (proxy 0.399, hurt); "
                             "'xattn' = shared gated cross-attention (0.408, helps IoU/hurts height); "
                             "'xattn_frac' = cross-attn routed to fraction decoder only (height stays "
                             "clean); 'xattn_frac_loc' = xattn_frac + soft Gaussian locality bias.")
    parser.add_argument("--cross-modal", type=str, default="off", choices=["off", "coarse", "fine"],
                        help="[Phase 12c] fresh_extract: bidirectional alpha<->tessera cross-attention "
                             "before the 1x1 fuse. 'coarse' = global at L3/L4 (null, f38); 'fine' = "
                             "windowed (8x8) at L0/L1 where building structure lives; 'off' = byte-identical.")
    parser.add_argument("--cross-modal-local", action="store_true",
                        help="[Phase 12c] add the soft Gaussian locality bias to the cross-modal blocks.")
    parser.add_argument("--height-bridge-alpha", type=float, default=0.2,
                        help="[Phase 10] GradScale alpha on the alpha(optical)->height-decoder "
                             "bridge: fraction of the height-loss gradient that reaches the alpha "
                             "encoder. Default 0.2 (byte-identical). Raise (0.5/1.0) to let the "
                             "optical encoder contribute more to height (watch IoU_B for the cost).")
    parser.add_argument("--task", type=str, default="all", choices=["all", "fraction", "height"],
                        help="[7A Phase 8 P4] Train all tasks (default), or a single task: "
                             "'fraction' = fraction+binary only (drops height loss); "
                             "'height' = height only. Single-task diagnostic for multi-task interference.")
    parser.add_argument("--strata-weights", type=str, default="1.0,1.5,2.0,3.0",
                        help="[7A Phase 8 P3a] WeightedRandomSampler weights for the 4 coverage "
                             "strata (empty,sparse,medium,dense). Default '1.0,1.5,2.0,3.0'. Try "
                             "'1,2,4,8' for more aggressive rare-class (building/water) oversampling.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="[7A Phase 8] bf16 mixed precision (autocast) for train+eval forward. "
                             "~halves activation memory (bs32 ~42GB -> ~24GB) so the 7A model fits "
                             "with headroom on a 48GB GPU. Default off (fp32, byte-identical).")
    # --- [Phase 9] Building-objective redesign (P1 + P2) ---
    parser.add_argument("--fraction-head", type=str, default="sigmoid3",
                        choices=["sigmoid3", "softmax4"],
                        help="[Phase 9 P2] Fraction head: 'sigmoid3' (default, byte-identical; "
                             "3 independent sigmoids) or 'softmax4' (building/veg/water/other "
                             "4-way softmax simplex; 'other'=1-b-v-w derived).")
    parser.add_argument("--building-overlap", type=str, default="dice",
                        choices=["dice", "tversky", "focal_tversky"],
                        help="[Phase 9 P1] Overlap loss on the building channel: 'dice' (default, "
                             "class-blind, byte-identical), 'tversky', or 'focal_tversky'.")
    parser.add_argument("--tversky-alpha", type=float, default=0.3,
                        help="[Phase 9 P1] Tversky FP weight (used by tversky/focal_tversky).")
    parser.add_argument("--tversky-beta", type=float, default=0.7,
                        help="[Phase 9 P1] Tversky FN weight (beta>alpha favours building recall).")
    parser.add_argument("--focal-tversky-gamma", type=float, default=1.333,
                        help="[Phase 9 P1] Focal-Tversky focusing exponent ((1-T)**(1/gamma)).")
    parser.add_argument("--building-overlap-weight", type=float, default=1.0,
                        help="[Phase 9 P1] Multiplier on the building overlap term vs veg/water.")
    parser.add_argument("--dice-k", type=float, default=5.0,
                        help="[Phase 9 P1] Sharpness of the shifted-sigmoid Dice relaxation "
                             "(sigmoid3 path). Default 5.0 = byte-identical.")
    parser.add_argument("--max-batches", type=int, default=0, help="If >0, cap batches per epoch (smoke testing).")
    parser.add_argument("--veg-height-boost", type=float, default=0.0,
                        help="[7A P5.1] Extra weight on masked Huber for vegetation pixels "
                             "(veg_frac>0.1). 0.0 = Phase-4 baseline; try 2-3 to lower RMSE_V.")
    parser.add_argument("--decouple-height", action="store_true",
                        help="[Phase 12] Separate building/veg height Huber terms (independent "
                             "weights, own normalisation) so boosting veg does not starve building.")
    parser.add_argument("--build-height-weight", type=float, default=1.0,
                        help="[Phase 12] Weight on the building-pixel height term (decoupled mode).")
    parser.add_argument("--veg-height-weight", type=float, default=1.0,
                        help="[Phase 12] Weight on the veg-pixel height term (decoupled mode).")
    parser.add_argument("--height-bins", type=int, default=0,
                        help="[Phase 13a] >0: adaptive-bin height head (N bins, soft-expectation) + "
                             "discrete-continuous loss (CE+Huber). 0 = legacy scalar regression.")
    parser.add_argument("--height-ce-weight", type=float, default=0.1,
                        help="[Phase 13a] weight on the bin cross-entropy term vs the Huber-on-expectation.")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="[7A] Directory for the memmap float16 tile cache. If set, "
                             "tiles are preprocessed once and served from the (page-cached) "
                             "memmap on later epochs — turns the IO-bound run compute-bound.")
    parser.add_argument("--scratch-dir", type=str, default=None,
                        help="[I/O] Write run outputs (checkpoints/params/curve) to this LOCAL "
                             "dir (e.g. NVMe) during training, then copy to the NFS runs/ dir at "
                             "the end. Avoids the per-epoch 74MB checkpoint + per-epoch param "
                             "writes to a busy NFS that can stall the loop. Tee the log to the "
                             "same local dir too. Default None = write directly to NFS runs/.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="[7A] Force rebuild of the tile cache even if a .done flag exists.")
    parser.add_argument("--no-height-bridge", action="store_true",
                        help="[7A Phase 5C] Disable the alpha-bottleneck cross-encoder bridge "
                             "into the height decoder (ablation; default keeps the bridge).")
    parser.add_argument("--no-binary-head", action="store_true",
                        help="[7A Phase 5C] Drop the auxiliary binary building loss term "
                             "(the head is left untrained, removed from task weighting).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Global RNG seed (torch / numpy / random) for reproducibility.")
    return parser.parse_args()


def worker_init_fn(worker_id):
    """Seed numpy/torch per worker so augmentation RNG differs across workers and epochs."""
    base_seed = torch.initial_seed() % (2**31 - 1)
    seed = (base_seed + worker_id) % (2**31 - 1)
    np.random.seed(seed)
    random.seed(seed)


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

    # 7A dispatch: fully self-contained path; never touches the legacy branches.
    # fresh_extract (Phase 11) shares the same data/loss/eval pipeline.
    if MODEL_TYPE in ("dual_enc_dec_fusion", "fresh_extract"):
        return train_7a(args)

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
    best_val_loss = float('inf')
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
                val_samples_seen += batch_size
                val_avg_live = val_running_loss / max(1, val_samples_seen)
                val_pbar.set_postfix(avg=f"{val_avg_live:.4f}")

        epoch_val_loss = val_running_loss / len(val_ds)
        epoch_comp = val_components / len(val_ds)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   >> Model Saved! (New Best Val Loss: {best_val_loss:.4f})")

        epoch_elapsed = time.time() - epoch_start_time
        epoch_times.append(epoch_elapsed)
        epoch_time_msg = f"   >> Epoch Time: {epoch_elapsed:.2f} seconds ({epoch_elapsed/60:.2f} minutes)\n"
        
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train: {epoch_loss:.4f} | Val: {epoch_val_loss:.4f}")
        print(epoch_time_msg, end="")
        print(
            f"   >> Val Breakdown: MAE:{epoch_comp[0]:.3f} | SSIM:{epoch_comp[1]:.3f} | Grad:{epoch_comp[2]:.3f} | Tversky:{epoch_comp[3]:.3f}")

        # Append epoch time to params log file
        with open(CONFIG_LOG_PATH, "a") as f:
            f.write(f"Epoch {epoch + 1} finished in {epoch_elapsed:.2f}s ({epoch_elapsed/60:.2f}m) | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}\n")

        # Clean memory to avoid leaks/fragmentation across epochs
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start_time
    total_time_msg = (
        f"\n=== TRAINING REPORT ===\n"
        f"Total Training Time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)\n"
    )
    for i, t in enumerate(epoch_times):
        total_time_msg += f"   >> Epoch {i+1}: {t:.2f}s ({t/60:.2f}m)\n"
    total_time_msg += "=======================\n"
    print(total_time_msg, end="")
    
    with open(CONFIG_LOG_PATH, "a") as f:
        f.write(total_time_msg)

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

# =============================================================================
# 7A — Training path for DualEncDualDecFusion
# =============================================================================

# Resolve via config so compute nodes read raw tiles/labels from their local
# disk (/mnt/nK/...) instead of the contended head NFS. Falls back to the head
# path on the head node (where config.TARGET_DRIVE already points there).
DATA_ROOT_7A = config.TARGET_DRIVE


def _runs_dir_7a():
    """Resolve runs output dir (config.py may point at a path that doesn't exist)."""
    head = "/mnt/head/users/bassam/data/geofmdata/runs"
    if os.path.isdir(os.path.dirname(head)):
        os.makedirs(head, exist_ok=True)
        return head
    os.makedirs(config.SHARED_RUNS_DIR, exist_ok=True)
    return config.SHARED_RUNS_DIR


@torch.no_grad()
def evaluate_7a(model, val_loader, criterion, device, C=4.0, amp=False,
                fraction_head="sigmoid3"):
    """Hard-IoU@0.5 (B/V/W), masked RMSE in metres (B/V), proxy (C=4.0), val losses."""
    model.eval()
    inter = torch.zeros(3, device=device)
    union = torch.zeros(3, device=device)
    se_b = torch.zeros((), device=device); n_b = torch.zeros((), device=device)
    se_v = torch.zeros((), device=device); n_v = torch.zeros((), device=device)
    task_sums = {"fraction": 0.0, "height": 0.0, "binary": 0.0}
    nb = 0

    ev_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
        if (amp and device.type == "cuda") else nullcontext
    for batch in val_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch["target"]
        with ev_ctx():
            out = model(batch)
            losses = criterion(out, target)
        out = {k: v.float() for k, v in out.items()}   # fp32 for metric math
        for k in task_sums:
            task_sums[k] += float(losses[k].detach())
        nb += 1

        if fraction_head == "softmax4":
            frac_prob = torch.softmax(out["fraction"], dim=1)[:, :3]
        else:
            frac_prob = torch.sigmoid(out["fraction"])
        h = out["height"]
        if h.shape[1] > 1:                      # [Phase 13a] bins -> soft-expectation
            h = model.expected_height(h)
        pred = torch.cat([frac_prob, h], dim=1)
        for c in range(3):
            p = pred[:, c] > 0.5
            t = target[:, c] > 0.5
            inter[c] += (p & t).sum()
            union[c] += (p | t).sum()

        pred_h = pred[:, 3] * HEIGHT_NORM_CONSTANT
        tgt_h = target[:, 3] * HEIGHT_NORM_CONSTANT
        sq = (pred_h - tgt_h) ** 2
        mb = target[:, 0] > 0
        mv = target[:, 1] > 0
        se_b += (sq * mb).sum(); n_b += mb.sum()
        se_v += (sq * mv).sum(); n_v += mv.sum()

    iou_b, iou_v, iou_w = (inter / union.clamp_min(1.0)).cpu().tolist()
    rmse_b = float(torch.sqrt(se_b / n_b.clamp_min(1.0)))
    rmse_v = float(torch.sqrt(se_v / n_v.clamp_min(1.0)))
    proxy = (0.25 * iou_b + 0.15 * iou_v + 0.15 * iou_w
             + 0.25 * max(0.0, 1.0 - rmse_b / C)
             + 0.20 * max(0.0, 1.0 - rmse_v / C))
    return {
        "iou_b": iou_b, "iou_v": iou_v, "iou_w": iou_w,
        "rmse_b": rmse_b, "rmse_v": rmse_v, "proxy": proxy,
        "val_task_losses": {k: v / max(1, nb) for k, v in task_sums.items()},
    }


def train_7a(args):
    # Global reproducibility seed (overrides the module-level default seed).
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Resolve patch-input names (terramind_s1/s2, thor_s1/s2). 'all' expands to
    # the full PATCH_DIR_MAP. THOR usage is inferred from the resolved names.
    _patch_str = args.patch_inputs.strip().lower()
    if _patch_str == "all":
        patch_names = list(PATCH_DIR_MAP.keys())
    else:
        patch_names = [p.strip() for p in _patch_str.split(",") if p.strip()]
    for p in patch_names:
        if p not in PATCH_DIR_MAP:
            raise ValueError(f"Unknown patch input '{p}'. Valid: {list(PATCH_DIR_MAP)}")
    if not any(p.endswith("_s1") for p in patch_names) and \
       not any(p.endswith("_s2") for p in patch_names):
        raise ValueError(f"--patch-inputs must include at least one *_s1 or *_s2 stream; got {patch_names}")
    use_thor = any(p.startswith("thor") for p in patch_names)
    print(f"   >> patch inputs: {patch_names}  (use_thor={use_thor}, "
          f"stem={args.patch_stem_version}, xattn_heads={args.xattn_heads})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 autocast context (Phase 8 --amp): halves activation memory so bs32 fits
    # with headroom. No GradScaler needed for bf16. nullcontext when off (fp32).
    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
        if (args.amp and device.type == "cuda") else nullcontext
    if args.amp:
        print("   >> AMP on (bf16 autocast, train+eval forward)")
    nfs_runs_dir = _runs_dir_7a()
    nfs_exp_dir = os.path.join(nfs_runs_dir, args.experiment_name)
    os.makedirs(nfs_exp_dir, exist_ok=True)
    # [I/O] When --scratch-dir is set, the HEAVY writers (74MB checkpoints + curve)
    # go to local NVMe during the run and are copied to NFS at the end. The tiny
    # per-epoch training_params.txt append stays on NFS so the run is monitorable
    # live and the metrics record is always there even if the run dies.
    out_root = args.scratch_dir if args.scratch_dir else nfs_runs_dir
    exp_dir = os.path.join(out_root, args.experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    if args.scratch_dir:
        print(f"   >> scratch I/O: checkpoints/curve -> {exp_dir} (local), copied to "
              f"{nfs_exp_dir} at end; training_params.txt stays on NFS for live monitoring")
    best_path = os.path.join(exp_dir, "model_best.pth")
    last_path = os.path.join(exp_dir, "model_last.pth")
    cfg_path = os.path.join(nfs_exp_dir, "training_params.txt")   # NFS: live monitoring + record
    curve_path = os.path.join(exp_dir, "loss_curve.png")

    with open(cfg_path, "w") as f:
        f.write(f"--- EXPERIMENT: {args.experiment_name} ---\n")
        f.write(f"MODEL_TYPE: {args.model_type}\n")
        f.write(f"PATCH_SIZE: {args.patch_size}\n")
        f.write(f"BATCH_SIZE: {args.batch_size}\n")
        f.write(f"EPOCHS: {args.epochs}\n")
        f.write(f"CV_FOLD: {args.cv_fold}\n")
        f.write(f"VEG_HEIGHT_BOOST: {args.veg_height_boost}\n")
        f.write(f"DECOUPLE_HEIGHT: {args.decouple_height}\n")
        f.write(f"BUILD_HEIGHT_WEIGHT: {args.build_height_weight}\n")
        f.write(f"VEG_HEIGHT_WEIGHT: {args.veg_height_weight}\n")
        f.write(f"HEIGHT_BINS: {args.height_bins}\n")
        f.write(f"HEIGHT_CE_WEIGHT: {args.height_ce_weight}\n")
        f.write(f"CACHE_DIR: {args.cache_dir}\n")
        f.write(f"USE_STRATIFIED_SAMPLER: {args.use_stratified_sampler}\n")
        f.write(f"USE_GRADNORM: {args.use_gradnorm}\n")
        f.write(f"USE_THOR: {use_thor}\n")
        f.write(f"PATCH_INPUTS: {','.join(patch_names)}\n")
        f.write(f"PATCH_STEM_VERSION: {args.patch_stem_version}\n")
        f.write(f"XATTN_HEADS: {args.xattn_heads}\n")
        f.write(f"PATCH_ROUTING: {args.patch_routing}\n")
        f.write(f"USE_FRACTION_BRIDGE: {args.use_fraction_bridge}\n")
        f.write(f"FRACTION_BRIDGE_ALPHA: {args.fraction_bridge_alpha}\n")
        f.write(f"HEIGHT_BRIDGE_ALPHA: {args.height_bridge_alpha}\n")
        f.write(f"USE_TERRAMIND: {args.use_terramind}\n")
        f.write(f"TERRAMIND_FUSION: {args.terramind_fusion}\n")
        f.write(f"CROSS_MODAL: {args.cross_modal}\n")
        f.write(f"CROSS_MODAL_LOCAL: {args.cross_modal_local}\n")
        f.write(f"TRAIN_FOLDS: {args.train_folds}\n")
        f.write(f"VAL_FOLDS: {args.val_folds}\n")
        f.write(f"STATIC_WEIGHTS: {args.static_weights}\n")
        f.write(f"NO_HEIGHT_BRIDGE: {args.no_height_bridge}\n")
        f.write(f"NO_BINARY_HEAD: {args.no_binary_head}\n")
        f.write(f"TASK: {args.task}\n")
        f.write(f"STRATA_WEIGHTS: {args.strata_weights}\n")
        f.write(f"AMP: {args.amp}\n")
        f.write(f"FRACTION_HEAD: {args.fraction_head}\n")
        f.write(f"BUILDING_OVERLAP: {args.building_overlap}\n")
        f.write(f"TVERSKY_ALPHA: {args.tversky_alpha}\n")
        f.write(f"TVERSKY_BETA: {args.tversky_beta}\n")
        f.write(f"FOCAL_TVERSKY_GAMMA: {args.focal_tversky_gamma}\n")
        f.write(f"BUILDING_OVERLAP_WEIGHT: {args.building_overlap_weight}\n")
        f.write(f"DICE_K: {args.dice_k}\n")
        f.write(f"SEED: {args.seed}\n")
        f.write(f"OPTIMIZER: AdamW lr={LEARNING_RATE} wd={WEIGHT_DECAY}\n")

    print("--- 7A Data Setup ---")
    # [no-holdout] cv_fold < 0 -> train on ALL tiles, no validation (final
    # submission model). The split logic already yields train=all (no fold == -1)
    # and val=empty; we just skip building/using the empty val set.
    no_holdout = args.cv_fold < 0
    fast = args.train_folds is not None  # [Phase 11] explicit fold-set screening
    tr_inc = [int(x) for x in args.train_folds.split(",")] if args.train_folds else None
    va_inc = [int(x) for x in args.val_folds.split(",")] if args.val_folds else None
    tiles = find_multimodal_train_tiles(DATA_ROOT_7A, use_thor=use_thor)
    train_ds = GeoFMDataset7A(tiles, is_train=True, cv_fold=args.cv_fold,
                              cache_dir=args.cache_dir, rebuild_cache=args.rebuild_cache,
                              patch_inputs=patch_names, include_folds=tr_inc)
    if fast:
        val_ds = (GeoFMDataset7A(tiles, is_train=False, cache_dir=args.cache_dir,
                                 rebuild_cache=args.rebuild_cache, patch_inputs=patch_names,
                                 include_folds=va_inc) if va_inc else None)
        no_holdout = (val_ds is None)
        print(f"   >> FAST-DEV: train folds={tr_inc} ({len(train_ds)})  "
              f"val folds={va_inc} ({len(val_ds) if val_ds else 0})")
    elif no_holdout:
        val_ds = None
        print(f"   >> matched={len(tiles)}  train={len(train_ds)}  val=0  "
              f"(NO-HOLDOUT: training on ALL tiles, cv_fold={args.cv_fold})")
    else:
        val_ds = GeoFMDataset7A(tiles, is_train=False, cv_fold=args.cv_fold,
                                cache_dir=args.cache_dir, rebuild_cache=args.rebuild_cache,
                                patch_inputs=patch_names)
        print(f"   >> matched={len(tiles)}  train={len(train_ds)}  val={len(val_ds)}  (cv_fold={args.cv_fold})")

    if args.use_stratified_sampler:
        _strata = [float(x) for x in args.strata_weights.split(",")]
        weights = train_ds.sampler_weights(weights=_strata)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = None if no_holdout else DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print("--- 7A Model Init ---")
    model, _ = build_model(args.model_type, n_channels=64, n_classes=4,
                           use_height_bridge=not args.no_height_bridge,
                           patch_inputs=patch_names,
                           patch_stem_version=args.patch_stem_version,
                           xattn_heads=args.xattn_heads,
                           patch_routing=args.patch_routing,
                           use_fraction_bridge=args.use_fraction_bridge,
                           fraction_bridge_alpha=args.fraction_bridge_alpha,
                           fraction_head=args.fraction_head,
                           bridge_alpha=args.height_bridge_alpha,
                           use_terramind=args.use_terramind,
                           terramind_fusion=args.terramind_fusion,
                           cross_modal=args.cross_modal,
                           cross_modal_local=args.cross_modal_local,
                           height_bins=args.height_bins)
    model = model.to(device)
    print(f"   >> params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
          f"  (height_bridge={'off' if args.no_height_bridge else 'on'})")

    criterion = DualPathLoss(veg_height_boost=args.veg_height_boost,
                             use_binary=not args.no_binary_head,
                             fraction_head=args.fraction_head,
                             building_overlap=args.building_overlap,
                             tversky_alpha=args.tversky_alpha,
                             tversky_beta=args.tversky_beta,
                             focal_tversky_gamma=args.focal_tversky_gamma,
                             building_overlap_weight=args.building_overlap_weight,
                             decouple_height=args.decouple_height,
                             build_height_weight=args.build_height_weight,
                             veg_height_weight=args.veg_height_weight,
                             height_bins=args.height_bins,
                             height_ce_weight=args.height_ce_weight,
                             dice_k=args.dice_k).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Drop the binary task entirely when the aux head is disabled, so GradNorm /
    # static weighting never sees a constant-zero loss (which would NaN GradNorm).
    task_names = ["fraction", "height"] if args.no_binary_head \
        else ["fraction", "height", "binary"]
    # [Phase 8 P4] Single-task diagnostic: keep only the active task's losses.
    if args.task == "fraction":
        task_names = [t for t in task_names if t in ("fraction", "binary")]
    elif args.task == "height":
        task_names = ["height"]
    print(f"   >> task={args.task}  active losses: {task_names}")
    use_gn = args.use_gradnorm
    if use_gn:
        balancer = GradNormBalancer(task_names, device, alpha=1.5, lr=0.025)
        ref_params = {
            "fraction": model.alpha_encoder.down4.block[0].weight,
            "height":   model.tessera_encoder.down4.block[0].weight,
        }
        if not args.no_binary_head:
            ref_params["binary"] = model.alpha_encoder.down4.block[0].weight
    else:
        balancer = None
        _sw = [float(x) for x in args.static_weights.split(",")]
        if len(_sw) != 3:
            raise ValueError(f"--static-weights expects 'w_f,w_h,w_b', got {args.static_weights!r}")
        static_w = {"fraction": _sw[0], "height": _sw[1], "binary": _sw[2]}
        print(f"   >> static task weights w[f/h/b] = {_sw[0]}/{_sw[1]}/{_sw[2]}")

    best_proxy = -1.0
    train_hist, proxy_hist = [], []
    total_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0; seen = 0
        ep_task = {k: 0.0 for k in task_names}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]", leave=True)
        for bi, batch in enumerate(pbar):
            if args.max_batches and bi >= args.max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch["target"]
            optimizer.zero_grad()
            with amp_ctx():
                out = model(batch)
                per_task = criterion(out, target)

                if use_gn:
                    balancer.step(per_task, ref_params)    # before model backward
                    w = balancer.weights()
                    loss = sum(w[i] * per_task[n] for i, n in enumerate(task_names))
                else:
                    loss = sum(static_w[n] * per_task[n] for n in task_names)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = target.size(0)
            running += float(loss.detach()) * bs; seen += bs
            for n in task_names:
                ep_task[n] += float(per_task[n].detach()) * bs
            pbar.set_postfix(loss=f"{float(loss.detach()):.4f}")

        scheduler.step()
        epoch_loss = running / max(1, seen)
        train_hist.append(epoch_loss)
        ep_task = {k: v / max(1, seen) for k, v in ep_task.items()}

        gn_str = ""
        if use_gn:
            wd = balancer.weight_dict()
            gn_str = "  w[f/h/b]=%.2f/%.2f/%.2f" % (
                wd["fraction"], wd["height"], wd.get("binary", 0.0))

        if no_holdout:
            # No validation set: save the current model every epoch (the final
            # model is the last epoch) and log train loss only.
            proxy_hist.append(epoch_loss)
            torch.save(model.state_dict(), best_path)   # = latest; predict.py loads model_best
            print(f"Epoch {epoch+1}/{args.epochs} | train {epoch_loss:.4f} | "
                  f"(no-holdout, saved){gn_str}")
            with open(cfg_path, "a") as f:
                f.write(f"Epoch {epoch+1}: train={epoch_loss:.4f} (no-holdout) "
                        f"task_train={ep_task}{gn_str}\n")
        else:
            metrics = evaluate_7a(model, val_loader, criterion, device, amp=args.amp,
                                  fraction_head=args.fraction_head)
            proxy_hist.append(metrics["proxy"])

            if metrics["proxy"] > best_proxy:
                best_proxy = metrics["proxy"]
                torch.save(model.state_dict(), best_path)
                tag = "  *** new best ***"
            else:
                tag = ""

            print(f"Epoch {epoch+1}/{args.epochs} | train {epoch_loss:.4f} | "
                  f"proxy {metrics['proxy']:.4f} | IoU B/V/W {metrics['iou_b']:.3f}/"
                  f"{metrics['iou_v']:.3f}/{metrics['iou_w']:.3f} | RMSE B/V "
                  f"{metrics['rmse_b']:.2f}/{metrics['rmse_v']:.2f}m{gn_str}{tag}")
            with open(cfg_path, "a") as f:
                f.write(f"Epoch {epoch+1}: train={epoch_loss:.4f} proxy={metrics['proxy']:.4f} "
                        f"IoU_B={metrics['iou_b']:.4f} IoU_V={metrics['iou_v']:.4f} "
                        f"IoU_W={metrics['iou_w']:.4f} RMSE_B={metrics['rmse_b']:.3f} "
                        f"RMSE_V={metrics['rmse_v']:.3f} task_train={ep_task} "
                        f"val_task={metrics['val_task_losses']}{gn_str}\n")

    torch.save(model.state_dict(), last_path)
    total_min = (time.time() - total_start) / 60
    summary = (f"final train loss={train_hist[-1]:.4f} (no-holdout)" if no_holdout
               else f"best proxy={best_proxy:.4f}")
    print(f"\n=== 7A DONE === {summary}  total={total_min:.1f}m")
    with open(cfg_path, "a") as f:
        if no_holdout:
            f.write(f"NO_HOLDOUT: True\nFINAL_TRAIN_LOSS: {train_hist[-1]:.4f}\nTOTAL_MIN: {total_min:.1f}\n")
        else:
            f.write(f"BEST_PROXY: {best_proxy:.4f}\nTOTAL_MIN: {total_min:.1f}\n")

    try:
        plt.figure()
        plt.plot(train_hist, label="train loss")
        plt.plot(proxy_hist, label="val proxy")
        plt.legend(); plt.title(f"7A {args.experiment_name}")
        plt.savefig(curve_path); plt.close()
    except Exception as e:
        print(f"   (curve plot skipped: {e})")

    # [I/O] Bulk-copy local scratch outputs to the NFS runs/ dir (one transfer at
    # the end instead of per-epoch NFS writes). Includes train.log if tee'd here.
    if args.scratch_dir and os.path.abspath(exp_dir) != os.path.abspath(nfs_exp_dir):
        import shutil
        try:
            os.makedirs(nfs_exp_dir, exist_ok=True)
            for fn in os.listdir(exp_dir):
                src = os.path.join(exp_dir, fn)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(nfs_exp_dir, fn))
            print(f"   >> copied run outputs to NFS: {nfs_exp_dir}")
        except Exception as e:
            print(f"   (NFS copy failed; outputs remain local at {exp_dir}: {e})")

    return best_proxy


if __name__ == "__main__":
    main()