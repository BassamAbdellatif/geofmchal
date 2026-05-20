import config
import os
import argparse
import glob
import re
import numpy as np
import rasterio
import torch
from tqdm.auto import tqdm

# --- IMPORT FROM CORE MODULES ---
from core.model import build_model
from core.dataset import _normalize_core_id, HEIGHT_NORM_CONSTANT

# --- DEFAULTS ---

EXPERIMENT_NAME = "terramind_decoder_run01"
BASE_DIR = "./runs"
TEST_EMBEDDINGS_DIR = None
MODEL_TYPE = "decoder_residual"
PATCH_SIZE = config.PATCH_SIZE
MAX_SAMPLES = 0

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained model and run inference on embeddings, saving predictions as .npy files."
    )
    parser.add_argument("--experiment-name", type=str, default=EXPERIMENT_NAME)
    parser.add_argument("--base-dir", type=str, default=BASE_DIR,
                        help="Root directory containing experiment subfolders.")
    parser.add_argument("--model-type", type=str, default=MODEL_TYPE,
                        choices=["auto", "lightunet", "decoder", "decoder_residual"],
                        help="Model architecture used during training.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to the .pth checkpoint. Defaults to <base-dir>/<experiment-name>/model_best.pth.")
    parser.add_argument("--test-embeddings-dir", type=str, default=None, required=False,
                        help="Directory containing embedding .tif files. Defaults to path in config.py based on model-type.")
    parser.add_argument("--test-targets-dir", type=str, default=None, required=False,
                        help="Kept for compatibility, but ignored in this inference script.")
    parser.add_argument("--predictions-dir", type=str, default=None,
                        help="Output directory for .npy predictions. Defaults to <base-dir>/<experiment-name>/predictions.")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                        help="Limit inference to N samples (0 = all).")
    return parser.parse_args()


def extract_core_id_from_filename(filepath):
    """
    Extracts the core ID from the filename.
    Matches the pattern: number_TwoCapitalLetters_number (e.g., 3474_PQ_2021).
    Falls back to the dataset's _normalize_core_id if the pattern is not found.
    """
    basename = os.path.basename(filepath)
    match = re.search(r'(\d+_[A-Z]{2}_\d+)', basename)
    if match:
        return match.group(1)
    return _normalize_core_id(filepath)


def process_embedding(emb_path, model_type, patch_size, scale_factor=16):
    """
    Reads the embedding .tif file and performs padding/cropping to match 
    the logic used in the dataset classes for inference (is_train=False).
    """
    with rasterio.open(emb_path) as src:
        image = src.read().astype(np.float32)
        
    image = np.nan_to_num(image)
    c, h, w = image.shape
    
    is_lightunet = model_type.lower() == "lightunet"
    if is_lightunet:
        target_size = patch_size
    else:
        target_size = patch_size // scale_factor
        
    # Pad if smaller than the required size
    if h < target_size or w < target_size:
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        h, w = image.shape[1], image.shape[2]
        
    # Center crop for evaluation
    top = (h - target_size) // 2
    left = (w - target_size) // 2
    
    image = image[:, top:top + target_size, left:left + target_size]
    
    return torch.from_numpy(image)


def main():
    args = parse_args()

    # Resolve test directory using config.py if not specified
    if args.test_embeddings_dir is None:
        if args.model_type == "lightunet":
            test_embeddings_dir = config.TESSERA_TEST_DIR
        else:
            # For decoder, decoder_residual, and auto defaults
            test_embeddings_dir = config.TERRAMIND_S1_TEST_DIR
    else:
        test_embeddings_dir = args.test_embeddings_dir

    exp_dir = os.path.join(args.base_dir, args.experiment_name)
    model_path = args.model_path or os.path.join(exp_dir, "model_best.pth")
    predictions_dir = args.predictions_dir or os.path.join(exp_dir, "predictions")

    os.makedirs(predictions_dir, exist_ok=True)

    print(f"Loading embedding files from: {test_embeddings_dir}")
    # Recursively find all .tif files
    emb_files = glob.glob(os.path.join(test_embeddings_dir, "**", "*.tif"), recursive=True)
    if not emb_files:
        raise RuntimeError(f"No .tif files found in test_embeddings_dir='{test_embeddings_dir}'.")
        
    if args.max_samples > 0:
        emb_files = emb_files[:args.max_samples]

    # Process the first embedding just to get the number of channels dynamically
    sample_tensor = process_embedding(emb_files[0], args.model_type, args.patch_size)
    n_channels = sample_tensor.shape[0]

    # --- Load model ---
    model, selected_model = build_model(args.model_type, n_channels=n_channels, n_classes=4)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model: {selected_model} from {model_path} (input channels={n_channels})")

    # --- Run inference ---
    print(f"Running inference on {len(emb_files)} samples...")
    with torch.no_grad():
        for emb_path in tqdm(emb_files, desc="Predicting"):
            img_tensor = process_embedding(emb_path, args.model_type, args.patch_size)
            img_batch = img_tensor.unsqueeze(0).to(DEVICE)

            output_batch = model(img_batch)
            pred_np = output_batch.squeeze().cpu().numpy().astype(np.float32)

            # Denormalize height channel: model output [0,1] -> physical meters
            pred_np[3] = pred_np[3] * HEIGHT_NORM_CONSTANT

            # Extract output filename ID based on regex or fallback
            core_id = extract_core_id_from_filename(emb_path)
            
            save_path = os.path.join(predictions_dir, f"{core_id}.npy")
            np.save(save_path, pred_np)

    print(f"Predictions saved to: {predictions_dir}")
    print(f"Output shape per file: {pred_np.shape} [building%, veg%, water%, height_m]")


if __name__ == "__main__":
    main()
