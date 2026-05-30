import config
import os
import argparse
import glob
import re
import numpy as np
import rasterio
import torch
from tqdm import tqdm

# --- IMPORT FROM CORE MODULES ---
from core.model import build_model
from core.dataset import _normalize_core_id, HEIGHT_NORM_CONSTANT

# --- DEFAULTS ---
EXPERIMENT_NAME = "terramind_decoder_run01"
BASE_DIR = config.SHARED_RUNS_DIR
PATCH_SIZE = config.PATCH_SIZE
MAX_SAMPLES = 0

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

PIXEL_TEST_DIR_MAP = {
    "tessera": config.TESSERA_TEST_DIR,
    "alpha_earth": config.ALPHA_EARTH_TEST_DIR,
}

PATCH_TEST_DIR_MAP = {
    "terramind_s1": config.TERRAMIND_S1_TEST_DIR,
    "terramind_s2": config.TERRAMIND_S2_TEST_DIR,
    "thor_s1": config.THOR_S1_TEST_DIR,
    "thor_s2": config.THOR_S2_TEST_DIR,
}

PIXEL_TRAIN_DIR_MAP = {
    "tessera": config.TESSERA_DIR,
    "alpha_earth": config.ALPHA_EARTH_DIR,
}

PATCH_TRAIN_DIR_MAP = {
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained model and run inference on embeddings, saving predictions as .npy files."
    )
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment to predict.")
    parser.add_argument("--pixel-inputs", type=str, default=None,
                        help="Comma-separated pixel embeddings (e.g. alpha_earth). "
                             "Overrides the value stored in training_params.txt.")
    parser.add_argument("--patch-inputs", type=str, default=None,
                        help="Comma-separated patch embeddings (e.g. terramind_s1,terramind_s2). "
                             "Overrides the value stored in training_params.txt.")
    parser.add_argument("--tta", action="store_true",
                        help="Enable 8-fold Test-Time Augmentation (4 rotations × 2 flips). "
                             "Averages predictions in logit space before sigmoid. No retraining needed.")
    return parser.parse_args()


def extract_core_id_from_filename(filepath):
    basename = os.path.basename(filepath)
    match = re.search(r'(\d+_[A-Z]{2}_\d+)', basename)
    if match:
        return match.group(1)
    return _normalize_core_id(filepath)


def find_test_pairs(pixel_dirs, patch_dirs):
    """
    Finds matching pairs of (pixel_embs, patch_embs) in the test set by intersecting normalized core IDs.
    """
    if isinstance(pixel_dirs, str):
        pixel_dirs = [pixel_dirs]
    if isinstance(patch_dirs, str):
        patch_dirs = [patch_dirs]

    pixel_maps = []
    for p_dir in pixel_dirs:
        files = glob.glob(os.path.join(p_dir, "**", "*.tif"), recursive=True)
        pixel_maps.append({ extract_core_id_from_filename(f): f for f in files })

    patch_maps = []
    for p_dir in patch_dirs:
        files = glob.glob(os.path.join(p_dir, "**", "*.tif"), recursive=True)
        patch_maps.append({ extract_core_id_from_filename(f): f for f in files })

    common_ids = None
    for p_map in pixel_maps:
        if common_ids is None:
            common_ids = set(p_map.keys())
        else:
            common_ids &= set(p_map.keys())
    for p_map in patch_maps:
        if common_ids is None:
            common_ids = set(p_map.keys())
        else:
            common_ids &= set(p_map.keys())

    pairs = []
    for cid in sorted(common_ids or set()):
        pixel_paths = [p_map[cid] for p_map in pixel_maps]
        patch_paths = [p_map[cid] for p_map in patch_maps]
        pairs.append((cid, pixel_paths, patch_paths))

    return pairs


def process_embedding(emb_path, model_type, patch_size, scale_factor=16):
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


def process_multi_embeddings(pixel_paths, patch_paths, patch_size, scale_factor=16):
    pixel_embs = []
    for p_path in pixel_paths:
        with rasterio.open(p_path) as src:
            emb = src.read().astype(np.float32)
        emb = np.nan_to_num(emb)
        c, h, w = emb.shape
        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            emb = np.pad(emb, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        top = (emb.shape[1] - patch_size) // 2
        left = (emb.shape[2] - patch_size) // 2
        emb = emb[:, top:top + patch_size, left:left + patch_size]
        pixel_embs.append(emb)
    pixel_tensor = torch.from_numpy(np.concatenate(pixel_embs, axis=0))

    emb_patch_size = patch_size // scale_factor
    patch_embs = []
    for p_path in patch_paths:
        with rasterio.open(p_path) as src:
            emb = src.read().astype(np.float32)
        emb = np.nan_to_num(emb)
        c, h, w = emb.shape
        if h < emb_patch_size or w < emb_patch_size:
            pad_h = max(0, emb_patch_size - h)
            pad_w = max(0, emb_patch_size - w)
            emb = np.pad(emb, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        top = (emb.shape[1] - emb_patch_size) // 2
        left = (emb.shape[2] - emb_patch_size) // 2
        emb = emb[:, top:top + emb_patch_size, left:left + emb_patch_size]
        patch_embs.append(emb)
    patch_tensor = torch.from_numpy(np.concatenate(patch_embs, axis=0))

    return pixel_tensor, patch_tensor


def tta_predict_dual(model, pixel_batch, patch_batch, device):
    """
    8-fold Test-Time Augmentation for dual-input models (ynet_attention_fusion / attention_fusion).
    Applies all D4 symmetry transforms (4 rotations × 2 flips) to both pixel and patch inputs,
    runs inference, applies the inverse transform to each output, then averages in RAW logit space
    before the caller applies sigmoid. This is more principled than averaging sigmoid outputs.
    """
    preds = []
    for k in range(4):          # rotations: 0°, 90°, 180°, 270°
        for flip in (False, True):
            p = torch.rot90(pixel_batch, k, dims=[-2, -1])
            q = torch.rot90(patch_batch, k, dims=[-2, -1])
            if flip:
                p = torch.flip(p, dims=[-1])
                q = torch.flip(q, dims=[-1])

            out = model(p, q)   # raw logits, shape [1, 4, H, W]

            # Invert the spatial transform on the output
            if flip:
                out = torch.flip(out, dims=[-1])
            if k > 0:
                out = torch.rot90(out, -k, dims=[-2, -1])

            preds.append(out)

    return torch.stack(preds).mean(dim=0)   # average in logit space


def tta_predict_single(model, img_batch, device):
    """
    8-fold TTA for single-input models (decoder_residual, lightunet, etc.).
    """
    preds = []
    for k in range(4):
        for flip in (False, True):
            x = torch.rot90(img_batch, k, dims=[-2, -1])
            if flip:
                x = torch.flip(x, dims=[-1])

            out = model(x)

            if flip:
                out = torch.flip(out, dims=[-1])
            if k > 0:
                out = torch.rot90(out, -k, dims=[-2, -1])

            preds.append(out)

    return torch.stack(preds).mean(dim=0)


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
    exp_dir = os.path.join(config.SHARED_RUNS_DIR, args.experiment_name)
    params = load_experiment_params(exp_dir)

    if not params:
        raise RuntimeError(f"Could not find or load training_params.txt in {exp_dir}")

    model_type = params.get("MODEL_TYPE", "decoder_residual").lower()
    # CLI args take priority; fall back to training_params.txt; empty string triggers fallback parser below
    pixel_inputs = args.pixel_inputs or params.get("PIXEL_INPUTS", "")
    patch_inputs = args.patch_inputs or params.get("PATCH_INPUTS", "")
    patch_size = int(params.get("PATCH_SIZE", "256"))

    use_tta = args.tta
    print(f"🔄 Auto-detected config from training_params.txt:")
    print(f"  Model: {model_type} | Patch Size: {patch_size} | TTA: {'ON (8-fold)' if use_tta else 'off'}")
    if args.pixel_inputs or args.patch_inputs:
        print(f"  (CLI override) pixel='{pixel_inputs}'  patch='{patch_inputs}'")

    p1 = os.path.join(exp_dir, "model_best_e1.pth")
    p2 = os.path.join(exp_dir, "model_best.pth")
    model_path = p1 if os.path.exists(p1) else p2

    # TTA predictions go to a separate folder so the base predictions are never overwritten.
    predictions_dir = os.path.join(exp_dir, "predictions_tta" if use_tta else "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    if model_type in ("attention_fusion", "ynet_attention_fusion", "ynet_tessera_xattn", "ynet_tessera_broadcast"):
        pixel_dir_map = PIXEL_TEST_DIR_MAP
        patch_dir_map = PATCH_TEST_DIR_MAP

        # Fallback: older ynet runs saved "Pixel:X_Patch:Y" in TRAIN_EMBEDDINGS_DIR
        # instead of separate PIXEL_INPUTS / PATCH_INPUTS keys.
        if not pixel_inputs or not patch_inputs:
            combined = params.get("TRAIN_EMBEDDINGS_DIR", "")
            import re as _re
            m = _re.match(r"Pixel:(.+?)_Patch:(.+)", combined)
            if m:
                pixel_inputs = m.group(1).strip()
                patch_inputs = m.group(2).strip()
                print(f"  (Fallback) Parsed from TRAIN_EMBEDDINGS_DIR: pixel='{pixel_inputs}' patch='{patch_inputs}'")
            else:
                raise RuntimeError(
                    f"Cannot determine pixel/patch inputs for model '{model_type}'. "
                    f"Expected PIXEL_INPUTS and PATCH_INPUTS in training_params.txt, "
                    f"or TRAIN_EMBEDDINGS_DIR in format 'Pixel:X_Patch:Y'. Got: '{combined}'"
                )

        pixel_dirs = resolve_dirs(pixel_inputs, pixel_dir_map)
        patch_dirs = resolve_dirs(patch_inputs, patch_dir_map)
        test_pairs = find_test_pairs(pixel_dirs, patch_dirs)
        
        print(f"🔍 Found {len(test_pairs)} matched test/train pairs.")
        if not test_pairs:
            raise RuntimeError("No matched pairs found. Check inputs.")
            
        # Get channel dimensions dynamically
        pixel_tensor, patch_tensor = process_multi_embeddings(
            test_pairs[0][1], test_pairs[0][2], patch_size
        )
        pixel_channels = pixel_tensor.shape[0]
        patch_channels = patch_tensor.shape[0]
        n_channels = pixel_channels

        # --- Load model ---
        model, selected_model = build_model(
            model_type,
            n_channels=n_channels,
            n_classes=4,
            pixel_channels=pixel_channels,
            patch_channels=patch_channels
        )
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Loaded model: {selected_model} from {model_path} (pixel channels={pixel_channels}, patch channels={patch_channels})")

        # --- Run inference ---
        desc = "Predicting (TTA 8x)" if use_tta else "Predicting"
        print(f"Running inference on {len(test_pairs)} samples...")
        with torch.no_grad():
            for core_id, pixel_paths, patch_paths in tqdm(test_pairs, desc=desc):
                pixel_tensor, patch_tensor = process_multi_embeddings(pixel_paths, patch_paths, patch_size)

                pixel_batch = pixel_tensor.unsqueeze(0).to(DEVICE)
                patch_batch = patch_tensor.unsqueeze(0).to(DEVICE)

                if use_tta:
                    output_batch = tta_predict_dual(model, pixel_batch, patch_batch, DEVICE)
                else:
                    output_batch = model(pixel_batch, patch_batch)

                preds = output_batch.squeeze()
                preds[:3] = torch.sigmoid(preds[:3])
                pred_np = preds.cpu().numpy().astype(np.float32)

                # Denormalize height channel: model output [0,1] -> physical meters
                pred_np[3] = pred_np[3] * HEIGHT_NORM_CONSTANT

                save_path = os.path.join(predictions_dir, f"{core_id}.npy")
                np.save(save_path, pred_np)

    else:
        # Resolve test/train directory using config.py
        if model_type == "lightunet":
            test_embeddings_dir = config.TESSERA_TEST_DIR
        else:
            test_embeddings_dir = config.TERRAMIND_S1_TEST_DIR

        print(f"Loading embedding files from: {test_embeddings_dir}")
        emb_files = glob.glob(os.path.join(test_embeddings_dir, "**", "*.tif"), recursive=True)
        if not emb_files:
            raise RuntimeError(f"No .tif files found in test_embeddings_dir='{test_embeddings_dir}'.")
            
        # Process the first embedding just to get the number of channels dynamically
        sample_tensor = process_embedding(emb_files[0], model_type, patch_size)
        n_channels = sample_tensor.shape[0]

        # --- Load model ---
        model, selected_model = build_model(model_type, n_channels=n_channels, n_classes=4)
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"Loaded model: {selected_model} from {model_path} (input channels={n_channels})")

        # --- Run inference ---
        desc = "Predicting (TTA 8x)" if use_tta else "Predicting"
        print(f"Running inference on {len(emb_files)} samples...")
        with torch.no_grad():
            for emb_path in tqdm(emb_files, desc=desc):
                img_tensor = process_embedding(emb_path, model_type, patch_size)
                img_batch = img_tensor.unsqueeze(0).to(DEVICE)

                if use_tta:
                    output_batch = tta_predict_single(model, img_batch, DEVICE)
                else:
                    output_batch = model(img_batch)

                preds = output_batch.squeeze()
                preds[:3] = torch.sigmoid(preds[:3])
                pred_np = preds.cpu().numpy().astype(np.float32)

                # Denormalize height channel: model output [0,1] -> physical meters
                pred_np[3] = pred_np[3] * HEIGHT_NORM_CONSTANT

                core_id = extract_core_id_from_filename(emb_path)
                save_path = os.path.join(predictions_dir, f"{core_id}.npy")
                np.save(save_path, pred_np)

    print(f"Predictions saved to: {predictions_dir}")
    print(f"Output shape per file: {pred_np.shape} [building%, veg%, water%, height_m]")


if __name__ == "__main__":
    main()
