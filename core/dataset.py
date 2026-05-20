import os
import glob
import re
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

HEIGHT_NORM_CONSTANT = 30.0

def _normalize_core_id(filename):
    """
    Extracts the pure core ID by stripping all known prefixes,
    embedding suffixes, and year suffixes.
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    # 1. Strip label prefix
    if base.startswith("label_"):
        base = base[len("label_"):]

    # 2. Strip embedding prefixes
    for prefix in ("gee_emb_", "tessera_emb_", "s2_", "s1_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    # 3. Strip trailing embedding suffixes (if any)
    if base.endswith("_embedding"):
        base = base[:-len("_embedding")]
    if base.endswith("_embeddings"):
        base = base[:-len("_embeddings")]
    if base.endswith("_merged"):
        base = base[:-len("_merged")]   

    # 4. Strip trailing year suffixes (e.g., '_2021', '_2023')
    base = re.sub(r'_\d{4}$', '', base)

    return base


def find_file_pairs(emb_dir, tar_dir):
    """
    Fast and robust O(N) file matching using a hash map and regex normalization.
    Searches recursively and guarantees a match regardless of prefixes/suffixes.
    """
    pairs = []

    # 1. Grab ALL files from the disk exactly ONCE
    emb_files = glob.glob(os.path.join(emb_dir, "**", "*.tif"), recursive=True)
    label_files = glob.glob(os.path.join(tar_dir, "**", "label_*.tif"), recursive=True)

    # 2. Build a fast lookup dictionary for the labels: {normalized_id: full_path}
    label_map = {}
    for l_path in label_files:
        norm_id = _normalize_core_id(l_path)
        label_map[norm_id] = l_path

    # 3. Match embeddings to the lookup dictionary instantly
    for e_path in emb_files:
        norm_id = _normalize_core_id(e_path)

        if norm_id in label_map:
            pairs.append((e_path, label_map[norm_id]))

    return pairs

# ---------------------------------------------------------
# DATASET 1: Pixel-Based (Alpha Earth, Tessera)
# 1:1 Spatial Resolution (e.g., 256x256 -> 256x256)
# ---------------------------------------------------------
class PixelEmbeddingDataset(Dataset):
    def __init__(self, file_pairs, patch_size=128, is_train=True):
        self.file_pairs = file_pairs
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        emb_path, tar_path = self.file_pairs[idx]

        with rasterio.open(emb_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(tar_path) as src:
            target = src.read().astype(np.float32)

        image, target = np.nan_to_num(image), np.nan_to_num(target)
        target[3, :, :] = np.clip(target[3, :, :] / HEIGHT_NORM_CONSTANT, 0.0, 1.5)

        # 1:1 Padding
        c, h, w = image.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            target = np.pad(target, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            h, w = image.shape[1], image.shape[2]

        # 1:1 Random Cropping
        if self.is_train:
            top = np.random.randint(0, h - self.patch_size + 1)
            left = np.random.randint(0, w - self.patch_size + 1)
        else:
            top = (h - self.patch_size) // 2
            left = (w - self.patch_size) // 2

        image = image[:, top:top + self.patch_size, left:left + self.patch_size]
        target = target[:, top:top + self.patch_size, left:left + self.patch_size]

        return torch.from_numpy(image), torch.from_numpy(target)

# ---------------------------------------------------------
# DATASET 2: Latent Token-Based (TerraMind, Thor)
# Upscaled Spatial Resolution (e.g., 16x16 -> 256x256)
# ---------------------------------------------------------
class LatentTokenDataset(Dataset):
    def __init__(self, file_pairs, patch_size=256, scale_factor=16, is_train=True):
        self.file_pairs = file_pairs
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        emb_path, tar_path = self.file_pairs[idx]

        with rasterio.open(emb_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(tar_path) as src:
            target = src.read().astype(np.float32)

        image, target = np.nan_to_num(image), np.nan_to_num(target)
        target[3, :, :] = np.clip(target[3, :, :] / HEIGHT_NORM_CONSTANT, 0.0, 1.5)

        emb_patch_size = self.patch_size // self.scale_factor

        # Pad Embedding to its specific small size
        c, h_emb, w_emb = image.shape
        if h_emb < emb_patch_size or w_emb < emb_patch_size:
            pad_h = max(0, emb_patch_size - h_emb)
            pad_w = max(0, emb_patch_size - w_emb)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            h_emb, w_emb = image.shape[1], image.shape[2]

        # Pad Target to full size
        _, h_tar, w_tar = target.shape
        if h_tar < self.patch_size or w_tar < self.patch_size:
            pad_h = max(0, self.patch_size - h_tar)
            pad_w = max(0, self.patch_size - w_tar)
            target = np.pad(target, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

        # Multi-scale Cropping
        if self.is_train:
            top_emb = np.random.randint(0, h_emb - emb_patch_size + 1)
            left_emb = np.random.randint(0, w_emb - emb_patch_size + 1)
        else:
            top_emb = (h_emb - emb_patch_size) // 2
            left_emb = (w_emb - emb_patch_size) // 2

        top_tar = top_emb * self.scale_factor
        left_tar = left_emb * self.scale_factor

        image = image[:, top_emb:top_emb + emb_patch_size, left_emb:left_emb + emb_patch_size]
        target = target[:, top_tar:top_tar + self.patch_size, left_tar:left_tar + self.patch_size]

        return torch.from_numpy(image), torch.from_numpy(target)


def find_triple_file_pairs(pixel_dirs, patch_dirs, label_dir):
    """
    Finds matching sets of (pixel_embs, patch_embs, label) using their normalized core IDs.
    Supports a list of pixel directories and patch directories.
    """
    if isinstance(pixel_dirs, str):
        pixel_dirs = [pixel_dirs]
    if isinstance(patch_dirs, str):
        patch_dirs = [patch_dirs]

    # Build maps: {normalized_id: full_path} for each directory
    pixel_maps = []
    for p_dir in pixel_dirs:
        files = glob.glob(os.path.join(p_dir, "**", "*.tif"), recursive=True)
        pixel_maps.append({ _normalize_core_id(f): f for f in files })

    patch_maps = []
    for p_dir in patch_dirs:
        files = glob.glob(os.path.join(p_dir, "**", "*.tif"), recursive=True)
        patch_maps.append({ _normalize_core_id(f): f for f in files })

    label_files = glob.glob(os.path.join(label_dir, "**", "label_*.tif"), recursive=True)
    label_map = { _normalize_core_id(f): f for f in label_files }

    # Intersection of all ID sets to find common IDs
    common_ids = set(label_map.keys())
    for p_map in pixel_maps:
        common_ids &= set(p_map.keys())
    for p_map in patch_maps:
        common_ids &= set(p_map.keys())

    pairs = []
    for cid in sorted(common_ids):
        pixel_paths = [p_map[cid] for p_map in pixel_maps]
        patch_paths = [p_map[cid] for p_map in patch_maps]
        pairs.append((pixel_paths, patch_paths, label_map[cid]))

    return pairs


class Emb2HeightsDataset(Dataset):
    def __init__(self, file_triplets, patch_size=256, scale_factor=16, is_train=True, cache_in_memory=False):
        self.file_triplets = file_triplets
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.cache_in_memory = cache_in_memory
        
        self.shared_pixel_embs = []
        self.shared_patch_embs = []
        self.shared_targets = []
        self.shared_loaded = []

        if self.cache_in_memory:
            print("🔍 Scanning metadata to pre-allocate CPU Shared Memory tensors...")
            
            # Determine channel counts from the first triplet
            first_pixel_paths, first_patch_paths, _ = file_triplets[0]
            pixel_channels = 0
            for p in first_pixel_paths:
                with rasterio.open(p) as src:
                    pixel_channels += src.count
            patch_channels = 0
            for p in first_patch_paths:
                with rasterio.open(p) as src:
                    patch_channels += src.count
            
            from tqdm import tqdm
            # We open only the first path in each list to get spatial height/width
            # This is extremely fast (takes < 2-3 seconds for the entire dataset)
            for pixel_paths, patch_paths, label_path in tqdm(file_triplets, desc="Pre-allocating shared memory"):
                # Get shape of pixel embedding
                with rasterio.open(pixel_paths[0]) as src:
                    h_pix, w_pix = src.height, src.width
                # Get shape of patch embedding
                with rasterio.open(patch_paths[0]) as src:
                    h_pat, w_pat = src.height, src.width
                
                # Calculate padded sizes
                pad_h_pix = max(self.patch_size, h_pix)
                pad_w_pix = max(self.patch_size, w_pix)
                
                emb_patch_size = self.patch_size // self.scale_factor
                pad_h_pat = max(emb_patch_size, h_pat)
                pad_w_pat = max(emb_patch_size, w_pat)
                
                # Pre-allocate shared tensors in float16 to save 50% CPU RAM
                t_pixel = torch.zeros((pixel_channels, pad_h_pix, pad_w_pix), dtype=torch.float16).share_memory_()
                t_patch = torch.zeros((patch_channels, pad_h_pat, pad_w_pat), dtype=torch.float16).share_memory_()
                t_target = torch.zeros((4, pad_h_pix, pad_w_pix), dtype=torch.float16).share_memory_()
                
                self.shared_pixel_embs.append(t_pixel)
                self.shared_patch_embs.append(t_patch)
                self.shared_targets.append(t_target)

            # Use a shared tensor for loaded flags
            self.shared_loaded = torch.zeros(len(file_triplets), dtype=torch.uint8).share_memory_()

    def __len__(self):
        return len(self.file_triplets)

    def __getitem__(self, idx):
        if self.cache_in_memory:
            if self.shared_loaded[idx] == 1:
                pixel_emb = self.shared_pixel_embs[idx]
                patch_emb = self.shared_patch_embs[idx]
                target = self.shared_targets[idx]
            else:
                pixel_paths, patch_paths, label_path = self.file_triplets[idx]

                # Load and pad individual pixel embeddings
                pixel_embs = []
                for p_path in pixel_paths:
                    with rasterio.open(p_path) as src:
                        emb = src.read().astype(np.float32)
                    emb = np.nan_to_num(emb)
                    c_pix, h_pix, w_pix = emb.shape
                    if h_pix < self.patch_size or w_pix < self.patch_size:
                        pad_h = max(0, self.patch_size - h_pix)
                        pad_w = max(0, self.patch_size - w_pix)
                        emb = np.pad(emb, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                    pixel_embs.append(emb)
                pixel_emb_np = np.concatenate(pixel_embs, axis=0)

                # Load and pad individual patch embeddings
                emb_patch_size = self.patch_size // self.scale_factor
                patch_embs = []
                for p_path in patch_paths:
                    with rasterio.open(p_path) as src:
                        emb = src.read().astype(np.float32)
                    emb = np.nan_to_num(emb)
                    c_pat, h_pat, w_pat = emb.shape
                    if h_pat < emb_patch_size or w_pat < emb_patch_size:
                        pad_h = max(0, emb_patch_size - h_pat)
                        pad_w = max(0, emb_patch_size - w_pat)
                        emb = np.pad(emb, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                    patch_embs.append(emb)
                patch_emb_np = np.concatenate(patch_embs, axis=0)

                # Load and pad target
                with rasterio.open(label_path) as src:
                    target_np = src.read().astype(np.float32)
                target_np = np.nan_to_num(target_np)
                
                # Normalize target height channel
                target_np[3, :, :] = np.clip(target_np[3, :, :] / HEIGHT_NORM_CONSTANT, 0.0, 1.5)

                c_tar, h_tar, w_tar = target_np.shape
                if h_tar < self.patch_size or w_tar < self.patch_size:
                    pad_h = max(0, self.patch_size - h_tar)
                    pad_w = max(0, self.patch_size - w_tar)
                    target_np = np.pad(target_np, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

                # Copy in-place to pre-allocated shared tensors (automatically handles cross-worker updates)
                self.shared_pixel_embs[idx].copy_(torch.from_numpy(pixel_emb_np).half())
                self.shared_patch_embs[idx].copy_(torch.from_numpy(patch_emb_np).half())
                self.shared_targets[idx].copy_(torch.from_numpy(target_np).half())
                
                # Mark as loaded in shared memory
                self.shared_loaded[idx] = 1

                pixel_emb = self.shared_pixel_embs[idx]
                patch_emb = self.shared_patch_embs[idx]
                target = self.shared_targets[idx]
        else:
            pixel_paths, patch_paths, label_path = self.file_triplets[idx]

            # Load and pad individual pixel embeddings
            pixel_embs = []
            for p_path in pixel_paths:
                with rasterio.open(p_path) as src:
                    emb = src.read().astype(np.float32)
                emb = np.nan_to_num(emb)
                c_pix, h_pix, w_pix = emb.shape
                if h_pix < self.patch_size or w_pix < self.patch_size:
                    pad_h = max(0, self.patch_size - h_pix)
                    pad_w = max(0, self.patch_size - w_pix)
                    emb = np.pad(emb, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                pixel_embs.append(emb)
            pixel_emb_np = np.concatenate(pixel_embs, axis=0)

            # Load and pad individual patch embeddings
            emb_patch_size = self.patch_size // self.scale_factor
            patch_embs = []
            for p_path in patch_paths:
                with rasterio.open(p_path) as src:
                    emb = src.read().astype(np.float32)
                emb = np.nan_to_num(emb)
                c_pat, h_pat, w_pat = emb.shape
                if h_pat < emb_patch_size or w_pat < emb_patch_size:
                    pad_h = max(0, emb_patch_size - h_pat)
                    pad_w = max(0, emb_patch_size - w_pat)
                    emb = np.pad(emb, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                patch_embs.append(emb)
            patch_emb_np = np.concatenate(patch_embs, axis=0)

            # Load and pad target
            with rasterio.open(label_path) as src:
                target_np = src.read().astype(np.float32)
            target_np = np.nan_to_num(target_np)
            
            # Normalize target height channel
            target_np[3, :, :] = np.clip(target_np[3, :, :] / HEIGHT_NORM_CONSTANT, 0.0, 1.5)

            c_tar, h_tar, w_tar = target_np.shape
            if h_tar < self.patch_size or w_tar < self.patch_size:
                pad_h = max(0, self.patch_size - h_tar)
                pad_w = max(0, self.patch_size - w_tar)
                target_np = np.pad(target_np, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

            pixel_emb = torch.from_numpy(pixel_emb_np)
            patch_emb = torch.from_numpy(patch_emb_np)
            target = torch.from_numpy(target_np)

        h_pix, w_pix = pixel_emb.shape[1], pixel_emb.shape[2]
        h_pat, w_pat = patch_emb.shape[1], patch_emb.shape[2]
        emb_patch_size = self.patch_size // self.scale_factor

        # Alignment-based Multi-scale Cropping
        max_top_pat = h_pat - emb_patch_size
        max_left_pat = w_pat - emb_patch_size

        if self.is_train:
            top_pat = np.random.randint(0, max_top_pat + 1) if max_top_pat > 0 else 0
            left_pat = np.random.randint(0, max_left_pat + 1) if max_left_pat > 0 else 0
        else:
            top_pat = max_top_pat // 2
            left_pat = max_left_pat // 2

        top_pix = top_pat * self.scale_factor
        left_pix = left_pat * self.scale_factor

        # Crop all arrays
        pixel_emb_crop = pixel_emb[:, top_pix:top_pix + self.patch_size, left_pix:left_pix + self.patch_size]
        patch_emb_crop = patch_emb[:, top_pat:top_pat + emb_patch_size, left_pat:left_pat + emb_patch_size]
        target_crop = target[:, top_pix:top_pix + self.patch_size, left_pix:left_pix + self.patch_size]

        # Convert back to float32 if loaded from float16 cache
        if self.cache_in_memory:
            return {
                "pixel_emb": pixel_emb_crop.float(),
                "patch_emb": patch_emb_crop.float(),
                "target": target_crop.float()
            }
        else:
            return {
                "pixel_emb": pixel_emb_crop,
                "patch_emb": patch_emb_crop,
                "target": target_crop
            }
