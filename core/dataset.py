import os
import glob
import re
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

HEIGHT_NORM_CONSTANT = 30.0


def augment_triplet(pixel_emb, patch_emb, target):
    """
    Applies identical random spatial augmentations to all three tensors,
    maintaining perfect spatial alignment between the 256x256 pixel grid
    and the 16x16 patch grid.

    Args:
        pixel_emb : [C_pix, H, W]       — full-resolution pixel embedding
        patch_emb : [C_pat, H//16, W//16] — downscaled patch embedding
        target    : [4, H, W]           — ground-truth label

    Returns:
        Tuple of (pixel_emb, patch_emb, target) after augmentation.
    """
    # --- Random Horizontal Flip (50% probability) ---
    if torch.rand(1).item() < 0.5:
        pixel_emb = torch.flip(pixel_emb, dims=[2])
        patch_emb = torch.flip(patch_emb, dims=[2])
        target    = torch.flip(target,    dims=[2])

    # --- Random Vertical Flip (50% probability) ---
    if torch.rand(1).item() < 0.5:
        pixel_emb = torch.flip(pixel_emb, dims=[1])
        patch_emb = torch.flip(patch_emb, dims=[1])
        target    = torch.flip(target,    dims=[1])

    # --- Random 90-degree Rotation (0 / 90 / 180 / 270 degrees) ---
    # rot90 on dims=[1,2] works on [C, H, W] tensors.
    # The same k applies to both grids — alignment is preserved because
    # rotating 90° on a 16x16 grid is the correct transform for a 256x256 grid.
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        pixel_emb = torch.rot90(pixel_emb, k, dims=[1, 2])
        patch_emb = torch.rot90(patch_emb, k, dims=[1, 2])
        target    = torch.rot90(target,    k, dims=[1, 2])

    return pixel_emb, patch_emb, target

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


VALID_CACHE_CATEGORIES = frozenset({"pixel", "patch", "target"})


def _parse_cache_spec(spec):
    """Normalise cache_in_memory= bool/str/None into a frozenset of categories."""
    if spec in (None, False):
        return frozenset()
    if spec is True or spec == "all":
        return VALID_CACHE_CATEGORIES
    if isinstance(spec, str):
        parts = {s.strip() for s in spec.split(",") if s.strip()}
        if "all" in parts:
            return VALID_CACHE_CATEGORIES
        unknown = parts - VALID_CACHE_CATEGORIES
        if unknown:
            raise ValueError(
                f"Unknown cache categories: {sorted(unknown)}. "
                f"Valid: {sorted(VALID_CACHE_CATEGORIES)} or 'all'."
            )
        return frozenset(parts)
    raise TypeError(f"cache_in_memory must be bool/str/None, got {type(spec).__name__}")


class Emb2HeightsDataset(Dataset):
    def __init__(self, file_triplets, patch_size=256, scale_factor=16, is_train=True, cache_in_memory=False, augment=False):
        self.file_triplets = file_triplets
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.is_train = is_train
        self.augment = augment

        self._cache_set = _parse_cache_spec(cache_in_memory)
        self.cache_in_memory = bool(self._cache_set)  # legacy boolean kept for callers

        self.shared_pixel_embs = None
        self.shared_patch_embs = None
        self.shared_targets = None
        self.shared_pixel_loaded = None
        self.shared_patch_loaded = None
        self.shared_target_loaded = None

        if not self.cache_in_memory:
            return

        print(f"🔍 Pre-allocating CPU shared memory for: {sorted(self._cache_set)}")

        first_pixel_paths, first_patch_paths, _ = file_triplets[0]
        pixel_channels = patch_channels = None
        if "pixel" in self._cache_set:
            pixel_channels = sum(rasterio.open(p).count for p in first_pixel_paths)
            self.shared_pixel_embs = []
        if "patch" in self._cache_set:
            patch_channels = sum(rasterio.open(p).count for p in first_patch_paths)
            self.shared_patch_embs = []
        if "target" in self._cache_set:
            self.shared_targets = []

        from tqdm import tqdm
        emb_patch_size = self.patch_size // self.scale_factor
        for pixel_paths, patch_paths, _label_path in tqdm(file_triplets, desc="Pre-allocating shared memory"):
            # Spatial sizes (read only what's needed for the categories we cache)
            if "pixel" in self._cache_set or "target" in self._cache_set:
                with rasterio.open(pixel_paths[0]) as src:
                    h_pix, w_pix = src.height, src.width
                pad_h_pix = max(self.patch_size, h_pix)
                pad_w_pix = max(self.patch_size, w_pix)
            if "patch" in self._cache_set:
                with rasterio.open(patch_paths[0]) as src:
                    h_pat, w_pat = src.height, src.width
                pad_h_pat = max(emb_patch_size, h_pat)
                pad_w_pat = max(emb_patch_size, w_pat)

            if "pixel" in self._cache_set:
                self.shared_pixel_embs.append(
                    torch.zeros((pixel_channels, pad_h_pix, pad_w_pix), dtype=torch.float16).share_memory_()
                )
            if "patch" in self._cache_set:
                self.shared_patch_embs.append(
                    torch.zeros((patch_channels, pad_h_pat, pad_w_pat), dtype=torch.float16).share_memory_()
                )
            if "target" in self._cache_set:
                self.shared_targets.append(
                    torch.zeros((4, pad_h_pix, pad_w_pix), dtype=torch.float16).share_memory_()
                )

        n = len(file_triplets)
        if "pixel" in self._cache_set:
            self.shared_pixel_loaded = torch.zeros(n, dtype=torch.uint8).share_memory_()
        if "patch" in self._cache_set:
            self.shared_patch_loaded = torch.zeros(n, dtype=torch.uint8).share_memory_()
        if "target" in self._cache_set:
            self.shared_target_loaded = torch.zeros(n, dtype=torch.uint8).share_memory_()

    def __len__(self):
        return len(self.file_triplets)

    def _load_pixel(self, idx):
        """Returns (tensor, is_half). Uses cache if present, fills cache on first miss."""
        if "pixel" in self._cache_set and self.shared_pixel_loaded[idx] == 1:
            return self.shared_pixel_embs[idx], True

        pixel_paths = self.file_triplets[idx][0]
        embs = []
        for p in pixel_paths:
            with rasterio.open(p) as src:
                a = src.read().astype(np.float32)
            a = np.nan_to_num(a)
            _, h, w = a.shape
            if h < self.patch_size or w < self.patch_size:
                a = np.pad(a, ((0, 0), (0, max(0, self.patch_size - h)), (0, max(0, self.patch_size - w))), mode="reflect")
            embs.append(a)
        t = torch.from_numpy(np.concatenate(embs, axis=0))

        if "pixel" in self._cache_set:
            self.shared_pixel_embs[idx].copy_(t.half())
            self.shared_pixel_loaded[idx] = 1
            return self.shared_pixel_embs[idx], True
        return t, False

    def _load_patch(self, idx):
        if "patch" in self._cache_set and self.shared_patch_loaded[idx] == 1:
            return self.shared_patch_embs[idx], True

        patch_paths = self.file_triplets[idx][1]
        emb_patch_size = self.patch_size // self.scale_factor
        embs = []
        for p in patch_paths:
            with rasterio.open(p) as src:
                a = src.read().astype(np.float32)
            a = np.nan_to_num(a)
            a = np.clip(a, -30.0, 30.0)  # terramind_s2 has channel outliers ~-330; clip before model
            _, h, w = a.shape
            if h < emb_patch_size or w < emb_patch_size:
                a = np.pad(a, ((0, 0), (0, max(0, emb_patch_size - h)), (0, max(0, emb_patch_size - w))), mode="reflect")
            embs.append(a)
        t = torch.from_numpy(np.concatenate(embs, axis=0))

        if "patch" in self._cache_set:
            self.shared_patch_embs[idx].copy_(t.half())
            self.shared_patch_loaded[idx] = 1
            return self.shared_patch_embs[idx], True
        return t, False

    def _load_target(self, idx):
        if "target" in self._cache_set and self.shared_target_loaded[idx] == 1:
            return self.shared_targets[idx], True

        label_path = self.file_triplets[idx][2]
        with rasterio.open(label_path) as src:
            arr = src.read().astype(np.float32)
        arr = np.nan_to_num(arr)
        arr[3, :, :] = np.clip(arr[3, :, :] / HEIGHT_NORM_CONSTANT, 0.0, 1.5)
        _, h, w = arr.shape
        if h < self.patch_size or w < self.patch_size:
            arr = np.pad(arr, ((0, 0), (0, max(0, self.patch_size - h)), (0, max(0, self.patch_size - w))), mode="reflect")
        t = torch.from_numpy(arr)

        if "target" in self._cache_set:
            self.shared_targets[idx].copy_(t.half())
            self.shared_target_loaded[idx] = 1
            return self.shared_targets[idx], True
        return t, False

    def __getitem__(self, idx):
        pixel_emb, pix_half = self._load_pixel(idx)
        patch_emb, pat_half = self._load_patch(idx)
        target,    tar_half = self._load_target(idx)

        h_pat, w_pat = patch_emb.shape[1], patch_emb.shape[2]
        emb_patch_size = self.patch_size // self.scale_factor

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

        pixel_emb_crop = pixel_emb[:, top_pix:top_pix + self.patch_size, left_pix:left_pix + self.patch_size]
        patch_emb_crop = patch_emb[:, top_pat:top_pat + emb_patch_size, left_pat:left_pat + emb_patch_size]
        target_crop    = target[:,    top_pix:top_pix + self.patch_size, left_pix:left_pix + self.patch_size]

        # Promote float16 cached tensors back to float32 for downstream training
        if pix_half: pixel_emb_crop = pixel_emb_crop.float()
        if pat_half: patch_emb_crop = patch_emb_crop.float()
        if tar_half: target_crop    = target_crop.float()

        if self.augment:
            pixel_emb_crop, patch_emb_crop, target_crop = augment_triplet(
                pixel_emb_crop, patch_emb_crop, target_crop
            )

        return {
            "pixel_emb": pixel_emb_crop,
            "patch_emb": patch_emb_crop,
            "target":    target_crop,
        }
