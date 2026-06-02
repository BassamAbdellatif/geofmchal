import os
import glob
import json
import re
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

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
    for prefix in ("gee_emb_", "tessera_emb_", "s2_", "s1_", "emb_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break

    # 3. Strip trailing embedding suffixes (if any)
    for suf in ("_embeddings", "_embedding", "_merged", "_quantized"):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break   

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


# =============================================================================
# 7A — Multi-modal dataset with geographic CV, stratified sampling,
#       domain-shift augmentation, and D4 geometric augmentation.
# =============================================================================

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_GEO_FOLDS = os.path.join(_REPO_ROOT, "data", "geo_folds.json")
_DEFAULT_NORM_STATS = os.path.join(_REPO_ROOT, "data", "norm_stats.json")

# Strata weights for WeightedRandomSampler: index = stratum (0=empty, 1=sparse, 2=medium, 3=dense)
_STRATA_WEIGHTS = [1.0, 1.5, 2.0, 3.0]


def find_multimodal_train_tiles(data_root):
    """
    Match tiles across alpha_earth, tessera, terramind_s1, terramind_s2, and labels.
    Returns a list of dicts: {core_id, alpha_path, tessera_path, tm_s1_path, tm_s2_path, label_path}
    """
    train_dir = os.path.join(data_root, "train")
    dirs = {
        "alpha":   os.path.join(train_dir, "alphaearth_emb"),
        "tessera": os.path.join(train_dir, "tessera_emb"),
        "tm_s1":   os.path.join(train_dir, "terramind_s1_emb"),
        "tm_s2":   os.path.join(train_dir, "terramind_s2_emb"),
        "label":   os.path.join(train_dir, "labels"),
    }

    id_maps = {}
    for key, d in dirs.items():
        files = glob.glob(os.path.join(d, "*.tif"))
        id_maps[key] = {_normalize_core_id(f): f for f in files}

    common = set(id_maps["alpha"].keys())
    for key in ("tessera", "tm_s1", "tm_s2", "label"):
        common &= set(id_maps[key].keys())

    tiles = []
    for cid in sorted(common):
        tiles.append({
            "core_id":      cid,
            "alpha_path":   id_maps["alpha"][cid],
            "tessera_path": id_maps["tessera"][cid],
            "tm_s1_path":   id_maps["tm_s1"][cid],
            "tm_s2_path":   id_maps["tm_s2"][cid],
            "label_path":   id_maps["label"][cid],
        })
    return tiles


def find_multimodal_test_tiles(data_root, use_thor=False):
    """
    Match test tiles across all pixel and patch modalities.
    Returns list of dicts keyed by modality name.
    """
    test_dir = os.path.join(data_root, "test")
    dirs = {
        "alpha":   os.path.join(test_dir, "alphaearth_test_emb"),
        "tessera": os.path.join(test_dir, "tessera_test_emb"),
        "tm_s1":   os.path.join(test_dir, "terramind_test_s1_emb"),
        "tm_s2":   os.path.join(test_dir, "terramind_test_s2_emb"),
    }
    if use_thor:
        dirs["thor_s1"] = os.path.join(test_dir, "thor_test_s1_emb")
        dirs["thor_s2"] = os.path.join(test_dir, "thor_test_s2_emb")

    id_maps = {}
    for key, d in dirs.items():
        files = glob.glob(os.path.join(d, "*.tif"))
        id_maps[key] = {_normalize_core_id(f): f for f in files}

    common = set(id_maps["alpha"].keys())
    for key in id_maps:
        common &= set(id_maps[key].keys())

    tiles = []
    for cid in sorted(common):
        entry = {"core_id": cid}
        for key in id_maps:
            entry[key + "_path"] = id_maps[key][cid]
        tiles.append(entry)
    return tiles


def _load_norm_stats(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _sanitize(arr, clip_mag=1e6):
    """
    Replace NaN/Inf and corrupt sentinel values with 0. At least one tessera
    tile (tessera_emb_1753_QE) carries a ~6.6e36 sentinel that nan_to_num does
    not catch; magnitudes above clip_mag are treated as corruption and zeroed.
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr[np.abs(arr) > clip_mag] = 0.0
    return arr


def _pad_to(arr, size=256):
    """
    Reflect-pad a (C, H, W) array up to (C, size, size). ~13% of the pixel
    tiles are 255 in one or both spatial dims; padding keeps pixel streams and
    the label aligned (they share identical dims per tile).
    """
    _, h, w = arr.shape
    if h == size and w == size:
        return arr
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")


def _apply_norm(arr, mean, std, channel_axis=0):
    """
    Normalise a numpy array using per-channel mean/std vectors.
    channel_axis=0  for pixel streams (C, H, W).
    channel_axis=-1 for patch tokens   (N, C)  where the feature dim is last.
    """
    mean = np.asarray(mean, dtype=np.float32)
    std  = np.asarray(std,  dtype=np.float32)
    shape = [1] * arr.ndim
    shape[channel_axis] = arr.shape[channel_axis]
    mean = mean.reshape(shape)
    std  = std.reshape(shape)
    return (arr - mean) / std


def augment_domain_shift(emb_dict, norm_stats, rng):
    """
    Embedding-space domain-shift augmentation (training only).
    Pixel streams: channel gain [0.9,1.1], offset [-0.1,0.1], channel dropout p=0.1,
                   Gaussian noise σ = 0.05 × per-channel std.
    Patch tokens: gain + dropout only.
    """
    for key in ("alpha_earth", "tessera"):
        if key not in emb_dict:
            continue
        emb = emb_dict[key]          # (C, H, W) float32
        C = emb.shape[0]
        if rng.random() < 0.5:
            gain   = rng.uniform(0.9, 1.1, size=(C, 1, 1)).astype(np.float32)
            offset = rng.uniform(-0.1, 0.1, size=(C, 1, 1)).astype(np.float32)
            emb = emb * gain + offset
        drop_mask = (rng.random(size=(C, 1, 1)) > 0.1).astype(np.float32)
        emb = emb * drop_mask
        if rng.random() < 0.5 and norm_stats and key in norm_stats:
            std_per_ch = np.array(norm_stats[key]["std"], dtype=np.float32).reshape(C, 1, 1)
            noise = rng.normal(0, 0.05 * std_per_ch, size=emb.shape).astype(np.float32)
            emb = emb + noise
        emb_dict[key] = emb

    for key in ("terramind_s1", "terramind_s2"):
        if key not in emb_dict:
            continue
        emb = emb_dict[key]          # (N, D) float32  N=256, D=768
        D = emb.shape[1]
        if rng.random() < 0.5:
            gain = rng.uniform(0.9, 1.1, size=(1, D)).astype(np.float32)
            emb = emb * gain
        drop_mask = (rng.random(size=(1, D)) > 0.1).astype(np.float32)
        emb = emb * drop_mask
        emb_dict[key] = emb

    return emb_dict


def _d4_transform_pixel(arr, k, flip):
    """Apply rot90 (k times) + optional h-flip to a (C, H, W) array."""
    arr = np.rot90(arr, k=k, axes=(1, 2))
    if flip:
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def _d4_transform_patch_grid(arr, k, flip):
    """
    Apply the same D4 transform to a (256, 768) patch-token array by treating
    256 = 16×16 as a spatial grid, permuting positions, and flattening back.
    """
    grid = arr.reshape(16, 16, -1)        # (16, 16, D)
    grid = np.rot90(grid, k=k, axes=(0, 1))
    if flip:
        grid = grid[:, ::-1, :]
    return np.ascontiguousarray(grid.reshape(256, -1))


class GeoFMDataset7A(Dataset):
    """
    7A multi-modal dataset.

    Output dict per sample:
        alpha_earth:  Tensor(64,  256, 256)
        tessera:      Tensor(128, 256, 256)
        terramind_s1: Tensor(256, 768)
        terramind_s2: Tensor(256, 768)
        target:       Tensor(4,   256, 256)
    """

    def __init__(
        self,
        tiles,
        is_train=True,
        geo_folds_path=_DEFAULT_GEO_FOLDS,
        norm_stats_path=_DEFAULT_NORM_STATS,
        cv_fold=0,
        cache_dir=None,
        rebuild_cache=False,
    ):
        with open(geo_folds_path) as f:
            folds = json.load(f)

        if is_train:
            self.tiles = [t for t in tiles if folds.get(t["core_id"], cv_fold) != cv_fold]
        else:
            self.tiles = [t for t in tiles if folds.get(t["core_id"], cv_fold) == cv_fold]

        self.is_train = is_train
        self.norm_stats = _load_norm_stats(norm_stats_path)

        self._coverage_scores = self._compute_coverage_scores()

        # Optional memmap-backed float16 cache of the deterministic preprocessed
        # tiles (sanitize+pad+reshape+norm). Built once; workers mmap it
        # read-only so the OS page cache holds a single shared copy (spawn-safe,
        # no per-worker duplication). Augmentation is still applied per __getitem__.
        self._cache = None
        if cache_dir is not None:
            tag = f"fold{cv_fold}_{'train' if is_train else 'val'}"
            self._init_cache(cache_dir, tag, rebuild_cache)

    # ------------------------------------------------------------------ caching
    _CACHE_SPECS = {
        "alpha_earth":  (64, 256, 256),
        "tessera":      (128, 256, 256),
        "terramind_s1": (256, 768),
        "terramind_s2": (256, 768),
        "target":       (4, 256, 256),
    }

    def _preprocess_tile(self, tile):
        """Deterministic read -> sanitize -> pad -> reshape -> norm. Returns f32 dict.
        This is the cacheable part; augmentation happens after, per __getitem__."""
        with rasterio.open(tile["alpha_path"]) as src:
            alpha = src.read().astype(np.float32)
        with rasterio.open(tile["tessera_path"]) as src:
            tessera = src.read().astype(np.float32)
        with rasterio.open(tile["tm_s1_path"]) as src:
            tm_s1 = src.read().astype(np.float32)
        with rasterio.open(tile["tm_s2_path"]) as src:
            tm_s2 = src.read().astype(np.float32)
        with rasterio.open(tile["label_path"]) as src:
            target = src.read().astype(np.float32)

        alpha   = _pad_to(_sanitize(alpha))
        tessera = _pad_to(_sanitize(tessera))
        tm_s1   = _sanitize(tm_s1).reshape(768, -1).T      # (256,768)
        tm_s2   = _sanitize(tm_s2).reshape(768, -1).T
        target  = _pad_to(np.nan_to_num(target))
        target[3] = np.clip(target[3] / HEIGHT_NORM_CONSTANT, 0.0, 1.5)

        if self.norm_stats:
            alpha   = _apply_norm(alpha,   self.norm_stats["alpha_earth"]["mean"],  self.norm_stats["alpha_earth"]["std"])
            tessera = _apply_norm(tessera, self.norm_stats["tessera"]["mean"],      self.norm_stats["tessera"]["std"])
            tm_s1   = _apply_norm(tm_s1,   self.norm_stats["terramind_s1"]["mean"], self.norm_stats["terramind_s1"]["std"], channel_axis=-1)
            tm_s2   = _apply_norm(tm_s2,   self.norm_stats["terramind_s2"]["mean"], self.norm_stats["terramind_s2"]["std"], channel_axis=-1)

        return {"alpha_earth": alpha, "tessera": tessera,
                "terramind_s1": tm_s1, "terramind_s2": tm_s2, "target": target}

    def _init_cache(self, cache_dir, tag, rebuild):
        os.makedirs(cache_dir, exist_ok=True)
        n = len(self.tiles)
        paths = {k: os.path.join(cache_dir, f"{tag}_{k}.f16.npy")
                 for k in self._CACHE_SPECS}
        done_flag = os.path.join(cache_dir, f"{tag}.done")

        need_build = rebuild or not os.path.exists(done_flag) or \
            any(not os.path.exists(p) for p in paths.values())

        if need_build:
            from tqdm import tqdm
            if os.path.exists(done_flag):
                os.remove(done_flag)
            mm = {k: np.lib.format.open_memmap(
                      paths[k], mode="w+", dtype=np.float16,
                      shape=(n, *self._CACHE_SPECS[k]))
                  for k in self._CACHE_SPECS}
            gb = sum(np.prod((n, *s)) for s in self._CACHE_SPECS.values()) * 2 / 1e9
            print(f"[cache] building {tag}: {n} tiles, ~{gb:.1f} GB f16 -> {cache_dir}")
            for i, tile in enumerate(tqdm(self.tiles, desc=f"cache {tag}", leave=False)):
                pp = self._preprocess_tile(tile)
                for k in self._CACHE_SPECS:
                    mm[k][i] = pp[k].astype(np.float16)
            for k in mm:
                mm[k].flush()
            del mm
            open(done_flag, "w").write(f"{n}\n")
            print(f"[cache] {tag} built.")

        # Store paths only; the read-only memmaps are opened lazily, once per
        # process (see _ensure_cache). They must NOT live in the pickled dataset
        # state: under the 'spawn' start method the DataLoader pickles the whole
        # dataset to each worker, and pickling a np.memmap serialises the FULL
        # array by value (~tens of GB) through a pipe, giving every worker its
        # own in-RAM copy and OOM-ing the node. Opening lazily in each worker
        # instead keeps a single shared copy in the OS page cache.
        self._cache_paths = dict(paths)
        self._cache = None

    def _ensure_cache(self):
        """Open the read-only f16 memmaps lazily, once per process."""
        if self._cache is None and getattr(self, "_cache_paths", None):
            self._cache = {k: np.load(p, mmap_mode="r")
                           for k, p in self._cache_paths.items()}
        return self._cache

    def __getstate__(self):
        # Never pickle live memmap handles into spawned workers; they are
        # reopened lazily per process from self._cache_paths.
        state = self.__dict__.copy()
        state["_cache"] = None
        return state

    def _compute_coverage_scores(self):
        """
        Continuous coverage = mean(building_fraction) + mean(water_fraction)
        over the 256x256 label map (unthresholded). Smoother than the
        thresholded spec version, which left ~92% of tiles at exactly 0.
        Building/water are the rare, hard classes (IoU_B/IoU_W are our worst
        metrics), so we enrich tiles that contain more of them.
        """
        scores = []
        for tile in self.tiles:
            with rasterio.open(tile["label_path"]) as src:
                lbl = src.read().astype(np.float32)
            build = float(np.mean(lbl[0]))
            water = float(np.mean(lbl[2]))
            scores.append(build + water)
        return scores

    def sampler_weights(self):
        """
        Per-sample weights for WeightedRandomSampler.
        Strata cut at quartiles of the continuous coverage distribution so
        each of the 4 strata holds a real population (~equal counts), then
        weighted [1.0, 1.5, 2.0, 3.0] from low to high coverage.
        """
        scores = np.asarray(self._coverage_scores, dtype=np.float64)
        # Stratum 0 = zero coverage. The positive part is split into 3 tertiles
        # (strata 1,2,3) so all four strata hold a real, roughly-equal population.
        pos = scores[scores > 0]
        if pos.size >= 3:
            q = np.quantile(pos, [1.0 / 3.0, 2.0 / 3.0])
        else:
            q = np.array([1e-6, 2e-6])

        def _stratum(s):
            if s <= 0:
                return 0
            return int(1 + np.searchsorted(q, s, side="right"))  # -> 1, 2, or 3

        self._stratum_edges = [0.0] + q.tolist()
        return [_STRATA_WEIGHTS[_stratum(s)] for s in scores]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        rng = np.random.default_rng()

        cache = self._ensure_cache()
        if cache is not None:
            # Read the preprocessed f16 tile from the shared memmap; upcast to f32
            # for augmentation/compute. (norm/sanitize/pad already baked in.)
            alpha   = np.asarray(cache["alpha_earth"][idx],  dtype=np.float32)
            tessera = np.asarray(cache["tessera"][idx],      dtype=np.float32)
            tm_s1   = np.asarray(cache["terramind_s1"][idx], dtype=np.float32)
            tm_s2   = np.asarray(cache["terramind_s2"][idx], dtype=np.float32)
            target  = np.asarray(cache["target"][idx],       dtype=np.float32)
        else:
            pp = self._preprocess_tile(tile)
            alpha, tessera = pp["alpha_earth"], pp["tessera"]
            tm_s1, tm_s2, target = pp["terramind_s1"], pp["terramind_s2"], pp["target"]

        if self.is_train:
            emb_dict = {
                "alpha_earth":  alpha,
                "tessera":      tessera,
                "terramind_s1": tm_s1,
                "terramind_s2": tm_s2,
            }
            emb_dict = augment_domain_shift(emb_dict, self.norm_stats, rng)

            k    = int(rng.integers(0, 4))
            flip = bool(rng.integers(0, 2))
            emb_dict["alpha_earth"]  = _d4_transform_pixel(emb_dict["alpha_earth"],  k, flip)
            emb_dict["tessera"]      = _d4_transform_pixel(emb_dict["tessera"],      k, flip)
            emb_dict["terramind_s1"] = _d4_transform_patch_grid(emb_dict["terramind_s1"], k, flip)
            emb_dict["terramind_s2"] = _d4_transform_patch_grid(emb_dict["terramind_s2"], k, flip)
            target = _d4_transform_pixel(target, k, flip)

            alpha   = emb_dict["alpha_earth"]
            tessera = emb_dict["tessera"]
            tm_s1   = emb_dict["terramind_s1"]
            tm_s2   = emb_dict["terramind_s2"]

        return {
            "alpha_earth":  torch.from_numpy(alpha),
            "tessera":      torch.from_numpy(tessera),
            "terramind_s1": torch.from_numpy(tm_s1),
            "terramind_s2": torch.from_numpy(tm_s2),
            "target":       torch.from_numpy(target),
        }
