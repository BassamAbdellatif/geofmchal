import os
import socket

# Raise file descriptor limit dynamically to prevent "Too many open files" errors
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(hard, 65536) if hard != resource.RLIM_INFINITY else 65536
    new_soft = max(soft, new_soft)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
except Exception:
    pass

# Optimize threading and GDAL/Rasterio caching to prevent memory leaks/bloat
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["GDAL_CACHEMAX"] = "64"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["GDAL_SHARED_FILE_LIMIT"] = "50"

from dataclasses import dataclass

LOCAL_DATA_ROOT = "/scratch/geofm_data"
HEAD_DATA_ROOT = "/mnt/head/users/bassam/data/geofmdata/embed2heights/data"
HEAD_RUNS_DIR = "/mnt/head/users/bassam/data/geofmdata/runs"
SHARED_DATA_ROOT = "/mnt/shared/geofm_data"

# Resolve the data root: fast local NVMe first, then the head-node NFS (which is
# effectively local on the head), then the generic shared NFS as a last resort.
if os.path.exists(LOCAL_DATA_ROOT):
    TARGET_DRIVE = LOCAL_DATA_ROOT
    SHARED_RUNS_DIR = "/mnt/shared/geofm_data/runs"
    _route_msg = f"[{socket.gethostname()}] Routing data to local NVMe -> {TARGET_DRIVE}"
elif os.path.exists(HEAD_DATA_ROOT):
    TARGET_DRIVE = HEAD_DATA_ROOT
    SHARED_RUNS_DIR = HEAD_RUNS_DIR
    _route_msg = f"[{socket.gethostname()}] Routing data to head NFS -> {TARGET_DRIVE}"
else:
    TARGET_DRIVE = SHARED_DATA_ROOT
    SHARED_RUNS_DIR = "/mnt/shared/geofm_data/runs"
    _route_msg = f"[{socket.gethostname()}] WARNING: no local/head data root found, using {TARGET_DRIVE}"

try:
    os.makedirs(SHARED_RUNS_DIR, exist_ok=True)
except Exception:
    pass

# Print the routing line once per process tree. DataLoader workers spawned after
# this import inherit GEOFM_CONFIG_QUIET=1 and stay silent, so the message no
# longer repeats per worker.
if not os.environ.get("GEOFM_CONFIG_QUIET"):
    print(_route_msg)
    os.environ["GEOFM_CONFIG_QUIET"] = "1"

# Paths to training data subdirectories
ALPHA_EARTH_DIR = os.path.join(TARGET_DRIVE, "train", "alphaearth_emb")
TESSERA_DIR = os.path.join(TARGET_DRIVE, "train", "tessera_emb")
TERRAMIND_S1_DIR = os.path.join(TARGET_DRIVE, "train", "terramind_s1_emb")
TERRAMIND_S2_DIR = os.path.join(TARGET_DRIVE, "train", "terramind_s2_emb")
THOR_S1_DIR = os.path.join(TARGET_DRIVE, "train", "thor_s1_emb")
THOR_S2_DIR = os.path.join(TARGET_DRIVE, "train", "thor_s2_emb")
LABELS_DIR = os.path.join(TARGET_DRIVE, "train", "labels")

# Paths to test data subdirectories (for cleaner predict command lines)
ALPHA_EARTH_TEST_DIR = os.path.join(TARGET_DRIVE, "test", "alphaearth_test_emb")
TESSERA_TEST_DIR = os.path.join(TARGET_DRIVE, "test", "tessera_test_emb")
TERRAMIND_S1_TEST_DIR = os.path.join(TARGET_DRIVE, "test", "terramind_test_s1_emb")
TERRAMIND_S2_TEST_DIR = os.path.join(TARGET_DRIVE, "test", "terramind_test_s2_emb")
THOR_S1_TEST_DIR = os.path.join(TARGET_DRIVE, "test", "thor_test_s1_emb")
THOR_S2_TEST_DIR = os.path.join(TARGET_DRIVE, "test", "thor_test_s2_emb")

# Default Hyperparameters
PATCH_SIZE = 256
BATCH_SIZE = 32
RUNS_DIR = SHARED_RUNS_DIR

@dataclass
class Config:
    TARGET_DRIVE: str = TARGET_DRIVE
    ALPHA_EARTH_DIR: str = ALPHA_EARTH_DIR
    TESSERA_DIR: str = TESSERA_DIR
    TERRAMIND_S1_DIR: str = TERRAMIND_S1_DIR
    TERRAMIND_S2_DIR: str = TERRAMIND_S2_DIR
    THOR_S1_DIR: str = THOR_S1_DIR
    THOR_S2_DIR: str = THOR_S2_DIR
    LABELS_DIR: str = LABELS_DIR
    ALPHA_EARTH_TEST_DIR: str = ALPHA_EARTH_TEST_DIR
    TESSERA_TEST_DIR: str = TESSERA_TEST_DIR
    TERRAMIND_S1_TEST_DIR: str = TERRAMIND_S1_TEST_DIR
    TERRAMIND_S2_TEST_DIR: str = TERRAMIND_S2_TEST_DIR
    THOR_S1_TEST_DIR: str = THOR_S1_TEST_DIR
    THOR_S2_TEST_DIR: str = THOR_S2_TEST_DIR
    PATCH_SIZE: int = PATCH_SIZE
    BATCH_SIZE: int = BATCH_SIZE
    RUNS_DIR: str = RUNS_DIR
