import os
import socket
import multiprocessing

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

# Define the shared master path for outputs
SHARED_DATA_ROOT = "/mnt/head/users/bassam/data/geofmdata"
SHARED_RUNS_DIR = os.path.join(SHARED_DATA_ROOT, "runs")

# Define EVERY possible local path across your cluster
LOCAL_PATHS = [
    "/mnt/n1/users/bassam/data/geofmdata",
    "/mnt/n2/users/bassam/data/geofmdata",
    "/mnt/n3/users/bassam/data/geofmdata",
    "/mnt/head/users/bassam/data/geofmdata" 
]

if not os.path.exists(SHARED_RUNS_DIR):
    try:
        os.makedirs(SHARED_RUNS_DIR, exist_ok=True)
    except Exception:
        pass

# The Auto-Router: Find the path that exists on THIS specific machine
TARGET_DRIVE = SHARED_DATA_ROOT  # Default to the slow shared drive just in case
is_fast_local = False

for path in LOCAL_PATHS:
    if os.path.exists(path):
        TARGET_DRIVE = path
        is_fast_local = True
        break

if multiprocessing.current_process().name == 'MainProcess':
    if is_fast_local:
        print(f"⚡ [{socket.gethostname()}] Fast data path selected: {TARGET_DRIVE}")
    else:
        print(f"⚠️ [{socket.gethostname()}] Local cache missing. Falling back to slow NFS: {TARGET_DRIVE}")

# Now build your embedding directories based on whatever TARGET_DRIVE became
# Ensuring we include the /data/ intermediate folder to match our architecture
BASE_EMB_DIR = os.path.join(TARGET_DRIVE, "embed2heights", "data")

# Paths to training data subdirectories
ALPHA_EARTH_DIR = os.path.join(BASE_EMB_DIR, "train", "alphaearth_emb")
TESSERA_DIR = os.path.join(BASE_EMB_DIR, "train", "tessera_emb")
TERRAMIND_S1_DIR = os.path.join(BASE_EMB_DIR, "train", "terramind_s1_emb")
TERRAMIND_S2_DIR = os.path.join(BASE_EMB_DIR, "train", "terramind_s2_emb")
THOR_S1_DIR = os.path.join(BASE_EMB_DIR, "train", "thor_s1_emb")
THOR_S2_DIR = os.path.join(BASE_EMB_DIR, "train", "thor_s2_emb")
LABELS_DIR = os.path.join(BASE_EMB_DIR, "train", "labels")

# Paths to test data subdirectories (for cleaner predict command lines)
ALPHA_EARTH_TEST_DIR = os.path.join(BASE_EMB_DIR, "test", "alphaearth_test_emb")
TESSERA_TEST_DIR = os.path.join(BASE_EMB_DIR, "test", "tessera_test_emb")
TERRAMIND_S1_TEST_DIR = os.path.join(BASE_EMB_DIR, "test", "terramind_test_s1_emb")
TERRAMIND_S2_TEST_DIR = os.path.join(BASE_EMB_DIR, "test", "terramind_test_s2_emb")
THOR_S1_TEST_DIR = os.path.join(BASE_EMB_DIR, "test", "thor_test_s1_emb")
THOR_S2_TEST_DIR = os.path.join(BASE_EMB_DIR, "test", "thor_test_s2_emb")

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
