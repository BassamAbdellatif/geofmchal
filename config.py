import os

# Automatically set MKL threading layer to avoid conflicts with OpenMP
os.environ["MKL_THREADING_LAYER"] = "GNU"

from dataclasses import dataclass

# Root path to the data
TARGET_DRIVE = "/mnt/head/users/bassam/data/geofmdata/embed2heights/data"

# Paths to training data subdirectories
TESSERA_DIR = os.path.join(TARGET_DRIVE, "train", "tessera_emb")
TERRAMIND_S1_DIR = os.path.join(TARGET_DRIVE, "train", "terramind_s1_emb")
TERRAMIND_S2_DIR = os.path.join(TARGET_DRIVE, "train", "terramind_s2_emb")
THOR_S1_DIR = os.path.join(TARGET_DRIVE, "train", "thor_s1_emb")
THOR_S2_DIR = os.path.join(TARGET_DRIVE, "train", "thor_s2_emb")
LABELS_DIR = os.path.join(TARGET_DRIVE, "train", "labels")

# Paths to test data subdirectories (for cleaner predict command lines)
TESSERA_TEST_DIR = os.path.join(TARGET_DRIVE, "test", "tessera_test_emb")
TERRAMIND_S1_TEST_DIR = os.path.join(TARGET_DRIVE, "test", "terramind_test_s1_emb")
TERRAMIND_S2_TEST_DIR = os.path.join(TARGET_DRIVE, "test", "terramind_test_s2_emb")
THOR_S1_TEST_DIR = os.path.join(TARGET_DRIVE, "test", "thor_test_s1_emb")
THOR_S2_TEST_DIR = os.path.join(TARGET_DRIVE, "test", "thor_test_s2_emb")

# Default Hyperparameters
PATCH_SIZE = 256
BATCH_SIZE = 16

@dataclass
class Config:
    TARGET_DRIVE: str = TARGET_DRIVE
    TESSERA_DIR: str = TESSERA_DIR
    TERRAMIND_S1_DIR: str = TERRAMIND_S1_DIR
    TERRAMIND_S2_DIR: str = TERRAMIND_S2_DIR
    THOR_S1_DIR: str = THOR_S1_DIR
    THOR_S2_DIR: str = THOR_S2_DIR
    LABELS_DIR: str = LABELS_DIR
    TESSERA_TEST_DIR: str = TESSERA_TEST_DIR
    TERRAMIND_S1_TEST_DIR: str = TERRAMIND_S1_TEST_DIR
    TERRAMIND_S2_TEST_DIR: str = TERRAMIND_S2_TEST_DIR
    THOR_S1_TEST_DIR: str = THOR_S1_TEST_DIR
    THOR_S2_TEST_DIR: str = THOR_S2_TEST_DIR
    PATCH_SIZE: int = PATCH_SIZE
    BATCH_SIZE: int = BATCH_SIZE
