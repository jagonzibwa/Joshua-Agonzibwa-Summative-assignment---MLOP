"""Central configuration for the student-engagement ML pipeline.

Every path and tunable lives here so the notebook, the training scripts and the
API all agree. Values that differ between a laptop and a cloud container can be
overridden with environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Paths -----------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
# Images held back from training and testing. They play the role of new labelled
# data arriving after deployment and are what the retraining demo uploads.
POOL_DIR = DATA_DIR / "retrain_pool"
RAW_DIR = DATA_DIR / "raw"            # downloaded archive, git-ignored
INSIGHTS_PATH = DATA_DIR / "insights.json"
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "pipeline.db"))

MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
MODEL_PATH = MODEL_DIR / "engagement_model.h5"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
ARCHIVE_DIR = MODEL_DIR / "archive"

# --- Dataset ---------------------------------------------------------------
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/joyee19/studentengagement"
DATASET_PAGE = "https://www.kaggle.com/datasets/joyee19/studentengagement"

# Alphabetical order, which is also the order of the model's output units.
CLASS_NAMES = ["bored", "confused", "drowsy", "engaged", "frustrated", "looking_away"]

# The dataset's own two-level labelling: a confused or frustrated student is
# still attending to the lesson, so those states count as engaged.
ENGAGEMENT_GROUP = {
    "bored": "Not engaged",
    "confused": "Engaged",
    "drowsy": "Not engaged",
    "engaged": "Engaged",
    "frustrated": "Engaged",
    "looking_away": "Not engaged",
}

# Folder names used inside the Kaggle archive -> our class names.
RAW_CLASS_FOLDERS = {
    "Engaged/confused": "confused",
    "Engaged/engaged": "engaged",
    "Engaged/frustrated": "frustrated",
    "Not engaged/Looking Away": "looking_away",
    "Not engaged/bored": "bored",
    "Not engaged/drowsy": "drowsy",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- Preprocessing ---------------------------------------------------------
IMG_SIZE = (160, 160)          # (height, width) fed to the network
STORE_SIZE = IMG_SIZE          # images are stored on disk at the network input size
TEST_FRACTION = 0.15
POOL_FRACTION = 0.06           # held back as "future" data for retraining
VAL_FRACTION = 0.15            # carved out of data/train, by frame block
SPLIT_BLOCK = 10               # consecutive frames kept together in a split
SEED = 42

# --- Training --------------------------------------------------------------
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
HEAD_EPOCHS = int(os.getenv("HEAD_EPOCHS", 12))
FINE_TUNE_EPOCHS = int(os.getenv("FINE_TUNE_EPOCHS", 10))
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-4
FINE_TUNE_LAYERS = 30          # top MobileNetV2 layers unfrozen in phase 2
DROPOUT = 0.3
L2_WEIGHT = 1e-4

# --- Retraining ------------------------------------------------------------
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", 3))
RETRAIN_LR = float(os.getenv("RETRAIN_LR", 1e-5))   # low: keeps class boundaries stable
# Old training images replayed per class next to the new uploads. 0 = use all.
REPLAY_PER_CLASS = int(os.getenv("REPLAY_PER_CLASS", 80))
# Automatic trigger: retrain once this many new images are waiting.
AUTO_RETRAIN_MIN_NEW = int(os.getenv("AUTO_RETRAIN_MIN_NEW", 200))
AUTO_RETRAIN_ENABLED = os.getenv("AUTO_RETRAIN_ENABLED", "true").lower() == "true"
# Drift trigger: mean confidence of the most recent predictions.
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", 50))
DRIFT_CONFIDENCE_FLOOR = float(os.getenv("DRIFT_CONFIDENCE_FLOOR", 0.60))
# A retrained model replaces the live one only if test accuracy does not drop
# by more than this.
PROMOTION_TOLERANCE = float(os.getenv("PROMOTION_TOLERANCE", 0.02))

# --- Serving ---------------------------------------------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Threads TensorFlow may use per process. Set to 1 in containers so that each
# replica behaves like a one-CPU unit and scaling out is what adds capacity.
TF_THREADS = int(os.getenv("TF_THREADS", 0))   # 0 = let TensorFlow decide


def ensure_dirs() -> None:
    for d in (DATA_DIR, TRAIN_DIR, TEST_DIR, MODEL_DIR, ARCHIVE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for name in CLASS_NAMES:
        (TRAIN_DIR / name).mkdir(parents=True, exist_ok=True)
        (TEST_DIR / name).mkdir(parents=True, exist_ok=True)
