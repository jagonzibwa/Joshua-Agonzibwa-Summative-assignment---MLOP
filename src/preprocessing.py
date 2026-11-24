"""Data acquisition and preprocessing for the student-engagement images.

Three jobs live here:

1. Acquisition: download the Kaggle archive and build ``data/train``,
   ``data/test`` and ``data/retrain_pool`` (``python -m src.preprocessing --acquire``).
2. Preprocessing shared by training, retraining and serving: decoding,
   validation, resizing, the train/validation split and ``tf.data`` pipelines.
3. Hand-crafted image features (brightness, contrast, sharpness, colour) used
   for the dataset visualisations.

TensorFlow is imported lazily so the light-weight helpers can be used without it.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import random
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a usable image."""


# ---------------------------------------------------------------------------
# Decoding and resizing
# ---------------------------------------------------------------------------
def decode_image(data: bytes) -> Image.Image:
    """Decode raw bytes into an upright RGB PIL image, or raise InvalidImageError."""
    if not data:
        raise InvalidImageError("The file is empty.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise InvalidImageError("The file is larger than 10 MB.")
    try:
        Image.open(io.BytesIO(data)).verify()          # cheap integrity check
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)          # honour phone rotation
        image = image.convert("RGB")                    # drop alpha / greyscale
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError("The file is not a readable image.") from exc
    if min(image.size) < 32:
        raise InvalidImageError("The image is smaller than 32 pixels.")
    return image


def square_crop(image: Image.Image) -> Image.Image:
    """Crop to a square without distorting the face.

    Webcam frames are 16:9 (landscape) or 9:16 (phone, portrait). Squashing them
    to a square would stretch faces, so the square is cut out instead: the
    centre of a landscape frame, and the upper part of a portrait frame, where
    the head is.
    """
    width, height = image.size
    if width == height:
        return image
    if width > height:
        left = (width - height) // 2
        return image.crop((left, 0, left + height, height))
    top = int((height - width) * 0.15)
    return image.crop((0, top, width, top + width))


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Square-crop, then resize to ``size`` (height, width) with a high-quality filter."""
    return square_crop(image).resize((size[1], size[0]), Image.LANCZOS)


def image_to_array(image: Image.Image) -> np.ndarray:
    """PIL image -> float32 array in [0, 255] at the network's input size.

    Scaling to [-1, 1] happens inside the model (a ``Rescaling`` layer), so the
    exact same preprocessing is applied at training and at serving time.
    """
    return np.asarray(resize_image(image, config.IMG_SIZE), dtype=np.float32)


def preprocess_bytes(data: bytes) -> np.ndarray:
    """Raw upload bytes -> batch of one, ready for ``model.predict``."""
    return image_to_array(decode_image(data))[np.newaxis, ...]


def preprocess_path(path: str | Path) -> np.ndarray:
    return preprocess_bytes(Path(path).read_bytes())


def save_training_image(data: bytes, label: str, name: str,
                        root: Path | None = None) -> Path:
    """Preprocess an uploaded image and store it under ``data/train/<label>``.

    This is the preprocessing step of the retraining pipeline: validate, fix
    orientation, convert to RGB, square-crop, resize and re-encode as JPEG.
    """
    if label not in config.CLASS_NAMES:
        raise ValueError(f"Unknown class label: {label}")
    image = resize_image(decode_image(data), config.STORE_SIZE)
    dest_dir = (root or config.TRAIN_DIR) / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{Path(name).stem}.jpg"
    image.save(dest, "JPEG", quality=90)
    return dest


# ---------------------------------------------------------------------------
# File listing and splits
# ---------------------------------------------------------------------------
def list_images(root: Path) -> tuple[list[str], list[int]]:
    """Return (paths, integer labels) for a ``root/<class>/<image>`` folder."""
    paths: list[str] = []
    labels: list[int] = []
    for index, name in enumerate(config.CLASS_NAMES):
        folder = Path(root) / name
        if not folder.is_dir():
            continue
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() in config.IMAGE_EXTENSIONS:
                paths.append(str(file))
                labels.append(index)
    return paths, labels


def class_counts(root: Path) -> dict[str, int]:
    paths, labels = list_images(root)
    counts = {name: 0 for name in config.CLASS_NAMES}
    for label in labels:
        counts[config.CLASS_NAMES[label]] += 1
    return counts


def _block_key(path: str | Path) -> str:
    """Key shared by neighbouring video frames of one class.

    The images are frames cut from webcam recordings, so frame 0412 and frame
    0413 are near-duplicates. Frames are grouped in blocks of ``SPLIT_BLOCK``
    consecutive numbers and a whole block always lands on the same side of a
    split. Uploaded images (no frame number) form their own one-image block.
    """
    p = Path(path)
    match = re.search(r"(\d+)$", p.stem)
    if match and not p.stem.startswith("upl"):
        return f"{p.parent.name}:{int(match.group(1)) // config.SPLIT_BLOCK}"
    return f"{p.parent.name}:{p.stem}"


def is_validation_file(path: str | Path) -> bool:
    """Deterministic train/validation assignment that is stable as data grows."""
    digest = hashlib.md5(_block_key(path).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < config.VAL_FRACTION


def train_val_split(paths: list[str], labels: list[int]):
    """Split a file list into (train_paths, train_labels, val_paths, val_labels)."""
    tr_p, tr_l, va_p, va_l = [], [], [], []
    for p, l in zip(paths, labels):
        if is_validation_file(p):
            va_p.append(p); va_l.append(l)
        else:
            tr_p.append(p); tr_l.append(l)
    return tr_p, tr_l, va_p, va_l


# ---------------------------------------------------------------------------
# tf.data input pipelines
# ---------------------------------------------------------------------------
def make_dataset(paths: list[str], labels: list[int], batch_size: int | None = None,
                 training: bool = False, seed: int = config.SEED):
    """Build a batched ``tf.data.Dataset`` of (image, label) pairs.

    Files under ``data/`` are already square-cropped and stored at ``IMG_SIZE``
    (see ``resize_image``), so the resize below is a no-op safety net. Pixels
    stay in [0, 255]; augmentation and scaling are layers of the model itself.
    """
    import tensorflow as tf

    batch_size = batch_size or config.BATCH_SIZE

    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image = tf.image.resize(image, config.IMG_SIZE, method="bilinear", antialias=True)
        image.set_shape((*config.IMG_SIZE, 3))
        return tf.cast(image, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compute_class_weights(labels: Iterable[int]) -> dict[int, float]:
    """Inverse-frequency class weights so the smaller classes are not ignored."""
    labels = np.asarray(list(labels))
    n_classes = len(config.CLASS_NAMES)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (n_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


# ---------------------------------------------------------------------------
# Hand-crafted features for the visualisations
# ---------------------------------------------------------------------------
def extract_features(image: Image.Image) -> dict[str, float]:
    """Interpretable per-image features.

    * brightness  - mean luminance (0 dark, 255 bright): how well lit the room is
    * contrast    - standard deviation of luminance
    * sharpness   - mean gradient magnitude: motion blur and focus
    * warmth      - red mean minus blue mean: colour temperature of the lighting
    * lower_minus_upper - brightness of the lower half minus the upper half, a
      rough cue for head position in the frame
    """
    small = np.asarray(image.convert("RGB").resize((96, 96), Image.BILINEAR), dtype=np.float32)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    grey = 0.299 * r + 0.587 * g + 0.114 * b
    gy, gx = np.gradient(grey)
    half = grey.shape[0] // 2
    return {
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
        "sharpness": float(np.hypot(gx, gy).mean()),
        "warmth": float(r.mean() - b.mean()),
        "lower_minus_upper": float(grey[half:].mean() - grey[:half].mean()),
    }


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def download_dataset(force: bool = False) -> Path:
    """Download the Kaggle archive into ``data/raw`` (about 40 MB)."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = config.RAW_DIR / "studentengagement.zip"
    if archive.exists() and not force:
        return archive
    print(f"Downloading {config.DATASET_URL}")
    request = urllib.request.Request(config.DATASET_URL, headers={"User-Agent": "python-urllib"})
    tmp = archive.with_suffix(".part")
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(archive)
    return archive


def build_splits(archive: Path, clean: bool = True) -> dict[str, dict[str, int]]:
    """Extract the archive into ``data/train``, ``data/test`` and ``data/retrain_pool``.

    Each class is cut into blocks of ``SPLIT_BLOCK`` consecutive frames and whole
    blocks are assigned to a split. Splitting single frames at random would put
    near-identical neighbouring frames on both sides and overstate test accuracy.

    * test (``TEST_FRACTION``)         - held out for evaluation, never trained on
    * retrain_pool (``POOL_FRACTION``) - held out as "future" data: it is not used
      by the initial training run and is uploaded later to exercise retraining
    * train                            - everything else (validation is carved
      out of it by ``train_val_split``)
    """
    roots = {"train": config.TRAIN_DIR, "test": config.TEST_DIR, "pool": config.POOL_DIR}
    if clean:
        for root in roots.values():
            for name in config.CLASS_NAMES:
                shutil.rmtree(root / name, ignore_errors=True)
    config.ensure_dirs()

    rng = random.Random(config.SEED)
    summary = {"train": {}, "test": {}, "pool": {}}
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if Path(n).suffix.lower() in config.IMAGE_EXTENSIONS]
        for raw_folder, label in config.RAW_CLASS_FOLDERS.items():
            members = sorted(n for n in names if f"/{raw_folder}/" in n)
            blocks = [members[i:i + config.SPLIT_BLOCK]
                      for i in range(0, len(members), config.SPLIT_BLOCK)]
            rng.shuffle(blocks)
            test_target = round(len(members) * config.TEST_FRACTION)
            pool_target = round(len(members) * config.POOL_FRACTION)
            assignment: dict[str, str] = {}
            n_test = n_pool = 0
            for block in blocks:
                if n_test < test_target:
                    split = "test"
                    n_test += len(block)
                elif n_pool < pool_target:
                    split = "pool"
                    n_pool += len(block)
                else:
                    split = "train"
                assignment.update({member: split for member in block})

            counts = {"train": 0, "test": 0, "pool": 0}
            for member in members:
                split = assignment[member]
                try:
                    image = decode_image(zf.read(member))
                except InvalidImageError:
                    continue                      # skip corrupt frames
                image = resize_image(image, config.STORE_SIZE)
                dest = roots[split] / label
                dest.mkdir(parents=True, exist_ok=True)
                image.save(dest / f"{label}_{Path(member).stem}.jpg", "JPEG", quality=90)
                counts[split] += 1
            for split in summary:
                summary[split][label] = counts[split]
            print(f"{label:13s} train={counts['train']:4d} test={counts['test']:4d} "
                  f"retrain_pool={counts['pool']:3d}")
    return summary


def acquire(force_download: bool = False) -> dict[str, dict[str, int]]:
    return build_splits(download_dataset(force=force_download))


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and preprocess the dataset")
    parser.add_argument("--acquire", action="store_true", help="download and build data/train, data/test")
    parser.add_argument("--if-missing", action="store_true", help="only acquire when data/train is empty")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    if args.acquire:
        if args.if_missing and sum(class_counts(config.TRAIN_DIR).values()) > 0:
            print("data/train already populated; nothing to do.")
            return
        acquire(args.force_download)
    else:
        print("train:       ", class_counts(config.TRAIN_DIR))
        print("test:        ", class_counts(config.TEST_DIR))
        print("retrain_pool:", class_counts(config.POOL_DIR))


if __name__ == "__main__":
    main()
