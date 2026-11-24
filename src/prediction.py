"""Prediction: load the saved model and classify a single image.

Used by the API (``Predictor`` keeps the model in memory and hot-reloads it when
a retraining run writes a new file) and from the command line::

    python -m src.prediction path/to/image.jpg
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import config, preprocessing


class ModelNotAvailable(RuntimeError):
    """Raised when no trained model file exists yet."""


class Predictor:
    """Thread-safe wrapper around the saved Keras model."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path or config.MODEL_PATH)
        self._lock = threading.Lock()
        self._model = None
        self._infer = None
        self._mtime: float | None = None
        self.metadata: dict[str, Any] = {}
        self.loaded_at: float | None = None

    # -- loading -----------------------------------------------------------
    def _file_mtime(self) -> float | None:
        try:
            return self.model_path.stat().st_mtime
        except FileNotFoundError:
            return None

    def load(self, force: bool = False) -> bool:
        """Load the model if it is not loaded, or reload if the file changed."""
        mtime = self._file_mtime()
        if mtime is None:
            raise ModelNotAvailable(
                f"No trained model file found ({self.model_path.name}). Train one with: python -m src.model")
        if not force and self._model is not None and mtime == self._mtime:
            return False
        with self._lock:
            if not force and self._model is not None and mtime == self._mtime:
                return False
            import tensorflow as tf
            from . import model as model_lib

            model = model_lib.load_model(self.model_path)
            spec = tf.TensorSpec((None, *config.IMG_SIZE, 3), tf.float32)
            # A traced function is several times faster than model.predict()
            # for one image at a time.
            infer = tf.function(lambda x: model(x, training=False), input_signature=[spec])
            infer(tf.zeros((1, *config.IMG_SIZE, 3)))      # warm-up / trace

            self._model, self._infer, self._mtime = model, infer, mtime
            self.metadata = model_lib.load_metadata()
            self.loaded_at = time.time()
        return True

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> int | None:
        return self.metadata.get("version")

    @property
    def class_names(self) -> list[str]:
        return self.metadata.get("class_names", config.CLASS_NAMES)

    # -- inference ---------------------------------------------------------
    def predict_array(self, batch: np.ndarray) -> np.ndarray:
        self.load()                       # cheap stat() unless the file changed
        return self._infer(batch).numpy()

    def predict_bytes(self, data: bytes) -> dict[str, Any]:
        """Classify one image given as raw bytes."""
        started = time.perf_counter()
        batch = preprocessing.preprocess_bytes(data)
        proba = self.predict_array(batch)[0]
        latency_ms = (time.perf_counter() - started) * 1000
        return format_prediction(proba, self.class_names, latency_ms, self.version)

    def predict_path(self, path: str | Path) -> dict[str, Any]:
        return self.predict_bytes(Path(path).read_bytes())


def format_prediction(proba: np.ndarray, class_names: list[str], latency_ms: float,
                      version: int | None) -> dict[str, Any]:
    top = int(np.argmax(proba))
    label = class_names[top]
    engaged_probability = float(sum(
        p for name, p in zip(class_names, proba) if config.ENGAGEMENT_GROUP.get(name) == "Engaged"))
    return {
        "label": label,
        "confidence": float(proba[top]),
        "engagement": config.ENGAGEMENT_GROUP.get(label, "Unknown"),
        "engaged_probability": engaged_probability,
        "probabilities": {name: float(p) for name, p in zip(class_names, proba)},
        "inference_ms": round(latency_ms, 2),
        "model_version": version,
    }


_default: Predictor | None = None


def get_predictor() -> Predictor:
    """Process-wide predictor (created lazily)."""
    global _default
    if _default is None:
        _default = Predictor()
    return _default


def predict_image(path: str | Path) -> dict[str, Any]:
    """Convenience function: classify the image at ``path``."""
    return get_predictor().predict_path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one image")
    parser.add_argument("image", help="path to a .jpg / .png image")
    args = parser.parse_args()
    print(json.dumps(predict_image(args.image), indent=2))


if __name__ == "__main__":
    main()
