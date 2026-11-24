"""Model creation, training, evaluation, saving and fine-tuning.

The classifier is MobileNetV2 (ImageNet weights) with a small regularised head.
Training runs in two phases: the head is trained on a frozen backbone, then the
top of the backbone is unfrozen and fine-tuned at a lower learning rate.

Run ``python -m src.model`` to train from ``data/train`` and write
``models/engagement_model.h5`` plus ``models/model_metadata.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, precision_recall_fscore_support, roc_auc_score)
from tensorflow import keras
from tensorflow.keras import layers

from . import config, preprocessing


def configure_tensorflow() -> None:
    """Apply the per-process thread limit (must run before the first TF op)."""
    if config.TF_THREADS > 0:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(config.TF_THREADS)
            tf.config.threading.set_inter_op_parallelism_threads(config.TF_THREADS)
        except RuntimeError:
            pass  # TensorFlow already initialised


configure_tensorflow()


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
def build_augmentation() -> keras.Sequential:
    """Light augmentation suited to webcam frames. Active only during training."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.15),
    ], name="augmentation")


def build_model(num_classes: int = len(config.CLASS_NAMES), dropout: float = config.DROPOUT,
                l2: float = config.L2_WEIGHT, dense_units: int = 128,
                weights: str | None = "imagenet") -> keras.Model:
    """MobileNetV2 backbone + regularised classification head.

    Input is an RGB image batch in [0, 255]; augmentation and the [-1, 1]
    scaling MobileNetV2 expects are layers of the model, so serving code only
    has to resize.
    """
    inputs = keras.Input(shape=(*config.IMG_SIZE, 3), name="image")
    x = build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale_to_minus1_1")(x)

    backbone = keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, 3), include_top=False, weights=weights)
    backbone.trainable = False
    # training=False keeps batch-norm statistics frozen, also while fine-tuning.
    x = backbone(x, training=False)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(l2), name="dense")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="engagement_mobilenetv2")


def get_backbone(model: keras.Model) -> keras.Model:
    for layer in model.layers:
        if isinstance(layer, keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    raise ValueError("The model has no MobileNetV2 backbone.")


def set_fine_tuning(model: keras.Model, n_layers: int = config.FINE_TUNE_LAYERS) -> None:
    """Unfreeze the top ``n_layers`` of the backbone (batch-norm layers stay frozen)."""
    backbone = get_backbone(model)
    backbone.trainable = n_layers > 0
    if n_layers <= 0:
        return
    cutoff = len(backbone.layers) - n_layers
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def default_callbacks(patience: int = 4) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                          min_lr=1e-6, verbose=1),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(train_paths, train_labels, val_paths, val_labels, *,
                head_epochs: int = config.HEAD_EPOCHS,
                fine_tune_epochs: int = config.FINE_TUNE_EPOCHS,
                dropout: float = config.DROPOUT, l2: float = config.L2_WEIGHT, dense_units: int = 128,
                head_lr: float = config.HEAD_LR, fine_tune_lr: float = config.FINE_TUNE_LR,
                batch_size: int = config.BATCH_SIZE, extra_callbacks=None, verbose: int = 2):
    """Two-phase transfer learning. Returns (model, history dict)."""
    keras.utils.set_random_seed(config.SEED)
    train_ds = preprocessing.make_dataset(train_paths, train_labels, batch_size, training=True)
    val_ds = preprocessing.make_dataset(val_paths, val_labels, batch_size)
    class_weight = preprocessing.compute_class_weights(train_labels)
    extra_callbacks = list(extra_callbacks or [])

    model = build_model(dropout=dropout, l2=l2, dense_units=dense_units)
    curves = ("loss", "accuracy", "val_loss", "val_accuracy")
    history: dict[str, Any] = {key: [] for key in curves}

    def run(epochs: int) -> None:
        if epochs <= 0:
            return
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
                      callbacks=default_callbacks() + extra_callbacks, verbose=verbose)
        for key in curves:
            history[key] += [float(v) for v in h.history[key]]

    # Phase 1: train the new head on frozen ImageNet features.
    compile_model(model, head_lr)
    run(head_epochs)
    history["phase1_epochs"] = len(history["loss"])

    # Phase 2: unfreeze the top of the backbone and fine-tune gently.
    if fine_tune_epochs > 0:
        set_fine_tuning(model)
        compile_model(model, fine_tune_lr)
        run(fine_tune_epochs)
    return model, history


def fine_tune_existing(model: keras.Model, train_paths, train_labels, val_paths, val_labels, *,
                       epochs: int = config.RETRAIN_EPOCHS, learning_rate: float = config.RETRAIN_LR,
                       batch_size: int = config.BATCH_SIZE, extra_callbacks=None, verbose: int = 2):
    """Retraining: continue training an already-trained model on new data.

    The saved production model is the starting point (it acts as the
    pre-trained model), so a few low-learning-rate epochs are enough.
    """
    train_ds = preprocessing.make_dataset(train_paths, train_labels, batch_size, training=True)
    val_ds = preprocessing.make_dataset(val_paths, val_labels, batch_size) if val_paths else None
    set_fine_tuning(model)
    compile_model(model, learning_rate)
    callbacks = list(extra_callbacks or [])
    if val_ds is not None:
        callbacks = default_callbacks(patience=2) + callbacks
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  class_weight=preprocessing.compute_class_weights(train_labels),
                  callbacks=callbacks, verbose=verbose)
    return model, {k: [float(v) for v in vals] for k, vals in h.history.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def predict_proba(model: keras.Model, paths: list[str], batch_size: int = 64) -> np.ndarray:
    ds = preprocessing.make_dataset(paths, [0] * len(paths), batch_size)
    return model.predict(ds, verbose=0)


def evaluate_model(model: keras.Model, paths: list[str], labels: list[int],
                   return_predictions: bool = False) -> dict[str, Any]:
    """Accuracy, loss, precision, recall, F1, ROC-AUC and the confusion matrix."""
    y_true = np.asarray(labels)
    proba = predict_proba(model, paths)
    y_pred = proba.argmax(axis=1)
    idx = list(range(len(config.CLASS_NAMES)))

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=idx, zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=idx, zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=idx))
    except ValueError:
        auc = None  # a class is missing from y_true

    # Engaged vs not engaged, obtained by collapsing the six classes.
    engaged = np.array([config.ENGAGEMENT_GROUP[c] == "Engaged" for c in config.CLASS_NAMES])
    binary_accuracy = float((engaged[y_true] == engaged[y_pred]).mean())

    report = classification_report(y_true, y_pred, labels=idx, target_names=config.CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(log_loss(y_true, proba, labels=idx)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "roc_auc_ovr": auc,
        "engaged_vs_not_accuracy": binary_accuracy,
        "per_class": {name: {"precision": report[name]["precision"],
                             "recall": report[name]["recall"],
                             "f1": report[name]["f1-score"],
                             "support": int(report[name]["support"])}
                      for name in config.CLASS_NAMES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }
    if return_predictions:
        metrics["_proba"] = proba
        metrics["_y_pred"] = y_pred
    return metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_metadata() -> dict[str, Any]:
    if config.METADATA_PATH.exists():
        return json.loads(config.METADATA_PATH.read_text())
    return {}


def load_model(path: Path | None = None) -> keras.Model:
    return keras.models.load_model(path or config.MODEL_PATH, compile=False)


def fresh_copy(model: keras.Model) -> keras.Model:
    """Rebuild the architecture from code and copy every weight across."""
    dense = model.get_layer("dense")
    regulariser = dense.kernel_regularizer
    fresh = build_model(
        num_classes=int(model.output_shape[-1]),
        dropout=float(model.get_layer("dropout_1").rate),
        l2=float(regulariser.l2) if regulariser is not None else 0.0,
        dense_units=int(dense.units),
        weights=None,
    )
    fresh.set_weights(model.get_weights())
    return fresh


def save_model(model: keras.Model, metrics: dict[str, Any], *, kind: str = "initial",
               train_images: int = 0, history: dict | None = None,
               extra: dict | None = None) -> dict[str, Any]:
    """Save the model as .h5, archive the previous version and write metadata."""
    config.ensure_dirs()
    previous = load_metadata()
    version = int(previous.get("version", 0)) + 1
    if config.MODEL_PATH.exists() and previous:
        shutil.copy2(config.MODEL_PATH, config.ARCHIVE_DIR / f"engagement_model_v{previous['version']}.h5")

    tmp = config.MODEL_DIR / "engagement_model.tmp.h5"
    try:
        model.save(tmp, include_optimizer=False)
    except TypeError:
        # Keras cannot write a model back to .h5 when the model object itself
        # was loaded from a .h5 file (the retraining case). Rebuilding the
        # architecture from code and copying the weights gives the same model.
        fresh_copy(model).save(tmp, include_optimizer=False)
    tmp.replace(config.MODEL_PATH)

    metadata = {
        "version": version,
        "kind": kind,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": "MobileNetV2 (ImageNet) + GAP + Dropout + Dense(128) + Dropout + Softmax",
        "class_names": config.CLASS_NAMES,
        "img_size": list(config.IMG_SIZE),
        "train_images": train_images,
        "tensorflow": tf.__version__,
        "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
        "history": history or {},
        **(extra or {}),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the engagement classifier")
    parser.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    args = parser.parse_args()

    paths, labels = preprocessing.list_images(config.TRAIN_DIR)
    if not paths:
        raise SystemExit("data/train is empty. Run: python -m src.preprocessing --acquire")
    tr_p, tr_l, va_p, va_l = preprocessing.train_val_split(paths, labels)
    te_p, te_l = preprocessing.list_images(config.TEST_DIR)
    print(f"train={len(tr_p)} val={len(va_p)} test={len(te_p)}")

    model, history = train_model(tr_p, tr_l, va_p, va_l, head_epochs=args.head_epochs,
                                 fine_tune_epochs=args.fine_tune_epochs)
    metrics = evaluate_model(model, te_p, te_l)
    meta = save_model(model, metrics, kind="initial", train_images=len(tr_p), history=history)
    print(f"Saved model v{meta['version']} to {config.MODEL_PATH}")
    for key in ("accuracy", "loss", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"):
        print(f"  {key:16s} {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
