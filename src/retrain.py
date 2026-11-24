"""Retraining pipeline and its triggers.

Flow of one run:

1. read the images users uploaded (``pending`` rows in the database)
2. preprocess them into ``data/train/<label>/`` (validate, RGB, resize, JPEG)
3. evaluate the live model on ``data/test``
4. fine-tune the live model (it is the pre-trained starting point) on the new
   images plus a replay sample of the existing training set
5. evaluate the candidate on the same test set
6. promote the candidate only if it is not worse than the live model

Triggers:

* manual  - ``POST /retrain`` (the button in the UI) or ``python -m src.retrain``
* volume  - automatically, once ``AUTO_RETRAIN_MIN_NEW`` (default 200) new images are waiting
* drift   - automatically, when the mean confidence of recent predictions falls
  below ``DRIFT_CONFIDENCE_FLOOR`` and there is new labelled data to learn from
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import threading
import time
import traceback
from typing import Any, Callable

from . import config, database, preprocessing

MIN_IMAGES_FOR_DRIFT_RETRAIN = 10
# Progress is mirrored to this file so that, when several API containers share
# the data volume, any of them can report on a run started by another.
STATE_PATH = config.DATA_DIR / "retrain_state.json"
STALE_AFTER_SECONDS = 900


class RetrainManager:
    """Runs at most one retraining job at a time and exposes its progress."""

    def __init__(self, on_promoted: Callable[[], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.on_promoted = on_promoted
        self.instance = os.getenv("INSTANCE_NAME") or socket.gethostname()
        self.state: dict[str, Any] = {
            "status": "idle", "stage": "", "progress": 0, "message": "No retraining has run yet.",
            "run_id": None, "trigger": None, "started_at": None, "finished_at": None,
            "log": [], "result": None, "instance": self.instance, "updated": 0.0,
        }

    # -- state helpers -----------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self.state["status"] == "running"

    def _update(self, stage: str | None = None, progress: int | None = None, message: str | None = None) -> None:
        if stage is not None:
            self.state["stage"] = stage
        if progress is not None:
            self.state["progress"] = progress
        if message is not None:
            self.state["message"] = message
            self.state["log"].append(f"{time.strftime('%H:%M:%S')}  {message}")
            print(f"[retrain] {message}", flush=True)
        self._persist()

    def _persist(self) -> None:
        self.state["updated"] = time.time()
        try:
            tmp = STATE_PATH.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.state, default=str))
            tmp.replace(STATE_PATH)
        except OSError:
            pass  # sharing progress is best-effort

    def snapshot(self) -> dict[str, Any]:
        """Current state; falls back to another replica's run if that is more recent."""
        if self.is_running:
            return self.state
        try:
            shared = json.loads(STATE_PATH.read_text())
        except (OSError, ValueError):
            return self.state
        if shared.get("instance") == self.instance or shared.get("updated", 0) <= self.state["updated"]:
            return self.state
        if shared.get("status") == "running" and time.time() - shared.get("updated", 0) > STALE_AFTER_SECONDS:
            return self.state                      # the other replica died mid-run
        return shared

    # -- public API --------------------------------------------------------
    def trigger(self, trigger: str = "manual", epochs: int | None = None,
                force: bool = False, background: bool = True) -> tuple[bool, str]:
        """Start a run. Returns (started, reason)."""
        with self._lock:
            if self.snapshot()["status"] == "running":
                return False, "A retraining run is already in progress."
            pending = database.pending_count()
            if pending == 0 and not force:
                return False, "There are no newly uploaded images to retrain on."
            if not config.MODEL_PATH.exists():
                return False, "There is no trained model to retrain. Run python -m src.model first."
            self.state.update(status="running", stage="Starting", progress=1, trigger=trigger,
                              started_at=database.now(), finished_at=None, log=[], result=None,
                              run_id=database.start_run("retrain", trigger))
            self._update(message=f"Retraining triggered ({trigger}); {pending} new image(s) waiting.")
        epochs = epochs or config.RETRAIN_EPOCHS
        if background:
            self._thread = threading.Thread(target=self._run, args=(epochs,), daemon=True)
            self._thread.start()
        else:
            self._run(epochs)
        return True, "Retraining started."

    def maybe_auto_trigger(self) -> tuple[bool, str]:
        """Evaluate the automatic triggers and start a run if one fires."""
        if not config.AUTO_RETRAIN_ENABLED or self.snapshot()["status"] == "running":
            return False, "Automatic retraining is disabled or a run is active."
        check = check_triggers()
        if check["volume_trigger"]:
            return self.trigger("auto:volume")
        if check["drift_trigger"]:
            return self.trigger("auto:drift")
        return False, "No trigger condition met."

    # -- the pipeline ------------------------------------------------------
    def _run(self, epochs: int) -> None:
        run_id = self.state["run_id"]
        try:
            result = self._pipeline(run_id, epochs)
            self.state.update(status="completed", progress=100, finished_at=database.now(), stage="Done",
                              result={k: v for k, v in result.items() if k != "history"})
            self._update(message=result["summary"])
        except Exception as exc:  # noqa: BLE001 - report every failure to the UI
            traceback.print_exc()
            database.finish_run(run_id, "failed", message=str(exc)[:500])
            self.state.update(status="failed", finished_at=database.now(), stage="Failed")
            self._update(message=f"Retraining failed: {exc}")

    def _pipeline(self, run_id: int, epochs: int) -> dict[str, Any]:
        from tensorflow import keras
        from . import insights, model as model_lib

        # 1 + 2. Uploaded data -> preprocessing -> data/train -------------------
        self._update("Loading uploads", 5, "Reading uploaded images from the database.")
        rows = database.pending_uploads()
        new_paths: list[str] = []
        accepted, rejected = [], []
        self._update("Preprocessing", 10, f"Preprocessing {len(rows)} uploaded image(s).")
        for row in rows:
            try:
                dest = preprocessing.save_training_image(row["content"], row["label"], f"upl_{row['id']:06d}")
                new_paths.append(str(dest))
                accepted.append(row["id"])
            except (preprocessing.InvalidImageError, ValueError):
                rejected.append(row["id"])
        database.mark_uploads(accepted, "processed", run_id)
        database.mark_uploads(rejected, "rejected", run_id)
        self._update(progress=18, message=f"Preprocessed {len(accepted)} image(s) into data/train"
                                          + (f"; rejected {len(rejected)}." if rejected else "."))

        # Assemble the training and validation sets --------------------------
        paths, labels = preprocessing.list_images(config.TRAIN_DIR)
        tr_p, tr_l, va_p, va_l = preprocessing.train_val_split(paths, labels)
        new_set = set(new_paths)
        rng = random.Random(config.SEED + run_id)
        train_pairs = [(p, l) for p, l in zip(tr_p, tr_l) if p in new_set]
        old_by_class: dict[int, list[str]] = {}
        for p, l in zip(tr_p, tr_l):
            if p not in new_set:
                old_by_class.setdefault(l, []).append(p)
        for label, items in old_by_class.items():
            if config.REPLAY_PER_CLASS > 0 and len(items) > config.REPLAY_PER_CLASS:
                items = rng.sample(items, config.REPLAY_PER_CLASS)
            train_pairs += [(p, label) for p in items]
        rng.shuffle(train_pairs)
        if not train_pairs:
            raise RuntimeError("There are no training images in data/train.")
        fit_paths = [p for p, _ in train_pairs]
        fit_labels = [l for _, l in train_pairs]
        self._update(progress=22, message=f"Training set: {len(fit_paths)} images "
                                          f"({len([p for p in fit_paths if p in new_set])} new + replay), "
                                          f"validation: {len(va_p)}.")

        # 3. Evaluate the live model -----------------------------------------
        te_p, te_l = preprocessing.list_images(config.TEST_DIR)
        self._update("Evaluating current model", 25, "Loading the current model as the pre-trained starting point.")
        model = model_lib.load_model()
        before = model_lib.evaluate_model(model, te_p, te_l) if te_p else None
        if before:
            self._update(progress=35, message=f"Current model: test accuracy {before['accuracy']:.3f}, "
                                              f"F1 {before['f1_macro']:.3f}.")

        # 4. Fine-tune -------------------------------------------------------
        manager = self

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                val = f", val_accuracy {logs['val_accuracy']:.3f}" if "val_accuracy" in logs else ""
                manager._update(progress=35 + int(45 * (epoch + 1) / epochs),
                                message=f"Epoch {epoch + 1}/{epochs}: loss {logs.get('loss', 0):.3f}, "
                                        f"accuracy {logs.get('accuracy', 0):.3f}{val}")

        self._update("Fine-tuning", 36, f"Fine-tuning for up to {epochs} epoch(s).")
        model, history = model_lib.fine_tune_existing(
            model, fit_paths, fit_labels, va_p, va_l, epochs=epochs,
            extra_callbacks=[Progress()], verbose=0)

        # 5. Evaluate the candidate ------------------------------------------
        self._update("Evaluating retrained model", 82, "Evaluating the retrained model on the test set.")
        after = model_lib.evaluate_model(model, te_p, te_l) if te_p else None

        # 6. Promote or keep -------------------------------------------------
        promoted = True
        if before and after:
            promoted = after["accuracy"] >= before["accuracy"] - config.PROMOTION_TOLERANCE
        version = model_lib.load_metadata().get("version")
        if promoted:
            self._update("Saving model", 90, "Saving the retrained model.")
            meta = model_lib.save_model(
                model, after or {}, kind="retrain", train_images=len(fit_paths), history=history,
                extra={"run_id": run_id, "new_images": len(accepted)})
            version = meta["version"]
            if self.on_promoted:
                self.on_promoted()
        try:
            insights.compute_insights()         # class counts changed
        except Exception:  # noqa: BLE001 - insights are not critical
            traceback.print_exc()

        if before and after:
            delta = after["accuracy"] - before["accuracy"]
            summary = (f"Model v{version} is live: test accuracy {before['accuracy']:.3f} -> "
                       f"{after['accuracy']:.3f} ({delta:+.3f})." if promoted else
                       f"Retrained model was not promoted: test accuracy {after['accuracy']:.3f} is below "
                       f"the live model's {before['accuracy']:.3f}. The live model stays in service.")
        else:
            summary = f"Model v{version} saved (no test set available for comparison)."

        strip = lambda m: {k: v for k, v in m.items() if k not in ("per_class",)} if m else None  # noqa: E731
        database.finish_run(run_id, "completed", new_images=len(accepted), train_images=len(fit_paths),
                            epochs=len(history.get("loss", [])), metrics_before=strip(before),
                            metrics_after=strip(after), promoted=int(promoted),
                            model_version=version, message=summary)
        return {"summary": summary, "promoted": promoted, "model_version": version,
                "new_images": len(accepted), "rejected_images": len(rejected),
                "train_images": len(fit_paths), "epochs_run": len(history.get("loss", [])),
                "before": strip(before), "after": strip(after), "history": history}


def check_triggers() -> dict[str, Any]:
    """Evaluate the conditions that call for retraining."""
    pending = database.pending_count()
    stats = database.prediction_stats()
    mean_conf = stats["mean_confidence"]
    window_full = stats["window"] >= config.DRIFT_WINDOW
    drift = bool(window_full and mean_conf is not None and mean_conf < config.DRIFT_CONFIDENCE_FLOOR)
    volume = pending >= config.AUTO_RETRAIN_MIN_NEW
    reasons = []
    if volume:
        reasons.append(f"{pending} new images are waiting (threshold {config.AUTO_RETRAIN_MIN_NEW}).")
    if drift:
        reasons.append(f"Mean confidence of the last {stats['window']} predictions is {mean_conf:.2f}, "
                       f"below {config.DRIFT_CONFIDENCE_FLOOR:.2f}.")
    return {
        "auto_enabled": config.AUTO_RETRAIN_ENABLED,
        "pending_images": pending,
        "volume_threshold": config.AUTO_RETRAIN_MIN_NEW,
        "volume_trigger": volume,
        "drift_window": config.DRIFT_WINDOW,
        "drift_window_filled": stats["window"],
        "drift_confidence_floor": config.DRIFT_CONFIDENCE_FLOOR,
        "mean_recent_confidence": mean_conf,
        "drift_detected": drift,
        "drift_trigger": drift and pending >= MIN_IMAGES_FOR_DRIFT_RETRAIN,
        "retraining_recommended": volume or drift,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the model on newly uploaded images")
    parser.add_argument("--epochs", type=int, default=config.RETRAIN_EPOCHS)
    parser.add_argument("--force", action="store_true", help="retrain even with no new uploads")
    args = parser.parse_args()
    database.init_db()
    manager = RetrainManager()
    started, reason = manager.trigger("cli", epochs=args.epochs, force=args.force, background=False)
    print(reason if not started else manager.state["message"])


if __name__ == "__main__":
    main()
