"""FastAPI service: prediction, bulk upload, retraining, monitoring and the web UI.

Run locally with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Interactive API docs are at ``/docs``; the dashboard is at ``/``.
"""
from __future__ import annotations

import io
import os
import socket
import statistics
import threading
import time
import zipfile
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import config, database, insights, preprocessing
from src.prediction import ModelNotAvailable, get_predictor
from src.retrain import RetrainManager, check_triggers

STATIC_DIR = Path(__file__).parent / "static"
START_TIME = time.time()
STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")
INSTANCE = os.getenv("INSTANCE_NAME") or socket.gethostname()
PROCESS = psutil.Process()

predictor = get_predictor()
manager = RetrainManager(on_promoted=lambda: predictor.load(force=True))

# In-memory request metrics for this instance.
_metrics_lock = threading.Lock()
_latencies: dict[str, deque] = {}
_counts: Counter = Counter()
_errors: Counter = Counter()
_last_evaluation: dict[str, Any] | None = None
TRACKED = ("/predict", "/health", "/status", "/upload", "/retrain", "/evaluate")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    database.init_db()
    try:
        predictor.load()
        print(f"[api] model v{predictor.version} loaded on {INSTANCE}", flush=True)
    except ModelNotAvailable as exc:
        print(f"[api] WARNING: {exc}", flush=True)
    yield


app = FastAPI(
    title="Student Engagement Classifier",
    description="Classifies a webcam frame of a student into one of six engagement states, "
                "accepts new labelled images and retrains the model on demand.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    started = time.perf_counter()
    path = request.url.path
    try:
        response = await call_next(request)
    except Exception:
        with _metrics_lock:
            _errors[path] += 1
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    if path in TRACKED:
        with _metrics_lock:
            _counts[path] += 1
            if response.status_code >= 500:
                _errors[path] += 1
            _latencies.setdefault(path, deque(maxlen=2000)).append(elapsed_ms)
    response.headers["X-Served-By"] = INSTANCE
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    return response


def _latency_summary(path: str) -> dict[str, Any]:
    with _metrics_lock:
        values = sorted(_latencies.get(path, []))
        count = _counts[path]
    if not values:
        return {"count": count, "avg_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {"count": count, "avg_ms": round(statistics.fmean(values), 1), "p50_ms": round(pick(0.50), 1),
            "p95_ms": round(pick(0.95), 1), "max_ms": round(values[-1], 1)}


# ---------------------------------------------------------------------------
# Health and status (model up-time)
# ---------------------------------------------------------------------------
@app.get("/health", tags=["monitoring"])
def health() -> dict[str, Any]:
    """Liveness probe used by the load balancer and the dashboard."""
    return {"status": "ok" if predictor.is_loaded else "degraded",
            "model_loaded": predictor.is_loaded,
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "instance": INSTANCE}


@app.get("/status", tags=["monitoring"])
def status() -> dict[str, Any]:
    """Up-time, model version and metrics, request latency and resource use."""
    meta = predictor.metadata or {}
    metrics = meta.get("metrics", {})
    mem = PROCESS.memory_info().rss / 1e6
    with _metrics_lock:
        total = sum(_counts.values())
        errors = sum(_errors.values())
    return {
        "status": "ok" if predictor.is_loaded else "degraded",
        "instance": INSTANCE,
        "started_at": STARTED_AT,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model": {
            "loaded": predictor.is_loaded,
            "version": meta.get("version"),
            "kind": meta.get("kind"),
            "trained_at": meta.get("trained_at"),
            "architecture": meta.get("architecture"),
            "train_images": meta.get("train_images"),
            "loaded_seconds_ago": round(time.time() - predictor.loaded_at, 1) if predictor.loaded_at else None,
            "test_metrics": {k: metrics.get(k) for k in
                             ("accuracy", "loss", "precision_macro", "recall_macro", "f1_macro",
                              "roc_auc_ovr", "engaged_vs_not_accuracy", "n_samples")},
        },
        "requests": {"total": total, "errors": errors,
                     "predict": _latency_summary("/predict"), "health": _latency_summary("/health")},
        "system": {"memory_mb": round(mem, 1), "cpu_percent": psutil.cpu_percent(interval=None),
                   "cpu_count": psutil.cpu_count()},
        "retraining": {k: manager.snapshot()[k] for k in ("status", "stage", "progress", "message")},
        "pending_uploads": database.pending_count(),
        "classes": config.CLASS_NAMES,
        "engagement_group": config.ENGAGEMENT_GROUP,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@app.post("/predict", tags=["prediction"])
def predict(file: UploadFile = File(..., description="One image (.jpg, .png, .webp)")) -> dict[str, Any]:
    """Classify a single image and return the class with all probabilities."""
    data = file.file.read()
    try:
        result = predictor.predict_bytes(data)
    except preprocessing.InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result["prediction_id"] = database.log_prediction(
        file.filename, result["label"], result["confidence"], result["inference_ms"], result["model_version"])
    result["filename"] = file.filename
    result["served_by"] = INSTANCE
    return result


class Feedback(BaseModel):
    prediction_id: int
    true_label: str


@app.post("/feedback", tags=["prediction"])
def feedback(body: Feedback) -> dict[str, Any]:
    """Record the correct label for a past prediction (used for live accuracy)."""
    if body.true_label not in config.CLASS_NAMES:
        raise HTTPException(status_code=400, detail="Unknown class label.")
    if not database.set_feedback(body.prediction_id, body.true_label):
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return {"saved": True}


@app.get("/samples", tags=["prediction"])
def samples(per_class: int = 2) -> list[dict[str, str]]:
    """A few held-out test images, so a prediction can be checked against its label."""
    out = []
    for name in config.CLASS_NAMES:
        files = sorted((config.TEST_DIR / name).glob("*.jpg"))
        step = max(1, len(files) // max(per_class, 1))
        for f in files[::step][:per_class]:
            out.append({"label": name, "name": f.name, "url": f"/samples/{name}/{f.name}"})
    return out


@app.get("/samples/{label}/{name}", tags=["prediction"])
def sample_file(label: str, name: str) -> FileResponse:
    if label not in config.CLASS_NAMES:
        raise HTTPException(status_code=404, detail="Unknown class.")
    path = (config.TEST_DIR / label / Path(name).name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No such sample.")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Bulk upload (saved to the database for retraining)
# ---------------------------------------------------------------------------
def _store(filename: str, label: str, data: bytes, report: dict[str, Any]) -> None:
    try:
        image = preprocessing.decode_image(data)
    except preprocessing.InvalidImageError as exc:
        report["rejected"].append({"file": filename, "reason": str(exc)})
        return
    _, created = database.save_upload(Path(filename).name, label, data, image.width, image.height)
    report["saved" if created else "duplicates"] += 1
    if created:
        report["by_label"][label] = report["by_label"].get(label, 0) + 1


@app.post("/upload", tags=["retraining"])
def upload(label: str = Form(..., description="Class of the uploaded images"),
           files: list[UploadFile] = File(..., description="Images, or a .zip of images")) -> dict[str, Any]:
    """Upload many labelled images. They are validated and stored in the database.

    A ``.zip`` may contain one sub-folder per class (folder name = class label);
    images outside such a folder get the label chosen in the form.
    """
    if label not in config.CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"label must be one of {config.CLASS_NAMES}")
    report: dict[str, Any] = {"saved": 0, "duplicates": 0, "rejected": [], "by_label": {}}
    for upload_file in files:
        data = upload_file.file.read()
        name = upload_file.filename or "upload"
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    members = [m for m in zf.infolist() if not m.is_dir()
                               and Path(m.filename).suffix.lower() in config.IMAGE_EXTENSIONS][:2000]
                    for member in members:
                        if member.file_size > config.MAX_UPLOAD_BYTES:
                            report["rejected"].append({"file": member.filename, "reason": "Larger than 10 MB."})
                            continue
                        folder = Path(member.filename).parent.name.lower().replace(" ", "_")
                        _store(member.filename, folder if folder in config.CLASS_NAMES else label,
                               zf.read(member), report)
            except zipfile.BadZipFile:
                report["rejected"].append({"file": name, "reason": "Not a valid zip archive."})
        else:
            _store(name, label, data, report)

    report["pending_total"] = database.pending_count()
    started, reason = manager.maybe_auto_trigger()
    report["auto_retrain_started"] = started
    report["auto_retrain_note"] = reason if started else None
    return report


@app.get("/uploads", tags=["retraining"])
def uploads() -> dict[str, Any]:
    """What is in the upload database: counts by class and status, latest rows."""
    return database.upload_summary()


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------
class RetrainRequest(BaseModel):
    epochs: int | None = None
    force: bool = False


@app.post("/retrain", status_code=202, tags=["retraining"])
def retrain(body: RetrainRequest | None = None) -> dict[str, Any]:
    """Trigger retraining on the uploaded images. Runs in the background."""
    body = body or RetrainRequest()
    if body.epochs is not None and not 1 <= body.epochs <= 30:
        raise HTTPException(status_code=400, detail="epochs must be between 1 and 30.")
    started, reason = manager.trigger("manual", epochs=body.epochs, force=body.force)
    if not started:
        raise HTTPException(status_code=409, detail=reason)
    return {"started": True, "message": reason, "run_id": manager.state["run_id"]}


@app.get("/retrain/status", tags=["retraining"])
def retrain_status() -> dict[str, Any]:
    """Progress of the current (or last) retraining run and the trigger conditions."""
    return {**manager.snapshot(), "triggers": check_triggers()}


@app.get("/retrain/history", tags=["retraining"])
def retrain_history() -> list[dict[str, Any]]:
    return database.run_history()


# ---------------------------------------------------------------------------
# Evaluation in production and monitoring
# ---------------------------------------------------------------------------
@app.post("/evaluate", tags=["monitoring"])
def evaluate() -> dict[str, Any]:
    """Score the live model on the held-out test set, inside the running service."""
    global _last_evaluation
    from src import model as model_lib

    paths, labels = preprocessing.list_images(config.TEST_DIR)
    if not paths:
        raise HTTPException(status_code=404, detail="data/test is empty on this instance.")
    try:
        predictor.load()
    except ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    started = time.perf_counter()
    metrics = model_lib.evaluate_model(predictor._model, paths, labels)
    _last_evaluation = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.perf_counter() - started, 2),
        "model_version": predictor.version,
        "class_names": config.CLASS_NAMES,
        **metrics,
    }
    return _last_evaluation


@app.get("/evaluate", tags=["monitoring"])
def last_evaluation() -> dict[str, Any]:
    """The most recent production evaluation, or the metrics stored at training time."""
    if _last_evaluation:
        return _last_evaluation
    meta = predictor.metadata or {}
    if not meta.get("metrics"):
        raise HTTPException(status_code=404, detail="No evaluation available yet.")
    return {"evaluated_at": meta.get("trained_at"), "model_version": meta.get("version"),
            "class_names": config.CLASS_NAMES, "from_training": True, **meta["metrics"]}


@app.get("/monitoring", tags=["monitoring"])
def monitoring() -> dict[str, Any]:
    """Live prediction log statistics: confidence, class mix, feedback accuracy."""
    return {**database.prediction_stats(), "triggers": check_triggers()}


# ---------------------------------------------------------------------------
# Data insights
# ---------------------------------------------------------------------------
@app.get("/insights", tags=["insights"])
def get_insights() -> dict[str, Any]:
    """Per-class statistics of interpretable image features, for the charts."""
    return insights.load_insights()


@app.get("/insights/mean/{name}", tags=["insights"])
def mean_image(name: str) -> FileResponse:
    path = config.DATA_DIR / "mean_images" / Path(name).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}"})
