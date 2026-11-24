"""SQLite persistence for the pipeline.

Three tables:

* ``uploaded_images``  - every image a user uploads for retraining, stored as a
  BLOB with its label. Rows start as ``pending`` and become ``processed`` once a
  retraining run has preprocessed them into ``data/train``.
* ``training_runs``    - one row per training / retraining run with its metrics.
* ``predictions``      - a log of served predictions, used for monitoring
  (confidence drift, latency) and for user feedback on correctness.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from . import config

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploaded_images (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT    NOT NULL,
    label        TEXT    NOT NULL,
    sha256       TEXT    NOT NULL UNIQUE,
    width        INTEGER,
    height       INTEGER,
    size_bytes   INTEGER,
    content      BLOB    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending',
    uploaded_at  TEXT    NOT NULL,
    processed_at TEXT,
    run_id       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_uploaded_status ON uploaded_images(status);

CREATE TABLE IF NOT EXISTS training_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL,
    trigger_type  TEXT NOT NULL,
    status        TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    new_images    INTEGER DEFAULT 0,
    train_images  INTEGER DEFAULT 0,
    epochs        INTEGER DEFAULT 0,
    metrics_before TEXT,
    metrics_after  TEXT,
    promoted      INTEGER DEFAULT 0,
    model_version INTEGER,
    message       TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    filename      TEXT,
    predicted     TEXT NOT NULL,
    confidence    REAL NOT NULL,
    latency_ms    REAL,
    model_version INTEGER,
    true_label    TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    # With write-ahead logging, NORMAL skips the disk sync on every commit. The
    # prediction log is written on each request, so this matters under load.
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _lock, connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")      # readers no longer block the writer
        conn.executescript(SCHEMA)
        # A run left 'running' by a crashed or restarted process can never finish.
        conn.execute(
            "UPDATE training_runs SET status='failed', finished_at=?, "
            "message='Interrupted by a restart' WHERE status='running'", (now(),))


# --- uploaded images -------------------------------------------------------
def save_upload(filename: str, label: str, content: bytes, width: int, height: int) -> tuple[int | None, bool]:
    """Store one uploaded image. Returns (row id, created); duplicates are skipped."""
    digest = hashlib.sha256(content).hexdigest()
    with _lock, connect() as conn:
        row = conn.execute("SELECT id FROM uploaded_images WHERE sha256=?", (digest,)).fetchone()
        if row:
            return row["id"], False
        cur = conn.execute(
            "INSERT INTO uploaded_images (filename, label, sha256, width, height, size_bytes, content, uploaded_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (filename, label, digest, width, height, len(content), content, now()))
        return cur.lastrowid, True


def pending_uploads() -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT id, filename, label, content FROM uploaded_images WHERE status='pending' ORDER BY id"
        ).fetchall()


def mark_uploads(ids: list[int], status: str, run_id: int | None = None) -> None:
    if not ids:
        return
    with _lock, connect() as conn:
        conn.executemany(
            "UPDATE uploaded_images SET status=?, processed_at=?, run_id=? WHERE id=?",
            [(status, now(), run_id, i) for i in ids])


def upload_summary() -> dict[str, Any]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT label, status, COUNT(*) AS n FROM uploaded_images GROUP BY label, status").fetchall()
        recent = conn.execute(
            "SELECT id, filename, label, status, width, height, size_bytes, uploaded_at "
            "FROM uploaded_images ORDER BY id DESC LIMIT 12").fetchall()
    by_label = {name: {"pending": 0, "processed": 0, "rejected": 0} for name in config.CLASS_NAMES}
    totals = {"pending": 0, "processed": 0, "rejected": 0}
    for r in rows:
        by_label.setdefault(r["label"], {"pending": 0, "processed": 0, "rejected": 0})
        by_label[r["label"]][r["status"]] = r["n"]
        totals[r["status"]] = totals.get(r["status"], 0) + r["n"]
    return {"totals": totals, "by_label": by_label, "recent": [dict(r) for r in recent]}


def pending_count() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM uploaded_images WHERE status='pending'").fetchone()[0]


# --- training runs ---------------------------------------------------------
def start_run(kind: str, trigger: str) -> int:
    with _lock, connect() as conn:
        cur = conn.execute(
            "INSERT INTO training_runs (kind, trigger_type, status, started_at) VALUES (?,?,?,?)",
            (kind, trigger, "running", now()))
        return cur.lastrowid


def finish_run(run_id: int, status: str, **fields: Any) -> None:
    fields = dict(fields, status=status, finished_at=now())
    for key in ("metrics_before", "metrics_after"):
        if key in fields and not isinstance(fields[key], (str, type(None))):
            fields[key] = json.dumps(fields[key])
    columns = ", ".join(f"{k}=?" for k in fields)
    with _lock, connect() as conn:
        conn.execute(f"UPDATE training_runs SET {columns} WHERE id=?", (*fields.values(), run_id))


def run_history(limit: int = 20) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for key in ("metrics_before", "metrics_after"):
            d[key] = json.loads(d[key]) if d[key] else None
        d["promoted"] = bool(d["promoted"])
        out.append(d)
    return out


# --- predictions -----------------------------------------------------------
def log_prediction(filename: str | None, predicted: str, confidence: float,
                   latency_ms: float, model_version: int | None) -> int:
    with _lock, connect() as conn:
        cur = conn.execute(
            "INSERT INTO predictions (created_at, filename, predicted, confidence, latency_ms, model_version) "
            "VALUES (?,?,?,?,?,?)",
            (now(), filename, predicted, confidence, latency_ms, model_version))
        return cur.lastrowid


def set_feedback(prediction_id: int, true_label: str) -> bool:
    with _lock, connect() as conn:
        cur = conn.execute("UPDATE predictions SET true_label=? WHERE id=?", (true_label, prediction_id))
        return cur.rowcount > 0


def prediction_stats(window: int = config.DRIFT_WINDOW) -> dict[str, Any]:
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        recent = conn.execute(
            "SELECT id, created_at, filename, predicted, confidence, latency_ms, true_label "
            "FROM predictions ORDER BY id DESC LIMIT ?", (window,)).fetchall()
        dist = conn.execute("SELECT predicted, COUNT(*) AS n FROM predictions GROUP BY predicted").fetchall()
        fb = conn.execute(
            "SELECT COUNT(*) AS n, SUM(CASE WHEN true_label = predicted THEN 1 ELSE 0 END) AS correct "
            "FROM predictions WHERE true_label IS NOT NULL").fetchone()
    confidences = [r["confidence"] for r in recent]
    return {
        "total": total,
        "window": len(recent),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "distribution": {r["predicted"]: r["n"] for r in dist},
        "feedback_count": fb["n"] or 0,
        "feedback_accuracy": (fb["correct"] / fb["n"]) if fb["n"] else None,
        "recent": [dict(r) for r in recent[:10]],
        "confidence_series": list(reversed(confidences)),
    }
