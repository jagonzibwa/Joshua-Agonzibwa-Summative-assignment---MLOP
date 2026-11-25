"""Locust load test for the EngageWatch API.

Simulates many users sending images to ``/predict`` (the heavy path) with an
occasional ``/health`` and ``/status`` call.

Interactive (opens the Locust web UI on http://localhost:8089)::

    locust -f load_tests/locustfile.py --host http://localhost:8080

Headless::

    locust -f load_tests/locustfile.py --host http://localhost:8080 \
           --headless -u 100 -r 20 -t 60s --csv load_tests/results/run

``scripts/run_load_test.py`` automates this for several container counts.

When several API replicas run without a load balancer in front, set
``TARGET_HOSTS=http://127.0.0.1:8001,http://127.0.0.1:8002``; users are then
spread across the replicas round-robin.
"""
from __future__ import annotations

import itertools
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "test"
_files = sorted(TEST_DIR.glob("*/*.jpg"))
if not _files:
    raise SystemExit(f"No test images found under {TEST_DIR}.")
random.Random(42).shuffle(_files)
# (true label, file name, bytes) for a spread of images across the six classes.
IMAGES = [(p.parent.name, p.name, p.read_bytes()) for p in _files[:60]]

_hosts = [h.strip() for h in os.getenv("TARGET_HOSTS", "").split(",") if h.strip()]
_host_cycle = itertools.cycle(_hosts) if _hosts else None


class EngagementUser(HttpUser):
    """One simulated client: uploads a frame roughly once a second."""

    wait_time = between(0.5, 1.5)

    def __init__(self, environment):
        if _host_cycle is not None:
            self.host = next(_host_cycle)
        super().__init__(environment)

    @task(10)
    def predict(self) -> None:
        label, name, data = random.choice(IMAGES)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")},
                              name="/predict", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            body = response.json()
            if "label" not in body:
                response.failure("response has no label")
            else:
                response.success()

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")

    @task(1)
    def status(self) -> None:
        self.client.get("/status", name="/status")
