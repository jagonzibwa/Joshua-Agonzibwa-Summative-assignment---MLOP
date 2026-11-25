"""Flood the API with Locust at several replica counts and tabulate the results.

Docker mode (the deployment described in docker-compose.yml)::

    python scripts/run_load_test.py --mode docker --replicas 1 2 4

    For each count it runs ``docker compose up -d --scale api=N``, waits until
    the load balancer is healthy, runs Locust headless against
    http://localhost:8080 and records latency and throughput.

Local mode (no Docker needed)::

    python scripts/run_load_test.py --mode local --replicas 1 2 4

    Starts N uvicorn processes instead of N containers. Each is pinned to one
    CPU core and one TensorFlow thread, which mirrors the one-CPU limit the
    compose file puts on a container. Locust spreads users across them.

Outputs go to ``load_tests/results/``: Locust CSV and HTML reports per run, plus
``summary_<mode>.json`` and ``summary_<mode>.md``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "load_tests" / "results"
LOCUSTFILE = ROOT / "load_tests" / "locustfile.py"


def wait_healthy(url: str, timeout: float = 240) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as r:
                if json.load(r).get("model_loaded"):
                    return
        except Exception:  # noqa: BLE001 - still starting
            pass
        time.sleep(2)
    raise RuntimeError(f"{url} did not become healthy within {timeout:.0f}s")


def run_locust(tag: str, users: int, spawn_rate: int, run_time: str,
               host: str | None, target_hosts: list[str] | None) -> dict:
    prefix = RESULTS / tag
    env = dict(os.environ)
    if target_hosts:
        env["TARGET_HOSTS"] = ",".join(target_hosts)
    cmd = [sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless",
           "-u", str(users), "-r", str(spawn_rate), "-t", run_time,
           "--host", host or target_hosts[0], "--csv", str(prefix), "--html", f"{prefix}.html",
           "--only-summary", "--stop-timeout", "10"]
    print("  $", " ".join(cmd[2:]))
    subprocess.run(cmd, env=env, cwd=ROOT, check=False)
    return parse_stats(Path(f"{prefix}_stats.csv"))


def parse_stats(path: Path) -> dict:
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["Name"]
            if name not in ("/predict", "Aggregated"):
                continue
            requests = int(row["Request Count"])
            failures = int(row["Failure Count"])
            out["predict" if name == "/predict" else "all"] = {
                "requests": requests,
                "failures": failures,
                "failure_pct": round(100 * failures / requests, 2) if requests else 0.0,
                "rps": round(float(row["Requests/s"]), 1),
                "avg_ms": round(float(row["Average Response Time"])),
                "median_ms": round(float(row["Median Response Time"])),
                "p90_ms": round(float(row["90%"])),
                "p95_ms": round(float(row["95%"])),
                "p99_ms": round(float(row["99%"])),
                "max_ms": round(float(row["Max Response Time"])),
            }
    return out


# ---------------------------------------------------------------------------
# Docker mode
# ---------------------------------------------------------------------------
def compose(*args: str) -> None:
    subprocess.run(["docker", "compose", *args], cwd=ROOT, check=True)


def run_docker(replicas: list[int], args) -> list[dict]:
    rows = []
    compose("build")
    try:
        for n in replicas:
            print(f"\n=== {n} container(s) ===")
            compose("up", "-d", "--scale", f"api={n}", "--force-recreate")
            wait_healthy("http://localhost:8080")
            time.sleep(5 + 3 * n)                  # let every replica finish loading
            stats = run_locust(f"docker_{n}", args.users, args.spawn_rate, args.run_time,
                               "http://localhost:8080", None)
            rows.append({"replicas": n, **stats})
    finally:
        compose("down")
    return rows


# ---------------------------------------------------------------------------
# Local mode
# ---------------------------------------------------------------------------
def run_local(replicas: list[int], args) -> list[dict]:
    import psutil

    rows = []
    cores = psutil.cpu_count(logical=True)
    tmp = RESULTS / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for n in replicas:
        print(f"\n=== {n} replica process(es) ===")
        procs, hosts = [], []
        env = dict(os.environ, TF_THREADS="1", AUTO_RETRAIN_ENABLED="false",
                   DB_PATH=str(tmp / f"loadtest_{n}.db"), TF_CPP_MIN_LOG_LEVEL="2")
        try:
            for i in range(n):
                port = args.base_port + i
                log = open(tmp / f"replica_{n}_{i}.log", "w")
                p = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1",
                     "--port", str(port), "--log-level", "warning"],
                    cwd=ROOT, env=dict(env, INSTANCE_NAME=f"replica-{i + 1}"), stdout=log, stderr=log)
                # One core per replica, counted from the top so core 0 stays free for Locust.
                psutil.Process(p.pid).cpu_affinity([cores - 1 - i])
                procs.append(p)
                hosts.append(f"http://127.0.0.1:{port}")
            for h in hosts:
                wait_healthy(h)
            stats = run_locust(f"local_{n}", args.users, args.spawn_rate, args.run_time, None, hosts)
            rows.append({"replicas": n, **stats})
        finally:
            for p in procs:
                p.terminate()
            for p in procs:
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
            time.sleep(2)
    return rows


# ---------------------------------------------------------------------------
def to_markdown(rows: list[dict], unit: str) -> str:
    lines = [f"| {unit} | Requests | Failures | Throughput (req/s) | Median (ms) | Average (ms) | p95 (ms) | p99 (ms) | Max (ms) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        s = r["predict"]
        lines.append(f"| {r['replicas']} | {s['requests']:,} | {s['failures']} ({s['failure_pct']}%) | {r['all']['rps']} | "
                     f"{s['median_ms']:,} | {s['avg_ms']:,} | {s['p95_ms']:,} | {s['p99_ms']:,} | {s['max_ms']:,} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["docker", "local"], default="docker")
    parser.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--users", type=int, default=100, help="concurrent simulated users")
    parser.add_argument("--spawn-rate", type=int, default=20, help="users started per second")
    parser.add_argument("--run-time", default="60s")
    parser.add_argument("--base-port", type=int, default=8101, help="first port in local mode")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = run_docker(args.replicas, args) if args.mode == "docker" else run_local(args.replicas, args)

    unit = "Docker containers" if args.mode == "docker" else "API replicas (processes)"
    summary = {"mode": args.mode, "users": args.users, "spawn_rate": args.spawn_rate,
               "run_time": args.run_time, "machine": f"{platform.system()} {platform.machine()}, "
               f"{os.cpu_count()} logical cores", "date": time.strftime("%Y-%m-%d %H:%M"), "runs": rows}
    (RESULTS / f"summary_{args.mode}.json").write_text(json.dumps(summary, indent=2))
    table = to_markdown(rows, unit)
    (RESULTS / f"summary_{args.mode}.md").write_text(
        f"Load: {args.users} users, spawn rate {args.spawn_rate}/s, {args.run_time} per run. "
        f"Latency columns are for `POST /predict`; throughput is all requests.\n\n{table}\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
