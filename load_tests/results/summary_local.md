Load: 100 users, spawn rate 20/s, 60s per run. Latency columns are for `POST /predict`; throughput is all requests.

| API replicas (processes) | Requests | Failures | Throughput (req/s) | Median (ms) | Average (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,113 | 0 (0.0%) | 21.4 | 3,900 | 3,777 | 4,600 | 4,800 | 5,552 |
| 2 | 2,007 | 0 (0.0%) | 39.9 | 1,700 | 1,659 | 2,600 | 2,900 | 3,237 |
| 4 | 2,998 | 0 (0.0%) | 61.1 | 650 | 683 | 1,300 | 1,500 | 2,116 |
