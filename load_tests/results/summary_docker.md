Load: 100 users, spawn rate 20/s, 60s per run. Latency columns are for `POST /predict`; throughput is all requests.

| Docker containers | Requests | Failures | Throughput (req/s) | Median (ms) | Average (ms) | p95 (ms) | p99 (ms) | Max (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,407 | 0 (0.0%) | 31.4 | 2,100 | 2,236 | 5,800 | 7,000 | 7,029 |
| 2 | 2,382 | 0 (0.0%) | 47.6 | 1,200 | 1,214 | 2,600 | 3,500 | 4,116 |
| 4 | 3,666 | 1 (0.03%) | 74.9 | 170 | 332 | 1,000 | 2,600 | 3,980 |
