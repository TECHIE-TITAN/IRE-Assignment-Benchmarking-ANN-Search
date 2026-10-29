Run A: ../results   Run B: ../results_v2

Every benchmark row present in both runs: how far recall, QPS and build time moved

| experiment | dataset | index | rows A | rows B | matched | max |d recall| | mean |d recall| | median d QPS | median |d QPS| | max |d QPS| |
|---|---|---|---|---|---|---|---|---|---|---|
| part1 | glove | Exact (Flat) | 3 | 3 | 3 | 0.0000 | 0.0000 | +15.8% | 15.8% | 41.9% |
| part1 | sift | Exact (Flat) | 3 | 3 | 3 | 0.0000 | 0.0000 | +13.7% | 13.7% | 31.1% |
| part2 | glove | HNSW | 42 | 42 | 42 | 0.0041 | 0.0012 | +27.8% | 27.8% | 77.2% |
| part2 | glove | IVF-Flat | 32 | 32 | 32 | 0.0000 | 0.0000 | +20.1% | 20.1% | 77.5% |
| part2 | glove | IVF-PQ | 63 | 63 | 63 | 0.0000 | 0.0000 | +2.7% | 3.1% | 24.4% |
| part2 | sift | HNSW | 26 | 26 | 26 | 0.0038 | 0.0008 | +50.5% | 50.5% | 65.3% |
| part2 | sift | IVF-Flat | 23 | 23 | 23 | 0.0000 | 0.0000 | +33.4% | 33.4% | 60.2% |
| part2 | sift | IVF-PQ | 57 | 57 | 57 | 0.0000 | 0.0000 | +17.8% | 17.8% | 48.0% |
| part3_scaling | sift | Exact (Flat) | 6 | 6 | 6 | 0.0000 | 0.0000 | +1.5% | 1.5% | 5.8% |
| part3_scaling | sift | HNSW | 43 | 43 | 43 | 0.0033 | 0.0005 | +6.3% | 6.3% | 9.5% |
| part3_scaling | sift | IVF-Flat | 42 | 42 | 42 | 0.0000 | 0.0000 | +4.4% | 4.9% | 24.5% |
| part4_drift | synthetic_gmm | HNSW | 8 | 8 | 8 | 0.0026 | 0.0009 | +30.4% | 30.4% | 34.7% |
| part4_drift | synthetic_gmm | IVF-Flat | 8 | 8 | 8 | 0.0000 | 0.0000 | +15.0% | 15.0% | 25.4% |
| part4_drift | synthetic_gmm | IVF-PQ | 8 | 8 | 8 | 0.0000 | 0.0000 | +7.9% | 7.9% | 9.2% |

Headline numbers (QPS = one thread, one query per call)

| Dataset | Index | recall target | QPS run A | QPS run B | change |
|---|---|---|---|---|---|
| SIFT-1M | Exact (1 thread, one query) | - | 23.8 | 27.0 | +13.7% |
| SIFT-1M | HNSW | 0.90 | 4816 | 6573 | +36.5% |
| SIFT-1M | HNSW | 0.95 | 4020 | 5173 | +28.7% |
| SIFT-1M | IVF-Flat | 0.90 | 1568 | 2366 | +50.8% |
| SIFT-1M | IVF-Flat | 0.95 | 1191 | 1439 | +20.8% |
| SIFT-1M | IVF-PQ | 0.90 | not reached | not reached | - |
| SIFT-1M | IVF-PQ | 0.95 | not reached | not reached | - |
| GloVe-100 | Exact (1 thread, one query) | - | 24.6 | 28.1 | +14.5% |
| GloVe-100 | HNSW | 0.90 | 982 | 1161 | +18.2% |
| GloVe-100 | HNSW | 0.95 | 551 | 635 | +15.3% |
| GloVe-100 | IVF-Flat | 0.90 | 383 | 461 | +20.3% |
| GloVe-100 | IVF-Flat | 0.95 | 196 | 236 | +20.7% |
| GloVe-100 | IVF-PQ | 0.90 | not reached | not reached | - |
| GloVe-100 | IVF-PQ | 0.95 | not reached | not reached | - |

Part 3: cost at recall >= 0.95 (ms per query)

| Index | n | ms/query A | ms/query B | change |
|---|---|---|---|---|
| Exact (Flat) | 100K | 3.869 | 3.657 | -5.5% |
| Exact (Flat) | 300K | 11.577 | 11.162 | -3.6% |
| Exact (Flat) | 1M | 42.105 | 37.023 | -12.1% |
| HNSW | 100K | 0.079 | 0.073 | -6.8% |
| HNSW | 300K | 0.137 | 0.129 | -6.2% |
| HNSW | 1M | 0.249 | 0.193 | -22.3% |
| IVF-Flat | 100K | 0.154 | 0.156 | +1.0% |
| IVF-Flat | 300K | 0.400 | 0.377 | -5.7% |
| IVF-Flat | 1M | 0.840 | 0.695 | -17.2% |

Growth 1M vs 100K

| Index | run A | run B |
|---|---|---|
| Exact (Flat) | 10.9x | 10.1x |
| HNSW | 3.2x | 2.6x |
| IVF-Flat | 5.4x | 4.5x |
