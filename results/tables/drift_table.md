Recall@10 against exact search over all vectors (final queries)

| Index | Variant | shift 0 | shift 1 | shift 2 | shift 4 |
|---|---|---|---|---|---|
| HNSW | stale | 0.959 | 0.985 | 0.972 | 0.978 |
| HNSW | refreshed | 0.960 | 0.989 | 0.991 | 0.991 |
| IVF-Flat | stale | 0.999 | 1.000 | 1.000 | 0.998 |
| IVF-Flat | refreshed | 1.000 | 1.000 | 1.000 | 0.998 |
| IVF-PQ | stale | 0.575 | 0.438 | 0.252 | 0.122 |
| IVF-PQ | refreshed | 0.576 | 0.607 | 0.600 | 0.565 |

QPS, one thread, one query per call (in brackets: relative to the same index on the control, stale)

| Index | Variant | shift 0 | shift 1 | shift 2 | shift 4 |
|---|---|---|---|---|---|
| HNSW | stale | 9194 (1.00x) | 15834 (1.72x) | 15083 (1.64x) | 17331 (1.88x) |
| HNSW | refreshed | 9088 (0.99x) | 15847 (1.72x) | 14768 (1.61x) | 17081 (1.86x) |
| IVF-Flat | stale | 9921 (1.00x) | 9778 (0.99x) | 9387 (0.95x) | 10713 (1.08x) |
| IVF-Flat | refreshed | 10512 (1.06x) | 11048 (1.11x) | 11226 (1.13x) | 9830 (0.99x) |
| IVF-PQ | stale | 6364 (1.00x) | 6303 (0.99x) | 6292 (0.99x) | 6296 (0.99x) |
| IVF-PQ | refreshed | 6928 (1.09x) | 7176 (1.13x) | 7457 (1.17x) | 5351 (0.84x) |

IVF lists: largest list / mean list / mean vectors scanned per query

| Index | Variant | shift 0 | shift 1 | shift 2 | shift 4 |
|---|---|---|---|---|---|
| IVF-Flat | stale | 2399 / 586 / 5579 | 2399 / 586 / 5642 | 2399 / 586 / 5588 | 2399 / 586 / 5351 |
| IVF-Flat | refreshed | 2340 / 586 / 5061 | 2340 / 586 / 4755 | 2340 / 586 / 4575 | 2316 / 586 / 7973 |
| IVF-PQ | stale | 2399 / 586 / 11037 | 2399 / 586 / 11006 | 2399 / 586 / 10893 | 2399 / 586 / 10972 |
| IVF-PQ | refreshed | 2340 / 586 / 9904 | 2340 / 586 / 9357 | 2340 / 586 / 8815 | 2316 / 586 / 13664 |

Search settings: HNSW efSearch=48, IVF-Flat nprobe=8, IVF-PQ nprobe=16
