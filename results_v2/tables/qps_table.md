| Dataset | Index | QPS @ recall 0.90 | QPS @ recall 0.95 | Config at 0.95 |
|---|---|---|---|---|
| SIFT-1M | Exact (Flat) | 27 (recall 1.0) | 27 (recall 1.0) | IndexFlatL2 |
| SIFT-1M | HNSW | 6573 | 5173 | M=16;efConstruction=200; efSearch=48 |
| SIFT-1M | IVF-Flat | 2366 | 1439 | nlist=2048; nprobe=32 |
| SIFT-1M | IVF-PQ | not reached (max 0.886) | not reached (max 0.886) | - |
| GloVe-100 | Exact (Flat) | 28 (recall 1.0) | 28 (recall 1.0) | IndexFlatL2 |
| GloVe-100 | HNSW | 1161 | 635 | M=32;efConstruction=200; efSearch=384 |
| GloVe-100 | IVF-Flat | 461 | 236 | nlist=2048; nprobe=256 |
| GloVe-100 | IVF-PQ | not reached (max 0.868) | not reached (max 0.868) | - |
