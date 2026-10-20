| Dataset | Index | QPS @ recall 0.90 | QPS @ recall 0.95 | Config at 0.95 |
|---|---|---|---|---|
| SIFT-1M | Exact (Flat) | 24 (recall 1.0) | 24 (recall 1.0) | IndexFlatL2 |
| SIFT-1M | HNSW | 4816 | 4020 | M=16;efConstruction=200; efSearch=48 |
| SIFT-1M | IVF-Flat | 1568 | 1191 | nlist=2048; nprobe=32 |
| SIFT-1M | IVF-PQ | not reached (max 0.886) | not reached (max 0.886) | - |
| GloVe-100 | Exact (Flat) | 25 (recall 1.0) | 25 (recall 1.0) | IndexFlatL2 |
| GloVe-100 | HNSW | 982 | 551 | M=32;efConstruction=200; efSearch=384 |
| GloVe-100 | IVF-Flat | 383 | 196 | nlist=4096; nprobe=512 |
| GloVe-100 | IVF-PQ | not reached (max 0.868) | not reached (max 0.868) | - |
