| Dataset | Index | Config | Build time (s) | Index size (MB) | Bytes / vector |
|---|---|---|---|---|---|
| SIFT-1M | HNSW | M=16;efConstruction=200; efSearch=48 | 164.7 | 656.3 | 656.3 |
| SIFT-1M | IVF-Flat | nlist=2048; nprobe=32 | 64.1 | 521.1 | 521.1 |
| SIFT-1M | IVF-PQ | nlist=4096;m=64;nbits=8; nprobe=1024 (0.95 not reached; best recall 0.886) | 205.9 | 74.3 | 74.3 |
| GloVe-100 | HNSW | M=32;efConstruction=200; efSearch=384 | 366.6 | 795.5 | 672.1 |
| GloVe-100 | IVF-Flat | nlist=4096; nprobe=512 | 172.8 | 484.5 | 409.4 |
| GloVe-100 | IVF-PQ | nlist=4096;m=50;nbits=8; nprobe=1024 (0.95 not reached; best recall 0.868) | 148.9 | 70.4 | 59.5 |
