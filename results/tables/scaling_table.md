Cost at recall@10 >= 0.95, one thread, one query per call (SIFT, final queries)

| Index | ms/query @ 100K | ms/query @ 300K | ms/query @ 1M | growth 300K vs 100K | growth 1M vs 100K | scaling exponent |
|---|---|---|---|---|---|---|
| Exact (Flat) | 3.657 | 11.162 | 37.023 | 3.1x | 10.1x | 1.01 |
| HNSW | 0.073 | 0.129 | 0.193 | 1.8x | 2.6x | 0.42 |
| IVF-Flat | 0.156 | 0.377 | 0.695 | 2.4x | 4.5x | 0.65 |

Speedup over exact search at the same size

| Index | 100K | 300K | 1M |
|---|---|---|---|
| HNSW | 50x | 87x | 192x |
| IVF-Flat | 23x | 30x | 53x |

Configuration, recall, build time and size behind each point

| Index | n | Config | recall@10 | Build time (s) | Size (MB) |
|---|---|---|---|---|---|
| Exact (Flat) | 100,000 | IndexFlatL2 | 1.0000 | 0.0 | 51.2 |
| Exact (Flat) | 300,000 | IndexFlatL2 | 1.0000 | 0.1 | 153.6 |
| Exact (Flat) | 1,000,000 | IndexFlatL2 | 1.0000 | 0.5 | 512.0 |
| HNSW | 100,000 | M=32;efConstruction=200; efSearch=24 | 0.9538 | 5.3 | 78.4 |
| HNSW | 300,000 | M=8;efConstruction=200; efSearch=64 | 0.9540 | 20.1 | 177.8 |
| HNSW | 1,000,000 | M=16;efConstruction=200; efSearch=48 | 0.9556 | 112.3 | 656.3 |
| IVF-Flat | 100,000 | nlist=1024; nprobe=32 | 0.9652 | 4.5 | 52.5 |
| IVF-Flat | 300,000 | nlist=512; nprobe=16 | 0.9562 | 3.3 | 156.3 |
| IVF-Flat | 1,000,000 | nlist=2048; nprobe=32 | 0.9611 | 46.2 | 521.1 |
