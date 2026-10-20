Cost at recall@10 >= 0.95, one thread, one query per call (SIFT, final queries)

| Index | ms/query @ 100K | ms/query @ 300K | ms/query @ 1M | growth 300K vs 100K | growth 1M vs 100K | scaling exponent |
|---|---|---|---|---|---|---|
| Exact (Flat) | 3.869 | 11.577 | 42.105 | 3.0x | 10.9x | 1.04 |
| HNSW | 0.079 | 0.137 | 0.249 | 1.7x | 3.2x | 0.50 |
| IVF-Flat | 0.154 | 0.400 | 0.840 | 2.6x | 5.4x | 0.74 |

Speedup over exact search at the same size

| Index | 100K | 300K | 1M |
|---|---|---|---|
| HNSW | 49x | 84x | 169x |
| IVF-Flat | 25x | 29x | 50x |

Configuration, recall, build time and size behind each point

| Index | n | Config | recall@10 | Build time (s) | Size (MB) |
|---|---|---|---|---|---|
| Exact (Flat) | 100,000 | IndexFlatL2 | 1.0000 | 0.1 | 51.2 |
| Exact (Flat) | 300,000 | IndexFlatL2 | 1.0000 | 0.1 | 153.6 |
| Exact (Flat) | 1,000,000 | IndexFlatL2 | 1.0000 | 0.5 | 512.0 |
| HNSW | 100,000 | M=32;efConstruction=200; efSearch=24 | 0.9536 | 5.7 | 78.4 |
| HNSW | 300,000 | M=8;efConstruction=200; efSearch=64 | 0.9547 | 21.1 | 177.8 |
| HNSW | 1,000,000 | M=16;efConstruction=200; efSearch=48 | 0.9563 | 164.7 | 656.3 |
| IVF-Flat | 100,000 | nlist=1024; nprobe=32 | 0.9652 | 4.2 | 52.5 |
| IVF-Flat | 300,000 | nlist=512; nprobe=16 | 0.9562 | 3.6 | 156.3 |
| IVF-Flat | 1,000,000 | nlist=2048; nprobe=32 | 0.9611 | 64.1 | 521.1 |
