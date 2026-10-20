# Hypotheses

## Part 3

**Scaling on SIFT.** The same benchmark is run on SIFT at 100K, 300K and 1M vectors to see how much slower each approach gets as the data grows: exact search, HNSW and IVF-Flat. "Cost" here means the time per query (one thread, one query at a time) of the fastest setting that still reaches recall@10 of 0.95. Build settings come from the same grid as Part 2, and the 1M point is the Part 2 result.

This was written after seeing the Part 2 results at 1M, so the expectations below lean on them a little.

**Expected outcome.** Going from 100K to 1M vectors:

- **Exact search** gets slower in direct proportion to the data. Every vector is checked for every query, and there are ten times as many vectors, so the cost should grow about tenfold.
- **IVF-Flat** gets slower by clearly less than exact search, but by more than HNSW. As the data grows, the number of lists grows too, so each list stays a similar size. More lists need to be probed to keep recall at 0.95, but not in proportion to the total, so the share of the data that gets scanned shrinks.
- **HNSW** grows the most slowly of the three. A graph search follows a path whose length grows only slowly with the number of points, and efSearch has to rise only a little to hold recall.

Overall, exact search should be hit hardest, then IVF-Flat, then HNSW, and the gap between ANN and exact search should widen as the data grows.

**What would count against it.**

- Exact search not growing roughly in proportion to n.
- IVF-Flat growing about as fast as exact search, which would mean it ends up scanning a fixed fraction of the data.
- HNSW growing faster than IVF-Flat.
- HNSW's speedup over exact search not getting bigger as the data grows.

**Not tested.** GloVe, other efConstruction values, multi-thread throughput, and sizes beyond 1M. Build time is reported as well, but no prediction is made about it.

## Part 4

_To be written before running the Part 4 experiment._
