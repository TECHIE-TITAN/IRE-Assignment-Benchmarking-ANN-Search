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

**Drifting data on synthetic clusters.** Every index is built on one distribution, after which vectors from a shifted version of that distribution are added without retraining, and queries come from the shifted version too. The question is which index loses quality, and whether retraining or rebuilding on all the data brings it back.

**Setup.** The data is a mixture of Gaussian clusters from a seeded generator (`src/synthetic_data.py`): 64 dimensions, 64 clusters, 100K base vectors and 50K vectors added later. "Drift" translates the whole mixture by a fixed vector, and the shift size is measured in cluster radii (one radius is the typical distance of a vector from its cluster centre, 8 in this setup). Four radii is about the typical distance between two cluster centres, so at that size the added vectors sit where there were no clusters before. Only the shift size changes between runs: the shifts are 0 (the unchanged control), 1, 2 and 4 radii, with the same base vectors, the same noise and the same shift direction every time. Recall is measured against exact search over all 150K vectors. The indexes are HNSW, IVF-Flat and IVF-PQ, each with its search setting fixed at the value that gives recall close to 0.95 on the control, the operating point used in the rest of the report (or as high as the index gets, if PQ cannot reach it). After the drifted run, each IVF index is retrained on all the data and HNSW is rebuilt from scratch, and the same measurements are repeated.

**Expected outcome.**

- **Control (no shift):** all three indexes behave as tuned, since nothing differs from the data they were built for.
- **IVF-PQ loses the most recall, and loses more as the shift grows.** It stores each vector as a code relative to its nearest centroid, using codebooks learned from the base data. Added vectors sit further from the old centroids than the training vectors ever did, so the codebooks describe them badly and the approximate distances rank neighbours wrongly. More shift means a larger error.
- **IVF-Flat gets uneven lists and slows down.** Added vectors are put into whichever old cell is nearest, so some cells absorb far more vectors than others, and a query that probes those cells scans much more. Recall at a fixed nprobe may also drop, because a drifted cluster can be spread over several old cells.
- **HNSW changes the least.** Each added vector is linked to its nearest neighbours in the graph when it arrives, so the graph follows the new data and nothing in it is tied to the old distribution. This is the least certain of the predictions, because a far-away group of added vectors might end up only weakly connected to the old part of the graph.
- **Retraining helps the IVF indexes:** new centroids and codebooks fit all the data, so lists even out and recall returns to about the control level. A rebuilt HNSW should show little to recover, since it lost little.

Overall, the ordering of damage should be IVF-PQ, then IVF-Flat, then HNSW, growing with the shift.

**What would count against it.**

- HNSW losing as much recall as IVF-Flat or more, or slowing down more.
- IVF-PQ not getting worse as the shift grows.
- Retraining not restoring recall for the IVF indexes.
- No visible change in any index at the largest shift, which would mean this kind of drift is harmless at this scale.

**Not tested.** Rotation of the data, new clusters appearing separately from a shift of the existing ones, gradual drift over time, deletions, other dimensions or cluster counts, drift in queries only, and other HNSW or IVF settings. List sizes after drift are recorded as a diagnostic for IVF-Flat, but no numbers are predicted for them.
