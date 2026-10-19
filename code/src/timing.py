"""Timing helpers: single-thread, one query per call, median over repeats.

`search_fn` takes a 2-D float32 array of queries, shape (m, d), and runs the
search. For per-query timing it is called with shape (1, d); for batch timing
with all queries at once.
"""
import statistics
import time

import numpy as np


def set_single_thread():
    """Restrict FAISS (OpenMP) to one thread. hnswlib needs
    index.set_num_threads(1) on the index object itself."""
    import faiss

    faiss.omp_set_num_threads(1)


def _split_rows(queries):
    q = np.ascontiguousarray(queries, dtype=np.float32)
    return [np.ascontiguousarray(q[i:i + 1]) for i in range(len(q))]


def time_single_queries(search_fn, queries, repeats=3, warmup=100):
    """Median queries/second over `repeats` passes of one-query-at-a-time search.

    Rows are sliced before the clock starts, so only search is timed.
    Returns dict with qps (median), all per-pass qps values and the repeat count.
    """
    rows = _split_rows(queries)
    for row in rows[:warmup]:
        search_fn(row)
    qps = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for row in rows:
            search_fn(row)
        elapsed = time.perf_counter() - t0
        qps.append(len(rows) / elapsed)
    return {"qps": statistics.median(qps), "qps_runs": qps, "repeats": repeats}


def time_batch(search_fn, queries, repeats=3):
    """Median queries/second over `repeats` runs of one all-queries-at-once call."""
    q = np.ascontiguousarray(queries, dtype=np.float32)
    search_fn(q[:min(len(q), 100)])
    qps = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        search_fn(q)
        elapsed = time.perf_counter() - t0
        qps.append(len(q) / elapsed)
    return {"qps": statistics.median(qps), "qps_runs": qps, "repeats": repeats}
