"""Tie-aware recall@k.

A returned id counts as correct if its true squared distance to the query is
at most the true k-th neighbour's squared distance (within a small relative
tolerance). Comparing by distance rather than by id means an equidistant
alternative at rank k is not counted as a mistake. Distances are recomputed
from the base vectors, so approximate distances reported by the index (e.g.
PQ) do not matter. Missing results (id -1, as FAISS pads) and repeated ids
count as misses.
"""
import numpy as np


def recall_at_k(found_ids, kth_dist_sq, base, queries, k=10, rtol=1e-5):
    """Mean recall@k over queries.

    found_ids:   (nq, >=k) ids returned by the index; only the first k are used.
    kth_dist_sq: (nq,) true squared distance to the k-th exact neighbour.
    base:        (n, d) indexed vectors.
    queries:     (nq, d) query vectors.
    """
    found_ids = np.asarray(found_ids)
    kth_dist_sq = np.asarray(kth_dist_sq, dtype=np.float64)
    nq = len(queries)
    if found_ids.ndim != 2 or found_ids.shape[0] != nq or found_ids.shape[1] < k:
        raise ValueError(f"found_ids must have shape ({nq}, >={k}), got {found_ids.shape}")
    if kth_dist_sq.shape != (nq,):
        raise ValueError(f"kth_dist_sq must have shape ({nq},), got {kth_dist_sq.shape}")

    ids = found_ids[:, :k]
    valid = ids >= 0

    dup = np.zeros_like(valid)
    for j in range(1, k):
        dup[:, j] = (ids[:, j:j + 1] == ids[:, :j]).any(axis=1)

    safe = np.where(valid, ids, 0)
    diff = base[safe].astype(np.float64) - queries[:, None, :].astype(np.float64)
    dist_sq = np.einsum("qkd,qkd->qk", diff, diff)

    threshold = kth_dist_sq[:, None] * (1.0 + rtol) + 1e-12
    correct = valid & ~dup & (dist_sq <= threshold)
    return float(correct.sum(axis=1).clip(max=k).mean() / k)
