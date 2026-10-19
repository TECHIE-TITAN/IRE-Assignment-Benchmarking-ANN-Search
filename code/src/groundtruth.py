"""Exact ground truth with FAISS IndexFlatL2, cached on disk under data/.

Distances are squared L2, as FAISS returns them. The cache is keyed by dataset
name, number of base vectors and number of queries.
"""
import numpy as np

from data import DATA_DIR


def exact_knn(base, queries, k=100):
    """Exact top-k: returns (squared distances, ids)."""
    import faiss

    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    return index.search(queries, k)


def load_or_compute_gt(name, base, queries, k=100):
    """Ground truth for `queries` against `base`, computed once and cached."""
    path = DATA_DIR / f"gt_{name}_{len(base)}_{len(queries)}_k{k}.npz"
    if path.exists():
        z = np.load(path)
        return z["D"], z["I"]
    D, I = exact_knn(base, queries, k)
    np.savez(path, D=D, I=I)
    return D, I


def kth_dist_sq(D, k=10):
    """True squared distance of the k-th neighbour, shape (nq,)."""
    return D[:, k - 1]
