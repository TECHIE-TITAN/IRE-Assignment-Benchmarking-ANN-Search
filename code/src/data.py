"""Dataset loading and the tune/final query split.

SIFT is used as-is (Euclidean). GloVe-100 is L2-normalised so that Euclidean
distance gives the same ordering as angular distance.

The `neighbors` array stored in the HDF5 files is only valid for the full
training set. For subsets (--quick, scaling) recompute ground truth with the
exact index instead.
"""
from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = {
    "sift": "sift-128-euclidean.hdf5",
    "glove": "glove-100-angular.hdf5",
}

N_QUERIES = 10_000
TUNE = slice(0, 5_000)
FINAL = slice(5_000, 10_000)


def normalise(x):
    """Row-wise L2 normalisation; zero rows are left as zeros."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def load_dataset(name, n=None):
    """Return dict with train, test, neighbors (None if n is given).

    n: keep only the first n training vectors.
    """
    if name not in FILES:
        raise ValueError(f"unknown dataset {name!r}, expected one of {sorted(FILES)}")
    path = DATA_DIR / FILES[name]
    with h5py.File(path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"])
    if name == "glove":
        train = normalise(train)
        test = normalise(test)
    if n is not None:
        train = np.ascontiguousarray(train[:n])
        neighbors = None
    return {"name": name, "train": train, "test": test, "neighbors": neighbors}


def split_queries(test):
    """Return (tune, final) query arrays: first 5,000 and last 5,000."""
    return (
        np.ascontiguousarray(test[TUNE]),
        np.ascontiguousarray(test[FINAL]),
    )
