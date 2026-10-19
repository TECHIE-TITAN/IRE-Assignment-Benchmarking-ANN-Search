"""FAISS index builders for HNSW, IVF-Flat and IVF-PQ, all with L2 distance.

Every index is built through build_index(), which returns a Built record with
build time (training plus adding) and serialised size in bytes. The
search-time parameter (efSearch for HNSW, nprobe for the IVF indexes) is set
with set_search_param().

Builds use all available threads. HNSW insertion order is therefore not
strictly deterministic across runs; k-means seeds are fixed.
"""
import time
from dataclasses import dataclass, field

import faiss

SEED = 1234
KINDS = ("hnsw", "ivfflat", "ivfpq")
SEARCH_PARAM = {"hnsw": "efSearch", "ivfflat": "nprobe", "ivfpq": "nprobe"}


@dataclass
class Built:
    kind: str
    index: object
    params: dict
    build_time_s: float
    size_bytes: int
    n: int
    # objects that must outlive the index (e.g. the IVF coarse quantizer)
    keepalive: list = field(default_factory=list, repr=False)

    @property
    def search_param_name(self):
        return SEARCH_PARAM[self.kind]


def params_str(params):
    return ";".join(f"{k}={v}" for k, v in params.items())


def build_index(kind, train, **params):
    """Build an index over `train` (float32, shape (n, d)).

    hnsw:    M, efConstruction (default 200)
    ivfflat: nlist
    ivfpq:   nlist, m (sub-vectors, must divide d), nbits (default 8)
    """
    n, d = train.shape
    keepalive = []
    t0 = time.perf_counter()

    if kind == "hnsw":
        params = {"M": params["M"], "efConstruction": params.get("efConstruction", 200)}
        index = faiss.IndexHNSWFlat(d, params["M"])
        index.hnsw.efConstruction = params["efConstruction"]
        index.add(train)

    elif kind == "ivfflat":
        params = {"nlist": params["nlist"]}
        quantizer = faiss.IndexFlatL2(d)
        keepalive.append(quantizer)
        index = faiss.IndexIVFFlat(quantizer, d, params["nlist"])
        index.cp.seed = SEED
        index.train(train)
        index.add(train)

    elif kind == "ivfpq":
        params = {"nlist": params["nlist"], "m": params["m"], "nbits": params.get("nbits", 8)}
        if d % params["m"]:
            raise ValueError(f"m={params['m']} does not divide d={d}")
        quantizer = faiss.IndexFlatL2(d)
        keepalive.append(quantizer)
        index = faiss.IndexIVFPQ(quantizer, d, params["nlist"], params["m"], params["nbits"])
        index.cp.seed = SEED
        index.pq.cp.seed = SEED
        index.train(train)
        index.add(train)

    else:
        raise ValueError(f"unknown index kind {kind!r}, expected one of {KINDS}")

    build_time = time.perf_counter() - t0
    size_bytes = int(faiss.serialize_index(index).nbytes)
    return Built(kind, index, params, build_time, size_bytes, n, keepalive)


def set_search_param(built, value):
    if built.kind == "hnsw":
        built.index.hnsw.efSearch = int(value)
    else:
        built.index.nprobe = int(value)


def search(built, queries, k=10):
    """Return ids only, shape (nq, k)."""
    return built.index.search(queries, k)[1]
