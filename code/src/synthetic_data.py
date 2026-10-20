"""Seeded synthetic data for Part 4 (drifting data).

The data is a mixture of Gaussian clusters in `d` dimensions:
  - cluster centers ~ N(0, center_scale^2) per coordinate,
  - each vector = its cluster's center + N(0, sigma^2) noise per coordinate.

"Drift" translates the whole mixture by a fixed vector. The shift size is given
in cluster radii, where one radius = sigma * sqrt(d) is the typical distance of
a vector from its cluster center.

One thing changes at a time: the base vectors, the noise drawn for the new
vectors and the queries, and the shift direction are all fixed by the seed.
Changing the shift size only translates the new vectors and the queries, so
a run with shift 0 is the unchanged control and every other run differs from it
by that translation alone.

Every random stream is derived from (seed, stream id), so the same config
always gives identical arrays.
"""
import math
from dataclasses import dataclass

import numpy as np

MAX_VECTORS = 500_000  # assignment limit on synthetic vectors

# stream ids: independent random streams under one seed
CENTERS, BASE, NEW, QUERIES, DIRECTION = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class DriftConfig:
    d: int = 64
    n_clusters: int = 64
    sigma: float = 1.0
    center_scale: float = 3.0
    n_base: int = 100_000    # vectors the index is first built on
    n_new: int = 50_000      # vectors added later, from the drifted mixture
    n_queries: int = 10_000  # queries, drawn from the drifted mixture
    seed: int = 0

    @property
    def radius(self):
        """Typical distance of a vector from its cluster center."""
        return self.sigma * math.sqrt(self.d)


def _rng(cfg, stream):
    return np.random.default_rng([cfg.seed, stream])


def make_centers(cfg):
    return _rng(cfg, CENTERS).normal(0.0, cfg.center_scale, (cfg.n_clusters, cfg.d))


def sample_mixture(cfg, centers, n, stream):
    """n vectors from the (unshifted) mixture, float64 until the caller casts."""
    rng = _rng(cfg, stream)
    labels = rng.integers(0, cfg.n_clusters, n)
    return centers[labels] + cfg.sigma * rng.standard_normal((n, cfg.d))


def shift_vector(cfg, radii):
    """Fixed random direction times radii * cluster radius."""
    u = _rng(cfg, DIRECTION).standard_normal(cfg.d)
    u /= np.linalg.norm(u)
    return u * radii * cfg.radius


def make_drift_dataset(cfg, shift_radii):
    """Return dict with base, new, queries (float32) and the applied shift.

    base:    drawn from the original mixture
    new:     drawn from the original mixture, then translated by the shift
    queries: drawn from the original mixture, then translated by the shift
    shift_radii = 0 gives new and queries from the original mixture (control).
    """
    if cfg.n_base + cfg.n_new > MAX_VECTORS:
        raise ValueError(
            f"n_base + n_new = {cfg.n_base + cfg.n_new} exceeds the {MAX_VECTORS} vector limit")
    centers = make_centers(cfg)
    shift = shift_vector(cfg, shift_radii)
    return {
        "base": sample_mixture(cfg, centers, cfg.n_base, BASE).astype(np.float32),
        "new": (sample_mixture(cfg, centers, cfg.n_new, NEW) + shift).astype(np.float32),
        "queries": (sample_mixture(cfg, centers, cfg.n_queries, QUERIES) + shift).astype(np.float32),
        "shift": shift,
        "shift_radii": shift_radii,
    }
