"""
================================================================================
FILE: base.py
================================================================================
PURPOSE:
    Abstract Descriptor interface shared by all fifteen shape descriptors, plus
    the small numeric helpers they all need (Hellinger histograms, distinct
    random index tuples, safe normalisation, optional torch device math).
    Keeping these here means every descriptor file contains only its own math.

INPUTS:
    - Descriptor params (dict) from [descriptors.<name>] in config.toml.
    - PreparedMesh (src/preprocessing.py) and a seeded np.random.Generator.

OUTPUTS:
    - Descriptor.compute(...) -> np.ndarray float32, shape (dim,).

DEPENDENCIES:
    stdlib      : abc, math
    third-party : numpy
    internal    : preprocessing (type only)

PHYSICS / DOMAIN CONTEXT:
    - Every vector is dimensionless (lengths / D_diam, areas / D_diam^2, ...),
      see CAD_Descriptors_Explained.pdf.
    - Histogram blocks are L1-normalised then square-rooted (Hellinger map),
      so the inner product of two blocks is the Bhattacharyya similarity.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.preprocessing import PreparedMesh

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
EPS = 1e-12                     # [-] guard for divisions by norms / masses
ALL_INVARIANCES = frozenset({"translation", "rotation", "scale", "reflection"})


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def angle_between_lines_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two undirected lines [deg], in [0, 90]."""
    c = min(1.0, abs(float(np.dot(a, b))))
    return math.degrees(math.acos(c))


def distinct_tuples(rng: np.random.Generator, n: int, S: int, k: int) -> np.ndarray:
    """
    Draw S index k-tuples from range(n) and drop tuples with repeated entries.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded generator.
    n : int
        Pool size. Must be >= k.
    S : int
        Number of tuples to draw.
    k : int
        Tuple length (2 pairs, 3 triples, 4 quadruples).

    Returns
    -------
    np.ndarray, shape (S', k), int64
        S' <= S tuples with pairwise distinct entries (collision rate ~k^2/2n).
    """
    if n < k:
        raise DescriptorError(f"need at least {k} surface samples, got {n}")
    idx = rng.integers(0, n, size=(S, k))
    ok = np.ones(S, dtype=bool)
    for a in range(k):
        for b in range(a + 1, k):
            ok &= idx[:, a] != idx[:, b]
    return idx[ok]


def hellinger_hist(values: np.ndarray, bins: int, lo: float, hi: float,
                   weights: np.ndarray | None = None) -> np.ndarray:
    """
    Histogram on [lo, hi] (values clipped into range), L1-normalised, sqrt-mapped.

    Parameters
    ----------
    values : np.ndarray, shape (N,)
        Dimensionless samples.
    bins : int
        Number of equal-width bins.
    lo, hi : float
        Histogram range.
    weights : np.ndarray, shape (N,), optional
        Non-negative sample weights.

    Returns
    -------
    np.ndarray, shape (bins,), float64
        sqrt(h / sum h); all zeros if the total mass is zero.
    """
    v = np.clip(values, lo, hi)
    h, _ = np.histogram(v, bins=bins, range=(lo, hi), weights=weights)
    return hellinger(h)


def hellinger(h: np.ndarray) -> np.ndarray:
    """L1-normalise a non-negative array and apply sqrt (zeros stay zeros)."""
    h = np.asarray(h, dtype=np.float64)
    s = h.sum()
    return np.sqrt(h / s) if s > EPS else np.zeros_like(h)


def resolve_torch_device(device: str) -> Any:
    """
    Return a torch.device for "cuda", or None for CPU / missing torch.

    torch is imported lazily so CPU-only runs never pay its import cost.
    """
    if device != "cuda":
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch.device("cuda") if torch.cuda.is_available() else None


def skewness(x: np.ndarray) -> float:
    """Sample skewness gamma_1 = E[(x-mu)^3]/sigma^3; 0 when sigma ~ 0."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return 0.0
    mu = x.mean()
    sigma = x.std()
    if sigma <= EPS * max(1.0, abs(mu)):
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


def soft_hellinger_hist(values: np.ndarray, bins: int, lo: float, hi: float,
                        weights: np.ndarray | None = None) -> np.ndarray:
    """
    Linearly interpolated (soft) histogram, L1-normalised and sqrt-mapped.

    Each value splits its weight between the two nearest bin centres, so a
    value exactly on a bin edge (a 90 deg CAD crease with 10 deg bins) lands
    half in each neighbour instead of flipping bins with rounding noise.
    The histogram is then continuous in the values.

    Parameters
    ----------
    values : np.ndarray, shape (N,)
    bins : int
    lo, hi : float
        Range; bin k is centred at lo + (k + 0.5)(hi - lo)/bins.
    weights : np.ndarray, shape (N,), optional

    Returns
    -------
    np.ndarray, shape (bins,)
    """
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=np.float64)
    u = np.clip((np.asarray(values, dtype=np.float64) - lo) / (hi - lo) * bins - 0.5, 0.0, bins - 1)
    k0 = np.floor(u).astype(np.int64)
    frac = u - k0
    k1 = np.minimum(k0 + 1, bins - 1)
    h = np.bincount(k0, weights=w * (1.0 - frac), minlength=bins) \
        + np.bincount(k1, weights=w * frac, minlength=bins)
    return hellinger(h)


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class DescriptorError(ValueError):
    """Raised when a mesh cannot produce a valid vector for a descriptor."""


class Descriptor(ABC):
    """
    One shape descriptor from CAD_Descriptors_Explained.pdf.

    Subclasses set the class attributes, declare ``defaults`` (PDF values),
    compute ``dim`` in ``_setup`` from their params, and implement ``_compute``.
    The first ``n_hist`` output dims are Hellinger-mapped histogram values;
    the remaining dims are raw scalars that the orchestrator z-scores with
    database statistics.
    """

    name: str = ""
    version: str = "1"
    metric: str = "hellinger"          # "hellinger" (histogram) | "l2" (z-scored)
    invariances: frozenset[str] = ALL_INVARIANCES
    defaults: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any] | None = None, device: str = "cpu") -> None:
        self.params: dict[str, Any] = {**self.defaults, **(params or {})}
        self.device = device
        self.dim: int = 0
        self.n_hist: int = 0
        self._setup()
        if self.dim <= 0:
            raise TypeError(f"{type(self).__name__} did not set dim")

    def _setup(self) -> None:
        """Derive ``dim`` and ``n_hist`` from ``self.params``."""
        raise NotImplementedError

    @abstractmethod
    def _compute(self, mesh: PreparedMesh, rng: np.random.Generator) -> np.ndarray:
        """Return the raw descriptor, shape (dim,)."""

    def compute(self, mesh: PreparedMesh, rng: np.random.Generator) -> np.ndarray:
        """
        Compute the descriptor and enforce the output contract.

        Returns
        -------
        np.ndarray, shape (dim,), float32
            Finite vector.

        Raises
        ------
        DescriptorError
            If the mesh is unusable or the result is not finite.
        """
        v = np.asarray(self._compute(mesh, rng), dtype=np.float64).ravel()
        if v.shape != (self.dim,):
            raise DescriptorError(f"{self.name}: shape {v.shape} != ({self.dim},)")
        if not np.all(np.isfinite(v)):
            raise DescriptorError(f"{self.name}: non-finite values in output")
        return v.astype(np.float32)

    def cache_params(self) -> dict[str, Any]:
        """Params that change the vector (hashed into the cache key)."""
        return dict(sorted(self.params.items()))

# ==============================================================================
# END OF FILE: base.py
# ==============================================================================
# SUMMARY:
#   Abstract Descriptor + shared numeric helpers (Hellinger hist, tuples, skew).
#
# USAGE EXAMPLE:
#   from src.descriptors.base import Descriptor, hellinger_hist
#   class MyDesc(Descriptor): ...
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Tuple drawing is vectorised (Theta(S) memory); stream in chunks if
#         S grows past ~1e7.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
