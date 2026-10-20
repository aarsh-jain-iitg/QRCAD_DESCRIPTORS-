"""
================================================================================
FILE: base.py
================================================================================
PURPOSE:
    The common parent class for all fifteen shape descriptors, plus the small
    number helpers they share: histograms, drawing random groups of points,
    skewness, and the optional GPU helpers.
    Keeping these here means each descriptor file holds only its own idea.

INPUTS:
    - Descriptor settings (dict) from [descriptors.<name>] in config.toml.
    - PreparedMesh (src/preprocessing.py) and a seeded random generator.

OUTPUTS:
    - Descriptor.compute(...) -> np.ndarray float32, shape (dim,).

DEPENDENCIES:
    stdlib      : abc, math
    third-party : numpy
    internal    : preprocessing (type only)

PHYSICS / DOMAIN CONTEXT:
    - Every output has no units: lengths are divided by the part size D_diam,
      areas by D_diam^2, and so on. A part and its scaled copy give the same
      vector.
    - Histograms are scaled to sum to 1, then square-rooted. With that, the
      dot product of two histograms measures how much they overlap
      (1 = identical, 0 = no overlap).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
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
EPS = 1e-12                     # [-] smallest value we are willing to divide by
ALL_INVARIANCES = frozenset({"translation", "rotation", "scale", "reflection"})


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def angle_between_lines_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two lines (direction sign ignored) [deg], in [0, 90]."""
    c = min(1.0, abs(float(np.dot(a, b))))
    return math.degrees(math.acos(c))


def distinct_tuples(rng: np.random.Generator, n: int, S: int, k: int) -> np.ndarray:
    """
    Pick S random groups of k point indices, dropping groups that repeat a point.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded generator.
    n : int
        Number of points to pick from. Must be >= k.
    S : int
        Number of groups to draw.
    k : int
        Group size (2 pairs, 3 triples, 4 quadruples).

    Returns
    -------
    np.ndarray, shape (S', k), int64
        S' <= S groups with no repeated point (about k^2/2n of them are dropped).
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
    Histogram on [lo, hi], scaled to sum 1, then square-rooted.

    Values outside the range are pushed into the first or last bin.

    Parameters
    ----------
    values : np.ndarray, shape (N,)
        Unitless values.
    bins : int
        Number of equal-width bins.
    lo, hi : float
        Histogram range.
    weights : np.ndarray, shape (N,), optional
        Non-negative weight of each value.

    Returns
    -------
    np.ndarray, shape (bins,), float64
        sqrt(h / sum h); all zeros if the histogram is empty.
    """
    v = np.clip(values, lo, hi)
    h, _ = np.histogram(v, bins=bins, range=(lo, hi), weights=weights)
    return hellinger(h)


def hellinger(h: np.ndarray) -> np.ndarray:
    """Scale a non-negative array to sum 1 and take the square root (zeros stay zeros)."""
    h = np.asarray(h, dtype=np.float64)
    s = h.sum()
    return np.sqrt(h / s) if s > EPS else np.zeros_like(h)


def resolve_torch_device(device: str) -> Any:
    """
    Return a torch GPU device for "cuda", or None to run on the CPU.

    torch is only imported when a GPU is asked for, so CPU runs stay fast to start.
    """
    if device != "cuda":
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch.device("cuda") if torch.cuda.is_available() else None


def skewness(x: np.ndarray) -> float:
    """Skewness (lopsidedness) E[(x-mu)^3]/sigma^3; 0 when all values are equal."""
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
    Histogram where each value is shared between its two nearest bins.

    A value exactly on a bin edge (a 90 deg CAD corner with 10 deg bins)
    lands half in each bin, instead of jumping from one bin to the other
    because of tiny rounding errors. Result is scaled to sum 1 and
    square-rooted, like hellinger_hist.

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


def torch_gather(P: np.ndarray, idx: np.ndarray, dev: Any) -> Any:
    """
    Copy the points to the GPU and look up each index group there, in float64.

    Parameters
    ----------
    P : np.ndarray, shape (n, 3)
        Points [L].
    idx : np.ndarray, shape (S, k), int
        Index groups (drawn on the CPU so CPU and GPU see the same groups).
    dev : torch.device

    Returns
    -------
    torch.Tensor, shape (S, k, 3), float64
    """
    import torch
    Pt = torch.as_tensor(P, dtype=torch.float64, device=dev)
    return Pt[torch.as_tensor(idx, device=dev)]


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class DescriptorError(ValueError):
    """Raised when a mesh cannot give a valid vector for a descriptor."""


class Descriptor(ABC):
    """
    One shape descriptor from CAD_Descriptors_Explained.pdf.

    A subclass sets the class attributes and its default settings, works out
    ``dim`` (output length) in ``_setup``, and does the maths in ``_compute``.
    The first ``n_hist`` output values are histogram values (scaled and
    square-rooted); the rest are plain numbers that the orchestrator later
    standardises using the whole database (subtract mean, divide by std).
    """

    name: str = ""
    version: str = "1"
    metric: str = "hellinger"          # "hellinger" (histogram) | "l2" (plain numbers)
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
        """Work out ``dim`` and ``n_hist`` from ``self.params``."""
        raise NotImplementedError

    @abstractmethod
    def _compute(self, mesh: PreparedMesh, rng: np.random.Generator) -> np.ndarray:
        """Return the raw descriptor, shape (dim,)."""

    def compute(self, mesh: PreparedMesh, rng: np.random.Generator) -> np.ndarray:
        """
        Compute the descriptor and check the result.

        Returns
        -------
        np.ndarray, shape (dim,), float32
            Vector with no NaN or infinite values.

        Raises
        ------
        DescriptorError
            If the mesh is unusable or the result has NaN / infinite values.
        """
        v = np.asarray(self._compute(mesh, rng), dtype=np.float64).ravel()
        if v.shape != (self.dim,):
            raise DescriptorError(f"{self.name}: shape {v.shape} != ({self.dim},)")
        if not np.all(np.isfinite(v)):
            raise DescriptorError(f"{self.name}: non-finite values in output")
        return v.astype(np.float32)

    def cache_params(self) -> dict[str, Any]:
        """Settings that change the vector (they go into the cache key)."""
        return dict(sorted(self.params.items()))

# ==============================================================================
# END OF FILE: base.py
# ==============================================================================
# SUMMARY:
#   Parent Descriptor class + shared helpers (histograms, random groups, skewness).
#
# USAGE EXAMPLE:
#   from src.descriptors.base import Descriptor, hellinger_hist
#   class MyDesc(Descriptor): ...
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] All random groups are drawn at once (memory grows with S); draw them
#         in chunks if S grows past ~1e7.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-04  torch_gather: shared GPU lookup for the point-group descriptors
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
