"""
================================================================================
FILE: metrics.py
================================================================================
PURPOSE:
    Vectorised distances between 781-d block vectors: one distance per sub-part
    (W1 for ordered histograms, Hellinger, L2, |delta|, |delta log|, L1 on log
    counts), combined into one distance per block with no hand weights:
    D_b = sum_j D_bj / median_R(D_bj), the medians being fitted on random pairs.

INPUTS:
    - x_a : (781,) or (K, 781) query rows; X_b : (K, 781) candidate rows.
    - dim_scale : (781,) robust per-dimension scale for scalar sub-parts (default 1).
    - medians : (S,) null medians of the sub-part distances (default 1).

OUTPUTS:
    - subpart_distances -> (K, S); block_distances -> (K, 15); all_block_distances -> (K, 15).
    All distances are >= 0 and smaller means more similar.

DEPENDENCIES:
    stdlib      : logging
    third-party : numpy
    internal    : blr.blocks

PHYSICS / DOMAIN CONTEXT:
    - W1(p, r) = dt * sum_k |cumsum(p - r)_k| (dt = bin width in the axis units) grows
      linearly when a peak moves, where cosine / Hellinger saturate at the maximum.
    - Histograms are stored sqrt-mapped (SubPart.stored_sqrt); W1 squares them back.
    - An all-zero histogram (flat part with no curved samples, no creases, empty
      persistence diagram) is "empty": empty vs empty = 0, empty vs non-empty = the
      metric's maximum. The raw metric functions raise ValueError on a zero sum.
    - |delta log| of scalars uses asinh(x / LOG_EPS) (a signed log): ~log(2x / LOG_EPS) for x >> LOG_EPS,
      finite at 0 and for negative values (total mean curvature H can be negative).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import logging

import numpy as np

from src.blr.blocks import BLOCKS, N_DIM, SubPart

logger = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
LOG_EPS = 1e-3                    # [-] scale of the signed log (asinh) for |delta log| scalars
SCALE_FLOOR = 1e-6                # [-] minimum robust scale of a scalar dimension

# (block index, sub-part) of every sub-part that carries a distance, in vector order
DIST_SUBPARTS: tuple[tuple[int, SubPart], ...] = tuple(
    (bi, sp) for bi, b in enumerate(BLOCKS) for sp in b.subparts if sp.metric != "none")
N_SUB = len(DIST_SUBPARTS)
SUB_BLOCK = np.array([bi for bi, _ in DIST_SUBPARTS])
BLOCK_START = np.searchsorted(SUB_BLOCK, np.arange(len(BLOCKS)))     # first column of each block


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _normalise(P: np.ndarray) -> np.ndarray:
    """Rows of P (probability weights) scaled to sum 1; raises ValueError on a zero sum."""
    P = np.asarray(P, dtype=np.float64)
    s = P.sum(axis=-1, keepdims=True)
    if np.any(s <= 0.0):
        raise ValueError("histogram with zero total mass")
    return P / s


def w1_hist(p: np.ndarray, r: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Wasserstein-1 distance between ordered histograms.

    Parameters
    ----------
    p, r : np.ndarray, shape (..., d)
        Non-negative weights (renormalised to sum 1); broadcastable.
    dt : float
        Bin width in the histogram's axis units.

    Returns
    -------
    np.ndarray, shape (...)
        dt * sum_k |cumsum(p - r)_k|  [axis units].
    """
    return dt * np.abs(np.cumsum(_normalise(p) - _normalise(r), axis=-1)).sum(axis=-1)


def hellinger(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """sqrt(max(0, 1 - sum_k sqrt(p_k r_k))) of histograms p, r, shape (..., d); in [0, 1]."""
    bc = np.sqrt(_normalise(p) * _normalise(r)).sum(axis=-1)
    return np.sqrt(np.maximum(0.0, 1.0 - bc))


def l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance over the last axis."""
    return np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64), axis=-1)


def log_count_l1(a: np.ndarray, b: np.ndarray, labels: tuple = ()) -> np.ndarray:
    """L1 between log1p counts; entries labelled "genus" are already log(1 + g)."""
    ta, tb = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    for j, lab in enumerate(labels):
        if lab != "genus":
            ta[..., j], tb[..., j] = np.log1p(ta[..., j]), np.log1p(tb[..., j])
    return np.abs(ta - tb).sum(axis=-1)


def _hist_pair(a: np.ndarray, b: np.ndarray, sp: SubPart, fn) -> np.ndarray:
    """fn on probability histograms; empty-vs-empty = 0, empty-vs-mass = the metric maximum."""
    a, b = (a ** 2, b ** 2) if sp.stored_sqrt else (a, b)
    sa, sb = a.sum(axis=-1), b.sum(axis=-1)
    ea, eb = sa <= 0.0, sb <= 0.0
    d = fn(np.where(ea[..., None], 1.0, a), np.where(eb[..., None], 1.0, b))
    dmax = (a.shape[-1] - 1) * (sp.labels[1] - sp.labels[0]) if sp.metric == "w1" else 1.0
    d = np.where(ea != eb, dmax, d)
    return np.where(ea & eb, 0.0, d)


def _dist(a: np.ndarray, b: np.ndarray, sp: SubPart, scale: np.ndarray) -> np.ndarray:
    """Distance of one sub-part; a, b have shape (K, width) (broadcastable rows)."""
    m = sp.metric
    if m == "w1":
        dt = sp.labels[1] - sp.labels[0]
        return _hist_pair(a, b, sp, lambda p, r: w1_hist(p, r, dt))
    if m == "hellinger":
        return _hist_pair(a, b, sp, hellinger)
    if m == "l2":
        s = scale[sp.sl] if sp.kind == "scalar" else 1.0
        return l2(a / s, b / s)
    if m == "abs":
        return (np.abs(a - b) / scale[sp.sl]).sum(axis=-1)
    if m == "abs_log":
        return np.abs(np.arcsinh(a / LOG_EPS) - np.arcsinh(b / LOG_EPS)).sum(axis=-1)
    if m == "log_count_l1":
        return log_count_l1(a, b, sp.labels)
    raise ValueError(f"unknown metric {m!r}")


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def subpart_distances(x_a: np.ndarray, X_b: np.ndarray,
                      dim_scale: np.ndarray | None = None) -> np.ndarray:
    """
    Distance of every distance-carrying sub-part between x_a and each row of X_b.

    Parameters
    ----------
    x_a : np.ndarray, shape (781,) or (K, 781)
    X_b : np.ndarray, shape (K, 781)
    dim_scale : np.ndarray, shape (781,), optional
        Robust scale of scalar dimensions (default ones).

    Returns
    -------
    np.ndarray, shape (K, N_SUB), float64, >= 0.
    """
    A = np.atleast_2d(np.asarray(x_a, dtype=np.float64))
    B = np.atleast_2d(np.asarray(X_b, dtype=np.float64))
    scale = np.ones(N_DIM) if dim_scale is None else np.asarray(dim_scale, dtype=np.float64)
    out = np.empty((max(len(A), len(B)), N_SUB))
    for j, (_, sp) in enumerate(DIST_SUBPARTS):
        out[:, j] = _dist(A[:, sp.sl], B[:, sp.sl], sp, scale)
    return out


def block_distances(sub_d: np.ndarray, medians: np.ndarray | None = None) -> np.ndarray:
    """
    Combine sub-part distances into block distances, D_b = sum_j D_bj / median_j.

    Parameters
    ----------
    sub_d : np.ndarray, shape (K, N_SUB)
    medians : np.ndarray, shape (N_SUB,), optional
        Null medians (default ones).

    Returns
    -------
    np.ndarray, shape (K, 15)
    """
    d = sub_d if medians is None else sub_d / np.asarray(medians)
    return np.add.reduceat(d, BLOCK_START, axis=1)


def all_block_distances(x_a: np.ndarray, X_b: np.ndarray, medians: np.ndarray | None = None,
                        dim_scale: np.ndarray | None = None) -> np.ndarray:
    """(K, 15) block distances between x_a and each row of X_b."""
    return block_distances(subpart_distances(x_a, X_b, dim_scale), medians)


def block_distance(spec, x_a: np.ndarray, X_b: np.ndarray, medians: np.ndarray | None = None,
                   dim_scale: np.ndarray | None = None) -> np.ndarray:
    """(K,) distance of one BlockSpec, D_b = sum_j D_bj / median_j."""
    return all_block_distances(x_a, X_b, medians, dim_scale)[:, BLOCKS.index(spec)]


# ==============================================================================
# END OF FILE: metrics.py
# ==============================================================================
# SUMMARY:
#   Per-sub-part distances (W1 / Hellinger / L2 / abs / abs-log / log-count L1) and their
#   null-median combination into 15 block distances; vectorised over K candidates.
#
# USAGE EXAMPLE:
#   D = all_block_distances(x_q, X_gallery)          # (K, 15), medians from null_model later
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Per-call O(K * 781) float64 copies; precompute squared / normalised gallery
#         histograms if latency at very large K matters.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
