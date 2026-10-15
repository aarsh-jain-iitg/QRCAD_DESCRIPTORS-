"""
================================================================================
FILE: baselines.py
================================================================================
PURPOSE:
    The label-free scoring ladder below the learned residual: L0 mean block cosine
    (the existing behaviour), L1 mean of (1 - u_b), L1r reciprocal rank fusion,
    L2 generalised mean M_p(1 - u_b), L3 = prior (prior.prior_scores).
    Larger score = more similar for every level.

INPUTS:
    - Gallery X (N, 781); per-candidate percentiles U (K, 15) and block distances D (K, 15).

OUTPUTS:
    - BlockCosine (L0), l1_scores, l1r_scores, gen_mean_scores, rank_matrix.

DEPENDENCIES:
    stdlib      : none
    third-party : numpy
    internal    : blr.blocks, blr.config

PHYSICS / DOMAIN CONTEXT:
    - L0 block vector: histogram dims as stored (Hellinger, inner product = Bhattacharyya),
      scalar dims z-scored with gallery statistics, unit length; cosine per block, mean over blocks.
    - M_p(v) = (mean v^p)^(1/p), M_0 = geometric mean; p < 0 punishes a single bad block.
    - RRF: sum_b 1 / (k + rank_b) with rank 1 = closest (Cormack 2009, k = 60).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import numpy as np

from src.blr.blocks import BLOCKS
from src.blr.config import BLRConfig

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
STD_FLOOR = 1e-9                  # [-] std floor for z-scored scalar dims
GEN_MEAN_P = (1.0, 0.0, -1.0, -4.0)   # [-] candidate exponents for L2


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def rank_matrix(D: np.ndarray) -> np.ndarray:
    """(K, 15) rank of every candidate under each block, 1 = smallest distance."""
    return np.argsort(np.argsort(D, axis=0, kind="stable"), axis=0, kind="stable") + 1


def l1_scores(U: np.ndarray) -> np.ndarray:
    """L1: mean over blocks of (1 - u_b)."""
    return (1.0 - U).mean(axis=1)


def l1r_scores(D: np.ndarray, k: int | None = None) -> np.ndarray:
    """L1r: reciprocal rank fusion of the 15 block rankings."""
    k = BLRConfig().rrf_k if k is None else k
    return (1.0 / (k + rank_matrix(D))).sum(axis=1)


def gen_mean_scores(U: np.ndarray, p: float) -> np.ndarray:
    """L2: generalised mean of (1 - u_b) with exponent p (p = 0: geometric mean)."""
    v = np.clip(1.0 - U, 1e-12, 1.0)
    return np.exp(np.log(v).mean(axis=1)) if p == 0 else np.mean(v ** p, axis=1) ** (1.0 / p)


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class BlockCosine:
    """L0: mean over the 15 blocks of the cosine of the block-transformed vectors."""

    def __init__(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        self.mean, self.std = np.zeros(X.shape[1]), np.ones(X.shape[1])
        self.scalar = np.zeros(X.shape[1], dtype=bool)
        for b in BLOCKS:
            for sp in b.subparts:
                if not sp.stored_sqrt:
                    self.scalar[sp.sl] = True
        self.mean[self.scalar] = X[:, self.scalar].mean(axis=0)
        sd = X[:, self.scalar].std(axis=0)
        self.std[self.scalar] = np.where(sd > STD_FLOOR, sd, 1.0)
        self.Z = self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """(K, 781) block-wise z-scored (scalars) and unit-normalised rows."""
        Z = (np.atleast_2d(np.asarray(X, np.float64)) - self.mean) / self.std
        for b in BLOCKS:
            n = np.linalg.norm(Z[:, b.sl], axis=1, keepdims=True)
            Z[:, b.sl] /= np.maximum(n, 1e-12)
        return Z

    def scores(self, x_q: np.ndarray, Z: np.ndarray | None = None) -> np.ndarray:
        """(N,) mean block cosine of the query against the gallery (or against Z)."""
        Z = self.Z if Z is None else Z
        zq = self.transform(x_q)[0]
        return np.mean([Z[:, b.sl] @ zq[b.sl] for b in BLOCKS], axis=0)


# ==============================================================================
# END OF FILE: baselines.py
# ==============================================================================
# SUMMARY:
#   L0 block cosine, L1 mean percentile, L1r RRF, L2 generalised mean.
#
# USAGE EXAMPLE:
#   s0 = BlockCosine(X).scores(xq); s1 = l1_scores(U); s2 = gen_mean_scores(U, -1.0)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] BlockCosine statistics are fitted on the whole gallery (transductive, as in main.py --eval).
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
