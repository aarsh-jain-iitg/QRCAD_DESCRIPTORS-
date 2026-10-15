"""
================================================================================
FILE: candidates.py
================================================================================
PURPOSE:
    Stage 1 of BLR: which gallery parts get re-ranked. At or below N_BRUTE_MAX
    parts every part is a candidate; above that a FAISS union of (a) the top-K of a
    beta-weighted fused index and (b) the top ceil(K / B) of every per-block index,
    so a part that only one block finds is not lost.

INPUTS:
    - Gallery X (N, 781), NullModel (beta_b), query vector (781,).

OUTPUTS:
    - brute(), Stage1 (build / query), recall_at_k(), choose_k().

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, faiss (faiss-cpu)
    internal    : blr.baselines, blr.blocks, blr.config, blr.null_model

PHYSICS / DOMAIN CONTEXT:
    - Fused index: every block unit-normalised (baselines.BlockCosine transform), scaled by
      sqrt(beta_b) with beta_b = Var_random / Var_augmentation, then concatenated;
      IndexFlatL2 on the result.
    - Self-matches are dropped by the caller via `exclude`.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import math

import faiss
import numpy as np

from src.blr.baselines import BlockCosine
from src.blr.blocks import BLOCKS
from src.blr.config import BLRConfig
from src.blr.null_model import NullModel

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
K_GRID = (25, 50, 100, 200)       # [-] candidate-list sizes reported in the recall table


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def brute(n: int, exclude: int | None = None) -> np.ndarray:
    """All gallery indices 0..n-1 except `exclude` (the query itself, if it is in the gallery)."""
    idx = np.arange(n)
    return idx if exclude is None else idx[idx != exclude]


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class Stage1:
    """FAISS candidate generator (beta-weighted fused index + per-block indexes)."""

    def __init__(self, X: np.ndarray, model: NullModel) -> None:
        self.cos = BlockCosine(X)
        self.w = np.ones(X.shape[1], dtype=np.float32)
        for b, spec in enumerate(BLOCKS):
            self.w[spec.sl] = math.sqrt(model.beta[b])
        Z = self.cos.Z.astype(np.float32)
        self.fused = faiss.IndexFlatL2(Z.shape[1])
        self.fused.add(np.ascontiguousarray(Z * self.w))
        self.per_block = []
        for spec in BLOCKS:
            ix = faiss.IndexFlatL2(spec.sl.stop - spec.sl.start)
            ix.add(np.ascontiguousarray(Z[:, spec.sl]))
            self.per_block.append(ix)

    def query(self, x_q: np.ndarray, k: int, exclude: int | None = None) -> np.ndarray:
        """
        Candidate indices for one query.

        Parameters
        ----------
        x_q : np.ndarray, shape (781,)
        k : int
            Fused top-k; every block adds its own top ceil(k / B).
        exclude : int, optional
            Gallery index to drop (the query itself).

        Returns
        -------
        np.ndarray of int, unique, ascending.
        """
        z = self.cos.transform(x_q).astype(np.float32)
        extra = 1 if exclude is not None else 0
        _, fi = self.fused.search(np.ascontiguousarray(z * self.w), k + extra)
        got = set(fi[0].tolist())
        per = math.ceil(k / len(BLOCKS)) + extra
        for spec, ix in zip(BLOCKS, self.per_block):
            _, bi = ix.search(np.ascontiguousarray(z[:, spec.sl]), per)
            got.update(bi[0].tolist())
        got.discard(-1)
        if exclude is not None:
            got.discard(exclude)
        return np.array(sorted(got), dtype=np.int64)


def recall_at_k(stage1: Stage1, Xq: np.ndarray, targets: list[set[int]], k: int,
                excludes: list[int | None] | None = None) -> float:
    """Mean over queries of |candidates ∩ targets| / |targets| for candidate size k."""
    r = []
    for i, x in enumerate(Xq):
        cand = set(stage1.query(x, k, None if excludes is None else excludes[i]).tolist())
        r.append(len(cand & targets[i]) / max(len(targets[i]), 1))
    return float(np.mean(r))


def choose_k(stage1: Stage1, Xq: np.ndarray, targets: list[set[int]], target: float | None = None,
             grid: tuple[int, ...] = K_GRID) -> int:
    """Smallest K in `grid` reaching the recall target (largest K if none does)."""
    target = BLRConfig().recall_target if target is None else target
    for k in grid:
        if recall_at_k(stage1, Xq, targets, k) >= target:
            return k
    return grid[-1]


# ==============================================================================
# END OF FILE: candidates.py
# ==============================================================================
# SUMMARY:
#   brute() candidate list and FAISS Stage1 union (fused beta-weighted + per-block).
#
# USAGE EXAMPLE:
#   s1 = Stage1(X, model); cand = s1.query(xq, 100, exclude=7)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] IndexFlatL2 only; switch to HNSW above ~1e5 parts (Phase 8).
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
