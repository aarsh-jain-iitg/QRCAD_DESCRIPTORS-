"""
================================================================================
FILE: numpy_store.py
================================================================================
PURPOSE:
    Pure-NumPy brute-force VectorStore: exact search that works when FAISS
    is not installed, and the reference the FAISS backends are tested against.

INPUTS:
    - float32 vectors (N, dim); queries (Q, dim).

OUTPUTS:
    - (scores, ids) top-k arrays; .npy file on save().

DEPENDENCIES:
    stdlib      : os, pathlib
    third-party : numpy
    internal    : database (VectorStore)

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.database import VectorStore


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class NumpyStore(VectorStore):
    """Exact brute-force search in float32."""

    def __init__(self, dim: int, metric: str = "cosine") -> None:
        super().__init__(dim, metric)
        self.X = np.zeros((0, self.dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        v = np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1, self.dim)
        self.X = np.vstack([self.X, v])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1, self.dim)
        if self.metric == "l2":
            # -|q - x|^2 = 2 q.x - |x|^2 - |q|^2
            S = 2.0 * q @ self.X.T - (self.X ** 2).sum(1)[None, :] - (q ** 2).sum(1)[:, None]
        else:
            S = q @ self.X.T
        k_eff = min(k, len(self.X))
        ids = np.full((len(q), k), -1, dtype=np.int64)
        scores = np.full((len(q), k), -np.inf, dtype=np.float32)
        if k_eff:
            top = np.argsort(-S, axis=1, kind="stable")[:, :k_eff]
            ids[:, :k_eff] = top
            scores[:, :k_eff] = np.take_along_axis(S, top, axis=1)
        return scores, ids

    def save(self, path: Path) -> None:
        tmp = Path(path).with_suffix(".tmp.npy")
        np.save(tmp, self.X)
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        self.X = np.load(Path(path)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

# ==============================================================================
# END OF FILE: numpy_store.py
# ==============================================================================
# SUMMARY:
#   Brute-force numpy VectorStore (exact, FAISS-free fallback).
#
# USAGE EXAMPLE:
#   from src.database.numpy_store import NumpyStore
#   st = NumpyStore(64); st.add(X); st.search(q, 5)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] O(N) per query; fine up to ~1e5 vectors.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
