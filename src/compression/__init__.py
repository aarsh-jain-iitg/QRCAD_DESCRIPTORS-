"""
================================================================================
FILE: __init__.py  (src/compression)
================================================================================
PURPOSE:
    Vector compressors applied to fused database vectors before indexing,
    behind one Compressor interface: identity ("none") and PCA fitted on the
    database and persisted with the cache. A new method
    only needs a subclass and a branch in make_compressor().

INPUTS:
    - Fused database matrix (N, d) for fit(); vectors (Q, d) for transform().
    - method name and target_dim from [compression].

OUTPUTS:
    - Compressed, re-L2-normalised float32 vectors; .npz state on save().

DEPENDENCIES:
    stdlib      : abc, logging, os, pathlib
    third-party : numpy
    internal    : (none)

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def l2_rows(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (zero rows stay zero), float32."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.where(n > 1e-12, n, 1.0)).astype(np.float32)


# ── CORE CLASSES / MAIN FUNCTIONS ─────────────────────────────────────────────
class Compressor(ABC):
    """Fit on the database, then map any vector to the compressed space."""

    name = "base"

    @abstractmethod
    def fit(self, X: np.ndarray) -> Compressor:
        """Learn parameters from the database matrix (N, d)."""

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map (Q, d) -> (Q, d_out), rows L2-normalised."""

    @property
    @abstractmethod
    def out_dim(self) -> int:
        ...

    def save(self, path: Path) -> None:
        """Persist state (default: nothing to save beyond the name)."""
        self._write(path, {})

    def load(self, path: Path) -> None:
        """Restore state written by save()."""

    def _write(self, path: Path, arrays: dict[str, np.ndarray]) -> None:
        path = Path(path)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, name=np.array(self.name), **arrays)
        os.replace(tmp, path)


class IdentityCompressor(Compressor):
    """No compression ("none")."""

    name = "none"

    def __init__(self) -> None:
        self._dim = 0

    def fit(self, X: np.ndarray) -> IdentityCompressor:
        self._dim = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return l2_rows(np.asarray(X, dtype=np.float64))

    @property
    def out_dim(self) -> int:
        return self._dim

    def save(self, path: Path) -> None:
        self._write(path, {"dim": np.array(self._dim)})

    def load(self, path: Path) -> None:
        self._dim = int(np.load(Path(path))["dim"])


class PCACompressor(Compressor):
    """PCA to target_dim (clamped to rank), fitted by SVD on the centred database."""

    name = "pca"

    def __init__(self, target_dim: int) -> None:
        self.target_dim = int(target_dim)
        self.mean = np.zeros(0)
        self.W = np.zeros((0, 0))

    def fit(self, X: np.ndarray) -> PCACompressor:
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        _, s, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        k = min(self.target_dim, int(np.sum(s > 1e-10 * max(s.max(initial=0.0), 1e-300))) or 1)
        if k < self.target_dim:
            logger.warning("pca: target_dim %d clamped to rank %d", self.target_dim, k)
        self.W = Vt[:k].T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return l2_rows((np.asarray(X, dtype=np.float64) - self.mean) @ self.W)

    @property
    def out_dim(self) -> int:
        return self.W.shape[1]

    def save(self, path: Path) -> None:
        self._write(path, {"mean": self.mean, "W": self.W})

    def load(self, path: Path) -> None:
        z = np.load(Path(path))
        self.mean, self.W = z["mean"], z["W"]


def make_compressor(method: str, target_dim: int) -> Compressor:
    """Factory for [compression].method."""
    if method == "none":
        return IdentityCompressor()
    if method == "pca":
        return PCACompressor(target_dim)
    raise ValueError(f"unknown compression method {method!r}")

# ==============================================================================
# END OF FILE: __init__.py  (src/compression)
# ==============================================================================
# SUMMARY:
#   Compressor ABC + identity and PCA implementations.
#
# USAGE EXAMPLE:
#   from src.compression import make_compressor
#   c = make_compressor("pca", 128).fit(X_db); Z = c.transform(X_query)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] FAISS PQ not implemented (config rejects "pq"); add a subclass
#         wrapping faiss.ProductQuantizer if index memory ever matters.
#   - [ ] PCA is global on the fused vector, not per block.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
