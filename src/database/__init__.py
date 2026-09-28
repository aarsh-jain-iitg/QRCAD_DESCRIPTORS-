"""
================================================================================
FILE: __init__.py  (src/database)
================================================================================
PURPOSE:
    Vector store interface and backend factory. Every backend stores float32
    row vectors and answers top-k queries with a similarity score (higher
    is better), so the orchestrator can swap backends from config alone.

INPUTS:
    - backend name ("faiss_flat" | "faiss_hnsw" | "numpy"), metric
      ("cosine" | "l2"), vector dimension.

OUTPUTS:
    - VectorStore instances via make_store().

DEPENDENCIES:
    stdlib      : abc, logging, pathlib
    third-party : numpy
    internal    : database.numpy_store, database.faiss_store (lazy)

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── CORE CLASSES / MAIN FUNCTIONS ─────────────────────────────────────────────
class VectorStore(ABC):
    """
    Minimal vector index.

    Scores returned by search() are similarities: the inner product for
    "cosine" (vectors are expected unit length), and -squared distance
    for "l2". Ranking is the same for unit vectors.
    """

    def __init__(self, dim: int, metric: str = "cosine") -> None:
        self.dim = int(dim)
        self.metric = metric

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Append rows, shape (N, dim)."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Top-k search.

        Returns
        -------
        scores : np.ndarray, shape (Q, k)
        ids : np.ndarray, shape (Q, k), int64 row ids (-1 for padding)
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist to a file (written atomically by the caller's convention)."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load from a file written by save()."""

    @abstractmethod
    def __len__(self) -> int:
        ...


def make_store(backend: str, dim: int, metric: str = "cosine") -> VectorStore:
    """
    Construct a backend by name; falls back to numpy if FAISS is missing.

    Parameters
    ----------
    backend : str
        "faiss_flat" | "faiss_hnsw" | "numpy".
    dim : int
        Vector length.
    metric : str
        "cosine" | "l2".
    """
    from src.database.numpy_store import NumpyStore
    if backend == "numpy":
        return NumpyStore(dim, metric)
    try:
        from src.database.faiss_store import FaissFlatStore, FaissHNSWStore
    except ImportError:
        logger.warning("faiss not installed; using the numpy backend instead of %s", backend)
        return NumpyStore(dim, metric)
    if backend == "faiss_flat":
        return FaissFlatStore(dim, metric)
    if backend == "faiss_hnsw":
        return FaissHNSWStore(dim, metric)
    raise ValueError(f"unknown backend {backend!r}")

# ==============================================================================
# END OF FILE: __init__.py  (src/database)
# ==============================================================================
# SUMMARY:
#   VectorStore ABC + make_store() factory (faiss_flat / faiss_hnsw / numpy).
#
# USAGE EXAMPLE:
#   from src.database import make_store
#   st = make_store("faiss_flat", 781); st.add(X); scores, ids = st.search(q, 10)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] faiss-gpu has no Windows wheels; GPU indexing is not wired in.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
