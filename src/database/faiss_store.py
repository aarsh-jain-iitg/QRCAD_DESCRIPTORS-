"""
================================================================================
FILE: faiss_store.py
================================================================================
PURPOSE:
    FAISS-backed VectorStores: exact flat index (IndexFlatIP / IndexFlatL2)
    and approximate HNSW (IndexHNSWFlat), with the score convention of the
    VectorStore interface (similarity, higher is better).

INPUTS:
    - float32 vectors (N, dim); queries (Q, dim).

OUTPUTS:
    - (scores, ids) top-k arrays; FAISS index file on save().

DEPENDENCIES:
    stdlib      : os, pathlib
    third-party : faiss-cpu, numpy
    internal    : database (VectorStore)

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import faiss
import numpy as np

from src.database import VectorStore

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
HNSW_M = 32                 # [-] graph links per node
HNSW_EF_SEARCH = 128        # [-] beam width at query time


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class _FaissStore(VectorStore):
    """Shared add / search / save / load around a faiss.Index."""

    def _new_index(self) -> faiss.Index:
        raise NotImplementedError

    def __init__(self, dim: int, metric: str = "cosine") -> None:
        super().__init__(dim, metric)
        self.index = self._new_index()

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1, self.dim))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1, self.dim)
        D, ids = self.index.search(q, k)
        scores = -D if self.metric == "l2" else D           # L2 index returns squared distances
        scores = np.where(ids < 0, -np.inf, scores).astype(np.float32)
        return scores, ids.astype(np.int64)

    def save(self, path: Path) -> None:
        tmp = Path(path).with_suffix(".tmp")
        faiss.write_index(self.index, str(tmp))
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(Path(path)))

    def __len__(self) -> int:
        return int(self.index.ntotal)


class FaissFlatStore(_FaissStore):
    """Exact search: IndexFlatIP (cosine on unit vectors) or IndexFlatL2."""

    def _new_index(self) -> faiss.Index:
        return faiss.IndexFlatL2(self.dim) if self.metric == "l2" else faiss.IndexFlatIP(self.dim)


class FaissHNSWStore(_FaissStore):
    """Approximate search: HNSW graph (M = 32, efSearch = 128)."""

    def _new_index(self) -> faiss.Index:
        m = faiss.METRIC_L2 if self.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        index = faiss.IndexHNSWFlat(self.dim, HNSW_M, m)
        index.hnsw.efSearch = HNSW_EF_SEARCH
        return index

# ==============================================================================
# END OF FILE: faiss_store.py
# ==============================================================================
# SUMMARY:
#   FAISS flat (exact) and HNSW (approximate) VectorStores.
#
# USAGE EXAMPLE:
#   from src.database.faiss_store import FaissFlatStore
#   st = FaissFlatStore(781); st.add(X); st.search(q, 10)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] HNSW params are constants; expose in config if databases grow past ~1e5.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
