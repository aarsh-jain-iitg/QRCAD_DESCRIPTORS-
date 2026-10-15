"""
QRCAD v2.0 — src/retrieval_engine.py
Abstract retrieval engine interface and classical implementation.

This abstraction allows future engines (learning-based, etc.) to be added
without restructuring the app. Currently only ClassicalEngine is implemented,
wrapping the existing PCA → MDOR → HOG+D1/D2+Curvature → FAISS pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


class RetrievalEngine(ABC):
    """
    Abstract base class for retrieval engines.
    All engines must implement search() and return results in standard format.
    """

    @abstractmethod
    def search(
        self,
        query_features: np.ndarray,
        database: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the database using query features.

        Args:
            query_features: Feature vector for the query (np.ndarray)
            database: List of database entries, each a dict with:
                - "filename": str
                - "feature_vector": np.ndarray
                - "metadata": dict (optional)
            top_k: Number of top results to return

        Returns:
            List of dicts with keys:
            - "filename": str
            - "similarity": float (0-1 range)
            - "metadata": dict
            Sorted by similarity (descending)
        """
        pass


class ClassicalEngine(RetrievalEngine):
    """
    Classical retrieval engine using the existing PCA-aligned, HOG-based pipeline.
    Wraps the core search logic without modification.
    """

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Numerically stable cosine similarity."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def search(
        self,
        query_features: np.ndarray,
        database: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the database using cosine similarity on feature vectors.
        No modifications to the core pipeline — just an interface wrapper.
        """
        if not database:
            return []

        # Score all entries
        scores = []
        for entry in database:
            # Handle both old format (tuple) and new format (dict)
            if isinstance(entry, dict):
                filename = entry["filename"]
                feature_vec = entry["feature_vector"]
                metadata = entry.get("metadata", {})
            else:
                # Old format: (filename, feature_vector)
                filename = entry[0]
                feature_vec = entry[1]
                metadata = {"filename": filename}

            similarity = self._cosine_similarity(query_features, feature_vec)
            scores.append({
                "filename": filename,
                "similarity": similarity,
                "metadata": metadata
            })

        # Sort by similarity (descending) and return top-k
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_k]


def get_engine(engine_type: str = "classical") -> RetrievalEngine:
    """
    Factory function to get a retrieval engine instance.
    Currently only "classical" is supported.

    Args:
        engine_type: Name of the engine ("classical")

    Returns:
        Instance of the requested engine

    Raises:
        ValueError: If engine_type is not recognized
    """
    if engine_type.lower() == "classical":
        return ClassicalEngine()
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")
