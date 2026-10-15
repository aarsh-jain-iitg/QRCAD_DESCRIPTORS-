"""
QRCAD  src/utils.py
Shared utility functions.
"""

import os
import sys
import numpy as np


def get_project_root() -> str:
    """
    Returns the project root directory.
    - Frozen exe: sys._MEIPASS (where PyInstaller extracts files)
    - Dev mode:   two levels up from src/
    """
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector. Returns zero vector if norm is near zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically stable cosine similarity between two vectors."""
    a_norm = normalize_vector(a)
    b_norm = normalize_vector(b)
    return float(np.dot(a_norm, b_norm))


def collect_stl_files(folder: str) -> list:
    """
    Recursively walk folder and return a list of absolute paths
    to every .stl file found.
    """
    stl_paths = []
    for root, _dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".stl"):
                stl_paths.append(os.path.join(root, fname))
    return stl_paths


def format_similarity(score: float) -> str:
    """Format a similarity score as a percentage string."""
    return f"{score * 100:.2f}%"
