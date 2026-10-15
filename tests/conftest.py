"""
================================================================================
FILE: conftest.py
================================================================================
PURPOSE:
    Shared pytest fixtures and helpers: repo import path, reduced-cost
    descriptor params for tests, test meshes (trimesh primitives plus files
    from a generated sample dataset), random similarity transforms, and a
    Config factory pointing at temporary folders.

INPUTS:
    - None (everything is generated).

OUTPUTS:
    - Fixtures: sample_dir, make_cfg; helpers: FAST_PARAMS, random_similarity,
      prepare_from_mesh, all_descriptors.

DEPENDENCIES:
    stdlib      : pathlib, sys, tomllib
    third-party : numpy, pytest, scipy, trimesh
    internal    : src.config, src.descriptors, src.preprocessing, src.sample_dataset

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import copy
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_CONFIG, config_from_dict  # noqa: E402
from src.descriptors import REGISTRY, make_descriptors  # noqa: E402
from src.preprocessing import prepare_mesh  # noqa: E402
from src.sample_dataset import generate_sample_dataset  # noqa: E402

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
N_SAMPLES = 10_000          # [-] test pool size (PDF default is 2e4)
FAST_PARAMS = {"d2": {"pairs": 60_000}, "triangle": {"triples": 60_000},
               "d4": {"quadruples": 60_000}, "ncf": {"pairs": 60_000},
               "crs_a": {"n_pool": 60_000}}


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def all_descriptors(device: str = "cpu"):
    """All 15 descriptors with the reduced-cost test params."""
    return make_descriptors(list(REGISTRY), FAST_PARAMS, device)


def prepare_from_mesh(mesh: trimesh.Trimesh, tmp: Path, name: str, seed: int = 42):
    """Export a mesh to STL (the real input path) and prepare it."""
    p = tmp / f"{name}.stl"
    mesh.export(p)
    return prepare_mesh(p, seed, N_SAMPLES)


def random_similarity(rng: np.random.Generator, rotate=True, translate=True,
                      scale=True, reflect=False) -> np.ndarray:
    """
    4x4 similarity transform: random SO(3), shift, scale in [0.01, 100], optional mirror.

    The shift is at most 10 part sizes (test parts are ~unit size, and the
    shift is applied after scaling). STL stores float32, so a shift t
    quantises vertices by ~2^-24 |t|; at |t| = 500 that tilts the normals
    of 0.05-wide facets by ~6e-4, which is input precision, not a
    descriptor property.
    """
    T = np.eye(4)
    R = Rotation.random(random_state=rng).as_matrix() if rotate else np.eye(3)
    if reflect:
        R = R @ np.diag([-1.0, 1.0, 1.0])
    s = 10.0 ** rng.uniform(-2, 2) if scale else 1.0
    T[:3, :3] = s * R
    if translate:
        T[:3, 3] = rng.uniform(-10, 10, 3) * s
    return T


# ── FIXTURES ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory) -> Path:
    """A generated sample dataset (2 parts per class) in a temp folder."""
    d = tmp_path_factory.mktemp("dataset") / "_sample"
    generate_sample_dataset(d, per_class=2)
    return d


@pytest.fixture
def make_cfg(tmp_path):
    """Factory: Config over a given dataset root with fast params in tmp folders."""
    base = tomllib.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def _make(dataset_root: Path, **pipeline):
        data = copy.deepcopy(base)
        data["paths"].update(dataset_root=str(dataset_root), cache_dir=str(tmp_path / "cache"),
                             output_dir=str(tmp_path / "out"))
        data["pipeline"].update({"n_samples": N_SAMPLES, "workers": 1, **pipeline})
        for name, params in FAST_PARAMS.items():
            data["descriptors"].setdefault(name, {}).update(params)
        return config_from_dict(data, ROOT_DIR)
    return _make

# ==============================================================================
# END OF FILE: conftest.py
# ==============================================================================
# SUMMARY:
#   Shared fixtures: fast params, sample dataset, similarity transforms, Config factory.
#
# USAGE EXAMPLE:
#   $ python -m pytest tests -q
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] None.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
