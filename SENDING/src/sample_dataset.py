"""
================================================================================
FILE: sample_dataset.py
================================================================================
PURPOSE:
    Generate a small, deterministic labelled STL dataset from trimesh
    primitives (dataset/_sample/<class>/*.stl) so the pipeline can be run
    and evaluated end to end before real CAD data is added. Every part gets
    random proportions, tessellation, rotation, translation and scale.

INPUTS:
    - out_dir : Path of the dataset folder to create (e.g. dataset/_sample).
    - per_class, seed.

OUTPUTS:
    - STL files, one folder per class; returns the number of files written.

DEPENDENCIES:
    stdlib      : logging, pathlib
    third-party : numpy, scipy.spatial.transform, trimesh
    internal    : (none)

PHYSICS / DOMAIN CONTEXT:
    - Units are arbitrary (scale is randomised over 3 decades on purpose).
    - "bracket" parts are two overlapping boxes concatenated without a
      boolean union: self-intersecting, not watertight. They exercise the
      robustness paths (T1, MTS).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CLASSES = ("block", "pin", "ball", "washer", "torus", "capsule", "bracket")

logger = logging.getLogger(__name__)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _make(cls: str, rng: np.random.Generator) -> trimesh.Trimesh:
    """One random part of a class, in its own frame."""
    u = rng.uniform
    if cls == "block":
        return trimesh.creation.box(extents=[u(0.8, 1.2), u(0.5, 0.8), u(0.2, 0.4)])
    if cls == "pin":
        return trimesh.creation.cylinder(radius=u(0.08, 0.15), height=u(1.5, 2.5),
                                         sections=int(rng.integers(48, 129)))
    if cls == "ball":
        m = trimesh.creation.icosphere(subdivisions=int(rng.integers(3, 5)))
        m.apply_scale([u(0.9, 1.1), u(0.9, 1.1), 1.0])
        return m
    if cls == "washer":
        r = u(0.8, 1.2)
        return trimesh.creation.annulus(r_min=r * u(0.4, 0.6), r_max=r, height=u(0.08, 0.2),
                                        sections=int(rng.integers(64, 129)))
    if cls == "torus":
        return trimesh.creation.torus(major_radius=1.0, minor_radius=u(0.2, 0.35),
                                      major_sections=int(rng.integers(48, 97)),
                                      minor_sections=int(rng.integers(24, 49)))
    if cls == "capsule":
        return trimesh.creation.capsule(height=u(1.0, 2.0), radius=u(0.3, 0.5),
                                        count=[int(rng.integers(24, 49))] * 2)
    if cls == "bracket":
        a = trimesh.creation.box(extents=[u(1.0, 1.4), 0.2, u(0.5, 0.7)])
        b = trimesh.creation.box(extents=[0.2, u(0.8, 1.2), u(0.5, 0.7)])
        b.apply_translation([a.extents[0] / 2 - 0.1, b.extents[1] / 2 - 0.1, 0.0])
        return trimesh.util.concatenate([a, b])
    raise ValueError(f"unknown class {cls!r}")


def _random_pose(m: trimesh.Trimesh, rng: np.random.Generator) -> trimesh.Trimesh:
    """Apply a random rotation, translation and uniform scale (10^-1 .. 10^2)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=rng).as_matrix() * 10.0 ** rng.uniform(-1, 2)
    T[:3, 3] = rng.uniform(-100, 100, 3)
    m.apply_transform(T)
    return m


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def generate_sample_dataset(out_dir: Path, per_class: int = 6, seed: int = 7) -> int:
    """
    Write per_class random parts for every class in CLASSES.

    Parameters
    ----------
    out_dir : Path
        Target dataset folder (created).
    per_class : int
        Parts per class.
    seed : int
        Generator seed (the dataset is reproducible).

    Returns
    -------
    int
        Number of STL files written.
    """
    rng = np.random.default_rng(seed)
    n = 0
    for cls in CLASSES:
        d = Path(out_dir) / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            _random_pose(_make(cls, rng), rng).export(d / f"{cls}_{i:02d}.stl")
            n += 1
    logger.info("sample dataset: wrote %d STL files to %s", n, out_dir)
    return n

# ==============================================================================
# END OF FILE: sample_dataset.py
# ==============================================================================
# SUMMARY:
#   Deterministic 7-class synthetic STL dataset from trimesh primitives.
#
# USAGE EXAMPLE:
#   from src.sample_dataset import generate_sample_dataset
#   generate_sample_dataset(Path("dataset/_sample"))
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Synthetic classes are far easier than real CAD classes; use it for
#         smoke tests, not to judge descriptors.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
