"""
================================================================================
FILE: augment.py
================================================================================
PURPOSE:
    Same-design augmenter T(p): produces a re-tessellated, rotated, slightly noisy,
    optionally decimated and slightly anisotropically scaled copy of a part. A
    part and its augmentations are "the same design"; their block distances give
    the null model its same-design distribution (Phase 3).

INPUTS:
    - trimesh.Trimesh (welded, normalised), np.random.Generator, BLRConfig.

OUTPUTS:
    - augment(mesh, rng, cfg) -> trimesh.Trimesh (input untouched).

DEPENDENCIES:
    stdlib      : logging
    third-party : numpy, scipy, trimesh, fast_simplification (optional, decimation)
    internal    : blr.config, blr.extract

PHYSICS / DOMAIN CONTEXT:
    Order (DOCS/BLR_PLAN.md Phase 2): remesh at edge x U(0.5, 2), uniform random SO(3)
    rotation, vertex noise N(0, s) with s ~ U(0, 0.01) D, with probability 0.5 quadric
    decimation to a U(0.25, 1) face fraction, anisotropic scale diag(U(0.95, 1.05)^3).
    Never mirrors. The remesh is a uniform 1:4 subdivision (it can refine, not coarsen;
    decimation supplies the coarsening); pymeshlab is not installed.
    D = bounding-box diagonal of the input.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from src.blr.config import BLRConfig
from src.blr.extract import precondition_mesh

logger = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
AUG_MAX_FACES = 40_000            # [-] face budget of the remesh step (extraction time)
DECIM_PROBABILITY = 0.5           # [-] chance that a part is decimated
MIN_FACES = 100                   # [-] never decimate below this


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def augment(mesh: trimesh.Trimesh, rng: np.random.Generator, cfg: BLRConfig) -> trimesh.Trimesh:
    """
    One same-design augmentation of a welded, normalised mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    rng : np.random.Generator
        Every random choice comes from this generator.
    cfg : BLRConfig

    Returns
    -------
    trimesh.Trimesh
        New mesh; not re-normalised (all blocks are scale / shift invariant).
    """
    D = float(np.linalg.norm(mesh.extents))
    edge = cfg.remesh_edge_frac * rng.uniform(*cfg.aug_edge_range)           # 1. remesh
    m = precondition_mesh(mesh, edge, AUG_MAX_FACES)
    v = np.asarray(m.vertices) @ Rotation.random(rng=rng).as_matrix().T      # 2. rotation
    v = v + rng.normal(0.0, rng.uniform(0.0, cfg.aug_noise_max) * D, v.shape)  # 3. noise
    out = trimesh.Trimesh(v, m.faces, process=False)
    if rng.random() < DECIM_PROBABILITY:                                      # 4. decimation
        target = max(MIN_FACES, int(len(out.faces) * rng.uniform(*cfg.aug_decim_range)))
        if target < len(out.faces):
            try:
                out = out.simplify_quadric_decimation(face_count=target)
            except Exception:                    # missing fast_simplification etc.: keep the mesh
                logger.warning("decimation skipped", exc_info=True)
    out.vertices = np.asarray(out.vertices) * rng.uniform(*cfg.aug_aniso_range, size=3)   # 5. scale
    return out


# ==============================================================================
# END OF FILE: augment.py
# ==============================================================================
# SUMMARY:
#   Same-design augmenter (remesh, rotate, noise, decimate, anisotropic scale; no mirror).
#
# USAGE EXAMPLE:
#   m2 = augment(load_normalized(path, "rms_radius"), np.random.default_rng(1), BLRConfig())
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Remesh only refines (uniform 1:4); an isotropic remesh needs pymeshlab.
#   - [ ] Step 6 (new extraction seed) is done by the caller (gallery.py).
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
