"""
================================================================================
FILE: lbe_laplace_beltrami.py
================================================================================
PURPOSE:
    LBE, the "sound" of the surface (PDF Algorithm 7, also called ShapeDNA).
    If the surface were a drum skin, it would vibrate at certain natural
    frequencies. This file computes the K lowest ones. Two parts with the
    same frequencies have very similar overall shape.

INPUTS:
    - PreparedMesh (cleaned mesh, surface area), seeded rng (start vector
      for the solver).
    - params: K (30), min_vertices (coarser meshes are refined up to this).

OUTPUTS:
    - float32 vector, shape (K,): log(lambda_k A / (4 pi k)), k = 1..K,
      i.e. how far each frequency is from what a typical shape of the same
      area would give.

DEPENDENCIES:
    stdlib      : logging, math
    third-party : numpy, scipy.sparse, robust_laplacian, trimesh
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Builds a "stiffness" matrix L and a "mass" matrix M from the mesh
      (eq. 14) and solves L phi = lambda M phi (eq. 15). Multiplying lambda
      by the area A removes size.
    - robust_laplacian builds these matrices in a way that also works on
      imperfect meshes. Its mass matrix gives each corner 1/3 of each
      triangle's area, a slightly simpler choice than the PDF's; both give
      the same answer on fine meshes.
    - Each separate solid piece adds one zero frequency; those are dropped.
    - Coarse CAD meshes (a box has 8 corners) cannot show 30 frequencies,
      so their triangles are split into smaller ones until there are at
      least min_vertices corners. This does not change the shape.
    - Depends only on the surface itself, so moving, turning or mirroring
      the part changes nothing.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import logging
import math

import numpy as np
import robust_laplacian
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

from src.descriptors.base import Descriptor, DescriptorError

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SHIFT = -1e-8               # [-] solver looks for frequencies closest to this value (just below 0)
MAX_SUBDIVISIONS = 8        # [-] at most 8 rounds of splitting each triangle into 4
LAMBDA_FLOOR = 1e-12        # [-] smallest value allowed before taking the log

logger = logging.getLogger(__name__)


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class LBELaplaceBeltrami(Descriptor):
    """LBE: lowest natural frequencies of the surface, compared to a typical shape."""
    # L = stiffness matrix (from the mesh angles), M = mass matrix (area per corner)
    # solve L phi_k = lambda_k M phi_k for the lowest lambda_k, then multiply by area A
    # output = [log(lambda_k A / (4 pi k))] for k = 1..30
    # points used: all mesh corners (refined up to >= 1000, params.min_vertices)

    name = "lbe"
    version = "1"
    metric = "l2"
    defaults = {"K": 30, "min_vertices": 1000}

    def _setup(self) -> None:
        self.dim = int(self.params["K"])
        self.n_hist = 0

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        K = self.dim
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int64)
        for _ in range(MAX_SUBDIVISIONS):
            if len(V) >= int(self.params["min_vertices"]):
                break
            V, F = trimesh.remesh.subdivide(V, F)
        L, M = robust_laplacian.mesh_laplacian(V, F)
        beta0, _ = connected_components(L, directed=False)
        k = K + beta0
        if len(V) <= k + 1:
            raise DescriptorError(f"lbe: {len(V)} vertices cannot give {K} eigenvalues")
        v0 = rng.standard_normal(len(V))
        # lowest frequencies of L phi = lambda M phi (eq. 15)
        lam = eigsh(L, k=k, M=M, sigma=SHIFT, which="LM", v0=v0,
                    return_eigenvectors=False)
        lam = np.sort(lam)[beta0:]
        if np.any(lam <= LAMBDA_FLOOR):
            logger.warning("lbe: %d numerically zero modes past beta_0=%d",
                           int(np.sum(lam <= LAMBDA_FLOOR)), beta0)
        ks = np.arange(1, K + 1)
        # compare to a typical shape of the same area, on a log scale (eq. 16)
        return np.log(np.maximum(lam, LAMBDA_FLOOR) * mesh.area / (4.0 * math.pi * ks))

# ==============================================================================
# END OF FILE: lbe_laplace_beltrami.py
# ==============================================================================
# SUMMARY:
#   LBE: 30 lowest natural frequencies of the surface (log, size-free).
#
# USAGE EXAMPLE:
#   from src.descriptors.lbe_laplace_beltrami import LBELaplaceBeltrami
#   v = LBELaplaceBeltrami().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] No limit on mesh size: time grows faster than the corner count
#         (~V^1.3); very fine meshes may hit the per-file timeout.
#   - [ ] Cannot tell a bump from a matching dent (boss vs pocket).
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
