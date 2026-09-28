"""
================================================================================
FILE: lbe_laplace_beltrami.py
================================================================================
PURPOSE:
    LBE Laplace-Beltrami eigenvalues ("ShapeDNA"),
    CAD_Descriptors_Explained.pdf, Algorithm 7: the K lowest non-zero
    eigenvalues of the cotangent Laplacian (generalised with the lumped mass
    matrix), made scale-free with the surface area and expressed as the log
    deviation from Weyl's law.

INPUTS:
    - PreparedMesh (welded mesh, area), seeded rng (Lanczos start vector).
    - params: K (30), min_vertices (subdivide coarser meshes up to this).

OUTPUTS:
    - float32 vector, shape (K,): log(lambda_k A / (4 pi k)), k = 1..K.

DEPENDENCIES:
    stdlib      : logging, math
    third-party : numpy, scipy.sparse, robust_laplacian, trimesh
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - L_ij = -1/2 (cot a_ij + cot b_ij), M lumped mass (eq. 14);
      L phi = lambda M phi (eq. 15); lambda^ = lambda A (scale-free).
    - robust_laplacian builds L on the intrinsic Delaunay triangulation (the
      edge-flip step the PDF asks for, so cotangent weights are >= 0) and
      handles non-manifold input via a tufted cover. Its lumped mass is
      barycentric (A_t/3 per corner) rather than the mixed-Voronoi cell of
      eq. (14); both converge to the same spectrum under refinement.
    - One zero eigenvalue per connected body is dropped (beta_0 = number of
      connected components of the vertex graph).
    - Coarse CAD meshes (a box has 8 vertices) cannot resolve 30 modes, so
      they are midpoint-subdivided until V >= min_vertices. Subdivision
      does not change the geometry, only the discretisation.
    - Intrinsic quantity: invariant to rotation, shift, reflection; lambda A
      removes scale.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
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
SHIFT = -1e-8               # [-] shift-invert sigma (PDF Algorithm 7)
MAX_SUBDIVISIONS = 8        # [-] cap on 4x midpoint refinements
LAMBDA_FLOOR = 1e-12        # [-] floor before the log for numerically zero modes

logger = logging.getLogger(__name__)


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class LBELaplaceBeltrami(Descriptor):
    """LBE: Laplace-Beltrami spectrum, Weyl-normalised."""
    # L_ij = -(cot a_ij + cot b_ij) / 2,  L_ii = -sum_{j != i} L_ij,  M = diag(A~_i)
    # L phi_k = lambda_k M phi_k,  0 = lambda_0 <= lambda_1 <= ...,  lambda^_k = lambda_k A
    # F_LBE = [log(lambda^_k / (4 pi k))]_{k=1..30}
    # points: all welded mesh vertices (subdivided up to >= 1000, params.min_vertices)

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
        # eq.(15): L phi = lambda M phi, shift-invert Lanczos near 0
        lam = eigsh(L, k=k, M=M, sigma=SHIFT, which="LM", v0=v0,
                    return_eigenvectors=False)
        lam = np.sort(lam)[beta0:]
        if np.any(lam <= LAMBDA_FLOOR):
            logger.warning("lbe: %d numerically zero modes past beta_0=%d",
                           int(np.sum(lam <= LAMBDA_FLOOR)), beta0)
        ks = np.arange(1, K + 1)
        # eq.(16): F = log(lambda_k A / (4 pi k))
        return np.log(np.maximum(lam, LAMBDA_FLOOR) * mesh.area / (4.0 * math.pi * ks))

# ==============================================================================
# END OF FILE: lbe_laplace_beltrami.py
# ==============================================================================
# SUMMARY:
#   LBE: 30 Weyl-normalised log Laplace-Beltrami eigenvalues.
#
# USAGE EXAMPLE:
#   from src.descriptors.lbe_laplace_beltrami import LBELaplaceBeltrami
#   v = LBELaplaceBeltrami().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] No decimation to a vertex budget: cost grows ~V^1.3 (Complexity
#         PDF); very fine meshes may hit the per-file timeout.
#   - [ ] Blind to mirror-undoable differences, e.g. boss vs pocket.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
