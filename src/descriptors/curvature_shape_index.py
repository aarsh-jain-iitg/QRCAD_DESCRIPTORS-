"""
================================================================================
FILE: curvature_shape_index.py
================================================================================
PURPOSE:
    Curvature descriptor (shape index + curvedness),
    CAD_Descriptors_Explained.pdf, Algorithm 9: principal curvatures from a
    local quadric fit at fixed physical scale around each surface sample,
    summarised as a 12 x 8 area-weighted (SI, log cCV) histogram over the
    smooth curved samples plus the flat share of the smooth area.

INPUTS:
    - PreparedMesh (default sample pool with normals, D_diam, area).
    - params: si_bins (12), cv_bins (8), radius_frac (r = 0.02 D_diam),
      k_max (neighbour cap, 30), tau_deg (crease, 20), flat_eps (0.1),
      cv_max (100), min_neighbors (6).

OUTPUTS:
    - float32 vector, shape (si_bins*cv_bins + 1,): [Hellinger hist | flat share].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, scipy.spatial.cKDTree
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Local frame (e_x, e_y, n_i); fit z = a x^2 + b xy + c y^2 + d x + e y.
      With z along the OUTWARD normal a convex surface curves toward -z, so
      kappa_{1,2} = eigenvalues of -[[2a, b], [b, 2c]], giving the PDF sign
      convention (kappa > 0 convex, sphere SI = +1).
    - SI = (2/pi) atan2(k1+k2, k1-k2), CV = sqrt((k1^2+k2^2)/2),
      cCV = CV sqrt(A/(4 pi)) (a ball has cCV = 1).
    - Neighbourhood: radius r = 0.02 D_diam capped at k_max
      nearest samples (Complexity PDF bound on cost). The two PDFs differ
      here (radius vs k-NN); the cap keeps the radius definition while
      bounding the cost.
    - Samples whose neighbourhood holds a normal > tau from n_i are creases
      (DAD's job) and are left out. Samples with < min_neighbors neighbours
      cannot support a 5-parameter fit and are also left out.
    - flat_eps and the log cCV range are not given in the PDF: flat means
      cCV < 0.1 (10x flatter than a ball of the same area); curved bins are
      log-spaced on [0.1, 100], values outside clipped to the end bins.
    - Coarse tessellations whose facets are wider than r read as flat.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from src.descriptors.base import Descriptor, hellinger


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _tangent_frames(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal tangents (e_x, e_y) for unit normals n (N,3)."""
    # points: all n = 2e4 surface samples (pipeline.n_samples), each fitted on <= 30 neighbours (params.k_max)
    helper = np.where(np.abs(n[:, [0]]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    ex = np.cross(n, helper)
    ex /= np.linalg.norm(ex, axis=1, keepdims=True)
    return ex, np.cross(n, ex)


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class CurvatureShapeIndex(Descriptor):
    """Curvature: joint (shape index, curvedness) histogram."""
    # fit z = a x^2 + b xy + c y^2 + d x + e y within r = 0.02 D_diam
    # k1 >= k2 = eigenvalues of [[2a, b], [b, 2c]]
    # SI = (2/pi) atan2(k1 + k2, k1 - k2) in [-1, 1]
    # CV = sqrt((k1^2 + k2^2) / 2),  cCV = CV sqrt(A / (4 pi))

    name = "curvature"
    version = "1"
    metric = "hellinger"
    defaults = {"si_bins": 12, "cv_bins": 8, "radius_frac": 0.02, "k_max": 30,
                "tau_deg": 20.0, "flat_eps": 0.1, "cv_max": 100.0, "min_neighbors": 6}

    def _setup(self) -> None:
        self.n_hist = int(self.params["si_bins"]) * int(self.params["cv_bins"])
        self.dim = self.n_hist + 1

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        p = self.params
        s = mesh.samples
        P, N = s.points, s.normals
        n = len(P)
        r = float(p["radius_frac"]) * mesh.diameter
        dist, idx = cKDTree(P).query(P, k=int(p["k_max"]) + 1, distance_upper_bound=r)
        idx, dist = idx[:, 1:], dist[:, 1:]                # drop self
        valid = idx < n
        safe = np.where(valid, idx, 0)
        # crease test: any neighbour normal more than tau away from n_i
        cosang = np.einsum("ikj,ij->ik", N[safe], N)
        crease = np.any(valid & (cosang < math.cos(math.radians(float(p["tau_deg"])))), axis=1)
        smooth = ~crease & (valid.sum(axis=1) >= int(p["min_neighbors"]))
        if not np.any(smooth):
            return np.zeros(self.dim)

        ex, ey = _tangent_frames(N)
        d = P[safe] - P[:, None, :]                          # (n, k, 3)
        x = np.einsum("ikj,ij->ik", d, ex)
        y = np.einsum("ikj,ij->ik", d, ey)
        z = np.einsum("ikj,ij->ik", d, N)
        # eq.(19): z = a x^2 + b xy + c y^2 + d x + e y (invalid rows zeroed)
        A = np.stack([x * x, x * y, y * y, x, y], axis=2) * valid[:, :, None]
        zt = z * valid
        AtA = np.einsum("ikp,ikq->ipq", A[smooth], A[smooth])
        Atz = np.einsum("ikp,ik->ip", A[smooth], zt[smooth])
        # tiny ridge keeps near-collinear neighbourhoods solvable
        ridge = (1e-12 * np.trace(AtA, axis1=1, axis2=2) + 1e-30)[:, None, None] * np.eye(5)
        coef = np.linalg.solve(AtA + ridge, Atz[:, :, None])[:, :, 0]
        a, b, c = coef[:, 0], coef[:, 1], coef[:, 2]
        # eq.(19): kappa_{1,2} = eig(-[[2a, b], [b, 2c]]), kappa_1 >= kappa_2
        mean = -(a + c)
        rad = np.sqrt((a - c) ** 2 + b ** 2)
        k1, k2 = mean + rad, mean - rad
        # eq.(20): SI, CV, normalised cCV
        si = (2.0 / math.pi) * np.arctan2(k1 + k2, k1 - k2)
        ccv = np.sqrt(0.5 * (k1 ** 2 + k2 ** 2)) * math.sqrt(mesh.area / (4.0 * math.pi))
        eps = float(p["flat_eps"])
        curved = ccv >= eps
        flat_share = 1.0 - float(curved.mean())
        H, _, _ = np.histogram2d(
            np.clip(si[curved], -1.0, 1.0),
            np.clip(np.log10(ccv[curved]), math.log10(eps), math.log10(float(p["cv_max"]))),
            bins=[int(p["si_bins"]), int(p["cv_bins"])],
            range=[[-1.0, 1.0], [math.log10(eps), math.log10(float(p["cv_max"]))]])
        return np.append(hellinger(H.ravel()), flat_share)

# ==============================================================================
# END OF FILE: curvature_shape_index.py
# ==============================================================================
# SUMMARY:
#   Curvature: 12x8 (SI, log cCV) histogram over smooth samples + flat share.
#
# USAGE EXAMPLE:
#   from src.descriptors.curvature_shape_index import CurvatureShapeIndex
#   v = CurvatureShapeIndex().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Facets wider than r = 0.02 D_diam read as flat (tessellation floor).
#   - [ ] Vectorised n x k x 5 design (~25 MB at defaults); chunk if n grows.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
