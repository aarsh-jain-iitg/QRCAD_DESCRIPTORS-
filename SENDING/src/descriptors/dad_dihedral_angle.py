"""
================================================================================
FILE: dad_dihedral_angle.py
================================================================================
PURPOSE:
    DAD dihedral angle distribution, CAD_Descriptors_Explained.pdf, Algorithm
    8: length-weighted histogram of fold angles over crease edges, plus total
    crease length and total mean curvature.

INPUTS:
    - PreparedMesh (welded mesh with consistent outward normals).
    - params: bins (36 bins of 10 deg), tau_deg (crease threshold, 20).

OUTPUTS:
    - float32 vector, shape (bins+2,): [Hellinger hist | Lambda | H].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - sigma_e = sign((b_j - b_i) . n_i), theta_e = pi + sigma_e arccos(n_i.n_j)
      in [0, 2 pi]: theta < pi convex, theta > pi concave (eq. 17).
    - Only edges with exactly two faces are used; creases are |theta-pi| > tau.
    - Binning is linear (soft): CAD creases are exactly 90 deg, which is a
      bin edge of the PDF's 36 x 10 deg bins, so hard binning flipped the
      whole crease mass between bins under rounding noise (measured 11 %
      vector change for a rotated box). Soft binning splits it evenly.
    - Lambda = sum_creases l_e / sqrt A, H = 1/2 sum_e (pi - theta_e) l_e / sqrt A
      (discrete total mean curvature, Steiner) (eq. 18).
    - Angles between normals and lengths / sqrt A: fully similarity-invariant.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from src.descriptors.base import Descriptor, soft_hellinger_hist


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class DADDihedralAngle(Descriptor):
    """DAD: crease-angle histogram + crease length + total bending."""
    # sigma_e = sign((b_j - b_i) . n_i),  theta_e = pi + sigma_e arccos(n_i . n_j) in [0, 2 pi]
    # F_DAD(b) = sum_{e in b, |theta_e - pi| > tau} l_e / sum_{|theta_e - pi| > tau} l_e  (36 x 10 deg)
    # Lambda = sum_creases l_e / sqrt A,  H = (1/2) sum_e (pi - theta_e) l_e / sqrt A
    # points: none sampled; every edge shared by two faces of the welded mesh

    name = "dad"
    version = "2"
    metric = "hellinger"
    defaults = {"bins": 36, "tau_deg": 20.0}

    def _setup(self) -> None:
        self.n_hist = int(self.params["bins"])
        self.dim = self.n_hist + 2

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        tm = mesh.mesh
        fa = np.asarray(tm.face_adjacency)                  # (E2, 2) manifold edges only
        if len(fa) == 0:
            return np.zeros(self.dim)
        N, B = mesh.face_normals, mesh.face_centers
        ni, nj = N[fa[:, 0]], N[fa[:, 1]]
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", ni, nj), -1.0, 1.0))
        sigma = np.sign(np.einsum("ij,ij->i", B[fa[:, 1]] - B[fa[:, 0]], ni))
        # eq.(17): theta_e = pi + sigma_e arccos(n_i . n_j)
        theta = math.pi + sigma * ang
        e = np.asarray(tm.face_adjacency_edges)
        length = np.linalg.norm(tm.vertices[e[:, 0]] - tm.vertices[e[:, 1]], axis=1)
        crease = np.abs(theta - math.pi) > math.radians(float(self.params["tau_deg"]))
        sqrtA = math.sqrt(mesh.area)
        h = soft_hellinger_hist(theta[crease], self.n_hist, 0.0, 2.0 * math.pi,
                                weights=length[crease])
        # eq.(18): Lambda = sum l_e / sqrt A, H = 1/2 sum (pi - theta) l_e / sqrt A
        lam = float(length[crease].sum()) / sqrtA
        H = 0.5 * float(((math.pi - theta) * length).sum()) / sqrtA
        return np.concatenate([h, [lam, H]])

# ==============================================================================
# END OF FILE: dad_dihedral_angle.py
# ==============================================================================
# SUMMARY:
#   DAD: 36-bin crease-angle histogram + crease length + total mean curvature.
#
# USAGE EXAMPLE:
#   from src.descriptors.dad_dihedral_angle import DADDihedralAngle
#   v = DADDihedralAngle().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Non-manifold edges (>2 faces) are ignored, as trimesh.face_adjacency does.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
