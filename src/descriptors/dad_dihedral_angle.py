"""
================================================================================
FILE: dad_dihedral_angle.py
================================================================================
PURPOSE:
    DAD, sharp edges and their angles (PDF Algorithm 8). Looks at every
    edge where two triangles meet and measures the fold angle there. Sharp
    edges (folds of more than 20 deg) go into a histogram weighted by edge
    length. Also stores the total length of sharp edges and the total
    amount of bending.

INPUTS:
    - PreparedMesh (cleaned mesh with normals pointing outward).
    - params: bins (36 bins of 10 deg), tau_deg (sharp-edge threshold, 20).

OUTPUTS:
    - float32 vector, shape (bins+2,): [angle histogram | sharp-edge length | total bending].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Fold angle from 0 to 360 deg: 180 is flat, below 180 is an outside
      corner (convex), above 180 is an inside corner (concave) (eq. 17).
    - Only edges shared by exactly two triangles are used. An edge is
      sharp when its angle is more than tau_deg away from flat.
    - Each angle is shared between its two nearest bins. CAD corners are
      exactly 90 deg, which falls right on a bin edge; without sharing,
      tiny rounding errors moved them between bins (measured 11 % change
      for a rotated box).
    - Sharp-edge length and total bending are divided by sqrt(area) so
      they do not depend on size (eq. 18).
    - Uses angles and length ratios only: moving, turning, mirroring or
      scaling the part changes nothing.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from src.descriptors.base import Descriptor, soft_hellinger_hist


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class DADDihedralAngle(Descriptor):
    """DAD: sharp-edge angle histogram + sharp-edge length + total bending."""
    # sign = +1 for an outside corner, -1 for an inside corner
    # angle = pi + sign * (angle between the two normals), in [0, 2 pi]
    # histogram: 36 x 10 deg bins of sharp-edge angles, weighted by edge length
    # length = sum of sharp-edge lengths / sqrt A,  bending = (1/2) sum (pi - angle) * length / sqrt A
    # points used: none sampled; every edge shared by two triangles

    name = "dad"
    version = "2"
    metric = "hellinger"
    defaults = {"bins": 36, "tau_deg": 20.0}

    def _setup(self) -> None:
        self.n_hist = int(self.params["bins"])
        self.dim = self.n_hist + 2

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        tm = mesh.mesh
        fa = np.asarray(tm.face_adjacency)                  # (E2, 2) edges with exactly two triangles
        if len(fa) == 0:
            return np.zeros(self.dim)
        N, B = mesh.face_normals, mesh.face_centers
        ni, nj = N[fa[:, 0]], N[fa[:, 1]]
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", ni, nj), -1.0, 1.0))
        sigma = np.sign(np.einsum("ij,ij->i", B[fa[:, 1]] - B[fa[:, 0]], ni))
        # fold angle: 180 deg flat, below = outside corner, above = inside corner (eq. 17)
        theta = math.pi + sigma * ang
        e = np.asarray(tm.face_adjacency_edges)
        length = np.linalg.norm(tm.vertices[e[:, 0]] - tm.vertices[e[:, 1]], axis=1)
        crease = np.abs(theta - math.pi) > math.radians(float(self.params["tau_deg"]))
        sqrtA = math.sqrt(mesh.area)
        h = soft_hellinger_hist(theta[crease], self.n_hist, 0.0, 2.0 * math.pi,
                                weights=length[crease])
        # sharp-edge length and total bending, divided by sqrt(area) (eq. 18)
        lam = float(length[crease].sum()) / sqrtA
        H = 0.5 * float(((math.pi - theta) * length).sum()) / sqrtA
        return np.concatenate([h, [lam, H]])

# ==============================================================================
# END OF FILE: dad_dihedral_angle.py
# ==============================================================================
# SUMMARY:
#   DAD: 36-bin sharp-edge angle histogram + sharp-edge length + total bending.
#
# USAGE EXAMPLE:
#   from src.descriptors.dad_dihedral_angle import DADDihedralAngle
#   v = DADDihedralAngle().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Edges shared by more than two triangles are skipped (as trimesh does).
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
