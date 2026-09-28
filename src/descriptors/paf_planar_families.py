"""
================================================================================
FILE: paf_planar_families.py
================================================================================
PURPOSE:
    PAF planar face families, CAD_Descriptors_Explained.pdf, Algorithm 14:
    global normal moments (Westin coefficients, I4, normal entropy) plus, for
    the 6 largest families of parallel planar faces, the area share, number of
    distinct planes, spacing regularity and angle to the main family.

INPUTS:
    - PreparedMesh (face normals, areas, centres, adjacency angles, D_diam).
    - params: family_tol_deg (0.25), plane_tol (1e-4 of D_diam),
      max_families (64, Complexity PDF cap), top_families (6),
      tau_deg (20), ico_subdivisions (4 -> 2562 cells).

OUTPUTS:
    - float32 vector, shape (7 + 4*top_families,) = 31:
      [c_l, c_p, c_s, I4, H, N_fam/16, Pi] + 6 x [alpha_s, N_s/10, CV_s, angle/90].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, scipy.spatial.cKDTree, trimesh
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - T2 = sum w_t n_t n_t^T, eigenvalues L1 >= L2 >= L3:
      c_l = L1-L2, c_p = 2(L2-L3), c_s = 3 L3, summing to 1.
    - I4 = sum_s sum_t w_s w_t (n_s.n_t)^4 = |T4|_F^2 (cube 1/3, ball 1/5).
    - H = entropy of the area-weighted normals on an icosphere grid,
      divided by log(G) so it lies in [0, 1]. The grid is fixed in space,
      so H is rotation invariant only up to cell quantisation; every
      other entry is exactly invariant.
    - "Planar face" is not defined for STL triangles in the PDF. Chosen: a
      triangle is planar unless it meets a neighbour at a small non-zero
      angle (0.25 deg < angle < tau), i.e. unless it is a facet of a
      tessellated curved surface. A flat CAD face (coplanar neighbours,
      creases at its rim) and a lone triangle are both planar.
    - Families: planar faces sorted by area; the largest unassigned face
      opens a family and claims every unassigned face with
      |n.m| >= cos(0.25 deg). This equals the PDF's greedy scan.
    - Plane offsets h = b.m / D_diam, clustered in 1-D with tolerance 1e-4.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from src.descriptors.base import Descriptor, angle_between_lines_deg


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _plane_stats(h: np.ndarray, tol: float) -> tuple[int, float]:
    """Number of distinct planes and gap CV from 1-D offsets h (dimensionless)."""
    # points: none sampled; all mesh faces (normals, areas, centres)
    hs = np.sort(h)
    starts = np.concatenate([[0], np.nonzero(np.diff(hs) > tol)[0] + 1])
    planes = np.add.reduceat(hs, starts) / np.diff(np.append(starts, len(hs)))
    Ns = len(planes)
    if Ns < 3:
        return Ns, 0.0
    gaps = np.diff(planes)
    mu = float(gaps.mean())
    return Ns, float(gaps.std() / mu) if mu > 0 else 0.0


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class PAFPlanarFamilies(Descriptor):
    """PAF: planar face families and normal moments."""
    # T2 = sum_t w_t n_t n_t^T,  Lambda_1 >= Lambda_2 >= Lambda_3
    # c_l = Lambda_1 - Lambda_2,  c_p = 2 (Lambda_2 - Lambda_3),  c_s = 3 Lambda_3
    # I4 = sum_{s,t} w_s w_t (n_s . n_t)^4

    name = "paf"
    version = "1"
    metric = "l2"
    defaults = {"family_tol_deg": 0.25, "plane_tol": 1e-4, "max_families": 64,
                "top_families": 6, "tau_deg": 20.0, "ico_subdivisions": 4}

    def _setup(self) -> None:
        self.dim = 7 + 4 * int(self.params["top_families"])
        self.n_hist = 0

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        p = self.params
        N = mesh.face_normals
        w = mesh.face_areas / mesh.area

        # eq.(28): T2 and Westin coefficients
        T2 = np.einsum("t,ti,tj->ij", w, N, N)
        L = np.sort(np.linalg.eigvalsh(T2))[::-1]
        cl, cp, cs = L[0] - L[1], 2.0 * (L[1] - L[2]), 3.0 * L[2]
        # I4 = |T4|_F^2, T4 = sum w n(x)n(x)n(x)n
        T4 = np.einsum("t,ti,tj,tk,tl->ijkl", w, N, N, N, N, optimize=True)
        I4 = float((T4 ** 2).sum())

        ico = trimesh.creation.icosphere(subdivisions=int(p["ico_subdivisions"]))
        _, cell = cKDTree(ico.vertices).query(N)
        q = np.bincount(cell, weights=w, minlength=len(ico.vertices))
        q = q[q > 0]
        H = float(-(q * np.log(q)).sum() / math.log(len(ico.vertices)))

        # planar faces: no small non-zero fold to a neighbour
        tol_rad = math.radians(float(p["family_tol_deg"]))
        planar = np.ones(len(N), dtype=bool)
        fa = np.asarray(mesh.mesh.face_adjacency)
        if len(fa):
            ang = np.asarray(mesh.mesh.face_adjacency_angles)
            soft = (ang > tol_rad) & (ang < math.radians(float(p["tau_deg"])))
            planar[fa[soft].ravel()] = False

        cos_tol = math.cos(tol_rad)
        order = np.argsort(-mesh.face_areas, kind="stable")
        order = order[planar[order]]
        free = np.ones(len(N), dtype=bool)
        families = []                                    # (alpha, m, members)
        for t in order:
            if len(families) >= int(p["max_families"]):
                break
            if not free[t]:
                continue
            m = N[t]
            members = free & planar & (np.abs(N @ m) >= cos_tol)
            free &= ~members
            families.append((float(w[members].sum()), m, np.nonzero(members)[0]))

        families.sort(key=lambda f: -f[0])
        Pi = float(sum(f[0] for f in families))
        head = [cl, cp, cs, I4, H, len(families) / 16.0, Pi]
        rows = np.zeros((int(p["top_families"]), 4))
        for k, (alpha, m, members) in enumerate(families[:int(p["top_families"])]):
            h = mesh.face_centers[members] @ m / mesh.diameter
            Ns, cv = _plane_stats(h, float(p["plane_tol"]))
            rows[k] = [alpha, Ns / 10.0, cv, angle_between_lines_deg(m, families[0][1]) / 90.0]
        return np.concatenate([head, rows.ravel()])

# ==============================================================================
# END OF FILE: paf_planar_families.py
# ==============================================================================
# SUMMARY:
#   PAF: Westin/I4/entropy normal moments + 6 largest parallel-plane families.
#
# USAGE EXAMPLE:
#   from src.descriptors.paf_planar_families import PAFPlanarFamilies
#   v = PAFPlanarFamilies().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Entropy H is quantised on a fixed grid (approximately rotation invariant).
#   - [ ] Family loop is O(F * N_fam) with N_fam <= 64 (Complexity PDF).
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
