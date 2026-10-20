"""
================================================================================
FILE: paf_planar_families.py
================================================================================
PURPOSE:
    PAF, flat faces and how they line up (PDF Algorithm 14). Two parts:
      1. Overall direction summary: is the surface mostly facing one way
         (rod-like), two ways (plate-like) or all ways (ball-like)?
      2. Groups of parallel flat faces (e.g. the top, bottom and step faces
         of a stepped block all face up). For the 6 biggest groups it
         records: share of total area, how many separate levels (planes)
         there are, how evenly those levels are spaced, and the angle to
         the biggest group.

INPUTS:
    - PreparedMesh (triangle normals, areas, centres, neighbour angles, D_diam).
    - params: family_tol_deg (0.25), plane_tol (1e-4 of D_diam),
      max_families (64), top_families (6), tau_deg (20),
      ico_subdivisions (4 -> 2562 direction cells).

OUTPUTS:
    - float32 vector, shape (7 + 4*top_families,) = 31:
      [rod-like, plate-like, ball-like, box-likeness I4, direction spread H,
       #groups/16, flat share] + 6 x [area share, #levels/10, spacing
       unevenness, angle to biggest group/90].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, scipy.spatial.cKDTree, trimesh, torch (optional, device = cuda)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Rod / plate / ball scores come from the average of n n^T over the
      surface (area-weighted normals n); the three scores add up to 1.
    - I4 (box-likeness) = average (n_s . n_t)^4 over all triangle pairs:
      1/3 for a cube, 1/5 for a ball.
    - H = how evenly the normals spread over a fixed grid of 2562
      directions, from 0 (one direction) to 1 (all directions). The grid is
      fixed, so H changes very slightly when the part is turned; every
      other value does not change at all.
    - The PDF does not define a "flat face" for STL triangles. Our rule: a
      triangle is flat unless it meets a neighbour at a small angle
      (between 0.25 and 20 deg), which is what pieces of a curved surface
      do. A flat CAD face and a lone triangle both count as flat.
    - Groups: take the biggest flat triangle not yet grouped; every
      ungrouped flat triangle facing the same way (within 0.25 deg) joins
      its group. Repeat.
    - Levels: each triangle's height along the group direction / D_diam,
      merged when closer than 1e-4.
    - GPU version (device = cuda) does the direction summary and grid
      lookup on the GPU; the grouping loop stays on the CPU (<= 64 steps).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from src.descriptors.base import Descriptor, angle_between_lines_deg, resolve_torch_device

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CELL_CHUNK = 8192           # [-] GPU: normals per batch for the grid lookup (8192 x 2562 float64 = 168 MB)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _plane_stats(h: np.ndarray, tol: float) -> tuple[int, float]:
    """Number of separate levels and how unevenly they are spaced, from heights h (unitless)."""
    # points used: none sampled; every triangle (normals, areas, centres)
    if len(h) == 0:                       # empty group (its first triangle was not flat)
        return 0, 0.0
    hs = np.sort(h)
    starts = np.concatenate([[0], np.nonzero(np.diff(hs) > tol)[0] + 1])
    planes = np.add.reduceat(hs, starts) / np.diff(np.append(starts, len(hs)))
    Ns = len(planes)
    if Ns < 3:
        return Ns, 0.0
    gaps = np.diff(planes)
    mu = float(gaps.mean())
    return Ns, float(gaps.std() / mu) if mu > 0 else 0.0


def _moments_cells_torch(N: np.ndarray, w: np.ndarray, ico: np.ndarray,
                        dev) -> tuple[np.ndarray, float, np.ndarray]:
    """
    GPU version: (T2, I4, grid cell of each normal), float64.

    Parameters
    ----------
    N : np.ndarray, shape (F, 3)
        Unit triangle normals [-].
    w : np.ndarray, shape (F,)
        Area weights [-].
    ico : np.ndarray, shape (G, 3)
        Unit directions of the grid.
    dev : torch.device

    Returns
    -------
    T2 : np.ndarray, shape (3, 3)
    I4 : float
    cell : np.ndarray, shape (F,), int64
    """
    import torch
    Nt = torch.as_tensor(N, dtype=torch.float64, device=dev)
    wt = torch.as_tensor(w, dtype=torch.float64, device=dev)
    Vt = torch.as_tensor(ico, dtype=torch.float64, device=dev)
    T2 = (Nt * wt[:, None]).T @ Nt
    M = (Nt[:, :, None] * Nt[:, None, :]).reshape(-1, 9)     # n n^T as 9 numbers
    T4 = (M * wt[:, None]).T @ M                              # 4th-order average as a 9 x 9 table
    cell = torch.cat([torch.argmax(Nt[s:s + CELL_CHUNK] @ Vt.T, dim=1)
                      for s in range(0, len(Nt), CELL_CHUNK)])
    return T2.cpu().numpy(), float((T4 ** 2).sum()), cell.cpu().numpy()


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class PAFPlanarFamilies(Descriptor):
    """PAF: groups of parallel flat faces + overall direction summary."""
    # T2 = sum of w n n^T over triangles; its three values L1 >= L2 >= L3
    # rod-like = L1 - L2,  plate-like = 2 (L2 - L3),  ball-like = 3 L3
    # I4 = sum over triangle pairs of w_s w_t (n_s . n_t)^4

    name = "paf"
    version = "2"
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

        ico = trimesh.creation.icosphere(subdivisions=int(p["ico_subdivisions"]))
        dev = resolve_torch_device(self.device)
        if dev is None:
            # area-weighted average of n n^T (eq. 28)
            T2 = np.einsum("t,ti,tj->ij", w, N, N)
            # box-likeness I4 from the 4th-order average
            T4 = np.einsum("t,ti,tj,tk,tl->ijkl", w, N, N, N, N, optimize=True)
            I4 = float((T4 ** 2).sum())
            _, cell = cKDTree(ico.vertices).query(N)
        else:
            T2, I4, cell = _moments_cells_torch(N, w, ico.vertices, dev)
        # rod / plate / ball scores (eq. 28)
        L = np.sort(np.linalg.eigvalsh(T2))[::-1]
        cl, cp, cs = L[0] - L[1], 2.0 * (L[1] - L[2]), 3.0 * L[2]

        q = np.bincount(cell, weights=w, minlength=len(ico.vertices))
        q = q[q > 0]
        H = float(-(q * np.log(q)).sum() / math.log(len(ico.vertices)))

        # flat triangles: no small (curved-surface) fold to any neighbour
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
        families = []                                    # (area share, direction, members)
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
#   PAF: overall direction summary + the 6 biggest groups of parallel flat faces.
#
# USAGE EXAMPLE:
#   from src.descriptors.paf_planar_families import PAFPlanarFamilies
#   v = PAFPlanarFamilies().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Direction spread H uses a fixed grid, so it changes very slightly when the part turns.
#   - [ ] Grouping loop time grows with triangles x groups (at most 64 groups).
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-09-30  v2: a group with no flat triangles no longer crashes
#   2026-10-04  optional GPU version (device = cuda)
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
