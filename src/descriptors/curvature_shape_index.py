"""
================================================================================
FILE: curvature_shape_index.py
================================================================================
PURPOSE:
    Curvature, what kind of curve and how strong (PDF Algorithm 9). Around
    each surface point it fits a small smooth patch to the nearby points and
    reads off how the surface bends there. Two numbers per point:
      - shape type (shape index): dent (-1) ... saddle (0) ... dome (+1),
      - bend strength (curvedness): how tightly it curves.
    Output: a 12 x 8 histogram of (shape type, bend strength) over the
    curved points, plus the share of the surface that is flat.

INPUTS:
    - PreparedMesh (surface sample points with normals, part size D_diam, area).
    - params: si_bins (12), cv_bins (8), radius_frac (patch radius = 0.02 D_diam),
      k_max (at most 30 neighbours), tau_deg (sharp-edge angle, 20),
      flat_eps (0.1), cv_max (100), min_neighbors (6).

OUTPUTS:
    - float32 vector, shape (si_bins*cv_bins + 1,): [histogram | flat share].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, scipy.spatial.cKDTree, torch (optional, device = cuda)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Each patch is z = a x^2 + b xy + c y^2 + d x + e y, with z along the
      outward normal. The two main bend values k1, k2 come from a, b, c.
      Sign: positive = bulging outward (a sphere has shape type +1).
    - Shape type = (2/pi) atan2(k1+k2, k1-k2).
      Bend strength = sqrt((k1^2+k2^2)/2), times sqrt(A/(4 pi)) to remove
      size (a ball then has bend strength 1).
    - Neighbours: points within 0.02 D_diam, at most 30 (keeps it fast).
    - Points next to a sharp edge (a neighbour faces > 20 deg away) are
      skipped; DAD handles edges. Points with fewer than 6 neighbours are
      skipped too (not enough to fit a patch).
    - Our choices (not in the PDF): "flat" means bend strength < 0.1 (10x
      flatter than a ball of the same area); curved values go into
      log-spaced bins from 0.1 to 100.
    - Meshes with triangles wider than the patch radius look flat.
    - GPU version (device = cuda) finds neighbours by brute force and
      solves all patches at once; same neighbours as the CPU (except exact ties).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from src.descriptors.base import Descriptor, hellinger, resolve_torch_device

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
KNN_CHUNK = 1024            # [-] GPU: points processed per batch (1024 x n float64 = 160 MB at n = 2e4)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _quadric_numpy(P: np.ndarray, N: np.ndarray, r: float, k_max: int, cos_tau: float,
                   min_nb: int) -> np.ndarray:
    """
    Fit a small patch at every smooth point and return its bend terms (a, b, c) (CPU).

    Parameters
    ----------
    P, N : np.ndarray, shape (n, 3)
        Sample points [L] and unit normals [-].
    r : float
        Patch radius [L].
    k_max : int
        Most neighbours to use.
    cos_tau : float
        Cosine of the sharp-edge angle.
    min_nb : int
        Fewest neighbours needed for a fit.

    Returns
    -------
    np.ndarray, shape (n_smooth, 3)
        a, b, c of z = a x^2 + b xy + c y^2 (+ d x + e y) [1/L].
    """
    n = len(P)
    dist, idx = cKDTree(P).query(P, k=k_max + 1, distance_upper_bound=r)
    idx, dist = idx[:, 1:], dist[:, 1:]                # drop the point itself
    valid = idx < n
    safe = np.where(valid, idx, 0)
    # near a sharp edge if any neighbour faces more than tau away
    cosang = np.einsum("ikj,ij->ik", N[safe], N)
    crease = np.any(valid & (cosang < cos_tau), axis=1)
    smooth = ~crease & (valid.sum(axis=1) >= min_nb)
    if not np.any(smooth):
        return np.zeros((0, 3))

    ex, ey = _tangent_frames(N)
    d = P[safe] - P[:, None, :]                          # (n, k, 3)
    x = np.einsum("ikj,ij->ik", d, ex)
    y = np.einsum("ikj,ij->ik", d, ey)
    z = np.einsum("ikj,ij->ik", d, N)
    # least-squares fit of z = a x^2 + b xy + c y^2 + d x + e y (missing neighbours zeroed) (eq. 19)
    A = np.stack([x * x, x * y, y * y, x, y], axis=2) * valid[:, :, None]
    zt = z * valid
    AtA = np.einsum("ikp,ikq->ipq", A[smooth], A[smooth])
    Atz = np.einsum("ikp,ik->ip", A[smooth], zt[smooth])
    # tiny extra term so nearly-in-a-line neighbours can still be solved
    ridge = (1e-12 * np.trace(AtA, axis1=1, axis2=2) + 1e-30)[:, None, None] * np.eye(5)
    return np.linalg.solve(AtA + ridge, Atz[:, :, None])[:, :3, 0]


def _quadric_torch(P: np.ndarray, N: np.ndarray, r: float, k_max: int, cos_tau: float,
                   min_nb: int, dev) -> np.ndarray:
    """GPU version of _quadric_numpy (same arguments, float64, brute-force neighbours)."""
    import torch
    Pt = torch.as_tensor(P, dtype=torch.float64, device=dev)
    Nt = torch.as_tensor(N, dtype=torch.float64, device=dev)
    k = min(k_max + 1, len(Pt))
    dist, idx = [], []
    for s in range(0, len(Pt), KNN_CHUNK):
        D = torch.cdist(Pt[s:s + KNN_CHUNK], Pt, compute_mode="donot_use_mm_for_euclid_dist")
        dk, ik = torch.topk(D, k, dim=1, largest=False)    # nearest first, so the point itself comes first
        dist.append(dk[:, 1:])
        idx.append(ik[:, 1:])
    dist, idx = torch.cat(dist), torch.cat(idx)
    valid = dist < r
    # near a sharp edge if any neighbour faces more than tau away
    cosang = (Nt[idx] * Nt[:, None, :]).sum(dim=2)
    crease = (valid & (cosang < cos_tau)).any(dim=1)
    smooth = ~crease & (valid.sum(dim=1) >= min_nb)
    if not bool(smooth.any()):
        return np.zeros((0, 3))

    Ps, Ns, idx, valid = Pt[smooth], Nt[smooth], idx[smooth], valid[smooth]
    helper = torch.zeros_like(Ns)
    xaxis = Ns[:, 0].abs() < 0.9
    helper[xaxis, 0] = 1.0
    helper[~xaxis, 1] = 1.0
    ex = torch.linalg.cross(Ns, helper)
    ex = ex / torch.linalg.vector_norm(ex, dim=1, keepdim=True)
    ey = torch.linalg.cross(Ns, ex)
    d = Pt[idx] - Ps[:, None, :]                         # (n_smooth, k, 3)
    x = (d * ex[:, None, :]).sum(dim=2)
    y = (d * ey[:, None, :]).sum(dim=2)
    z = (d * Ns[:, None, :]).sum(dim=2)
    # least-squares fit of z = a x^2 + b xy + c y^2 + d x + e y (missing neighbours zeroed) (eq. 19)
    A = torch.stack([x * x, x * y, y * y, x, y], dim=2) * valid[:, :, None]
    AtA = A.transpose(1, 2) @ A
    Atz = (A * (z * valid)[:, :, None]).sum(dim=1)
    ridge = (1e-12 * AtA.diagonal(dim1=1, dim2=2).sum(dim=1) + 1e-30)[:, None, None] \
        * torch.eye(5, dtype=torch.float64, device=dev)
    return torch.linalg.solve(AtA + ridge, Atz)[:, :3].cpu().numpy()


def _tangent_frames(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two fixed directions (e_x, e_y) lying flat on the surface, for unit normals n (N,3)."""
    # points used: all 2e4 surface samples (pipeline.n_samples), each fitted on <= 30 neighbours (params.k_max)
    helper = np.where(np.abs(n[:, [0]]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    ex = np.cross(n, helper)
    ex /= np.linalg.norm(ex, axis=1, keepdims=True)
    return ex, np.cross(n, ex)


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class CurvatureShapeIndex(Descriptor):
    """Curvature: histogram of (shape type, bend strength) over the surface."""
    # fit z = a x^2 + b xy + c y^2 + d x + e y within r = 0.02 D_diam
    # k1 >= k2 = main bend values from [[2a, b], [b, 2c]]
    # shape type = (2/pi) atan2(k1 + k2, k1 - k2), in [-1, 1]
    # bend strength = sqrt((k1^2 + k2^2) / 2) * sqrt(A / (4 pi))

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
        r = float(p["radius_frac"]) * mesh.diameter
        args = (s.points, s.normals, r, int(p["k_max"]),
                math.cos(math.radians(float(p["tau_deg"]))), int(p["min_neighbors"]))
        dev = resolve_torch_device(self.device)
        coef = _quadric_numpy(*args) if dev is None else _quadric_torch(*args, dev)
        if len(coef) == 0:
            return np.zeros(self.dim)
        a, b, c = coef[:, 0], coef[:, 1], coef[:, 2]
        # main bend values k1 >= k2 of the patch, positive = bulging outward (eq. 19)
        mean = -(a + c)
        rad = np.sqrt((a - c) ** 2 + b ** 2)
        k1, k2 = mean + rad, mean - rad
        # shape type and size-free bend strength (eq. 20)
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
#   Curvature: 12 x 8 (shape type, bend strength) histogram + flat share.
#
# USAGE EXAMPLE:
#   from src.descriptors.curvature_shape_index import CurvatureShapeIndex
#   v = CurvatureShapeIndex().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Triangles wider than 0.02 D_diam look flat (coarse meshes).
#   - [ ] All patches are fitted at once (~25 MB at defaults); do it in chunks if n grows.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-04  optional GPU version (device = cuda)
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
