"""
================================================================================
FILE: d4_tetrahedron_volume.py
================================================================================
PURPOSE:
    D4 random tetrahedron volume with handedness,
    CAD_Descriptors_Explained.pdf, Algorithm 4: for S random sample quadruples
    ordered by distance from the centroid, the signed volume V. Output is the
    histogram of cbrt(|V| / D_diam^3) plus the skewness of V.

INPUTS:
    - PreparedMesh (default sample pool, centroid, D_diam), seeded rng.
    - params: bins (32), quadruples S (2e5).
    - device: "cuda" runs the gather / triple product in torch float64.

OUTPUTS:
    - float32 vector, shape (bins+1,): [Hellinger hist | gamma_1(V)].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, torch (optional, GPU path only)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - V = det[p2-p1, p3-p1, p4-p1]/6 with p1..p4 sorted by |p - c|.
    - det(RM) = det(M) for rotations; a mirror flips the sign of V, so
      gamma_1 flips sign: D4 is deliberately NOT reflection invariant.
    - Histogram range [0, (6 sqrt2)^(-1/3)]: the regular tetrahedron of
      edge D_diam is the largest.
    - The CPU and GPU paths use the same indices and float64 maths; they
      agree to ~1e-15 per volume (tested), so only values that sit exactly
      on a bin edge could differ.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from src.descriptors.base import (ALL_INVARIANCES, Descriptor, distinct_tuples,
                                  hellinger_hist, resolve_torch_device, skewness)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CBRT_VOL_MAX = (6.0 * math.sqrt(2.0)) ** (-1.0 / 3.0)   # [-] upper histogram edge


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _signed_volumes_numpy(P: np.ndarray, c: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Signed volumes of ordered tetrahedra [L^3], shape (S,)."""
    # points: S = 2e5 random quadruples (params.quadruples) drawn from n = 2e4 surface samples (pipeline.n_samples)
    q = P[idx]                                            # (S, 4, 3)
    r = np.linalg.norm(q - c, axis=2)
    order = np.argsort(r, axis=1, kind="stable")
    q = np.take_along_axis(q, order[:, :, None], axis=1)
    e1, e2, e3 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]
    # D4 eq.(9): V = ((p2-p1) x (p3-p1)) . (p4-p1) / 6
    return np.einsum("ij,ij->i", np.cross(e1, e2), e3) / 6.0


def _signed_volumes_torch(P: np.ndarray, c: np.ndarray, idx: np.ndarray, dev) -> np.ndarray:
    """GPU twin of _signed_volumes_numpy (float64, stable sort)."""
    import torch
    Pt = torch.as_tensor(P, dtype=torch.float64, device=dev)
    ct = torch.as_tensor(c, dtype=torch.float64, device=dev)
    q = Pt[torch.as_tensor(idx, device=dev)]
    r = torch.linalg.norm(q - ct, dim=2)
    order = torch.argsort(r, dim=1, stable=True)
    q = torch.take_along_dim(q, order[:, :, None], dim=1)
    e1, e2, e3 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]
    V = (torch.linalg.cross(e1, e2) * e3).sum(dim=1) / 6.0
    return V.cpu().numpy()


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class D4TetrahedronVolume(Descriptor):
    """D4: random tetrahedron volume with handedness."""
    # p1..p4 ordered by |p - c|;  V = det[p2 - p1, p3 - p1, p4 - p1] / 6
    # V^ = |V| / D_diam^3 in [0, 1/(6 sqrt2)]
    # F_D4 = [hist32({cbrt V^}; [0, (6 sqrt2)^(-1/3)]), gamma_1({V / sigma_V})]

    name = "d4"
    version = "1"
    metric = "hellinger"
    invariances = ALL_INVARIANCES - {"reflection"}
    defaults = {"bins": 32, "quadruples": 200_000}

    def _setup(self) -> None:
        self.n_hist = int(self.params["bins"])
        self.dim = self.n_hist + 1

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        P = mesh.samples.points
        idx = distinct_tuples(rng, len(P), int(self.params["quadruples"]), 4)
        dev = resolve_torch_device(self.device)
        if dev is None:
            V = _signed_volumes_numpy(P, mesh.centroid, idx)
        else:
            V = _signed_volumes_torch(P, mesh.centroid, idx, dev)
        # D4 eq.(10): hist of cbrt(|V| / D_diam^3), plus gamma_1(V / sigma_V)
        v_hat = np.cbrt(np.abs(V) / mesh.diameter ** 3)
        h = hellinger_hist(v_hat, self.n_hist, 0.0, CBRT_VOL_MAX)
        return np.append(h, skewness(V))

# ==============================================================================
# END OF FILE: d4_tetrahedron_volume.py
# ==============================================================================
# SUMMARY:
#   D4 ordered-tetrahedron volume histogram + handedness skewness (33 values).
#
# USAGE EXAMPLE:
#   from src.descriptors.d4_tetrahedron_volume import D4TetrahedronVolume
#   v = D4TetrahedronVolume(device="cuda").compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Vectorised S x 4 x 3 gather (~80 MB peak at S=2e5); stream in
#         chunks of 1e4 if memory matters (Complexity PDF).
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
