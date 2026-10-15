"""
================================================================================
FILE: d4_tetrahedron_volume.py
================================================================================
PURPOSE:
    D4 random tetrahedron volume (unsigned),
    CAD_Descriptors_Explained.pdf, Algorithm 4: for S random sample quadruples the
    volume V = |det[p2-p1, p3-p1, p4-p1]| / 6 (never negative). Output is the
    histogram of cbrt(V / D_diam^3) plus the skewness of V.

INPUTS:
    - PreparedMesh (default sample pool, centroid, D_diam), seeded rng.
    - params: bins (32), quadruples S (2e5).
    - device: "cuda" runs the gather / triple product in torch float64.

OUTPUTS:
    - float32 vector, shape (bins+1,): [Hellinger hist | gamma_1(V)], gamma_1 of the unsigned volumes.

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, torch (optional, GPU path only)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - V = |det[p2-p1, p3-p1, p4-p1]|/6: a volume is never negative, and the
      absolute value makes D4 invariant to rotation and reflection (v1 kept the
      sign as handedness and was not mirror invariant).
    - Histogram range [0, (6 sqrt2)^(-1/3)]: the regular tetrahedron of
      edge D_diam is the largest.
    - The CPU and GPU paths use the same indices and float64 maths; they
      agree to ~1e-15 per volume (tested), so only values that sit exactly
      on a bin edge could differ.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-04
================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from src.descriptors.base import (ALL_INVARIANCES, Descriptor, distinct_tuples,
                                  hellinger_hist, resolve_torch_device, skewness,
                                  torch_gather)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CBRT_VOL_MAX = (6.0 * math.sqrt(2.0)) ** (-1.0 / 3.0)   # [-] upper histogram edge


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _volumes_numpy(P: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Volumes of tetrahedra [L^3], shape (S,), all >= 0."""
    # points: S = 2e5 random quadruples (params.quadruples) drawn from n = 2e4 surface samples (pipeline.n_samples)
    q = P[idx]                                            # (S, 4, 3)
    e1, e2, e3 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]
    # D4 eq.(9): V = |((p2-p1) x (p3-p1)) . (p4-p1)| / 6
    return np.abs(np.einsum("ij,ij->i", np.cross(e1, e2), e3)) / 6.0


def _volumes_torch(P: np.ndarray, idx: np.ndarray, dev) -> np.ndarray:
    """GPU twin of _volumes_numpy (float64)."""
    import torch
    q = torch_gather(P, idx, dev)
    e1, e2, e3 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]
    V = (torch.linalg.cross(e1, e2) * e3).sum(dim=1).abs() / 6.0
    return V.cpu().numpy()


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class D4TetrahedronVolume(Descriptor):
    """D4: random tetrahedron volume (unsigned)."""
    # V = |det[p2 - p1, p3 - p1, p4 - p1]| / 6
    # V^ = V / D_diam^3 in [0, 1/(6 sqrt2)]
    # F_D4 = [hist32({cbrt V^}; [0, (6 sqrt2)^(-1/3)]), gamma_1({V / sigma_V})]

    name = "d4"
    version = "2"
    metric = "hellinger"
    invariances = ALL_INVARIANCES
    defaults = {"bins": 32, "quadruples": 200_000}

    def _setup(self) -> None:
        self.n_hist = int(self.params["bins"])
        self.dim = self.n_hist + 1

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        P = mesh.samples.points
        idx = distinct_tuples(rng, len(P), int(self.params["quadruples"]), 4)
        dev = resolve_torch_device(self.device)
        if dev is None:
            V = _volumes_numpy(P, idx)
        else:
            V = _volumes_torch(P, idx, dev)
        # D4 eq.(10): hist of cbrt(V / D_diam^3), plus gamma_1(V / sigma_V)
        v_hat = np.cbrt(V / mesh.diameter ** 3)
        h = hellinger_hist(v_hat, self.n_hist, 0.0, CBRT_VOL_MAX)
        return np.append(h, skewness(V))

# ==============================================================================
# END OF FILE: d4_tetrahedron_volume.py
# ==============================================================================
# SUMMARY:
#   D4 tetrahedron volume histogram + skewness (unsigned volumes) (33 values).
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
#   2026-09-30  v2: unsigned volumes, no handedness; now mirror invariant
#   2026-10-04  GPU twin uses the shared base.torch_gather
# ==============================================================================
