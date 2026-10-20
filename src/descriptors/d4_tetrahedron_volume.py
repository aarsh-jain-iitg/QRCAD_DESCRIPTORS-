"""
================================================================================
FILE: d4_tetrahedron_volume.py
================================================================================
PURPOSE:
    D4, volume of random 4-point pyramids (PDF Algorithm 4). Picks many
    random sets of four surface points and measures the volume of the
    pyramid (tetrahedron) they form. Flat parts give small volumes, chunky
    parts give large ones. Output: histogram of the volumes plus one number
    for how lopsided they are.

INPUTS:
    - PreparedMesh (surface sample points, part size D_diam), seeded rng.
    - params: bins (32), quadruples (2e5).
    - device: "cuda" runs the volume maths on the GPU.

OUTPUTS:
    - float32 vector, shape (bins+1,): [volume histogram | skewness of the volumes].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, torch (optional, GPU only)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - V = |det[p2-p1, p3-p1, p4-p1]| / 6. The absolute value means a volume
      is never negative, so a mirrored part gives the same result
      (version 1 kept the sign and could tell left from right).
    - The histogram is of cbrt(V / D_diam^3) (cube root, so it behaves like
      a length). The largest possible pyramid is the regular one with edge
      D_diam, which sets the top of the range.
    - CPU and GPU use the same points and float64 maths; they agree to
      ~1e-15 per volume (tested).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from src.descriptors.base import (ALL_INVARIANCES, Descriptor, distinct_tuples,
                                  hellinger_hist, resolve_torch_device, skewness,
                                  torch_gather)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CBRT_VOL_MAX = (6.0 * math.sqrt(2.0)) ** (-1.0 / 3.0)   # [-] top of the histogram (regular pyramid)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _volumes_numpy(P: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Volume of each 4-point pyramid [L^3], shape (S,), all >= 0."""
    # points used: 2e5 random quadruples (params.quadruples) from the 2e4 surface samples (pipeline.n_samples)
    q = P[idx]                                            # (S, 4, 3)
    e1, e2, e3 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]
    # pyramid volume = |(e1 x e2) . e3| / 6 (eq. 9)
    return np.abs(np.einsum("ij,ij->i", np.cross(e1, e2), e3)) / 6.0


def _volumes_torch(P: np.ndarray, idx: np.ndarray, dev) -> np.ndarray:
    """GPU version of _volumes_numpy (float64)."""
    import torch
    q = torch_gather(P, idx, dev)
    e1, e2, e3 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]
    V = (torch.linalg.cross(e1, e2) * e3).sum(dim=1).abs() / 6.0
    return V.cpu().numpy()


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class D4TetrahedronVolume(Descriptor):
    """D4: volumes of random 4-point pyramids (never negative)."""
    # V = |det[p2 - p1, p3 - p1, p4 - p1]| / 6
    # output = [32-bin histogram of cbrt(V / D_diam^3), skewness of V]

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
        # histogram of cube-root volume / part size, plus skewness (eq. 10)
        v_hat = np.cbrt(V / mesh.diameter ** 3)
        h = hellinger_hist(v_hat, self.n_hist, 0.0, CBRT_VOL_MAX)
        return np.append(h, skewness(V))

# ==============================================================================
# END OF FILE: d4_tetrahedron_volume.py
# ==============================================================================
# SUMMARY:
#   D4: volume histogram of random 4-point pyramids + skewness (33 values).
#
# USAGE EXAMPLE:
#   from src.descriptors.d4_tetrahedron_volume import D4TetrahedronVolume
#   v = D4TetrahedronVolume(device="cuda").compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] All 2e5 x 4 points are loaded at once (~80 MB peak); process
#         in chunks of 1e4 if memory matters.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-09-30  v2: volumes never negative, so left/right mirror copies match
#   2026-10-04  GPU version uses the shared base.torch_gather
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
