"""
================================================================================
FILE: triangle_a3_d3.py
================================================================================
PURPOSE:
    Triangle descriptor (A3 angle + D3 area), CAD_Descriptors_Explained.pdf,
    Algorithm 3: for S random sample triples, the corner angle at p1 and the
    square root of the triangle area / D_diam^2.

INPUTS:
    - PreparedMesh (default sample pool, D_diam), seeded rng.
    - params: bins (32 per reading), triples S (2e5).

OUTPUTS:
    - float32 vector, shape (2*bins,): [hist(theta) | hist(sqrt A^)], each
      Hellinger-mapped.

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, torch (optional, device = cuda)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - theta = atan2(|u x w|, u.w) in [0, pi] (atan2 keeps 0 and pi accurate).
    - A^ = 0.5 |u x w| / D_diam^2 in [0, sqrt3/4]; histogram of sqrt(A^) on
      [0, sqrt(sqrt3/4)] (equilateral triangle of side D_diam is the maximum).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-04
================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from src.descriptors.base import (Descriptor, distinct_tuples, hellinger_hist,
                                  resolve_torch_device, torch_gather)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SQRT_AREA_MAX = math.sqrt(math.sqrt(3.0) / 4.0)   # [-] upper edge of the D3 histogram


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _cross_dot_torch(P: np.ndarray, idx: np.ndarray, dev) -> tuple[np.ndarray, np.ndarray]:
    """GPU twin of (|u x w| [L^2], u . w [L^2]) for index triples idx (S, 3), float64."""
    import torch
    q = torch_gather(P, idx, dev)
    u, w = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    cross = torch.linalg.vector_norm(torch.linalg.cross(u, w), dim=1)
    return cross.cpu().numpy(), (u * w).sum(dim=1).cpu().numpy()


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class TriangleA3D3(Descriptor):
    """Triangle descriptor: A3 angle + D3 area."""
    # u = p2 - p1,  w = p3 - p1
    # theta = atan2(|u x w|, u . w) in [0, pi],  F_A3 = hist32({theta}; [0, pi])
    # A^ = |u x w| / (2 D_diam^2) in [0, sqrt3/4],  F_D3 = hist32({sqrt A^}; [0, sqrt(sqrt3/4)])
    # points: S = 2e5 random triples (params.triples) drawn from n = 2e4 surface samples (pipeline.n_samples)

    name = "triangle"
    version = "1"
    metric = "hellinger"
    defaults = {"bins": 32, "triples": 200_000}

    def _setup(self) -> None:
        self.dim = self.n_hist = 2 * int(self.params["bins"])

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        P = mesh.samples.points
        idx = distinct_tuples(rng, len(P), int(self.params["triples"]), 3)
        dev = resolve_torch_device(self.device)
        if dev is None:
            u = P[idx[:, 1]] - P[idx[:, 0]]
            w = P[idx[:, 2]] - P[idx[:, 0]]
            cross = np.linalg.norm(np.cross(u, w), axis=1)
            dot = np.einsum("ij,ij->i", u, w)
        else:
            cross, dot = _cross_dot_torch(P, idx, dev)
        # A3 eq.(7): theta = atan2(|u x w|, u.w)
        theta = np.arctan2(cross, dot)
        # D3 eq.(8): sqrt(A^), A^ = 0.5 |u x w| / D_diam^2
        a = np.sqrt(0.5 * cross / mesh.diameter ** 2)
        b = int(self.params["bins"])
        return np.concatenate([hellinger_hist(theta, b, 0.0, math.pi),
                               hellinger_hist(a, b, 0.0, SQRT_AREA_MAX)])

# ==============================================================================
# END OF FILE: triangle_a3_d3.py
# ==============================================================================
# SUMMARY:
#   A3 + D3 random-triangle histograms (2 x 32 bins, Hellinger).
#
# USAGE EXAMPLE:
#   from src.descriptors.triangle_a3_d3 import TriangleA3D3
#   v = TriangleA3D3().compute(prepared_mesh, np.random.default_rng(0))
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Cannot see departure from a plane; that is D4's job.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-04  optional GPU path for the triple cross/dot products (device = cuda)
# ==============================================================================
