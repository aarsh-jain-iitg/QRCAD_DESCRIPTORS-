"""
================================================================================
FILE: triangle_a3_d3.py
================================================================================
PURPOSE:
    Triangle descriptor, A3 angle + D3 area (PDF Algorithm 3). Picks many
    random sets of three surface points. For each triangle it records the
    angle at the first corner and the size of the triangle, and stores both
    as histograms.

INPUTS:
    - PreparedMesh (surface sample points, part size D_diam), seeded rng.
    - params: bins (32 for each histogram), triples (2e5).

OUTPUTS:
    - float32 vector, shape (2*bins,): [angle histogram | area histogram].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, torch (optional, device = cuda)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Angle at corner p1, between 0 and 180 deg. Computed with atan2,
      which stays accurate for angles near 0 and 180 deg.
    - Area is divided by D_diam^2; the histogram is of its square root
      (so it behaves like a length). The largest possible triangle is the
      equilateral one with side D_diam, which sets the top of the range.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from src.descriptors.base import (Descriptor, distinct_tuples, hellinger_hist,
                                  resolve_torch_device, torch_gather)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SQRT_AREA_MAX = math.sqrt(math.sqrt(3.0) / 4.0)   # [-] top of the area histogram (equilateral triangle)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _cross_dot_torch(P: np.ndarray, idx: np.ndarray, dev) -> tuple[np.ndarray, np.ndarray]:
    """GPU version: (|u x w| [L^2], u . w [L^2]) for each index triple in idx (S, 3)."""
    import torch
    q = torch_gather(P, idx, dev)
    u, w = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    cross = torch.linalg.vector_norm(torch.linalg.cross(u, w), dim=1)
    return cross.cpu().numpy(), (u * w).sum(dim=1).cpu().numpy()


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class TriangleA3D3(Descriptor):
    """Triangle descriptor: corner angle (A3) + triangle size (D3)."""
    # u = p2 - p1,  w = p3 - p1  (the two sides that meet at p1)
    # angle = atan2(|u x w|, u . w) in [0, pi]           -> 32-bin histogram
    # area = |u x w| / (2 D_diam^2); sqrt(area) on [0, sqrt(sqrt3/4)] -> 32-bin histogram
    # points used: 2e5 random triples (params.triples) from the 2e4 surface samples (pipeline.n_samples)

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
        # corner angle at p1 (eq. 7)
        theta = np.arctan2(cross, dot)
        # square root of triangle area / D_diam^2 (eq. 8)
        a = np.sqrt(0.5 * cross / mesh.diameter ** 2)
        b = int(self.params["bins"])
        return np.concatenate([hellinger_hist(theta, b, 0.0, math.pi),
                               hellinger_hist(a, b, 0.0, SQRT_AREA_MAX)])

# ==============================================================================
# END OF FILE: triangle_a3_d3.py
# ==============================================================================
# SUMMARY:
#   Triangle: angle + size histograms of random point triangles (2 x 32 bins).
#
# USAGE EXAMPLE:
#   from src.descriptors.triangle_a3_d3 import TriangleA3D3
#   v = TriangleA3D3().compute(prepared_mesh, np.random.default_rng(0))
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Cannot tell how far a shape is from flat; D4 covers that.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-04  optional GPU version (device = cuda)
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
