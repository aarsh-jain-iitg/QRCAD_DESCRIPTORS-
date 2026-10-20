"""
================================================================================
FILE: d2_pairwise_distance.py
================================================================================
PURPOSE:
    D2, distance between random point pairs (PDF Algorithm 2). Picks many
    random pairs of surface points, measures how far apart each pair is
    (divided by the part size D_diam), and stores those values as a histogram.

INPUTS:
    - PreparedMesh (surface sample points, part size D_diam), seeded rng.
    - params: bins (32), pairs (2e5).

OUTPUTS:
    - float32 vector, shape (bins,): histogram of pair distances on [0, 1].

DEPENDENCIES:
    stdlib      : (none)
    third-party : numpy, torch (optional, device = cuda)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - A distance between two points does not change when the part is moved,
      turned or mirrored; dividing by D_diam removes size.
    - Example: for a sphere the histogram rises in a straight line.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import numpy as np

from src.descriptors.base import (Descriptor, distinct_tuples, hellinger_hist,
                                  resolve_torch_device, torch_gather)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _distances_torch(P: np.ndarray, idx: np.ndarray, dev) -> np.ndarray:
    """GPU version: distance |p_i - p_j| for each index pair in idx (S, 2) [L], shape (S,)."""
    import torch
    q = torch_gather(P, idx, dev)
    return torch.linalg.vector_norm(q[:, 0] - q[:, 1], dim=1).cpu().numpy()


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class D2PairwiseDistance(Descriptor):
    """D2: distances between random pairs of surface points."""
    # d_ij = |p_i - p_j| / D_diam, a value in [0, 1]; output = 32-bin histogram of d_ij
    # points used: 2e5 random pairs (params.pairs) from the 2e4 surface samples (pipeline.n_samples)

    name = "d2"
    version = "1"
    metric = "hellinger"
    defaults = {"bins": 32, "pairs": 200_000}

    def _setup(self) -> None:
        self.dim = self.n_hist = int(self.params["bins"])

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        P = mesh.samples.points
        idx = distinct_tuples(rng, len(P), int(self.params["pairs"]), 2)
        # distance of each pair, divided by the part size (eq. 6)
        dev = resolve_torch_device(self.device)
        if dev is None:
            d = np.linalg.norm(P[idx[:, 0]] - P[idx[:, 1]], axis=1) / mesh.diameter
        else:
            d = _distances_torch(P, idx, dev) / mesh.diameter
        return hellinger_hist(d, self.dim, 0.0, 1.0)

# ==============================================================================
# END OF FILE: d2_pairwise_distance.py
# ==============================================================================
# SUMMARY:
#   D2: 32-bin histogram of random pair distances.
#
# USAGE EXAMPLE:
#   from src.descriptors.d2_pairwise_distance import D2PairwiseDistance
#   v = D2PairwiseDistance().compute(prepared_mesh, np.random.default_rng(0))
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Barely notices bending: a bent bar and a straight bar look alike.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-04  optional GPU version of the pair distances (device = cuda)
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
