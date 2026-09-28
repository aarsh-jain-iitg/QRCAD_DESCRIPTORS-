"""
================================================================================
FILE: d2_pairwise_distance.py
================================================================================
PURPOSE:
    D2 pairwise distance distribution, CAD_Descriptors_Explained.pdf,
    Algorithm 2: histogram of |p_i - p_j| / D_diam over S random pairs of
    surface samples.

INPUTS:
    - PreparedMesh (default sample pool, D_diam), seeded rng.
    - params: bins (32), pairs S (2e5).

OUTPUTS:
    - float32 vector, shape (bins,), Hellinger-mapped histogram on [0, 1].

DEPENDENCIES:
    stdlib      : (none)
    third-party : numpy
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Differences cancel translation, norms cancel rotation/reflection,
      D_diam cancels scale. Sphere reference density p(d) = d/(2R^2).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import numpy as np

from src.descriptors.base import Descriptor, distinct_tuples, hellinger_hist


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class D2PairwiseDistance(Descriptor):
    """D2: pairwise distance distribution."""
    # d^_ij = |p_i - p_j| / D_diam in [0, 1],  F_D2 = hist32({d^_ij}; [0, 1])
    # points: S = 2e5 random pairs (params.pairs) drawn from n = 2e4 surface samples (pipeline.n_samples)

    name = "d2"
    version = "1"
    metric = "hellinger"
    defaults = {"bins": 32, "pairs": 200_000}

    def _setup(self) -> None:
        self.dim = self.n_hist = int(self.params["bins"])

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        P = mesh.samples.points
        idx = distinct_tuples(rng, len(P), int(self.params["pairs"]), 2)
        # D2 eq.(6): d^_ij = |p_i - p_j| / D_diam
        d = np.linalg.norm(P[idx[:, 0]] - P[idx[:, 1]], axis=1) / mesh.diameter
        return hellinger_hist(d, self.dim, 0.0, 1.0)

# ==============================================================================
# END OF FILE: d2_pairwise_distance.py
# ==============================================================================
# SUMMARY:
#   D2 pairwise distance histogram (32 bins, Hellinger).
#
# USAGE EXAMPLE:
#   from src.descriptors.d2_pairwise_distance import D2PairwiseDistance
#   v = D2PairwiseDistance().compute(prepared_mesh, np.random.default_rng(0))
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Second-order blind to bending (PDF "What it still cannot see").
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
