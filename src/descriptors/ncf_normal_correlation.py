"""
================================================================================
FILE: ncf_normal_correlation.py
================================================================================
PURPOSE:
    NCF normal correlation over distance, CAD_Descriptors_Explained.pdf,
    Algorithm 6: for S random sample pairs, accumulate the normal agreement
    n_i . n_j into the bin of their normalised distance.

INPUTS:
    - PreparedMesh (default sample pool with normals, D_diam), seeded rng.
    - params: bins (32), pairs S (2e5).

OUTPUTS:
    - float32 vector, shape (bins,): g(m) = (1/S) sum (n_i . n_j) 1[r_ij in B_m].
      Signed values; not a histogram, so it is z-scored downstream.

DEPENDENCIES:
    stdlib      : (none)
    third-party : numpy
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - r and n_i . n_j are unchanged by rotation, reflection, shift, scale.
    - sum_m g(m) = |E n|^2 = 0 for a closed part.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import numpy as np

from src.descriptors.base import Descriptor, distinct_tuples


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class NCFNormalCorrelation(Descriptor):
    """NCF: normal agreement as a function of distance."""
    # r^_ij = |p_i - p_j| / D_diam
    # g(m) = (1/S) sum_(i,j) (n_i . n_j) 1[r^_ij in B_m],  m = 1..32
    # points: S = 2e5 random pairs (params.pairs) drawn from n = 2e4 surface samples (pipeline.n_samples)

    name = "ncf"
    version = "1"
    metric = "l2"
    defaults = {"bins": 32, "pairs": 200_000}

    def _setup(self) -> None:
        self.dim = int(self.params["bins"])
        self.n_hist = 0

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        s = mesh.samples
        idx = distinct_tuples(rng, len(s.points), int(self.params["pairs"]), 2)
        i, j = idx[:, 0], idx[:, 1]
        r = np.linalg.norm(s.points[i] - s.points[j], axis=1) / mesh.diameter
        m = np.clip((r * self.dim).astype(np.int64), 0, self.dim - 1)
        # eq.(13): g(m) = (1/S) sum (n_i . n_j) 1[r_ij in B_m]
        g = np.bincount(m, weights=np.einsum("ij,ij->i", s.normals[i], s.normals[j]),
                        minlength=self.dim)
        return g / len(idx)

# ==============================================================================
# END OF FILE: ncf_normal_correlation.py
# ==============================================================================
# SUMMARY:
#   NCF: 32-bin signed normal-agreement profile over normalised distance.
#
# USAGE EXAMPLE:
#   from src.descriptors.ncf_normal_correlation import NCFNormalCorrelation
#   v = NCFNormalCorrelation().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Measures positions and directions only, not connectivity.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
