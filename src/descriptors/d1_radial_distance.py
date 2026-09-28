"""
================================================================================
FILE: d1_radial_distance.py
================================================================================
PURPOSE:
    D1 radial distance distribution, CAD_Descriptors_Explained.pdf, Algorithm
    1: histogram of the distances from the area-weighted surface centroid to
    the surface samples, scaled by the largest distance.

INPUTS:
    - PreparedMesh (default sample pool, centroid).
    - params: bins (PDF: 32).

OUTPUTS:
    - float32 vector, shape (bins,), Hellinger-mapped histogram on [0, 1].

DEPENDENCIES:
    stdlib      : (none)
    third-party : numpy
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - d_i = |p_i - c|, d^_i = d_i / max_j d_j. Invariant to translation (c
      moves with the part), rotation and reflection (norms), scale (ratio).
    - Reference: sphere -> all mass in the last bin; cube -> support
      [1/sqrt 3, 1].

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import numpy as np

from src.descriptors.base import EPS, Descriptor, DescriptorError, hellinger_hist


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class D1RadialDistance(Descriptor):
    """D1: radial distance distribution."""
    # c = sum_t A_t b_t / A,  d_i = |p_i - c|,  d^_i = d_i / max_j d_j in [0, 1]
    # F_D1 = hist32({d^_i}; [0, 1])
    # points: all n = 2e4 surface samples (pipeline.n_samples)

    name = "d1"
    version = "1"
    metric = "hellinger"
    defaults = {"bins": 32}

    def _setup(self) -> None:
        self.dim = self.n_hist = int(self.params["bins"])

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        # D1 eq.(5): d_i = |p_i - c|, normalised by max_j d_j
        d = np.linalg.norm(mesh.samples.points - mesh.centroid, axis=1)
        dmax = float(d.max())
        if dmax <= EPS:
            raise DescriptorError("d1: all samples coincide with the centroid")
        return hellinger_hist(d / dmax, self.dim, 0.0, 1.0)

# ==============================================================================
# END OF FILE: d1_radial_distance.py
# ==============================================================================
# SUMMARY:
#   D1 radial distance histogram (32 bins, Hellinger).
#
# USAGE EXAMPLE:
#   from src.descriptors.d1_radial_distance import D1RadialDistance
#   v = D1RadialDistance().compute(prepared_mesh, np.random.default_rng(0))
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Blind to direction: rotating one arm about the centre leaves D1 unchanged (by design).
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
