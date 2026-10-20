"""
================================================================================
FILE: d1_radial_distance.py
================================================================================
PURPOSE:
    D1, distance from the centre (PDF Algorithm 1). Measures how far each
    surface point is from the part's centre, divides by the largest such
    distance, and stores the spread of those values as a histogram.

INPUTS:
    - PreparedMesh (surface sample points, centre).
    - params: bins (32).

OUTPUTS:
    - float32 vector, shape (bins,): histogram of the distances on [0, 1].

DEPENDENCIES:
    stdlib      : (none)
    third-party : numpy
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - d_i = |p_i - c|, divided by the largest d. The centre moves with the
      part, distances do not change when the part is turned or mirrored,
      and dividing by the largest distance removes size.
    - Examples: a sphere puts everything in the last bin; a cube fills
      [0.58, 1] (0.58 = 1/sqrt 3, centre of a face vs a corner).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import numpy as np

from src.descriptors.base import EPS, Descriptor, DescriptorError, hellinger_hist


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class D1RadialDistance(Descriptor):
    """D1: how far surface points are from the centre."""
    # c = area-weighted centre of the surface
    # d_i = |p_i - c| / (largest d), a value in [0, 1]; output = 32-bin histogram of d_i
    # points used: all 2e4 surface samples (pipeline.n_samples)

    name = "d1"
    version = "1"
    metric = "hellinger"
    defaults = {"bins": 32}

    def _setup(self) -> None:
        self.dim = self.n_hist = int(self.params["bins"])

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        # distance of each point from the centre, divided by the largest one (eq. 5)
        d = np.linalg.norm(mesh.samples.points - mesh.centroid, axis=1)
        dmax = float(d.max())
        if dmax <= EPS:
            raise DescriptorError("d1: all samples coincide with the centroid")
        return hellinger_hist(d / dmax, self.dim, 0.0, 1.0)

# ==============================================================================
# END OF FILE: d1_radial_distance.py
# ==============================================================================
# SUMMARY:
#   D1: 32-bin histogram of distance from the centre.
#
# USAGE EXAMPLE:
#   from src.descriptors.d1_radial_distance import D1RadialDistance
#   v = D1RadialDistance().compute(prepared_mesh, np.random.default_rng(0))
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Ignores direction: swinging one arm around the centre does not change D1 (by design).
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
