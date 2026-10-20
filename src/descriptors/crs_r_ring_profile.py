"""
================================================================================
FILE: crs_r_ring_profile.py
================================================================================
PURPOSE:
    CRS-R, distance from the part's main axes (PDF Algorithm 10). Finds the
    part's three main axes (its natural length / width / height
    directions) and picks the one most likely to be a turning axis (for
    a shaft or a flange, the axis it spins around). Then it histograms how
    far every surface point is from that axis (64 bins) and from each of
    the other two axes (32 bins each). Turned parts show sharp peaks at
    each radius they have.
    This file also has symmetry_axis(), the axis choice reused by PHD and CRS-A.

INPUTS:
    - PreparedMesh (surface sample points, main axes, part size D_diam).
    - params: bins_axis (64), bins_other (32), rho_max (0.75).

OUTPUTS:
    - float32 vector, shape (bins_axis + 2*bins_other,): three histograms,
      each scaled on its own.

DEPENDENCIES:
    stdlib      : (none)
    third-party : numpy
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Axis choice (eq. 22): a round (turned) part is equally wide in two
      directions, so the axis whose other two directions have the most
      similar spread is taken as the turning axis.
    - Distance to an axis does not depend on which way the axis points.
    - Range [0, 0.75]: no point of a part can be more than 0.75 D_diam
      from its centre, so nothing is cut off (the PDF gives no range).
    - Known issue: when the part is equally spread in all three directions
      (cube, sphere) there is no single best axis, and the result depends
      on which axis the solver happens to return. See CLAUDE.md.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import numpy as np

from src.descriptors.base import Descriptor, hellinger_hist


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def symmetry_axis(eigvals: np.ndarray) -> tuple[int, list[int]]:
    """
    Pick the turning axis (eq. 22) and return the other two in order.

    Parameters
    ----------
    eigvals : np.ndarray, shape (3,)
        Spread of the part along each main axis, largest first [L^2].

    Returns
    -------
    s : int
        Index of the turning axis.
    others : list[int]
        The other two indices, largest spread first.
    """
    lam1 = max(float(eigvals[0]), 1e-300)
    gaps = [abs(eigvals[(a + 1) % 3] - eigvals[(a + 2) % 3]) / lam1 for a in range(3)]
    s = int(np.argmin(gaps))
    return s, [a for a in range(3) if a != s]


def axis_distance(X: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Distance of points X (N,3), measured from the centre, to the line along unit e [L]."""
    # points used: all 2e4 surface samples (pipeline.n_samples)
    return np.linalg.norm(np.cross(X, e), axis=1)


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class CRSRRingProfile(Descriptor):
    """CRS-R: histograms of distance from the three main axes."""
    # s = turning axis (the one whose other two spreads are most alike)
    # rho = distance from point to axis / D_diam
    # output = [64-bin hist (turning axis), 32-bin hist (axis 2), 32-bin hist (axis 3)]

    name = "crs_r"
    version = "1"
    metric = "hellinger"
    defaults = {"bins_axis": 64, "bins_other": 32, "rho_max": 0.75}

    def _setup(self) -> None:
        self.dim = self.n_hist = int(self.params["bins_axis"]) + 2 * int(self.params["bins_other"])

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        center, w, V = mesh.axes
        s, others = symmetry_axis(w)
        X = mesh.samples.points - center
        hi = float(self.params["rho_max"])
        parts = []
        for a, bins in ((s, self.params["bins_axis"]), (others[0], self.params["bins_other"]),
                        (others[1], self.params["bins_other"])):
            # distance from each point to this axis, divided by part size
            rho = axis_distance(X, V[:, a]) / mesh.diameter
            parts.append(hellinger_hist(rho, int(bins), 0.0, hi))
        return np.concatenate(parts)

# ==============================================================================
# END OF FILE: crs_r_ring_profile.py
# ==============================================================================
# SUMMARY:
#   CRS-R: 64 + 32 + 32 bin distance-from-axis histograms; symmetry_axis() helper.
#
# USAGE EXAMPLE:
#   from src.descriptors.crs_r_ring_profile import CRSRRingProfile, symmetry_axis
#   v = CRSRRingProfile().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] No single axis for parts equally spread in all directions (cube, sphere); see header.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
