"""
================================================================================
FILE: crs_r_ring_profile.py
================================================================================
PURPOSE:
    CRS-R concentric ring profile, CAD_Descriptors_Explained.pdf, Algorithm
    10: histograms of the distance from each surface sample to the part's own
    symmetry axis (64 bins) and to the two other principal axes (32 bins
    each). Also owns symmetry_axis(), the eq. (22) axis rule reused by PHD and
    CRS-A.

INPUTS:
    - PreparedMesh (default sample pool, principal axes, D_diam).
    - params: bins_axis (64), bins_other (32), rho_max (0.75).

OUTPUTS:
    - float32 vector, shape (bins_axis + 2*bins_other,): three Hellinger-mapped
      profiles, each L1-normalised on its own.

DEPENDENCIES:
    stdlib      : (none)
    third-party : numpy
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Axis choice eq.(22): s = argmin_a |lambda_b - lambda_c| / lambda_1; a body
      of revolution has two equal eigenvalues and s is its axis.
    - rho_i^(a) = |(p_i - c) x e_a| / D_diam: independent of the axis sign and
      of the in-plane eigenvector choice for a body of revolution.
    - Range: the distance from the centroid to any point of the part is at
      most 3/4 D_diam in R^3 (Jung-type centroid bound), so [0, 0.75]
      loses nothing; the PDF gives no range.
    - Known ambiguity: when all three eigenvalues are (nearly) equal (cube,
      sphere, regular solids) eq.(22) does not pick a unique axis; the
      profile then depends on the eigen-solver's choice. Documented in
      CLAUDE.md.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import numpy as np

from src.descriptors.base import Descriptor, hellinger_hist


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def symmetry_axis(eigvals: np.ndarray) -> tuple[int, list[int]]:
    """
    Pick the symmetry axis index per eq.(22) and the other two in eigen order.

    Parameters
    ----------
    eigvals : np.ndarray, shape (3,)
        Covariance eigenvalues in descending order [L^2].

    Returns
    -------
    s : int
        Index of the symmetry axis.
    others : list[int]
        The two remaining indices, in eigenvalue order.
    """
    lam1 = max(float(eigvals[0]), 1e-300)
    gaps = [abs(eigvals[(a + 1) % 3] - eigvals[(a + 2) % 3]) / lam1 for a in range(3)]
    s = int(np.argmin(gaps))
    return s, [a for a in range(3) if a != s]


def axis_distance(X: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Distance of centred points X (N,3) to the line through 0 along unit e [L]."""
    # points: all n = 2e4 surface samples (pipeline.n_samples)
    return np.linalg.norm(np.cross(X, e), axis=1)


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class CRSRRingProfile(Descriptor):
    """CRS-R: distance-to-axis profiles about the principal axes."""
    # s = argmin_a |lambda_b - lambda_c| / lambda_1  ({a, b, c} = {1, 2, 3})
    # rho_i^(a) = |(p_i - c) x e_a| / D_diam;  F = [hist64(rho^(s)), hist32(rho^(b)), hist32(rho^(c))]
    # turned part: R(r) = sum_i 2 pi r_i h_i delta(r - r_i) + 2 pi r m(r)

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
            # rho^(a) = |(p - c) x e_a| / D_diam
            rho = axis_distance(X, V[:, a]) / mesh.diameter
            parts.append(hellinger_hist(rho, int(bins), 0.0, hi))
        return np.concatenate(parts)

# ==============================================================================
# END OF FILE: crs_r_ring_profile.py
# ==============================================================================
# SUMMARY:
#   CRS-R: 64 + 32 + 32 distance-to-axis profiles; symmetry_axis() helper.
#
# USAGE EXAMPLE:
#   from src.descriptors.crs_r_ring_profile import CRSRRingProfile, symmetry_axis
#   v = CRSRRingProfile().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Axis undefined for isotropic covariance (cube, sphere); see header.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
