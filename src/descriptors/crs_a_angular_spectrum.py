"""
================================================================================
FILE: crs_a_angular_spectrum.py
================================================================================
PURPOSE:
    CRS-A angular pattern spectrum, CAD_Descriptors_Explained.pdf, Algorithm
    13: azimuth histograms about the symmetry axis in four radial bands,
    reduced to Fourier magnitudes |a_q(m)| / |a_q(0)|, m = 1..32, so N-fold
    patterns (teeth, bolt circles) show as harmonic N.

INPUTS:
    - PreparedMesh: its own larger pool (definition: n ~ 2e5) plus the
      principal axes / D_diam of the default pool.
    - params: n_pool (2e5), bands (4), n_phi (256), harmonics (32).

OUTPUTS:
    - float32 vector, shape (bands * harmonics,).

DEPENDENCIES:
    stdlib      : math
    third-party : numpy
    internal    : descriptors.base, descriptors.crs_r_ring_profile

PHYSICS / DOMAIN CONTEXT:
    - phi_i = atan2((p_i - c).e_b, (p_i - c).e_c); a rotation about e_s shifts
      phi, multiplying a^(m) by a unit phase; flipping e_b or e_c maps
      phi -> -phi or phi + pi; magnitudes are unchanged in every case.
    - Bands split [0, max rho] of the distance to e_s into equal parts.
    - Samples are area-uniform, so plain counts are area weights.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from src.descriptors.base import EPS, Descriptor
from src.descriptors.crs_r_ring_profile import axis_distance, symmetry_axis


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class CRSAAngularSpectrum(Descriptor):
    """CRS-A: per-band azimuthal Fourier magnitudes."""
    # phi_i = atan2((p_i - c) . e_b, (p_i - c) . e_c),  a_q(j) = hist256(phi) in radius band q
    # a^_q(m) = sum_{j=0}^{N_phi-1} a_q(j) exp(-2 pi i m j / N_phi)
    # F_CRS-A = [|a^_q(m)| / |a^_q(0)|]_{q=1..4, m=1..32}
    # points: n = 2e5 surface samples (params.n_pool, larger pool than the default 2e4)

    name = "crs_a"
    version = "1"
    metric = "l2"
    defaults = {"n_pool": 200_000, "bands": 4, "n_phi": 256, "harmonics": 32}

    def _setup(self) -> None:
        self.dim = int(self.params["bands"]) * int(self.params["harmonics"])
        self.n_hist = 0

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        center, w, V = mesh.axes
        s, (b_idx, c_idx) = symmetry_axis(w)
        X = mesh.sample_pool(int(self.params["n_pool"])).points - center
        phi = np.arctan2(X @ V[:, b_idx], X @ V[:, c_idx])
        rho = axis_distance(X, V[:, s])
        nb, nphi, M = int(self.params["bands"]), int(self.params["n_phi"]), int(self.params["harmonics"])
        band = np.minimum((rho / max(float(rho.max()), EPS) * nb).astype(np.int64), nb - 1)
        out = np.zeros((nb, M))
        for q in range(nb):
            a, _ = np.histogram(phi[band == q], bins=nphi, range=(-math.pi, math.pi))
            A = np.abs(np.fft.rfft(a))
            # eq.(27): |a_q(m)| / |a_q(0)|, m = 1..M
            if A[0] > 0:
                out[q] = A[1:M + 1] / A[0]
        return out.ravel()

# ==============================================================================
# END OF FILE: crs_a_angular_spectrum.py
# ==============================================================================
# SUMMARY:
#   CRS-A: 4 bands x 32 azimuthal harmonic magnitudes about the symmetry axis.
#
# USAGE EXAMPLE:
#   from src.descriptors.crs_a_angular_spectrum import CRSAAngularSpectrum
#   v = CRSAAngularSpectrum().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Axis inherited from CRS-R: undefined for isotropic covariance.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
