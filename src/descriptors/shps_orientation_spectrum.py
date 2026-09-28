"""
================================================================================
FILE: shps_orientation_spectrum.py
================================================================================
PURPOSE:
    SHPS orientation spectrum, CAD_Descriptors_Explained.pdf, Algorithm 5:
    rotation-invariant spherical-harmonic band powers P~_l of the
    area-weighted Extended Gaussian Image, computed exactly from the faces,
    l = 2..L.

INPUTS:
    - PreparedMesh (face normals, face areas).
    - params: L (band limit, PDF: 16), chunk (faces per SH evaluation batch).

OUTPUTS:
    - float32 vector, shape (L-1,) = [P~_2, ..., P~_L] (15 values at L=16).

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, scipy.special.sph_harm_y_all
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - c_lm = sum_t w_t Y_lm(n_t), P~_l = 4 pi/(2l+1) sum_m |c_lm|^2 in [0,1].
      By the addition theorem this equals sum_s sum_t w_s w_t P_l(n_s . n_t),
      so it depends only on angles between normals: rotation and reflection
      invariant; normals ignore shift and scale.
    - sph_harm_y_all evaluates all (l, m) by recurrence, which the
      Complexity PDF requires (27x faster than per-harmonic calls).
      Complex and real harmonics give the same band power.
    - Reference: cube P~_4 = 0.583, P~_6 = 0.125, P~_8 = 0.516; ball ~0.
    - PDF prose mentions l = 2,4,6,...; the formula and the stated length 15
      use every l = 2..16, which is what is implemented.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import sph_harm_y_all

from src.descriptors.base import Descriptor


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class SHPSOrientationSpectrum(Descriptor):
    """SHPS: spherical-harmonic power spectrum of the EGI."""
    # c_lm = sum_t w_t Y_lm(n_t),  P~_l = 4 pi / (2l + 1) sum_m c_lm^2
    #      = sum_s sum_t w_s w_t P_l(n_s . n_t) in [0, 1];  F = [P~_2 .. P~_16]
    # points: none sampled; exact sum over all mesh faces (normals, areas)

    name = "shps"
    version = "1"
    metric = "l2"
    defaults = {"L": 16, "chunk": 4096}

    def _setup(self) -> None:
        self.dim = int(self.params["L"]) - 1
        self.n_hist = 0

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        L = int(self.params["L"])
        n = mesh.face_normals
        w = mesh.face_areas / mesh.area
        theta = np.arccos(np.clip(n[:, 2], -1.0, 1.0))      # polar angle
        phi = np.arctan2(n[:, 1], n[:, 0])                  # azimuth
        c = np.zeros((L + 1, 2 * L + 1), dtype=np.complex128)
        step = int(self.params["chunk"])
        for s in range(0, len(w), step):
            Y = sph_harm_y_all(L, L, theta[s:s + step], phi[s:s + step])
            # eq.(11): c_lm = sum_t w_t Y_lm(n_t)
            c += Y @ w[s:s + step]
        ell = np.arange(L + 1)
        # eq.(11): P~_l = 4 pi / (2l+1) * sum_m |c_lm|^2
        P = 4.0 * math.pi / (2 * ell + 1) * (np.abs(c) ** 2).sum(axis=1)
        return P[2:]

# ==============================================================================
# END OF FILE: shps_orientation_spectrum.py
# ==============================================================================
# SUMMARY:
#   SHPS: exact EGI spherical-harmonic band powers l = 2..16.
#
# USAGE EXAMPLE:
#   from src.descriptors.shps_orientation_spectrum import SHPSOrientationSpectrum
#   v = SHPSOrientationSpectrum().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Knows which directions occur, not where.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
