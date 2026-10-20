"""
================================================================================
FILE: shps_orientation_spectrum.py
================================================================================
PURPOSE:
    SHPS, which way the surfaces face (PDF Algorithm 5). Collects the
    facing direction (normal) of every triangle, weighted by its area, as
    points on a sphere. It then measures how much of that pattern repeats
    at each level of detail l = 2..L. A box (6 directions, 90 deg apart)
    and a ball (all directions) give very different numbers.

INPUTS:
    - PreparedMesh (triangle normals, triangle areas).
    - params: L (highest level of detail, 16), chunk (triangles per batch).

OUTPUTS:
    - float32 vector, shape (L-1,): one strength per level l = 2..L (15 values at L=16).

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, scipy.special.sph_harm_y_all, torch (optional, device = cuda)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Uses spherical harmonics (the sphere version of a Fourier series).
      The strength at level l only depends on the angles between the
      normals, so turning or mirroring the part does not change it;
      normals ignore position and size.
    - Each strength is between 0 and 1. Examples: cube l=4: 0.583,
      l=6: 0.125, l=8: 0.516; ball: all ~0.
    - Exact: every triangle is used, no random sampling.
    - sph_harm_y_all computes all levels in one pass (27x faster than one
      call per harmonic).
    - GPU version (device = cuda) does the same maths with a step-by-step
      formula in torch; matches the CPU result to ~1e-12.
    - The PDF text mentions only even l (2, 4, 6, ...) but its formula and
      its stated length of 15 use every l = 2..16; we use every l.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import sph_harm_y_all

from src.descriptors.base import Descriptor, resolve_torch_device


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _band_powers_torch(n: np.ndarray, w: np.ndarray, L: int, dev) -> np.ndarray:
    """
    GPU version: strength of each level l = 0..L, float64.

    Parameters
    ----------
    n : np.ndarray, shape (F, 3)
        Unit triangle normals [-].
    w : np.ndarray, shape (F,)
        Area weights, summing to 1 [-].
    L : int
        Highest level.
    dev : torch.device

    Returns
    -------
    np.ndarray, shape (L+1,)
        Strength of level l = 4 pi / (2l+1) * sum over m of |c_lm|^2.
    """
    import torch
    nt = torch.as_tensor(n, dtype=torch.float64, device=dev)
    wt = torch.as_tensor(w, dtype=torch.float64, device=dev)
    x = nt[:, 2].clamp(-1.0, 1.0)                           # cos(theta)
    sx = torch.sqrt((1.0 - x * x).clamp_min(0.0))           # sin(theta)
    phi = torch.atan2(nt[:, 1], nt[:, 0])
    power = torch.zeros(L + 1, dtype=torch.float64, device=dev)
    pmm = torch.full_like(x, 1.0 / math.sqrt(4.0 * math.pi))  # starting value for l = m = 0
    for m in range(L + 1):
        if m > 0:
            pmm = math.sqrt((2 * m + 1) / (2 * m)) * sx * pmm
        rows = [pmm]
        if m < L:
            rows.append(math.sqrt(2 * m + 3) * x * pmm)
        for ell in range(m + 2, L + 1):
            # next level from the two before it: P_l = a (x P_{l-1} - b P_{l-2})
            a = math.sqrt((4 * ell * ell - 1) / (ell * ell - m * m))
            b = math.sqrt(((ell - 1) ** 2 - m * m) / (4 * (ell - 1) ** 2 - 1))
            rows.append(a * (x * rows[-1] - b * rows[-2]))
        Pm = torch.stack(rows, dim=1)                       # (F, L-m+1), l = m..L
        # |c_lm|^2 from the cos and sin parts; m and -m give the same amount (eq. 11)
        c2 = ((wt * torch.cos(m * phi)) @ Pm) ** 2 + ((wt * torch.sin(m * phi)) @ Pm) ** 2
        power[m:] += c2 if m == 0 else 2.0 * c2
    ell = torch.arange(L + 1, dtype=torch.float64, device=dev)
    return (4.0 * math.pi / (2 * ell + 1) * power).cpu().numpy()


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class SHPSOrientationSpectrum(Descriptor):
    """SHPS: how strongly the surface directions repeat at each level of detail."""
    # c_lm = sum over triangles of (area weight * Y_lm(normal))
    # strength_l = 4 pi / (2l + 1) * sum over m of c_lm^2, in [0, 1];  output = [l = 2 .. 16]
    # points used: none sampled; every triangle of the mesh

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
        dev = resolve_torch_device(self.device)
        if dev is not None:
            return _band_powers_torch(n, w, L, dev)[2:]
        theta = np.arccos(np.clip(n[:, 2], -1.0, 1.0))      # angle from the z axis
        phi = np.arctan2(n[:, 1], n[:, 0])                  # angle around the z axis
        c = np.zeros((L + 1, 2 * L + 1), dtype=np.complex128)
        step = int(self.params["chunk"])
        for s in range(0, len(w), step):
            Y = sph_harm_y_all(L, L, theta[s:s + step], phi[s:s + step])
            # add up area weight * harmonic value over the triangles (eq. 11)
            c += Y @ w[s:s + step]
        ell = np.arange(L + 1)
        # strength of each level (eq. 11)
        P = 4.0 * math.pi / (2 * ell + 1) * (np.abs(c) ** 2).sum(axis=1)
        return P[2:]

# ==============================================================================
# END OF FILE: shps_orientation_spectrum.py
# ==============================================================================
# SUMMARY:
#   SHPS: 15 numbers describing which directions the surfaces face.
#
# USAGE EXAMPLE:
#   from src.descriptors.shps_orientation_spectrum import SHPSOrientationSpectrum
#   v = SHPSOrientationSpectrum().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Knows which directions the surfaces face, not where those surfaces are.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-04  optional GPU version (device = cuda)
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
