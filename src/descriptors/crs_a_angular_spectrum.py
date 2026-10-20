"""
================================================================================
FILE: crs_a_angular_spectrum.py
================================================================================
PURPOSE:
    CRS-A, repeating patterns around the axis (PDF Algorithm 13). Walks
    around the turning axis and counts how much surface sits at each angle,
    separately in four rings from the axis outward. It then checks which
    repeat counts are present: a gear with 20 teeth lights up value 20,
    a flange with 6 bolt holes lights up value 6.

INPUTS:
    - PreparedMesh: a larger set of 2e5 surface points, plus the main axes
      and part size D_diam.
    - params: n_pool (2e5), bands (4 rings), n_phi (256 angle bins), harmonics (32).

OUTPUTS:
    - float32 vector, shape (bands * harmonics,): for each ring, the
      strength of every repeat count 1..32 (relative to the total).

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, torch (optional, device = cuda)
    internal    : descriptors.base, descriptors.crs_r_ring_profile

PHYSICS / DOMAIN CONTEXT:
    - Angle around the axis = atan2 of the point's position along the
      other two axes. Turning the part about its axis just shifts all
      angles; flipping an axis mirrors them. Neither changes the repeat
      strengths (from a Fourier transform, ignoring phase).
    - The four rings split [0, largest distance from the axis] evenly.
    - Points are spread evenly over the surface, so plain counts already
      weight by area.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from src.descriptors.base import EPS, Descriptor, resolve_torch_device
from src.descriptors.crs_r_ring_profile import axis_distance, symmetry_axis


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _phi_rho_torch(X: np.ndarray, eb: np.ndarray, ec: np.ndarray, es: np.ndarray,
                   dev) -> tuple[np.ndarray, np.ndarray]:
    """GPU version: (angle around axis [rad], distance to axis e_s [L]) of points X (N, 3)."""
    import torch
    Xt = torch.as_tensor(X, dtype=torch.float64, device=dev)
    E = torch.as_tensor(np.stack([eb, ec, es]), dtype=torch.float64, device=dev)
    phi = torch.atan2(Xt @ E[0], Xt @ E[1])
    rho = torch.linalg.vector_norm(torch.linalg.cross(Xt, E[2].expand_as(Xt)), dim=1)
    return phi.cpu().numpy(), rho.cpu().numpy()


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class CRSAAngularSpectrum(Descriptor):
    """CRS-A: strength of each repeat count around the axis, per ring."""
    # angle of each point around the turning axis -> 256-bin histogram per ring
    # Fourier transform of that histogram -> strength of repeat count m
    # output = [strength(m) / strength(0)] for 4 rings x m = 1..32
    # points used: 2e5 surface samples (params.n_pool, larger than the default 2e4)

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
        dev = resolve_torch_device(self.device)
        if dev is None:
            phi = np.arctan2(X @ V[:, b_idx], X @ V[:, c_idx])
            rho = axis_distance(X, V[:, s])
        else:
            phi, rho = _phi_rho_torch(X, V[:, b_idx], V[:, c_idx], V[:, s], dev)
        nb, nphi, M = int(self.params["bands"]), int(self.params["n_phi"]), int(self.params["harmonics"])
        band = np.minimum((rho / max(float(rho.max()), EPS) * nb).astype(np.int64), nb - 1)
        out = np.zeros((nb, M))
        for q in range(nb):
            a, _ = np.histogram(phi[band == q], bins=nphi, range=(-math.pi, math.pi))
            A = np.abs(np.fft.rfft(a))
            # repeat strengths 1..M divided by the total (eq. 27)
            if A[0] > 0:
                out[q] = A[1:M + 1] / A[0]
        return out.ravel()

# ==============================================================================
# END OF FILE: crs_a_angular_spectrum.py
# ==============================================================================
# SUMMARY:
#   CRS-A: 4 rings x 32 repeat strengths around the turning axis.
#
# USAGE EXAMPLE:
#   from src.descriptors.crs_a_angular_spectrum import CRSAAngularSpectrum
#   v = CRSAAngularSpectrum().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Uses the CRS-R axis: no single axis for cube- or sphere-like parts.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-04  optional GPU version (device = cuda)
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
