"""
================================================================================
FILE: mts_wall_thickness.py
================================================================================
PURPOSE:
    MTS wall thickness spectrum (Shape Diameter Function),
    CAD_Descriptors_Explained.pdf, Algorithm 15: from each surface sample cast
    a cone of rays into the material, take a robust median of the
    cosine-projected first-hit distances as the local wall thickness, and
    summarise log10(thickness / D_diam).

INPUTS:
    - PreparedMesh (welded mesh for ray casting, default samples, D_diam).
    - params: rays (16), cone_deg (30), eps (1e-6 of D_diam), bins (32),
      log_lo (-3), log_hi (0), thin (0.01).

OUTPUTS:
    - float32 vector, shape (bins + 8,):
      [Hellinger hist of log10 w^ on [-3, 0] | mu, sigma, gamma_1, Q1, Q2, Q3,
       f_thin, f_miss].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, trimesh (Embree via embreex when installed)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - w_i = median_j (tau_ij cos angle(d_ij, -n_i)) after dropping values more
      than one standard deviation from the mean; w^ = w / D_diam (eq. 29).
    - Ray pattern (16 = 1 + 5 + 10): one ray along -n, five at 15 deg and
      ten at 30 deg, at azimuths symmetric about the tangent direction
      that points toward the centroid. The pattern moves with the part and
      is mirror-symmetric, so rotations and reflections leave it unchanged
      (a fixed-world pattern would not be).
    - f_thin = Pr[w^ < 0.01]; f_miss = share of rays that hit nothing
      (0 for a watertight part, a mesh-quality flag).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from src.descriptors.base import EPS, Descriptor, hellinger_hist, skewness


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _cone_pattern(n_rays: int, cone_deg: float) -> np.ndarray:
    """
    Unit directions (n_rays, 3) in a local frame (t, b, -n) -> components
    (along t, along b, along -n). Layout: 1 axial ray, then rings at
    cone/2 and cone with 1/3 and 2/3 of the remaining rays, azimuths
    starting at 0 so the set is symmetric under b -> -b.
    """
    rest = n_rays - 1
    inner = rest // 3
    rings = [(0.0, 1), (cone_deg / 2.0, inner), (cone_deg, rest - inner)]
    dirs = []
    for polar_deg, count in rings:
        th = math.radians(polar_deg)
        for k in range(count):
            az = 2.0 * math.pi * k / max(count, 1)
            dirs.append((math.sin(th) * math.cos(az), math.sin(th) * math.sin(az), math.cos(th)))
    return np.asarray(dirs)


def _robust_median(tau: np.ndarray) -> np.ndarray:
    """
    Row-wise median of finite entries after rejecting |x - mean| > std.

    Parameters
    ----------
    tau : np.ndarray, shape (n, c)
        Projected hit distances, NaN for misses.

    Returns
    -------
    np.ndarray, shape (n,)
        Robust medians; NaN where a row has no hits.
    """
    out = np.full(len(tau), np.nan)
    hit = np.any(np.isfinite(tau), axis=1)
    t = tau[hit]
    mu = np.nanmean(t, axis=1, keepdims=True)
    sd = np.nanstd(t, axis=1, keepdims=True)
    kept = np.where(np.abs(t - mu) <= sd + 1e-15, t, np.nan)   # the mean-nearest value always survives
    out[hit] = np.nanmedian(kept, axis=1)
    return out


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class MTSWallThickness(Descriptor):
    """MTS: log wall-thickness spectrum from the shape diameter function."""
    # w_i = median_j(tau_ij cos angle(d_ij, -n_i)),  w^_i = w_i / D_diam in (0, 1]
    # F_MTS = [hist32(log10 w^; [-3, 0]), mu, sigma, gamma_1, Q1, Q2, Q3 of log10 w^,
    #          f_thin, f_miss]
    # points: all n = 2e4 surface samples (pipeline.n_samples), 16 rays each (params.rays) = 3.2e5 rays

    name = "mts"
    version = "1"
    metric = "hellinger"
    defaults = {"rays": 16, "cone_deg": 30.0, "eps": 1e-6, "bins": 32,
                "log_lo": -3.0, "log_hi": 0.0, "thin": 0.01}

    def _setup(self) -> None:
        self.n_hist = int(self.params["bins"])
        self.dim = self.n_hist + 8

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        p = self.params
        s = mesh.samples
        P, Nn = s.points, s.normals
        n = len(P)
        D = mesh.diameter
        # tangent t toward the centroid (rotation-covariant), fallback any perpendicular
        t = (mesh.centroid - P)
        t -= np.einsum("ij,ij->i", t, Nn)[:, None] * Nn
        tn = np.linalg.norm(t, axis=1)
        fallback = np.cross(Nn, np.where(np.abs(Nn[:, [0]]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
        t = np.where((tn > 1e-9 * D)[:, None], t, fallback)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        b = np.cross(Nn, t)
        pat = _cone_pattern(int(p["rays"]), float(p["cone_deg"]))          # (c, 3)
        c = len(pat)
        dirs = (pat[None, :, 0:1] * t[:, None] + pat[None, :, 1:2] * b[:, None]
                - pat[None, :, 2:3] * Nn[:, None]).reshape(-1, 3)
        origins = np.repeat(P - float(p["eps"]) * D * Nn, c, axis=0)

        locs, ray_id, _ = mesh.mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        tau = np.full(n * c, np.nan)
        tau[ray_id] = np.linalg.norm(locs - origins[ray_id], axis=1)
        f_miss = float(np.isnan(tau).mean())
        # eq.(29): tau_ij * cos angle(d_ij, -n_i); the cosine is the pattern's third component
        proj = tau.reshape(n, c) * pat[None, :, 2]
        w_hat = _robust_median(proj) / D
        w_hat = w_hat[np.isfinite(w_hat) & (w_hat > 0)]
        if len(w_hat) == 0:
            return np.concatenate([np.zeros(self.n_hist + 7), [f_miss]])

        lw = np.log10(np.maximum(w_hat, EPS))
        h = hellinger_hist(lw, self.n_hist, float(p["log_lo"]), float(p["log_hi"]))
        q1, q2, q3 = np.quantile(lw, [0.25, 0.5, 0.75])
        f_thin = float(np.mean(w_hat < float(p["thin"])))
        # eq.(30): [hist, mu, sigma, gamma_1, Q1, Q2, Q3, f_thin, f_miss]
        return np.concatenate([h, [lw.mean(), lw.std(), skewness(lw), q1, q2, q3, f_thin, f_miss]])

# ==============================================================================
# END OF FILE: mts_wall_thickness.py
# ==============================================================================
# SUMMARY:
#   MTS: log10 wall-thickness histogram + 8 summary values via ray casting.
#
# USAGE EXAMPLE:
#   from src.descriptors.mts_wall_thickness import MTSWallThickness
#   v = MTSWallThickness().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Without embreex, trimesh falls back to a much slower ray engine.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
