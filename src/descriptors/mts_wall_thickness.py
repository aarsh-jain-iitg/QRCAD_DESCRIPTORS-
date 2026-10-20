"""
================================================================================
FILE: mts_wall_thickness.py
================================================================================
PURPOSE:
    MTS, wall thickness (PDF Algorithm 15). From each surface point, shoot
    16 rays straight into the material in a small cone and see how far
    each one travels before it comes out the other side. The middle value
    is the wall thickness at that point. Output: histogram of thickness
    (log scale, divided by part size) plus summary numbers.

INPUTS:
    - PreparedMesh (cleaned mesh for the rays, surface sample points, part size D_diam).
    - params: rays (16), cone_deg (30), eps (start offset, 1e-6 of D_diam),
      bins (32), log_lo (-3), log_hi (0), thin (0.01).

OUTPUTS:
    - float32 vector, shape (bins + 8,):
      [thickness histogram on log10 scale -3..0 | mean, spread, skewness,
       25 %, 50 %, 75 % values, share of thin walls, share of rays that missed].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, trimesh (uses embreex for fast rays when installed)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Thickness at a point: each ray's length is projected onto the inward
      direction, values far from the average (more than one standard
      deviation) are dropped, and the middle value of the rest is kept;
      then divided by D_diam (eq. 29).
    - Ray layout (16 = 1 + 5 + 10): one straight in, five tilted 15 deg,
      ten tilted 30 deg. The layout is lined up with the direction toward
      the part's centre, so it turns and mirrors with the part and the
      result does not change.
    - Thin share = fraction of points thinner than 0.01 D_diam. Missed share
      = fraction of rays that hit nothing (0 for a closed part; a high
      value warns of a broken mesh).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import math

import numpy as np

from src.descriptors.base import EPS, Descriptor, hellinger_hist, skewness


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _cone_pattern(n_rays: int, cone_deg: float) -> np.ndarray:
    """
    Ray directions (n_rays, 3) as (along t, along b, inward). One straight
    ray, then a ring at cone/2 with 1/3 of the rest and a ring at cone with
    2/3 of the rest. Each ring starts at angle 0, so the set is the same
    when mirrored across t.
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
    Middle value per row, after dropping values more than one std from the mean.

    Parameters
    ----------
    tau : np.ndarray, shape (n, c)
        Projected ray lengths, NaN where a ray missed.

    Returns
    -------
    np.ndarray, shape (n,)
        Middle values; NaN where every ray in the row missed.
    """
    out = np.full(len(tau), np.nan)
    hit = np.any(np.isfinite(tau), axis=1)
    t = tau[hit]
    mu = np.nanmean(t, axis=1, keepdims=True)
    sd = np.nanstd(t, axis=1, keepdims=True)
    kept = np.where(np.abs(t - mu) <= sd + 1e-15, t, np.nan)   # the value closest to the mean is always kept
    out[hit] = np.nanmedian(kept, axis=1)
    return out


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class MTSWallThickness(Descriptor):
    """MTS: wall thickness histogram from rays shot into the material."""
    # thickness = middle value of (ray length * cos(tilt)) / D_diam, in (0, 1]
    # output = [32-bin hist of log10 thickness on [-3, 0], mean, spread, skewness,
    #           25/50/75 % values, thin share, missed share]
    # points used: all 2e4 surface samples (pipeline.n_samples), 16 rays each (params.rays) = 3.2e5 rays

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
        # surface direction t pointing toward the centre (turns with the part); any flat direction if none
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
        # ray length times cos(tilt); the cosine is the pattern's third component (eq. 29)
        proj = tau.reshape(n, c) * pat[None, :, 2]
        w_hat = _robust_median(proj) / D
        w_hat = w_hat[np.isfinite(w_hat) & (w_hat > 0)]
        if len(w_hat) == 0:
            return np.concatenate([np.zeros(self.n_hist + 7), [f_miss]])

        lw = np.log10(np.maximum(w_hat, EPS))
        h = hellinger_hist(lw, self.n_hist, float(p["log_lo"]), float(p["log_hi"]))
        q1, q2, q3 = np.quantile(lw, [0.25, 0.5, 0.75])
        f_thin = float(np.mean(w_hat < float(p["thin"])))
        # [histogram, mean, spread, skewness, 25/50/75 %, thin share, missed share] (eq. 30)
        return np.concatenate([h, [lw.mean(), lw.std(), skewness(lw), q1, q2, q3, f_thin, f_miss]])

# ==============================================================================
# END OF FILE: mts_wall_thickness.py
# ==============================================================================
# SUMMARY:
#   MTS: log wall-thickness histogram + 8 summary numbers, from ray casting.
#
# USAGE EXAMPLE:
#   from src.descriptors.mts_wall_thickness import MTSWallThickness
#   v = MTSWallThickness().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Without embreex installed, ray casting is much slower.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
