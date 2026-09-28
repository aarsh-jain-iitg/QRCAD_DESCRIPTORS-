"""
================================================================================
FILE: phd_persistent_homology.py
================================================================================
PURPOSE:
    PHD persistent homology of the radial sweep,
    CAD_Descriptors_Explained.pdf, Algorithm 12: lower-star filtration of the
    welded mesh by distance to the symmetry axis, extended persistence
    (GUDHI), three 5x5 persistence images plus two scalars.

INPUTS:
    - PreparedMesh (welded mesh, principal axes, D_diam).
    - params: grid (5), b_max (0.5), p_grid_max (0.25), sigma (0.05),
      p_max (0.1), loop_threshold (0.02), min_persistence (1e-9).

OUTPUTS:
    - float32 vector, shape (3*grid^2 + 2,):
      [img(PD0) | img(PD1) | img(Ext1) | #loops(p > 0.02) | total persistence].
      Each image is L1-normalised and sqrt-mapped (the PDF lists PHD images
      as histogram blocks); an empty diagram gives a zero image.

DEPENDENCIES:
    stdlib      : (none)
    third-party : numpy, gudhi
    internal    : descriptors.base, descriptors.crs_r_ring_profile

PHYSICS / DOMAIN CONTEXT:
    - f(v) = |(x_v - c) x e_s| / D_diam; edges / triangles take the max of
      their vertices (lower star).
    - GUDHI extended_persistence returns [Ordinary, Relative, Extended+,
      Extended-]. Mapping chosen for the PDF's three diagrams:
        PD0  = ordinary dim-0 pairs (pieces),
        PD1  = ordinary + relative dim-1 pairs (loops),
        Ext1 = extended (+ and -) dim-1 pairs (the 2g through-hole loops).
      Each pair maps to (b, p) = (min(birth, death), |death - birth|).
    - I(x,y) = sum w(p) exp(-((x-b)^2+(y-p)^2)/(2 sigma^2)),
      w(p) = min(1, p/p_max) (eq. 26). sigma, p_max and the grid extent are
      not given in the PDF; chosen: b in [0, 0.5] (max radius about an
      axis through the centroid ~ D_diam/2), p in [0, 0.25], sigma = one
      grid cell (0.05 in p), p_max = 0.1.
    - Stability: bottleneck distance <= |f - g|_inf, so mesh noise delta moves
      the diagram by at most delta.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import gudhi
import numpy as np

from src.descriptors.base import Descriptor, hellinger
from src.descriptors.crs_r_ring_profile import axis_distance, symmetry_axis


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _as_bp(pairs: list) -> np.ndarray:
    """(birth, death) list -> array (n, 2) of (b, p) = (min, |d - b|)."""
    # points: none sampled; filtration over all welded mesh vertices, edges, faces
    if not pairs:
        return np.zeros((0, 2))
    a = np.asarray([bd for _, bd in pairs], dtype=np.float64)
    a = a[np.all(np.isfinite(a), axis=1)]
    return np.column_stack([a.min(axis=1), np.abs(a[:, 1] - a[:, 0])])


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class PHDPersistentHomology(Descriptor):
    """PHD: extended persistence images of the radial sweep."""
    # f(p) = |(p - c) x e_s| / D_diam  (lower-star filtration, extended persistence)
    # I(x, y) = sum_(b,p) w(p) exp(-((x - b)^2 + (y - p)^2) / (2 sigma^2))
    # w(p) = min(1, p / p_max),  p = d - b

    name = "phd"
    version = "1"
    metric = "hellinger"
    defaults = {"grid": 5, "b_max": 0.5, "p_grid_max": 0.25, "sigma": 0.05,
                "p_max": 0.1, "loop_threshold": 0.02, "min_persistence": 1e-9}

    def _setup(self) -> None:
        self.n_hist = 3 * int(self.params["grid"]) ** 2
        self.dim = self.n_hist + 2

    def _image(self, bp: np.ndarray) -> np.ndarray:
        """Persistence image eq.(26) on the grid, Hellinger-mapped."""
        g = int(self.params["grid"])
        xs = (np.arange(g) + 0.5) * float(self.params["b_max"]) / g
        ys = (np.arange(g) + 0.5) * float(self.params["p_grid_max"]) / g
        if len(bp) == 0:
            return np.zeros(g * g)
        b, p = bp[:, 0], bp[:, 1]
        w = np.minimum(1.0, p / float(self.params["p_max"]))
        s2 = 2.0 * float(self.params["sigma"]) ** 2
        # eq.(26): I(x, y) = sum w(p) exp(-((x-b)^2 + (y-p)^2) / (2 sigma^2))
        gx = np.exp(-(xs[None, :] - b[:, None]) ** 2 / s2)      # (n, g)
        gy = np.exp(-(ys[None, :] - p[:, None]) ** 2 / s2)      # (n, g)
        img = np.einsum("n,nx,ny->xy", w, gx, gy)
        return hellinger(img.ravel())

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        center, w, V = mesh.axes
        s, _ = symmetry_axis(w)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        f = axis_distance(verts - center, V[:, s]) / mesh.diameter
        edges = np.asarray(mesh.mesh.edges_unique, dtype=np.int64)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        st = gudhi.SimplexTree()
        st.insert_batch(np.arange(len(f))[None, :], f)
        st.insert_batch(edges.T, f[edges].max(axis=1))
        st.insert_batch(faces.T, f[faces].max(axis=1))
        st.extend_filtration()
        ordinary, relative, ext_plus, ext_minus = st.extended_persistence(
            min_persistence=float(self.params["min_persistence"]))

        pd0 = _as_bp([x for x in ordinary if x[0] == 0])
        pd1 = _as_bp([x for x in ordinary + relative if x[0] == 1])
        ext1 = _as_bp([x for x in ext_plus + ext_minus if x[0] == 1])

        loops = np.concatenate([pd1[:, 1], ext1[:, 1]])
        n_loops = float(np.sum(loops > float(self.params["loop_threshold"])))
        total = float(pd0[:, 1].sum() + pd1[:, 1].sum() + ext1[:, 1].sum())
        return np.concatenate([self._image(pd0), self._image(pd1), self._image(ext1),
                               [n_loops, total]])

# ==============================================================================
# END OF FILE: phd_persistent_homology.py
# ==============================================================================
# SUMMARY:
#   PHD: extended persistence of the radial sweep -> 3 persistence images + 2 scalars.
#
# USAGE EXAMPLE:
#   from src.descriptors.phd_persistent_homology import PHDPersistentHomology
#   v = PHDPersistentHomology().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Axis inherited from CRS-R: undefined for isotropic covariance.
#   - [ ] Worst-case O(n_s^3) reduction (not seen on mesh filtrations).
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
