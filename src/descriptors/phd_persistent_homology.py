"""
================================================================================
FILE: phd_persistent_homology.py
================================================================================
PURPOSE:
    PHD, growing the part outward from its axis (PDF Algorithm 12).
    Imagine a cylinder around the turning axis that slowly gets wider, and
    keep only the surface inside it. As it grows, pieces appear and join
    and loops form and close. We record when each feature appears and how
    long it lasts. Long-lasting features are real (a flange, a groove);
    short ones are noise. The result is turned into three small 5 x 5
    pictures plus two numbers.

INPUTS:
    - PreparedMesh (cleaned mesh, main axes, part size D_diam).
    - params: grid (5), b_max (0.5), p_grid_max (0.25), sigma (0.05),
      p_max (0.1), loop_threshold (0.02), min_persistence (1e-9).

OUTPUTS:
    - float32 vector, shape (3*grid^2 + 2,):
      [picture: pieces | picture: loops | picture: through-hole loops |
       #loops lasting > 0.02 | total lifetime of all features].
      Each picture is scaled to sum 1 and square-rooted, like a histogram;
      an empty one is all zeros.

DEPENDENCIES:
    stdlib      : (none)
    third-party : numpy, gudhi
    internal    : descriptors.base, descriptors.crs_r_ring_profile

PHYSICS / DOMAIN CONTEXT:
    - Each corner's "time" = its distance from the turning axis / D_diam;
      an edge or triangle appears when its last corner does.
    - GUDHI tracks the features and returns four lists. We use them as:
        pieces            = first list, separate pieces,
        loops             = first + second lists, loops,
        through-hole loops = third + fourth lists, loops that go round a hole.
      Each feature becomes (start time, lifetime).
    - Each picture places a soft blob at every feature's (start, lifetime),
      bigger for longer-lived ones (eq. 26). Blob size, weights and picture
      range are our choices (not in the PDF): start in [0, 0.5], lifetime in
      [0, 0.25], blob width 0.05, full weight from lifetime 0.1 up.
    - Stable: moving the mesh points by a small amount moves every feature
      by at most that amount.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import gudhi
import numpy as np

from src.descriptors.base import Descriptor, hellinger
from src.descriptors.crs_r_ring_profile import axis_distance, symmetry_axis


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _as_bp(pairs: list) -> np.ndarray:
    """List of (start, end) times -> array (n, 2) of (earlier time, lifetime)."""
    # points used: none sampled; every mesh corner, edge and triangle
    if not pairs:
        return np.zeros((0, 2))
    a = np.asarray([bd for _, bd in pairs], dtype=np.float64)
    a = a[np.all(np.isfinite(a), axis=1)]
    return np.column_stack([a.min(axis=1), np.abs(a[:, 1] - a[:, 0])])


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class PHDPersistentHomology(Descriptor):
    """PHD: pictures of how long shape features last as the part grows from its axis."""
    # time of a point = distance to turning axis / D_diam
    # picture(x, y) = sum over features of weight * exp(-((x - start)^2 + (y - life)^2) / (2 sigma^2))
    # weight = min(1, life / p_max)

    name = "phd"
    version = "1"
    metric = "hellinger"
    defaults = {"grid": 5, "b_max": 0.5, "p_grid_max": 0.25, "sigma": 0.05,
                "p_max": 0.1, "loop_threshold": 0.02, "min_persistence": 1e-9}

    def _setup(self) -> None:
        self.n_hist = 3 * int(self.params["grid"]) ** 2
        self.dim = self.n_hist + 2

    def _image(self, bp: np.ndarray) -> np.ndarray:
        """Turn a feature list into a small picture (eq. 26), scaled and square-rooted."""
        g = int(self.params["grid"])
        xs = (np.arange(g) + 0.5) * float(self.params["b_max"]) / g
        ys = (np.arange(g) + 0.5) * float(self.params["p_grid_max"]) / g
        if len(bp) == 0:
            return np.zeros(g * g)
        b, p = bp[:, 0], bp[:, 1]
        w = np.minimum(1.0, p / float(self.params["p_max"]))
        s2 = 2.0 * float(self.params["sigma"]) ** 2
        # one soft blob per feature, weighted by its lifetime (eq. 26)
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
#   PHD: 3 small feature pictures + 2 numbers from growing the part out of its axis.
#
# USAGE EXAMPLE:
#   from src.descriptors.phd_persistent_homology import PHDPersistentHomology
#   v = PHDPersistentHomology().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Uses the CRS-R axis: no single axis for cube- or sphere-like parts.
#   - [ ] Worst case is very slow (cube of the mesh size); not seen on real meshes.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
