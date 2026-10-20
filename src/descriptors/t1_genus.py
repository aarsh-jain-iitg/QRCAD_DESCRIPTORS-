"""
================================================================================
FILE: t1_genus.py
================================================================================
PURPOSE:
    T1, counting pieces, holes and hollows (PDF Algorithm 11). Four counts:
      - how many separate solid pieces the file holds,
      - how many through-holes there are (a washer has 1, a plate with
        4 bolt holes has 4), stored as log(1 + holes),
      - how many sealed hollow spaces are inside (closed cavities),
      - how many open boundary loops the surface has (0 for a closed part).

INPUTS:
    - PreparedMesh (cleaned mesh with consistent outward normals).

OUTPUTS:
    - float32 vector, shape (4,): [#pieces, log(1+holes), #hollows, #open loops].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, scipy.sparse
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Holes come from counting corners (V), edges (E) and triangles (F) of
      each connected surface: holes = (2 - (V - E + F) - open loops) / 2
      (eq. 24). Never below 0, even for broken meshes.
    - A connected surface is a set of triangles joined through shared edges.
    - Each surface's enclosed volume tells outside skin (positive) from the
      wall of an inner hollow (negative). Near-zero volumes (open sheets)
      count as pieces.
    - Pure counts: moving, turning, mirroring or scaling changes nothing.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.descriptors.base import Descriptor

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
VOLUME_TOL = 1e-9        # [-] |volume| / area^(3/2) below this means an open sheet


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _components(n: int, pairs: np.ndarray) -> tuple[int, np.ndarray]:
    """Group n items into connected groups, given linked pairs (E,2)."""
    # points used: none sampled; counts over all mesh corners, edges and triangles
    if len(pairs) == 0:
        return n, np.arange(n)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class T1Genus(Descriptor):
    """T1: pieces, through-holes, hollows, open loops."""
    # per connected surface: holes = (2 - (V - E + F) - open loops) / 2
    # output = [#pieces, log(1 + total holes), #hollows, total open loops]

    name = "t1"
    version = "1"
    metric = "l2"
    defaults: dict = {}

    def _setup(self) -> None:
        self.dim = 4
        self.n_hist = 0

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        tm = mesh.mesh
        F = np.asarray(tm.faces)
        nF = len(F)
        n_comp, comp = _components(nF, np.asarray(tm.face_adjacency))

        # count corners, edges and triangles of each connected surface (integer keys)
        nV = np.int64(len(tm.vertices))
        comp = comp.astype(np.int64)
        vkeys = np.unique(np.repeat(comp, 3) * nV + F.ravel())
        Vc = np.bincount(vkeys // nV, minlength=n_comp)
        Fc = np.bincount(comp, minlength=n_comp)
        e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1).astype(np.int64)
        ekey = (np.repeat(comp, 3) * nV + e[:, 0]) * nV + e[:, 1]
        ekeys, ecount = np.unique(ekey, return_counts=True)
        Ec = np.bincount(ekeys // (nV * nV), minlength=n_comp)

        # open loops: chains of edges that have a triangle on one side only
        bkeys = ekeys[ecount == 1]
        bc = np.zeros(n_comp, dtype=np.int64)
        if len(bkeys):
            bcomp = bkeys // (nV * nV)
            bedges = np.column_stack([(bkeys // nV) % nV, bkeys % nV])
            verts, inv = np.unique(bedges, return_inverse=True)
            n_loops, loop_of = _components(len(verts), inv.reshape(-1, 2))
            loop_comp = np.zeros(n_loops, dtype=np.int64)
            loop_comp[loop_of[inv.reshape(-1, 2)[:, 0]]] = bcomp
            bc = np.bincount(loop_comp, minlength=n_comp)

        # through-holes per surface (eq. 24)
        chi = Vc - Ec + Fc
        g = np.maximum(0.0, (2 - chi - bc) / 2.0)

        tri = tm.triangles
        vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
        Vol = np.bincount(comp, weights=vol, minlength=n_comp)
        tol = VOLUME_TOL * mesh.area ** 1.5
        cavities = int(np.sum(Vol < -tol))
        # [#pieces, log(1 + holes), #hollows, #open loops] (eq. 25)
        return np.array([n_comp - cavities, math.log1p(float(g.sum())), cavities, float(bc.sum())])

# ==============================================================================
# END OF FILE: t1_genus.py
# ==============================================================================
# SUMMARY:
#   T1: [pieces, log(1+holes), hollows, open loops].
#
# USAGE EXAMPLE:
#   from src.descriptors.t1_genus import T1Genus
#   v = T1Genus().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Edges shared by more than two triangles split surfaces; hole count may then be too low.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
