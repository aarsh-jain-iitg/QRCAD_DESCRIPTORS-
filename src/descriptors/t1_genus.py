"""
================================================================================
FILE: t1_genus.py
================================================================================
PURPOSE:
    T1 genus and body count, CAD_Descriptors_Explained.pdf, Algorithm 11: per
    connected surface component, Euler characteristic, boundary loops and
    genus; bodies vs sealed cavities from the sign of each component's
    enclosed volume.

INPUTS:
    - PreparedMesh (welded mesh with consistent orientation).

OUTPUTS:
    - float32 vector, shape (4,): [#bodies, log(1+g), #cavities, sum b_c].

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, scipy.sparse
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - chi_c = V_c - E_c + F_c = 2 - 2 g_c - b_c  =>  g_c = (2 - chi_c - b_c)/2
      (eq. 24), clipped at 0 for non-manifold input.
    - Components are face sets connected through shared (manifold) edges.
    - V_c = 1/6 sum x0 . (x1 x x2): > 0 outer skin, < 0 cavity wall. Volumes
      with |V_c| <= 1e-9 A^(3/2) (open sheets) count as bodies.
    - Topological counts: invariant under every similarity transform.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.descriptors.base import Descriptor

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
VOLUME_TOL = 1e-9        # [-] |V_c| / A^(3/2) below which a component is a sheet


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _components(n: int, pairs: np.ndarray) -> tuple[int, np.ndarray]:
    """Connected components of an undirected graph on n nodes given edge pairs (E,2)."""
    # points: none sampled; counts over all welded vertices, edges and faces
    if len(pairs) == 0:
        return n, np.arange(n)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class T1Genus(Descriptor):
    """T1: bodies, genus, cavities, boundary loops."""
    # chi_c = V_c - E_c + F_c = 2 - 2 g_c - b_c  =>  g_c = (2 - chi_c - b_c) / 2
    # F_T1 = [#bodies, log(1 + g), #cavities, sum_c b_c],  g = sum_c g_c

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

        # vertices, edges and faces counted per component (1-D integer keys)
        nV = np.int64(len(tm.vertices))
        comp = comp.astype(np.int64)
        vkeys = np.unique(np.repeat(comp, 3) * nV + F.ravel())
        Vc = np.bincount(vkeys // nV, minlength=n_comp)
        Fc = np.bincount(comp, minlength=n_comp)
        e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1).astype(np.int64)
        ekey = (np.repeat(comp, 3) * nV + e[:, 0]) * nV + e[:, 1]
        ekeys, ecount = np.unique(ekey, return_counts=True)
        Ec = np.bincount(ekeys // (nV * nV), minlength=n_comp)

        # boundary loops: components of the graph of one-sided edges
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

        # eq.(24): g_c = (2 - chi_c - b_c) / 2
        chi = Vc - Ec + Fc
        g = np.maximum(0.0, (2 - chi - bc) / 2.0)

        tri = tm.triangles
        vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
        Vol = np.bincount(comp, weights=vol, minlength=n_comp)
        tol = VOLUME_TOL * mesh.area ** 1.5
        cavities = int(np.sum(Vol < -tol))
        # eq.(25): [#bodies, log(1 + g), #cavities, sum b_c]
        return np.array([n_comp - cavities, math.log1p(float(g.sum())), cavities, float(bc.sum())])

# ==============================================================================
# END OF FILE: t1_genus.py
# ==============================================================================
# SUMMARY:
#   T1: [bodies, log(1+genus), cavities, boundary loops] from Euler-Poincare.
#
# USAGE EXAMPLE:
#   from src.descriptors.t1_genus import T1Genus
#   v = T1Genus().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Non-manifold edges split components; genus is then a lower bound.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
