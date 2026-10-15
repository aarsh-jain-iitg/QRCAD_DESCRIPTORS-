"""
================================================================================
FILE: quality.py
================================================================================
PURPOSE:
    Per-part quality statistics that gauge how far a block can be trusted
    (block reliability omega_b, Phase 3). Pure functions on a mesh or on
    already-extracted values; the file-level entry point is
    extract.quality_stats().

INPUTS:
    - trimesh.Trimesh (welded, normalised), covariance eigenvalues, the 781-d vector.

OUTPUTS:
    - QUALITY_KEYS and the individual statistics (all floats).

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, trimesh
    internal    : blr.blocks

PHYSICS / DOMAIN CONTEXT:
    - boundary_edge_frac: fraction of edges with exactly one incident face; 0 for a
      watertight part. Open surfaces break volume-like descriptors (LBE, DAD, CURV, PHD).
    - facet_angle_deg: median dihedral over non-crease edges (< tau); a coarse
      tessellation of a curved surface shows up as a large value, and LBE/DAD/CURV
      then read facet artefacts instead of the design.
    - eigen_gap: min(l1 - l2, l2 - l3) / l1 of the sample covariance; near 0 the
      principal axes are ambiguous and the axis-based blocks (CRS-R, CRS-A, PHD)
      are unreliable (CLAUDE.md, "Axis-based blocks").

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from src.blr.blocks import BLOCK_BY_NAME

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
QUALITY_KEYS = ("n_vertices", "boundary_edge_frac", "facet_angle", "eigen_gap",
                "f_miss", "boundary_loops")
CREASE_TAU_DEG = 20.0                 # [deg] dihedral above this is a crease, not a facet step


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def boundary_edge_frac(mesh: trimesh.Trimesh) -> float:
    """Fraction of unique edges with a single incident face [-]; 0 if there are no edges."""
    if len(mesh.faces) == 0:
        return 0.0
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    return float(np.mean(counts == 1))


def facet_angle_deg(mesh: trimesh.Trimesh, tau_deg: float = CREASE_TAU_DEG) -> float:
    """Median dihedral angle [deg] over edges below tau_deg; 0 if every edge is a crease."""
    a = np.degrees(mesh.face_adjacency_angles)
    a = a[a < tau_deg]
    return float(np.median(a)) if len(a) else 0.0


def eigen_gap(eigvals: np.ndarray) -> float:
    """min(l1 - l2, l2 - l3) / l1 for descending eigenvalues, shape (3,) [-]."""
    l1 = max(float(eigvals[0]), 1e-300)
    return float(min(eigvals[0] - eigvals[1], eigvals[1] - eigvals[2]) / l1)


def block_scalar(x: np.ndarray, block: str, sub: str, j: int = 0) -> float:
    """Value j of sub-part `sub` of `block` in the 781-d vector x, shape (781,)."""
    sp = next(s for s in BLOCK_BY_NAME[block].subparts if s.name == sub)
    return float(x[sp.sl][j])


def quality_from(mesh: trimesh.Trimesh, eigvals: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """
    All QUALITY_KEYS for one part.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Welded, normalised mesh BEFORE preconditioning (facet and boundary statistics
        must describe the input tessellation, not the subdivided one).
    eigvals : np.ndarray, shape (3,)
        Descending covariance eigenvalues of the surface samples.
    x : np.ndarray, shape (781,)
        Extracted blocks (f_miss from MTS, boundary loops from T1).

    Returns
    -------
    dict[str, float]
    """
    return {"n_vertices": float(len(mesh.vertices)),
            "boundary_edge_frac": boundary_edge_frac(mesh),
            "facet_angle": facet_angle_deg(mesh),
            "eigen_gap": eigen_gap(eigvals),
            "f_miss": block_scalar(x, "mts", "f_miss"),
            "boundary_loops": block_scalar(x, "t1", "boundary_loops")}


# ==============================================================================
# END OF FILE: quality.py
# ==============================================================================
# SUMMARY:
#   Quality statistics (mesh openness, tessellation coarseness, axis ambiguity,
#   ray misses, boundary loops) used to gate block reliability.
#
# USAGE EXAMPLE:
#   q = quality_from(mesh, eigvals, x)          # dict with QUALITY_KEYS
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] face_adjacency_angles needs a welded mesh; non-manifold edges (>2 faces) are
#         ignored by trimesh's adjacency.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
