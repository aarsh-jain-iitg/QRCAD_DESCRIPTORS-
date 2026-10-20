"""
================================================================================
FILE: __init__.py  (src/descriptors)
================================================================================
PURPOSE:
    List of the fifteen descriptors: maps each descriptor's config name to
    its class, in the order of the PDF, so the orchestrator can create them
    by name.

INPUTS:
    - Descriptor names (from [pipeline].enabled) and their settings.

OUTPUTS:
    - REGISTRY : dict[str, type[Descriptor]]
    - make_descriptors(names, params, device) -> list[Descriptor]

DEPENDENCIES:
    stdlib      : (none)
    third-party : (none)
    internal    : every descriptor module

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

from typing import Any

from src.descriptors.base import Descriptor, DescriptorError
from src.descriptors.crs_a_angular_spectrum import CRSAAngularSpectrum
from src.descriptors.crs_r_ring_profile import CRSRRingProfile
from src.descriptors.curvature_shape_index import CurvatureShapeIndex
from src.descriptors.d1_radial_distance import D1RadialDistance
from src.descriptors.d2_pairwise_distance import D2PairwiseDistance
from src.descriptors.d4_tetrahedron_volume import D4TetrahedronVolume
from src.descriptors.dad_dihedral_angle import DADDihedralAngle
from src.descriptors.lbe_laplace_beltrami import LBELaplaceBeltrami
from src.descriptors.mts_wall_thickness import MTSWallThickness
from src.descriptors.ncf_normal_correlation import NCFNormalCorrelation
from src.descriptors.paf_planar_families import PAFPlanarFamilies
from src.descriptors.phd_persistent_homology import PHDPersistentHomology
from src.descriptors.shps_orientation_spectrum import SHPSOrientationSpectrum
from src.descriptors.t1_genus import T1Genus
from src.descriptors.triangle_a3_d3 import TriangleA3D3

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# PDF order (D1 -> MTS); the combined vector stores the blocks in this same order.
REGISTRY: dict[str, type[Descriptor]] = {cls.name: cls for cls in (
    D1RadialDistance, D2PairwiseDistance, TriangleA3D3, D4TetrahedronVolume,
    SHPSOrientationSpectrum, NCFNormalCorrelation, LBELaplaceBeltrami,
    DADDihedralAngle, CurvatureShapeIndex, CRSRRingProfile, T1Genus,
    PHDPersistentHomology, CRSAAngularSpectrum, PAFPlanarFamilies, MTSWallThickness,
)}


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def make_descriptors(names: list[str], params: dict[str, dict[str, Any]],
                     device: str = "cpu") -> list[Descriptor]:
    """
    Create descriptors by name, in registry order.

    Parameters
    ----------
    names : list[str]
        Names of the descriptors to turn on.
    params : dict[str, dict]
        Per-descriptor settings ([descriptors.<name>]).
    device : str
        "cpu" or "cuda" (only descriptors with a GPU version use it).

    Returns
    -------
    list[Descriptor]
    """
    unknown = set(names) - set(REGISTRY)
    if unknown:
        raise ValueError(f"unknown descriptor(s): {sorted(unknown)}")
    return [REGISTRY[n](params.get(n, {}), device) for n in REGISTRY if n in names]


__all__ = ["REGISTRY", "Descriptor", "DescriptorError", "make_descriptors"]

# ==============================================================================
# END OF FILE: __init__.py  (src/descriptors)
# ==============================================================================
# SUMMARY:
#   Name -> class list of the 15 descriptors.
#
# USAGE EXAMPLE:
#   from src.descriptors import make_descriptors
#   descs = make_descriptors(["d1", "shps"], {}, "cpu")
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] None.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
