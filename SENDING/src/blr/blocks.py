"""
================================================================================
FILE: blocks.py
================================================================================
PURPOSE:
    Registry of the 15 descriptor blocks of the 781-d QRCAD vector: where each
    block and sub-part sits, how it is measured (metric), which quality
    statistics gauge its reliability, and what a difference in it means in plain
    language (used by the difference report). The layout is the concatenation
    of src.descriptors.REGISTRY in registry order (d1 ... mts); tests compare
    it with the real descriptors.

INPUTS:
    - None (static layout of the default src/config.toml parameters).

OUTPUTS:
    - SubPart, BlockSpec, BLOCKS (15 blocks), BLOCK_BY_NAME, N_DIM (= 781),
      METRICS (metric names implemented in metrics.py, Phase 1).

DEPENDENCIES:
    stdlib      : dataclasses, math
    third-party : none
    internal    : none

PHYSICS / DOMAIN CONTEXT:
    - Histogram sub-parts are stored sqrt-mapped (Hellinger map, L1-normalised
      before the sqrt): SubPart.stored_sqrt = True. W1 must square them back to
      probabilities first; Hellinger and L2 use the stored values directly.
    - Every value is dimensionless (lengths / D_diam), so axis labels are
      fractions of the reference length unless the unit says otherwise.
    - D4 is unsigned (v2): LH and RH parts look identical to every block.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
N_DIM = 781                                     # [-] total vector length
METRICS = frozenset({
    "w1",           # Wasserstein-1 between ordered histograms (needs stored_sqrt squared)
    "hellinger",    # Hellinger distance between sqrt-mapped histograms
    "l2",           # Euclidean distance
    "abs",          # |a - b| of a scalar (per null-scaled entry)
    "abs_log",      # |log a - log b| of a positive scalar
    "log_count_l1", # L1 on log1p counts
    "none",         # not a distance input (quality statistic or redundant summary)
})
KINDS = frozenset({"hist", "hist2d", "spectrum", "scalar", "count"})
SQRT_AREA_MAX = math.sqrt(math.sqrt(3.0) / 4.0)  # [-] D3 histogram upper edge (triangle_a3_d3.py)
CBRT_VOL_MAX = (6.0 * math.sqrt(2.0)) ** (-1.0 / 3.0)  # [-] D4 histogram upper edge


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SubPart:
    """A contiguous run of dims inside a block, with one metric and axis labels."""

    name: str
    sl: slice                       # absolute slice into the 781-d vector
    kind: str                       # one of KINDS
    labels: tuple                   # one entry per dim: bin centre (axis units) or index name
    unit: str
    metric: str                     # one of METRICS
    stored_sqrt: bool = False       # True: stored as sqrt(p); W1 needs p = stored**2


@dataclass(frozen=True)
class BlockSpec:
    """One descriptor block of the 781-d vector."""

    name: str                       # descriptor key in src.descriptors.REGISTRY
    plain_name: str
    what_it_sees: str               # "a difference means ..."
    sl: slice
    subparts: tuple[SubPart, ...]
    metric: str                     # main metric (that of the first distance-carrying sub-part)
    family: str                     # point | normal | intrinsic | local | axis | topology | thickness
    quality_keys: tuple[str, ...]   # keys of quality.QUALITY_KEYS gating the block's reliability


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def centres(n: int, lo: float, hi: float) -> tuple[float, ...]:
    """Bin centres of n equal bins on [lo, hi]."""
    w = (hi - lo) / n
    return tuple(lo + (i + 0.5) * w for i in range(n))


def _names(prefix: str, n: int, start: int = 0) -> tuple[str, ...]:
    return tuple(f"{prefix}{i}" for i in range(start, start + n))


def _grid(n_a: int, n_b: int, la: tuple[float, float], lb: tuple[float, float],
          fa: str, fb: str, log_b: bool = False) -> tuple[str, ...]:
    """Labels of a row-major (n_a x n_b) grid, e.g. 'SI=+0.83 CV=0.2'."""
    ca, cb = centres(n_a, *la), centres(n_b, *lb)
    return tuple(f"{fa}={a:+.2f} {fb}={(10 ** b if log_b else b):.3g}" for a in ca for b in cb)


def _build() -> tuple[BlockSpec, ...]:
    """Assemble BLOCKS: parts are (name, width, kind, labels, unit, metric, stored_sqrt)."""
    deg = tuple(math.degrees(c) for c in centres(32, 0.0, math.pi))
    spec = [
        ("d1", "D1 radial distance", "the surface sits at different distances from the centre "
         "(solid vs hollow, round vs square)", "point", (), [
            ("hist", 32, "hist", centres(32, 0, 1), "distance from centre / max", "w1", True)]),
        ("d2", "D2 pairwise distance", "different overall proportions or spacing between features "
         "(arm layout, elongation)", "point", (), [
            ("hist", 32, "hist", centres(32, 0, 1), "point-pair distance / D", "w1", True)]),
        ("triangle", "A3/D3 triangles", "the part bends differently (straight vs bowed), "
         "different local triangle shapes", "point", (), [
            ("a3", 32, "hist", deg, "deg", "w1", True),
            ("d3", 32, "hist", centres(32, 0, SQRT_AREA_MAX), "sqrt(area) / D", "w1", True)]),
        ("d4", "D4 tetrahedron volume", "different 3-D thickness or camber (unsigned volumes: "
         "left- and right-hand parts look identical)", "point", (), [
            ("hist", 32, "hist", centres(32, 0, CBRT_VOL_MAX), "cbrt(V / D^3)", "w1", True),
            ("skew", 1, "scalar", ("gamma1",), "-", "abs", False)]),
        ("shps", "SHPS orientation spectrum", "surfaces face in different directions "
         "(grooves, knurls, facets)", "normal", (), [
            ("spectrum", 15, "spectrum", tuple(f"l={l}" for l in range(2, 17)), "harmonic order", "l2", False)]),
        ("ncf", "NCF normal correlation", "thin walls vs solid material "
         "(opposing normals across a wall)", "normal", (), [
            ("g", 32, "spectrum", centres(32, 0, 1), "pair distance / D", "l2", False)]),
        ("lbe", "LBE Laplace-Beltrami", "different connectivity: slots, flexures, separate lobes",
         "intrinsic", ("n_vertices", "boundary_edge_frac"), [
            ("spectrum", 30, "spectrum", tuple(f"k={k}" for k in range(1, 31)), "eigenvalue index", "l2", False)]),
        ("dad", "DAD dihedral angles", "different edge treatment (sharp vs chamfer, crease angles, "
         "crease length)", "local", ("boundary_edge_frac", "facet_angle"), [
            ("hist", 36, "hist", tuple(math.degrees(c) for c in centres(36, 0, 2 * math.pi)),
             "dihedral angle [deg]", "w1", True),
            ("crease_length", 1, "scalar", ("Lambda",), "crease length / D", "abs_log", False),
            ("total_h", 1, "scalar", ("H",), "total mean curvature", "abs_log", False)]),
        ("curvature", "CURV shape index", "curved regions differ: convex vs concave, fillets, "
         "sphere vs cylinder", "local", ("boundary_edge_frac",), [
            ("si_cv", 96, "hist2d", _grid(12, 8, (-1, 1), (-1, 2), "SI", "cCV", log_b=True),
             "(shape index, curvedness)", "hellinger", True),
            ("flat_share", 1, "scalar", ("flat",), "flat fraction", "abs", False)]),
        ("crs_r", "CRS-R ring profile", "turned features have different radii or are not concentric",
         "axis", ("eigen_gap",), [
            ("axis", 64, "hist", centres(64, 0, 0.75), "radius / D (symmetry axis)", "w1", True),
            ("other1", 32, "hist", centres(32, 0, 0.75), "radius / D (2nd axis)", "w1", True),
            ("other2", 32, "hist", centres(32, 0, 0.75), "radius / D (3rd axis)", "w1", True)]),
        ("t1", "T1 genus / topology", "different number of bodies, through-holes (genus) "
         "or internal cavities", "topology", ("boundary_loops",), [
            ("counts", 3, "count", ("bodies", "genus", "cavities"), "count", "log_count_l1", False),
            ("boundary_loops", 1, "scalar", ("boundary_loops",), "count", "none", False)]),
        ("phd", "PHD persistent homology", "holes or loops sit at different positions or sizes "
         "around the axis", "axis", ("eigen_gap", "boundary_edge_frac"), [
            (f"img_{n}", 25, "hist2d", _grid(5, 5, (0, 0.5), (0, 0.25), "birth", "pers"),
             "(birth, persistence) / D", "l2", True) for n in ("pd0", "pd1", "ext1")]
            + [("loops", 1, "count", ("loops",), "count", "log_count_l1", False),
               ("total_persistence", 1, "scalar", ("total",), "/ D", "abs", False)]),
        ("crs_a", "CRS-A angular spectrum", "features are arranged in a different angular pattern "
         "(hole pitch, grouping)", "axis", ("eigen_gap",), [
            (f"band{r}", 32, "spectrum", tuple(f"band={r} m={m}" for m in range(1, 33)),
             "harmonic m", "l2", False) for r in range(4)]),
        ("paf", "PAF planar families", "different planar faces and datum planes "
         "(coplanarity, step heights)", "normal", (), [
            ("head", 7, "scalar", ("c_l", "c_p", "c_s", "I4", "H", "n_families/16", "Pi"),
             "Westin / entropy", "l2", False),
            ("families", 24, "scalar", tuple(f"f{k}_{n}" for k in range(6)
                                             for n in ("alpha", "n_planes/10", "cv", "angle/90")),
             "per-family stats", "l2", False)]),
        ("mts", "MTS wall thickness", "different wall thickness", "thickness", ("f_miss",), [
            ("hist", 32, "hist", centres(32, -3, 0), "log10(thickness / D)", "w1", True),
            ("moments", 6, "scalar", ("mu", "sigma", "gamma1", "Q1", "Q2", "Q3"), "log10 w", "none", False),
            ("f_thin", 1, "scalar", ("f_thin",), "fraction", "abs", False),
            ("f_miss", 1, "scalar", ("f_miss",), "fraction", "none", False)]),
    ]
    blocks, off = [], 0
    for name, plain, sees, family, quality, parts in spec:
        subs, start = [], off
        for pname, width, kind, labels, unit, metric, sq in parts:
            subs.append(SubPart(pname, slice(off, off + width), kind, tuple(labels), unit, metric, sq))
            off += width
        metric = next(s.metric for s in subs if s.metric != "none")
        blocks.append(BlockSpec(name, plain, sees, slice(start, off), tuple(subs), metric, family, quality))
    return tuple(blocks)


BLOCKS: tuple[BlockSpec, ...] = _build()
BLOCK_BY_NAME: dict[str, BlockSpec] = {b.name: b for b in BLOCKS}

# ==============================================================================
# END OF FILE: blocks.py
# ==============================================================================
# SUMMARY:
#   Static 15-block registry (slices, sub-parts, metrics, families, quality keys).
#
# USAGE EXAMPLE:
#   from src.blr.blocks import BLOCKS
#   x[BLOCKS[7].sl]                        # the DAD block of a 781-d vector
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Layout follows the default config.toml params; a params change (bins,
#         top_families, ...) needs this file updated (tests/test_blocks.py checks it).
#   - [ ] MTS moments (mu, sigma, gamma1, Q1-Q3) get metric "none": redundant with the
#         histogram; revisit in Phase 1 if a scalar distance is wanted.
#   - [ ] Block order is the descriptor registry order, not the plan's section 3 order.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
