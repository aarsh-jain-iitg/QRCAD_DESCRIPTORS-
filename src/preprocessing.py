"""
================================================================================
FILE: preprocessing.py
================================================================================
PURPOSE:
    Turn one STL file into a PreparedMesh that every descriptor shares: loaded
    once, normalised (area-weighted centroid at the origin, config scale
    rule), welded and repaired, with seeded area-weighted surface samples,
    the reference length D_diam and the principal axes computed lazily and
    cached. No descriptor reloads or resamples the mesh except CRS-A, whose
    definition asks for a larger pool (it calls sample_pool(n)).

INPUTS:
    - path : Path to an .stl file.
    - seed, n_samples, scale_rule : from [pipeline] in config.toml.

OUTPUTS:
    - prepare_mesh(...) -> PreparedMesh
    - Raises DescriptorError if the mesh is empty / zero-area after repair.

DEPENDENCIES:
    stdlib      : dataclasses, functools, logging, pathlib
    third-party : numpy, trimesh
    internal    : descriptors.base (DescriptorError)

PHYSICS / DOMAIN CONTEXT:
    - Units: whatever the STL uses; everything downstream is dimensionless.
    - Sampling follows CAD_Descriptors_Explained.pdf eq. (2): triangle t with
      probability A_t/A, then p = (1-sqrt r1) x0 + sqrt r1 (1-r2) x1 + sqrt r1 r2 x2.
    - D_diam uses the double-normal iteration (Complexity PDF), which
      matches the exact hull diameter to ~0.02 % at O(n) instead of O(n^2).
    - Normals are made consistent with trimesh.repair.fix_normals(multibody=
      False): winding is unified per component, and the whole mesh is
      inverted only if its total volume is negative, so the inward-facing
      walls of sealed cavities keep their orientation (T1 needs that).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import trimesh

from src.descriptors.base import EPS, DescriptorError

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SCALE_RULES = ("bbox_diagonal", "rms_radius", "unit_area")
DIAMETER_ITERATIONS = 4             # [-] double-normal jumps (Complexity PDF)
AXIS_DEGENERACY_TOL = 1e-3          # [-] relative eigen-gap below which axes are ambiguous

logger = logging.getLogger(__name__)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _triangle_stats(tri: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Per-face areas [L^2], total area [L^2], area-weighted centroid [L]."""
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    A = float(areas.sum())
    if not np.isfinite(A) or A <= EPS:
        raise DescriptorError("mesh has zero surface area")
    c = (areas[:, None] * tri.mean(axis=1)).sum(axis=0) / A
    return areas, A, c


def _scale_factor(tri: np.ndarray, areas: np.ndarray, A: float, rule: str) -> float:
    """
    Length that is divided out during normalisation [L].

    bbox_diagonal : diagonal of the axis-aligned bounding box (not rotation
                    invariant, harmless because every descriptor re-normalises).
    rms_radius    : sqrt(E|p|^2) over the surface, exact per triangle:
                    E|p|^2 = (|x0|^2+|x1|^2+|x2|^2+|x0+x1+x2|^2)/12.
    unit_area     : sqrt(A), so the normalised surface has area 1.
    """
    if rule == "bbox_diagonal":
        pts = tri.reshape(-1, 3)
        s = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    elif rule == "rms_radius":
        sq = (tri ** 2).sum(axis=2).sum(axis=1) + (tri.sum(axis=1) ** 2).sum(axis=1)
        s = float(np.sqrt((areas * sq / 12.0).sum() / A))
    elif rule == "unit_area":
        s = float(np.sqrt(A))
    else:
        raise ValueError(f"unknown scale rule {rule!r}; expected one of {SCALE_RULES}")
    if not np.isfinite(s) or s <= EPS:
        raise DescriptorError(f"degenerate scale ({rule} = {s:.3e})")
    return s


# ── CORE CLASSES / MAIN FUNCTIONS ─────────────────────────────────────────────
@dataclass
class SurfaceSamples:
    """Area-uniform surface samples: points (n,3), unit normals (n,3), face ids (n,)."""

    points: np.ndarray
    normals: np.ndarray
    face_index: np.ndarray


@dataclass
class PreparedMesh:
    """
    A normalised, welded mesh plus cached shared quantities.

    Attributes
    ----------
    mesh : trimesh.Trimesh
        Welded, repaired mesh; centroid at the origin, unit scale per rule.
    is_watertight : bool
        trimesh watertightness after repair.
    seed : int
        Base seed for the sample pools.
    n_samples : int
        Size of the default sample pool.
    """

    mesh: trimesh.Trimesh
    is_watertight: bool
    seed: int
    n_samples: int
    _pools: dict[int, SurfaceSamples] = field(default_factory=dict, repr=False)

    @property
    def vertices(self) -> np.ndarray:
        return self.mesh.vertices

    @property
    def faces(self) -> np.ndarray:
        return self.mesh.faces

    @cached_property
    def face_areas(self) -> np.ndarray:
        return np.asarray(self.mesh.area_faces, dtype=np.float64)

    @cached_property
    def face_normals(self) -> np.ndarray:
        return np.asarray(self.mesh.face_normals, dtype=np.float64)

    @cached_property
    def face_centers(self) -> np.ndarray:
        return np.asarray(self.mesh.triangles_center, dtype=np.float64)

    @cached_property
    def area(self) -> float:
        return float(self.face_areas.sum())

    @cached_property
    def centroid(self) -> np.ndarray:
        """Area-weighted surface centroid c = sum A_t b_t / A."""
        return (self.face_areas[:, None] * self.face_centers).sum(axis=0) / self.area

    @property
    def samples(self) -> SurfaceSamples:
        """The default pool of n_samples points (cached)."""
        return self.sample_pool(self.n_samples)

    def sample_pool(self, n: int) -> SurfaceSamples:
        """
        Seeded area-weighted surface samples, PDF eq. (2). Cached per n.

        The generator is seeded with (seed, n), so a pool of a given size is
        identical no matter which descriptors ran before.
        """
        if n in self._pools:
            return self._pools[n]
        rng = np.random.default_rng([self.seed, 0x5A4D, n])
        p = self.face_areas / self.area
        fi = rng.choice(len(p), size=n, p=p)
        r1 = np.sqrt(rng.random(n))[:, None]
        r2 = rng.random(n)[:, None]
        tri = self.mesh.triangles[fi]
        pts = (1.0 - r1) * tri[:, 0] + r1 * (1.0 - r2) * tri[:, 1] + r1 * r2 * tri[:, 2]
        pool = SurfaceSamples(pts, self.face_normals[fi], fi)
        self._pools[n] = pool
        return pool

    @cached_property
    def diameter(self) -> float:
        """
        Reference length D_diam = max_ij |p_i - p_j| of the default pool [L].

        Double-normal iteration: start at the sample farthest from the
        centroid, jump to the farthest sample from the current one,
        DIAMETER_ITERATIONS times, keep the largest distance seen.
        """
        P = self.samples.points
        i = int(np.argmax(np.linalg.norm(P - self.centroid, axis=1)))
        best = 0.0
        for _ in range(DIAMETER_ITERATIONS):
            d = np.linalg.norm(P - P[i], axis=1)
            j = int(np.argmax(d))
            best = max(best, float(d[j]))
            i = j
        if best <= EPS:
            raise DescriptorError("reference length D_diam is zero")
        return best

    @cached_property
    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Principal axes of the default pool (Complexity PDF, shared by
        CRS-R, PHD, CRS-A).

        Returns
        -------
        center : np.ndarray, shape (3,)
            Mean of the samples [L].
        eigvals : np.ndarray, shape (3,)
            Covariance eigenvalues, descending [L^2].
        eigvecs : np.ndarray, shape (3, 3)
            Columns are unit eigenvectors in eigvals order. Signs are fixed
            so that the third moment of the projections is >= 0 (the
            skewness convention); the frame is then made right-handed. The
            axis-based descriptors only use axis *lines*, so they do not
            depend on these signs. When two eigenvalues are within
            AXIS_DEGENERACY_TOL (relative), the eigenvectors in that plane
            are arbitrary; this is logged at DEBUG level.
        """
        P = self.samples.points
        center = P.mean(axis=0)
        X = P - center
        C = X.T @ X / len(X)
        w, V = np.linalg.eigh(C)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        for a in range(3):
            if np.sum((X @ V[:, a]) ** 3) < 0:
                V[:, a] = -V[:, a]
        if np.linalg.det(V) < 0:
            V[:, 2] = -V[:, 2]
        scale = max(float(w[0]), EPS)
        if min(w[0] - w[1], w[1] - w[2]) / scale < AXIS_DEGENERACY_TOL:
            logger.debug("near-degenerate principal axes, eigvals=%s", w)
        return center, w, V


def load_normalized(path: Path, scale_rule: str) -> trimesh.Trimesh:
    """
    Load an STL, normalise it, then weld and repair it.

    Normalisation happens before welding so that trimesh's absolute merge
    tolerance behaves the same for a 1e-6 m part and a 1e6 mm part.

    Parameters
    ----------
    path : Path
        STL file.
    scale_rule : str
        One of SCALE_RULES.

    Returns
    -------
    trimesh.Trimesh
        Welded mesh with degenerate / duplicate faces removed and
        consistent outward normals.

    Raises
    ------
    DescriptorError
        If the file has no usable faces.
    """
    raw = trimesh.load(Path(path), force="mesh", process=False)
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
        raise DescriptorError(f"no triangles in {Path(path).name}")
    V = np.asarray(raw.vertices, dtype=np.float64)
    Fc = np.asarray(raw.faces, dtype=np.int64)
    if not np.all(np.isfinite(V)):
        raise DescriptorError("non-finite vertex coordinates")
    tri = V[Fc]
    areas, A, c = _triangle_stats(tri)
    s = _scale_factor(tri - c, areas, A, scale_rule)
    mesh = trimesh.Trimesh((V - c) / s, Fc, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0 or mesh.area <= EPS:
        raise DescriptorError("no non-degenerate faces after repair")
    trimesh.repair.fix_normals(mesh, multibody=False)
    return mesh


def prepare_mesh(path: Path, seed: int, n_samples: int,
                 scale_rule: str = "rms_radius") -> PreparedMesh:
    """
    Load and prepare one STL for descriptor extraction.

    Parameters
    ----------
    path : Path
        STL file.
    seed : int
        Base seed for sampling.
    n_samples : int
        Default pool size (PDF: 2e4).
    scale_rule : str
        Normalisation rule, see _scale_factor.

    Returns
    -------
    PreparedMesh
    """
    mesh = load_normalized(path, scale_rule)
    return PreparedMesh(mesh=mesh, is_watertight=bool(mesh.is_watertight),
                        seed=int(seed), n_samples=int(n_samples))

# ==============================================================================
# END OF FILE: preprocessing.py
# ==============================================================================
# SUMMARY:
#   STL -> normalised, welded PreparedMesh with cached samples / D_diam / axes.
#
# USAGE EXAMPLE:
#   from src.preprocessing import prepare_mesh
#   pm = prepare_mesh(Path("dataset/_sample/box/box_00.stl"), seed=42, n_samples=20000)
#   pm.samples.points.shape, pm.diameter
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] fix_normals(multibody=False) cannot repair a single body exported
#         inside-out inside an otherwise correct multi-body file.
#   - [ ] No remeshing to a vertex budget (Complexity PDF item 2); LBE cost
#         grows ~V^1.3 on very fine meshes.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
