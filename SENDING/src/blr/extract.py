"""
================================================================================
FILE: extract.py
================================================================================
PURPOSE:
    File-level entry points of BLR: STL -> (optionally preconditioned mesh) ->
    the 781-d block vector and its quality statistics, using the existing
    src.descriptors / src.preprocessing pipeline (no extractor is duplicated).

INPUTS:
    - STL path, integer seed, optional Config (default: src/config.toml).

OUTPUTS:
    - extract_blocks(path, seed) -> float32 (781,)
    - quality_stats(path, seed)  -> dict[str, float] (QUALITY_KEYS)
    - extract_with_quality(path, seed) -> (x, q) computed in one pass.

DEPENDENCIES:
    stdlib      : logging, pathlib
    third-party : numpy, trimesh
    internal    : src.config, src.descriptors, src.orchestrator, src.preprocessing, blr.*

PHYSICS / DOMAIN CONTEXT:
    - precondition_mesh: midpoint-subdivides the whole mesh (never only some faces:
      that would open T-junction cracks) until the longest edge is <= edge_frac * D
      (D = bounding-box diagonal) or the face budget is reached. Coarse flat parts
      (12-76 triangles) then give LBE, DAD and CURV enough vertices. This is a
      uniform 1:4 split, not an isotropic remesh (pymeshlab is not installed).
    - Every descriptor draws from its own rng default_rng([seed, crc32(name)]) exactly as
      orchestrator.extract_file does, so a fixed seed is deterministic.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh

from src.blr.blocks import N_DIM
from src.blr.config import BLRConfig
from src.blr.quality import quality_from
from src.config import DEFAULT_CONFIG, Config, load_config
from src.descriptors import REGISTRY, make_descriptors
from src.orchestrator import descriptor_rng
from src.preprocessing import PreparedMesh, load_normalized

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def resolve_device(device: str) -> str:
    """"auto" -> "cuda" when torch sees a GPU, else "cpu"; explicit values pass through."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def precondition_mesh(mesh: trimesh.Trimesh, edge_frac: float, max_faces: int) -> trimesh.Trimesh:
    """
    Uniformly subdivide until the longest edge <= edge_frac * D or max_faces is reached.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Welded mesh.
    edge_frac : float
        Target maximum edge length / bounding-box diagonal D [-].
    max_faces : int
        Do not perform a subdivision pass that would exceed this many faces.

    Returns
    -------
    trimesh.Trimesh
        New mesh (input untouched); geometry is unchanged, only the tessellation.
    """
    D = float(np.linalg.norm(mesh.extents))
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    longest = float(mesh.edges_unique_length.max())
    while longest > edge_frac * D and 4 * len(f) <= max_faces:
        v, f = trimesh.remesh.subdivide(v, f)
        longest /= 2.0
    return trimesh.Trimesh(v, f, process=False)


def _extract(path: Path, seed: int, cfg: Config, edge_frac: float, max_faces: int,
             precondition: bool, device: str) -> tuple[np.ndarray, dict[str, float]]:
    raw = load_normalized(path, cfg.scale_rule)
    mesh = precondition_mesh(raw, edge_frac, max_faces) if precondition else raw
    pm = PreparedMesh(mesh=mesh, is_watertight=bool(mesh.is_watertight),
                      seed=int(seed), n_samples=cfg.n_samples)
    parts = [d.compute(pm, descriptor_rng(seed, d.name))
             for d in make_descriptors(list(REGISTRY), cfg.descriptor_params, device)]
    x = np.concatenate(parts).astype(np.float32)
    if x.shape != (N_DIM,):
        raise RuntimeError(f"expected {N_DIM} dims, got {x.shape}")
    return x, quality_from(raw, pm.axes[1], x)


def extract_with_quality(path: Path, seed: int, cfg: Config | None = None,
                         blr: BLRConfig | None = None,
                         precondition: bool = True, device: str = "cpu") -> tuple[np.ndarray, dict[str, float]]:
    """One-pass 781-d vector (float32, shape (781,)) and its quality dict; "cpu" (GPU measured slower for every heavy descriptor)."""
    blr = blr or BLRConfig()
    cfg = cfg or load_config(DEFAULT_CONFIG, ROOT_DIR)
    return _extract(Path(path), seed, cfg, blr.remesh_edge_frac, blr.remesh_max_faces, precondition,
                    resolve_device(device))


def extract_blocks(mesh_path: Path, seed: int, **kw) -> np.ndarray:
    """781-d block vector of one STL, float32, shape (781,); deterministic for a fixed seed."""
    return extract_with_quality(mesh_path, seed, **kw)[0]


def quality_stats(mesh_path: Path, seed: int, **kw) -> dict[str, float]:
    """Quality statistics (blr.quality.QUALITY_KEYS) of one STL."""
    return extract_with_quality(mesh_path, seed, **kw)[1]


# ==============================================================================
# END OF FILE: extract.py
# ==============================================================================
# SUMMARY:
#   STL -> preconditioned mesh -> 781-d block vector + quality statistics.
#
# USAGE EXAMPLE:
#   x, q = extract_with_quality(Path("dataset/_sample/pin/pin_00.stl"), seed=42)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Uniform 1:4 subdivision, not an isotropic remesh; install pymeshlab for that.
#   - [ ] quality_stats and extract_blocks each run a full extraction; use
#         extract_with_quality when both are needed.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
