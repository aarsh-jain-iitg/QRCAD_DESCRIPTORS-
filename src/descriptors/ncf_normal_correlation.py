"""
================================================================================
FILE: ncf_normal_correlation.py
================================================================================
PURPOSE:
    NCF, do nearby and far-apart surfaces face the same way? (PDF Algorithm
    6). Picks many random pairs of surface points. For each pair it checks
    how much their facing directions agree (+1 same, 0 at right angles,
    -1 opposite) and adds that to a bin chosen by how far apart they are.

INPUTS:
    - PreparedMesh (surface sample points with normals, part size D_diam), seeded rng.
    - params: bins (32), pairs (2e5).

OUTPUTS:
    - float32 vector, shape (bins,): average agreement in each distance bin.
      Values can be negative, so this is not a histogram.

DEPENDENCIES:
    stdlib      : (none)
    third-party : numpy, torch (optional, device = cuda)
    internal    : descriptors.base

PHYSICS / DOMAIN CONTEXT:
    - Distances and angles between normals do not change when the part is
      moved, turned, mirrored or scaled (distances are divided by D_diam).
    - For a closed part all bins add up to 0.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-21
================================================================================
"""

from __future__ import annotations

import numpy as np

from src.descriptors.base import Descriptor, distinct_tuples, resolve_torch_device, torch_gather


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _dist_dot_torch(P: np.ndarray, Nn: np.ndarray, idx: np.ndarray,
                    dev) -> tuple[np.ndarray, np.ndarray]:
    """GPU version: (distance [L], normal agreement n_i . n_j [-]) for each pair in idx (S, 2)."""
    import torch
    q, nq = torch_gather(P, idx, dev), torch_gather(Nn, idx, dev)
    r = torch.linalg.vector_norm(q[:, 0] - q[:, 1], dim=1)
    return r.cpu().numpy(), (nq[:, 0] * nq[:, 1]).sum(dim=1).cpu().numpy()


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class NCFNormalCorrelation(Descriptor):
    """NCF: how much surface directions agree, by distance."""
    # r_ij = |p_i - p_j| / D_diam picks the bin m
    # g(m) = (1/S) * sum of (n_i . n_j) over the pairs in bin m,  m = 1..32
    # points used: 2e5 random pairs (params.pairs) from the 2e4 surface samples (pipeline.n_samples)

    name = "ncf"
    version = "1"
    metric = "l2"
    defaults = {"bins": 32, "pairs": 200_000}

    def _setup(self) -> None:
        self.dim = int(self.params["bins"])
        self.n_hist = 0

    def _compute(self, mesh, rng: np.random.Generator) -> np.ndarray:
        s = mesh.samples
        idx = distinct_tuples(rng, len(s.points), int(self.params["pairs"]), 2)
        dev = resolve_torch_device(self.device)
        if dev is None:
            i, j = idx[:, 0], idx[:, 1]
            r = np.linalg.norm(s.points[i] - s.points[j], axis=1)
            dots = np.einsum("ij,ij->i", s.normals[i], s.normals[j])
        else:
            r, dots = _dist_dot_torch(s.points, s.normals, idx, dev)
        m = np.clip((r / mesh.diameter * self.dim).astype(np.int64), 0, self.dim - 1)
        # add each pair's agreement to its distance bin, divide by the number of pairs (eq. 13)
        g = np.bincount(m, weights=dots, minlength=self.dim)
        return g / len(idx)

# ==============================================================================
# END OF FILE: ncf_normal_correlation.py
# ==============================================================================
# SUMMARY:
#   NCF: 32 values, normal agreement of point pairs by distance.
#
# USAGE EXAMPLE:
#   from src.descriptors.ncf_normal_correlation import NCFNormalCorrelation
#   v = NCFNormalCorrelation().compute(prepared_mesh, rng)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Uses positions and directions only, not which surfaces are connected.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
#   2026-10-04  optional GPU version (device = cuda)
#   2026-10-21  Plain-language comments and docstrings (no code change)
# ==============================================================================
