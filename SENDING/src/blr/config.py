"""
================================================================================
FILE: config.py
================================================================================
PURPOSE:
    All BLR (Block-aware Likelihood Re-ranker) constants in one frozen
    dataclass, plus a stable hash so cached artefacts are tied to the exact
    settings that produced them.

INPUTS:
    - None (defaults from DOCS/BLR_PLAN.md section 6).

OUTPUTS:
    - BLRConfig; config_hash() -> 12-char hex; cache_dir(dataset_dir) -> Path.

DEPENDENCIES:
    stdlib      : dataclasses, hashlib, json, pathlib
    third-party : none
    internal    : none

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
CACHE_SUBDIR = Path(".qrcad_cache") / "blr"    # <dataset>/.qrcad_cache/blr/<hash>/


# ── CORE CLASSES / MAIN FUNCTIONS ─────────────────────────────────────────────
@dataclass(frozen=True)
class BLRConfig:
    """BLR settings; every number is dimensionless unless a comment says otherwise."""

    n_null_pairs: int = 100_000          # [-] random gallery pairs for F0_b
    n_aug_draws: int = 4                 # [-] augmentations per part
    k_nn_scale: int = 7                  # [-] neighbour index for local scale sigma_b
    n_folds: int = 5                     # [-] part-disjoint folds
    n_brute_max: int = 5_000             # [-] re-rank all candidates at or below this gallery size
    recall_target: float = 0.99          # [-] R@K target used by choose_K
    remesh_edge_frac: float = 0.02       # [-] precondition edge length / bbox diagonal D
    remesh_max_faces: int = 40_000       # [-] precondition never makes a pass that would exceed this face count
    aug_edge_range: tuple[float, float] = (0.5, 2.0)     # [-] remesh edge multiplier
    aug_noise_max: float = 0.001         # [-] vertex noise sigma / D (0.01 destroyed normal-based blocks, see plan status)
    aug_decim_range: tuple[float, float] = (0.25, 1.0)   # [-] kept face fraction
    aug_aniso_range: tuple[float, float] = (0.95, 1.05)  # [-] per-axis scale
    block_dropout: float = 0.15          # [-] per-row probability
    n_hard_neg: int = 100                # [-] hard negatives per training query
    n_rand_neg: int = 20                 # [-] random negatives per training query
    alpha: float = 0.05                  # [-] target false NO_MATCH rate
    alpha_diff: float = 0.05             # [-] family-wise error rate for "differs" (weighted Holm)
    drift_delta: float = 0.01            # [-] KS test level for null refit
    rrf_k: int = 60                      # [-] reciprocal rank fusion constant (Cormack 2009)
    seed: int = 20260930                 # [-] master seed

    def config_hash(self) -> str:
        """SHA-1 (first 12 hex chars) of the sorted JSON of all fields."""
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def cache_dir(dataset_dir: Path, cfg: BLRConfig) -> Path:
    """Artefact folder of one dataset for one config: <dataset>/.qrcad_cache/blr/<hash>."""
    return Path(dataset_dir) / CACHE_SUBDIR / cfg.config_hash()


# ==============================================================================
# END OF FILE: config.py
# ==============================================================================
# SUMMARY:
#   BLRConfig (all constants) + config_hash + per-config cache folder.
#
# USAGE EXAMPLE:
#   cfg = BLRConfig(); cache_dir(Path("dataset/mine"), cfg)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] None.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
