"""
================================================================================
FILE: prior.py
================================================================================
PURPOSE:
    Stage 2/3 of BLR without learning: pair features (block distances, percentiles,
    reliability weights) and the label-free prior score
    S = sum_F (m_eff_F / m_F) sum_{b in F} w_b g0_b(u_b), split per block into
    evidence e_b in nats (sum_b e_b = S).

INPUTS:
    - NullModel, query vector x_q (781,) with quality q_q (6,), candidates X_c (K, 781), Q_c (K, 6).

OUTPUTS:
    - pair_stats -> (D, U, W), each (K, 15); prior_scores -> (S (K,), e (K, 15)).

DEPENDENCIES:
    stdlib      : none
    third-party : numpy
    internal    : blr.metrics, blr.null_model

PHYSICS / DOMAIN CONTEXT:
    - g0_b(u) = log pA_b(u): log-likelihood of the percentile under "same design"
      (the null density of u is uniform, so this is a log likelihood ratio).
    - w_b = omega_b(q) * omega_b(c): a block with w = 0 contributes exactly 0.
    - Blocks in a correlated family are down-weighted by m_eff_F / m_F so that four
      point-statistics do not count as four independent witnesses.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import numpy as np

from src.blr.metrics import all_block_distances
from src.blr.null_model import NullModel


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def pair_stats(model: NullModel, x_q: np.ndarray, q_q: np.ndarray, X_c: np.ndarray,
               Q_c: np.ndarray, D: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Block distances, percentiles and reliability weights of a query against K candidates.

    Parameters
    ----------
    model : NullModel
    x_q : np.ndarray, shape (781,)
    q_q : np.ndarray, shape (6,)
        Quality statistics of the query (quality.QUALITY_KEYS order).
    X_c : np.ndarray, shape (K, 781)
    Q_c : np.ndarray, shape (K, 6)
    D : np.ndarray, shape (K, 15), optional
        Precomputed block distances (skips the distance computation).

    Returns
    -------
    D, U, W : np.ndarray, each shape (K, 15)
    """
    if D is None:
        D = all_block_distances(x_q, X_c, model.space.medians, model.space.scale)
    U = model.space.u(D)
    W = model.omega_of(q_q)[0][None, :] * model.omega_of(Q_c)
    return D, U, W


def prior_scores(U: np.ndarray, W: np.ndarray, model: NullModel) -> tuple[np.ndarray, np.ndarray]:
    """
    Prior score and per-block evidence.

    Parameters
    ----------
    U, W : np.ndarray, shape (K, 15)
    model : NullModel

    Returns
    -------
    S : np.ndarray, shape (K,) [nats]
    e : np.ndarray, shape (K, 15) [nats], S = e.sum(axis=1)
    """
    e = model.family_factor() * W * model.g0(U)
    return e.sum(axis=1), e


# ==============================================================================
# END OF FILE: prior.py
# ==============================================================================
# SUMMARY:
#   pair_stats (D, U, W) and prior_scores (S, e) of BLR level L3.
#
# USAGE EXAMPLE:
#   D, U, W = pair_stats(model, xq, qq, X, Q); S, e = prior_scores(U, W, model)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] None.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
