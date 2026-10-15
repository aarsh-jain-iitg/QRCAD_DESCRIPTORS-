"""
================================================================================
FILE: test_null_model.py
================================================================================
PURPOSE:
    BLR Phase 3: null model fit, p-values, family correction, prior score properties, save/load.

INPUTS:
    - Synthetic data only.

OUTPUTS:
    - pytest results.

DEPENDENCIES:
    stdlib      : dataclasses
    third-party : numpy, pytest
    internal    : src.blr.null_model, src.blr.prior, tests.test_metrics

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.blr.blocks import BLOCKS
from src.blr.config import BLRConfig
from src.blr.null_model import (N_BLOCKS, NullModel, Space, _family_counts, fit_null_model, fit_space)
from src.blr.prior import pair_stats, prior_scores
from src.blr.quality import QUALITY_KEYS
from test_metrics import random_vectors

CFG = dataclasses.replace(BLRConfig(), n_null_pairs=3000)


def _fit(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = random_vectors(n, seed)
    src = np.repeat(np.arange(n), 2)
    Xa = X[src] * (1.0 + 0.01 * rng.standard_normal((len(src), 781)))
    Q = np.abs(rng.standard_normal((n, len(QUALITY_KEYS)))) + 0.1
    Qa = Q[src] * (1.0 + 0.05 * rng.random((len(src), len(QUALITY_KEYS))))
    return X, Q, Xa, Qa, src, fit_null_model(X, Q, Xa, Qa, src, CFG, rng)


def test_fit_shapes_and_ranges():
    X, Q, Xa, Qa, src, m = _fit()
    assert m.U1.shape == (N_BLOCKS, len(src)) and np.all(np.diff(m.U1, axis=1) >= 0)
    assert np.all(m.eps > 0) and np.all(m.beta > 0) and m.space.sigma.shape == (60, N_BLOCKS)
    D, U, W = pair_stats(m, X[0], Q[0], X, Q)
    assert D.shape == U.shape == W.shape == (60, N_BLOCKS)
    assert np.all((U >= 0) & (U <= 1)) and np.all((W >= 0) & (W <= 1))
    assert np.allclose(D[0], 0.0) and U[0].max() < 0.01              # a part is closest to itself


def test_pvalue_bounds():
    _, _, _, _, src, m = _fit()
    M = len(src)
    assert np.allclose(m.pvalue(np.zeros(N_BLOCKS)), 1.0)
    assert np.allclose(m.pvalue(np.ones(N_BLOCKS)), 1.0 / (M + 1))


def test_save_load_round_trip(tmp_path):
    X, Q, _, _, _, m = _fit()
    m.save(tmp_path / "nm")
    r = NullModel.load(tmp_path / "nm")
    D, U, W = pair_stats(m, X[3], Q[3], X, Q)
    D2, U2, W2 = pair_stats(r, X[3], Q[3], X, Q)
    assert np.allclose(U, U2) and np.allclose(W, W2) and np.allclose(m.g0(U), r.g0(U2))
    assert m.m_eff == r.m_eff and np.allclose(m.eps, r.eps) and np.allclose(m.pvalue(U), r.pvalue(U2))


def _synthetic_model(D_null):
    F0 = np.sort(D_null, axis=0).T.copy()
    sp = Space(np.ones(781), np.ones(1), F0, np.zeros((1, N_BLOCKS)), D_null, np.zeros((1, 1, N_BLOCKS), np.float32))
    kx, h = np.array([0.0, 0.5, 1.0]), np.array([1.5, 0.5])
    return NullModel(sp, np.zeros((N_BLOCKS, 1)), [kx] * N_BLOCKS, [h] * N_BLOCKS, np.full(N_BLOCKS, 0.5),
                     np.ones(N_BLOCKS), {}, {}, {}, "")


def test_m_eff_independent_vs_duplicated():
    rng = np.random.default_rng(4)
    ind = _synthetic_model(rng.random((20_000, N_BLOCKS)))
    m_fam, m_eff = _family_counts(ind, ind.space)
    assert abs(m_eff["point"] - m_fam["point"]) < 0.4                # 4 independent blocks
    D = rng.random((20_000, N_BLOCKS))
    for i, b in enumerate(BLOCKS):
        if b.family == "point":
            D[:, i] = D[:, 0]                                        # duplicated blocks
    dup = _synthetic_model(D)
    assert abs(_family_counts(dup, dup.space)[1]["point"] - 1.0) < 1e-6


def test_prior_invalid_block_contributes_zero_and_sum_of_evidence():
    _, _, _, _, _, m = _fit()
    rng = np.random.default_rng(5)
    U, W = rng.random((10, N_BLOCKS)), rng.random((10, N_BLOCKS))
    W[:, 4] = 0.0
    S, e = prior_scores(U, W, m)
    assert np.all(e[:, 4] == 0.0) and np.allclose(S, e.sum(axis=1))


def test_closer_candidate_never_scores_lower():
    """Direction guard: smaller u in every block must not lower the prior score."""
    _, _, _, _, _, m = _fit()
    rng = np.random.default_rng(6)
    U = rng.random((50, N_BLOCKS))
    W = np.ones_like(U)
    S1, _ = prior_scores(U, W, m)
    S2, _ = prior_scores(U * 0.5, W, m)
    assert np.all(S2 >= S1 - 1e-12)


# ==============================================================================
# END OF FILE: test_null_model.py
# ==============================================================================
# SUMMARY:
#   Null model, prior and family-correction tests.
#
# USAGE EXAMPLE:
#   $ python -m pytest tests/test_null_model.py -q
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] None.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
