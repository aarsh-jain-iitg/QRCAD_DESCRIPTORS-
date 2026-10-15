"""
================================================================================
FILE: test_metrics.py
================================================================================
PURPOSE:
    BLR Phase 1 tests: metric identities, W1 bin-shift behaviour, vectorised ==
    per-row loop, empty-histogram handling, block combination and timing.

INPUTS:
    - Synthetic vectors only.

OUTPUTS:
    - pytest results (timing is logged).

DEPENDENCIES:
    stdlib      : logging, time
    third-party : numpy, pytest
    internal    : src.blr.blocks, src.blr.metrics

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from src.blr.blocks import BLOCKS, N_DIM
from src.blr.metrics import (N_SUB, all_block_distances, block_distances, hellinger, l2,
                             log_count_l1, subpart_distances, w1_hist)

logger = logging.getLogger(__name__)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def random_vectors(n: int, seed: int = 0) -> np.ndarray:
    """(n, 781) plausible vectors: sqrt-mapped normalised histograms, positive scalars."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, N_DIM))
    for b in BLOCKS:
        for sp in b.subparts:
            if sp.stored_sqrt:
                h = X[:, sp.sl] ** 3
                X[:, sp.sl] = np.sqrt(h / h.sum(axis=1, keepdims=True))
    return X


# ── TESTS ─────────────────────────────────────────────────────────────────────
def test_identity_and_symmetry():
    X = random_vectors(20)
    D = all_block_distances(X[0], X[:1])
    assert np.allclose(D, 0.0)
    a = all_block_distances(X[0], X[1:])
    b = np.stack([all_block_distances(x, X[:1])[0] for x in X[1:]])
    assert np.allclose(a, b, atol=1e-12) and np.all(a >= 0.0)


def test_w1_spikes_scale_linearly_where_hellinger_saturates():
    for k in (1, 3, 7):
        p, r = np.zeros(32), np.zeros(32)
        p[5], r[5 + k] = 1.0, 1.0
        assert w1_hist(p, r, 0.25) == pytest.approx(k * 0.25)
    p, r = np.zeros(32), np.zeros(32)
    p[2], r[4] = 1.0, 1.0
    p2, r2 = np.zeros(32), np.zeros(32)
    p2[2], r2[20] = 1.0, 1.0
    assert hellinger(p, r) == pytest.approx(1.0) == hellinger(p2, r2)


def test_hellinger_range_and_l2():
    rng = np.random.default_rng(1)
    P, R = rng.random((50, 16)), rng.random((50, 16))
    h = hellinger(P, R)
    assert np.all((h >= 0) & (h <= 1)) and np.allclose(hellinger(P, P), 0.0, atol=1e-7)
    assert l2(np.zeros(3), np.array([3.0, 4.0, 0.0])) == pytest.approx(5.0)


def test_zero_sum_histogram_raises():
    with pytest.raises(ValueError):
        w1_hist(np.zeros(8), np.ones(8))
    with pytest.raises(ValueError):
        hellinger(np.ones(8), np.zeros(8))


def test_empty_histograms_in_blocks_are_finite():
    X = random_vectors(3)
    empty = X[0].copy()
    for b in BLOCKS:
        for sp in b.subparts:
            if sp.stored_sqrt:
                empty[sp.sl] = 0.0
    D = subpart_distances(empty, np.stack([empty, X[1]]))
    assert np.all(np.isfinite(D)) and np.allclose(D[0], 0.0) and D[1].max() > 0.0


def test_vectorised_equals_row_loop():
    X = random_vectors(40, seed=3)
    med = np.random.default_rng(5).random(N_SUB) + 0.5
    scale = np.random.default_rng(6).random(N_DIM) + 0.5
    full = all_block_distances(X[0], X[1:], med, scale)
    loop = np.stack([all_block_distances(X[0], X[i:i + 1], med, scale)[0] for i in range(1, 40)])
    assert np.max(np.abs(full - loop)) < 1e-12


def test_block_combination_divides_by_medians():
    sub = np.ones((2, N_SUB))
    med = np.full(N_SUB, 2.0)
    D = block_distances(sub, med)
    n_sub = np.array([sum(1 for sp in b.subparts if sp.metric != "none") for b in BLOCKS])
    assert np.allclose(D, n_sub / 2.0) and D.shape == (2, 15)


def test_log_count_l1_genus_is_not_relogged():
    a, b = np.array([1.0, 0.5, 0.0]), np.array([3.0, 0.5, 0.0])
    assert log_count_l1(a, b, ("bodies", "genus", "cavities")) == pytest.approx(np.log(4) - np.log(2))


def test_timing_k599(caplog):
    X = random_vectors(600)
    t = time.perf_counter()
    for _ in range(20):
        all_block_distances(X[0], X[1:])
    ms = (time.perf_counter() - t) / 20 * 1e3
    logger.info("all_block_distances K=599: %.2f ms", ms)
    print(f"\nK=599 block distances: {ms:.2f} ms")
    assert ms < 50.0


# ==============================================================================
# END OF FILE: test_metrics.py
# ==============================================================================
# SUMMARY:
#   Phase 1 metric tests.
#
# USAGE EXAMPLE:
#   $ python -m pytest tests/test_metrics.py -q -s
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Timing bound is loose (50 ms) so CI machines do not flake; the plan target is 3 ms.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
