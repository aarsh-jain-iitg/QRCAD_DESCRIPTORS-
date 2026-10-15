"""
================================================================================
FILE: test_diff_report.py
================================================================================
PURPOSE:
    BLR Phase 3b: weighted Holm, compare() invariants (self, symmetry, one-block
    swap, w = 0, null false-positive rate).

INPUTS:
    - Synthetic data only.

OUTPUTS:
    - pytest results.

DEPENDENCIES:
    stdlib      : dataclasses
    third-party : numpy, pytest
    internal    : src.blr.diff_report, src.blr.null_model, tests.test_metrics

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.blr.blocks import BLOCKS
from src.blr.config import BLRConfig
from src.blr.diff_report import compare, weighted_holm
from src.blr.null_model import fit_null_model, fit_space
from src.blr.quality import QUALITY_KEYS
from test_metrics import random_vectors

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CFG = dataclasses.replace(BLRConfig(), n_null_pairs=4000)
N, A = 300, 2                     # 600 same-design pairs: p_min = 1/601 < alpha / 15


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _setup(seed=0):
    rng = np.random.default_rng(seed)
    X = random_vectors(N, seed)
    src = np.repeat(np.arange(N), A)
    Xa = X[src] * (1.0 + 0.005 * rng.standard_normal((N * A, 781)))
    Q = np.abs(rng.standard_normal((N, len(QUALITY_KEYS)))) + 0.1
    Qa = Q[src] * (1.0 + 0.02 * rng.random((N * A, len(QUALITY_KEYS))))
    space = fit_space(X, CFG, rng)
    return rng, X, Q, Xa, Qa, src, space


def _brute_holm(p, w, alpha):
    """Independent reference: sort by p/w, reject while p_j <= alpha w_j / sum of remaining weights."""
    ids = sorted([i for i in range(len(p)) if w[i] > 0], key=lambda i: p[i] / w[i])
    rej = set()
    for k, i in enumerate(ids):
        if p[i] <= alpha * w[i] / sum(w[j] for j in ids[k:]):
            rej.add(i)
        else:
            break
    return rej


# ── TESTS ─────────────────────────────────────────────────────────────────────
def test_weighted_holm_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(1000):
        p = rng.random(15) ** rng.uniform(1, 6)
        w = np.where(rng.random(15) < 0.2, 0.0, rng.random(15))
        rej, _ = weighted_holm(p, w, 0.05)
        assert set(np.nonzero(rej)[0].tolist()) == _brute_holm(p, w, 0.05)
        assert not np.any(rej & (w == 0))


def test_compare_invariants_and_block_swap():
    rng, X, Q, Xa, Qa, src, space = _setup()
    model = fit_null_model(X, Q, Xa, Qa, src, CFG, rng, space)
    assert len(compare(X[0], X[0], Q[0], Q[0], model).differing) == 0          # identical parts
    rab, rba = compare(X[0], X[1], Q[0], Q[1], model), compare(X[1], X[0], Q[1], Q[0], model)
    assert {d.block for d in rab.differing} == {d.block for d in rba.differing}
    assert {d.block for d in rab.consistent} == {d.block for d in rba.consistent}
    b = 5                                                                       # one block from a far part
    y = X[0].copy()
    y[BLOCKS[b].sl] = X[1][BLOCKS[b].sl]
    rep = compare(X[0], y, Q[0], Q[0], model)
    assert [d.block for d in rep.differing] == [BLOCKS[b].name], rep.summary
    assert rep.differing[0].text and "->" in rep.differing[0].text
    assert rep.to_dict()["differing"][0]["block"] == BLOCKS[b].name


def test_zero_weight_block_is_not_evaluated():
    rng, X, Q, Xa, Qa, src, space = _setup(2)
    model = fit_null_model(X, Q, Xa, Qa, src, CFG, rng, space)
    b = [i for i, s in enumerate(BLOCKS) if s.name == "lbe"][0]
    model.omega[(b, "n_vertices")] = (np.array([-1e9, 1e9]), np.zeros(2))
    rep = compare(X[0], X[1], Q[0], Q[1], model)
    assert [d.block for d in rep.not_evaluated] == ["lbe"] and all(d.block != "lbe" for d in rep.differing)


def test_null_false_positive_rate_is_controlled():
    rng, X, Q, Xa, Qa, src, space = _setup(3)
    model = fit_null_model(X, Q, Xa, Qa, src, CFG, rng, space, train_parts=np.arange(N // 2))
    held = np.nonzero(src >= N // 2)[0]
    hits = sum(len(compare(X[src[i]], Xa[i], Q[src[i]], Qa[i], model).differing) > 0 for i in held)
    rate, se = hits / len(held), np.sqrt(0.05 * 0.95 / len(held))
    assert rate <= 0.05 + 2 * se, rate


# ==============================================================================
# END OF FILE: test_diff_report.py
# ==============================================================================
# SUMMARY:
#   Difference-report tests.
#
# USAGE EXAMPLE:
#   $ python -m pytest tests/test_diff_report.py -q
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] tests/test_diff_report_ladder.py (R01-R15 pairs) needs the STL folder, which is not in the repo.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
