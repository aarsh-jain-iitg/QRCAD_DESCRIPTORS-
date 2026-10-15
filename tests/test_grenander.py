"""
================================================================================
FILE: test_grenander.py
================================================================================
PURPOSE:
    BLR Phase 3: the Grenander density estimator (shape, normalisation, recovery, uniform limit).

INPUTS:
    - Synthetic data only.

OUTPUTS:
    - pytest results.

DEPENDENCIES:
    stdlib      : dataclasses
    third-party : numpy, pytest
    internal    : src.blr.null_model

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import numpy as np

from src.blr.null_model import grenander, pav_increasing


def _density(kx, h, grid):
    return h[np.clip(np.searchsorted(kx, grid, side="right") - 1, 0, len(h) - 1)]


def test_non_increasing_and_integrates_to_one():
    rng = np.random.default_rng(0)
    for u in (rng.random(500), rng.beta(1, 5, 500), np.full(50, 0.5), np.zeros(30), np.ones(30)):
        kx, h = grenander(u, rng)
        assert np.all(np.diff(h) <= 1e-12) and np.all(h > 0) and np.all(np.isfinite(h))
        assert abs(float(np.sum(h * np.diff(kx))) - 1.0) < 1e-9
        assert kx[0] == 0.0 and kx[-1] == 1.0


def test_recovers_beta_1_5_density():
    rng = np.random.default_rng(1)
    kx, h = grenander(rng.beta(1, 5, 10_000), rng)
    grid = (np.arange(20_000) + 0.5) / 20_000
    err = np.mean(np.abs(_density(kx, h, grid) - 5 * (1 - grid) ** 4))
    assert err < 0.05, err


def test_uniform_gives_flat_density():
    rng = np.random.default_rng(2)
    kx, h = grenander(rng.random(20_000), rng)
    assert abs(h[-1] - 1.0) < 0.15                       # eps ~ 1
    grid = (np.arange(1000) + 0.5) / 1000
    assert np.mean(np.abs(np.log(_density(kx, h, grid)))) < 0.1   # g0 ~ 0


def test_pav_is_monotone_and_keeps_monotone_data():
    rng = np.random.default_rng(3)
    x, y = rng.random(200), rng.random(200)
    _, f = pav_increasing(x, y)
    assert np.all(np.diff(f) >= -1e-12)
    xs = np.arange(10.0)
    assert np.allclose(pav_increasing(xs, xs * 2)[1], xs * 2)


# ==============================================================================
# END OF FILE: test_grenander.py
# ==============================================================================
# SUMMARY:
#   Grenander / PAV tests.
#
# USAGE EXAMPLE:
#   $ python -m pytest tests/test_grenander.py -q
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] None.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
