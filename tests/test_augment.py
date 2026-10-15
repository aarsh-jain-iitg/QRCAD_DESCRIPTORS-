"""
================================================================================
FILE: test_augment.py
================================================================================
PURPOSE:
    BLR Phase 2: the augmenter keeps a part recognisable, is deterministic and
    never mirrors.

INPUTS:
    - trimesh primitives.

OUTPUTS:
    - pytest results.

DEPENDENCIES:
    stdlib      : none
    third-party : numpy, trimesh
    internal    : src.blr.augment, src.blr.config

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import numpy as np
import trimesh

from src.blr.augment import augment
from src.blr.config import BLRConfig


def test_augmented_mesh_is_valid_and_similar():
    m = trimesh.creation.box([2.0, 1.0, 0.5])
    for seed in range(6):
        a = augment(m, np.random.default_rng(seed), BLRConfig())
        assert np.all(np.isfinite(a.vertices)) and len(a.faces) > 0
        assert 0.7 < abs(a.volume) / m.volume < 1.4        # noise + decimation + anisotropic scale


def test_deterministic_for_a_seed():
    m = trimesh.creation.box([2.0, 1.0, 0.5])
    a = augment(m, np.random.default_rng(7), BLRConfig())
    b = augment(m, np.random.default_rng(7), BLRConfig())
    assert np.array_equal(a.vertices, b.vertices) and np.array_equal(a.faces, b.faces)


def test_never_mirrors():
    """Signed volume stays positive: rotation and positive scaling only, no reflection."""
    m = trimesh.creation.icosphere(subdivisions=2)
    for seed in range(10):
        assert augment(m, np.random.default_rng(seed), BLRConfig()).volume > 0


# ==============================================================================
# END OF FILE: test_augment.py
# ==============================================================================
# SUMMARY:
#   Augmenter tests.
#
# USAGE EXAMPLE:
#   $ python -m pytest tests/test_augment.py -q
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] None.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
