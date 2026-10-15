"""
================================================================================
FILE: test_blocks.py
================================================================================
PURPOSE:
    Phase 0 tests of BLR: registry consistency with the real descriptors, config
    hash, mesh preconditioning, and deterministic 781-d extraction + quality
    statistics on dataset parts (and optional test parts).

INPUTS:
    - Generated sample dataset (conftest.sample_dir).
    - Optional env BLR_TEST_PARTS_DIR: folder with the R01-R15 STL files (5 are used).

OUTPUTS:
    - pytest results.

DEPENDENCIES:
    stdlib      : os, pathlib
    third-party : numpy, pytest, trimesh
    internal    : src.blr.*, tests.conftest

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.blr.blocks import BLOCKS, KINDS, METRICS, N_DIM
from src.blr.config import BLRConfig, cache_dir
from src.blr.extract import extract_blocks, extract_with_quality, precondition_mesh
from src.blr.quality import QUALITY_KEYS
from src.descriptors import REGISTRY, make_descriptors

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SEED = 42
N_PARTS = 5


# ── TESTS: registry ───────────────────────────────────────────────────────────
def test_offsets_contiguous_and_total():
    pos = 0
    for b in BLOCKS:
        assert b.sl.start == pos, b.name
        s = b.sl.start
        for sp in b.subparts:
            assert sp.sl.start == s and sp.sl.stop > sp.sl.start, (b.name, sp.name)
            s = sp.sl.stop
        assert s == b.sl.stop, b.name
        pos = b.sl.stop
    assert pos == N_DIM == 781 and len(BLOCKS) == 15


def test_registry_matches_real_descriptors():
    descs = make_descriptors(list(REGISTRY), {}, "cpu")
    assert [b.name for b in BLOCKS] == [d.name for d in descs]
    for b, d in zip(BLOCKS, descs):
        assert b.sl.stop - b.sl.start == d.dim, b.name
        hist_dims = sum(sp.sl.stop - sp.sl.start for sp in b.subparts if sp.stored_sqrt)
        assert hist_dims == d.n_hist, (b.name, hist_dims, d.n_hist)


def test_subpart_fields_valid():
    for b in BLOCKS:
        assert b.what_it_sees.strip() and b.plain_name.strip() and b.family
        assert b.metric in METRICS
        for sp in b.subparts:
            assert sp.metric in METRICS and sp.kind in KINDS, (b.name, sp.name)
            assert len(sp.labels) == sp.sl.stop - sp.sl.start, (b.name, sp.name)
        for k in b.quality_keys:
            assert k in QUALITY_KEYS, (b.name, k)


def test_config_hash_stable_and_sensitive(tmp_path):
    a, b = BLRConfig(), BLRConfig(alpha_diff=0.01)
    assert a.config_hash() == BLRConfig().config_hash() != b.config_hash()
    assert a.config_hash() in str(cache_dir(tmp_path, a))


# ── TESTS: preconditioning ────────────────────────────────────────────────────
def test_precondition_subdivides_flat_box_and_keeps_geometry():
    box = trimesh.creation.box([2.0, 1.0, 0.5])
    out = precondition_mesh(box, 0.02, 200_000)
    assert len(out.faces) > 1000 and out.is_watertight
    assert abs(out.area - box.area) < 1e-9 and abs(out.volume - box.volume) < 1e-9
    assert out.edges_unique_length.max() <= 0.02 * np.linalg.norm(box.extents) + 1e-12


def test_precondition_respects_face_budget():
    box = trimesh.creation.box([1.0, 1.0, 1.0])
    out = precondition_mesh(box, 1e-6, 500)
    assert len(out.faces) <= 500


# ── TESTS: extraction ─────────────────────────────────────────────────────────
def _parts(sample_dir: Path) -> list[Path]:
    files = sorted(sample_dir.rglob("*.stl"))
    dataset = files[::max(1, len(files) // N_PARTS)][:N_PARTS]
    extra = os.environ.get("BLR_TEST_PARTS_DIR")
    tests = sorted(Path(extra).glob("*.stl"))[:N_PARTS] if extra else []
    return dataset + tests


def test_extract_shape_finite_deterministic(sample_dir):
    parts = _parts(sample_dir)
    assert len(parts) >= N_PARTS
    for p in parts:
        x, q = extract_with_quality(p, SEED)
        assert x.shape == (781,) and x.dtype == np.float32 and np.all(np.isfinite(x)), p.name
        assert set(q) == set(QUALITY_KEYS) and all(np.isfinite(v) for v in q.values()), p.name
        assert 0.0 <= q["boundary_edge_frac"] <= 1.0 and 0.0 <= q["eigen_gap"] <= 1.0
    p = parts[0]
    assert np.array_equal(extract_blocks(p, SEED), extract_blocks(p, SEED))


def test_extract_seed_changes_sampled_blocks_only(sample_dir):
    p = _parts(sample_dir)[0]
    a, b = extract_blocks(p, 1), extract_blocks(p, 2)
    assert not np.array_equal(a, b)
    exact = [bl for bl in BLOCKS if bl.name in ("dad", "t1", "shps", "paf")]
    for bl in exact:                                 # all-face / exact blocks ignore the seed
        assert np.allclose(a[bl.sl], b[bl.sl], atol=1e-6), bl.name


def test_quality_flags_open_mesh(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=3)
    open_m = sphere.copy()
    open_m.update_faces(np.arange(len(open_m.faces)) > 40)     # cut a hole
    open_m.remove_unreferenced_vertices()
    closed, holed = tmp_path / "closed.stl", tmp_path / "holed.stl"
    sphere.export(closed)
    open_m.export(holed)
    assert extract_with_quality(closed, SEED)[1]["boundary_edge_frac"] == 0.0
    assert extract_with_quality(holed, SEED)[1]["boundary_edge_frac"] > 0.0


# ==============================================================================
# END OF FILE: test_blocks.py
# ==============================================================================
# SUMMARY:
#   Registry / config / precondition / extraction tests for BLR Phase 0.
#
# USAGE EXAMPLE:
#   $ python -m pytest tests/test_blocks.py -q
#   $ BLR_TEST_PARTS_DIR=path/to/R_parts python -m pytest tests/test_blocks.py -q
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] R01-R15 STL files are not in the repo; the test-part branch runs only
#         when BLR_TEST_PARTS_DIR is set.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
