"""
================================================================================
FILE: test_infrastructure.py
================================================================================
PURPOSE:
    Tests for everything around the descriptors: config validation, cache
    invalidation (file edit, version bump, deletion), vector-store backend
    equivalence and persistence, compression, and an end-to-end build /
    query / report / evaluate run on a small generated dataset with the
    process pool.

INPUTS:
    - Generated sample dataset (conftest.sample_dir) and temp folders.

OUTPUTS:
    - pytest results.

DEPENDENCIES:
    stdlib      : copy, shutil, tomllib
    third-party : numpy, pytest, trimesh
    internal    : src.cache, src.compression, src.config, src.database,
                  src.descriptors, src.orchestrator, src.report

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import copy
import shutil
import tomllib

import numpy as np
import pytest
import trimesh

from conftest import ROOT_DIR
from src.cache import DatasetCache
from src.compression import make_compressor
from src.config import DEFAULT_CONFIG, config_from_dict
from src.database import make_store
from src.descriptors.d1_radial_distance import D1RadialDistance
from src.orchestrator import build, evaluate, query
from src.report import format_table, write_report

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CHEAP = ["d1", "shps", "t1"]


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def base_config() -> dict:
    return tomllib.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def small_dataset(tmp_path, sample_dir, n: int = 3):
    """dataset/<ds>/<class>/ with n files copied from the sample dataset."""
    root = tmp_path / "data"
    files = sorted(sample_dir.rglob("*.stl"))[:n]
    for f in files:
        dst = root / "ds" / f.parent.name / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dst)
    return root


# ── TESTS: config ─────────────────────────────────────────────────────────────
def test_config_loads_default():
    data = base_config()
    data["paths"]["dataset_root"] = str(ROOT_DIR)
    cfg = config_from_dict(data, ROOT_DIR)
    assert len(cfg.enabled) == 15 and cfg.fingerprint()["n_samples"] == 20000


@pytest.mark.parametrize("mutate, key", [
    (lambda d: d["pipeline"].update(bogus=1), "bogus"),
    (lambda d: d["database"].update(top_k="10"), "top_k"),
    (lambda d: d["paths"].update(dataset_root="does/not/exist"), "dataset_root"),
    (lambda d: d["descriptors"]["d1"].update(bins="32"), "bins"),
    (lambda d: d["descriptors"]["d1"].update(binz=32), "binz"),
    (lambda d: d["pipeline"].update(enabled=["d9"]), "enabled"),
    (lambda d: d["database"].update(backend="annoy"), "backend"),
])
def test_config_rejects(mutate, key):
    data = base_config()
    data["paths"]["dataset_root"] = str(ROOT_DIR)
    mutate(data)
    with pytest.raises(ValueError, match=key):
        config_from_dict(data, ROOT_DIR)


# ── TESTS: cache ──────────────────────────────────────────────────────────────
def test_cache_invalidation(tmp_path, sample_dir, make_cfg, monkeypatch):
    root = small_dataset(tmp_path, sample_dir, 3)
    cfg = make_cfg(root, enabled=CHEAP)

    r1 = build(cfg)
    assert (r1.n_indexed, r1.n_files_extracted, r1.n_vectors_computed) == (3, 3, 9)

    r2 = build(cfg)                                     # unchanged: nothing recomputed
    assert (r2.n_files_extracted, r2.n_vectors_computed) == (0, 0)

    victim = sorted(root.rglob("*.stl"))[0]             # changed file: its 3 vectors only
    trimesh.creation.box([1.0, 2.0, 3.0]).export(victim)
    r3 = build(cfg)
    assert (r3.n_files_extracted, r3.n_vectors_computed) == (1, 3)

    monkeypatch.setattr(D1RadialDistance, "version", "99")   # version bump: d1 on every file
    r4 = build(cfg)
    assert (r4.n_files_extracted, r4.n_vectors_computed) == (3, 3)

    victim.unlink()                                     # deletion: pruned
    r5 = build(cfg)
    assert (r5.n_files, r5.n_vectors_computed) == (2, 0)
    c = DatasetCache.load(cfg.cache_dir / "ds")
    assert len(c.files) == 2 and len(c.vectors) == 2 * len(CHEAP)


def test_cache_param_change_invalidates_only_that_descriptor(tmp_path, sample_dir, make_cfg):
    root = small_dataset(tmp_path, sample_dir, 2)
    build(make_cfg(root, enabled=CHEAP))
    data_cfg = make_cfg(root, enabled=CHEAP)
    params = copy.deepcopy(data_cfg.descriptor_params)
    params["d1"] = {"bins": 16}
    cfg2 = type(data_cfg)(**{**data_cfg.__dict__, "descriptor_params": params})
    r = build(cfg2)
    assert (r.n_files_extracted, r.n_vectors_computed) == (2, 2)


# ── TESTS: backends / compression ─────────────────────────────────────────────
@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_backends_agree(tmp_path, metric):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 48)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = X[:20] + 0.05 * rng.normal(size=(20, 48)).astype(np.float32)
    flat, ref, hnsw = (make_store(b, 48, metric) for b in ("faiss_flat", "numpy", "faiss_hnsw"))
    for s in (flat, ref, hnsw):
        s.add(X)
    _, i_flat = flat.search(Q, 10)
    s_ref, i_ref = ref.search(Q, 10)
    np.testing.assert_array_equal(i_flat, i_ref)
    _, i_h = hnsw.search(Q, 10)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(i_h, i_ref)])
    assert recall >= 0.9
    for name, s in (("flat", flat), ("np", ref)):              # save / load round trip
        s.save(tmp_path / f"{name}.bin")
        t = make_store("faiss_flat" if name == "flat" else "numpy", 48, metric)
        t.load(tmp_path / f"{name}.bin")
        assert len(t) == 400
        np.testing.assert_array_equal(t.search(Q, 10)[1], i_ref)


def test_pca_compressor_roundtrip(tmp_path):
    X = np.random.default_rng(1).normal(size=(50, 30))
    c = make_compressor("pca", 8).fit(X)
    Z = c.transform(X)
    assert Z.shape == (50, 8)
    np.testing.assert_allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)
    c.save(tmp_path / "c.npz")
    c2 = make_compressor("pca", 8)
    c2.load(tmp_path / "c.npz")
    np.testing.assert_allclose(c2.transform(X), Z, atol=1e-6)


# ── TESTS: end to end ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("fusion, compression", [("weighted_zscore", "none"),
                                                  ("rank_fusion", "none"),
                                                  ("weighted_zscore", "pca")])
def test_end_to_end(tmp_path, sample_dir, make_cfg, fusion, compression):
    cfg = make_cfg(sample_dir.parent, workers=2, fusion=fusion)
    cfg = type(cfg)(**{**cfg.__dict__, "compression": compression, "target_dim": 8})
    rep = build(cfg)
    assert rep.n_indexed == rep.n_files == 14
    q = sample_dir / "torus" / "torus_01.stl"
    res = query(q, cfg, top_k=5)
    assert res.query_in_database
    assert res.hits[0].exact_match and res.hits[0].label == "torus"
    assert len(res.hits) == 5 and len(res.hits[0].top_descriptors) == 3
    assert set(res.hits[0].per_descriptor) == set(cfg.enabled)
    assert "torus_01" in format_table(res)
    out = write_report(res, ROOT_DIR, tmp_path / "report")
    for f in ("report.html", "results.png", "results.json"):
        assert (out / f).stat().st_size > 0
    ev = evaluate(cfg)
    assert ev["n_models"] == 14 and 0.0 <= ev["metrics"]["fused"]["mAP"] <= 1.0

# ==============================================================================
# END OF FILE: test_infrastructure.py
# ==============================================================================
# SUMMARY:
#   Config, cache invalidation, backends, compression and end-to-end tests.
#
# USAGE EXAMPLE:
#   $ python -m pytest tests/test_infrastructure.py -q
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] None.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
