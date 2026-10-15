"""
================================================================================
FILE: test_descriptors.py
================================================================================
PURPOSE:
    Correctness tests for the 15 descriptors: output contract (shape, dtype,
    finite), every claimed invariance under random similarity transforms,
    determinism, degenerate inputs, PDF reference values, and CPU/GPU
    equivalence of the D4 GPU path.

INPUTS:
    - Test meshes: trimesh primitives + generated sample-dataset files.

OUTPUTS:
    - pytest results.

DEPENDENCIES:
    stdlib      : math, pathlib
    third-party : numpy, pytest, trimesh
    internal    : tests.conftest, src.descriptors, src.orchestrator

PHYSICS / DOMAIN CONTEXT:
    - Invariance metric: relative L2 distance |a - b| / |a|.
    - Tolerance, one rule for all descriptors: 3 x max(seed noise, input
      noise), floor 1e-4. Seed noise = distance between two sampling seeds
      of the same part (0 for deterministic descriptors). Input noise =
      distance after jittering the vertices by 1e-7 of the part size, i.e.
      float32 STL precision; a transformed STL is only known to that
      precision. Example: LBE moves 4e-4 under such a jitter because
      intrinsic-Delaunay flips tie-break on exactly cocircular quads.
      So an invariance violation must hide inside noise the descriptor
      shows anyway. PAF's normal entropy H lives on a fixed icosphere grid
      and may move by 0.02 (quantisation); its other entries use
      the same rule.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from conftest import all_descriptors, prepare_from_mesh, random_similarity
from src.descriptors import DescriptorError
from src.orchestrator import descriptor_rng

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
PAF_H_INDEX = 4                     # position of the entropy H in the PAF vector
PAF_H_TOL = 0.02                    # [-] icosphere-grid quantisation of H
TOL_FLOOR = 1e-4                    # [-] relative tolerance floor
NOISE_FACTOR = 3.0                  # [-] tolerance = 3 x max(seed, input) noise
JITTER = 1e-7                       # [-] vertex jitter / part size (float32 STL precision)
TRANSFORMS = {
    "rotation": dict(rotate=True, translate=False, scale=False),
    "translation": dict(rotate=False, translate=True, scale=False),
    "scale": dict(rotate=False, translate=False, scale=True),
    "reflection": dict(rotate=True, translate=True, scale=True, reflect=True),
}


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def compute_all(pm, seed: int = 42) -> dict[str, np.ndarray]:
    return {d.name: d.compute(pm, descriptor_rng(seed, d.name)) for d in all_descriptors()}


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b) / (np.linalg.norm(a) + 1e-12))


def primitives() -> dict[str, trimesh.Trimesh]:
    return {
        "box321": trimesh.creation.box([3.0, 2.0, 1.0]),
        "cylinder": trimesh.creation.cylinder(0.3, 2.0, sections=96),
        "torus": trimesh.creation.torus(1.0, 0.3, major_sections=64, minor_sections=32),
        "capsule": trimesh.creation.capsule(height=1.5, radius=0.4, count=[32, 32]),
    }


# ── FIXTURES ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def vectors(tmp_path_factory, sample_dir):
    """
    Vectors for every test mesh: original, second seed, and each transform.

    Returns dict mesh_name -> {"base", "seed2", <transform>: vector dicts}.
    """
    tmp = tmp_path_factory.mktemp("inv")
    meshes = primitives()
    for rel_path in ("washer/washer_00.stl", "bracket/bracket_00.stl", "pin/pin_00.stl"):
        meshes[rel_path.split("/")[0] + "_file"] = trimesh.load(sample_dir / rel_path, force="mesh")
    rng = np.random.default_rng(2026)
    out = {}
    for name, m in meshes.items():
        pm = prepare_from_mesh(m, tmp, name)
        rec = {"base": compute_all(pm), "seed2": compute_all(prepare_from_mesh(m, tmp, name, seed=7), 7)}
        mj = m.copy()
        mj.vertices = mj.vertices + rng.normal(scale=JITTER * float(np.ptp(m.vertices)),
                                               size=mj.vertices.shape)
        rec["jitter"] = compute_all(prepare_from_mesh(mj, tmp, f"{name}_jitter"))
        for tname, kw in TRANSFORMS.items():
            mt = m.copy()
            mt.apply_transform(random_similarity(rng, **kw))
            rec[tname] = compute_all(prepare_from_mesh(mt, tmp, f"{name}_{tname}"))
        out[name] = rec
    return out


# ── TESTS ─────────────────────────────────────────────────────────────────────
def test_output_contract(vectors):
    dims = {d.name: d.dim for d in all_descriptors()}
    assert sum(dims.values()) == 781                       # PDF total
    for rec in vectors.values():
        for variant in rec.values():
            for name, v in variant.items():
                assert v.shape == (dims[name],), name
                assert v.dtype == np.float32, name
                assert np.all(np.isfinite(v)), name


@pytest.mark.parametrize("tname", list(TRANSFORMS))
@pytest.mark.parametrize("dname", [d.name for d in all_descriptors()])
def test_invariance(vectors, dname, tname):
    desc = next(d for d in all_descriptors() if d.name == dname)
    if tname not in desc.invariances:
        pytest.skip(f"{dname} does not claim {tname} invariance")
    for mesh_name, rec in vectors.items():
        a, b = rec["base"][dname], rec[tname][dname]
        mask = np.ones(len(a), dtype=bool)
        if dname == "paf":
            mask[PAF_H_INDEX] = False
            dh = abs(float(a[PAF_H_INDEX]) - float(b[PAF_H_INDEX]))
            assert dh < PAF_H_TOL, f"{mesh_name}: paf H moved {dh:.3f}"
        noise = max(rel(a[mask], rec["seed2"][dname][mask]), rel(a[mask], rec["jitter"][dname][mask]))
        tol = max(NOISE_FACTOR * noise, TOL_FLOOR)
        r = rel(a[mask], b[mask])
        assert r <= tol, (f"{mesh_name}: {dname} {tname} rel={r:.3e} tol={tol:.3e}\n"
                          f"base={np.round(a, 4)}\n{tname}={np.round(b, 4)}")


def test_determinism(tmp_path):
    m = trimesh.creation.torus(1.0, 0.3, major_sections=48, minor_sections=24)
    a = compute_all(prepare_from_mesh(m, tmp_path, "a"))
    b = compute_all(prepare_from_mesh(m, tmp_path, "b"))
    c = compute_all(prepare_from_mesh(m, tmp_path, "c", seed=99), 99)
    for name in a:
        assert np.array_equal(a[name], b[name]), f"{name} not bit-identical"
        r = rel(a[name], c[name])
        # relative bound, or an absolute one for noise-floor vectors (CRS-A of
        # an axisymmetric part is pure sampling noise of size ~1/sqrt(n))
        small = float(np.abs(a[name] - c[name]).max()) < 0.05
        assert r < 0.3 or small, f"{name}: seed change moved the vector by {r:.3f}"
        if name in {"shps", "dad", "t1", "paf"}:
            assert r == 0.0, f"{name} should not depend on the seed"


DEGENERATE = {
    "single_triangle": lambda: trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]),
    "flat_plate": lambda: trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 2, 3]]),
    "open_box": lambda: trimesh.Trimesh(trimesh.creation.box().vertices, trimesh.creation.box().faces[2:]),
    "disconnected": lambda: trimesh.util.concatenate([
        trimesh.creation.box(), trimesh.creation.icosphere(2).apply_translation([5, 0, 0])]),
    "tiny": lambda: trimesh.creation.cylinder(1.0, 2.0).apply_scale(1e-6),
    "huge": lambda: trimesh.creation.cylinder(1.0, 2.0).apply_scale(1e6),
}


@pytest.mark.parametrize("case", list(DEGENERATE))
def test_degenerate_inputs(tmp_path, case):
    pm = prepare_from_mesh(DEGENERATE[case](), tmp_path, case)
    for d in all_descriptors():
        try:
            v = d.compute(pm, descriptor_rng(42, d.name))
        except DescriptorError:
            continue
        assert v.shape == (d.dim,) and np.all(np.isfinite(v)), f"{case}: {d.name}"


def test_zero_area_mesh_raises(tmp_path):
    m = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]], process=False)
    with pytest.raises(DescriptorError):
        prepare_from_mesh(m, tmp_path, "line")


def test_pdf_reference_values(tmp_path):
    """Spot checks against the reference shapes stated in the PDF."""
    get = lambda m, n: compute_all(prepare_from_mesh(m, tmp_path, n))
    cube = get(trimesh.creation.box(), "cube")
    np.testing.assert_allclose(cube["shps"][[0, 2, 4, 6]], [0.0, 0.583, 0.125, 0.516], atol=2e-3)
    assert (cube["dad"][[8, 9]] ** 2).sum() > 0.999                  # all crease mass at 90 deg
    np.testing.assert_allclose(cube["paf"][[2, 3]], [1.0, 1 / 3], atol=1e-6)  # c_s, I4
    assert cube["paf"][7 + 1] == pytest.approx(0.2)                 # 2 planes per family

    sphere = get(trimesh.creation.icosphere(5), "sphere")
    assert np.all(np.abs(sphere["shps"]) < 1e-3)                    # ball: P_l ~ 0
    assert sphere["d1"][-1] ** 2 > 0.95                             # D1 spike at 1
    assert sphere["lbe"][0] == pytest.approx(math.log(2.0), abs=0.02)   # lambda_1 A / 4 pi = 2
    si_mass = (sphere["curvature"][:96].reshape(12, 8) ** 2).sum(axis=1)
    assert int(np.argmax(si_mass)) == 11 and si_mass[8:].sum() > 0.9   # mode at cap (+1), convex

    torus = get(trimesh.creation.torus(1.0, 0.3), "torus")
    assert torus["t1"][1] == pytest.approx(math.log(2.0))           # genus 1

    plate = get(trimesh.creation.box([1.0, 1.0, 0.02]), "plate")
    D = math.sqrt(2.0)                                              # ~ diameter of the plate
    assert plate["mts"][32 + 4] == pytest.approx(math.log10(0.02 / D), abs=0.05)  # median log thickness

    hexprism = get(trimesh.creation.cylinder(1.0, 0.5, sections=6), "hex")
    outer = hexprism["crs_a"].reshape(4, 32)[3]
    assert outer[5] > 5 * outer[4]                                  # harmonic 6 peak


def test_d4_gpu_matches_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from src.descriptors.d4_tetrahedron_volume import D4TetrahedronVolume
    pm = prepare_from_mesh(trimesh.creation.torus(1.0, 0.3), tmp_path, "t")
    a = D4TetrahedronVolume(device="cpu").compute(pm, descriptor_rng(1, "d4"))
    b = D4TetrahedronVolume(device="cuda").compute(pm, descriptor_rng(1, "d4"))
    np.testing.assert_allclose(a, b, atol=1e-6)

# ==============================================================================
# END OF FILE: test_descriptors.py
# ==============================================================================
# SUMMARY:
#   Contract, invariance, determinism, degenerate-input and reference tests.
#
# USAGE EXAMPLE:
#   $ python -m pytest tests/test_descriptors.py -q
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Invariance meshes avoid isotropic covariance (cube, sphere) for the
#         axis-based descriptors; see CLAUDE.md known limitations.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
