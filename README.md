# QRCAD: 15-Descriptor CAD Retrieval Pipeline

## What This Does
Training-free shape retrieval for STL triangle meshes. Every part is turned into
fifteen space-invariant descriptor blocks (781 numbers) defined in
`DOCS/CAD_Descriptors_Explained.pdf`: point-set statistics (D1, D2, A3/D3, D4),
normal-based spectra (SHPS, NCF), the Laplace–Beltrami spectrum (LBE), edge and
curvature histograms (DAD, SI/CV), and CAD-specific blocks (CRS-R, T1 genus,
persistent homology PHD, CRS-A, planar families PAF, wall thickness MTS). Blocks
are fused into one vector and searched with FAISS. Vectors are cached per file
content, so re-runs only recompute new or changed files. Every result lists which
descriptors drove the match.

## Setup
```bash
python -m pip install -r requirements.txt     # Python >= 3.11 (tested on 3.14, Windows)
```
`embreex` makes the wall-thickness ray casting fast; `faiss-cpu` is optional
(a NumPy backend is used if it is missing); `torch` is only needed for
`pipeline.device = "cuda"`.

## Web app
```bash
python app.py      # opens http://127.0.0.1:5000
```
Add datasets (folder of `class/part.stl`, STL files or zip) → cache builds in the background → drop one STL to search. Each of the top-10 hits shows the descriptors that make it similar (green), those where it differs (red), and all 15 per-descriptor z-scores.

## Usage
```bash
python main.py                                   # build/refresh cache, query QUERY_FILE, write report
python main.py --query dataset/_sample/pin/pin_03.stl --top-k 5
python main.py --eval                            # + leave-one-out P@1/5/10, mAP table
python main.py --rebuild                         # ignore the cache
python -m pytest tests -q                        # test suite
```
- Put data in `dataset/<dataset_name>/<class_name>/*.stl` (see `dataset/README.md`).
  With no STL files, a synthetic sample dataset is generated in `dataset/_sample/`.
- The default query is `QUERY_FILE` at the top of `main.py`.
- Settings live in `src/config.toml` (datasets, backend, cache location,
  compression, fusion, weights, every descriptor parameter).
- Output: a terminal table (rank, file, class, fused score, top-3 descriptors) and
  `outputs/<query>_<timestamp>/` with `report.html` (interactive three.js viewers +
  per-descriptor heatmap; open in a browser or VS Code Live Preview), `results.png`,
  `results.json` (and `evaluation.md/json` with `--eval`).

## Evaluation
Synthetic sample dataset (`dataset/_sample`, 7 classes × 6 parts, random pose and
scale), leave-one-out, class = folder name. P@10 is capped at 0.5 here because each
class has only 5 other members. ms/mesh = mean extraction time per descriptor
(CPU, 8 worker processes in parallel, n = 2e4 samples). These classes are far easier than
real CAD classes; treat the numbers as a smoke test, not a benchmark.

Leave-one-out on 42 models, 7 classes (`python main.py --rebuild --eval`, 2026-09-23).

| Method | P@1 | P@5 | P@10 | mAP | ms/mesh |
|---|---|---|---|---|---|
| d1 | 0.952 | 0.752 | 0.460 | 0.837 | 36 |
| d2 | 0.976 | 0.838 | 0.474 | 0.900 | 129 |
| triangle | 0.952 | 0.852 | 0.460 | 0.900 | 239 |
| d4 | 0.667 | 0.386 | 0.271 | 0.488 | 698 |
| shps | 0.905 | 0.724 | 0.429 | 0.808 | 347 |
| ncf | 0.976 | 0.819 | 0.486 | 0.908 | 195 |
| lbe | 0.929 | 0.767 | 0.469 | 0.828 | 471 |
| dad | 0.881 | 0.824 | 0.483 | 0.881 | 4 |
| curvature | 0.929 | 0.781 | 0.474 | 0.867 | 1401 |
| crs_r | 0.905 | 0.648 | 0.412 | 0.748 | 35 |
| t1 | 0.429 | 0.429 | 0.329 | 0.566 | 9 |
| phd | 0.810 | 0.671 | 0.460 | 0.769 | 52 |
| crs_a | 0.786 | 0.638 | 0.424 | 0.724 | 443 |
| paf | 1.000 | 1.000 | 0.500 | 1.000 | 36 |
| mts | 0.952 | 0.767 | 0.445 | 0.841 | 1717 |
| **fused** (weighted_zscore, no compression) | 1.000 | 1.000 | 0.500 | 1.000 | ≈6.2 s total |
| fused + PCA (41 dims = data rank) | 1.000 | 1.000 | 0.500 | 1.000 | |
| rank_fusion | 1.000 | 0.986 | 0.500 | 0.996 | |

PCA to the data rank loses nothing here; with only 42 models `target_dim = 128`
is clamped to 41, so compression is only meaningful on larger databases.
### Descriptor screenshots for slides
```bash
python generate_ss.py                    # every A/B pair in "testing files", with re-seed noise
python generate_ss.py --pairs R02 R08    # only these pairs
python generate_ss.py --no-noise         # about 2x faster, no noise column
python cosine_table.py                   # -> outputs/descriptor_ss/cosine_table.html (double-click to open)
```
Writes `outputs/descriptor_ss/<pair>/<stl>/d<NN>_<name>.png` (one white-background figure per descriptor), `part.png` (shaded picture of the part), `metrics.csv` (cosine, relative change, re-seed noise, same/different per descriptor) and `manifest.json`. Re-runs reuse `vectors.npz` unless `--force`.

A warm single-process query costs ≈0.5–2 s of extraction and ~1 ms of search.

## File Map
| File | Purpose |
|---|---|
| `main.py` | Entry point: build/refresh cache → query → table + HTML/PNG/JSON (+ `--eval`) |
| `src/config.toml` | User settings; descriptor defaults = PDF values |
| `src/config.py` | tomllib loader, strict validation → `Config` |
| `src/preprocessing.py` | STL → `PreparedMesh`: normalise, weld, repair, samples, D∅, axes |
| `src/descriptors/base.py` | `Descriptor` ABC, Hellinger histograms, tuple sampling |
| `src/descriptors/__init__.py` | Registry name → class (PDF ladder order) |
| `src/descriptors/d1_radial_distance.py` | D1 |
| `src/descriptors/d2_pairwise_distance.py` | D2 |
| `src/descriptors/triangle_a3_d3.py` | A3 + D3 |
| `src/descriptors/d4_tetrahedron_volume.py` | D4 (unsigned volumes), optional GPU path |
| `src/descriptors/shps_orientation_spectrum.py` | SHPS |
| `src/descriptors/ncf_normal_correlation.py` | NCF |
| `src/descriptors/lbe_laplace_beltrami.py` | LBE / ShapeDNA |
| `src/descriptors/dad_dihedral_angle.py` | DAD |
| `src/descriptors/curvature_shape_index.py` | Shape index + curvedness |
| `src/descriptors/crs_r_ring_profile.py` | CRS-R + shared axis rule |
| `src/descriptors/t1_genus.py` | T1 genus / bodies / cavities |
| `src/descriptors/phd_persistent_homology.py` | PHD extended persistence |
| `src/descriptors/crs_a_angular_spectrum.py` | CRS-A |
| `src/descriptors/paf_planar_families.py` | PAF |
| `src/descriptors/mts_wall_thickness.py` | MTS shape-diameter function |
| `testing.py` | Compute selected descriptors (1..15) for one STL; scrollable report page served from memory (no files) |
| `generate_ss.py` | Batch `testing.py` for slides: per-descriptor PNGs + part pictures + A-vs-B metrics for every pair in `testing files` |
| `cosine_table.py` | Static HTML table from `generate_ss.py` output: rows = A/B pairs (part pictures), columns = 15 descriptors, cells = cosine(A, B) |
| `src/cache.py` | Content-addressed incremental vector cache (atomic writes) |
| `src/orchestrator.py` | build / query / evaluate, process pool, fusion |
| `src/database/__init__.py` | `VectorStore` ABC + backend factory |
| `src/database/numpy_store.py` | Brute-force NumPy backend |
| `src/database/faiss_store.py` | FAISS flat (exact) and HNSW backends |
| `src/compression/__init__.py` | `Compressor` ABC, identity and PCA |
| `src/report.py` | Terminal table, report.html, results.png, results.json |
| `src/sample_dataset.py` | Synthetic sample dataset generator |
| `tests/conftest.py` | Shared fixtures (fast params, transforms, config factory) |
| `tests/test_descriptors.py` | Contract, invariance, determinism, degenerate inputs, PDF reference values |
| `tests/test_infrastructure.py` | Config, cache invalidation, backends, compression, end-to-end |
| `dataset/README.md` | Dataset folder convention |
| `CLAUDE.md` | Architecture, descriptor table, cache keys, extension guide, limitations |

## Physics / Domain Assumptions
- Input is an STL triangle mesh in any length unit; every descriptor is dimensionless
  (lengths / D∅, areas / D∅² or A, curvature × √A), so translation, rotation and
  scale drop out; all blocks are also mirror invariant.
- Surfaces are sampled uniformly by area (PDF eq. 2), n = 2·10⁴, seeded: the same
  file + config gives a bit-identical vector.
- Normals are assumed outward after repair (`fix_normals(multibody=False)` keeps
  sealed-cavity walls inward for T1).
- Histogram blocks are compared with the Bhattacharyya similarity (Hellinger map);
  scalar blocks are z-scored with database statistics (PDF eq. 31).
- Axis-based blocks (CRS-R, PHD, CRS-A) need a distinguishable symmetry axis; for
  parts with isotropic covariance (cube, sphere) the axis choice is arbitrary.
- Curvature needs facets finer than 0.02·D∅; coarser tessellations read as flat.
- GPU: measured no gain at PDF sizes, so `device = "auto"` runs on CPU (see CLAUDE.md).

## Changelog
| Date | Change |
|---|---|
| 2026-09-23 | Initial modular 15-descriptor pipeline: config, cache, FAISS/NumPy backends, PCA compressor, fusion, HTML/PNG/JSON reports, evaluation, tests |
| 2026-09-23 | Removed the unused `pdf_section` class attribute from `Descriptor` and all 15 descriptors; PDF section refs replaced by the equations (class comments, config.toml, CLAUDE.md table) |
| 2026-09-25 | Added `testing.py`: interactive per-descriptor inspection/plots for one STL |
| 2026-09-25 | `testing.py`: per-block panels on physical axes + single scrollable local report page |
| 2026-09-27 | `testing.py`: `folder_mode` bool, one report tab per STL in `input_folder` |
| 2026-09-30 | Web app (`app.py`, `static/index.html`): dataset upload, background cache build, single-STL search, top-k with per-descriptor why-similar / why-different; `Hit.per_descriptor` gains `z` |
| 2026-09-30 | D4 v2: tetrahedron volumes are unsigned (never negative); handedness dropped, D4 is now mirror invariant; removed handedness test |
| 2026-09-30 | BLR Phase 0 (`src/blr/`: config, 15-block registry, quality stats, mesh preconditioning, `extract_blocks` / `quality_stats`, `tests/test_blocks.py`; plan in `DOCS/BLR_PLAN.md`). Web app: per-descriptor bars now show cosine similarity coloured by z-score |
| 2026-09-30 | BLR Phases 1-5 (metrics, augmenter, null model, difference report, Stage-1 candidates, evaluation harness, `scripts/blr_eval_ladder.py`, `scripts/blr_compare.py`); Gate 5 reached, H1 not met (see DOCS/BLR_PLAN.md status). PAF v2 fixes an empty-plane crash. Extraction preconditioning capped at 40k faces (about 3x faster) |
| 2026-10-01 | BLR augmentation noise 0.01 D -> 0.001 D; Gate 5 rerun: L3 MRR 0.869 vs L1 0.843 (Holm p 0.059), L1r 0.874; DAD, PAF, CURV, T1 still unreliable (eps_b > 0.2) |
| 2026-10-04 | Added `generate_ss.py`: batch per-descriptor PNGs, part renders, A/B cosine / change / re-seed noise for the descriptor-alignment deck |
| 2026-10-15 | Added `cosine_table.py` (double-clickable A/B cosine table per descriptor); regenerated `outputs/descriptor_ss` with the current descriptors |
| 2026-10-21 | Descriptors: plain-language comments and docstrings in all 15 descriptor files + `base.py` / `__init__.py` (no code change, AST-identical). `SENDING/` and `SENDING.zip` removed from the repo (kept locally, git-ignored) |
