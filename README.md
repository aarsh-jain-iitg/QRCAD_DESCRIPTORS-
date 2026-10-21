# QRCAD: 15-Descriptor CAD Retrieval Pipeline

## What This Does
Finds similar parts in a folder of STL files, with no machine learning or
training. Each part is measured in fifteen different ways (the "descriptors",
781 numbers in total, defined in `DOCS/CAD_Descriptors_Explained.pdf`). The
numbers are joined into one list per part, and the parts whose lists are
closest to the query come back first (search via FAISS). Results are saved per
file, so a re-run only measures new or changed files. Every result shows which
descriptors made the match.

## The 15 Descriptors, in Plain Words
Every measurement gives the same answer if the part is moved, turned, mirrored
or scaled, so only the shape counts. "Size" below means the part's overall size.
The example pairs are in `testing files/` (A vs B): each pair differs in the
one way that descriptor is built to notice.

| # | Name | What it measures | Good at telling apart | Example pair |
|---|---|---|---|---|
| 1 | D1 | How far each surface point is from the part's centre | Round vs boxy | R01 ball vs cube |
| 2 | D2 | Distance between many random pairs of surface points | Overall spread and proportions | R02 spider, arms at 90° vs paired |
| 3 | Triangle (A3 + D3) | Angle and size of triangles made from random point triples | Straight vs bent, slender vs stubby | R03 straight vs bowed pin |
| 4 | D4 | Volume of pyramids made from random groups of four points | Flat vs bulky (does the part leave a plane?) | R04alt flat vs cambered strip |
| 5 | SHPS | Which directions the surfaces face, and how regular that pattern is | Plain vs grooved, boxy vs smooth | R05 plain vs V-grooved knob |
| 6 | NCF | Whether near and far surface points face the same way | Solid vs hollow, thin plates | R06 solid vs hollow cube |
| 7 | LBE | The part's natural "ringing" frequencies, as if the surface were a drum | Overall form, slots and openings | R07 plain vs slotted plate |
| 8 | DAD | Sharp edges: their angle (inside or outside corner) and total length | Sharp vs chamfered or rounded edges | R08 sharp vs chamfered block |
| 9 | Curvature | At each point: dome, saddle or dent, and how tightly it curves | Bulging out vs dished in | R09 crown vs dish plate |
| 10 | CRS-R | Distance of the surface from the part's main axes | Turned parts: each diameter shows as a peak | R10 concentric vs eccentric shaft |
| 11 | T1 | Counts: separate pieces, through-holes, sealed hollows, open edges | Through-holes vs blind holes | R11 4 through vs 4 blind holes |
| 12 | PHD | Grows a cylinder out from the axis and records when rings and holes appear and how long they last | Where features sit radially | R12 bolt circle at R25 vs R35 |
| 13 | CRS-A | Repeating patterns around the axis (teeth, bolt holes) | 6 equal holes vs 6 grouped holes, gear tooth counts | R13 equal vs grouped holes |
| 14 | PAF | Groups of parallel flat faces: how many levels, how evenly spaced | Coplanar vs stepped faces | R14 coplanar vs stepped feet |
| 15 | MTS | Wall thickness, by shooting short rays into the material | Thin vs thick walls, shells | R15 dome cover 1 mm vs 2 mm |

Some of these have known weak spots:
- D1–D4 and NCF use random points, so they change very slightly from run to
  run. The random seed is fixed, so the same file always gives the same numbers.
- CRS-R, PHD and CRS-A need a clear main axis. For parts that are equally spread
  in every direction (cube, sphere) the axis choice is arbitrary.
- Curvature cannot see curves on coarse meshes whose triangles are wider than
  2 % of the part size; they read as flat.
- More detail (formulas, settings, design choices) is in each file's header in
  `src/descriptors/` and in `CLAUDE.md`.

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
| File | What it does |
|---|---|
| `main.py` | Run from the command line: update the saved results, search for one part, write the report (`--eval` also scores the search) |
| `app.py`, `static/index.html` | Web app: upload parts, search, see why parts match |
| `src/config.toml` | All settings (which datasets, search method, every descriptor setting) |
| `src/config.py` | Reads the settings and stops with a clear message if one is wrong |
| `src/preprocessing.py` | Loads an STL and cleans it up (fixes gaps, picks surface points, finds size and main axes) |
| `src/descriptors/` | One file per descriptor, named after it (e.g. `d1_radial_distance.py`, `mts_wall_thickness.py`); see the table above |
| `src/descriptors/base.py` | Shared parent class and helpers for all descriptors |
| `src/descriptors/__init__.py` | List of all 15 descriptors, in order |
| `src/cache.py` | Saves each part's numbers so unchanged files are never measured twice |
| `src/orchestrator.py` | Runs everything: measures parts in parallel, joins the numbers, searches |
| `src/database/` | Search engines: plain NumPy, or FAISS (exact or fast-approximate) |
| `src/compression/` | Optional shrinking of the number list (PCA) |
| `src/report.py` | Writes the results table, `report.html` (3D views), `results.png` and `results.json` |
| `src/sample_dataset.py` | Makes a small made-up dataset when `dataset/` is empty |
| `testing.py` | Look at the descriptors of one STL in detail, with plots |
| `generate_ss.py` | Makes slide pictures for every A/B pair in `testing files/` |
| `cosine_table.py` | Builds one HTML table: how similar A and B are, per descriptor |
| `tests/` | Automatic checks: descriptors give the same answer after moving / turning / scaling a part, saving and searching work |
| `dataset/README.md` | How to lay out the dataset folder |
| `CLAUDE.md` | Developer notes: design, settings, known limits, how to add a descriptor |

## Assumptions
- Input is an STL file in any unit (mm, inch, ...). All measurements are
  divided by the part's size, so moving, turning, mirroring or scaling a part
  never changes its numbers.
- Each part is measured on 20,000 points spread evenly over its surface. The
  random choice is fixed, so the same file and settings always give exactly the
  same numbers.
- After cleanup, surface directions are assumed to point outward (the walls of
  sealed inner hollows point inward, which is how T1 counts them).
- When two parts are compared, every descriptor counts fairly: histograms are
  compared by how much they overlap, and plain numbers are first rescaled
  against the whole dataset so no single descriptor dominates.
- The known weak spots are listed under the descriptor table above.
- A GPU gave no speed-up in tests, so everything runs on the CPU by default.

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
| 2026-10-21 | README: plain-language "What This Does" and a new "The 15 Descriptors, in Plain Words" table (what each measures, what it tells apart, example test pair) |
| 2026-10-21 | README: File Map and Assumptions (was Physics / Domain Assumptions) rewritten in plain language |
