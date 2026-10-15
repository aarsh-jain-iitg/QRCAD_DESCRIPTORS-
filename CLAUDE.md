# CLAUDE.md: QRCAD 15-descriptor CAD retrieval (project memory)

Training-free STL retrieval: every mesh → 15 space-invariant descriptor blocks
(781 dims, `DOCS/CAD_Descriptors_Explained.pdf`) → fused vector → FAISS search.
Coding standard: `SKILLS/SKILL.md` (header + footer on every .py, PATH VARIABLES
block above CONSTANTS, logging not print in `src/`, README changelog each session).
`reference/` is an older project: **read-only, never import from it.**

## Layout

```
main.py                 entry point: build/refresh cache → query → report (+ --eval)
src/config.toml         all user settings (PDF defaults for descriptor params)
src/config.py           tomllib loader + strict validation → frozen Config
src/preprocessing.py    STL → PreparedMesh (normalise, weld, repair, samples, D_diam, axes)
src/descriptors/        base.py (Descriptor ABC + helpers), __init__.py (REGISTRY), 15 files
src/cache.py            per-dataset content-addressed vector cache (npz + manifest.json)
src/orchestrator.py     build / query / evaluate, process pool, fusion, IndexState
src/database/           VectorStore ABC; numpy_store.py; faiss_store.py (flat, HNSW)
src/compression/        Compressor ABC; identity ("none") and PCA
src/report.py           terminal table, report.html (three.js), results.png, results.json
src/sample_dataset.py   synthetic 7-class dataset used when dataset/ is empty
tests/                  pytest: descriptors (invariance etc.), infrastructure (cache, backends, e2e)
dataset/<ds>/<class>/*.stl   input (class label = parent folder)
cache/                  generated: <ds>/{vectors.npz, manifest.json}, _index/ (fitted index)
outputs/                generated query reports
```

## Data flow

**build(cfg)**: discover `dataset/<ds>/**/*.stl` → sha256 per file (reused when
size+mtime unchanged) → cache key per (file, descriptor) → files with missing
keys go to a `ProcessPoolExecutor` (`extract_file`: `prepare_mesh` once, then each
missing descriptor with `rng = default_rng([seed, crc32(name)])`) → vectors,
timings and failures into the dataset cache → prune stale entries → atomic save →
stack rows of files that have **all** enabled descriptors → per-descriptor
mean/std of the scalar dims → block transform + fusion → compressor fit →
index → persist to `cache/_index/`.

**query(path, cfg)**: load `_index` (signature must match the config) → same
extraction on the query (not inserted) → block transform → fused vector →
compressor → index search (`weighted_zscore`), or reciprocal rank fusion over
exact per-descriptor rankings (`rank_fusion`) → per hit: per-descriptor cosine
+ rank, top-3 descriptors by w·z-score of the similarity, exact-match flag by sha256.

**Block transform (PDF eq. 31):** the first `n_hist` dims of each descriptor are
Hellinger histograms (left as is: inner product = Bhattacharyya); the remaining
scalar dims are z-scored with database stats; then each block is L2-normalised,
multiplied by √w, concatenated and L2-normalised (PDF eq. 31).

## Descriptors

| # | name (config key) | file | dim | n_hist | metric | invariances | equation | points used |
|---|---|---|---|---|---|---|---|---|
| 1 | d1 | d1_radial_distance.py | 32 | 32 | hellinger | T R S M | d̂ᵢ = ‖pᵢ − c‖ / maxⱼ dⱼ | n = 2e4 samples |
| 2 | d2 | d2_pairwise_distance.py | 32 | 32 | hellinger | T R S M | d̂ᵢⱼ = ‖pᵢ − pⱼ‖ / D∅ | S = 2e5 pairs of n = 2e4 |
| 3 | triangle | triangle_a3_d3.py | 64 | 64 | hellinger | T R S M | θ = atan2(‖u×w‖, u·w); Â = ½‖u×w‖ / D∅² | S = 2e5 triples of n = 2e4 |
| 4 | d4 | d4_tetrahedron_volume.py | 33 | 32 | hellinger | T R S M | V = ⅙ \|det[p₂−p₁, p₃−p₁, p₄−p₁]\|; [hist(∛(V/D∅³)), γ₁(V)] | S = 2e5 quadruples of n = 2e4 |
| 5 | shps | shps_orientation_spectrum.py | 15 | 0 | l2 | T R S M | P̃ₗ = 4π/(2ℓ+1) Σₘ c²ₗₘ, ℓ = 2..16 | all faces (exact) |
| 6 | ncf | ncf_normal_correlation.py | 32 | 0 | l2 | T R S M | g(m) = (1/S) Σ (n̂ᵢ·n̂ⱼ) 1[r̂ᵢⱼ ∈ Bₘ] | S = 2e5 pairs of n = 2e4 |
| 7 | lbe | lbe_laplace_beltrami.py | 30 | 0 | l2 | T R S M | L φₖ = λₖ M φₖ; log(λₖ A / 4πk) | all vertices (>= 1000) |
| 8 | dad | dad_dihedral_angle.py | 38 | 36 | hellinger | T R S M | θₑ = π + σₑ arccos(n̂ᵢ·n̂ⱼ); Λ, H = ½ Σ (π − θₑ) ℓₑ / √A | all edges (exact) |
| 9 | curvature | curvature_shape_index.py | 97 | 96 | hellinger | T R S M | SI = (2/π) atan2(κ₁+κ₂, κ₁−κ₂); CV = √(½(κ₁²+κ₂²)) | n = 2e4 samples, ≤ 30 nbrs each |
| 10 | crs_r | crs_r_ring_profile.py | 128 | 128 | hellinger | T R S M | ρ⁽ᵃ⁾ = ‖(p − c) × eₐ‖ / D∅ | n = 2e4 samples |
| 11 | t1 | t1_genus.py | 4 | 0 | l2 | T R S M | gᶜ = ½(2 − χᶜ − bᶜ); [#bodies, log(1+g), #cavities, Σ bᶜ] | whole mesh (exact) |
| 12 | phd | phd_persistent_homology.py | 77 | 75 | hellinger | T R S M | I(x,y) = Σ w(p) exp(−((x−b)² + (y−p)²) / 2σ²) | all vertices (exact) |
| 13 | crs_a | crs_a_angular_spectrum.py | 128 | 0 | l2 | T R S M | \|âq(m)\| / \|âq(0)\|, âq = DFT of azimuth hist | n = 2e5 samples |
| 14 | paf | paf_planar_families.py | 31 | 0 | l2 | T R S M (H ≈) | T₂ = Σ wₜ n̂ₜ n̂ₜᵀ; cₗ, cₚ, cₛ, I₄ | all faces (exact) |
| 15 | mts | mts_wall_thickness.py | 40 | 32 | hellinger | T R S M | wᵢ = medianⱼ τᵢⱼ cos∠(dᵢⱼ, −n̂ᵢ); hist(log₁₀ ŵ) | n = 2e4 samples × 16 rays |

T translation, R rotation, S scale, M reflection. Total 781.

## Config keys (`src/config.toml`)

- `[paths]` dataset_root, datasets (`["all"]` or names), cache_dir, output_dir.
- `[database]` backend (`faiss_flat` | `faiss_hnsw` | `numpy`), metric (`cosine` | `l2`), top_k.
- `[compression]` method (`none` | `pca`), target_dim (clamped to the data rank).
- `[pipeline]` seed, device (`auto`=cpu | `cuda` | `cpu`), n_samples (2e4), timeout_s,
  workers (0 = min(cpu, 8)), scale_rule (`rms_radius` | `bbox_diagonal` | `unit_area`),
  enabled, fusion (`weighted_zscore` | `rank_fusion`), `[pipeline.weights]` name = w.
- `[descriptors.<name>]` params; key set and types must match the class `defaults`.
Unknown keys / wrong types / missing dataset paths → `ValueError` naming the key.

## Cache key scheme

`"<sha256(file bytes)>:<descriptor.name>:<descriptor.version>:<h>"` with
`h = sha256(json({params, pipeline: {seed, n_samples, scale_rule}}))[:16]`.
Content-addressed: renames reuse vectors; edits, version bumps and param changes
invalidate exactly the affected (file, descriptor) entries; entries of deleted files
are pruned on save. Per-descriptor failures are cached under the same key
(retried only with `--rebuild`); preprocessing failures under
`<sha>:preprocess:<fp hash>`; timeouts are not cached. `cache/_index/meta.json`
stores a signature of the config; `query()` refuses a mismatched index.

## How to extend

- **New descriptor:** new file in `src/descriptors/` subclassing `Descriptor`
  (set `name`, `version`, `metric`, `invariances`, `defaults`;
  `_setup` sets `dim`/`n_hist` with histogram dims first; `_compute` returns the raw
  vector, using `mesh.samples` / `mesh.diameter` / `mesh.axes`, not reloading) → add
  the class to `REGISTRY` in `descriptors/__init__.py` → add
  `[descriptors.<name>]` to config.toml → it is picked up automatically by the
  invariance / contract / determinism tests in `tests/test_descriptors.py`
  (add a reference-value check there too). Bump `version` whenever the math changes.
- **New DB backend:** subclass `VectorStore` (add/search/save/load/__len__, scores =
  similarity) → branch in `database.make_store` → add name to `BACKENDS` in config.py
  → add to `test_backends_agree`.
- **New compressor:** subclass `Compressor` (fit/transform/out_dim/save/load) → branch
  in `compression.make_compressor` → add to `COMPRESSIONS` in config.py.

## Decisions, PDF ambiguities, known limitations

- **Axis-based blocks (CRS-R, PHD, CRS-A):** eq.(22) picks no unique axis when the
  sample covariance is isotropic (cube, sphere, regular solids); the eigen-solver's
  choice then leaks into these blocks (not rotation invariant for such parts).
- **Curvature:** radius r = 0.02·D_diam (PDF) capped at 30 nearest neighbours
  (Complexity PDF uses k-NN); flat threshold cCV < 0.1 and log bins on [0.1, 100]
  are my choices (not in the PDF); κ > 0 convex. Facets wider than r read as flat.
- **SHPS:** l = 2..16 including odd l (PDF formula / length 15; prose says even).
- **PHD:** grid b∈[0,0.5], p∈[0,0.25], σ = 0.05, p_max = 0.1 (unspecified in PDF).
  PD0 = ordinary H0, PD1 = ordinary + relative H1, Ext1 = extended H1.
- **PAF:** "planar face" = triangle with no neighbour at a small non-zero fold
  (0.25° < angle < 20°). Entropy H on a fixed icosphere grid is only approximately
  rotation invariant (tests allow 0.02).
- **DAD:** soft (linear) binning (v2): exact 90° creases sit on a 10° bin edge; hard
  bins flipped under rounding (11 % vector change on a rotated box).
- **LBE:** robust_laplacian (intrinsic Delaunay, barycentric lumped mass instead
  of mixed Voronoi); meshes below 1000 vertices are midpoint-subdivided. Sensitive
  at ~4e-4 to float32-level vertex noise (Delaunay ties). No vertex budget → cost
  ~V^1.3 on very fine meshes.
- **MTS:** 16-ray cone pattern (1 + 5 @15° + 10 @30°) oriented toward the centroid,
  so it is rotation/reflection covariant. Needs `embreex` for speed.
- **Fusion:** z-score applies to scalar dims only (histograms stay Hellinger, PDF eq. 31);
  PCA is global on the fused vector, not per block. PQ not implemented.
- **GPU:** only D4 has a torch path; measured no faster than CPU (86 vs 77 ms) plus
  ~5 s CUDA start-up per worker, so `device="auto"` → cpu. faiss-gpu has no Windows wheels.
- **Tests:** invariance tolerance = 3 × max(seed noise, float32-jitter noise), floor 1e-4.
  Translations in tests ≤ 10 part sizes (STL float32 quantisation).
- A file is indexed only if every enabled descriptor succeeded for it.
- The hard per-file timeout kills the pool via the private
  `ProcessPoolExecutor._processes`; a cooperative check also runs between descriptors.
- Evaluation fits z-score stats / PCA on the full database (transductive).

## Active work: BLR re-ranker
- Plan: DOCS/BLR_PLAN.md. Do ONE phase per session. Stop at the phase gate and report.
- Follow SKILLS/SKILL.md for every .py file.
- Never tune anything on class labels unless the plan says so. Class labels are for evaluation only.
- New code goes in src/blr/. Do not edit descriptor extractors except for Phase 0 fixes.
- D4 (v2, 2026-09-30) is unsigned |V| and mirror-invariant (decided): no handedness anywhere; the plan's R04 pair is dropped.
