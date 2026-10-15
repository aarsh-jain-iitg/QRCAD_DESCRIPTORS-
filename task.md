# TASK: Modular 15-Descriptor CAD Retrieval Pipeline (STL)

## 0. Read before writing any code

Do these in order. Do not write code until all are done.

1. `docs/*.pdf`: the descriptor document. Read it fully. Extract all **15 descriptors**: name, math definition, parameters (bins, samples, resolution, radii), invariances claimed, and any normalization or distance metric the paper specifies. If the PDF gives fewer or more than 15, stop and report the discrepancy before proceeding.
2. `docs/context.md`: project context. Lower relevance, but use it to resolve ambiguity.
3. `skills/` (repo root): read every `SKILL.md`. These are binding coding standards (style, headers, docstrings, README rules, pre-delivery checklist). Where this file and a skill conflict on style, the skill wins.
4. `reference/`: a separate, earlier project showing intent. **Read-only. Do not modify, move, import from, or reformat anything in `reference/`.** Reuse ideas, not files. If you port logic, re-implement it in `src/` under the current standards.
5. `CLAUDE.md` (repo root): current project memory. You will update it at the end (Section 9).

Before coding, write a short plan in your reply: the 15 descriptors as a table (PDF section → file name → output dim → invariances), plus any PDF ambiguity and the interpretation you chose.

## 1. Hard constraints

- Input: STL triangle meshes only. Python 3.11+. GPU is allowed: use it where it gives a real speedup (e.g. PyTorch/CuPy for heavy descriptors, `faiss-gpu` for the index), selected by `pipeline.device` in config (`"auto"` | `"cuda"` | `"cpu"`). Every GPU path must have a CPU fallback that produces the same vectors within tolerance, so the pipeline still runs on a machine without CUDA.
- Use established libraries where they are correct and fast: NumPy, SciPy, trimesh, scikit-learn, scikit-image, FAISS (`faiss-cpu`); add others (e.g. `robust-laplacian`, `gudhi`, `POT`, `open3d`) only if a descriptor needs them. Pin everything in `requirements.txt`.
- Deterministic: every stochastic step takes a seed from config. Same input + same config means a bit-identical vector.
- No hardcoded absolute paths. All paths resolve relative to the repo root via `Path(__file__).resolve()`.
- `logging`, not `print`, inside `src/`. `main.py` may print the final results table.

## 2. Target layout

```
repo/
├── main.py                  # single entry point (Section 7)
├── CLAUDE.md                # update (Section 9)
├── README.md                # update per skills
├── requirements.txt
├── dataset/                 # NEW; user adds STL class folders here
│   └── README.md            # explains: dataset/<dataset_name>/<class_name>/*.stl
├── outputs/                 # generated query reports (gitignored)
├── docs/ reference/ skills/ # existing; do not modify reference/
├── tests/
└── src/
    ├── config.toml          # user-editable settings (Section 5)
    ├── config.py            # loads and validates config.toml into a dataclass
    ├── orchestrator.py      # build, cache, query, fuse (Section 6)
    ├── preprocessing.py     # shared mesh loading, repair, normalization, sampling
    ├── database/            # vector store backends (Section 5)
    ├── compression/         # vector compressor(s) (Section 5)
    ├── report.py            # output rendering (Section 7)
    └── descriptors/         # NEW
        ├── __init__.py      # registry: name -> class
        ├── base.py          # abstract Descriptor interface
        ├── <descriptor_01>.py
        ├── ...
        └── <descriptor_15>.py   # one file per descriptor, 15 total
```

Name each descriptor file after the descriptor in the PDF (snake_case, e.g. `d2_shape_distribution.py`), not `descriptor_01.py`.

## 3. Descriptor interface (`src/descriptors/base.py`)

Every descriptor subclasses one abstract base:

```python
class Descriptor(ABC):
    name: str            # unique registry key, matches config
    version: str         # bump when the math changes; used in cache keys
    dim: int             # output length (fixed, known before compute)
    metric: str          # "l2" | "cosine" | "hellinger" | "chi2" ...
    invariances: set[str]  # subset of {"translation", "rotation", "scale", "reflection"}

    def compute(self, mesh: PreparedMesh, rng: np.random.Generator) -> np.ndarray: ...
```

Rules for each descriptor file:

- Implements exactly one descriptor from the PDF, faithful to its definition. Cite the PDF section and equation in the module docstring.
- Returns a fixed-length `float32` vector of length `dim`. Never NaN or Inf: guard divisions, empty slices, zero-area meshes; raise `DescriptorError` with a clear message if the mesh is unusable.
- Parameters come from `config.toml` under `[descriptors.<name>]`, with PDF values as defaults.
- Histogram-type descriptors: normalize to unit mass, then apply a square-root (Hellinger) map so L2 or inner product in the index approximates the correct histogram distance.
- Shares preprocessing via `PreparedMesh` (vertices, faces, face areas, normals, cached surface samples, optional PCA frame). Do not reload or resample the mesh inside a descriptor unless the definition requires its own sampling.
- Registered in `descriptors/__init__.py` so the orchestrator discovers it by name.

## 4. Preprocessing (`src/preprocessing.py`)

- Load with trimesh (`force="mesh"`); merge duplicate vertices, drop degenerate and zero-area faces, fix winding/normals where possible. Record `is_watertight`.
- Normalize: translate the area-weighted centroid to the origin; scale by a config-selectable rule (`bbox_diagonal` | `rms_radius` | `unit_area`).
- Surface sampling: area-weighted, seeded, `n_samples` from config. Samples are computed once and cached on `PreparedMesh`.
- PCA alignment is **optional** and computed lazily only if a descriptor requests it. Handle near-degenerate eigenvalues and sign ambiguity explicitly and document the choice. Rotation-invariant descriptors must not depend on it.
- A per-file timeout (config) and try/except around each mesh: one bad file logs a warning and is skipped; it never kills the build.

## 5. Config, database, cache, compressor

`src/config.toml`, parsed with stdlib `tomllib` and validated in `config.py` (unknown keys, bad types, and missing paths raise `ValueError` with the key name):

```toml
[paths]
dataset_root = "dataset"          # relative to repo root
datasets     = ["all"]            # or a list of subfolder names under dataset/
cache_dir    = "cache"            # where vector embeddings are cached
output_dir   = "outputs"

[database]
backend  = "faiss_flat"           # "faiss_flat" | "faiss_hnsw" | "numpy"
metric   = "cosine"
top_k    = 10

[compression]
method   = "none"                 # "none" | "pca" | "pq"
target_dim = 128

[pipeline]
seed       = 42
device     = "auto"               # "auto" | "cuda" | "cpu"
n_samples  = 10000
timeout_s  = 30
enabled    = ["all"]              # or a list of descriptor names
fusion     = "weighted_zscore"    # see Section 6
[pipeline.weights]                # optional per-descriptor weights, default 1.0

[descriptors.<name>]              # one table per descriptor, PDF defaults
```

**Database (`src/database/`)**: one abstract `VectorStore` (`add`, `search`, `save`, `load`, `__len__`) with at least three backends: FAISS flat (exact), FAISS HNSW (approximate), and a pure-NumPy brute-force fallback that works if FAISS is missing. Swapping `backend` in config must require no code changes.

**Cache**: per file, per descriptor, store the raw vector in `cache_dir` (e.g. one `.npz` per dataset plus a `manifest.json`). The cache key is `sha256(file bytes) + descriptor.name + descriptor.version + hash(descriptor params)`. Rebuilds are incremental: only new or changed files, or changed descriptors, are recomputed. Deleted STL files are pruned from the cache. Writes are atomic (write to a temp file, then rename).

**Compressor**: you do not *have* to implement a vector compressor right now, but you may. At minimum, create `src/compression/` with a `Compressor` interface and an identity (`none`) implementation wired into the pipeline so one can be added later without refactoring. If you implement one, do PCA (fit on the database, persisted with the cache) and optionally FAISS PQ, and report retrieval-quality loss versus `none` in the evaluation.

## 6. Orchestrator (`src/orchestrator.py`)

- `build(config) -> BuildReport`: discover STL files under the configured datasets, prepare each mesh once, run all enabled descriptors, fill and persist the cache, fit per-descriptor statistics (mean and std over the database) and the compressor, then build the index. Use `concurrent.futures.ProcessPoolExecutor` for per-file work (configurable worker count; fall back to serial on error). Show progress with a counter or `tqdm`.
- `query(path, config) -> QueryResult`: prepare the query mesh, compute the same descriptors with the same params and seed, apply the same normalization and compression, then search.
- Fusion, two modes:
  - `weighted_zscore`: z-score each descriptor block with database stats, multiply by `sqrt(weight)`, concatenate, L2-normalize, and run a single index search.
  - `rank_fusion`: search each descriptor separately and combine with reciprocal rank fusion. Slower, but more robust when descriptors disagree.
- `QueryResult` holds the fused ranking **and** per-descriptor similarity and rank for each hit, so outputs are explainable (which descriptors drove the match).
- The query file is never inserted into the database. If it is already in the database, flag it as an exact match.

## 7. `main.py`

Behaviour on `python main.py`:

1. Load `src/config.toml`.
2. Build or refresh the cache for every dataset in the config (incremental, fast on rerun).
3. Query: a clearly marked variable at the top, `QUERY_FILE = "..."` (relative path), also overridable with `python main.py --query path/to/file.stl`. Optional flags: `--rebuild` (ignore cache), `--top-k`, `--eval`.
4. Output:
   - A terminal table: rank, file, class (parent folder), fused score, and the top-3 contributing descriptors.
   - `outputs/<query_stem>_<timestamp>/report.html`: query and top-k meshes in interactive three.js viewers (STL embedded as base64 so the file is self-contained), plus a per-descriptor score heatmap. It must be openable in VS Code (Live Preview or Open in Browser).
   - `outputs/.../results.png`: a static grid of query and top-k renders (matplotlib `Poly3DCollection`, no GPU or display needed), so results are visible directly in the VS Code image viewer.
   - `outputs/.../results.json`: machine-readable full result.

## 8. Correctness and robustness (double-check everything)

`tests/` with `pytest`:

- **Invariance tests per descriptor**: for each claimed invariance, apply a random SO(3) rotation, translation, uniform scale, and (if claimed) reflection to test meshes (trimesh primitives plus a few dataset files). Assert that vector distance stays under a tolerance, and note tolerances for stochastic descriptors. A descriptor that fails its claimed invariance is a bug, not a tolerance to loosen.
- **Determinism**: same file and seed produce identical vectors; a different seed produces a small bounded change.
- **Sanity**: output shape equals `dim`, dtype is float32, all values finite. Degenerate inputs (single triangle, flat plate, non-watertight, disconnected parts, very small or very large scale) raise `DescriptorError` or return valid vectors, and never crash.
- **Cache**: modifying a file or bumping a descriptor version invalidates exactly the right entries.
- **Backends**: FAISS flat and NumPy return identical top-k on the same data.
- **Evaluation (`--eval`)**: leave-one-out P@1, P@5, P@10, and mAP using class folder names as labels. Report per-descriptor, fused, and with/without compression. Log per-descriptor extraction time (ms/mesh). Put the table in the README.

Before handing back: run the full test suite, run `main.py` end to end on a small sample dataset (generate one from trimesh primitives in `dataset/_sample/` if `dataset/` is empty), run the pre-delivery checklist from `skills/`, and fix everything found.

## 9. Documentation updates

- **`CLAUDE.md`**: add the new architecture (layout tree, data flow build→query), the descriptor table (name → file → dim → metric → invariances → PDF section), config keys, cache-key scheme, how to add a new descriptor (subclass, register, add config table, add invariance test), how to add a DB backend or compressor, and known limitations or open PDF ambiguities.
- **`README.md`**: follow the README rules in `skills/` (What This Does, Setup, Usage, File Map, Domain Assumptions, Changelog row), plus the evaluation table.
- **`dataset/README.md`**: the folder convention and the note that class labels come from folder names.

## 10. Definition of done

- [ ] 15 descriptor files in `src/descriptors/`, each faithful to the PDF, registered, and passing its invariance tests
- [ ] `config.toml` controls datasets, DB backend, cache location, compressor, and descriptor params
- [ ] `python main.py` caches all configured datasets, then queries and writes a terminal table, HTML, PNG, and JSON
- [ ] Rerun with no changes skips recomputation
- [ ] `reference/` untouched (`git diff --stat reference/` is empty)
- [ ] All tests pass; evaluation table produced
- [ ] `CLAUDE.md`, `README.md`, `dataset/README.md` updated; skills checklist passed
