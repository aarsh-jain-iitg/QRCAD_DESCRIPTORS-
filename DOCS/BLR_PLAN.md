# BLR: Block-aware Likelihood Re-ranker, implementation plan for Claude Code

Spec: "Block-Aware Learned Re-ranking for QRCAD" (Claude Doc). Project summary: `claude/blr-reranker-spec.md`.
Target descriptor set: the **15-block space-invariant set, 781-d**. **Decision 2026-09-30: D4 volumes are unsigned (|V|), so D4 is mirror-invariant (D4 v2). There is no handedness signal in any block; left/right-hand pairs (R04) are not distinguishable and are dropped from the ladder acceptance.** Mode: **design first** (label-free); family mode is a later phase.
Coding standard: `aarsh-coding-style` (in this repo: `SKILLS/SKILL.md`). That means a header and footer on every `.py` file, a PATH VARIABLES block above CONSTANTS, NumPy docstrings with units and array shapes, type hints, `logging` instead of print, `unittest`, and a README update at the end of every session.

---

## 0. How to use this file in Claude Code

1. Keep this file at `DOCS/BLR_PLAN.md` (the repo's docs folder is `DOCS/`).
2. `CLAUDE.md` at the repo root carries the "Active work: BLR re-ranker" section.
3. Start each session in **plan mode** with this prompt:
   ```text
   Read DOCS/BLR_PLAN.md and CLAUDE.md. Execute Phase <N> only.
   First list the files you will create or edit and the tests you will add, then wait for my OK.
   Finish with: tests green, the phase gate checked, a README changelog row, and ticked boxes in DOCS/BLR_PLAN.md.
   ```
4. Tick the `- [ ]` boxes as you go. This file is the single source of progress.
5. Phase 5 is a **decision gate**. Do not start Phase 6 until the L0–L3 results have been reviewed.

> **Status 2026-10-01 (Gate 5 reached, Phase 6 NOT started).** 299 parts x 2 augmentations of `dataset/small_set`, label-free design protocol only (no class labels exist).
> Results: `results/ladder_L0_L3.md`, `phase2_aug_sanity.md`, `phase4_recall.md`, `g0_curves.png`.
> Augmentation noise lowered from 0.01 D to 0.001 D (0.01 destroyed the normal-based blocks; first run had L3 0.467 < L1 0.479).
> After the fix: MRR L0 0.774, L1 0.843, L1r 0.874, L3 0.869. H1 (L3 > L1): raw Wilcoxon p = 0.030, Holm p = 0.059 over the 5 listed comparisons
> (L2's best exponent is p = 1, i.e. L2 = L1, so that family contains a duplicate; with the 4 distinct comparisons Holm p = 0.030): borderline. L1r vs L3 not tested.
> Blocks with eps_b > 0.2: DAD 0.97, PAF 0.67, CURV 0.58, T1 0.27. Latency S2+S3 2.3 ms (K = N = 299), 1.3 ms (K = 100). Stage-1 R@K = 1.0 for K >= 25.
> Not done: R01-R15 ladder acceptance test (STL files absent), Phase 8 (needs labels / MCB-A).

---

## 1. What is being built (one screen)

```
query mesh ──► S0 extract 15 blocks + quality stats
          ──► S1 candidates: brute force if N ≤ 5000, else FAISS weighted-L2 top-K ∪ per-block top-⌈K/B⌉
          ──► S2 pair features per block: D_b, u_b, ℓ_b, z_b, ρ_b, w_b
          ──► S3 score  S = Σ_F (m_eff_F/m_F) Σ_{b∈F} w_b·g0_b(u_b)  +  λ·Σ_b f_b(φ_b)
          ──► S4 conformal NO_MATCH gate + per-block evidence e_b in nats (Σ_b e_b = S)
          ──► S5 DIFFERENCE REPORT: every block whose distance is too large for "same design",
                 each with a verdict, a p-value, its evidence and a plain-language, numeric
                 description of what changed (for example "DAD: crease angle 90° → 135°")
```

| Symbol | Definition | Shape / unit |
|---|---|---|
| `D_b` | block distance (metric per block, §3) | (K,), ≥ 0, smaller means more similar |
| `u_b` | `F0_b(D_b)`: ECDF of D_b over random gallery pairs | (K,), [0,1] |
| `ℓ_b` | `D_b / sqrt(σ_b[q]·σ_b[c])`, where σ_b is the 7th-NN distance | (K,), dimensionless |
| `z_b` | `(D_b − median_C D_b) / (1.4826·MAD_C D_b + eps)` over the candidate list | (K,) |
| `ρ_b` | rank of c among C under block b, divided by K | (K,), (0,1] |
| `w_b` | `ω_b(q)·ω_b(c)`: block reliability | (K,), [0,1] |
| `g0_b` | `log pA_b(u)`: Grenander density of augmentation-pair percentiles | nats |
| `p_b` | `(1 + #{a : U1_b(a) ≥ u_b}) / (\|A_b\| + 1)`: how often a same-design pair is at least this far apart in block b | [0,1] |
| `ε_b` | `pA_b(1⁻)`: flat tail height, which is the block's failure rate | [0,1] |
| `m_eff_F` | `m_F² / (1ᵀ R_F 1)`, where R_F is the correlation of g0 within the family over random pairs | count |

---

## 2. Target file layout

```
src/blr/
  __init__.py
  config.py          BLRConfig dataclass (all constants, units), config_hash()
  blocks.py          BlockSpec registry for the 15 blocks: slices, sub-parts, metric, family, quality stat
  metrics.py         vectorised distances: w1_hist, l2, hellinger, log_count_l1, combine_subparts
  quality.py         per-part quality statistics
  augment.py         same-design augmenter T(p) at mesh level
  null_model.py      F0_b, σ_b, Grenander, ε_b, g0_b, ω_b, m_eff_F, β_b  (fit/save/load)
  candidates.py      Stage 1: brute or FAISS, per-block union, choose_K()
  pair_features.py   φ(q, C) → array (K, B, 5) + w (K, B)
  prior.py           S_prior and per-block e_b (L3)
  baselines.py       L0, L1, L1r (RRF), L2 (generalised mean), L3 = prior
  gam.py             LightGBM GA2M fit, compile to lookup tables, table predict (L4)
  conformal.py       τ_α fit / apply
  diff_report.py     compare(a, b) -> DiffReport: per-block verdict, p-value, evidence, "what changed" text
  engine.py          BLREngine.fit() / .query() / .add()
  evaluate.py        mAP, nDCG@10, P@5/10, R@K, MRR, Wilcoxon + Holm, class-cluster bootstrap
scripts/
  blr_build_null.py  extract gallery + augmentations, fit null model
  blr_eval_ladder.py run L0–L3 (and L4 later), write results JSON + markdown table
  blr_fit_gam.py     fit L4 with part-disjoint folds
  blr_compare.py     difference report for any two STL files
tests/
  test_blocks.py test_metrics.py test_grenander.py test_null_model.py
  test_pair_features.py test_prior.py test_gam_compile.py test_conformal.py test_engine_smoke.py
  test_diff_report.py test_diff_report_ladder.py
```

Artifacts go under `<dataset>/.qrcad_cache/blr/<config_hash>/` as `.npz` and `.json` files. Never pickle model objects: LightGBM goes out as `model.txt` plus the compiled `tables.npz`.

Third-party packages, each justified: `numpy`, `scipy` (Wilcoxon, `scipy.spatial.transform.Rotation`), `trimesh` (already used), `faiss-cpu` (already used), `lightgbm` (Phase 6 only), and `pymeshlab` (isotropic remesh and decimation; check whether it is installed, and fall back to `trimesh.remesh.subdivide_to_size` plus `fast_simplification`).

---

## 3. Block registry (fill offsets from the real extractor)

The table below gives the **expected** layout. Phase 0 must read the actual 15-block extractor and correct the offsets. `assert sum(d_b) == 781`.

| # | Block | d_b | Sub-parts | Metric (D_b) | Family | Quality stat → ω_b |
|---|---|---|---|---|---|---|
| 1 | D1 | 32 | hist | W1 | point | — (ω = 1) |
| 2 | D2 | 32 | hist | W1 | point | — |
| 3 | TRI (A3+D3) | 64 | A3 hist 32, D3 hist 32 | W1 each, combined | point | — |
| 4 | D4 | 33 | \|V\| hist 32, skew γ₁ of unsigned volumes 1 | W1 + abs diff | point | — |
| 5 | SHPS | 15 | spectrum | L2 | normal | — |
| 6 | NCF | 32 | g(m) profile | L2 | normal | — |
| 7 | PAF | 31 | Westin, I4, families, offsets | L2 on null-robust-scaled entries | normal | — |
| 8 | LBE | 30 | spectrum | L2 | intrinsic | n_vertices, boundary-edge fraction |
| 9 | DAD | 38 | hist 36, crease length, total H | W1 + \|Δ log\| scalars | local | boundary-edge fraction, facet angle |
| 10 | CURV | 97 | joint SI×CV 12×8 hist, flat fraction | Hellinger + abs diff | local | boundary-edge fraction |
| 11 | CRS-R | 128 | hists 64, 32, 32 | W1 each | axis | eigen-gap γ |
| 12 | CRS-A | 128 | 4 bands × 32 harmonics | L2 per band | axis | eigen-gap γ |
| 13 | PHD | 77 | persistence image | L2 | axis | eigen-gap γ, boundary-edge fraction |
| 14 | T1 | 4 | bodies, log(1+g), cavities, boundary loops | L1 on the first 3 (log1p counts) | topology | boundary loops (the 4th entry is a quality stat, **not** a distance input) |
| 15 | MTS | 40 | log₁₀-thickness hist, f_thin, f_miss | W1 + \|Δ f_thin\| | thickness | f_miss |

**Plain-language meaning (goes into `BlockSpec.what_it_sees`, used by the difference report):**

| Block | A difference means… | Axis / index label for the detail line |
|---|---|---|
| D1 | the surface sits at different distances from the centre (solid vs hollow, round vs square) | distance from centre / D∅ |
| D2 | different overall proportions or spacing between features (arm layout, elongation) | point-pair distance / D∅ |
| TRI | the part bends differently (straight vs bowed), different local triangle shapes | A3 angle [deg]; D3 √area / D∅ |
| D4 | different 3-D thickness or camber; (unsigned volumes: LH and RH parts look identical to D4) | ∛(V / D∅³); γ₁ |
| SHPS | surfaces face in different directions (grooves, knurls, facets) | harmonic order ℓ |
| NCF | thin walls vs solid material (opposing normals across a wall) | pair distance bin m |
| PAF | different planar faces and datum planes (coplanarity, step heights) | entry name (Westin c_l/c_p/c_s, I4, family count, offset gap) |
| LBE | different connectivity: slots, flexures, separate lobes | eigenvalue index k |
| DAD | different edge treatment (sharp vs chamfer, crease angles, crease length) | dihedral angle [deg]; crease length / D∅; total H |
| CURV | curved regions differ: convex vs concave, fillets, sphere vs cylinder | (shape index SI, curvedness CV) cell; flat fraction |
| CRS-R | turned features have different radii or are not concentric | radius from symmetry axis / D∅ |
| CRS-A | features arranged in a different angular pattern (hole pitch, grouping) | band r, harmonic m |
| PHD | holes or loops sit at different positions or sizes around the axis | persistence-image pixel (birth, persistence) / D∅ |
| T1 | different number of bodies, through-holes (genus) or internal cavities | count name |
| MTS | different wall thickness | log₁₀(thickness / D∅); f_thin |

`SubPart` also carries `kind ∈ {hist, hist2d, spectrum, scalar, count}`, `labels` (the bin centres or index names, in axis units) and `unit`. Phase 0 fills these from the extractor.

**Sub-part combination (no hand weights):** `D_b = Σ_j D_bj / median_R(D_bj)`, i.e. each sub-distance is divided by its median over random pairs. The medians are stored in the null model.

**Metric definitions (smaller means more similar):**
- `W1(p, r) = Δt · Σ_k |cumsum(p − r)_k|`, with `Δt` the bin width in the histogram's axis units. Renormalise each histogram to sum to 1 first (guard a zero sum by raising `ValueError`).
- `hellinger(p, r) = sqrt(max(0, 1 − Σ_k sqrt(p_k·r_k)))`.
- `l2(a, b) = ‖a − b‖₂`.
- Everything is vectorised: `metric(x_q: (d,), X_c: (K, d)) -> (K,)`.

---

## 4. TODO by phase (each with its how-to and a gate)

### Phase 0: Recon and plumbing
- [x] Find the 15-block extractor in the repo. It may not live in `src/features.py`, which is the old 960-d app. Record the module, the function name and the exact output layout in `blocks.py`.
      *How:* grep for `SHPS`, `NCF`, `MTS`, `PHD`. If an extractor is **missing**, stop and report which ones. Do not write extractors inside this plan.
- [x] `blocks.py`:
  - `@dataclass(frozen=True) BlockSpec(name, plain_name, what_it_sees, sl: slice, subparts: tuple[SubPart,...], metric: str, family: str, quality_keys: tuple[str,...])`;
  - `SubPart(name, sl, kind, labels: tuple, unit: str, metric: str)`;
  - a `BLOCKS` tuple in the order of §3.

  *Test:* the offsets are contiguous, non-overlapping and sum to 781; every metric name resolves; `len(labels)` matches the sub-part width; every block has a non-empty `what_it_sees`.
- [x] `config.py`: a `BLRConfig` dataclass holding every constant in §6, plus `config_hash()` (SHA-1 of the sorted JSON). The cache path includes the hash.
- [x] Plumbing bugs: not applicable in this repo (per-descriptor seeds `default_rng([seed, crc32(name)])`, cache keys carry the config hash, no `pca.py`). Notes 2026-09-30: extractor = `src.descriptors.REGISTRY` + `src.preprocessing`, wrapped by `src/blr/extract.py`; block order = registry order (d1..mts); `precondition_mesh` is a uniform 1:4 subdivision (pymeshlab not installed). Original bullets:
  - `features.py` reuses seed 42 in every function. Make the seed a parameter.
  - `cache.py` has no config key.
  - `pca.py` Step 9 can mirror the part. Log it; the fix belongs to another ticket unless it is trivial.
- [x] **Mesh preconditioning:** parts with fewer than 31 eigenvalues' worth of vertices fail LBE (the flat test parts export with 12–76 triangles). Add `precondition_mesh(mesh, edge_max=0.02·D∅)` (isotropic remesh) before extraction, and record `n_vertices` as a quality stat.
- **Gate 0 (status 2026-09-30: passed on 5 sample-dataset parts, 9 tests green; the 5 R01-R15 test parts are not in the repo, so that half runs when `BLR_TEST_PARTS_DIR` is set):** `extract_blocks(mesh_path: Path, seed: int) -> np.ndarray (781,)` and `quality_stats(mesh_path) -> dict[str, float]` run on 5 dataset parts and 5 test parts (R01–R15 STLs), deterministically for a fixed seed. `tests/test_blocks.py` passes.

### Phase 1: Metrics
- [x] `metrics.py`: `w1_hist`, `l2`, `hellinger`, `log_count_l1`, and `block_distance(spec, x_q, X_c, medians) -> (K,)`.
- [x] Tests (`test_metrics.py`):
  - identity gives 0;
  - symmetry;
  - W1 of two unit spikes k bins apart equals `k·Δt` (fixes the bin-shift problem: cosine and Hellinger give the maximum distance here, W1 grows linearly);
  - Hellinger is in [0,1];
  - the vectorised result equals a per-row loop to 1e-12;
  - a zero-sum histogram raises `ValueError`.
- **Gate 1:** all tests green. Timing is ≤ 3 ms for K = 599 over the 15 blocks (log it).

### Phase 2: Augmenter and quality statistics
- [x] `augment.py`: `augment(mesh, rng) -> trimesh.Trimesh` applying, in order:
  1. isotropic remesh at edge × U(0.5, 2);
  2. a uniform random SO(3) rotation (`Rotation.random`);
  3. vertex noise N(0, σ) with σ ~ U(0, 0.001)·D∅;
  4. with probability 0.5, decimation to a U(0.25, 1) face fraction;
  5. anisotropic scale diag(U(0.95, 1.05)³);
  6. a new extraction seed.

  Mirroring is not part of the augmenter (a mirrored part is a different design for the user, even though every block is now mirror-invariant).
- [x] `quality.py`:
  - `boundary_edge_frac` (edges with a single incident face / all edges);
  - `n_vertices`;
  - `eigen_gap` γ = min_{i≠j} |λ_i − λ_j| / λ₁ of the covariance;
  - `facet_angle` (median dihedral over non-crease edges);
  - `f_miss` (from MTS);
  - `boundary_loops` (T1 entry 4).
- [x] Tests:
  - an augmented part's blocks lie closer to the original than the median random pair for ≥ 90 % of the 600 parts, per block; log the failing blocks, since they are the ε_b candidates;
  - rotation invariance: `extract(R·mesh)` vs `extract(mesh)` gives D_b below the 5th percentile of the null for every block.
- **Gate 2:** a per-block table of the median augmentation percentile, saved to `results/phase2_aug_sanity.md`.

### Phase 3: Null model (label-free, gives L1–L3)
- [x] `null_model.py`, function `fit_null_model(X, Q, X_aug, pairs_aug, cfg) -> NullModel`:
  1. **Random pairs R:** `N_NULL_PAIRS` pairs (i ≠ j), seeded. Compute D_b and the sub-part medians.
  2. **F0_b:** ECDF, stored as sorted distances. `u = searchsorted(right) / n`. For ties use a mid-rank.
  3. **σ_b[p]:** the 7th-NN distance of every gallery part in block b (brute force N×N per block is fine at N = 600; above that, use FAISS on the block).
  4. **U1_b:** `F0_b(D_b(p, T(p)))` over the augmentation pairs.
  5. **Grenander** `pA_b`: the non-increasing density MLE on [0,1] from `U1_b ∪ {one pseudo-point ~ U(0,1)}`. Take the least concave majorant of the ECDF (convex-hull / PAV on the cumulative), then its left derivatives. Store it as a step function (knots, heights).
  6. `ε_b = pA_b(1⁻)` and `g0_b(u) = log pA_b(u)`. **Also store `U1_b` sorted**; the difference report's tail p-values p_b need it.
  7. **ω_b(stat):** responsibility `r(a) = 1 − ε_b / pA_b(u_b(a))` (the probability that the pair came from the "same" component, not the failure floor). Then fit `IsotonicRegression`-style PAV (implement in numpy) of r on the block's quality statistic, increasing in quality. Blocks without a quality stat get ω ≡ 1. If a block has several stats, take the product of the per-stat isotonic fits.
  8. **β_b** = `Var_R(D_b) / Var_A(D_b)` (for Stage 1).
  9. **m_eff_F:** `R_F = corrcoef(g0_b(u_b) over R, b ∈ F)`, then `m_eff = m² / R_F.sum()`.
  10. Save to `.npz` and `.json`, and load them back.
- [x] `prior.py`: `prior_scores(u: (K,B), w: (K,B), model) -> (S: (K,), e: (K,B))`, with `e_b = (m_eff_F/m_F)·w_b·g0_b(u_b)` and `S = e.sum(1)`.
- [x] `baselines.py`:
  - L0 = the mean of the per-block cosines (the current behaviour, on the 781-d blocks);
  - L1 = mean(1 − u_b);
  - L1r = Σ 1/(60 + r_b);
  - L2 = M_p(1 − u_b) for p ∈ {1, 0, −1, −4}, with p chosen by MRR on augmentation queries, not by labels;
  - L3 = prior.
- [x] Tests (`test_grenander.py`, `test_null_model.py`):
  - Grenander output is non-increasing and integrates to 1 ± 1e-9;
  - on samples from Beta(1, 5) it recovers the density (L1 error < 0.05 at n = 1e4);
  - on uniform data it gives ε ≈ 1 and g0 ≈ 0;
  - m_eff = m for independent blocks, m_eff = 1 for duplicated blocks;
  - an invalid block (w = 0) contributes exactly 0.
- **Gate 3:** the null model is fitted on the 600 using **part-disjoint 5-fold** augmentation pairs. The F0 ECDF uses all gallery random pairs, which is transductive and allowed. Plot each g0_b curve to `results/g0_curves.png`, 15 panels, with ε_b in each title.

### Phase 3b: Difference report ("which descriptors make these two parts different?")
This phase depends only on the registry and the null model, so it works before any learning and with every scorer L0–L4.

- [x] `src/blr/diff_report.py`, `compare(x_a: (781,), x_b: (781,), q_a: dict, q_b: dict, model, alpha_diff=cfg.ALPHA_DIFF) -> DiffReport`.
      The same function is used on every retrieved result (query vs result) and on any two parts directly.
- [x] **Test per block.** The null hypothesis H_b is "in block b, these two parts are the same design". The statistic is u_b.
  ```
  p_b = (1 + #{a ∈ A_b : U1_b(a) ≥ u_b}) / (|A_b| + 1)
  ```
  This is a conformal tail probability: how often a real same-design pair is at least this far apart. It is valid in finite samples, and U1 already contains the augmentation failures, so p_b is honest about noisy blocks.
- [x] **Multiple testing across the 15 blocks, weighted by reliability** (weighted Holm, Holm 1979):
  - Blocks with `w_b = 0` get the verdict `not_evaluated` (invalid for this pair) and are excluded.
  - Sort the rest by `p_b / w_b` ascending.
  - Step down: reject H_(j), giving the verdict `differs`, while `p_(j) ≤ α · w_(j) / Σ_{i≥j} w_(i)`. Stop at the first failure; that block and all later ones get the verdict `consistent`.
  - Result: family-wise error ≤ α_diff. A low-reliability block needs stronger evidence to be called different. The only number set by hand is α_diff, a stated error rate.
- [x] **What changed, per differing block (deterministic, no LLM).** For each sub-part, compute its share `D_bj/median_R(D_bj) / D_b` of the block distance, then describe it by `kind`:

  | kind | Detail line | Example output |
  |---|---|---|
  | `hist` (ordered) | mass-weighted median of A and of B in axis units, Δ; plus the bin with the largest mass loss → the bin with the largest gain | "DAD: crease angle 90° → 135° (100 % of crease length moved)" |
  | `hist2d` | largest-loss cell → largest-gain cell, as (SI, CV) labels | "CURV: convex cap (SI +1) → concave cup (SI −1)" |
  | `spectrum` | the index with the largest \|Δ\|, its label, both values | "CRS-A: dominant harmonic m = 6 → m = 2" |
  | `scalar` | both values and Δ | "D4: γ₁ 1.28 → 0.40 (fewer very flat tetrahedra)" |
  | `count` | invert log1p to integers and print both | "T1: genus 4 → 0 (through-holes closed)" |

  Each line is built from `what_it_sees` plus the numbers.
- [x] Dataclasses:
  ```python
  @dataclass(frozen=True)
  class SubpartDiff:  name: str; kind: str; share: float; a_summary: str; b_summary: str; delta: float; unit: str; text: str
  @dataclass(frozen=True)
  class BlockDiff:    block: str; plain_name: str; verdict: str   # differs | consistent | not_evaluated
                      p_value: float; p_threshold: float; evidence_nats: float; reliability: float
                      u: float; subparts: tuple[SubpartDiff, ...]; text: str
  @dataclass(frozen=True)
  class DiffReport:   differing: tuple[BlockDiff, ...]      # sorted by evidence_nats ascending (strongest "against" first)
                      consistent: tuple[BlockDiff, ...]; not_evaluated: tuple[BlockDiff, ...]
                      summary: str                          # e.g. "Differs in 2 of 14 evaluated descriptors: DAD (edge treatment), MTS (wall thickness)"
                      def to_dict(self) -> dict: ...        # JSON-safe, for Flask
  ```
- [x] Unit tests (`tests/test_diff_report.py`):
  - `compare(x, x)` gives no differing blocks;
  - the verdicts of `compare(a, b)` equal those of `compare(b, a)`;
  - synthetic: take a gallery part, replace one block with that block from a far part, and only that block becomes `differs`;
  - `w_b = 0` gives `not_evaluated`;
  - weighted Holm matches a brute-force reference on 1,000 random (p, w) sets;
  - under the null (augmentation pairs from held-out parts), the rate of ≥ 1 `differs` is ≤ α_diff + 2·SE.
- [x] CLI script `scripts/blr_compare.py --a A.stl --b B.stl --model-dir <path>`, which prints the report as a table and writes JSON.
- [ ] **Acceptance test on the built ladder pairs** (`tests/test_diff_report_ladder.py`, STL folder passed as `--test-parts-dir`, never hard-coded). Expected results from `claude/descriptor-pair-catalog.md`:

  | Pair | Must be `differs` | Must be `consistent` (`= EXACT` ties) |
  |---|---|---|
  | R01 ball vs cube | D1 | — |
  | R02 spider 90° vs 0/30/180/210° | D2 | D1 |
  | R03 pin straight vs bowed | TRI | — |
  | ~~R04 bracket RH vs LH~~ | dropped (D4 v2 is mirror-invariant; no block separates LH from RH) | every block must be `consistent` (all are mirror-invariant) |
  | R04alt strip flat vs cambered | D4 | — |
  | R05 knob plain vs grooved | SHPS | — |
  | R06 solid vs hollow cube | NCF | SHPS |
  | R07 plate vs slotted plate | LBE | — |
  | R08 sharp vs chamfered block | DAD | — |
  | R09 crown vs dish | CURV | SHPS, LBE, DAD |
  | R10 concentric vs eccentric shaft | CRS-R | SHPS, DAD, CURV |
  | R11 through vs blind holes | T1 | — |
  | R12 bolt circle R25 vs R35 | PHD | T1, DAD, CURV, SHPS |
  | R13 equal vs grouped hole pitch | CRS-A | D1, SHPS, DAD, CURV, T1, PHD, CRS-R |
  | R14 coplanar vs stepped feet | PAF | CURV, SHPS, DAD, T1, PHD |
  | R15 dome 1 mm vs 2 mm wall | MTS | — |

  Blocks with `≈ NOISE` or `≈ BOUND` badges should be `consistent` but are only **reported**, not asserted. Known disclosed shadows: D1 on R10 and R12. A failure on an `= EXACT` tie means a real bug (tessellation artefact, metric, or ω). Log it; do not loosen the test.
- **Gate 3b:** `results/ladder_diff_report.md` with one row per pair (differing blocks, p-values, detail lines). The pass count against the table above is printed. The R07, R06, R08 and R14 LBE cases require the Phase 0 remesh.

### Phase 4: Stage-1 candidates
- [x] `candidates.py`:
  - `brute(q) -> all N−1` (**drop the self-match**);
  - `faiss_union(q, K)`: one `IndexFlatL2` on `concat_b sqrt(β_b)·x_b` (blocks unit-normalised first), plus 15 per-block flat indexes each giving top-⌈K/B⌉; return the union;
  - `choose_K(aug_queries, target=0.99)`.
- [x] Report R@K for K ∈ {25, 50, 100, 200} on augmentation queries (does the original come back?) and on class relevance (LOO).
- **Gate 4:** an R@K table in `results/phase4_recall.md`. At N = 600 the engine uses brute force; the table only characterises Stage 1.

### Phase 5: Evaluation harness and ladder L0–L3 — DECISION GATE
- [x] `evaluate.py`:
  - `average_precision`, `ndcg_at_k` (graded: 2 = the original of an augmented query, 1 = same class, 0 otherwise), `precision_at_k`, `recall_at_k`, `mrr`;
  - `paired_wilcoxon(ap_a, ap_b)` with Holm correction across adjacent rungs;
  - `cluster_bootstrap_ci(per_query, class_ids, n=2000)`, which resamples classes.
- [x] `scripts/blr_eval_ladder.py` runs two protocols:
  - **Design:** a query is a fresh augmentation of part p (a seed never used in fitting, in p's held-out fold); the gallery is the 600 originals; the metrics are MRR and R@1 of p.
  - **Family (evaluation only):** the original LOO over the 600 with class relevance; the metrics are mAP, nDCG@10, P@5 and P@10 (P@10 ceiling 97.5 %). L0–L3 use no labels, so every class is unseen by construction.

  Output: `results/ladder_L0_L3.json` plus a markdown table with the CIs and Wilcoxon p-values.
- **Gate 5 (stop and report):**
  - H1: L3 > L1 on both protocols (p < 0.05 after Holm)?
  - Which blocks have ε_b > 0.2? They are unreliable and need their quality stats reviewed.
  - Latency p50/p95 per query for S2 + S3.

  If L3 ≤ L1, debug the metric, g0 and ω choices before any learning.

### Phase 6: L4, the monotone GA2M residual
- [ ] `pair_features.py`: `pair_features(q_idx or x_q, C, model) -> φ: (K, B, 5), w: (K, B)`. Guard MAD = 0: when every candidate is equidistant, set z = 0.
- [ ] Training rows, part-disjoint 5 folds. For each training part p and each augmentation a: query = T_a(p), candidates C = the gallery. Label 1 for the original p, 0 for every other part. There are **no class labels**.
      Subsample negatives to the top 100 by prior score (hard negatives) plus 20 random negatives. Apply block dropout: for each row, with probability 0.15, pick a random block and set w_b = 0 and its φ_b to NaN.
- [ ] `gam.py`: LightGBM with these parameters:
  ```python
  params = dict(objective="binary", learning_rate=0.03, max_depth=2, num_leaves=4,
                min_data_in_leaf=200, lambda_l2=1.0, bagging_fraction=0.8, bagging_freq=1,
                feature_fraction=0.8, monotone_constraints=MONO, monotone_constraints_method="advanced",
                interaction_constraints=GROUPS, verbose=-1, seed=cfg.seed)
  # feature order per block b: [u_b, ell_b, z_b, rho_b, w_b]
  # MONO: -1 for u, ell, z, rho ; 0 for w
  # GROUPS: for each b: [idx(u_b), idx(w_b)], [idx(ell_b)], [idx(z_b)], [idx(rho_b)]
  # init_score = S_prior + logit(pi1)   where pi1 = positive fraction in rows (keeps the intercept honest)
  ```
  Early stopping uses the held-out fold's logloss.
- [ ] **Compile to tables:** read `booster.dump_model()`. Every tree uses features from a single group (assert this). For each group, build the grid of all split thresholds used on its features, evaluate every tree of that group on the grid cell midpoints, and sum into a 1-D table (singletons) or a 2-D table (u, w). Predict with `searchsorted` and a lookup.
      *Test (`test_gam_compile.py`):* compiled table scores equal `booster.predict(X, raw_score=True) − init` to within 1e-9 on 10k random rows. The assertion fails if any tree crosses groups.
- [ ] λ: a grid over [0, 1] in steps of 0.1, choosing the λ that maximises held-out design MRR. Report family mAP at that λ; never select on it.
- **Gate 6:**
  - the ladder now includes L4, with the Wilcoxon comparison L4 vs L3;
  - H3 check: L4 − L3 ≤ 2 mAP points is the expected outcome, so report whether it holds;
  - latency ≤ 2 ms at K = 100 for S2 + S3 on the i9.

### Phase 7: Gate, explanations, integration
- [ ] `conformal.py`: `fit_tau(scores_true_match: (n,), alpha) -> τ`, taken as the ⌊α(n+1)⌋-th smallest; `is_no_match(S_top, τ)`. Calibrate on a split of held-out design queries.
      *Test:* on simulated exchangeable data the empirical false-reject rate is ≤ α + 2·SE.
- [ ] Measure the true-reject rate on an OOD set: the 8 TPMS parts, the R01–R15 test parts, and a leave-one-class-out gallery.
- [ ] `engine.py`: a `BLREngine` class exposing `fit(dataset_dir: Path)`, `query(mesh_path: Path, k=10) -> list[Result]` and `add(mesh_path: Path)`.
  - `Result` fields: `filename, S, rank, evidence: dict[block, nats], top_for: list, top_against: list, no_match: bool, differences: DiffReport` (from Phase 3b, computed for the top-k results only).
  - Add `compare(path_a: Path, path_b: Path) -> DiffReport` for any two parts.
  - `add` does no refit. It updates σ_b for the new part and its reverse 7-NN, and appends to a null reservoir. A per-block KS test at δ = 0.01 triggers a closed-form refit of `null_model` only.
- [ ] Wire it into `retrieval_engine.py` / `main.search_query` behind a `engine="blr"` switch, keeping the classical engine as the default until Gate 7. The Flask response gets `evidence` and `differences` (`DiffReport.to_dict()`) fields. UI suggestion: a "Why different?" expander per result listing the differing blocks' text lines. It is optional and happens later.
- **Gate 7:** `test_engine_smoke.py` runs fit → query → add → query on 30 parts in under 60 s. The evidence sums to S within 1e-9. Every result carries a DiffReport, and the report for a part compared with itself has zero differing blocks.

### Phase 8 (later): Family mode and scale
- [ ] Add MCB-A: deduplicate against the 600 (GrabCAD overlap), then build 4 class-disjoint folds of 17 classes.
- [ ] Family mode: grade-1 positives are same class **on training classes only**. Select λ on held-out classes, and report the seen/unseen gap Δ.
- [ ] Switch Stage 1 to HNSW for N > 1e5.
- [ ] Ablations A1–A11 from the spec.

### Every session ends with
- [ ] README.md: What This Does, Setup, Usage, File Map, Domain Assumptions, and a Changelog row.
- [ ] Every item on the aarsh-coding-style pre-delivery checklist.
- [ ] Ticked boxes in `DOCS/BLR_PLAN.md`.

---

## 5. Pitfalls to check (Claude Code: re-read before each phase)

1. **Direction:** every D_b is a distance, so monotone constraints are **−1** on u, ℓ, z, ρ because the label is relevance. A sign flip passes the tests silently and inverts the ranking; add a test that a strictly closer candidate never scores lower.
2. **Self-match:** remove q from its own candidate list in every LOO evaluation and in the training rows.
3. **Leakage:**
   - an augmentation of a held-out part must never appear in the fit of g0, ω or the GAM;
   - F0 (random pairs) may use the whole gallery;
   - never choose p (L2), λ or the tree count on class metrics.
4. **ECDF ties** (T1 counts are discrete): use mid-ranks, or the Grenander fit sees fake mass at 0.
5. **Grenander at u → 0:** the density can spike. Cap the first step at the empirical mass / (first knot width) using the pseudo-point, and never return +inf.
6. **Zero variance:** T1 boundary loops is always 0 on watertight parts, so do not z-score it; it is a quality stat. Guard MAD = 0 and a zero histogram sum.
7. **init_score must be in log-odds:** `S_prior + logit(pi1)`. Without the logit, the model spends its first trees learning the intercept, which breaks the nats interpretation.
8. **Interaction constraints:** confirm they hold after training (the compile assertion). A tree mixing blocks breaks the exact per-block evidence.
9. **Mirror:** all 15 blocks are mirror-invariant (D4 v2). The augmenter still does not reflect.
10. **Coarse STL:** without remeshing, LBE, DAD and curvature read facet artefacts; ω should go low on these parts. Check this on R06, R07, R08 and R14.
11. **Difference ≠ negative evidence.** Call a block `differs` from its p-value p_b with weighted Holm, **not** from `e_b < 0`. Evidence mixes in the learned residual and the family down-weighting. A p-value tests one thing: "too far apart for the same design".
12. **Self-compare sanity:** `compare(x, x)` must return zero differing blocks, and `compare(a, b)` must equal `compare(b, a)` in verdicts. The detail text swaps the A/B direction; the verdicts must not change.
13. **Determinism:** thread `rng: np.random.Generator` everywhere. Never use the global `np.random`.

---

## 6. Constants (`BLRConfig` defaults)

```python
N_NULL_PAIRS: int = 100_000      # [-]    random gallery pairs for F0_b
N_AUG_DRAWS: int = 4             # [-]    augmentations per part
K_NN_SCALE: int = 7              # [-]    neighbour index for local scale σ_b
N_FOLDS: int = 5                 # [-]    part-disjoint folds
N_BRUTE_MAX: int = 5_000         # [-]    re-rank all candidates at or below this gallery size
RECALL_TARGET: float = 0.99      # [-]    R@K target used by choose_K
REMESH_EDGE_FRAC: float = 0.02   # [-]    precondition edge length / bounding-box diagonal D∅
AUG_EDGE_RANGE = (0.5, 2.0)      # [-]    remesh edge multiplier
AUG_NOISE_MAX: float = 0.001     # [-]    vertex noise σ / D∅ (was 0.01; that destroyed the normal-based blocks, 2026-10-01)
AUG_DECIM_RANGE = (0.25, 1.0)    # [-]    kept face fraction
AUG_ANISO_RANGE = (0.95, 1.05)   # [-]    per-axis scale
BLOCK_DROPOUT: float = 0.15      # [-]    per-row probability
N_HARD_NEG: int = 100            # [-]    hard negatives per training query
N_RAND_NEG: int = 20             # [-]    random negatives per training query
ALPHA: float = 0.05              # [-]    target false NO_MATCH rate
ALPHA_DIFF: float = 0.05         # [-]    family-wise error rate for calling a block "differs" (weighted Holm)
DRIFT_DELTA: float = 0.01        # [-]    KS test level for null refit
RRF_K: int = 60                  # [-]    reciprocal rank fusion constant (Cormack 2009)
SEED: int = 20260930             # [-]    master seed
```

---

## 7. Acceptance summary

| Phase | Done when |
|---|---|
| 0 | 781-d extraction is deterministic; registry tests pass |
| 1 | Metric tests pass; ≤ 3 ms for K = 599 |
| 2 | Augmentation sanity table written |
| 3 | g0 curves plotted; null model round-trips |
| 3b | `compare(a, b)` works; the ladder-pair test passes on every target block and every `= EXACT` tie |
| 4 | R@K table written |
| 5 | **Ladder L0–L3 reviewed by Aarsh (go/no-go)** |
| 6 | L4 in the ladder; compiled tables match LightGBM; ≤ 2 ms at K = 100 |
| 7 | Engine smoke test passes; the conformal rate holds; the `engine="blr"` switch works |
| 8 | MCB-A class-disjoint results; family mode; seen/unseen gap |
