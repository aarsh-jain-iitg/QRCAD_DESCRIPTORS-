"""
================================================================================
FILE: null_model.py
================================================================================
PURPOSE:
    Label-free null model of the 15 block distances. From random gallery pairs
    (different designs) and augmentation pairs (same design) it fits, per block:
    the ECDF F0_b that turns a distance into a percentile u_b, the local scale
    sigma_b, the Grenander density pA_b of same-design percentiles (with its
    failure floor eps_b and log-density g0_b), the sorted same-design percentiles
    U1_b (for conformal p-values), the reliability curves omega_b(quality), the
    Stage-1 weight beta_b and the family correlation correction m_eff_F.

INPUTS:
    - X (N, 781) gallery, Q (N, 6) quality, X_aug (M, 781) augmented rows, Q_aug (M, 6),
      pairs_aug (M,) index of the source part of every augmented row, BLRConfig.

OUTPUTS:
    - Space (parts that depend on random pairs only) and NullModel (adds the
      augmentation-pair parts); save() / load() as .npz + .json (no pickle).

DEPENDENCIES:
    stdlib      : dataclasses, json, logging, pathlib
    third-party : numpy
    internal    : blr.blocks, blr.config, blr.metrics, blr.quality

PHYSICS / DOMAIN CONTEXT:
    - u_b = F0_b(D_b) with mid-rank ties: small u = unusually close.
    - Grenander: non-increasing density MLE on [0,1] = left derivative of the least
      concave majorant of the ECDF (upper convex hull), fitted on U1_b plus one pseudo-point
      ~ U(0,1); then mixed with a uniform of weight 1/(n+1) so the density is never 0
      (log stays finite) and still integrates to 1. First knot is clipped to >= 0.5/n so
      the first step cannot spike to infinity.
    - eps_b = height of the flat tail on [0.9, 1] (the density there is replaced by its mean, which keeps
      the mass and the monotonicity; the raw MLE always ends in a zero-density sliver): the share of
      same-design pairs that a block fails to keep close, i.e. its failure rate.
    - omega_b: responsibility r = 1 - eps_b / pA_b(u) of a same-design pair, regressed
      isotonically (increasing) on the quality-oriented statistic (worse of the pair);
      the pair weight at query time is omega_b(q) * omega_b(c).
    - m_eff_F = m_F^2 / (1' R_F 1), R_F the correlation of g0_b(u_b) over random pairs.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.blr.blocks import BLOCKS, N_DIM
from src.blr.config import BLRConfig
from src.blr.metrics import N_SUB, all_block_distances, block_distances, subpart_distances
from src.blr.quality import QUALITY_KEYS

logger = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
N_BLOCKS = len(BLOCKS)
CHUNK = 5_000                     # [-] rows per vectorised distance call
MIN_ISO = 30                      # [-] fewer pairs than this: omega = 1
TAIL_START = 0.9                  # [-] percentile where the flat tail (failure floor eps_b) starts
QUALITY_SIGN = {"n_vertices": +1.0, "eigen_gap": +1.0, "boundary_edge_frac": -1.0,
                "facet_angle": -1.0, "f_miss": -1.0, "boundary_loops": -1.0}   # + = higher is better
FAMILIES = tuple(dict.fromkeys(b.family for b in BLOCKS))


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def assign_folds(n: int, n_folds: int, seed: int) -> np.ndarray:
    """Part-disjoint fold id (n,) in [0, n_folds), random but balanced."""
    return np.random.default_rng(seed).permutation(n) % n_folds


def oriented(Q: np.ndarray, key: str) -> np.ndarray:
    """Quality column `key` of Q (..., 6) oriented so that larger = better (n_vertices logged)."""
    v = Q[..., QUALITY_KEYS.index(key)]
    return QUALITY_SIGN[key] * (np.log1p(v) if key == "n_vertices" else v)


def robust_scale(X: np.ndarray) -> np.ndarray:
    """(781,) 1.4826 * MAD of every scalar-kind dimension over the gallery, else 1."""
    s = np.ones(N_DIM)
    for b in BLOCKS:
        for sp in b.subparts:
            if sp.kind == "scalar":
                v = X[:, sp.sl].astype(np.float64)
                mad = 1.4826 * np.median(np.abs(v - np.median(v, axis=0)), axis=0)
                s[sp.sl] = np.where(mad > 1e-6, mad, np.maximum(v.std(axis=0), 1e-6))
    return s


def pav_increasing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Isotonic (non-decreasing) regression of y on x by pool-adjacent-violators.

    Returns
    -------
    (xs, ys) : unique sorted x values and the fitted non-decreasing values.
    """
    order = np.argsort(x, kind="stable")
    xs_all, ys_all = np.asarray(x, np.float64)[order], np.asarray(y, np.float64)[order]
    ux, inv = np.unique(xs_all, return_inverse=True)
    w = np.bincount(inv).astype(np.float64)
    m = np.bincount(inv, weights=ys_all) / w
    val, wt, cnt = [], [], []                       # blocks of pooled values
    for mi, wi in zip(m, w):
        val.append(mi), wt.append(wi), cnt.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            v2, w2, c2 = val.pop(), wt.pop(), cnt.pop()
            val[-1] = (val[-1] * wt[-1] + v2 * w2) / (wt[-1] + w2)
            wt[-1] += w2
            cnt[-1] += c2
    return ux, np.repeat(val, cnt)


def grenander(u: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Non-increasing density MLE on [0, 1] from percentiles u plus one pseudo-point.

    Parameters
    ----------
    u : np.ndarray, shape (n,)
        Values in [0, 1].
    rng : np.random.Generator
        Draws the U(0, 1) pseudo-point.

    Returns
    -------
    (knots, heights) : knots (m+1,) from 0 to 1, heights (m,) non-increasing, integral 1.
    """
    pts = np.append(np.clip(np.asarray(u, np.float64), 0.0, 1.0), rng.random())
    n = len(pts)
    ut, cnt = np.unique(np.maximum(pts, 0.5 / n), return_counts=True)
    xs = np.concatenate([[0.0], ut])
    ys = np.concatenate([[0.0], np.cumsum(cnt) / n])
    if xs[-1] < 1.0:
        xs, ys = np.append(xs, 1.0), np.append(ys, 1.0)
    hull = [0]
    for k in range(1, len(xs)):                     # upper concave hull (monotone chain)
        while len(hull) >= 2:
            i, j = hull[-2], hull[-1]
            if (xs[j] - xs[i]) * (ys[k] - ys[i]) - (ys[j] - ys[i]) * (xs[k] - xs[i]) >= 0.0:
                hull.pop()
            else:
                break
        hull.append(k)
    kx, ky = xs[hull], ys[hull]
    h = np.diff(ky) / np.diff(kx)
    lam = 1.0 / (n + 1.0)
    h = (1.0 - lam) * h + lam
    m = int(np.sum(kx < TAIL_START))                # segments starting below the tail (kx[0] = 0)
    cum = np.concatenate([[0.0], np.cumsum(h * np.diff(kx))])
    eps = (1.0 - float(np.interp(TAIL_START, kx, cum))) / (1.0 - TAIL_START)
    return np.concatenate([kx[:m], [TAIL_START, 1.0]]), np.concatenate([h[:m], [eps]])


def _pair_distances(XA: np.ndarray, XB: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(K, N_SUB) sub-part distances of row-aligned pairs, chunked."""
    return np.concatenate([subpart_distances(XA[i:i + CHUNK], XB[i:i + CHUNK], scale)
                           for i in range(0, len(XA), CHUNK)])


def gallery_distances(X: np.ndarray, medians: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(N, N, 15) float32 block distances between all gallery parts (diagonal = 0)."""
    D = np.empty((len(X), len(X), N_BLOCKS), dtype=np.float32)
    for i in range(len(X)):
        D[i] = all_block_distances(X[i], X, medians, scale)
    return D


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
@dataclass
class Space:
    """Parts of the null model that depend on random gallery pairs only."""

    scale: np.ndarray                 # (781,) robust scalar scales
    medians: np.ndarray               # (N_SUB,) null medians of sub-part distances
    F0: np.ndarray                    # (15, n_null) sorted random-pair block distances
    sigma: np.ndarray                 # (N, 15) 7th-NN block distance of every gallery part
    D_null: np.ndarray                # (n_null, 15) random-pair block distances (unsorted)
    D_gal: np.ndarray                 # (N, N, 15) float32 gallery block distances

    def u(self, D: np.ndarray) -> np.ndarray:
        """Mid-rank ECDF percentile of block distances D (..., 15) -> (..., 15) in (0, 1)."""
        out = np.empty(D.shape)
        for b in range(N_BLOCKS):
            lo = np.searchsorted(self.F0[b], D[..., b], side="left")
            hi = np.searchsorted(self.F0[b], D[..., b], side="right")
            out[..., b] = (lo + hi) / (2.0 * self.F0.shape[1])
        return out


@dataclass
class NullModel:
    """Fitted null model (see module header)."""

    space: Space
    U1: np.ndarray                    # (15, M) sorted same-design percentiles
    knots: list[np.ndarray]           # per block, Grenander knots
    heights: list[np.ndarray]         # per block, Grenander heights
    eps: np.ndarray                   # (15,)
    beta: np.ndarray                  # (15,)
    omega: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]   # (block, key) -> isotonic curve
    m_eff: dict[str, float]           # per family
    m_fam: dict[str, int]
    config_hash: str = ""
    extra: dict = field(default_factory=dict)

    # -- lookups -------------------------------------------------------------
    def g0(self, U: np.ndarray) -> np.ndarray:
        """log pA_b(u_b), U (..., 15) -> (..., 15) [nats]."""
        out = np.empty(U.shape)
        for b in range(N_BLOCKS):
            idx = np.clip(np.searchsorted(self.knots[b], U[..., b], side="right") - 1,
                          0, len(self.heights[b]) - 1)
            out[..., b] = np.log(self.heights[b][idx])
        return out

    def pvalue(self, U: np.ndarray) -> np.ndarray:
        """Conformal tail p_b = (1 + #{a: U1_b(a) >= u_b}) / (M + 1); U (..., 15)."""
        M = self.U1.shape[1]
        out = np.empty(U.shape)
        for b in range(N_BLOCKS):
            out[..., b] = (1 + M - np.searchsorted(self.U1[b], U[..., b], side="left")) / (M + 1.0)
        return out

    def omega_of(self, Q: np.ndarray) -> np.ndarray:
        """Per-part block reliability, Q (K, 6) -> (K, 15) in [0, 1] (product over quality keys)."""
        Q = np.atleast_2d(Q)
        out = np.ones((len(Q), N_BLOCKS))
        for (b, key), (xs, ys) in self.omega.items():
            out[:, b] *= np.interp(oriented(Q, key), xs, ys)
        return out

    def family_factor(self) -> np.ndarray:
        """(15,) m_eff_F / m_F of each block's family."""
        return np.array([self.m_eff[b.family] / self.m_fam[b.family] for b in BLOCKS])

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path) -> None:
        """Write path.npz (arrays) and path.json (scalars); no pickled objects."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = {"scale": self.space.scale, "medians": self.space.medians, "F0": self.space.F0,
               "sigma": self.space.sigma, "U1": self.U1, "eps": self.eps, "beta": self.beta,
               "D_null": self.space.D_null, "D_gal": self.space.D_gal,
               "knot_len": np.array([len(k) for k in self.knots]),
               "knots": np.concatenate(self.knots), "heights": np.concatenate(self.heights)}
        keys = []
        for i, ((b, key), (xs, ys)) in enumerate(self.omega.items()):
            arr[f"om_x{i}"], arr[f"om_y{i}"] = xs, ys
            keys.append([b, key])
        np.savez(path.with_suffix(".npz"), **arr)
        path.with_suffix(".json").write_text(json.dumps(
            {"omega_keys": keys, "m_eff": self.m_eff, "m_fam": self.m_fam,
             "config_hash": self.config_hash, "extra": self.extra}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> NullModel:
        path = Path(path)
        z = np.load(path.with_suffix(".npz"), allow_pickle=False)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        kl = z["knot_len"]
        cut = np.cumsum(kl)[:-1]
        cut_h = np.cumsum(kl - 1)[:-1]
        omega = {(int(b), key): (z[f"om_x{i}"], z[f"om_y{i}"])
                 for i, (b, key) in enumerate(meta["omega_keys"])}
        sp = Space(z["scale"], z["medians"], z["F0"], z["sigma"], z["D_null"], z["D_gal"])
        return cls(sp, z["U1"], np.split(z["knots"], cut), np.split(z["heights"], cut_h),
                   z["eps"], z["beta"], omega, meta["m_eff"], {k: int(v) for k, v in meta["m_fam"].items()},
                   meta["config_hash"], meta["extra"])


# ── FITTING ───────────────────────────────────────────────────────────────────
def fit_space(X: np.ndarray, cfg: BLRConfig, rng: np.random.Generator) -> Space:
    """
    Random-pair part of the null model: scales, sub-part medians, F0_b, sigma_b.

    Parameters
    ----------
    X : np.ndarray, shape (N, 781)
    cfg : BLRConfig
    rng : np.random.Generator

    Returns
    -------
    Space
    """
    N = len(X)
    scale = robust_scale(X)
    i = rng.integers(0, N, cfg.n_null_pairs)
    j = (i + rng.integers(1, N, cfg.n_null_pairs)) % N                 # j != i
    sub = _pair_distances(X[i], X[j], scale)
    medians = np.median(sub, axis=0)
    medians = np.where(medians > 1e-12, medians, np.maximum(sub.mean(axis=0), 1e-12))
    D_null = block_distances(sub, medians)
    F0 = np.sort(D_null, axis=0).T.copy()
    D_gal = gallery_distances(X, medians, scale)
    Dm = D_gal.astype(np.float64).copy()
    Dm[np.arange(N), np.arange(N)] = np.inf
    k = min(cfg.k_nn_scale, N - 1) - 1
    sigma = np.partition(Dm, k, axis=1)[:, k, :]
    logger.info("space: %d random pairs, %d parts", cfg.n_null_pairs, N)
    return Space(scale, medians, F0, sigma, D_null, D_gal)


def fit_null_model(X: np.ndarray, Q: np.ndarray, X_aug: np.ndarray, Q_aug: np.ndarray,
                   pairs_aug: np.ndarray, cfg: BLRConfig, rng: np.random.Generator | None = None,
                   space: Space | None = None, train_parts: np.ndarray | None = None) -> NullModel:
    """
    Fit the full null model.

    Parameters
    ----------
    X, Q : np.ndarray, shapes (N, 781), (N, 6)
        Gallery and its quality statistics.
    X_aug, Q_aug : np.ndarray, shapes (M, 781), (M, 6)
        Augmented rows and their quality.
    pairs_aug : np.ndarray, shape (M,)
        Source part index (into X) of every augmented row.
    cfg : BLRConfig
    rng : np.random.Generator, optional
        Default: default_rng(cfg.seed).
    space : Space, optional
        Reuse a fitted random-pair part (fold models share it).
    train_parts : np.ndarray of int, optional
        Part indices whose augmentations may be used (part-disjoint folds); default all.

    Returns
    -------
    NullModel
    """
    rng = rng or np.random.default_rng(cfg.seed)
    space = space or fit_space(X, cfg, rng)
    use = np.ones(len(pairs_aug), bool) if train_parts is None else np.isin(pairs_aug, train_parts)
    src, Xa, Qa = pairs_aug[use], X_aug[use], Q_aug[use]
    DA = block_distances(_pair_distances(X[src], Xa, space.scale), space.medians)     # (M', 15)
    UA = space.u(DA)
    knots, heights, eps = [], [], np.empty(N_BLOCKS)
    for b in range(N_BLOCKS):
        kx, h = grenander(UA[:, b], rng)
        knots.append(kx), heights.append(h)
        eps[b] = h[-1]
    model = NullModel(space, np.sort(UA, axis=0).T.copy(), knots, heights, eps,
                      _beta(space.D_null, DA), {}, {}, {}, cfg.config_hash())
    model.omega = _fit_omega(model, UA, Q[src], Qa)
    model.m_fam, model.m_eff = _family_counts(model, space)
    logger.info("null model: %d augmentation pairs, eps=%s", len(src), np.round(eps, 3))
    return model


def _beta(D_null: np.ndarray, DA: np.ndarray) -> np.ndarray:
    """Var over random pairs / Var over augmentation pairs, per block (Stage-1 weight)."""
    return D_null.var(axis=0) / np.maximum(DA.var(axis=0), 1e-12)


def _fit_omega(model: NullModel, UA: np.ndarray, Q_orig: np.ndarray, Q_aug: np.ndarray) -> dict:
    """Isotonic omega_b(quality) curves from the responsibility of same-design pairs."""
    out = {}
    g = model.g0(UA)
    for b, spec in enumerate(BLOCKS):
        r = np.clip(1.0 - model.eps[b] / np.exp(g[:, b]), 0.0, 1.0)
        for key in spec.quality_keys:
            s = np.minimum(oriented(Q_orig, key), oriented(Q_aug, key))
            if len(s) >= MIN_ISO and np.ptp(s) > 0:
                out[(b, key)] = pav_increasing(s, r)
    return out


def _family_counts(model: NullModel, space: Space) -> tuple[dict, dict]:
    """m_F and m_eff_F = m^2 / sum(R_F) with R_F the correlation of g0(u) over random pairs."""
    G = model.g0(space.u(space.D_null))
    m_fam, m_eff = {}, {}
    for fam in FAMILIES:
        idx = [i for i, b in enumerate(BLOCKS) if b.family == fam]
        m = len(idx)
        R = np.ones((1, 1)) if m == 1 else np.nan_to_num(np.corrcoef(G[:, idx].T), nan=0.0)
        m_fam[fam], m_eff[fam] = m, float(m * m / max(R.sum(), 1e-9))
    return m_fam, m_eff


# ==============================================================================
# END OF FILE: null_model.py
# ==============================================================================
# SUMMARY:
#   Random-pair space (F0, sigma, medians) + augmentation-pair fit (Grenander g0, eps,
#   omega, beta, m_eff) with .npz/.json round trip.
#
# USAGE EXAMPLE:
#   sp = fit_space(X, cfg, rng); nm = fit_null_model(X, Q, Xa, Qa, src, cfg, rng, sp, train_parts)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] omega is fitted on the worse of (original, augmented) quality but applied as
#         omega(q) * omega(c) per part; this over-penalises pairs of two mediocre parts slightly.
#   - [ ] D_gal (N, N, 15) is stored in the npz: 30 MB at N = 713, quadratic in N.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
