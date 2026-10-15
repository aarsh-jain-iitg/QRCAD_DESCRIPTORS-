"""
================================================================================
FILE: evaluate.py
================================================================================
PURPOSE:
    Retrieval metrics and statistics for the BLR ladder: average precision, graded
    nDCG, P@k, R@k, MRR, paired Wilcoxon with Holm correction, and a cluster
    bootstrap confidence interval.

INPUTS:
    - Score vectors (N,) (higher = more similar) and relevance labels.

OUTPUTS:
    - ranking(), average_precision(), ndcg_at_k(), precision_at_k(), recall_at_k(),
      reciprocal_rank(), paired_wilcoxon(), holm(), cluster_bootstrap_ci().

DEPENDENCIES:
    stdlib      : math
    third-party : numpy, scipy
    internal    : none

PHYSICS / DOMAIN CONTEXT:
    - Ranking is by descending score with a stable tie order (lowest index first).
    - Graded relevance for nDCG: 2 = the original of an augmented query, 1 = same class,
      0 otherwise; gain (2^g - 1) / log2(rank + 1).
    - Cluster bootstrap resamples clusters (classes, or parts when there are no classes) with
      replacement, so several queries of one cluster are not treated as independent.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import wilcoxon


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def ranking(scores: np.ndarray) -> np.ndarray:
    """Indices sorted by descending score, ties broken by lower index."""
    return np.argsort(-np.asarray(scores, np.float64), kind="stable")


def average_precision(rel_ranked: np.ndarray) -> float:
    """AP of a ranked binary relevance vector (0 if nothing is relevant)."""
    r = np.asarray(rel_ranked, bool)
    if not r.any():
        return 0.0
    hits = np.cumsum(r)
    return float(np.sum(hits[r] / (np.nonzero(r)[0] + 1)) / r.sum())


def ndcg_at_k(gain_ranked: np.ndarray, k: int, ideal: np.ndarray | None = None) -> float:
    """Graded nDCG@k of a ranked gain vector; `ideal` = all gains of the list (default: the vector)."""
    def dcg(g: np.ndarray) -> float:
        g = np.asarray(g, np.float64)[:k]
        return float(np.sum((2.0 ** g - 1.0) / np.log2(np.arange(2, len(g) + 2))))
    best = dcg(np.sort(np.asarray(gain_ranked if ideal is None else ideal, np.float64))[::-1])
    return dcg(gain_ranked) / best if best > 0 else 0.0


def precision_at_k(rel_ranked: np.ndarray, k: int) -> float:
    return float(np.mean(np.asarray(rel_ranked, bool)[:k]))


def recall_at_k(rel_ranked: np.ndarray, k: int) -> float:
    r = np.asarray(rel_ranked, bool)
    return float(r[:k].sum() / r.sum()) if r.any() else 0.0


def reciprocal_rank(rel_ranked: np.ndarray) -> float:
    """1 / rank of the first relevant item (0 if none)."""
    nz = np.nonzero(np.asarray(rel_ranked, bool))[0]
    return float(1.0 / (nz[0] + 1)) if len(nz) else 0.0


def holm(p: list[float]) -> list[float]:
    """Holm-adjusted p-values (step-down, monotone, capped at 1)."""
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value of per-query metrics a vs b (1.0 if identical)."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return 1.0 if np.all(d == 0) else float(wilcoxon(a, b, zero_method="wilcox").pvalue)


def cluster_bootstrap_ci(per_query: np.ndarray, cluster: np.ndarray, n: int = 2000, alpha: float = 0.05,
                         rng: np.random.Generator | None = None) -> tuple[float, float]:
    """
    Percentile CI of the mean of a per-query metric, resampling clusters.

    Parameters
    ----------
    per_query : np.ndarray, shape (Q,)
    cluster : np.ndarray, shape (Q,)
        Cluster id (class, or part index) of every query.
    n : int
        Bootstrap replicates.
    alpha : float
        Two-sided level.
    rng : np.random.Generator, optional

    Returns
    -------
    (lo, hi)
    """
    rng = rng or np.random.default_rng(0)
    ids = np.unique(cluster)
    sums = np.array([per_query[cluster == c].sum() for c in ids])
    cnt = np.array([(cluster == c).sum() for c in ids])
    pick = rng.integers(0, len(ids), (n, len(ids)))
    means = sums[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ==============================================================================
# END OF FILE: evaluate.py
# ==============================================================================
# SUMMARY:
#   AP / nDCG / P@k / R@k / MRR, paired Wilcoxon + Holm, cluster bootstrap CI.
#
# USAGE EXAMPLE:
#   rel = truth[ranking(scores)]; rr = reciprocal_rank(rel)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Wilcoxon uses scipy's default (exact for small n, normal approximation otherwise).
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
