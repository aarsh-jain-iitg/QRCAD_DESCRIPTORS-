"""
================================================================================
FILE: blr_eval_ladder.py
================================================================================
PURPOSE:
    Gate script for BLR Phases 2-5 (label-free): from the cached gallery it fits the
    null model (part-disjoint folds), writes the augmentation sanity table, the g0
    curves, the Stage-1 recall table and evaluates the ladder L0-L3 with the DESIGN
    protocol (a fresh augmentation of part p is the query, the gallery holds the
    originals, the target is p itself).

INPUTS:
    --dataset DIR, --max-parts N, --aug-draws A (must match blr_build_null.py), --out DIR.

OUTPUTS:
    - results/phase2_aug_sanity.md, results/g0_curves.png, results/phase4_recall.md,
      results/ladder_L0_L3.json, results/ladder_L0_L3.md
    - <dataset>/.qrcad_cache/blr/<hash>/null_model.{npz,json} (model fitted on all parts)

DEPENDENCIES:
    stdlib      : argparse, dataclasses, json, logging, pathlib, sys, time
    third-party : numpy, matplotlib
    internal    : src.blr.*

PHYSICS / DOMAIN CONTEXT:
    - Every query is scored by the model of ITS fold, fitted only on augmentations of parts in the
      other folds (part-disjoint); F0 uses all gallery pairs (transductive, allowed).
    - L2's exponent p is chosen by MRR on these same augmentation queries (label-free but not
      held out: mild optimism for L2 among four candidate values); no class labels exist.
    - Latency is S2 + S3 (block distances, percentiles, weights, prior) per query.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.blr.baselines import GEN_MEAN_P, BlockCosine, gen_mean_scores, l1_scores, l1r_scores  # noqa: E402
from src.blr.blocks import BLOCKS  # noqa: E402
from src.blr.candidates import K_GRID, Stage1  # noqa: E402
from src.blr.config import BLRConfig, cache_dir  # noqa: E402
from src.blr.evaluate import cluster_bootstrap_ci, holm, paired_wilcoxon  # noqa: E402
from src.blr.gallery import build_gallery  # noqa: E402
from src.blr.metrics import all_block_distances  # noqa: E402
from src.blr.null_model import assign_folds, fit_null_model, fit_space  # noqa: E402
from src.blr.prior import pair_stats, prior_scores  # noqa: E402

logger = logging.getLogger("blr.eval")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
EPS_UNRELIABLE = 0.2              # [-] eps_b above this = block fails often (Gate 5)
N_LATENCY = 50                    # [-] queries timed
K_LAT = 100                       # [-] candidate-list size for the second latency figure
LEVELS = ("L0", "L1", "L1r", "L2", "L3")


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def target_rank(scores: np.ndarray, t: int) -> int:
    """1-based rank of index t under descending scores, ties broken by lower index."""
    return int((scores > scores[t]).sum() + (scores[:t] == scores[t]).sum() + 1)


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_aug_sanity(model, out: Path) -> list[str]:
    """Phase 2 gate table: how close augmentations sit relative to random pairs, per block."""
    rows, weak = [], []
    for b, spec in enumerate(BLOCKS):
        u = model.U1[b]
        frac = float(np.mean(u < 0.5))
        if frac < 0.9:
            weak.append(spec.name)
        rows.append([spec.name, f"{np.median(u):.4f}", f"{np.percentile(u, 90):.3f}", f"{100 * frac:.1f}%",
                     f"{model.eps[b]:.3f}"])
    text = ("# Phase 2 augmentation sanity\n\nPercentile u of (part, augmentation) distances within the random-pair "
            "distribution of each block (0 = closest possible). Target: at least 90 % of pairs below the "
            "median random pair (u < 0.5). Blocks below that are eps_b candidates.\n\n"
            + md_table(["block", "median u", "p90 u", "pairs with u<0.5", "eps_b"], rows)
            + f"\n\nBlocks under 90 %: {', '.join(weak) if weak else 'none'}\n")
    out.write_text(text, encoding="utf-8")
    return weak


def plot_g0(model, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = (np.arange(500) + 0.5) / 500
    fig, axes = plt.subplots(3, 5, figsize=(16, 8), sharex=True)
    for b, ax in enumerate(axes.ravel()):
        ax.plot(u, model.g0(np.tile(u[:, None], (1, len(BLOCKS))))[:, b])
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{BLOCKS[b].name}  eps={model.eps[b]:.2f}", fontsize=9)
    fig.supxlabel("percentile u of the block distance (0 = closest)")
    fig.supylabel("g0 = log pA(u)  [nats]")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BLR gate script (Phases 2-5, design protocol)")
    ap.add_argument("--dataset", type=Path, default=ROOT_DIR / "dataset" / "small_set")
    ap.add_argument("--max-parts", type=int, default=None)
    ap.add_argument("--aug-draws", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT_DIR / "results")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = BLRConfig() if args.aug_draws is None else dataclasses.replace(BLRConfig(), n_aug_draws=args.aug_draws)
    g = build_gallery(args.dataset, cfg, max_parts=args.max_parts)
    N, A = len(g.paths), g.X_aug.shape[1]
    X, Q = g.X.astype(np.float64), g.Q
    Xa, Qa = g.X_aug.reshape(N * A, -1).astype(np.float64), g.Q_aug.reshape(N * A, -1)
    src = np.repeat(np.arange(N), A)
    logger.info("gallery: %d parts x %d augmentations", N, A)

    rng = np.random.default_rng(cfg.seed)
    space = fit_space(X, cfg, rng)
    folds = assign_folds(N, cfg.n_folds, cfg.seed)
    models = {f: fit_null_model(X, Q, Xa, Qa, src, cfg, rng, space, np.nonzero(folds != f)[0])
              for f in range(cfg.n_folds)}
    full = fit_null_model(X, Q, Xa, Qa, src, cfg, rng, space)
    full.save(cache_dir(args.dataset, cfg) / "null_model")

    weak = write_aug_sanity(full, args.out / "phase2_aug_sanity.md")
    plot_g0(full, args.out / "g0_curves.png")

    # ---- design protocol: query = augmentation of p, target = p ------------------------------------
    cos = BlockCosine(X)
    Uq, Dq, Wq, S3, S0, S1, S1r, S2 = ([] for _ in range(8))
    per = {p: [] for p in GEN_MEAN_P}
    ranks = {lv: np.zeros(N * A, int) for lv in LEVELS}
    t_lat, t_lat100 = [], []
    for q in range(N * A):
        p, xq = int(src[q]), Xa[q]
        model = models[int(folds[p])]
        t0 = time.perf_counter()
        D, U, W = pair_stats(model, xq, Qa[q], X, Q)
        s3, _ = prior_scores(U, W, model)
        if q < N_LATENCY:
            t_lat.append((time.perf_counter() - t0) * 1e3)
            sel = rng.choice(N, min(K_LAT, N), replace=False)
            t1 = time.perf_counter()
            D2, U2, W2 = pair_stats(model, xq, Qa[q], X[sel], Q[sel])
            prior_scores(U2, W2, model)
            t_lat100.append((time.perf_counter() - t1) * 1e3)
        ranks["L0"][q] = target_rank(cos.scores(xq), p)
        ranks["L1"][q] = target_rank(l1_scores(U), p)
        ranks["L1r"][q] = target_rank(l1r_scores(D), p)
        ranks["L3"][q] = target_rank(s3, p)
        for pw in GEN_MEAN_P:
            per[pw].append(target_rank(gen_mean_scores(U, pw), p))
    rr_p = {pw: np.mean(1.0 / np.array(r)) for pw, r in per.items()}
    p_star = max(rr_p, key=rr_p.get)
    ranks["L2"] = np.array(per[p_star])

    rr = {lv: 1.0 / ranks[lv] for lv in LEVELS}
    cluster = src
    stats = {}
    for lv in LEVELS:
        lo, hi = cluster_bootstrap_ci(rr[lv], cluster, rng=rng)
        stats[lv] = {"MRR": float(rr[lv].mean()), "MRR_CI": [lo, hi], "R@1": float((ranks[lv] == 1).mean()),
                     "R@5": float((ranks[lv] <= 5).mean()), "R@10": float((ranks[lv] <= 10).mean()),
                     "median_rank": float(np.median(ranks[lv]))}
    pairs = [("L0", "L1"), ("L1", "L1r"), ("L1r", "L2"), ("L2", "L3"), ("L1", "L3")]
    raw = [paired_wilcoxon(rr[a], rr[b]) for a, b in pairs]
    adj = holm(raw)
    wil = [{"a": a, "b": b, "p": r, "p_holm": h} for (a, b), r, h in zip(pairs, raw, adj)]

    unreliable = [BLOCKS[b].name for b in range(len(BLOCKS)) if full.eps[b] > EPS_UNRELIABLE]
    lat = {"K=N p50": float(np.percentile(t_lat, 50)), "K=N p95": float(np.percentile(t_lat, 95)),
           f"K={K_LAT} p50": float(np.percentile(t_lat100, 50)), f"K={K_LAT} p95": float(np.percentile(t_lat100, 95))}

    # ---- Phase 4: Stage-1 recall of the original for augmentation queries -------------------------
    s1 = Stage1(X, full)
    rec = {k: float(np.mean([p in set(s1.query(Xa[q], min(k, N)).tolist()) for q, p in enumerate(src)]))
           for k in K_GRID}
    (args.out / "phase4_recall.md").write_text(
        "# Phase 4 Stage-1 recall\n\nDoes the original part come back in the candidate list of its "
        f"augmentation? (N = {N}; the engine uses brute force at N <= {cfg.n_brute_max}, this only "
        "characterises Stage 1.) Class relevance is not measured: the dataset has no class labels.\n\n"
        + md_table(["K", "R@K (original)"], [[k, f"{v:.4f}"] for k, v in rec.items()]) + "\n", encoding="utf-8")

    result = {"n_parts": N, "n_aug": A, "p_star": p_star, "L2_MRR_by_p": {str(k): float(v) for k, v in rr_p.items()},
              "levels": stats, "wilcoxon_on_RR": wil, "eps": {BLOCKS[b].name: float(full.eps[b]) for b in range(15)},
              "eps_over_0.2": unreliable, "blocks_under_90pct_aug_sanity": weak, "latency_ms": lat,
              "stage1_recall": {str(k): v for k, v in rec.items()},
              "m_eff": full.m_eff, "config_hash": cfg.config_hash()}
    (args.out / "ladder_L0_L3.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    md = ["# BLR ladder L0-L3, design protocol", "",
          f"{N} parts x {A} augmentations = {N * A} queries; gallery = the {N} originals; target = the source part.",
          f"Cluster bootstrap (by part) 95% CI on MRR. L2 exponent p* = {p_star} (chosen on these queries).", "",
          md_table(["level", "MRR", "95% CI", "R@1", "R@5", "R@10", "median rank"],
                   [[lv, f"{s['MRR']:.4f}", f"[{s['MRR_CI'][0]:.4f}, {s['MRR_CI'][1]:.4f}]", f"{s['R@1']:.4f}",
                     f"{s['R@5']:.4f}", f"{s['R@10']:.4f}", f"{s['median_rank']:.0f}"] for lv, s in stats.items()]),
          "", "Paired Wilcoxon on per-query reciprocal rank (Holm across these comparisons):", "",
          md_table(["pair", "p", "p (Holm)"], [[f"{w['a']} vs {w['b']}", f"{w['p']:.3g}", f"{w['p_holm']:.3g}"]
                                               for w in wil]),
          "", f"Blocks with eps_b > {EPS_UNRELIABLE}: {', '.join(unreliable) if unreliable else 'none'}", "",
          md_table(["block", "eps_b"], [[BLOCKS[b].name, f"{full.eps[b]:.3f}"] for b in range(15)]), "",
          "Latency (S2 + S3, ms per query): " + ", ".join(f"{k} {v:.2f}" for k, v in lat.items()), ""]
    (args.out / "ladder_L0_L3.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ==============================================================================
# END OF FILE: blr_eval_ladder.py
# ==============================================================================
# SUMMARY:
#   Fits the fold null models and reports Phase 2-5 gate artefacts and the L0-L3 ladder.
#
# USAGE EXAMPLE:
#   $ python scripts/blr_eval_ladder.py --max-parts 300 --aug-draws 2
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Design protocol only: the family protocol needs class labels.
#   - [ ] Rotation-invariance check of Phase 2 is a separate test (tests/test_augment.py), not run here.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
