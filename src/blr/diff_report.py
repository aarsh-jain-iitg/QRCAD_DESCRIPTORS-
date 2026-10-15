"""
================================================================================
FILE: diff_report.py
================================================================================
PURPOSE:
    "Which descriptors make these two parts different?" For two parts (query vs
    result, or any pair) decide per block whether the pair is too far apart to be
    "the same design", and say in numbers what changed. Works with any scorer
    because it only needs the registry and the null model.

INPUTS:
    - x_a, x_b : (781,) vectors; q_a, q_b : (6,) quality statistics; NullModel.

OUTPUTS:
    - compare(...) -> DiffReport (differing / consistent / not_evaluated blocks,
      summary text, to_dict() for JSON).

DEPENDENCIES:
    stdlib      : dataclasses
    third-party : numpy
    internal    : blr.blocks, blr.config, blr.metrics, blr.null_model

PHYSICS / DOMAIN CONTEXT:
    - Test per block: H_b = "in block b these two parts are the same design", statistic
      u_b; conformal tail p_b = (1 + #{a: U1_b(a) >= u_b}) / (|A_b| + 1) from real same-design
      (augmentation) pairs, so noisy blocks are honest.
    - Weighted Holm across blocks, weights w_b = omega_b(a) omega_b(b): blocks with w = 0 are
      "not_evaluated"; the rest are sorted by p_b / w_b and rejected ("differs") while
      p_(j) <= alpha w_(j) / sum_{i >= j} w_(i); the first failure and everything after is
      "consistent". Family-wise error <= alpha_diff.
    - "differs" comes from p_b, never from the sign of the evidence e_b.
    - Text is deterministic (no LLM): built from the sub-part kind and the numbers.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.blr.blocks import BLOCKS, BlockSpec, SubPart
from src.blr.config import BLRConfig
from src.blr.metrics import DIST_SUBPARTS, subpart_distances
from src.blr.null_model import NullModel

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MIN_SHARE = 0.2                   # [-] sub-parts below this share of the block distance are not listed
MAX_LINES = 3                     # [-] sub-part lines per block


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SubpartDiff:
    name: str
    kind: str
    share: float
    a_summary: str
    b_summary: str
    delta: float
    unit: str
    text: str


@dataclass(frozen=True)
class BlockDiff:
    block: str
    plain_name: str
    verdict: str                      # differs | consistent | not_evaluated
    p_value: float
    p_threshold: float
    evidence_nats: float
    reliability: float
    u: float
    subparts: tuple[SubpartDiff, ...]
    text: str


@dataclass(frozen=True)
class DiffReport:
    differing: tuple[BlockDiff, ...]          # sorted by evidence ascending (strongest "against" first)
    consistent: tuple[BlockDiff, ...]
    not_evaluated: tuple[BlockDiff, ...]
    summary: str

    def to_dict(self) -> dict:
        """JSON-safe dict."""
        return asdict(self)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def weighted_holm(p: np.ndarray, w: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted Holm step-down (Holm 1979) with reliability weights.

    Parameters
    ----------
    p : np.ndarray, shape (m,)
        p-values.
    w : np.ndarray, shape (m,)
        Weights >= 0; blocks with w == 0 are excluded (never rejected).
    alpha : float
        Family-wise error rate.

    Returns
    -------
    reject : np.ndarray of bool, shape (m,)
    threshold : np.ndarray, shape (m,)
        alpha * w_(j) / sum_{i>=j} w_(i) at the block's step-down position (0 if excluded).
    """
    m = len(p)
    reject, thr = np.zeros(m, bool), np.zeros(m)
    idx = np.array([i for i in np.argsort(p / np.where(w > 0, w, np.inf), kind="stable") if w[i] > 0], int)
    tail = np.cumsum(w[idx][::-1])[::-1]
    thr[idx] = alpha * w[idx] / tail
    for k, i in enumerate(idx):
        if p[i] <= thr[i]:
            reject[i] = True
        else:
            break
    return reject, thr


def _probs(v: np.ndarray, sp: SubPart) -> np.ndarray:
    a = np.asarray(v, np.float64) ** 2 if sp.stored_sqrt else np.asarray(v, np.float64)
    s = a.sum()
    return a / s if s > 0 else a


def _median_axis(pr: np.ndarray, centres: tuple) -> float:
    """Mass-weighted median of a histogram over its bin centres (axis units)."""
    return float(centres[int(np.searchsorted(np.cumsum(pr), 0.5))]) if pr.sum() > 0 else float("nan")


def _fmt(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.3g}"


def describe(sp: SubPart, a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> tuple[str, str, float, str]:
    """(a_summary, b_summary, delta, detail line) of one sub-part from a to b."""
    if sp.kind in ("hist", "hist2d"):
        pa, pb = _probs(a, sp), _probs(b, sp)
        if pa.sum() == 0 or pb.sum() == 0:
            sa, sb = ("no data" if pa.sum() == 0 else "data"), ("no data" if pb.sum() == 0 else "data")
            return sa, sb, float("nan"), f"{sp.name}: {sa} -> {sb}"
        d = pb - pa
        lose, gain = int(np.argmin(d)), int(np.argmax(d))
        if sp.kind == "hist":
            ma, mb = _median_axis(pa, sp.labels), _median_axis(pb, sp.labels)
            return (_fmt(ma), _fmt(mb), mb - ma,
                    f"{sp.unit}: median {_fmt(ma)} -> {_fmt(mb)} (mass moved from {_fmt(sp.labels[lose])} "
                    f"to {_fmt(sp.labels[gain])}, {100 * float(np.abs(d).sum() / 2):.0f}% shifted)")
        return (str(sp.labels[lose]), str(sp.labels[gain]), float(d[gain]),
                f"{sp.unit}: {sp.labels[lose]} -> {sp.labels[gain]} ({100 * float(np.abs(d).sum() / 2):.0f}% shifted)")
    if sp.kind == "spectrum":
        j = int(np.argmax(np.abs(np.asarray(b, float) - np.asarray(a, float))))
        return (f"{a[j]:.3g}", f"{b[j]:.3g}", float(b[j] - a[j]),
                f"{sp.labels[j]}: {a[j]:.3g} -> {b[j]:.3g}")
    if sp.kind == "count":
        def n(v, lab):        # only genus is stored as log(1 + g); bodies, cavities, loops are raw counts
            return round(float(np.expm1(v))) if lab == "genus" else round(float(v))
        j = int(np.argmax([abs(n(a[i], sp.labels[i]) - n(b[i], sp.labels[i])) for i in range(len(a))]))
        na, nb = n(a[j], sp.labels[j]), n(b[j], sp.labels[j])
        return str(na), str(nb), float(nb - na), f"{sp.labels[j]} {na} -> {nb}"
    z = (np.asarray(b, float) - np.asarray(a, float)) / scale[sp.sl]              # scalar
    j = int(np.argmax(np.abs(z)))
    return (f"{a[j]:.3g}", f"{b[j]:.3g}", float(b[j] - a[j]),
            f"{sp.labels[j]}: {a[j]:.3g} -> {b[j]:.3g}")


def _block_text(spec: BlockSpec, subs: tuple[SubpartDiff, ...]) -> str:
    lines = "; ".join(s.text for s in subs)
    return f"{spec.plain_name}: {spec.what_it_sees}" + (f" [{lines}]" if lines else "")


# ── CORE FUNCTION ─────────────────────────────────────────────────────────────
def compare(x_a: np.ndarray, x_b: np.ndarray, q_a: np.ndarray, q_b: np.ndarray, model: NullModel,
            alpha_diff: float | None = None) -> DiffReport:
    """
    Per-block difference report between two parts.

    Parameters
    ----------
    x_a, x_b : np.ndarray, shape (781,)
    q_a, q_b : np.ndarray, shape (6,)
        Quality statistics (quality.QUALITY_KEYS order).
    model : NullModel
    alpha_diff : float, optional
        Family-wise error rate (default BLRConfig().alpha_diff).

    Returns
    -------
    DiffReport
        Verdicts are symmetric in (a, b); only the "a -> b" wording swaps.
    """
    alpha = BLRConfig().alpha_diff if alpha_diff is None else alpha_diff
    sp = model.space
    sub = subpart_distances(x_a, x_b[None], sp.scale)[0] / sp.medians
    D = np.array([sub[[j for j, (bi, _) in enumerate(DIST_SUBPARTS) if bi == b]].sum() for b in range(len(BLOCKS))])
    U = sp.u(D)
    p = model.pvalue(U)
    w = model.omega_of(q_a)[0] * model.omega_of(q_b)[0]
    e = model.family_factor() * w * model.g0(U)
    reject, thr = weighted_holm(p, w, alpha)
    diffs, cons, none = [], [], []
    for b, spec in enumerate(BLOCKS):
        verdict = "not_evaluated" if w[b] <= 0 else ("differs" if reject[b] else "consistent")
        subs = ()
        if verdict == "differs":
            rows = [(j, s) for j, (bi, s) in enumerate(DIST_SUBPARTS) if bi == b]
            rows.sort(key=lambda r: -sub[r[0]])
            picked = [r for r in rows if sub[r[0]] / max(D[b], 1e-12) >= MIN_SHARE][:MAX_LINES] or rows[:1]
            subs = tuple(SubpartDiff(s.name, s.kind, float(sub[j] / max(D[b], 1e-12)), *d[:3], s.unit, d[3])
                         for j, s in picked for d in [describe(s, x_a[s.sl], x_b[s.sl], sp.scale)])
        bd = BlockDiff(spec.name, spec.plain_name, verdict, float(p[b]), float(thr[b]), float(e[b]),
                       float(w[b]), float(U[b]), subs, _block_text(spec, subs) if subs else spec.plain_name)
        {"differs": diffs, "consistent": cons, "not_evaluated": none}[verdict].append(bd)
    diffs.sort(key=lambda d: d.evidence_nats)
    n_eval = len(diffs) + len(cons)
    summary = (f"Differs in {len(diffs)} of {n_eval} evaluated descriptors"
               + (": " + ", ".join(d.plain_name for d in diffs) if diffs else " (same design in every block)")
               + (f"; {len(none)} not evaluated (unreliable for this pair)" if none else ""))
    return DiffReport(tuple(diffs), tuple(cons), tuple(none), summary)


# ==============================================================================
# END OF FILE: diff_report.py
# ==============================================================================
# SUMMARY:
#   Per-block same-design test (conformal p + weighted Holm) with numeric "what changed" text.
#
# USAGE EXAMPLE:
#   rep = compare(xa, xb, qa, qb, model); print(rep.summary); rep.to_dict()
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] T1 counts: bodies / cavities are stored raw, genus as log(1 + g); text inverts accordingly.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
