"""
================================================================================
FILE: blr_compare.py
================================================================================
PURPOSE:
    Difference report for any two STL files: which descriptor blocks make them
    different, with p-values and numeric "what changed" lines.

INPUTS:
    --a A.stl --b B.stl --model-dir DIR (folder holding null_model.npz/.json, e.g.
    <dataset>/.qrcad_cache/blr/<hash>), --json out.json (optional).

OUTPUTS:
    - Table on stdout; optional JSON (DiffReport.to_dict()).

DEPENDENCIES:
    stdlib      : argparse, json, logging, pathlib, sys
    third-party : numpy
    internal    : src.blr.diff_report, src.blr.extract, src.blr.null_model

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.blr.config import BLRConfig  # noqa: E402
from src.blr.diff_report import compare  # noqa: E402
from src.blr.extract import extract_with_quality  # noqa: E402
from src.blr.null_model import NullModel  # noqa: E402
from src.blr.quality import QUALITY_KEYS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BLR: which descriptors make two parts different?")
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    model = NullModel.load(args.model_dir / "null_model")
    cfg = BLRConfig()
    (xa, qa), (xb, qb) = (extract_with_quality(p, cfg.seed, blr=cfg) for p in (args.a, args.b))
    rep = compare(xa, xb, np.array([qa[k] for k in QUALITY_KEYS]), np.array([qb[k] for k in QUALITY_KEYS]), model)
    print(rep.summary, "\n")
    print(f"{'block':<28}{'verdict':<15}{'p':>9}{'thr':>9}{'evid[nats]':>12}{'w':>6}")
    for d in (*rep.differing, *rep.consistent, *rep.not_evaluated):
        print(f"{d.plain_name:<28}{d.verdict:<15}{d.p_value:>9.4f}{d.p_threshold:>9.4f}"
              f"{d.evidence_nats:>12.2f}{d.reliability:>6.2f}")
    for d in rep.differing:
        print(f"\n* {d.text}")
    if args.json:
        args.json.write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ==============================================================================
# END OF FILE: blr_compare.py
# ==============================================================================
# SUMMARY:
#   CLI difference report for two STL files against a fitted null model.
#
# USAGE EXAMPLE:
#   $ python scripts/blr_compare.py --a a.stl --b b.stl --model-dir dataset/small_set/.qrcad_cache/blr/<hash>
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Needs a null model written by scripts/blr_eval_ladder.py.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
