"""
================================================================================
FILE: blr_build_null.py
================================================================================
PURPOSE:
    Extract the gallery + augmentations of a dataset (cached) and fit the BLR
    null model (added in Phase 3). Long step: about 5 s per extraction, pooled.

INPUTS:
    --dataset DIR (default dataset/small_set), --workers N, --rebuild.

OUTPUTS:
    - <dataset>/.qrcad_cache/blr/<hash>/gallery.npz (and null_model.* once fitted).

DEPENDENCIES:
    stdlib      : argparse, dataclasses, logging, pathlib, sys
    third-party : none
    internal    : src.blr.config, src.blr.gallery

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.blr.config import BLRConfig  # noqa: E402
from src.blr.gallery import build_gallery  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BLR: extract gallery + augmentations")
    ap.add_argument("--dataset", type=Path, default=ROOT_DIR / "dataset" / "small_set")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--max-parts", type=int, default=None, help="seeded random subset")
    ap.add_argument("--aug-draws", type=int, default=None, help="augmentations per part (default 4)")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = BLRConfig() if args.aug_draws is None else dataclasses.replace(BLRConfig(), n_aug_draws=args.aug_draws)
    g = build_gallery(args.dataset, cfg, args.workers, args.rebuild, args.retry_failed, args.max_parts)
    logging.getLogger("blr").info("gallery ready: %d parts", len(g.paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ==============================================================================
# END OF FILE: blr_build_null.py
# ==============================================================================
# SUMMARY:
#   CLI: build the BLR gallery cache.
#
# USAGE EXAMPLE:
#   $ python scripts/blr_build_null.py --dataset dataset/small_set
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Null-model fit is added in Phase 3.
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
