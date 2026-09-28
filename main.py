"""
================================================================================
FILE: main.py
================================================================================
PURPOSE:
    Single entry point of the QRCAD 15-descriptor retrieval pipeline:
    load src/config.toml, build or refresh the descriptor cache for every
    configured dataset (incremental), query one STL, print a results table
    and write an HTML / PNG / JSON report. --eval adds a leave-one-out
    evaluation table.

INPUTS:
    - QUERY_FILE below, or --query path/to/file.stl (relative to the repo root).
    - --rebuild (ignore cache), --top-k N, --eval, --config path.
    - src/config.toml; STL files under dataset/<dataset>/<class>/.

OUTPUTS:
    - Terminal results table (and evaluation table with --eval).
    - outputs/<query_stem>_<timestamp>/{report.html, results.png, results.json}
      (+ evaluation.json / evaluation.md with --eval).
    - cache/ (descriptor vectors and index).

DEPENDENCIES:
    stdlib      : argparse, datetime, json, logging, pathlib, sys
    third-party : (via src)
    internal    : src.config, src.orchestrator, src.report, src.sample_dataset

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config import DEFAULT_CONFIG, load_config
from src.orchestrator import build, evaluate, query
from src.report import format_table, write_report
from src.sample_dataset import generate_sample_dataset

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent                  # repo root
DATASET_DIR = ROOT_DIR / "dataset"
SAMPLE_DIR = DATASET_DIR / "_sample"

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# ▶▶ QUERY FILE: path relative to the repo root (or pass --query) ◀◀
QUERY_FILE = "dataset/_sample/washer/washer_00.stl"


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def eval_markdown(ev: dict) -> str:
    """Evaluation result as a Markdown table (also pasted into README.md)."""
    lines = [f"Leave-one-out on {ev['n_models']} models, {ev['n_classes']} classes.", "",
             "| Method | P@1 | P@5 | P@10 | mAP | ms/mesh |", "|---|---|---|---|---|---|"]
    for name, m in ev["metrics"].items():
        ms = ev["ms_per_mesh"].get(name)
        ms_s = f"{ms:.0f}" if ms is not None else "-"
        lines.append(f"| {name} | {m.get('P@1', 0):.3f} | {m.get('P@5', 0):.3f} | "
                     f"{m.get('P@10', 0):.3f} | {m['mAP']:.3f} | {ms_s} |")
    return "\n".join(lines)


def has_stl(folder: Path) -> bool:
    return any(p.suffix.lower() == ".stl" for p in folder.rglob("*") if p.is_file())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="QRCAD 15-descriptor CAD retrieval")
    ap.add_argument("--query", type=Path, default=Path(QUERY_FILE), help="query STL")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="config.toml")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cache")
    ap.add_argument("--top-k", type=int, default=None, help="override [database].top_k")
    ap.add_argument("--eval", action="store_true", help="leave-one-out evaluation")
    return ap.parse_args(argv)


# ── CLI ENTRYPOINT ────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    DATASET_DIR.mkdir(exist_ok=True)
    if not has_stl(DATASET_DIR):
        print(f"dataset/ is empty: generating a synthetic sample dataset in {SAMPLE_DIR}")
        generate_sample_dataset(SAMPLE_DIR)

    cfg = load_config(args.config, ROOT_DIR)
    report = build(cfg, rebuild=args.rebuild)
    print(f"\nbuild: {report.n_indexed}/{report.n_files} files indexed from {report.datasets}, "
          f"{report.n_files_extracted} (re)extracted, {report.seconds:.1f} s, device={report.device}")
    for name, msg in report.failures:
        print(f"  skipped/failed: {name}: {msg}")

    qpath = args.query if args.query.is_absolute() else ROOT_DIR / args.query
    if not qpath.is_file():
        print(f"query file not found: {qpath}", file=sys.stderr)
        return 2
    result = query(qpath, cfg, args.top_k)
    print()
    print(format_table(result))

    out = cfg.output_dir / f"{qpath.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    write_report(result, ROOT_DIR, out)
    print(f"\nreport: {out / 'report.html'}\n        {out / 'results.png'}\n        {out / 'results.json'}")

    if args.eval:
        ev = evaluate(cfg)
        md = eval_markdown(ev)
        (out / "evaluation.json").write_text(json.dumps(ev, indent=1), encoding="utf-8")
        (out / "evaluation.md").write_text(md + "\n", encoding="utf-8")
        print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ==============================================================================
# END OF FILE: main.py
# ==============================================================================
# SUMMARY:
#   Build/refresh cache -> query -> table + HTML/PNG/JSON report (+ --eval).
#
# USAGE EXAMPLE:
#   $ python main.py
#   $ python main.py --query dataset/_sample/pin/pin_03.stl --top-k 5 --eval
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] None.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
