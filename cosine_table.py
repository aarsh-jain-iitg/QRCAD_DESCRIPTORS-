"""
================================================================================
FILE: cosine_table.py
================================================================================
PURPOSE:
    One-page overview of the descriptor-alignment pairs: a table with one row
    per A/B pair (both part pictures in the first cell) and one column per
    descriptor (D1 ... MTS), each cell the plain cosine similarity
    a.b / (|a||b|) of that descriptor's vectors for part A and part B.
    The page is a static HTML file next to the images, so a double-click
    opens it in any browser (no server, no internet).

INPUTS:
    - <out-dir>/manifest.json                     pairs + part.png paths (from generate_ss.py)
    - <out-dir>/<pair>/<stl-stem>/vectors.npz     raw vectors v01..v15 (from generate_ss.py)
    - CLI: --out-dir DIR   folder written by generate_ss.py (default: outputs/descriptor_ss)

OUTPUTS:
    - <out-dir>/cosine_table.html   (image links are relative, so keep it in that folder)

DEPENDENCIES:
    stdlib      : argparse, html, json, logging, math, pathlib
    third-party : numpy
    internal    : generate_ss (cosine, DESCRIPTORS, OUT_DIR), testing (DESCRIPTOR_MAP, FULL_NAMES)

PHYSICS / DOMAIN CONTEXT:
    - Cosine of the raw descriptor vectors exactly as generate_ss.py stores them
      (Hellinger histograms + unscaled scalars, no database z-scoring) [-].
      1 = the descriptor cannot tell A from B.
    - Cell shade encodes 1 - cos on a log scale, because most values sit in
      0.99..1.0 and a linear scale would paint them all the same.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-15
================================================================================
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
from pathlib import Path

import numpy as np

import testing as T
from generate_ss import DESCRIPTORS, OUT_DIR, cosine

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent                    # project root
PAGE_NAME = "cosine_table.html"                               # written inside --out-dir

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
LOG_DIFF_MIN = -4.0          # [-] log10(1 - cos) at or below this -> lightest shade (identical)
LOG_DIFF_MAX = 0.0           # [-] log10(1 - cos) at or above this -> darkest shade (orthogonal)
SHADES = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # deck palette, light -> dark
DARK_TEXT_FROM = 3           # [-] shade index from which cell text turns white

logger = logging.getLogger(__name__)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def shade_index(cos: float) -> int:
    """
    Map a cosine similarity to a palette index (0 = identical .. 5 = orthogonal).

    Parameters
    ----------
    cos : float
        Cosine similarity [-].

    Returns
    -------
    int
        Index into SHADES.
    """
    diff = max(1.0 - cos, 10.0 ** LOG_DIFF_MIN)
    t = (math.log10(diff) - LOG_DIFF_MIN) / (LOG_DIFF_MAX - LOG_DIFF_MIN)
    return min(len(SHADES) - 1, max(0, round(t * (len(SHADES) - 1))))


def side_label(stem: str) -> str:
    """'R08A_block_sharp' -> 'block sharp'."""
    return stem.split("_", 1)[1].replace("_", " ") if "_" in stem else stem


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def pair_cosines(out_dir: Path, key: str, entry: dict) -> list[float | None]:
    """
    Cosine A vs B for every descriptor of one pair.

    Parameters
    ----------
    out_dir : Path
        Folder written by generate_ss.py.
    key : str
        Pair key, e.g. "R08".
    entry : dict
        manifest["pairs"][key] (needs "A" and "B" STL stems).

    Returns
    -------
    list[float | None]
        One value per descriptor in DESCRIPTORS order; None if a vector is missing.
    """
    va = np.load(out_dir / key / entry["A"] / "vectors.npz")
    vb = np.load(out_dir / key / entry["B"] / "vectors.npz")
    out: list[float | None] = []
    for idx in DESCRIPTORS:
        k = f"v{idx:02d}"
        out.append(cosine(va[k], vb[k]) if k in va and k in vb else None)
    return out


def build_page(out_dir: Path) -> Path:
    """
    Write cosine_table.html into out_dir.

    Parameters
    ----------
    out_dir : Path
        Folder written by generate_ss.py (holds manifest.json and the pair folders).

    Returns
    -------
    Path
        Path of the written page.

    Raises
    ------
    FileNotFoundError
        If manifest.json is missing (run generate_ss.py first).
    """
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    head_cells = []
    for idx in DESCRIPTORS:
        code, _, long = T.FULL_NAMES[T.DESCRIPTOR_MAP[idx].name].partition(" · ")
        head_cells.append(f'<th><span class="idx">{idx}</span><b>{html.escape(code)}</b>'
                          f'<small>{html.escape(long)}</small></th>')

    rows = []
    for key, entry in sorted(manifest["pairs"].items()):
        a, b = side_label(entry["A"]), side_label(entry["B"])
        imgs = "".join(
            f'<a href="{html.escape(entry["parts"][s])}" target="_blank"><figure>'
            f'<img src="{html.escape(entry["parts"][s])}" alt="{s}: {html.escape(lbl)}" loading="lazy">'
            f'<figcaption><b>{s}</b> {html.escape(lbl)}</figcaption></figure></a>'
            for s, lbl in (("A", a), ("B", b)))
        cells = []
        for idx, cos in zip(DESCRIPTORS, pair_cosines(out_dir, key, entry)):
            if cos is None:
                cells.append('<td class="na">failed</td>')
                continue
            s = shade_index(cos)
            cls = ' class="inv"' if s >= DARK_TEXT_FROM else ""
            cells.append(f'<td{cls} style="background:{SHADES[s]}" '
                         f'title="{html.escape(key)} · descriptor {idx} · cos = {cos:.6f}">{cos:.4f}</td>')
        rows.append(f'<tr><th class="pair"><div class="key">{html.escape(key)}</div>'
                    f'<div class="parts">{imgs}</div></th>{"".join(cells)}</tr>')

    step = (LOG_DIFF_MAX - LOG_DIFF_MIN) / (len(SHADES) - 1)   # shade centres, back to cos
    legend = "".join(
        f'<span style="background:{c}"{" class=inv" if i >= DARK_TEXT_FROM else ""}>'
        f'{1.0 - 10.0 ** (LOG_DIFF_MIN + i * step):.4f}</span>'
        for i, c in enumerate(SHADES))
    page = PAGE_TEMPLATE.format(
        n_pairs=len(rows), seed=manifest.get("seed"), n_samples=manifest.get("n_samples"),
        legend=legend, head="".join(head_cells), rows="\n".join(rows))
    page_path = out_dir / PAGE_NAME
    page_path.write_text(page, encoding="utf-8")
    return page_path


PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Descriptor Alignment · cosine table</title>
<style>
:root {{ --ink:#1a1a1a; --ink2:#6b6b6b; --mute:#8c8c8c; --line:#e4e3df; --bg:#ffffff; --soft:#f1f1f1; --accent:#2a78d6; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Calibri,Carlito,"Segoe UI",Arial,sans-serif; }}
header {{ padding:28px 32px 12px; border-bottom:4px solid var(--accent); }}
h1 {{ margin:0; font-size:34px; letter-spacing:.06em; }}
.sub {{ color:var(--mute); font-size:15px; margin-top:4px; }}
.legend {{ display:flex; align-items:center; gap:0; margin-top:14px; font-size:13px; color:var(--ink2); flex-wrap:wrap; }}
.legend span {{ padding:4px 10px; border:1px solid var(--line); margin-right:-1px; color:var(--ink); }}
.legend .inv {{ color:#fff; }}
.legend em {{ font-style:normal; margin-right:10px; }}
.wrap {{ overflow:auto; max-height:calc(100vh - 150px); margin:16px 32px 32px; border:1px solid var(--line); }}
table {{ border-collapse:separate; border-spacing:0; font-variant-numeric:tabular-nums; }}
thead th {{ position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:2px solid var(--ink);
  padding:10px 8px; min-width:92px; vertical-align:bottom; text-align:center; font-weight:normal; }}
thead th b {{ display:block; font-size:17px; }}
thead th small {{ display:block; color:var(--mute); font-size:12px; line-height:1.2; margin-top:2px; }}
thead th .idx {{ display:block; color:var(--accent); font-size:12px; font-weight:bold; }}
thead th.corner {{ left:0; z-index:3; text-align:left; }}
th.pair {{ position:sticky; left:0; z-index:1; background:var(--bg); border-right:2px solid var(--ink);
  border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; font-weight:normal; }}
.key {{ font-weight:bold; color:var(--accent); font-size:15px; }}
.parts {{ display:flex; gap:8px; }}
.parts a {{ color:inherit; text-decoration:none; }}
figure {{ margin:0; width:120px; }}
figure img {{ width:120px; height:93px; object-fit:contain; display:block; border:1px solid var(--line); background:#fff; }}
figure img:hover {{ border-color:var(--accent); }}
figcaption {{ font-size:12px; color:var(--ink2); line-height:1.2; margin-top:2px; }}
figcaption b {{ color:var(--ink); }}
td {{ text-align:center; font-size:15px; padding:0 8px; border-bottom:1px solid var(--line); border-right:1px solid var(--line); }}
td.inv {{ color:#fff; }}
td.na {{ color:var(--mute); background:var(--soft); font-style:italic; }}
tbody tr:hover th.pair {{ background:var(--soft); }}
</style></head>
<body>
<header>
  <h1>DESCRIPTOR ALIGNMENT</h1>
  <div class="sub">Cosine similarity of part A vs part B, per descriptor · {n_pairs} pairs · seed {seed}, {n_samples:,} samples · click a part to enlarge</div>
  <div class="legend"><em>cos(A, B):</em>{legend}<em style="margin-left:10px">light = A and B look the same to this descriptor · dark = it tells them apart</em></div>
</header>
<div class="wrap"><table>
<thead><tr><th class="corner">Pair (A · B)</th>{head}</tr></thead>
<tbody>
{rows}
</tbody></table></div>
</body></html>
"""


# ── CLI ENTRYPOINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="A/B cosine table (HTML) for the descriptor-alignment pairs")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="folder written by generate_ss.py")
    args = ap.parse_args()
    assert shade_index(1.0) == 0 and shade_index(0.0) == len(SHADES) - 1 and shade_index(-1.0) == len(SHADES) - 1
    logger.info("wrote %s", build_page(args.out_dir))

# ==============================================================================
# END OF FILE: cosine_table.py
# ==============================================================================
# SUMMARY:
#   Builds a double-clickable HTML table: rows = A/B part pairs (with pictures),
#   columns = the 15 descriptors, cells = cosine(A, B) of the raw vectors.
#
# USAGE EXAMPLE:
#   $ python generate_ss.py          # vectors + part pictures (skip if up to date)
#   $ python cosine_table.py         # -> outputs/descriptor_ss/cosine_table.html
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Reads raw vectors, so scalar dims are not z-scored as in the database
#         fusion; the cosine here is per-descriptor, not the retrieval score.
#
# CHANGELOG:
#   2026-10-15  Initial implementation
# ==============================================================================
