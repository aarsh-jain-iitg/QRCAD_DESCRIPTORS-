"""
================================================================================
FILE: testing.py
================================================================================
PURPOSE:
    Interactive inspection of the 15 space-invariant descriptors on ONE STL
    file. Set `input_file` and `target_descriptors` (integers 1..15, PDF ladder
    order) below, run the script, and every requested descriptor vector is
    computed with the exact pipeline code path (same preprocessing, params,
    seeds) and shown on one scrollable local web page. Each vector is split into its sub-blocks
    (histograms, 2-D images, spectra, scalars) and every block is drawn on
    its physical axis (d_hat, theta [deg], log10 w_hat, ...), per
    CAD_Descriptors_Explained.pdf. Nothing is written to disk.

INPUTS:
    - folder_mode        : False -> inspect input_file; True -> every STL in input_folder.
    - input_file         : path of the STL to inspect (relative to this file or absolute).
    - input_folder       : folder searched recursively for *.stl in folder mode.
    - target_descriptors : list[int] in 1..15 (see DESCRIPTOR_MAP).
    - src/config.toml    : descriptor params, seed, n_samples, scale_rule.

OUTPUTS:
    - Terminal: mesh summary + one aligned summary table of all descriptors.
    - One HTML page per STL served from memory at http://127.0.0.1:<port>/<i>,
      each opened in its own browser tab (one card per descriptor, inline SVG).
      No files saved.

DEPENDENCIES:
    stdlib      : gc, html, http.server, io, logging, math, pathlib, threading, time,
                  webbrowser
    third-party : numpy, matplotlib
    internal    : src.config, src.preprocessing, src.orchestrator, src.descriptors.*

PHYSICS / DOMAIN CONTEXT:
    - Vectors are dimensionless (lengths / D_diam, areas / D_diam^2, ...).
      The first n_hist dims are sqrt(L1-normalised) histograms (Hellinger map);
      the rest are raw scalars (z-scored only inside the database, not here).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-25
================================================================================
"""

from __future__ import annotations

import gc
import html
import http.server
import io
import logging
import math
import threading
import time
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # render off-screen; figures go to the page, not windows
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.config import load_config
from src.orchestrator import descriptor_rng
from src.preprocessing import prepare_mesh

# ── DESCRIPTOR MODULES (relink here if your files live elsewhere) ─────────────
from src.descriptors.base import Descriptor
from src.descriptors.d1_radial_distance import D1RadialDistance              # 1
from src.descriptors.d2_pairwise_distance import D2PairwiseDistance          # 2
from src.descriptors.triangle_a3_d3 import SQRT_AREA_MAX, TriangleA3D3       # 3
from src.descriptors.d4_tetrahedron_volume import CBRT_VOL_MAX, D4TetrahedronVolume  # 4
from src.descriptors.shps_orientation_spectrum import SHPSOrientationSpectrum  # 5
from src.descriptors.ncf_normal_correlation import NCFNormalCorrelation      # 6
from src.descriptors.lbe_laplace_beltrami import LBELaplaceBeltrami          # 7
from src.descriptors.dad_dihedral_angle import DADDihedralAngle              # 8
from src.descriptors.curvature_shape_index import CurvatureShapeIndex        # 9
from src.descriptors.crs_r_ring_profile import CRSRRingProfile               # 10
from src.descriptors.t1_genus import T1Genus                                 # 11
from src.descriptors.phd_persistent_homology import PHDPersistentHomology    # 12
from src.descriptors.crs_a_angular_spectrum import CRSAAngularSpectrum       # 13
from src.descriptors.paf_planar_families import PAFPlanarFamilies            # 14
from src.descriptors.mts_wall_thickness import MTSWallThickness              # 15

# ── USER INPUTS ───────────────────────────────────────────────────────────────
folder_mode = True                     # False: input_file only; True: every STL in input_folder
input_file = r"C:\Users\aarsh\Desktop\PARIKALP\NEW_QCAD\testing files\R07A_plate_plain.stl"   # one STL, relative to this file
input_folder = r"C:\Users\aarsh\Desktop\PARIKALP\NEW_QCAD\testing files"          # relative to this file, or absolute
target_descriptors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent                # project root
CONFIG_PATH = ROOT_DIR / "src" / "config.toml"            # descriptor params

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
DESCRIPTOR_MAP: dict[int, type[Descriptor]] = {
    1: D1RadialDistance, 2: D2PairwiseDistance, 3: TriangleA3D3,
    4: D4TetrahedronVolume, 5: SHPSOrientationSpectrum, 6: NCFNormalCorrelation,
    7: LBELaplaceBeltrami, 8: DADDihedralAngle, 9: CurvatureShapeIndex,
    10: CRSRRingProfile, 11: T1Genus, 12: PHDPersistentHomology,
    13: CRSAAngularSpectrum, 14: PAFPlanarFamilies, 15: MTSWallThickness,
}
FULL_NAMES: dict[str, str] = {
    "d1": "D1 · radial distance", "d2": "D2 · pairwise distance",
    "triangle": "A3 / D3 · random triangles", "d4": "D4 · tetrahedron volume",
    "shps": "SHPS · normal orientation spectrum", "ncf": "NCF · normal correlation",
    "lbe": "LBE · Laplace-Beltrami eigenvalues", "dad": "DAD · dihedral angles",
    "curvature": "Curvature · shape index × curvedness", "crs_r": "CRS-R · ring profiles",
    "t1": "T1 · topology", "phd": "PHD · persistence images",
    "crs_a": "CRS-A · angular spectrum", "paf": "PAF · planar families",
    "mts": "MTS · wall thickness",
}

# Palette: dataviz reference instance (light mode), validated there.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"                 # titles, values
INK_2 = "#52514e"               # axis labels, ticks
GRID = "#e4e3df"
SERIES = "#2a78d6"              # single-series marks (blue)
NEGATIVE = "#e34948"            # diverging pole for signed values (red)
FIG_SIZE_IN = (13.0, 3.4)       # [in] every figure, so fonts match across cards
SEQ_CMAP = LinearSegmentedColormap.from_list(   # sequential blue, near-zero recedes
    "seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# Panel = (kind, start, stop, title, opts); kinds: hist | profile | grid | scalars.
Panel = tuple[str, int, int, str, dict]

logger = logging.getLogger(__name__)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def apply_style() -> None:
    """Recessive axes and grid, one text ink, shared by every figure."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "font.size": 9,
    })


def layout(d: Descriptor) -> list[Panel]:
    """
    Split a descriptor vector into labelled sub-blocks on their physical axes.

    Offsets follow each descriptor's OUTPUTS contract (histogram dims first).
    Unknown descriptors fall back to [histogram | scalars] on bin index.

    Returns
    -------
    list[Panel]
        Panels whose (start, stop) ranges tile [0, d.dim).
    """
    p, n, dim = d.params, d.n_hist, d.dim
    match d.name:
        case "d1":
            return [("hist", 0, n, r"$\hat d_i = \|p_i-c\| / \max_j d_j$",
                     {"lo": 0.0, "hi": 1.0, "xlabel": r"$\hat d$"})]
        case "d2":
            return [("hist", 0, n, r"$\hat d_{ij} = \|p_i-p_j\| / D_\varnothing$",
                     {"lo": 0.0, "hi": 1.0, "xlabel": r"$\hat d_{ij}$"})]
        case "triangle":
            b = int(p["bins"])
            return [("hist", 0, b, r"A3 · angle $\theta$", {"lo": 0.0, "hi": 180.0, "xlabel": r"$\theta$ [deg]"}),
                    ("hist", b, 2 * b, r"D3 · $\sqrt{\hat A}$, $\hat A = \frac{1}{2}\|u\times w\|/D_\varnothing^2$",
                     {"lo": 0.0, "hi": SQRT_AREA_MAX, "xlabel": r"$\sqrt{\hat A}$"})]
        case "d4":
            return [("hist", 0, n, r"$\sqrt[3]{|V| / D_\varnothing^3}$",
                     {"lo": 0.0, "hi": CBRT_VOL_MAX, "xlabel": r"$\sqrt[3]{|V|/D_\varnothing^3}$"}),
                    ("scalars", n, dim, "Handedness", {"labels": [r"$\gamma_1(V)$"]})]
        case "shps":
            return [("profile", 0, dim, r"$\tilde P_\ell = \frac{4\pi}{2\ell+1}\sum_m c_{\ell m}^2$",
                     {"x": np.arange(2, 2 + dim), "xlabel": r"degree $\ell$"})]
        case "ncf":
            return [("profile", 0, dim, r"$g(m) = \frac{1}{S}\sum \hat n_i\cdot\hat n_j\,[\hat r_{ij}\in B_m]$",
                     {"x": (np.arange(dim) + 0.5) / dim, "xlabel": r"$\hat r_{ij}$", "signed": True})]
        case "lbe":
            return [("profile", 0, dim, r"$\log(\lambda_k A / 4\pi k)$",
                     {"x": np.arange(1, dim + 1), "xlabel": "eigenvalue index k", "line": True})]
        case "dad":
            return [("hist", 0, n, r"Crease angle $\theta_e$ (length-weighted)",
                     {"lo": 0.0, "hi": 360.0, "xlabel": r"$\theta_e$ [deg]  (<180 convex, >180 concave)"}),
                    ("scalars", n, dim, "Crease totals", {"labels": [r"$\Lambda$", "H"]})]
        case "curvature":
            si, cv = int(p["si_bins"]), int(p["cv_bins"])
            return [("grid", 0, n, "Joint histogram (smooth samples)",
                     {"shape": (si, cv), "extent": [-1, 1, math.log10(p["flat_eps"]), math.log10(p["cv_max"])],
                      "xlabel": "shape index SI  (−1 cup … +1 cap)", "ylabel": r"$\log_{10}$ cCV"}),
                    ("scalars", n, dim, "Flat share", {"labels": [r"cCV < $\epsilon$"]})]
        case "crs_r":
            a, o, hi = int(p["bins_axis"]), int(p["bins_other"]), float(p["rho_max"])
            xl = r"$\hat\rho = \|(p-c)\times e_a\| / D_\varnothing$"
            return [("hist", 0, a, "Symmetry axis", {"lo": 0.0, "hi": hi, "xlabel": xl}),
                    ("hist", a, a + o, "Principal axis 2", {"lo": 0.0, "hi": hi, "xlabel": xl}),
                    ("hist", a + o, n, "Principal axis 3", {"lo": 0.0, "hi": hi, "xlabel": xl})]
        case "t1":
            return [("scalars", 0, dim, "Topology",
                     {"labels": ["#bodies", r"$\log(1+g)$", "#cavities", r"$\Sigma\, b_c$"]})]
        case "phd":
            g, m = int(p["grid"]), int(p["grid"]) ** 2
            opts = {"shape": (g, g), "extent": [0, p["b_max"], 0, p["p_grid_max"]],
                    "xlabel": "birth b", "ylabel": "persistence p"}
            return [("grid", 0, m, "PD0", opts), ("grid", m, 2 * m, "PD1", opts),
                    ("grid", 2 * m, n, "Ext1", opts),
                    ("scalars", n, dim, "Loops", {"labels": ["#loops", "total pers."]})]
        case "crs_a":
            bands, M = int(p["bands"]), int(p["harmonics"])
            return [("grid", 0, dim, r"$|\hat a_q(m)| / |\hat a_q(0)|$",
                     {"shape": (bands, M), "extent": [0.5, M + 0.5, 0.5, bands + 0.5], "rows": True,
                      "xlabel": "azimuthal harmonic m", "ylabel": "radial band q"})]
        case "paf":
            k = int(p["top_families"])
            return [("scalars", 0, 7, "Global orientation",
                     {"labels": [r"$c_l$", r"$c_p$", r"$c_s$", r"$I_4$", "H", r"$N_{fam}/16$", r"$\Pi$"]}),
                    ("grid", 7, 7 + 4 * k, f"Top {k} planar families",
                     {"shape": (k, 4), "extent": [0.5, 4.5, 0.5, k + 0.5], "rows": True,
                      "xticks": [r"$\alpha_s$", r"$N_s/10$", r"$CV_s$", "angle/90"],
                      "xlabel": "", "ylabel": "family s", "annotate": True})]
        case "mts":
            return [("hist", 0, n, r"Wall thickness $\hat w = w / D_\varnothing$",
                     {"lo": p["log_lo"], "hi": p["log_hi"], "xlabel": r"$\log_{10}\hat w$"}),
                    ("scalars", n, dim, "Summary",
                     {"labels": [r"$\mu$", r"$\sigma$", r"$\gamma_1$", "Q1", "Q2", "Q3",
                                 r"$f_{thin}$", r"$f_{miss}$"]})]
    out: list[Panel] = [("hist", 0, n, "Histogram", {"lo": 0, "hi": n, "xlabel": "bin"})] if n else []
    return out + ([("scalars", n, dim, "Scalars", {})] if dim > n else [])


def draw_hist(ax: plt.Axes, h: np.ndarray, o: dict) -> None:
    """Hellinger histogram as touching bars on the physical bin range; mode marked."""
    w = (o["hi"] - o["lo"]) / len(h)
    x = o["lo"] + (np.arange(len(h)) + 0.5) * w
    ax.bar(x, h, width=w, color=SERIES, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xlim(o["lo"], o["hi"])
    ax.set_ylim(0, 1.15 * float(h.max()) if h.any() else 1.0)   # headroom for the mode label
    ax.set_ylabel(r"$\sqrt{h/\Sigma h}$")
    if h.any():
        k = int(np.argmax(h))
        ax.annotate(f"mode {x[k]:.3g}", (x[k], h[k]), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    else:
        ax.text(0.5, 0.5, "empty (no mass)", transform=ax.transAxes, ha="center", color=INK_2)


def draw_profile(ax: plt.Axes, y: np.ndarray, o: dict) -> None:
    """Spectrum / signed profile over its natural index."""
    x = o["x"]
    if o.get("line"):
        ax.plot(x, y, color=SERIES, lw=2, marker="o", ms=4)
    else:
        colors = np.where(y < 0, NEGATIVE, SERIES) if o.get("signed") else SERIES
        ax.bar(x, y, width=(x[1] - x[0]) * 0.8 if len(x) > 1 else 0.8, color=colors)
    ax.axhline(0, color=INK_2, lw=0.6)
    ax.set_ylabel("value")


def draw_grid(fig: plt.Figure, ax: plt.Axes, v: np.ndarray, o: dict) -> None:
    """2-D block as a sequential heatmap; x-major blocks are transposed to (y, x)."""
    img = v.reshape(o["shape"])
    img = img if o.get("rows") else img.T
    im = ax.imshow(img, origin="lower", aspect="auto", cmap=SEQ_CMAP, extent=o["extent"],
                   interpolation="nearest", vmin=0.0, vmax=max(float(img.max()), 1e-12))
    ax.grid(False)
    if img.any():
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=INK_2)
    else:
        ax.text(0.5, 0.5, "empty (no mass)", transform=ax.transAxes, ha="center", color=INK_2)
    if "xticks" in o:
        ax.set_xticks(np.arange(1, len(o["xticks"]) + 1), o["xticks"])
    if o.get("rows"):
        ax.set_yticks(np.arange(1, img.shape[0] + 1))
    if o.get("annotate"):
        thr = 0.6 * (img.max() or 1.0)
        for (r, c), val in np.ndenumerate(img):
            ax.text(c + 1, r + 1, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if val > thr else INK)


def draw_scalars(ax: plt.Axes, s: np.ndarray, o: dict) -> None:
    """Named scalar dims as horizontal bars with direct value labels."""
    labels = o.get("labels") or [str(i) for i in range(len(s))]
    y = np.arange(len(s))[::-1]
    ax.barh(y, s, height=0.6, color=np.where(s < 0, NEGATIVE, SERIES))
    ax.set_yticks(y, labels)
    ax.axvline(0, color=INK_2, lw=0.6)
    ax.grid(axis="x"), ax.grid(axis="y", visible=False)
    span = float(np.ptp(np.append(s, 0.0))) or 1.0
    ax.set_xlim(min(0.0, s.min()) - 0.25 * span, max(0.0, s.max()) + 0.25 * span)
    for yi, val in zip(y, s):
        ax.text(val, yi, f" {val:.4g} " , va="center", ha="left" if val >= 0 else "right",
                fontsize=8, color=INK)
    ax.set_xlabel("raw value (not z-scored)")


def plot_descriptor(d: Descriptor, v: np.ndarray) -> plt.Figure:
    """
    One figure per descriptor, one panel per sub-block of the vector.

    Every figure has the same width (FIG_SIZE_IN) so text renders at one size
    on the page; the descriptor title and metadata live in the HTML card.

    Parameters
    ----------
    d : Descriptor
        Instance that produced v (gives n_hist, params).
    v : np.ndarray, shape (d.dim,), float32
        Raw descriptor vector.
    """
    panels = layout(d)
    ratios = [0.55 if kind == "scalars" else 1.0 for kind, *_ in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=FIG_SIZE_IN, squeeze=False,
                             gridspec_kw={"width_ratios": ratios})
    draw = {"hist": draw_hist, "profile": draw_profile, "scalars": draw_scalars}
    for ax, (kind, a, b, sub, o) in zip(axes[0], panels):
        if kind == "grid":
            draw_grid(fig, ax, v[a:b], o)
        else:
            draw[kind](ax, v[a:b], o)
        ax.set_title(sub, loc="left", pad=10)
        if o.get("xlabel"):
            ax.set_xlabel(o["xlabel"])
        if o.get("ylabel"):
            ax.set_ylabel(o["ylabel"])
    fig.tight_layout(w_pad=2.5)
    return fig


def summary_table(rows: list[tuple]) -> str:
    """Fixed-width ASCII table (Windows consoles lack most math glyphs)."""
    head = ("#", "descriptor", "dim", "hist", "scal", "|v|", "min", "max", "ms", "status")
    fmt = "{:>2}  {:<10} {:>4} {:>4} {:>4}  {:>8} {:>9} {:>9} {:>7}  {}"
    lines = [fmt.format(*head), "-" * 78]
    for r in rows:
        lines.append(fmt.format(*r))
    return "\n".join(lines)


def fig_to_svg(fig: plt.Figure) -> str:
    """Render a figure to an inline SVG string in memory and free it."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]                 # drop XML prolog / DOCTYPE for inlining


def build_page(stl_name: str, mesh_info: str, rows: list[tuple], sections: list[tuple]) -> str:
    """
    One scrollable HTML page: header, jump chips, summary table, one card per descriptor.

    Parameters
    ----------
    rows : list[tuple]
        Summary-table rows (same tuples as the terminal table).
    sections : list[tuple[int, str, str, str]]
        (idx, name, meta line, inline svg) per successful descriptor.
    """
    e = html.escape
    chips = "".join(f'<a href="#d{i}">{i}&nbsp;{e(n)}</a>' for i, n, _, _ in sections)
    trs = "".join(
        "<tr" + ("" if r[-1] == "ok" else ' class="fail"') + ">"
        + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    cards = "".join(
        f'<section id="d{i}"><header><h2>{i}. {e(FULL_NAMES.get(n, n))}</h2>'
        f'<span class="inv">{e(meta)}</span></header>'
        f'<div class="fig">{svg}</div></section>' for i, n, meta, svg in sections)
    head = "".join(f"<th>{h}</th>" for h in
                   ("#", "descriptor", "dim", "hist", "scalar", "|v|", "min", "max", "ms", "status"))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Descriptors · {e(stl_name)}</title>
<style>
:root {{ --surface:{SURFACE}; --card:{SURFACE}; --ink:{INK}; --ink2:{INK_2}; --line:{GRID};
        --accent:{SERIES}; --bad:{NEGATIVE}; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--surface); color:var(--ink);
       font:14px/1.45 system-ui, "Segoe UI", sans-serif; }}
.top {{ position:sticky; top:0; z-index:2; background:var(--surface);
        border-bottom:1px solid var(--line); padding:12px 24px 10px; }}
h1 {{ margin:0; font-size:18px; }}
.meta {{ color:var(--ink2); font-size:13px; margin:2px 0 8px; }}
nav {{ display:flex; flex-wrap:wrap; gap:6px; }}
nav a {{ font-size:12px; padding:3px 9px; border:1px solid var(--line); border-radius:999px;
         color:var(--ink); text-decoration:none; background:var(--card); }}
nav a:hover {{ border-color:var(--accent); color:var(--accent); }}
main {{ max-width:1500px; margin:0 auto; padding:20px 24px 60px; }}
table {{ border-collapse:collapse; width:100%; background:var(--card); font-size:13px;
         font-variant-numeric:tabular-nums; border:1px solid var(--line); }}
th, td {{ padding:6px 10px; text-align:right; border-bottom:1px solid var(--line); }}
th:nth-child(2), td:nth-child(2), th:last-child, td:last-child {{ text-align:left; }}
th {{ color:var(--ink2); font-weight:600; }}
tr.fail td {{ color:var(--bad); }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
           margin-top:20px; padding:14px 18px; scroll-margin-top:120px; }}
section header {{ display:flex; justify-content:space-between; align-items:baseline; gap:12px; }}
h2 {{ margin:0 0 6px; font-size:16px; }}
.inv {{ color:var(--ink2); font-size:12px; }}
.fig svg {{ width:100%; height:auto; }}
</style></head><body>
<div class="top"><h1>{e(stl_name)}</h1><div class="meta">{e(mesh_info)}</div><nav>{chips}</nav></div>
<main><table><thead><tr>{head}</tr></thead><tbody>{trs}</tbody></table>{cards}</main>
</body></html>"""


def start_server(pages: dict[str, bytes]) -> tuple[http.server.ThreadingHTTPServer, str]:
    """
    Serve in-memory pages on 127.0.0.1 (random free port) from a daemon thread.

    Parameters
    ----------
    pages : dict[str, bytes]
        URL path ("/1", "/2", ...) -> UTF-8 HTML. Filled while the server runs,
        so tabs can open as each file finishes. Nothing touches the disk.

    Returns
    -------
    (server, base_url)
    """
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = pages.get(self.path)
            if body is None:
                self.send_error(404, "no report at this path")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:       # keep the terminal clean
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def stl_paths(folder_mode: bool, file: str, folder: str) -> list[Path]:
    """
    Resolve the STL list: one file, or every *.stl (any case) under a folder.

    Raises
    ------
    ValueError
        Missing folder or folder without STLs (a missing file is caught by run()).
    """
    if not folder_mode:
        return [ROOT_DIR / file]
    root = ROOT_DIR / folder
    if not root.is_dir():
        raise ValueError(f"input_folder not found: {root}")
    paths = sorted(q for q in root.rglob("*") if q.is_file() and q.suffix.lower() == ".stl")
    if not paths:
        raise ValueError(f"no .stl files in input_folder: {root}")
    return paths


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def run(stl_path: Path, targets: list[int]) -> tuple[dict[int, np.ndarray], str]:
    """
    Prepare one mesh, compute the requested descriptors, build the report page.

    Parameters
    ----------
    stl_path : Path
        STL file.
    targets : list[int]
        Ladder numbers 1..15.

    Returns
    -------
    dict[int, np.ndarray]
        idx -> float32 vector (only descriptors that succeeded).
    str
        Self-contained HTML page (inline SVG figures).

    Raises
    ------
    ValueError
        Unknown descriptor number or missing file.
    """
    bad = sorted(set(targets) - set(DESCRIPTOR_MAP))
    if bad:
        raise ValueError(f"target_descriptors must be in 1..15, got {bad}")
    if not stl_path.is_file():
        raise ValueError(f"input_file not found: {stl_path}")

    cfg = load_config(CONFIG_PATH)
    t0 = time.perf_counter()
    pm = prepare_mesh(stl_path, cfg.seed, cfg.n_samples, cfg.scale_rule)
    mesh_info = (f"{len(pm.vertices):,} vertices · {len(pm.faces):,} faces · "
                 f"D_diam = {pm.diameter:.4g} (normalised, {cfg.scale_rule}) · "
                 f"n_samples = {cfg.n_samples:,} · seed {cfg.seed} · "
                 f"prepared in {(time.perf_counter() - t0) * 1e3:.0f} ms")
    print(f"\n{stl_path.name}: {mesh_info.replace(' · ', ', ')}\n")

    apply_style()
    out: dict[int, np.ndarray] = {}
    rows: list[tuple] = []
    sections: list[tuple[int, str, str, str]] = []
    for idx in sorted(set(targets)):
        cls = DESCRIPTOR_MAP[idx]
        d = cls(cfg.descriptor_params.get(cls.name, {}), "cpu")
        t0 = time.perf_counter()
        try:
            # Same per-descriptor seed as the pipeline -> identical vectors to the cache.
            v = d.compute(pm, descriptor_rng(cfg.seed, d.name))
        except Exception as exc:
            logger.debug("descriptor %d (%s) failed", idx, d.name, exc_info=True)
            rows.append((idx, d.name, d.dim, d.n_hist, d.dim - d.n_hist, "-", "-", "-", "-",
                         f"FAILED {type(exc).__name__}: {exc}"[:60]))
            continue
        ms = (time.perf_counter() - t0) * 1e3
        rows.append((idx, d.name, d.dim, d.n_hist, d.dim - d.n_hist,
                     f"{np.linalg.norm(v):.4f}", f"{v.min():.4g}", f"{v.max():.4g}", f"{ms:.0f}", "ok"))
        out[idx] = v
        meta = (f"dim {d.dim} = {d.n_hist} hist + {d.dim - d.n_hist} scalar · {d.metric} · "
                f"{ms:.0f} ms · invariant to {', '.join(sorted(d.invariances)) or 'nothing'}")
        sections.append((idx, d.name, meta, fig_to_svg(plot_descriptor(d, v))))
        gc.collect()                    # drop per-descriptor temporaries (pair/ray arrays)

    del pm                              # free mesh + sample pools before the GUI loop
    gc.collect()
    print(summary_table(rows))
    print(f"\ntotal dims: {sum(len(v) for v in out.values())}   "
          f"ok: {len(out)}/{len(rows)}   (run with DEBUG logging for failure tracebacks)")
    return out, build_page(stl_path.name, mesh_info, rows, sections)


# ── CLI ENTRYPOINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    paths = stl_paths(folder_mode, input_file, input_folder)
    pages: dict[str, bytes] = {}
    srv, base = start_server(pages)
    skipped: list[str] = []
    for i, path in enumerate(paths, start=1):
        if folder_mode:
            print(f"\n=== [{i}/{len(paths)}] {path.name} ===")
        try:
            _, page = run(path, target_descriptors)
        except Exception:
            if not folder_mode:
                raise
            logger.exception("SKIPPED %s", path.name)   # one bad STL must not stop the batch
            skipped.append(path.name)
            continue
        pages[f"/{i}"] = page.encode("utf-8")
        webbrowser.open(f"{base}/{i}")                  # tab opens as soon as this file is done
    print(f"\n{len(pages)} page(s) served at {base}/1..{len(paths)}"
          + (f"   skipped: {', '.join(skipped)}" if skipped else "") + "   (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)                               # Ctrl+C-interruptible on Windows
    except KeyboardInterrupt:
        srv.shutdown()
        print("stopped")

# ==============================================================================
# END OF FILE: testing.py
# ==============================================================================
# SUMMARY:
#   Compute selected descriptors (1..15) for one STL or a folder; one local page per STL.
#
# USAGE EXAMPLE:
#   edit folder_mode / input_file / input_folder / target_descriptors at the top, then
#   $ python testing.py
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Scalar dims are shown raw; database z-scoring needs the fitted index.
#   - [ ] layout() mirrors each descriptor's OUTPUTS contract; update it when a
#         descriptor's block order changes (unknown names fall back to bin index).
#
# CHANGELOG:
#   2026-09-25  Initial implementation
#   2026-09-25  Per-block panels on physical axes, shared palette, summary table
#   2026-09-25  Single scrollable report page served from memory (no windows, no files)
#   2026-09-27  folder_mode: one report tab per STL in input_folder
# ==============================================================================
