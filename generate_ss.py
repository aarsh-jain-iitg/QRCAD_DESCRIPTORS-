"""
================================================================================
FILE: generate_ss.py
================================================================================
PURPOSE:
    Batch version of testing.py for the slide deck "descriptor alignment".
    For every A/B part pair found in the testing-files folder it computes all
    15 descriptors with the exact pipeline code path (same preprocessing,
    params, seeds as testing.py), saves one clean white-background PNG per
    (STL, descriptor) plus a shaded picture of the part (part.png), and writes the numbers behind the pictures: cosine
    similarity A vs B, relative change, re-seeding noise, and a same/different
    verdict per descriptor. The deck builder reads manifest.json and the PNGs;
    nothing else is needed.

INPUTS:
    - STL files named R<NN>[alt|orig]<A|B>_<label>.stl in --stl-dir
      (default: <root>/testing files), e.g. R08A_block_sharp.stl / R08B_...
    - src/config.toml (descriptor params, seed, n_samples, scale_rule).
    - CLI: --pairs R02 R08 ...  only these pair keys (default: all)
           --stl-dir DIR        STL folder (default: testing files)
           --out-dir DIR        output folder (default: outputs/descriptor_ss)
           --no-noise           skip the re-seeded run (halves run time)
           --force              recompute even if vectors.npz already exists

OUTPUTS:
    <out-dir>/<pair>/<stl-stem>/d<NN>_<name>.png   one figure per descriptor
    <out-dir>/<pair>/<stl-stem>/part.png           shaded 3-D picture of the part
    <out-dir>/<pair>/<stl-stem>/vectors.npz        raw vectors (v<NN>, n<NN> = re-seeded)
    <out-dir>/metrics.csv                          pair x descriptor numbers
    <out-dir>/manifest.json                        pairs, image paths, metrics (deck input)

DEPENDENCIES:
    stdlib      : argparse, csv, gc, json, logging, re, time, pathlib
    third-party : numpy, matplotlib, trimesh, Pillow
    internal    : testing (plot_descriptor, layout, DESCRIPTOR_MAP), src.config,
                  src.orchestrator, src.preprocessing

PHYSICS / DOMAIN CONTEXT:
    - Descriptor vectors are dimensionless (lengths / D_diam, areas / D_diam^2).
      The first n_hist dims are sqrt(L1-normalised) histograms (Hellinger map);
      the rest are raw scalars, exactly as testing.py shows them.
    - change  = ||a - b|| / (0.5 (||a|| + ||b||))                       [-]
    - noise   = mean over A, B of ||x - x'|| / ||x||, x' = same part
                recomputed with seed + 1 (new surface samples)           [-]
    - snr     = change / max(noise, NOISE_FLOOR); verdict "different"
                if snr >= SNR_DIFFERENT, else "same". SNR_DIFFERENT is a
                presentation heuristic, not a physical constant.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-10-04
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # off-screen; figures go to PNG, not windows
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import testing as T
from src.config import load_config
from src.orchestrator import descriptor_rng
from src.preprocessing import prepare_mesh

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent                    # project root
STL_DIR = ROOT_DIR / "testing files"                          # A/B test parts
OUT_DIR = ROOT_DIR / "outputs" / "descriptor_ss"              # PNGs + metrics
CONFIG_PATH = ROOT_DIR / "src" / "config.toml"                # descriptor params

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
DESCRIPTORS: tuple[int, ...] = tuple(range(1, 16))            # ladder order, PDF
FIG_HEIGHT_IN = 3.0               # [in] every figure, so rows line up on a slide
FIG_WIDTH_IN = {1: 7.5, 2: 7.5, 3: 9.5, 4: 11.0}  # [in] by number of panels (PHD has 4)
DPI = 200                         # [px/in]
SURFACE = "#ffffff"               # white background for slides
RESEED_OFFSET = 1                 # [-] seed increment for the noise run
NOISE_FLOOR = 1e-3                # [-] relative; exact descriptors have noise = 0
SNR_DIFFERENT = 3.0               # [-] change / noise needed to call a pair "different"
VIEW_AZIM_DEG = -40.0             # [deg] camera azimuth for part.png
VIEW_ELEV_DEG = 25.0              # [deg] camera elevation for part.png
RENDER_PX = (900, 700)            # [px] part.png size (width, height)
RENDER_SUPERSAMPLE = 2            # [-] anti-aliasing factor
PART_RGB = (110, 150, 205)        # [0-255] base colour of the part
CREASE_SHADE_JUMP = 0.12          # [-] shade difference that draws a crease line (0..1 scale)
PAIR_PATTERN = re.compile(r"^(R\d{2}(?:alt|orig)?)([AB])_(.+)\.stl$", re.IGNORECASE)
SEQ_CMAP = LinearSegmentedColormap.from_list(                 # white-based sequential blue
    "seq_blue_white", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

logger = logging.getLogger(__name__)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity of two descriptor vectors.

    Parameters
    ----------
    a, b : np.ndarray, shape (dim,)
        Descriptor vectors [-].

    Returns
    -------
    float
        a.b / (|a||b|) [-]; 1.0 if either vector is all zeros (nothing to compare).
    """
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def discover_pairs(stl_dir: Path) -> dict[str, dict[str, Path]]:
    """
    Group STL files into A/B pairs by their R-number key.

    Parameters
    ----------
    stl_dir : Path
        Folder searched (non-recursively) for R<NN>[alt|orig]<A|B>_*.stl.

    Returns
    -------
    dict[str, dict[str, Path]]
        {"R08": {"A": path, "B": path}, ...}, only keys with both A and B.

    Raises
    ------
    ValueError
        If stl_dir is missing or holds no complete pair.
    """
    if not stl_dir.is_dir():
        raise ValueError(f"STL folder not found: {stl_dir}")
    found: dict[str, dict[str, Path]] = {}
    for p in sorted(stl_dir.iterdir()):
        m = PAIR_PATTERN.match(p.name)
        if m:
            found.setdefault(m.group(1), {})[m.group(2).upper()] = p
    pairs = {k: v for k, v in sorted(found.items()) if {"A", "B"} <= set(v)}
    if not pairs:
        raise ValueError(f"no complete A/B pair (R<NN>A_*.stl + R<NN>B_*.stl) in {stl_dir}")
    return pairs


def fig_size_for(n_panels: int) -> tuple[float, float]:
    """
    Figure size for a descriptor with n_panels sub-blocks.

    Parameters
    ----------
    n_panels : int
        Number of panels in testing.layout(d) (>= 1).

    Returns
    -------
    tuple[float, float]
        (width, height) [in]; width grows with panel count so labels stay readable.
    """
    return FIG_WIDTH_IN[min(max(n_panels, 1), max(FIG_WIDTH_IN))], FIG_HEIGHT_IN


def rel_change(a: np.ndarray, b: np.ndarray) -> float:
    """
    Symmetric relative L2 change between two vectors.

    Parameters
    ----------
    a, b : np.ndarray, shape (dim,)
        Descriptor vectors [-].

    Returns
    -------
    float
        ||a - b|| / (0.5 (||a|| + ||b||)) [-]; 0.0 if both are all zeros.
    """
    scale = 0.5 * (float(np.linalg.norm(a)) + float(np.linalg.norm(b)))
    return float(np.linalg.norm(a - b) / scale) if scale > 0.0 else 0.0


def render_mesh(vertices: np.ndarray, faces: np.ndarray, png_path: Path) -> None:
    """
    Shaded orthographic picture of a triangle mesh (software z-buffer, no GPU).

    Parameters
    ----------
    vertices : np.ndarray, shape (V, 3)
        Vertex coordinates [file units]; only the shape matters, the view is fitted.
    faces : np.ndarray, shape (M, 3)
        Triangle vertex indices (int), M >= 1.
    png_path : Path
        Output PNG, RENDER_PX pixels, white background.

    Raises
    ------
    ValueError
        If the mesh is empty or has zero extent in the view plane.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    if verts.size == 0 or tris.size == 0:
        raise ValueError(f"cannot render an empty mesh: {png_path}")
    azim, elev = np.radians(VIEW_AZIM_DEG), np.radians(VIEW_ELEV_DEG)
    cam = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])  # unit, scene -> camera
    right = np.cross([0.0, 0.0, 1.0], cam)
    right /= np.linalg.norm(right)
    up = np.cross(cam, right)
    xs, ys, zs = verts @ right, verts @ up, verts @ cam      # screen x, screen y, depth toward camera

    corner = verts[tris]                                     # shape (M, 3, 3)
    normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    area2 = np.linalg.norm(normal, axis=1)                   # 2 x triangle area
    normal /= np.where(area2 > 0.0, area2, 1.0)[:, None]
    light = cam + 0.6 * right + 0.4 * up
    light /= np.linalg.norm(light)
    shade = 0.30 + 0.70 * np.abs(normal @ light)             # two-sided Lambert; flipped normals still shade
    shade[area2 == 0.0] = 0.30

    width, height = RENDER_PX
    wpx, hpx = width * RENDER_SUPERSAMPLE, height * RENDER_SUPERSAMPLE
    span_x, span_y = float(np.ptp(xs)), float(np.ptp(ys))
    if span_x <= 0.0 and span_y <= 0.0:
        raise ValueError(f"mesh has zero extent in the view plane: {png_path}")
    scale = 0.88 * min(wpx / max(span_x, 1e-12), hpx / max(span_y, 1e-12))   # [px / file unit]
    px = (xs - 0.5 * (xs.max() + xs.min())) * scale + 0.5 * wpx
    py = 0.5 * hpx - (ys - 0.5 * (ys.max() + ys.min())) * scale

    zbuf = np.full((hpx, wpx), -np.inf)                      # nearest depth per pixel
    sbuf = np.full((hpx, wpx), np.nan)                       # shade of the nearest triangle
    for k in range(len(tris)):
        ia, ib, ic = tris[k]
        xa, ya, xb, yb, xc, yc = px[ia], py[ia], px[ib], py[ib], px[ic], py[ic]
        x0, x1 = max(int(np.floor(min(xa, xb, xc))), 0), min(int(np.ceil(max(xa, xb, xc))) + 1, wpx)
        y0, y1 = max(int(np.floor(min(ya, yb, yc))), 0), min(int(np.ceil(max(ya, yb, yc))) + 1, hpx)
        if x1 <= x0 or y1 <= y0:
            continue
        det = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)  # twice the signed screen area
        if abs(det) < 1e-12:
            continue                                         # edge-on triangle
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        la = ((yb - yc) * (gx - xc) + (xc - xb) * (gy - yc)) / det   # barycentric weights
        lb = ((yc - ya) * (gx - xc) + (xa - xc) * (gy - yc)) / det
        lc = 1.0 - la - lb
        inside = (la >= -1e-3) & (lb >= -1e-3) & (lc >= -1e-3)
        if not inside.any():                                 # sub-pixel triangle: colour its centroid pixel
            cx, cy = int((xa + xb + xc) / 3.0), int((ya + yb + yc) / 3.0)
            if 0 <= cx < wpx and 0 <= cy < hpx and zs[ia] > zbuf[cy, cx]:
                zbuf[cy, cx], sbuf[cy, cx] = zs[ia], shade[k]
            continue
        depth = la * zs[ia] + lb * zs[ib] + lc * zs[ic]
        zwin, swin = zbuf[y0:y1, x0:x1], sbuf[y0:y1, x0:x1]  # views into the buffers
        nearer = inside & (depth > zwin)
        zwin[nearer] = depth[nearer]
        swin[nearer] = shade[k]

    base = np.array(PART_RGB, dtype=np.float64)
    covered = ~np.isnan(sbuf)
    rgb = np.full((hpx, wpx, 3), 255.0)
    rgb[covered] = base * sbuf[covered][:, None]
    # Crease lines: darken pixels where the shade jumps between neighbouring covered pixels.
    jump = np.zeros((hpx, wpx), dtype=bool)
    dx = covered[:, 1:] & covered[:, :-1] & (np.abs(np.diff(sbuf, axis=1)) > CREASE_SHADE_JUMP)
    dy = covered[1:, :] & covered[:-1, :] & (np.abs(np.diff(sbuf, axis=0)) > CREASE_SHADE_JUMP)
    jump[:, 1:] |= dx
    jump[1:, :] |= dy
    rgb[jump] *= 0.55
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    img.resize((width, height), Image.LANCZOS).save(png_path)


def setup_plot_style() -> None:
    """Point testing.py's plotting at a white background (figure size is set per descriptor)."""
    T.SURFACE = SURFACE
    T.SEQ_CMAP = SEQ_CMAP
    T.apply_style()


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def compute_part(
    stl_path: Path,
    out_dir: Path,
    cfg,
    with_noise: bool,
    force: bool,
) -> dict[str, np.ndarray]:
    """
    Compute all descriptors for one STL, save PNGs and vectors.

    Parameters
    ----------
    stl_path : Path
        STL file (normalised internally by prepare_mesh; lengths in file units).
    out_dir : Path
        Folder for this part: part.png, d<NN>_<name>.png and vectors.npz.
    cfg : Config
        Loaded src/config.toml (seed, n_samples, scale_rule, descriptor_params).
    with_noise : bool
        Also recompute every descriptor with seed + RESEED_OFFSET.
    force : bool
        Ignore an existing vectors.npz and recompute.

    Returns
    -------
    dict[str, np.ndarray]
        Keys "v<NN>" (main run) and "n<NN>" (re-seeded run), each shape (dim,) [-].
        A descriptor that failed is simply absent.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    part_png = out_dir / "part.png"
    if force or not part_png.is_file():
        part = trimesh.load(stl_path, force="mesh")          # raw file mesh, original units
        render_mesh(part.vertices, part.faces, part_png)
        del part
    npz_path = out_dir / "vectors.npz"
    if npz_path.is_file() and not force:
        with np.load(npz_path) as z:
            cached = {k: z[k] for k in z.files}
        if all(f"v{i:02d}" in cached for i in DESCRIPTORS) and (
            not with_noise or all(f"n{i:02d}" in cached for i in DESCRIPTORS)
        ):
            logger.info("%s: reusing %s", stl_path.name, npz_path.name)
            return cached

    vecs: dict[str, np.ndarray] = {}
    runs = [("v", cfg.seed)] + ([("n", cfg.seed + RESEED_OFFSET)] if with_noise else [])
    for prefix, seed in runs:
        t0 = time.perf_counter()
        pm = prepare_mesh(stl_path, seed, cfg.n_samples, cfg.scale_rule)
        for idx in DESCRIPTORS:
            cls = T.DESCRIPTOR_MAP[idx]
            d = cls(cfg.descriptor_params.get(cls.name, {}), "cpu")
            try:
                # Same per-descriptor seed rule as the pipeline and testing.py.
                v = d.compute(pm, descriptor_rng(seed, d.name))
            except Exception:                       # logged with traceback, run continues
                logger.exception("%s: descriptor %d (%s) FAILED", stl_path.name, idx, d.name)
                continue
            vecs[f"{prefix}{idx:02d}"] = np.asarray(v, dtype=np.float64)
            if prefix == "v":
                T.FIG_SIZE_IN = fig_size_for(len(T.layout(d)))   # read by T.plot_descriptor
                fig = T.plot_descriptor(d, v)
                fig.savefig(out_dir / f"d{idx:02d}_{d.name}.png", dpi=DPI,
                            bbox_inches="tight", facecolor=SURFACE)
                plt.close(fig)
            gc.collect()
        del pm
        gc.collect()
        logger.info("%s: %s run seed %d done in %.1f s", stl_path.name,
                    "main" if prefix == "v" else "re-seeded", seed, time.perf_counter() - t0)
    np.savez_compressed(npz_path, **vecs)
    return vecs


def pair_metrics(
    vec_a: dict[str, np.ndarray],
    vec_b: dict[str, np.ndarray],
    with_noise: bool,
) -> list[dict]:
    """
    Per-descriptor numbers comparing part A with part B.

    Parameters
    ----------
    vec_a, vec_b : dict[str, np.ndarray]
        Output of compute_part for A and B.
    with_noise : bool
        Whether "n<NN>" keys are present and should set the noise level.

    Returns
    -------
    list[dict]
        One row per descriptor: idx, name, cosine [-], change [-], noise [-],
        snr [-], verdict ("same" | "different" | "failed").
    """
    rows: list[dict] = []
    for idx in DESCRIPTORS:
        key = f"{idx:02d}"
        name = T.DESCRIPTOR_MAP[idx].name
        if f"v{key}" not in vec_a or f"v{key}" not in vec_b:
            rows.append({"idx": idx, "name": name, "cosine": None, "change": None,
                         "noise": None, "snr": None, "verdict": "failed"})
            continue
        a, b = vec_a[f"v{key}"], vec_b[f"v{key}"]
        change = rel_change(a, b)
        noise = 0.0
        if with_noise and f"n{key}" in vec_a and f"n{key}" in vec_b:
            noise = 0.5 * (rel_change(a, vec_a[f"n{key}"]) + rel_change(b, vec_b[f"n{key}"]))
        snr = change / max(noise, NOISE_FLOOR)
        rows.append({
            "idx": idx, "name": name, "cosine": cosine(a, b), "change": change,
            "noise": noise, "snr": snr,
            "verdict": "different" if snr >= SNR_DIFFERENT else "same",
        })
    return rows


def run(
    stl_dir: Path,
    out_dir: Path,
    only_pairs: list[str],
    with_noise: bool,
    force: bool,
) -> dict:
    """
    Process every requested pair and write metrics.csv + manifest.json.

    Parameters
    ----------
    stl_dir : Path
        Folder with the A/B STL files.
    out_dir : Path
        Output root (created if missing).
    only_pairs : list[str]
        Pair keys to process (e.g. ["R02", "R08"]); empty = all found.
    with_noise : bool
        Run the re-seeded pass too.
    force : bool
        Recompute cached parts.

    Returns
    -------
    dict
        The manifest (also written to out_dir / "manifest.json").

    Raises
    ------
    ValueError
        If a requested pair key does not exist in stl_dir.
    """
    pairs = discover_pairs(stl_dir)
    if only_pairs:
        lookup = {k.lower(): k for k in pairs}                 # case-insensitive: r09ORIG -> R09orig
        missing = sorted(k for k in only_pairs if k.lower() not in lookup)
        if missing:
            raise ValueError(f"unknown pair key(s) {missing}; found {sorted(pairs)}")
        pairs = {lookup[k.lower()]: pairs[lookup[k.lower()]] for k in only_pairs}
    cfg = load_config(CONFIG_PATH)
    setup_plot_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    old_pairs: dict = {}
    if manifest_path.is_file():                             # keep pairs from earlier partial runs
        old_pairs = json.loads(manifest_path.read_text(encoding="utf-8")).get("pairs", {})
    manifest: dict = {"seed": cfg.seed, "n_samples": cfg.n_samples, "snr_different": SNR_DIFFERENT,
                      "noise_floor": NOISE_FLOOR, "with_noise": with_noise, "pairs": old_pairs}
    for n, (key, files) in enumerate(pairs.items(), start=1):
        logger.info("=== [%d/%d] %s: %s vs %s ===", n, len(pairs), key,
                    files["A"].name, files["B"].name)
        vecs = {}
        for side in ("A", "B"):
            stem = files[side].stem
            vecs[side] = compute_part(files[side], out_dir / key / stem, cfg, with_noise, force)
        rows = pair_metrics(vecs["A"], vecs["B"], with_noise)
        images = {}
        for idx in DESCRIPTORS:
            name = T.DESCRIPTOR_MAP[idx].name
            images[str(idx)] = {
                s: (Path(key) / files[s].stem / f"d{idx:02d}_{name}.png").as_posix() for s in ("A", "B")}
        manifest["pairs"][key] = {
            "A": files["A"].stem, "B": files["B"].stem, "images": images, "metrics": rows,
            "parts": {s: (Path(key) / files[s].stem / "part.png").as_posix() for s in ("A", "B")}}
        # Write after every pair so a crash on a big STL keeps the finished ones.
        write_outputs(out_dir, manifest)
    return manifest


def write_outputs(out_dir: Path, manifest: dict) -> None:
    """
    Write manifest.json and metrics.csv (one CSV row per pair x descriptor).

    Parameters
    ----------
    out_dir : Path
        Output root.
    manifest : dict
        JSON-serialisable manifest; manifest["pairs"][key]["metrics"] feeds the CSV.
    """
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fields = ["pair", "idx", "name", "cosine", "change", "noise", "snr", "verdict"]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key, entry in sorted(manifest["pairs"].items()):
            w.writerows({"pair": key, **r} for r in entry["metrics"])


# ── CLI ENTRYPOINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Per-descriptor PNGs + A/B metrics for the deck")
    ap.add_argument("--pairs", nargs="*", default=[], help="pair keys, e.g. R02 R08 R09orig")
    ap.add_argument("--stl-dir", type=Path, default=STL_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-noise", action="store_true", help="skip the re-seeded run")
    ap.add_argument("--force", action="store_true", help="recompute cached parts")
    args = ap.parse_args()
    result = run(args.stl_dir, args.out_dir, args.pairs, not args.no_noise, args.force)
    logger.info("done: %d pair(s) -> %s", len(result["pairs"]), args.out_dir)

# ==============================================================================
# END OF FILE: generate_ss.py
# ==============================================================================
# SUMMARY:
#   Batch-computes the 15 descriptors for every A/B pair in "testing files",
#   saves white-background PNGs per descriptor and the A-vs-B numbers (cosine,
#   change, re-seed noise, verdict) for the descriptor-alignment deck.
#
# USAGE EXAMPLE:
#   $ python generate_ss.py                       # all pairs, with noise run
#   $ python generate_ss.py --pairs R02 R08       # just these pairs
#   $ python generate_ss.py --no-noise            # faster, no noise column
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Big STLs (R03, R05B, R15, ~9 MB) dominate run time; the re-seeded
#         pass doubles it (use --no-noise for a quick look).
#   - [ ] Axis-based blocks (CRS-R, PHD, CRS-A) are arbitrary on isotropic parts
#         (cube, ball) - see CLAUDE.md.
#
# CHANGELOG:
#   2026-10-04  Initial implementation (descriptor PNGs, metrics, manifest, part.png renders)
# ==============================================================================
