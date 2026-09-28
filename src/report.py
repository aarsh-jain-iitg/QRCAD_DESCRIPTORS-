"""
================================================================================
FILE: report.py
================================================================================
PURPOSE:
    Render a QueryResult for people: a terminal table, a self-contained
    report.html (interactive three.js viewers with the STL files embedded as
    base64, plus a per-descriptor similarity heatmap), a static results.png
    grid (matplotlib Poly3DCollection, no GPU or display needed) and a
    machine-readable results.json.

INPUTS:
    - QueryResult (src/orchestrator.py), repo root, output directory.

OUTPUTS:
    - <out_dir>/report.html, results.png, results.json
    - format_table(...) -> str for the terminal.

DEPENDENCIES:
    stdlib      : base64, dataclasses, html, json, logging, pathlib
    third-party : matplotlib (Agg), numpy, trimesh
    internal    : orchestrator (types only)

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import base64
import html
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if TYPE_CHECKING:
    from src.orchestrator import QueryResult

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MAX_RENDER_FACES = 6000          # [-] faces drawn per mesh in results.png
THREE_VERSION = "0.160.0"

logger = logging.getLogger(__name__)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _heat_color(t: float) -> str:
    """Map t in [0, 1] to a white -> teal background colour."""
    t = float(np.clip(t, 0.0, 1.0))
    r, g, b = (int(255 + (c - 255) * t) for c in (13, 148, 136))
    return f"rgb({r},{g},{b})"


def _heatmap_html(result: QueryResult) -> str:
    """Hits x descriptors table, cell colour = similarity scaled per column."""
    names = result.descriptors
    sims = np.array([[h.per_descriptor[n]["similarity"] for n in names] for h in result.hits])
    lo, hi = sims.min(axis=0), sims.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    body = []
    for i, h in enumerate(result.hits):
        cells = "".join(
            f'<td style="background:{_heat_color((sims[i, j] - lo[j]) / span[j])}" '
            f'title="rank {h.per_descriptor[n]["rank"]}">{sims[i, j]:.3f}'
            f'<small>#{h.per_descriptor[n]["rank"]}</small></td>'
            for j, n in enumerate(names))
        body.append(f"<tr><th>{h.rank}. {html.escape(h.id)}</th>{cells}</tr>")
    return f"<table class='heat'><tr><th>hit</th>{head}</tr>{''.join(body)}</table>"


def _load_for_render(path: Path) -> trimesh.Trimesh | None:
    try:
        m = trimesh.load(path, force="mesh")
        return m if len(m.faces) else None
    except Exception:  # a broken file only loses its thumbnail
        logger.warning("cannot render %s", path, exc_info=True)
        return None


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def format_table(result: QueryResult) -> str:
    """Plain-text results table: rank, file, class, fused score, top-3 descriptors."""
    rows = [("rank", "file", "class", "score", "top descriptors")]
    for h in result.hits:
        rows.append((str(h.rank), h.id + (" [exact match]" if h.exact_match else ""), h.label,
                     f"{h.score:.4f}", ", ".join(h.top_descriptors)))
    w = [max(len(r[i]) for r in rows) for i in range(5)]
    line = "  ".join("-" * x for x in w)
    fmt = lambda r: "  ".join(c.ljust(x) for c, x in zip(r, w))
    head = (f"query: {result.query_path}   fusion: {result.fusion}   "
            f"extract {result.timing_ms.get('extract', 0):.0f} ms, "
            f"search {result.timing_ms.get('search', 0):.1f} ms")
    if result.query_in_database:
        head += "\nnote: the query file itself is in the database (exact match flagged)"
    return "\n".join([head, fmt(rows[0]), line] + [fmt(r) for r in rows[1:]])


def write_png(result: QueryResult, root: Path, out: Path) -> None:
    """Static grid of the query and top-k meshes (results.png)."""
    items = [("QUERY", Path(result.query_path))] + [
        (f"#{h.rank} {h.label}  {h.score:.3f}", root / h.path) for h in result.hits]
    cols = min(4, len(items))
    rows = int(np.ceil(len(items) / cols))
    fig = plt.figure(figsize=(3.2 * cols, 3.2 * rows))
    for i, (title, p) in enumerate(items):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)
        m = _load_for_render(p)
        if m is None:
            continue
        faces = m.faces[:: max(1, len(m.faces) // MAX_RENDER_FACES)]
        V = m.vertices - m.bounds.mean(axis=0)
        tri = V[faces]
        light = np.abs(m.face_normals[:: max(1, len(m.faces) // MAX_RENDER_FACES)] @ [0.4, 0.3, 0.87])
        colors = plt.cm.Greys(0.35 + 0.5 * light)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=colors, edgecolor="none"))
        r = float(np.abs(V).max()) or 1.0
        ax.set_xlim(-r, r), ax.set_ylim(-r, r), ax.set_zlim(-r, r)
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def write_html(result: QueryResult, root: Path, out: Path) -> None:
    """Self-contained report.html with three.js viewers and the heatmap."""
    models = [{"title": "QUERY: " + Path(result.query_path).name, "path": Path(result.query_path)}]
    models += [{"title": f"#{h.rank} {h.id} ({h.score:.3f})"
                         + (" exact match" if h.exact_match else ""),
                "path": root / h.path} for h in result.hits]
    payload = []
    for m in models:
        try:
            b64 = base64.b64encode(Path(m["path"]).read_bytes()).decode("ascii")
        except OSError:
            b64 = ""
        payload.append({"title": m["title"], "stl": b64})
    cards = "".join(f'<div class="card"><div class="v" id="v{i}"></div>'
                    f'<p>{html.escape(m["title"])}</p></div>' for i, m in enumerate(payload))
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>QRCAD Query Report</title>
<style>
:root{{--bg:#f7f7f5;--fg:#1c1c1c;--card:#fff;--line:#ddd}}
@media (prefers-color-scheme: dark){{:root{{--bg:#16181a;--fg:#e8e8e8;--card:#212427;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px}}
.v{{height:220px}} .card p{{margin:4px;font-size:12px;word-break:break-all}}
.heat{{border-collapse:collapse;font-size:11px;margin-top:16px;color:#111}}
.heat th,.heat td{{border:1px solid var(--line);padding:3px 5px;text-align:right}}
.heat th{{background:var(--card);color:var(--fg);text-align:left}}
.heat small{{opacity:.6;margin-left:3px}} .wrap{{overflow-x:auto}}
</style></head><body>
<h2>QRCAD query: {html.escape(Path(result.query_path).name)}</h2>
<p>fusion: {html.escape(result.fusion)} &middot; drag to rotate, scroll to zoom</p>
<div class="grid">{cards}</div>
<h3>Per-descriptor similarity (cell: cosine of the block, #rank within that descriptor)</h3>
<div class="wrap">{_heatmap_html(result)}</div>
<script type="importmap">{{"imports":{{
 "three":"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js",
 "three/addons/":"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from 'three';
import {{STLLoader}} from 'three/addons/loaders/STLLoader.js';
import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';
const models = {json.dumps(payload)};
const loader = new STLLoader();
models.forEach((m, i) => {{
  const el = document.getElementById('v' + i);
  if (!m.stl) {{ el.textContent = 'file missing'; return; }}
  const bin = Uint8Array.from(atob(m.stl), c => c.charCodeAt(0));
  const geo = loader.parse(bin.buffer);
  geo.computeBoundingSphere(); geo.center();
  const r = geo.boundingSphere.radius || 1;
  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));
  const d = new THREE.DirectionalLight(0xffffff, 1.5); d.position.set(1, 2, 3); scene.add(d);
  scene.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({{color: i ? 0x8aa4b8 : 0xd08a3c, metalness: .2, roughness: .6}})));
  const cam = new THREE.PerspectiveCamera(40, el.clientWidth / el.clientHeight, r / 100, r * 100);
  cam.position.set(r * 1.8, r * 1.4, r * 2.2);
  const ren = new THREE.WebGLRenderer({{antialias: true, alpha: true}});
  ren.setSize(el.clientWidth, el.clientHeight); el.appendChild(ren.domElement);
  const ctl = new OrbitControls(cam, ren.domElement);
  const draw = () => ren.render(scene, cam);
  ctl.addEventListener('change', draw); draw();
}});
</script></body></html>"""
    out.write_text(doc, encoding="utf-8")


def write_report(result: QueryResult, root: Path, out_dir: Path) -> Path:
    """
    Write report.html, results.png and results.json into out_dir.

    Parameters
    ----------
    result : QueryResult
    root : Path
        Repo root (hit paths are stored relative to it).
    out_dir : Path
        Target folder (created).

    Returns
    -------
    Path
        out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(asdict(result), indent=1), encoding="utf-8")
    write_html(result, root, out_dir / "report.html")
    write_png(result, root, out_dir / "results.png")
    return out_dir

# ==============================================================================
# END OF FILE: report.py
# ==============================================================================
# SUMMARY:
#   Terminal table + report.html (three.js, heatmap) + results.png + results.json.
#
# USAGE EXAMPLE:
#   from src.report import format_table, write_report
#   print(format_table(res)); write_report(res, ROOT_DIR, Path("outputs/q_20260923"))
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] three.js loads from the jsDelivr CDN; the viewers need internet
#         (the STL data itself is embedded).
#   - [ ] Very large STLs are embedded whole; the HTML grows with them.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
