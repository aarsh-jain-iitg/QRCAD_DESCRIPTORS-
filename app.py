"""
================================================================================
FILE: app.py
================================================================================
PURPOSE:
    Local web app around the QRCAD pipeline: add datasets (folders / STL files /
    zip), build the descriptor cache in the background, search one STL and see the
    top-k similar parts with a per-descriptor explanation of why they match or differ.

INPUTS:
    - src/config.toml, dataset/<name>/<class>/*.stl, uploaded query STL.

OUTPUTS:
    - Web UI on http://127.0.0.1:5000 (static/index.html), JSON API under /api.
    - dataset/<name>/ (uploads), cache/ (vectors + index), cache/_query/ (last query).

DEPENDENCIES:
    stdlib      : collections, dataclasses, logging, pathlib, re, shutil, threading, webbrowser, zipfile
    third-party : flask
    internal    : src.config, src.orchestrator, src.sample_dataset

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import webbrowser
import zipfile
from collections import deque
from dataclasses import asdict
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

from src.config import DEFAULT_CONFIG, load_config
from src.orchestrator import STATE_DIR, build, query
from src.sample_dataset import generate_sample_dataset

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
STATIC_DIR = ROOT_DIR / "static"

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
HOST, PORT = "127.0.0.1", 5000
LOG_LINES = 12
MAX_UPLOAD_MB = 2048
SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]")

logger = logging.getLogger("app")
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
JOB: dict = {"state": "idle", "message": "", "report": None}   # ponytail: one global job, add a queue if multi-user
LOG: deque[str] = deque(maxlen=LOG_LINES)
LOCK = threading.Lock()


class _Capture(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        LOG.append(record.getMessage())


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def cfg():
    return load_config(DEFAULT_CONFIG, ROOT_DIR)


def clean(part: str) -> str:
    return SAFE_NAME.sub("_", part).strip(" .") or "_"


def safe_rel(name: str) -> Path | None:
    """Sanitised relative path of an uploaded STL (class folder kept), else None."""
    parts = [clean(p) for p in name.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    if not parts or not parts[-1].lower().endswith(".stl"):
        return None
    return Path(*parts) if len(parts) > 1 else Path("unlabeled") / parts[0]


def dataset_dir(name: str) -> Path:
    d = DATASET_DIR / clean(name)
    if d.parent != DATASET_DIR:
        abort(400)
    return d


def run_build(rebuild: bool) -> None:
    try:
        report = build(cfg(), rebuild=rebuild)
        JOB.update(state="idle", report={**asdict(report)}, message="")
    except Exception as e:                      # surfaced in the UI, not swallowed
        logger.exception("build failed")
        JOB.update(state="error", message=str(e))


def start_build(rebuild: bool = False) -> bool:
    with LOCK:
        if JOB["state"] == "building":
            return False
        JOB.update(state="building", message="", report=None)
        LOG.clear()
    threading.Thread(target=run_build, args=(rebuild,), daemon=True).start()
    return True


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/datasets")
def datasets():
    out = []
    for d in sorted(p for p in DATASET_DIR.glob("*") if p.is_dir()):
        files = [f for f in d.rglob("*") if f.suffix.lower() == ".stl"]
        out.append({"name": d.name, "files": len(files),
                    "classes": sorted({f.parent.name for f in files})})
    return jsonify(datasets=out, indexed=(cfg().cache_dir / STATE_DIR / "meta.json").is_file())


@app.post("/api/datasets")
def add_dataset():
    """multipart: name, files[] (filename = relative path incl. class folder) or a .zip."""
    root = dataset_dir(request.form.get("name", "").strip() or "uploads")
    n = 0
    for f in request.files.getlist("files"):
        if f.filename.lower().endswith(".zip"):
            with zipfile.ZipFile(f.stream) as z:
                for info in z.infolist():
                    rel = safe_rel(info.filename)
                    if rel and not info.is_dir():
                        dst = root / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(z.read(info))
                        n += 1
        elif (rel := safe_rel(f.filename)) is not None:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            f.save(dst)
            n += 1
    if n == 0:
        return jsonify(error="no .stl files found in the upload"), 400
    return jsonify(added=n, building=start_build())


@app.delete("/api/datasets/<name>")
def delete_dataset(name: str):
    d = dataset_dir(name)
    if not d.is_dir():
        abort(404)
    shutil.rmtree(d)
    shutil.rmtree(cfg().cache_dir / d.name, ignore_errors=True)
    if any(p.is_dir() for p in DATASET_DIR.glob("*")):
        start_build()
    else:
        shutil.rmtree(cfg().cache_dir / STATE_DIR, ignore_errors=True)
    return jsonify(ok=True)


@app.post("/api/sample")
def sample():
    generate_sample_dataset(DATASET_DIR / "_sample")
    return jsonify(building=start_build())


@app.post("/api/build")
def rebuild():
    return jsonify(building=start_build((request.get_json(silent=True) or {}).get("rebuild", False)))


@app.get("/api/status")
def status():
    return jsonify(**JOB, log=list(LOG))


@app.post("/api/query")
def search():
    if JOB["state"] == "building":
        return jsonify(error="cache is still building, try again when it finishes"), 409
    f = request.files.get("file")
    if f is None or not f.filename.lower().endswith(".stl"):
        return jsonify(error="upload one .stl file"), 400
    c = cfg()
    qdir = c.cache_dir / "_query"
    qdir.mkdir(parents=True, exist_ok=True)
    qpath = qdir / "query.stl"
    f.save(qpath)
    try:
        res = query(qpath, c, int(request.form.get("top_k", 10)))
    except FileNotFoundError:
        return jsonify(error="no index yet: add a dataset first"), 409
    except Exception as e:
        logger.exception("query failed")
        return jsonify(error=str(e)), 500
    out = asdict(res)
    out["query_name"] = f.filename
    return jsonify(out)


@app.get("/api/query_stl")
def query_stl():
    return send_from_directory(cfg().cache_dir / "_query", "query.stl")


@app.get("/api/stl")
def stl():
    """Serve a dataset STL; the path must resolve inside dataset/."""
    p = (ROOT_DIR / request.args.get("path", "")).resolve()
    if DATASET_DIR.resolve() not in p.parents or p.suffix.lower() != ".stl":
        abort(404)
    return send_from_directory(p.parent, p.name)


# ── ENTRYPOINT ────────────────────────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("src").addHandler(_Capture())
    DATASET_DIR.mkdir(exist_ok=True)
    if any(p.is_dir() for p in DATASET_DIR.glob("*")) and not (cfg().cache_dir / STATE_DIR).is_dir():
        start_build()
    threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    app.run(HOST, PORT, threaded=True)


if __name__ == "__main__":
    main()

# ==============================================================================
# END OF FILE: app.py
# ==============================================================================
# SUMMARY:
#   Flask server: dataset upload -> background build -> single-STL query with
#   per-descriptor explanation. UI is static/index.html.
#
# USAGE EXAMPLE:
#   $ python app.py
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Single user, one build at a time; uploads of huge datasets go through
#         the browser (drop a folder into dataset/ and press Rebuild instead).
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
