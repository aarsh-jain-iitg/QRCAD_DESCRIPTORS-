"""
================================================================================
FILE: cache.py
================================================================================
PURPOSE:
    Per-dataset, per-file, per-descriptor vector cache. One vectors.npz plus
    one manifest.json per dataset under cache_dir/<dataset>/. Vectors are
    keyed by content, so an unchanged file is never recomputed, a changed
    file or a bumped descriptor version / param invalidates exactly its own
    entries, and entries no current file refers to are pruned on save.

INPUTS:
    - cache_dir / dataset name; file bytes; Descriptor instances;
      Config.fingerprint() (seed, n_samples, scale rule).

OUTPUTS:
    - cache_dir/<dataset>/vectors.npz  : arrays "d<dim>__keys", "d<dim>__vecs"
    - cache_dir/<dataset>/manifest.json: files (sha, size, mtime), failures,
      per-key extraction time [ms].

DEPENDENCIES:
    stdlib      : hashlib, json, logging, os, pathlib
    third-party : numpy
    internal    : descriptors.base (type only)

PHYSICS / DOMAIN CONTEXT:
    - Cache key = "<sha256(file bytes)>:<name>:<version>:<param hash>", with
      the param hash = sha256 of the descriptor params and the pipeline
      fingerprint, first 16 hex characters.
    - Writes go to a temp file, then os.replace (atomic on the same volume).

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MANIFEST = "manifest.json"
VECTORS = "vectors.npz"
FORMAT_VERSION = 1
READ_CHUNK = 1 << 20          # [bytes] file hashing block size

logger = logging.getLogger(__name__)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to path via a temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def cache_key(file_sha: str, desc: Any, fingerprint: dict[str, Any]) -> str:
    """Content-addressed key for one (file, descriptor, params, pipeline) vector."""
    payload = json.dumps({"params": desc.cache_params(), "pipeline": fingerprint},
                         sort_keys=True, default=str)
    ph = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{file_sha}:{desc.name}:{desc.version}:{ph}"


def file_sha256(path: Path) -> str:
    """sha256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(READ_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
class DatasetCache:
    """
    In-memory view of one dataset's cache directory.

    Attributes
    ----------
    vectors : dict[str, np.ndarray]
        cache key -> float32 vector.
    failures : dict[str, str]
        cache key (or "<sha>:preprocess:<fp>") -> error message.
    timing_ms : dict[str, float]
        cache key -> extraction time [ms].
    files : dict[str, dict]
        relative path -> {"sha256", "size", "mtime_ns"}.
    """

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.vectors: dict[str, np.ndarray] = {}
        self.failures: dict[str, str] = {}
        self.timing_ms: dict[str, float] = {}
        self.files: dict[str, dict[str, Any]] = {}

    @classmethod
    def load(cls, directory: Path) -> DatasetCache:
        """Load a cache directory; a missing or unreadable cache loads empty."""
        c = cls(directory)
        man, vec = c.dir / MANIFEST, c.dir / VECTORS
        if not man.exists():
            return c
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
            if m.get("format") != FORMAT_VERSION:
                logger.warning("cache %s has another format; ignoring it", c.dir)
                return c
            c.files, c.failures, c.timing_ms = m["files"], m["failures"], m["timing_ms"]
            if vec.exists():
                with np.load(vec) as z:
                    for name in z.files:
                        if name.endswith("__keys"):
                            stem = name[: -len("__keys")]
                            for k, v in zip(z[name], z[stem + "__vecs"]):
                                c.vectors[str(k)] = v
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("cache %s unreadable (%s); starting empty", c.dir, exc)
            return cls(directory)
        return c

    def file_sha(self, rel: str, path: Path) -> str:
        """sha256 of a file, reused from the manifest when size and mtime match."""
        st = path.stat()
        rec = self.files.get(rel)
        if rec and rec["size"] == st.st_size and rec["mtime_ns"] == st.st_mtime_ns:
            return rec["sha256"]
        sha = file_sha256(path)
        self.files[rel] = {"sha256": sha, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        return sha

    def prune(self, live_files: set[str], live_keys: set[str]) -> int:
        """Drop records of deleted files and entries no live key refers to."""
        self.files = {k: v for k, v in self.files.items() if k in live_files}
        dead = [k for k in self.vectors if k not in live_keys]
        for k in dead:
            del self.vectors[k]
        self.failures = {k: v for k, v in self.failures.items() if k in live_keys}
        self.timing_ms = {k: v for k, v in self.timing_ms.items() if k in live_keys}
        return len(dead)

    def save(self) -> None:
        """Atomically write vectors.npz and manifest.json."""
        self.dir.mkdir(parents=True, exist_ok=True)
        groups: dict[int, list[str]] = {}
        for k, v in self.vectors.items():
            groups.setdefault(len(v), []).append(k)
        arrays: dict[str, np.ndarray] = {}
        for dim, keys in groups.items():
            keys.sort()
            arrays[f"d{dim}__keys"] = np.array(keys)
            arrays[f"d{dim}__vecs"] = np.stack([self.vectors[k] for k in keys]).astype(np.float32)
        tmp = self.dir / (VECTORS + ".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, self.dir / VECTORS)
        manifest = {"format": FORMAT_VERSION, "files": self.files,
                    "failures": self.failures, "timing_ms": self.timing_ms}
        atomic_write_bytes(self.dir / MANIFEST, json.dumps(manifest, indent=1).encode())

# ==============================================================================
# END OF FILE: cache.py
# ==============================================================================
# SUMMARY:
#   Content-addressed, incremental, atomically-written descriptor vector cache.
#
# USAGE EXAMPLE:
#   from src.cache import DatasetCache, cache_key
#   c = DatasetCache.load(cfg.cache_dir / "_sample")
#   key = cache_key(c.file_sha(rel, path), desc, cfg.fingerprint())
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Whole npz is rewritten on save; fine up to ~1e5 vectors per dataset.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
