"""
================================================================================
FILE: gallery.py
================================================================================
PURPOSE:
    Extract the gallery (one 781-d vector + quality stats per part) and N_AUG_DRAWS
    augmentations of every part in a process pool, and cache the result under
    <dataset>/.qrcad_cache/blr/<config_hash>/gallery.npz.

INPUTS:
    - Dataset folder (all *.stl below it), BLRConfig, worker count.

OUTPUTS:
    - Gallery(paths, X (N, 781), Q (N, 6), X_aug (N, A, 781), Q_aug (N, A, 6)).
      Parts whose extraction failed (original or any augmentation) are dropped.

DEPENDENCIES:
    stdlib      : concurrent.futures, logging, os, pathlib, tempfile, time
    third-party : numpy
    internal    : blr.augment, blr.config, blr.extract, blr.quality, src.preprocessing

PHYSICS / DOMAIN CONTEXT:
    - Original seed = cfg.seed for every part; augmentation (part i, draw a) uses
      default_rng([cfg.seed, i, a]) for the augmenter and its extraction seed, so the
      whole build is deterministic.
    - Augmented meshes go through a temporary STL (float32), the real input path.
    - Quality arrays follow quality.QUALITY_KEYS order.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-30
================================================================================
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.blr.augment import augment
from src.blr.config import BLRConfig, cache_dir
from src.blr.extract import extract_with_quality
from src.blr.quality import QUALITY_KEYS
from src.preprocessing import load_normalized

logger = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MAX_WORKERS = 6                   # [-] process pool cap (leave the machine usable)
LOG_EVERY = 100                   # [-] progress log interval (tasks)
SCALE_RULE = "rms_radius"         # [-] must match src/config.toml [pipeline].scale_rule


# ── CORE CLASSES ──────────────────────────────────────────────────────────────
@dataclass
class Gallery:
    """Extracted gallery and augmentations (see module header for shapes)."""

    paths: list[str]
    X: np.ndarray
    Q: np.ndarray
    X_aug: np.ndarray
    Q_aug: np.ndarray

    def quality_dicts(self, Q: np.ndarray) -> list[dict[str, float]]:
        return [dict(zip(QUALITY_KEYS, map(float, row))) for row in Q]


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def list_parts(dataset_dir: Path) -> list[str]:
    """Sorted absolute paths of all STL files below dataset_dir."""
    return sorted(str(p) for p in Path(dataset_dir).rglob("*") if p.suffix.lower() == ".stl")


def _qvec(q: dict[str, float]) -> np.ndarray:
    return np.array([q[k] for k in QUALITY_KEYS], dtype=np.float64)


def _low_priority() -> None:
    """Pool initialiser: run at idle priority (Windows): only spare cycles are used, foreground apps always win."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
    except Exception:                       # non-Windows: nothing to do
        pass


def _task(args: tuple[str, int, int, BLRConfig]):
    """Worker: (path, part index, draw or -1 for the original, cfg) -> (i, a, x, q) or error."""
    path, i, a, cfg = args
    try:
        if a < 0:
            x, q = extract_with_quality(Path(path), cfg.seed, blr=cfg)
        else:
            rng = np.random.default_rng([cfg.seed, i, a])
            mesh = augment(load_normalized(Path(path), SCALE_RULE), rng, cfg)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "aug.stl"
                mesh.export(p)
                x, q = extract_with_quality(p, int(rng.integers(2 ** 31)), blr=cfg)
        return i, a, x, _qvec(q), None
    except Exception as exc:                       # one bad part must not kill the build
        return i, a, None, None, f"{type(exc).__name__}: {exc}"


# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
def _extract_parts(paths: list[str], cfg: BLRConfig, workers: int | None) -> Gallery:
    """Extract originals + augmentations of `paths`; incomplete parts are dropped."""
    n, A = len(paths), cfg.n_aug_draws
    X = np.full((n, 781), np.nan, dtype=np.float32)
    Q = np.full((n, len(QUALITY_KEYS)), np.nan)
    X_aug = np.full((n, A, 781), np.nan, dtype=np.float32)
    Q_aug = np.full((n, A, len(QUALITY_KEYS)), np.nan)
    tasks = [(p, i, a, cfg) for i, p in enumerate(paths) for a in [-1, *range(A)]]
    os.environ.setdefault("OMP_NUM_THREADS", "1")      # inherited by workers: no BLAS oversubscription
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    t0, done, failed = time.perf_counter(), 0, []
    with ProcessPoolExecutor(max_workers=workers or min(os.cpu_count() or 1, MAX_WORKERS),
                            initializer=_low_priority) as ex:
        for fut in as_completed([ex.submit(_task, t) for t in tasks]):
            i, a, x, q, err = fut.result()
            done += 1
            if err:
                failed.append((paths[i], a, err))
                logger.warning("failed %s draw %d: %s", Path(paths[i]).name, a, err)
            elif a < 0:
                X[i], Q[i] = x, q
            else:
                X_aug[i, a], Q_aug[i, a] = x, q
            if done % LOG_EVERY == 0:
                logger.info("gallery: %d/%d tasks, %.0f s", done, len(tasks), time.perf_counter() - t0)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(X_aug).all(axis=(1, 2))
    logger.info("gallery: %d/%d parts complete, %d failed tasks, %.0f s",
                int(ok.sum()), n, len(failed), time.perf_counter() - t0)
    return Gallery([p for p, k in zip(paths, ok) if k], X[ok], Q[ok], X_aug[ok], Q_aug[ok])


def _save(cache: Path, paths: list[str], g: Gallery) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, paths=np.array(paths), kept=np.array(g.paths), X=g.X, Q=g.Q,
             X_aug=g.X_aug, Q_aug=g.Q_aug)


def _extend(g: Gallery, paths: list[str], missing: list[str], cfg: BLRConfig,
            workers: int | None, cache: Path) -> Gallery:
    """Extract only `missing` and merge with g, keeping the dataset listing order."""
    new = _extract_parts(missing, cfg, workers)
    keep = {p: k for k, p in enumerate(g.paths)} | {p: len(g.paths) + k for k, p in enumerate(new.paths)}
    order = [p for p in paths if p in keep]
    idx = np.array([keep[p] for p in order])
    cat = lambda a, b: np.concatenate([a, b])[idx]                      # noqa: E731
    out = Gallery(order, cat(g.X, new.X), cat(g.Q, new.Q), cat(g.X_aug, new.X_aug), cat(g.Q_aug, new.Q_aug))
    _save(cache, paths, out)
    return out


def build_gallery(dataset_dir: Path, cfg: BLRConfig, workers: int | None = None,
                  rebuild: bool = False, retry_failed: bool = False,
                  max_parts: int | None = None) -> Gallery:
    """
    Extract (or load from cache) the gallery and its augmentations.

    Parameters
    ----------
    dataset_dir : Path
    cfg : BLRConfig
    workers : int, optional
        Pool size (default min(cpu, MAX_WORKERS)).
    rebuild : bool
        Ignore the cache file.
    max_parts : int, optional
        Use a seeded random subset of this many parts (sorted), for a quick run.
    retry_failed : bool
        Re-extract parts that failed in an earlier build and merge them into the cache.

    Returns
    -------
    Gallery
    """
    paths = list_parts(dataset_dir)
    if max_parts and max_parts < len(paths):
        keep = np.sort(np.random.default_rng(cfg.seed).choice(len(paths), max_parts, replace=False))
        paths = [paths[i] for i in keep]
    cache = cache_dir(dataset_dir, cfg) / "gallery.npz"
    if cache.is_file() and not rebuild:
        z = np.load(cache, allow_pickle=False)
        if list(z["paths"]) == paths:
            g = Gallery([str(p) for p in z["kept"]], z["X"], z["Q"], z["X_aug"], z["Q_aug"])
            missing = [p for p in paths if p not in set(g.paths)]
            if not (retry_failed and missing):
                logger.info("gallery loaded from %s (%d of %d parts)", cache, len(g.paths), len(paths))
                return g
            return _extend(g, paths, missing, cfg, workers, cache)
        logger.info("gallery cache does not match the dataset; rebuilding")

    g = _extract_parts(paths, cfg, workers)
    _save(cache, paths, g)
    return g


# ==============================================================================
# END OF FILE: gallery.py
# ==============================================================================
# SUMMARY:
#   Process-pool extraction of gallery + augmentations with an npz cache.
#
# USAGE EXAMPLE:
#   g = build_gallery(Path("dataset/small_set"), BLRConfig())
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Cache is all-or-nothing: adding a part re-extracts everything (engine.add,
#         Phase 7, appends without a refit).
#
# CHANGELOG:
#   2026-09-30  Initial implementation
# ==============================================================================
