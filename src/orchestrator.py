"""
================================================================================
FILE: orchestrator.py
================================================================================
PURPOSE:
    The pipeline's control flow. build() discovers STL files, extracts every
    enabled descriptor once per file (process pool, per-file timeout,
    incremental cache), fits per-descriptor database statistics and the
    compressor, and persists the search index. query() runs the same
    extraction on one STL and returns a fused ranking with per-descriptor
    similarity and rank for every hit, so results are explainable.

INPUTS:
    - Config (src/config.py); STL files under dataset_root/<dataset>/<class>/.
    - query(): path to a query STL.

OUTPUTS:
    - cache_dir/<dataset>/{vectors.npz, manifest.json} (see cache.py)
    - cache_dir/_index/{meta.json, state.npz, compressor.npz, index_<backend>.bin}
    - BuildReport, QueryResult dataclasses.

DEPENDENCIES:
    stdlib      : collections, concurrent.futures, dataclasses, hashlib, json,
                  logging, math, os, pathlib, time, zlib
    third-party : numpy, tqdm, torch (optional, device detection only)
    internal    : cache, compression, config, database, descriptors,
                  preprocessing

PHYSICS / DOMAIN CONTEXT:
    - Block transform (PDF eq. 31): histogram dims keep their Hellinger values
      (inner product = Bhattacharyya similarity); scalar dims are z-scored
      with database mean / std; each block is then scaled to unit length.
    - weighted_zscore: x = concat(sqrt(w_b) x^_b), L2-normalised, so
      <x, x'> = sum_b w_b <x^_b, x^'_b> / sum_b w_b (PDF eq. 31); one index search.
    - rank_fusion: reciprocal rank fusion sum_b w_b / (60 + rank_b) over the
      exact per-descriptor rankings.
    - Each descriptor gets its own rng seeded by (seed, crc32(name)), so a
      vector does not depend on which other descriptors are enabled.

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import zlib
from collections import deque
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.cache import DatasetCache, atomic_write_bytes, cache_key, file_sha256
from src.compression import Compressor, l2_rows, make_compressor
from src.config import Config
from src.database import VectorStore, make_store
from src.descriptors import Descriptor, DescriptorError, make_descriptors
from src.preprocessing import prepare_mesh

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
STATE_DIR = "_index"            # subfolder of cache_dir holding the fitted index
RRF_K = 60                      # [-] reciprocal-rank-fusion constant
HARD_TIMEOUT_FACTOR = 2.0       # [-] pool kills a task after factor * timeout_s + grace
HARD_TIMEOUT_GRACE_S = 20.0     # [s] allowance for worker start-up (imports)
STD_FLOOR = 1e-9                # [-] z-score std floor

logger = logging.getLogger(__name__)


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def descriptor_rng(seed: int, name: str) -> np.random.Generator:
    """Independent, reproducible generator per (seed, descriptor)."""
    return np.random.default_rng([int(seed), zlib.crc32(name.encode())])


def extract_file(path: str, names: list[str], params: dict[str, dict], device: str,
                 seed: int, n_samples: int, scale_rule: str, timeout_s: float) -> dict[str, Any]:
    """
    Prepare one mesh and compute the named descriptors (process-pool worker).

    Returns
    -------
    dict with keys
        "ok"         : {name: (float32 vector, ms)}
        "err"        : {name: message} for DescriptorError / unexpected errors
        "file_error" : message if the mesh could not be prepared or timed out
        "timeout"    : True if the cooperative deadline was hit
    """
    out: dict[str, Any] = {"ok": {}, "err": {}, "file_error": None, "timeout": False}
    t_start = time.perf_counter()
    try:
        pm = prepare_mesh(Path(path), seed, n_samples, scale_rule)
    except DescriptorError as exc:
        out["file_error"] = f"unusable mesh: {exc}"
        return out
    except Exception as exc:  # corrupt STL: trimesh raises many types; never kill the build
        logger.warning("could not load %s", path, exc_info=True)
        out["file_error"] = f"{type(exc).__name__}: {exc}"
        return out
    for d in make_descriptors(names, params, device):
        if time.perf_counter() - t_start > timeout_s:
            out["file_error"], out["timeout"] = f"timeout after {timeout_s:.0f} s", True
            return out
        t0 = time.perf_counter()
        try:
            v = d.compute(pm, descriptor_rng(seed, d.name))
            out["ok"][d.name] = (v, (time.perf_counter() - t0) * 1e3)
        except DescriptorError as exc:
            out["err"][d.name] = str(exc)
        except Exception as exc:  # one descriptor bug must not kill the build
            logger.warning("%s on %s failed", d.name, path, exc_info=True)
            out["err"][d.name] = f"{type(exc).__name__}: {exc}"
    return out


def loo_metrics(S: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    Leave-one-out P@1, P@5, P@10 and mAP from a similarity matrix.

    Parameters
    ----------
    S : np.ndarray, shape (N, N)
        Similarities (higher = closer); the diagonal is ignored.
    labels : np.ndarray, shape (N,)
        Class labels.
    """
    S = np.array(S, dtype=np.float64)
    np.fill_diagonal(S, -np.inf)
    N = len(S)
    order = np.argsort(-S, axis=1, kind="stable")[:, : N - 1]
    rel = labels[order] == labels[:, None]
    out = {f"P@{k}": float(rel[:, :k].mean()) for k in (1, 5, 10) if k <= N - 1}
    n_rel = rel.sum(axis=1)
    hits = np.cumsum(rel, axis=1)
    prec = hits / np.arange(1, N)[None, :]
    ap = (prec * rel).sum(axis=1) / np.maximum(n_rel, 1)
    out["mAP"] = float(ap[n_rel > 0].mean()) if np.any(n_rel > 0) else 0.0
    return out


def rank_matrix(S: np.ndarray) -> np.ndarray:
    """1-based rank of each column within each row of S (Q, N); highest similarity = 1."""
    order = np.argsort(-S, axis=1, kind="stable")
    R = np.empty_like(order)
    np.put_along_axis(R, order, np.broadcast_to(np.arange(1, S.shape[1] + 1), order.shape), axis=1)
    return R


def repo_relative(p: str | Path, root: Path) -> str:
    """Path relative to the repo root when inside it (posix), else as given."""
    rp, rr = Path(p).resolve(), root.resolve()
    return rp.relative_to(rr).as_posix() if rp.is_relative_to(rr) else str(p)


def resolve_device(device: str) -> str:
    """
    Map "auto" / "cuda" / "cpu" to the device used for extraction.

    "auto" resolves to "cpu": at the PDF defaults the only GPU path (D4)
    measured 86 ms on an RTX 4060 vs 77 ms on CPU, plus ~5 s of CUDA
    start-up in every worker process, so no descriptor gains from the GPU.
    "cuda" forces the GPU path (bit-identical vectors, tested).
    """
    if device != "cuda":
        return "cpu"
    try:
        import torch
        ok = bool(torch.cuda.is_available())
    except ImportError:
        ok = False
    if device == "cuda" and not ok:
        logger.warning("device = cuda requested but CUDA is unavailable; using cpu")
    return "cuda" if ok else "cpu"


def _signature(cfg: Config, descs: list[Descriptor]) -> str:
    """Hash of everything that must match between build and query."""
    payload = {"fp": cfg.fingerprint(), "descs": [(d.name, d.version, d.cache_params()) for d in descs],
               "weights": {d.name: cfg.weight(d.name) for d in descs},
               "compression": [cfg.compression, cfg.target_dim], "backend": cfg.backend,
               "metric": cfg.metric}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# ── CORE CLASSES / MAIN FUNCTIONS ─────────────────────────────────────────────
@dataclass
class BuildReport:
    """Summary of one build() call."""

    datasets: list[str]
    n_files: int
    n_indexed: int
    n_files_extracted: int
    n_vectors_computed: int
    failures: list[tuple[str, str]]
    seconds: float
    device: str


@dataclass
class Hit:
    """One ranked database item with its per-descriptor explanation."""

    rank: int
    id: str
    label: str
    path: str
    score: float
    exact_match: bool
    per_descriptor: dict[str, dict[str, float]]
    top_descriptors: list[str]


@dataclass
class QueryResult:
    """Fused ranking + per-descriptor detail for one query."""

    query_path: str
    query_sha256: str
    query_in_database: bool
    fusion: str
    descriptors: list[str]
    hits: list[Hit]
    timing_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class IndexState:
    """Everything query() / evaluate() need, persisted under cache_dir/_index."""

    names: list[str]
    dims: list[int]
    n_hist: list[int]
    weights: list[float]
    ids: list[str]
    labels: list[str]
    shas: list[str]
    paths: list[str]
    raw: dict[str, np.ndarray]
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    signature: str
    timing_ms: dict[str, float]

    def block(self, name: str, X: np.ndarray) -> np.ndarray:
        """PDF eq. 31 block transform: z-score scalar dims, unit-length rows."""
        h = self.n_hist[self.names.index(name)]
        Z = np.array(X, dtype=np.float64, ndmin=2)
        Z[:, h:] = (Z[:, h:] - self.mean[name]) / self.std[name]
        return l2_rows(Z)

    def fuse(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        """PDF eq.(31): concat sqrt(w_b) x^_b, then L2-normalise."""
        parts = [math.sqrt(w) * blocks[n] for n, w in zip(self.names, self.weights)]
        return l2_rows(np.concatenate(parts, axis=1))

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for n in self.names:
            arrays[f"raw__{n}"], arrays[f"mean__{n}"], arrays[f"std__{n}"] = \
                self.raw[n], self.mean[n], self.std[n]
        tmp = d / "state.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, d / "state.npz")
        meta = {k: getattr(self, k) for k in ("names", "dims", "n_hist", "weights", "ids",
                                              "labels", "shas", "paths", "signature", "timing_ms")}
        atomic_write_bytes(d / "meta.json", json.dumps(meta, indent=1).encode())

    @classmethod
    def load(cls, d: Path) -> IndexState:
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"no index at {d}; run build first")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with np.load(d / "state.npz") as z:
            raw = {n: z[f"raw__{n}"] for n in meta["names"]}
            mean = {n: z[f"mean__{n}"] for n in meta["names"]}
            std = {n: z[f"std__{n}"] for n in meta["names"]}
        return cls(raw=raw, mean=mean, std=std, **meta)


def _run_jobs(jobs: list[dict[str, Any]], cfg: Config, device: str) -> Iterator[tuple[dict, dict]]:
    """
    Run extraction jobs; yield (job, result) as they finish.

    Parallel path: at most `workers` tasks in flight, so submit time ~ start
    time. A task running longer than HARD_TIMEOUT_FACTOR * timeout_s + grace
    is treated as hung: the pool is torn down and rebuilt, the hung file is
    reported as a timeout, and the other in-flight files are resubmitted.
    Any pool failure falls back to serial processing of what is left.
    """
    common = (cfg.descriptor_params, device, cfg.seed, cfg.n_samples, cfg.scale_rule, cfg.timeout_s)
    workers = min(cfg.n_workers(), len(jobs))
    pending = deque(jobs)
    if workers > 1:
        hard = HARD_TIMEOUT_FACTOR * cfg.timeout_s + HARD_TIMEOUT_GRACE_S
        ex = ProcessPoolExecutor(max_workers=workers)
        running: dict[Any, tuple[dict, float]] = {}
        try:
            while pending or running:
                while pending and len(running) < workers:
                    job = pending.popleft()
                    fut = ex.submit(extract_file, job["path"], job["names"], *common)
                    running[fut] = (job, time.monotonic())
                done, _ = wait(list(running), timeout=0.5, return_when=FIRST_COMPLETED)
                for fut in done:
                    job, _ = running.pop(fut)
                    yield job, fut.result()
                now = time.monotonic()
                hung = [f for f, (_, t0) in running.items() if now - t0 > hard]
                if hung:
                    for f in hung:
                        job, _ = running.pop(f)
                        yield job, {"ok": {}, "err": {}, "timeout": True,
                                    "file_error": f"hard timeout after {hard:.0f} s"}
                    pending.extendleft(job for job, _ in running.values())
                    running.clear()
                    for p in list(getattr(ex, "_processes", {}).values()):   # private: only way to kill a hung task
                        p.terminate()
                    ex.shutdown(wait=False, cancel_futures=True)
                    ex = ProcessPoolExecutor(max_workers=workers)
        except (BrokenProcessPool, OSError) as exc:
            logger.warning("process pool failed (%s); continuing serially", exc)
            pending.extendleft(job for job, _ in running.values())
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    while pending:
        job = pending.popleft()
        yield job, extract_file(job["path"], job["names"], *common)


def build(cfg: Config, rebuild: bool = False) -> BuildReport:
    """
    Build or incrementally refresh the cache and index for all configured datasets.

    Parameters
    ----------
    cfg : Config
    rebuild : bool
        Ignore existing caches (recompute everything).

    Returns
    -------
    BuildReport
    """
    t_start = time.perf_counter()
    device = resolve_device(cfg.device)
    descs = make_descriptors(list(cfg.enabled), cfg.descriptor_params, device)
    fp = cfg.fingerprint()
    fp_hash = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    datasets = cfg.dataset_names()
    if not datasets:
        raise ValueError(f"no dataset folders under {cfg.dataset_root}")

    caches: dict[str, DatasetCache] = {}
    entries: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for ds in datasets:
        cdir = cfg.cache_dir / ds
        cache = DatasetCache(cdir) if rebuild else DatasetCache.load(cdir)
        caches[ds] = cache
        root = cfg.dataset_root / ds
        for p in sorted(q for q in root.rglob("*") if q.is_file() and q.suffix.lower() == ".stl"):
            rel = p.relative_to(root).as_posix()
            sha = cache.file_sha(rel, p)
            keys = {d.name: cache_key(sha, d, fp) for d in descs}
            e = {"dataset": ds, "rel": rel, "path": str(p), "sha": sha, "keys": keys,
                 "label": p.parent.name, "pre_key": f"{sha}:preprocess:{fp_hash}"}
            entries.append(e)
            if e["pre_key"] in cache.failures:
                continue
            missing = [n for n, k in keys.items() if k not in cache.vectors and k not in cache.failures]
            if missing:
                jobs.append({**e, "names": missing})

    logger.info("build: %d files in %s, %d need extraction (device=%s)",
                len(entries), datasets, len(jobs), device)
    failures: list[tuple[str, str]] = []
    n_vec = 0
    if jobs:
        with tqdm(total=len(jobs), desc="extracting", unit="file") as bar:
            for job, res in _run_jobs(jobs, cfg, device):
                cache = caches[job["dataset"]]
                name = f"{job['dataset']}/{job['rel']}"
                if res["file_error"]:
                    failures.append((name, res["file_error"]))
                    logger.warning("skipped %s: %s", name, res["file_error"])
                    if not res["timeout"]:
                        cache.failures[job["pre_key"]] = res["file_error"]
                for dname, (vec, ms) in res["ok"].items():
                    k = job["keys"][dname]
                    cache.vectors[k], cache.timing_ms[k] = vec, ms
                    n_vec += 1
                for dname, msg in res["err"].items():
                    cache.failures[job["keys"][dname]] = msg
                    failures.append((name, f"{dname}: {msg}"))
                    logger.warning("%s: descriptor %s failed: %s", name, dname, msg)
                bar.update(1)

    for ds, cache in caches.items():
        mine = [e for e in entries if e["dataset"] == ds]
        live = {k for e in mine for k in e["keys"].values()} | {e["pre_key"] for e in mine}
        pruned = cache.prune({e["rel"] for e in mine}, live)
        if pruned:
            logger.info("cache %s: pruned %d stale vectors", ds, pruned)
        cache.save()

    rows = [e for e in entries
            if all(k in caches[e["dataset"]].vectors for k in e["keys"].values())]
    if not rows:
        raise RuntimeError("no STL file produced a complete descriptor set; see warnings")
    state = _fit_state(cfg, descs, rows, caches)
    _fit_index(cfg, state)
    report = BuildReport(datasets, len(entries), len(rows), len(jobs), n_vec, failures,
                         time.perf_counter() - t_start, device)
    logger.info("build: %d/%d files indexed, %d vectors computed, %.1f s",
                report.n_indexed, report.n_files, n_vec, report.seconds)
    return report


def _fit_state(cfg: Config, descs: list[Descriptor], rows: list[dict],
               caches: dict[str, DatasetCache]) -> IndexState:
    """Stack cached vectors and fit per-descriptor mean / std of the scalar dims."""
    raw, mean, std, timing = {}, {}, {}, {}
    for d in descs:
        X = np.stack([caches[e["dataset"]].vectors[e["keys"][d.name]] for e in rows]).astype(np.float32)
        raw[d.name] = X
        S = X[:, d.n_hist:].astype(np.float64)
        mean[d.name] = S.mean(axis=0)
        sd = S.std(axis=0)
        std[d.name] = np.where(sd > STD_FLOOR, sd, 1.0)
        ts = [caches[e["dataset"]].timing_ms.get(e["keys"][d.name]) for e in rows]
        ts = [t for t in ts if t is not None]
        timing[d.name] = float(np.mean(ts)) if ts else float("nan")
    return IndexState(
        names=[d.name for d in descs], dims=[d.dim for d in descs], n_hist=[d.n_hist for d in descs],
        weights=[cfg.weight(d.name) for d in descs],
        ids=[f"{e['dataset']}/{e['rel']}" for e in rows], labels=[e["label"] for e in rows],
        shas=[e["sha"] for e in rows], paths=[repo_relative(e["path"], cfg.root) for e in rows],
        raw=raw, mean=mean, std=std, signature=_signature(cfg, descs), timing_ms=timing)


def _fit_index(cfg: Config, state: IndexState) -> tuple[Compressor, VectorStore]:
    """Fuse the database, fit the compressor, build and persist the index."""
    fused = state.fuse({n: state.block(n, state.raw[n]) for n in state.names})
    comp = make_compressor(cfg.compression, cfg.target_dim).fit(fused)
    Z = comp.transform(fused)
    store = make_store(cfg.backend, Z.shape[1], cfg.metric)
    store.add(Z)
    d = cfg.cache_dir / STATE_DIR
    state.save(d)
    comp.save(d / "compressor.npz")
    store.save(d / f"index_{cfg.backend}.bin")
    return comp, store


def load_state(cfg: Config) -> tuple[IndexState, Compressor, VectorStore]:
    """Load the persisted index; raise if it was built with another config."""
    d = cfg.cache_dir / STATE_DIR
    state = IndexState.load(d)
    descs = make_descriptors(state.names, cfg.descriptor_params, "cpu")
    if state.signature != _signature(cfg, descs) or tuple(state.names) != tuple(cfg.enabled):
        raise RuntimeError("index was built with a different config; run build() again")
    comp = make_compressor(cfg.compression, cfg.target_dim)
    comp.load(d / "compressor.npz")
    store = make_store(cfg.backend, comp.out_dim, cfg.metric)
    store.load(d / f"index_{cfg.backend}.bin")
    return state, comp, store


def query(path: Path, cfg: Config, top_k: int | None = None) -> QueryResult:
    """
    Retrieve the top-k database items for one STL.

    Parameters
    ----------
    path : Path
        Query STL (never inserted into the database).
    cfg : Config
        Same config as the build.
    top_k : int, optional
        Overrides [database].top_k.

    Returns
    -------
    QueryResult
    """
    k = int(top_k or cfg.top_k)
    state, comp, store = load_state(cfg)
    N = len(state.ids)
    k = min(k, N)
    t0 = time.perf_counter()
    device = resolve_device(cfg.device)
    res = extract_file(str(path), state.names, cfg.descriptor_params, device,
                       cfg.seed, cfg.n_samples, cfg.scale_rule, math.inf)
    if res["file_error"] or res["err"]:
        raise DescriptorError(f"query {path}: {res['file_error'] or res['err']}")
    t_extract = (time.perf_counter() - t0) * 1e3

    t1 = time.perf_counter()
    qb = {n: state.block(n, res["ok"][n][0]) for n in state.names}
    sims = {n: (state.block(n, state.raw[n]) @ qb[n][0]).astype(np.float64) for n in state.names}
    ranks = {n: rank_matrix(sims[n][None, :])[0] for n in state.names}
    if cfg.fusion == "rank_fusion":
        rrf = sum(w / (RRF_K + ranks[n]) for n, w in zip(state.names, state.weights))
        order = np.argsort(-rrf, kind="stable")[:k]
        scores = rrf[order]
    else:
        qz = comp.transform(state.fuse(qb))
        scores, ids = store.search(qz, k)
        order, scores = ids[0], scores[0].astype(np.float64)
        if cfg.metric == "l2":
            scores = 1.0 + scores / 2.0          # unit vectors: cos = 1 - d^2 / 2
    t_search = (time.perf_counter() - t1) * 1e3

    qsha = file_sha256(Path(path))
    zs = {n: (sims[n] - sims[n].mean()) / (sims[n].std() + 1e-12) for n in state.names}
    hits = []
    for r, (i, s) in enumerate(zip(order, scores), start=1):
        i = int(i)
        if i < 0:
            continue
        contrib = {n: w * zs[n][i] for n, w in zip(state.names, state.weights)}
        hits.append(Hit(
            rank=r, id=state.ids[i], label=state.labels[i], path=state.paths[i], score=float(s),
            exact_match=state.shas[i] == qsha,
            per_descriptor={n: {"similarity": float(sims[n][i]), "rank": int(ranks[n][i]),
                                "z": float(zs[n][i])} for n in state.names},
            top_descriptors=sorted(contrib, key=contrib.get, reverse=True)[:3]))
    return QueryResult(query_path=str(path), query_sha256=qsha,
                       query_in_database=qsha in set(state.shas), fusion=cfg.fusion,
                       descriptors=list(state.names), hits=hits,
                       timing_ms={"extract": t_extract, "search": t_search})


def evaluate(cfg: Config) -> dict[str, Any]:
    """
    Leave-one-out retrieval quality on the built database (labels = class folders).

    Reports per-descriptor, fused (weighted_zscore), fused + PCA(target_dim),
    and rank fusion, plus mean extraction time per descriptor [ms/mesh].
    Database statistics and PCA are fitted on the whole database
    (transductive; the descriptors themselves are training-free).
    """
    state, _, _ = load_state(cfg)
    labels = np.asarray(state.labels)
    blocks = {n: state.block(n, state.raw[n]).astype(np.float64) for n in state.names}
    rows: dict[str, dict[str, float]] = {}
    R = {}
    for n in state.names:
        S = blocks[n] @ blocks[n].T
        rows[n] = loo_metrics(S, labels)
        np.fill_diagonal(S, -np.inf)
        R[n] = rank_matrix(S)
    fused = state.fuse(blocks).astype(np.float64)
    rows["fused"] = loo_metrics(fused @ fused.T, labels)
    pca = make_compressor("pca", cfg.target_dim).fit(fused)
    Z = pca.transform(fused).astype(np.float64)
    rows[f"fused+pca{pca.out_dim}"] = loo_metrics(Z @ Z.T, labels)
    rrf = sum(w / (RRF_K + R[n]) for n, w in zip(state.names, state.weights))
    rows["rank_fusion"] = loo_metrics(rrf, labels)
    return {"n_models": len(labels), "n_classes": int(len(set(state.labels))),
            "metrics": rows, "ms_per_mesh": state.timing_ms}

# ==============================================================================
# END OF FILE: orchestrator.py
# ==============================================================================
# SUMMARY:
#   build (parallel, incremental, cached) -> fitted index; query; evaluate.
#
# USAGE EXAMPLE:
#   from src.config import load_config
#   from src.orchestrator import build, query
#   cfg = load_config(); build(cfg); res = query(Path("q.stl"), cfg)
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] Hung-task kill uses ProcessPoolExecutor._processes (private attribute).
#   - [ ] evaluate() holds N x N matrices; fine up to ~1e4 models.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
