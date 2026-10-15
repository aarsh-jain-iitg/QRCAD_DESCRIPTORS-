"""
================================================================================
FILE: config.py
================================================================================
PURPOSE:
    Load src/config.toml with stdlib tomllib and validate it into a frozen
    Config dataclass. Unknown keys, wrong types, out-of-range values and
    missing input paths raise ValueError naming the offending key, so a
    typo never silently falls back to a default.

INPUTS:
    - config.toml path (default: src/config.toml next to this file).
    - Repo root (paths in the file are relative to it).

OUTPUTS:
    - Config dataclass; Config.fingerprint() for cache keys.

DEPENDENCIES:
    stdlib      : dataclasses, os, pathlib, tomllib
    third-party : (none)
    internal    : descriptors (REGISTRY, for names and param defaults)

AUTHOR: Aarsh Jain
LAST MODIFIED: 2026-09-23
================================================================================
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.descriptors import REGISTRY
from src.preprocessing import SCALE_RULES

# ── PATH VARIABLES ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent          # repo root
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.toml"

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
BACKENDS = ("faiss_flat", "faiss_hnsw", "numpy")
METRICS = ("cosine", "l2")
COMPRESSIONS = ("none", "pca")
DEVICES = ("auto", "cuda", "cpu")
FUSIONS = ("weighted_zscore", "rank_fusion")
MAX_AUTO_WORKERS = 8

SCHEMA: dict[str, dict[str, type | tuple[type, ...]]] = {
    "paths": {"dataset_root": str, "datasets": list, "cache_dir": str, "output_dir": str},
    "database": {"backend": str, "metric": str, "top_k": int},
    "compression": {"method": str, "target_dim": int},
    "pipeline": {"seed": int, "device": str, "n_samples": int, "timeout_s": (int, float),
                 "workers": int, "scale_rule": str, "enabled": list, "fusion": str,
                 "weights": dict},
}


# ── HELPER / UTILITY FUNCTIONS ────────────────────────────────────────────────
def _check_choice(key: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{key} = {value!r}: expected one of {list(choices)}")


def _check_table(section: str, table: dict[str, Any]) -> None:
    """Reject unknown / missing keys and wrong types in one [section]."""
    spec = SCHEMA[section]
    for k in table:
        if k not in spec:
            raise ValueError(f"unknown key [{section}].{k}")
    for k, typ in spec.items():
        if k not in table:
            if k == "weights":
                continue
            raise ValueError(f"missing key [{section}].{k}")
        v = table[k]
        if isinstance(v, bool) or not isinstance(v, typ):
            raise ValueError(f"[{section}].{k} has type {type(v).__name__}, expected {typ}")


def _descriptor_params(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate [descriptors.<name>] tables against each class's PDF defaults."""
    out: dict[str, dict[str, Any]] = {}
    for name, table in raw.items():
        if name not in REGISTRY:
            raise ValueError(f"unknown descriptor table [descriptors.{name}]")
        if not isinstance(table, dict):
            raise ValueError(f"[descriptors.{name}] must be a table")
        defaults = REGISTRY[name].defaults
        for k, v in table.items():
            if k not in defaults:
                raise ValueError(f"unknown key [descriptors.{name}].{k}")
            want = type(defaults[k])
            ok = isinstance(v, (int, float)) and not isinstance(v, bool) if want is float \
                else isinstance(v, want) and not isinstance(v, bool)
            if not ok:
                raise ValueError(f"[descriptors.{name}].{k} has type {type(v).__name__}, "
                                 f"expected {want.__name__}")
            if isinstance(v, (int, float)) and v <= 0 and k not in ("log_lo", "log_hi"):
                raise ValueError(f"[descriptors.{name}].{k} must be > 0, got {v}")
        out[name] = {k: (float(v) if type(defaults[k]) is float else v) for k, v in table.items()}
    return out


# ── CORE CLASSES / MAIN FUNCTIONS ─────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    """Validated pipeline configuration (see config.toml for meanings)."""

    root: Path
    dataset_root: Path
    datasets: tuple[str, ...]
    cache_dir: Path
    output_dir: Path
    backend: str
    metric: str
    top_k: int
    compression: str
    target_dim: int
    seed: int
    device: str
    n_samples: int
    timeout_s: float
    workers: int
    scale_rule: str
    enabled: tuple[str, ...]
    fusion: str
    weights: dict[str, float] = field(default_factory=dict)
    descriptor_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def dataset_names(self) -> list[str]:
        """Resolve datasets = ["all"] to the current subfolders of dataset_root."""
        if self.datasets != ("all",):
            return list(self.datasets)
        return sorted(p.name for p in self.dataset_root.iterdir() if p.is_dir())

    def fingerprint(self) -> dict[str, Any]:
        """Pipeline settings that change every descriptor vector (cache keys)."""
        return {"seed": self.seed, "n_samples": self.n_samples, "scale_rule": self.scale_rule}

    def weight(self, name: str) -> float:
        return float(self.weights.get(name, 1.0))

    def n_workers(self) -> int:
        return self.workers if self.workers > 0 else min(os.cpu_count() or 1, MAX_AUTO_WORKERS)


def config_from_dict(data: dict[str, Any], root: Path = ROOT_DIR) -> Config:
    """
    Validate a parsed TOML dict into a Config.

    Parameters
    ----------
    data : dict
        Parsed config.toml content.
    root : Path
        Directory that relative paths resolve against (the repo root).

    Returns
    -------
    Config

    Raises
    ------
    ValueError
        On unknown keys, wrong types, bad choices or missing input paths.
    """
    for section in data:
        if section not in SCHEMA and section != "descriptors":
            raise ValueError(f"unknown section [{section}]")
    for section in SCHEMA:
        if section not in data:
            raise ValueError(f"missing section [{section}]")
        _check_table(section, data[section])
    pa, db, co, pi = data["paths"], data["database"], data["compression"], data["pipeline"]

    dataset_root = root / pa["dataset_root"]
    if not dataset_root.is_dir():
        raise ValueError(f"[paths].dataset_root: directory not found: {dataset_root}")
    datasets = tuple(pa["datasets"])
    if not datasets or not all(isinstance(d, str) for d in datasets):
        raise ValueError("[paths].datasets must be a non-empty list of strings")
    if datasets != ("all",):
        for d in datasets:
            if not (dataset_root / d).is_dir():
                raise ValueError(f"[paths].datasets: {d!r} not found under {dataset_root}")

    _check_choice("[database].backend", db["backend"], BACKENDS)
    _check_choice("[database].metric", db["metric"], METRICS)
    _check_choice("[compression].method", co["method"], COMPRESSIONS)
    _check_choice("[pipeline].device", pi["device"], DEVICES)
    _check_choice("[pipeline].fusion", pi["fusion"], FUSIONS)
    _check_choice("[pipeline].scale_rule", pi["scale_rule"], SCALE_RULES)
    for key, v in (("[database].top_k", db["top_k"]), ("[compression].target_dim", co["target_dim"]),
                   ("[pipeline].n_samples", pi["n_samples"]), ("[pipeline].timeout_s", pi["timeout_s"])):
        if v <= 0:
            raise ValueError(f"{key} must be > 0, got {v}")
    if pi["workers"] < 0:
        raise ValueError(f"[pipeline].workers must be >= 0, got {pi['workers']}")
    if pi["n_samples"] < 10:
        raise ValueError(f"[pipeline].n_samples must be >= 10, got {pi['n_samples']}")

    enabled = tuple(REGISTRY) if list(pi["enabled"]) == ["all"] else tuple(pi["enabled"])
    for n in enabled:
        if n not in REGISTRY:
            raise ValueError(f"[pipeline].enabled: unknown descriptor {n!r}")
    if not enabled:
        raise ValueError("[pipeline].enabled is empty")
    weights = dict(pi.get("weights", {}))
    for n, w in weights.items():
        if n not in REGISTRY:
            raise ValueError(f"[pipeline.weights]: unknown descriptor {n!r}")
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0:
            raise ValueError(f"[pipeline.weights].{n} must be a number >= 0")

    return Config(
        root=root, dataset_root=dataset_root, datasets=datasets,
        cache_dir=root / pa["cache_dir"], output_dir=root / pa["output_dir"],
        backend=db["backend"], metric=db["metric"], top_k=db["top_k"],
        compression=co["method"], target_dim=co["target_dim"],
        seed=pi["seed"], device=pi["device"], n_samples=pi["n_samples"],
        timeout_s=float(pi["timeout_s"]), workers=pi["workers"], scale_rule=pi["scale_rule"],
        enabled=enabled, fusion=pi["fusion"], weights={k: float(v) for k, v in weights.items()},
        descriptor_params=_descriptor_params(data.get("descriptors", {})),
    )


def load_config(path: Path = DEFAULT_CONFIG, root: Path = ROOT_DIR) -> Config:
    """Read and validate a config.toml file."""
    with Path(path).open("rb") as fh:
        return config_from_dict(tomllib.load(fh), root)

# ==============================================================================
# END OF FILE: config.py
# ==============================================================================
# SUMMARY:
#   tomllib loader + strict validation -> frozen Config dataclass.
#
# USAGE EXAMPLE:
#   from src.config import load_config
#   cfg = load_config()
#   cfg.dataset_names(), cfg.fingerprint()
#
# KNOWN LIMITATIONS / TODO:
#   - [ ] None.
#
# CHANGELOG:
#   2026-09-23  Initial implementation
# ==============================================================================
