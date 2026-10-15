"""
QRCAD v2.0 — src/main.py
Core logic: build_database, search_query, get_cache_status, delete_cache_file
All functions are callable from app.py (Flask) or CLI.
NO script-level execution code here.

CACHE FORMAT v2.0:
Each entry is now a dict with keys:
  - "filename": str
  - "feature_vector": np.ndarray
  - "metadata": dict (see metadata.py for structure)
"""

import os
import concurrent.futures
import numpy as np
import time
import threading

from src.loader import load_mesh, sample_points
from src.pca import pca_align
from src.features import extract_feature_vector
from src.cache import save_cache, load_cache, cache_exists, delete_cache
from src.metadata import compute_mesh_metadata


# Global progress tracking (in-memory, single build at a time)
_build_progress = {
    "status": "idle",  # idle | building | done
    "total_files": 0,
    "files_completed": 0,
    "dataset_path": None,
    "lock": threading.Lock()
}


def get_build_progress():
    """
    Get current cache build progress.
    Returns dict with status and counts only (no time information).
    """
    with _build_progress["lock"]:
        progress = {
            "status": _build_progress["status"],
            "total_files": _build_progress["total_files"],
            "files_completed": _build_progress["files_completed"],
            "dataset_path": _build_progress["dataset_path"]
        }
        
        return progress


def _update_build_progress(status=None, total=None, completed=None, dataset_path=None):
    """Internal helper to update build progress."""
    with _build_progress["lock"]:
        if status is not None:
            _build_progress["status"] = status
        if total is not None:
            _build_progress["total_files"] = total
        if completed is not None:
            _build_progress["files_completed"] = completed
        if dataset_path is not None:
            _build_progress["dataset_path"] = dataset_path


# 
# INTERNAL HELPERS
# 

def _safe_load_and_sample(path: str, num_points: int = 5000):
    """Load an STL, sample points. Returns np.ndarray or None."""
    try:
        mesh = load_mesh(path)
        if mesh is None:
            return None
        points = sample_points(mesh, num_points)
        return points
    except Exception:
        return None


def _process_single_file(path: str, timeout: float = 10.0):
    """
    Load, sample, PCA-align, extract features, and compute metadata for one STL file.
    
    Returns dict with structure:
      {
        "filename": str,
        "feature_vector": np.ndarray,
        "metadata": dict
      }
    or None on failure/timeout.
    """
    filename = os.path.basename(path)
    
    # Load mesh for both feature extraction and metadata
    try:
        mesh = load_mesh(path)
        if mesh is None:
            print(f"  [SKIP]    {filename}  failed to load")
            return None
    except Exception as e:
        print(f"  [ERROR]   {filename}: {e}")
        return None
    
    # Extract feature vector (with timeout)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(sample_points, mesh)
        try:
            points = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"  [TIMEOUT] {filename}")
            return None
        except Exception as e:
            print(f"  [ERROR]   {filename}: {e}")
            return None

    if points is None or len(points) == 0:
        print(f"  [SKIP]    {filename}  no valid geometry")
        return None

    try:
        aligned = pca_align(points)
        feature_vec = extract_feature_vector(aligned)
    except Exception as e:
        print(f"  [FEAT ERR] {filename}: {e}")
        return None
    
    # Compute metadata (reuse the already-loaded mesh)
    metadata = compute_mesh_metadata(path, mesh)
    
    # Store relative path from project root instead of just basename
    # This prevents collisions in nested folder structures
    return {
        "filepath": path,  # Full path for now, will be converted to relative in build_database
        "filename": filename,
        "feature_vector": feature_vec,
        "metadata": metadata
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Numerically stable cosine similarity.
    DEPRECATED: Use RetrievalEngine.search() instead.
    Kept for backwards compatibility.
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


# 
# PUBLIC API  (called by app.py)
# 

def build_database(dataset_path: str) -> dict:
    """
    Walk dataset_path, process every .stl file, save cache with metadata.
    Returns a summary dict.
    
    Cache format v2.0: List of dicts, each containing filename, feature_vector, and metadata.
    """
    build_start = time.time()
    
    dataset_path = os.path.abspath(dataset_path)

    if not os.path.isdir(dataset_path):
        return {"error": f"Dataset path does not exist: {dataset_path}"}

    # Collect all STL paths first
    stl_files = []
    for root, _dirs, files in os.walk(dataset_path):
        for fname in files:
            if fname.lower().endswith(".stl"):
                stl_files.append(os.path.join(root, fname))

    total = len(stl_files)
    if total == 0:
        return {"error": "No .stl files found in the specified folder."}

    print(f"\n[BUILD] Found {total} STL files in: {dataset_path}")
    
    # Initialize progress tracking
    _update_build_progress(status="building", total=total, completed=0, dataset_path=dataset_path)

    database = []
    valid = 0
    skipped = 0

    for path in stl_files:
        result = _process_single_file(path)
        if result is not None:
            # Convert absolute filepath to relative path from dataset root
            result["filename"] = os.path.relpath(result["filepath"], dataset_path)
            # Remove the temporary filepath key
            del result["filepath"]
            
            database.append(result)
            valid += 1
            print(f"  [OK]      {result['filename']}  ({valid}/{total})")
        else:
            skipped += 1
        
        # Update progress after each file
        _update_build_progress(completed=valid + skipped)

    if not database:
        _update_build_progress(status="idle")
        return {"error": "No STL files could be processed. Check your dataset."}

    # **8B FIX: Save cache with metadata including file list for validation**
    metadata = {
        "model_count": len(database),
        "files": [
            {
                "name": entry["filename"],
                "path": entry["filename"],
                "mtime": os.path.getmtime(os.path.join(dataset_path, entry["filename"]))
            }
            for entry in database
        ]
    }
    
    # **BUG 2 DEBUG: Log the actual saved file paths**
    print(f"\n[DEBUG BUILD] Saving metadata with {len(metadata['files'])} files:")
    for i, f in enumerate(metadata['files'][:5]):  # Show first 5
        print(f"  [{i}] path={repr(f['path'])}")
    if len(metadata['files']) > 5:
        print(f"  ... and {len(metadata['files']) - 5} more")
    
    save_cache(database, dataset_path, metadata)
    
    build_duration = time.time() - build_start
    
    # Mark as done
    _update_build_progress(status="done")

    print(f"\n[BUILD] Complete: {valid} indexed, {skipped} skipped.")
    return {
        "valid_files": valid,
        "skipped_files": skipped,
        "total_files": total,
        "models": valid,
        "build_duration": round(build_duration, 2)
    }


def search_query(query_path: str, dataset_path: str, top_k: int = 5) -> dict:
    """
    Load query STL, compute features and metadata, compare against cached database.
    Uses RetrievalEngine.search() for the actual retrieval (no pipeline modifications).
    Public API for single-query search (used by /search endpoint).
    
    Returns dict with structure:
      {
        "query_metadata": dict,
        "results": [
          {
            "filename": str,
            "similarity": float,
            "metadata": dict
          },
          ...
        ],
        "feature_extraction_time": float,
        "retrieval_time": float,
        "models_compared": int,
        "feature_dimension": int
      }
    """
    from src.retrieval_engine import get_engine
    import time
    
    query_path = os.path.abspath(query_path)
    dataset_path = os.path.abspath(dataset_path)

    if not os.path.isfile(query_path):
        raise FileNotFoundError(f"Query file not found: {query_path}")

    if not cache_exists(dataset_path):
        raise RuntimeError(
            "No cache found for this dataset. Please build the cache first."
        )

    # Load query mesh for both features and metadata
    extraction_start = time.time()
    
    mesh = load_mesh(query_path)
    if mesh is None:
        raise ValueError("Query STL could not be loaded.")
    
    points = sample_points(mesh)
    if points is None or len(points) == 0:
        raise ValueError("Query STL has no valid geometry.")

    aligned = pca_align(points)
    query_vec = extract_feature_vector(aligned)
    
    feature_extraction_time = time.time() - extraction_start
    
    # Compute query metadata (reuse loaded mesh)
    query_metadata = compute_mesh_metadata(query_path, mesh)

    # Load database and perform retrieval
    retrieval_start = time.time()
    
    database = load_cache(dataset_path)
    if not database:
        raise RuntimeError("Cache is empty. Please rebuild the cache.")

    # Use retrieval engine for search (no pipeline modifications)
    engine = get_engine("classical")
    results = engine.search(query_vec, database, top_k)
    
    retrieval_time = time.time() - retrieval_start

    return {
        "query_metadata": query_metadata,
        "results": results,
        "feature_extraction_time": round(feature_extraction_time, 3),
        "retrieval_time": round(retrieval_time, 3),
        "models_compared": len(database),
        "feature_dimension": len(query_vec)
    }


def get_cache_status(dataset_path: str) -> dict:
    """
    Returns cache existence and model count for dataset_path.
    Also indicates if cache needs rebuild (old format detected).
    """
    dataset_path = os.path.abspath(dataset_path)
    exists = cache_exists(dataset_path)

    if not exists:
        return {"exists": False, "models": 0}

    db = load_cache(dataset_path)
    
    # Check if this is an old-format cache (tuple-based)
    needs_rebuild = False
    if db and len(db) > 0:
        first_entry = db[0]
        if not isinstance(first_entry, dict):
            needs_rebuild = True
    
    return {
        "exists": True,
        "models": len(db),
        "needs_rebuild": needs_rebuild,
        "cache_version": "v1.0 (no metadata)" if needs_rebuild else "v2.0 (with metadata)"
    }


def delete_cache_file(dataset_path: str) -> bool:
    """
    Delete the cache for dataset_path. Returns True if deleted.
    """
    dataset_path = os.path.abspath(dataset_path)
    return delete_cache(dataset_path)
