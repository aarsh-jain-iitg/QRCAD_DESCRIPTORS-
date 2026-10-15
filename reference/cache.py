"""
QRCAD src/cache.py
Handles feature cache persistence using pickle.
**8B FIX**: Cache is now stored INSIDE the database folder itself at <dataset_path>/.qrcad_cache/
This makes cache detection restart-proof and eliminates any path-to-cache mapping sync issues.
"""

import os
import sys
import pickle
import json
import time

# **8B: Cache subdirectory name inside each database folder**
CACHE_FOLDER_NAME = ".qrcad_cache"
FEATURES_FILENAME = "features.pkl"
METADATA_FILENAME = "metadata.json"


def get_cache_path(dataset_path: str) -> str:
    """
    **8B FIX**: Return the cache directory path INSIDE the database folder.
    Structure: <dataset_path>/.qrcad_cache/
    This folder contains:
      - features.pkl (the actual feature database)
      - metadata.json (model count, timestamp, file list for validation)
    """
    cache_dir = os.path.join(os.path.abspath(dataset_path), CACHE_FOLDER_NAME)
    return cache_dir


def _ensure_cache_dir(dataset_path: str) -> str:
    """Create cache directory inside the database folder if it doesn't exist."""
    cache_dir = get_cache_path(dataset_path)
    os.makedirs(cache_dir, exist_ok=True)
    
    # **8B: On Windows, mark as hidden to avoid cluttering folder browser**
    if sys.platform == 'win32':
        try:
            import ctypes
            # FILE_ATTRIBUTE_HIDDEN = 0x02
            ctypes.windll.kernel32.SetFileAttributesW(cache_dir, 0x02)
        except Exception:
            pass  # Ignore if we can't set hidden attribute
    
    return cache_dir


def save_cache(database: list, dataset_path: str, metadata: dict = None) -> None:
    """
    **8B FIX**: Persist feature database to disk INSIDE the database folder.
    Also saves metadata for cache validation.
    """
    cache_dir = _ensure_cache_dir(dataset_path)
    features_path = os.path.join(cache_dir, FEATURES_FILENAME)
    metadata_path = os.path.join(cache_dir, METADATA_FILENAME)
    
    # Save feature database
    with open(features_path, "wb") as f:
        pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save metadata for validation
    if metadata is None:
        metadata = {}
    
    metadata['timestamp'] = time.time() * 1000  # **BUG 4 FIX**: Convert to milliseconds for JS compatibility
    metadata['model_count'] = len(database)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[CACHE] Saved to {cache_dir} ({len(database)} models)")


def load_cache(dataset_path: str) -> list:
    """
    **8B FIX**: Load and return feature database from INSIDE the database folder.
    Returns [] if missing.
    """
    cache_dir = get_cache_path(dataset_path)
    features_path = os.path.join(cache_dir, FEATURES_FILENAME)
    
    if not os.path.exists(features_path):
        return []
    
    try:
        with open(features_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[CACHE] Error loading cache: {e}")
        return []


def cache_exists(dataset_path: str) -> bool:
    """
    **8B FIX**: Return True if cache exists INSIDE the database folder.
    Checks for both features.pkl and metadata.json.
    """
    cache_dir = get_cache_path(dataset_path)
    features_path = os.path.join(cache_dir, FEATURES_FILENAME)
    metadata_path = os.path.join(cache_dir, METADATA_FILENAME)
    
    return (os.path.exists(features_path) and os.path.exists(metadata_path))


def delete_cache(dataset_path: str) -> bool:
    """
    **8B FIX**: Delete cache directory INSIDE the database folder.
    Returns True if it existed and was removed.
    """
    cache_dir = get_cache_path(dataset_path)
    
    if os.path.exists(cache_dir):
        import shutil
        try:
            shutil.rmtree(cache_dir)
            print(f"[CACHE] Deleted {cache_dir}")
            return True
        except Exception as e:
            print(f"[CACHE] Error deleting cache: {e}")
            return False
    
    return False
