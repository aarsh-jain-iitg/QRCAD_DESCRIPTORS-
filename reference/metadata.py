"""
QRCAD v2.0 — src/metadata.py
Compute and extract metadata from 3D mesh files using trimesh.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
import trimesh


def compute_mesh_metadata(file_path: str, mesh: trimesh.Trimesh = None) -> Dict[str, Any]:
    """
    Compute comprehensive metadata for a 3D mesh file.
    
    Args:
        file_path: Path to the mesh file on disk
        mesh: Optional pre-loaded trimesh object (to avoid reloading)
        
    Returns:
        Dictionary containing all metadata fields
    """
    metadata = {
        "filename": os.path.basename(file_path),
        "file_size_bytes": None,
        "vertex_count": None,
        "face_count": None,
        "bounding_box": None,  # {"x": float, "y": float, "z": float}
        "surface_area": None,
        "volume": None,
        "watertight": False,
        "date_modified": None,  # ISO format string
        "file_format": None,
    }
    
    try:
        # File-level metadata
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            metadata["file_size_bytes"] = stat.st_size
            # Use modification time (mtime) as the consistent timestamp
            metadata["date_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            metadata["file_format"] = os.path.splitext(file_path)[1].lower()
        
        # Mesh-level metadata
        if mesh is not None and hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            # Vertex and face counts
            metadata["vertex_count"] = len(mesh.vertices)
            metadata["face_count"] = len(mesh.faces)
            
            # Bounding box dimensions
            try:
                bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
                extents = mesh.extents  # [x_extent, y_extent, z_extent]
                metadata["bounding_box"] = {
                    "x": float(extents[0]),
                    "y": float(extents[1]),
                    "z": float(extents[2])
                }
            except Exception:
                pass
            
            # Surface area
            try:
                metadata["surface_area"] = float(mesh.area)
            except Exception:
                pass
            
            # Volume (only if watertight)
            try:
                is_watertight = mesh.is_watertight
                metadata["watertight"] = bool(is_watertight)
                
                if is_watertight:
                    # Only compute volume for watertight meshes
                    metadata["volume"] = float(mesh.volume)
                else:
                    # Explicitly set to None for non-watertight meshes
                    metadata["volume"] = None
            except Exception:
                # If watertight check fails, assume not watertight
                metadata["watertight"] = False
                metadata["volume"] = None
        
    except Exception as e:
        # If any metadata computation fails, return what we have
        print(f"  [METADATA WARNING] {metadata['filename']}: {e}")
    
    return metadata


def format_metadata_for_display(metadata: Dict[str, Any]) -> Dict[str, str]:
    """
    Format metadata values for human-readable display.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Dictionary with formatted string values
    """
    formatted = {}
    
    # File size in human-readable format
    size_bytes = metadata.get("file_size_bytes")
    if size_bytes is not None:
        if size_bytes < 1024:
            formatted["file_size"] = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            formatted["file_size"] = f"{size_bytes / 1024:.2f} KB"
        else:
            formatted["file_size"] = f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        formatted["file_size"] = "N/A"
    
    # Vertex and face counts
    vertex_count = metadata.get("vertex_count")
    formatted["vertices"] = f"{vertex_count:,}" if vertex_count is not None else "N/A"
    
    face_count = metadata.get("face_count")
    formatted["faces"] = f"{face_count:,}" if face_count is not None else "N/A"
    
    # Bounding box
    bbox = metadata.get("bounding_box")
    if bbox and all(k in bbox for k in ['x', 'y', 'z']):
        formatted["bbox"] = f"{bbox['x']:.2f} × {bbox['y']:.2f} × {bbox['z']:.2f}"
    else:
        formatted["bbox"] = "N/A"
    
    # Surface area
    area = metadata.get("surface_area")
    formatted["surface_area"] = f"{area:.2f}" if area is not None else "N/A"
    
    # Volume (with watertight indication)
    volume = metadata.get("volume")
    watertight = metadata.get("watertight", False)
    if volume is not None and watertight:
        formatted["volume"] = f"{volume:.2f}"
    else:
        formatted["volume"] = "N/A (not watertight)" if not watertight else "N/A"
    
    # Date modified
    date_str = metadata.get("date_modified")
    if date_str:
        try:
            dt = datetime.fromisoformat(date_str)
            formatted["date"] = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            formatted["date"] = date_str
    else:
        formatted["date"] = "N/A"
    
    # File format
    fmt = metadata.get("file_format")
    formatted["format"] = fmt if fmt else "N/A"
    
    return formatted
