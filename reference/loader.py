import trimesh
import numpy as np
import os


def load_mesh(file_path):
    try:
        # Skip very small files (likely corrupted)
        if os.path.getsize(file_path) < 100:
            return None

        # Load mesh safely
        mesh = trimesh.load(file_path, force='mesh', skip_materials=True)

        # Handle scenes
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return None
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        # Validate mesh
        if mesh is None or len(mesh.vertices) == 0:
            return None

        return mesh

    except:
        return None


def sample_points(mesh, num_points=5000):
    try:
        if mesh is None:
            return None

        if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            return None

        # Use tessellation-independent even sampling with fallback
        try:
            points, _ = trimesh.sample.sample_surface_even(mesh, count=num_points)
        except Exception:
            points = mesh.sample(num_points)
        
        return points

    except Exception as e:
        print(f"[SAMPLE ERROR] {e}")
        return None
