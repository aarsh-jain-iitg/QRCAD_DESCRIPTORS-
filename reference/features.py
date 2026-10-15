import numpy as np
from scipy.ndimage import label

# ---------- GLOBAL RANDOM SEED FOR REPRODUCIBILITY ----------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- SLICE CREATION (unchanged, but note: axis is used correctly now) ----------


def create_slices(points, num_slices, axis):
    vals = points[:, axis]
    mn, mx = vals.min(), vals.max()
    if mx - mn < 1e-10:
        return [np.empty((0, 3))] * num_slices
    edges = np.linspace(mn, mx, num_slices + 1)
    slices = []
    for i in range(num_slices):
        if i < num_slices - 1:
            mask = (vals >= edges[i]) & (vals < edges[i+1])
        else:
            mask = (vals >= edges[i]) & (vals <= edges[i+1])
        slices.append(points[mask])
    return slices

# ---------- TOPOLOGY FEATURES (FIXED: correct projection axes) ----------


def slice_features(slice_pts, axis):
    """
    axis: which axis was used for slicing (0=X,1=Y,2=Z)
    We project onto the two remaining axes.
    """
    if len(slice_pts) == 0:
        return [0, 0, 0, 0, 0, 0]

    # Determine projection axes (the two that are NOT the slicing axis)
    proj_axes = [0, 1, 2]
    # e.g., if axis=0 (X), proj_axes = [1,2] -> YZ plane
    proj_axes.remove(axis)
    xy = slice_pts[:, proj_axes]    # now correctly 2D projection

    # Normalize to [0,1] range for grid
    min_xy = xy.min(0)
    max_xy = xy.max(0)
    range_xy = max_xy - min_xy
    range_xy[range_xy < 1e-8] = 1.0
    xy = (xy - min_xy) / range_xy
    grid = np.zeros((32, 32))
    idx = (xy * 31).astype(int)
    idx = np.clip(idx, 0, 31)
    grid[idx[:, 0], idx[:, 1]] = 1

    labeled, num = label(grid)
    sizes = [np.sum(labeled == i) for i in range(1, num+1)]
    sizes = sorted(sizes, reverse=True)
    sizes = sizes[:2] + [0]*(2-len(sizes))

    density = np.sum(grid) / (32 * 32)
    # Eccentricity (spread)
    if len(slice_pts) > 2:
        cov = np.cov(xy.T)
        ev = np.linalg.eigvalsh(cov)
        eccentricity = np.sqrt(1 - (ev.min() / (ev.max() + 1e-8)))
    else:
        eccentricity = 0

    return [num, sizes[0], sizes[1], np.sum(grid), density, eccentricity]

# ---------- MULTI VIEW + MULTI RES (now passes axis to slice_features) ----------


def slicing_features(points):
    feats = []
    for axis in [0, 1, 2]:
        # coarse
        slices = create_slices(points, 8, axis)
        for s in slices:
            feats.extend(slice_features(s, axis))
        # fine
        slices = create_slices(points, 40, axis)
        for s in slices:
            feats.extend(slice_features(s, axis))
    return np.array(feats)

# ---------- D2 SHAPE (VECTORIZED, NO LOOP, WITH SEED) ----------


def compute_d2(points, samples=2000, bins=32):
    np.random.seed(RANDOM_SEED)   # ensure reproducibility inside function
    n = len(points)
    if n < 2:
        return np.zeros(bins)

    # Vectorized sampling
    idx = np.random.randint(0, n, (samples, 2))
    d = np.linalg.norm(points[idx[:, 0]] - points[idx[:, 1]], axis=1)
    d = d / (np.max(d) + 1e-8)
    hist, _ = np.histogram(d, bins=bins, range=(0, 1), density=True)
    return hist

# ---------- A3 ANGLE FEATURE (FULLY VECTORIZED  NO PYTHON LOOP) ----------


def compute_a3(points, samples=2000, bins=32):
    np.random.seed(RANDOM_SEED)
    n = len(points)
    if n < 3:
        return np.zeros(bins)

    # Sample all triangles at once
    idx = np.random.randint(0, n, (samples, 3))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]

    v1 = p2 - p1
    v2 = p3 - p1
    dot = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)

    cos_angle = np.clip(dot / (norm1 * norm2 + 1e-8), -1, 1)
    angles = np.arccos(cos_angle)

    hist, _ = np.histogram(angles, bins=bins, range=(0, np.pi), density=True)
    return hist

# ---------- D1 RADIAL DISTRIBUTION (VECTORIZED) ----------


def compute_d1(points, samples=2000, bins=32):
    np.random.seed(RANDOM_SEED)
    n = len(points)
    if n < 1:
        return np.zeros(bins)

    centroid = np.mean(points, axis=0)
    idx = np.random.randint(0, n, min(samples, n))
    distances = np.linalg.norm(points[idx] - centroid, axis=1)
    distances = distances / (np.max(distances) + 1e-8)
    hist, _ = np.histogram(distances, bins=bins, range=(0, 1), density=True)
    return hist

# ---------- HELPER: NORMALIZE EACH BLOCK SEPARATELY ----------


def normalize_block(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v

# ---------- FINAL FEATURE VECTOR (CRITICAL FIX: BLOCKWISE NORMALIZATION) ----------


def extract_feature_vector(points):
    slice_feat = slicing_features(points)
    d2_feat = compute_d2(points)
    a3_feat = compute_a3(points)
    d1_feat = compute_d1(points)

    #  Each block normalized independently  balanced contribution
    slice_feat = normalize_block(slice_feat)
    d2_feat = normalize_block(d2_feat)
    a3_feat = normalize_block(a3_feat)
    d1_feat = normalize_block(d1_feat)

    vec = np.concatenate([slice_feat, d2_feat, a3_feat, d1_feat])
    # Optional: final overall normalization (keeps balance, just scales whole vector)
    vec = normalize_block(vec)
    return vec
