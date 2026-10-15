import numpy as np
from scipy.stats import skew


def pca_align(points):
    """
    Robust, deterministic PCA alignment for 3D point clouds.

    Pipeline (unchanged variable names, same overall stage order):
      1. Center
      2. Scale by bounding-box diagonal
      3. Covariance
      4. Eigendecomposition (stable, symmetric solver)
      5. Sort eigenvalues/vectors descending
      6. Degenerate-eigenvalue handling (near-symmetric shapes)
      7. Right-handed system enforcement
      8. Align (project points onto eigenbasis)
      9. Sign fix (mass-imbalance, then skew tie-break)
    """
    points = np.asarray(points, dtype=np.float64)

    # ----------------------------------------------------------------
    # STEP 1 - CENTER
    # ----------------------------------------------------------------
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # ----------------------------------------------------------------
    # STEP 2 - SCALE (BOUNDING BOX DIAGONAL)
    # ----------------------------------------------------------------
    bbox_min = centered.min(axis=0)
    bbox_max = centered.max(axis=0)
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    if diagonal > 1e-12:
        centered = centered / diagonal
    else:
        return centered  # degenerate case: single point / zero extent

    # ----------------------------------------------------------------
    # STEP 3 - COVARIANCE
    # ----------------------------------------------------------------
    cov = np.cov(centered.T)

    # ----------------------------------------------------------------
    # STEP 4 - EIGEN DECOMPOSITION (STABLE)
    # ----------------------------------------------------------------
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # ----------------------------------------------------------------
    # STEP 5 - SORT DESCENDING
    # ----------------------------------------------------------------
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # ----------------------------------------------------------------
    # STEP 6 - HANDLE DEGENERACY (SYMMETRY FIX)
    #
    # FIX: the previous version flipped eigenvectors[:, i] based only on
    # the sign of that vector's single largest-magnitude component. When
    # eigenvalues are close, eigh() can return ANY orthonormal basis for
    # that subspace (not just a sign-flip of some "canonical" pair), so a
    # sign-only correction does not make alignment reproducible across
    # near-identical point clouds. It could also touch the same column
    # twice when BOTH (0,1) and (1,2) pairs are degenerate at once (e.g.
    # near-spherical shapes), since axis 1 is "j" in one pair and "i" in
    # the next, making its final sign order-dependent.
    #
    # Instead: for any axis involved in a degenerate pair, re-derive its
    # sign from a moment of the point cloud ITSELF (skew of the points
    # projected onto that eigenvector), not from eigh's incidental basis
    # choice. Each axis is touched at most once in this pass, then any
    # pair that was jointly degenerate is re-orthonormalized via a single
    # Gram-Schmidt step so the basis stays exactly orthonormal.
    # ----------------------------------------------------------------
    DEGENERATE_THRESHOLD = 1e-4
    degenerate_axes = set()
    for i, j in [(0, 1), (1, 2)]:
        if abs(eigenvalues[i] - eigenvalues[j]) < DEGENERATE_THRESHOLD:
            degenerate_axes.add(i)
            degenerate_axes.add(j)

    if degenerate_axes:
        for axis_pos in sorted(degenerate_axes):
            v = eigenvectors[:, axis_pos]
            proj = centered @ v
            s = skew(proj) if proj.size > 2 else 0.0
            if not np.isnan(s) and s < 0:
                eigenvectors[:, axis_pos] = -v

        for i, j in [(0, 1), (1, 2)]:
            if i in degenerate_axes and j in degenerate_axes:
                vi = eigenvectors[:, i]
                vj = eigenvectors[:, j]
                vj = vj - np.dot(vj, vi) * vi
                norm_vj = np.linalg.norm(vj)
                if norm_vj > 1e-12:
                    eigenvectors[:, j] = vj / norm_vj

    # ----------------------------------------------------------------
    # STEP 7 - RIGHT-HANDED SYSTEM (NO MIRROR)
    # ----------------------------------------------------------------
    cross = np.cross(eigenvectors[:, 0], eigenvectors[:, 1])
    if np.dot(cross, eigenvectors[:, 2]) < 0:
        eigenvectors[:, 2] *= -1

    # ----------------------------------------------------------------
    # STEP 8 - ALIGN
    # ----------------------------------------------------------------
    aligned = centered @ eigenvectors

    # ----------------------------------------------------------------
    # STEP 9 - SIGN FIX (MASS -> SKEW -> KEEP)
    #
    # Unchanged logic, kept as the final deterministic tie-break. For
    # degenerate axes this is a harmless second pass (Step 6 already
    # oriented them by skew); for normal axes it remains the primary
    # sign-resolution mechanism, exactly as before.
    # ----------------------------------------------------------------
    for i in range(3):
        pos = np.sum(aligned[:, i] > 0)
        neg = np.sum(aligned[:, i] < 0)
        if pos != neg:
            if neg > pos:
                aligned[:, i] *= -1
        else:
            s = skew(aligned[:, i])
            if not np.isnan(s) and abs(s) > 1e-6 and s < 0:
                aligned[:, i] *= -1

    return aligned