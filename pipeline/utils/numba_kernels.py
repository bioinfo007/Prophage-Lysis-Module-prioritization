"""
numba_kernels.py
================
JIT-compiled kernels for the two most CPU-intensive operations:
  1. Pairwise cosine similarity (M07 redundancy collapse)
  2. MaxMin diversity selection (M08)

Auto-detects GPU (CUDA) and falls back to parallel CPU transparently.
Same Python interface regardless of hardware.

Usage:
    from pipeline.utils.numba_kernels import cosine_similarity_matrix, maxmin_select
"""

import logging
import numpy as np
from typing import List, Tuple

log = logging.getLogger("numba_kernels")

# ── GPU/CPU detection ─────────────────────────────────────────────────────────

def _detect_cuda() -> bool:
    try:
        from numba import cuda
        if cuda.is_available():
            log.info("CUDA GPU detected — using GPU kernels")
            return True
    except Exception:
        pass
    return False

_HAS_CUDA = _detect_cuda()

try:
    from numba import njit, prange, float32, int32, boolean
    _HAS_NUMBA = True
    log.info("Numba JIT available — compiled kernels active")
except ImportError:
    _HAS_NUMBA = False
    log.warning("Numba not installed — falling back to NumPy (slower)")


# ── Cosine similarity — CPU parallel ─────────────────────────────────────────

if _HAS_NUMBA:
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=True)
    def _cosine_sim_cpu(A: np.ndarray, out: np.ndarray) -> None:
        """
        Compute symmetric N×N cosine similarity matrix in-place.
        A: (N, D) float32, L2-normalized rows.
        out: (N, N) float32 output.
        Parallel over rows, computes upper triangle + mirrors.
        """
        N = A.shape[0]
        for i in prange(N):
            out[i, i] = 1.0
            for j in range(i + 1, N):
                s = 0.0
                for k in range(A.shape[1]):
                    s += A[i, k] * A[j, k]
                # Clamp to [-1, 1] to handle floating point drift
                if s > 1.0:  s = 1.0
                if s < -1.0: s = -1.0
                out[i, j] = s
                out[j, i] = s

    @njit(parallel=True, cache=True, fastmath=True)
    def _maxmin_kernel(
        dist:         np.ndarray,   # (N, N) cosine DISTANCE matrix
        selected_idx: np.ndarray,   # (max_sel,) int32, pre-allocated
        n_selected:   int,
        remaining:    np.ndarray,   # (N,) boolean mask
        n_remaining:  int,
    ) -> Tuple[int, float]:
        """
        Single MaxMin step: find remaining candidate with max min-distance
        to all currently selected candidates.

        Uses per-row array to avoid prange race condition on shared scalars.
        prange parallelizes the min-distance computation; argmax is sequential.
        """
        N = dist.shape[0]
        min_dists = np.full(N, -1.0)

        for i in prange(N):
            if not remaining[i]:
                continue
            min_d = 1e10
            for s in range(n_selected):
                d = dist[i, selected_idx[s]]
                if d < min_d:
                    min_d = d
            min_dists[i] = min_d

        # Sequential argmax — no race condition
        best_idx  = -1
        best_dist = -1.0
        for i in range(N):
            if min_dists[i] > best_dist:
                best_dist = min_dists[i]
                best_idx  = i

        return best_idx, best_dist

else:
    # NumPy fallback — no Numba
    def _cosine_sim_cpu(A, out):
        np.dot(A, A.T, out=out)
        np.clip(out, -1.0, 1.0, out=out)

    def _maxmin_kernel(dist, selected_idx, n_selected, remaining, n_remaining):
        sel = selected_idx[:n_selected]
        min_dists = dist[:, sel].min(axis=1)
        min_dists[~remaining] = -1.0
        best_idx  = int(np.argmax(min_dists))
        best_dist = float(min_dists[best_idx])
        return best_idx, best_dist


# ── GPU kernel (optional, only if CUDA available) ────────────────────────────

if _HAS_CUDA:
    try:
        from numba import cuda
        import math

        @cuda.jit
        def _cosine_sim_gpu(A, out):
            i, j = cuda.grid(2)
            N, D = A.shape
            if i >= N or j >= N:
                return
            s = 0.0
            for k in range(D):
                s += A[i, k] * A[j, k]
            if s > 1.0:  s = 1.0
            if s < -1.0: s = -1.0
            out[i, j] = s

        _GPU_KERNEL_OK = True
    except Exception as e:
        log.warning(f"CUDA kernel compilation failed: {e} — using CPU")
        _HAS_CUDA = False
        _GPU_KERNEL_OK = False
else:
    _GPU_KERNEL_OK = False


# ── Public API ────────────────────────────────────────────────────────────────

def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row. Returns float32."""
    matrix = matrix.astype(np.float32)
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms  = np.where(norms < 1e-10, 1e-10, norms)
    return (matrix / norms).astype(np.float32)


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute full N×N cosine similarity matrix.

    Automatically uses:
      - CUDA GPU kernel if available
      - Numba parallel CPU kernel otherwise
      - NumPy fallback if Numba not installed

    For large N (>5000), consider block_cosine_similarity instead.

    Args:
        matrix: (N, D) float array — need not be normalized

    Returns:
        (N, N) float32 cosine similarity matrix
    """
    A   = l2_normalize(matrix)
    N   = A.shape[0]
    out = np.empty((N, N), dtype=np.float32)

    if _GPU_KERNEL_OK:
        import math
        from numba import cuda
        d_A   = cuda.to_device(A)
        d_out = cuda.device_array((N, N), dtype=np.float32)
        threads = (16, 16)
        blocks  = (math.ceil(N / 16), math.ceil(N / 16))
        _cosine_sim_gpu[blocks, threads](d_A, d_out)
        out = d_out.copy_to_host()
    else:
        _cosine_sim_cpu(A, out)

    return out


def block_cosine_similarity(
    matrix:     np.ndarray,
    threshold:  float,
    block_size: int = 512,
) -> np.ndarray:
    """
    Memory-efficient blocked cosine similarity — builds only the
    edges (i, j) where similarity > threshold, without materializing
    the full N×N matrix.

    Returns: (M, 3) array of [i, j, similarity] pairs (upper triangle only).
    Use this for N > 5000 to avoid OOM.
    """
    A      = l2_normalize(matrix)
    N      = A.shape[0]
    edges  = []

    for i_start in range(0, N, block_size):
        i_end = min(i_start + block_size, N)
        A_i   = A[i_start:i_end]   # (block, D)

        for j_start in range(i_start, N, block_size):
            j_end  = min(j_start + block_size, N)
            A_j    = A[j_start:j_end]   # (block, D)

            block  = A_i @ A_j.T        # (block_i, block_j)
            np.clip(block, -1.0, 1.0, out=block)

            ii, jj = np.where(block > threshold)

            for k in range(len(ii)):
                gi = i_start + int(ii[k])
                gj = j_start + int(jj[k])
                if gi < gj:   # upper triangle only
                    edges.append([gi, gj, float(block[ii[k], jj[k]])])

    if not edges:
        return np.empty((0, 3), dtype=np.float32)
    return np.array(edges, dtype=np.float32)


def maxmin_select(
    matrix:    np.ndarray,
    max_n:     int,
    seed_idx:  int = -1,
) -> Tuple[List[int], List[float]]:
    """
    MaxMin (Farthest Point Sampling) in cosine distance space.

    Args:
        matrix:   (N, D) embedding matrix, need not be normalized
        max_n:    maximum candidates to select
        seed_idx: index of seed candidate (-1 = centroid-nearest)

    Returns:
        selected_indices: list of selected row indices in selection order
        marginal_dists:   list of marginal distances (one per selection step)
    """
    A    = l2_normalize(matrix)
    N    = A.shape[0]
    dist = np.ones((N, N), dtype=np.float32)   # cosine distance = 1 - cosine_sim
    sim  = np.empty((N, N), dtype=np.float32)
    _cosine_sim_cpu(A, sim)
    np.subtract(1.0, sim, out=dist)
    np.fill_diagonal(dist, 0.0)

    # Seed
    if seed_idx < 0:
        centroid    = A.mean(axis=0)
        cn          = centroid / (np.linalg.norm(centroid) + 1e-10)
        seed_idx    = int(np.argmax(A @ cn))

    selected_idx  = np.full(max_n, -1, dtype=np.int32)
    selected_idx[0] = seed_idx
    remaining     = np.ones(N, dtype=bool)
    remaining[seed_idx] = False

    sel_list  = [seed_idx]
    dist_list = [0.0]

    for step in range(1, max_n):
        if remaining.sum() == 0:
            break
        best_i, best_d = _maxmin_kernel(
            dist, selected_idx, len(sel_list), remaining, int(remaining.sum())
        )
        if best_i < 0:
            break
        selected_idx[len(sel_list)] = best_i
        remaining[best_i]            = False
        sel_list.append(int(best_i))
        dist_list.append(float(best_d))

    return sel_list, dist_list


def compute_mean_pairwise_distance(
    matrix:       np.ndarray,
    selected_idx: List[int],
) -> float:
    """Mean pairwise cosine distance within a selected subset."""
    if len(selected_idx) < 2:
        return 0.0
    A    = l2_normalize(matrix[selected_idx])
    sim  = A @ A.T
    np.fill_diagonal(sim, 1.0)   # exclude self
    dist = 1.0 - sim
    np.fill_diagonal(dist, np.nan)
    return float(np.nanmean(dist))
