
from typing import Sequence, Dict
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean

# --- Optional third-party libs (install as needed) ----------------
import ot                                 # POT  for OT / GW
from tslearn.metrics import dtw           # fast Cython DTW
# from fastdtw import fastdtw            # ← 近似 DTW を使うならこちら

# ------------------------------------------------------------------
def _series_arrays(
    raw: pd.DataFrame,
    series_ids: Sequence[int],
    group_idx: Dict[int, np.ndarray],
    X_cols: Sequence[str],
):
    """
    Helper generator -> yields (sid, TrajArray) pairs
    TrajArray has shape (T_i, n_dim)
    """
    for sid in series_ids:
        Xi = raw.loc[group_idx[sid], X_cols].values
        yield sid, Xi


# ------------------------------------------------------------------
def compute_ot_matrix(raw, series_ids, group_idx, X_cols):
    """
    Wasserstein-1 (Earth-Mover’s) distance between each pair of series.
    Uses POT's ot.emd2 (exact) -> O(T_i * T_j).
    """
    N = len(series_ids)
    W = np.zeros((N, N))
    traj = dict(_series_arrays(raw, series_ids, group_idx, X_cols))

    for i, sid_i in enumerate(series_ids):
        Xi = traj[sid_i]
        mu = np.ones(len(Xi)) / len(Xi)
        for j, sid_j in enumerate(series_ids[i + 1 :], i + 1):
            Xj = traj[sid_j]
            nu = np.ones(len(Xj)) / len(Xj)
            C  = cdist(Xi, Xj, metric="euclidean")
            W_ij = ot.emd2(mu, nu, C)        # exact EMD cost
            W[i, j] = W[j, i] = W_ij
    return W


# ------------------------------------------------------------------
def compute_dtw_matrix(raw, series_ids, group_idx, X_cols):
    """
    Multivariate DTW distance (exact) between each pair of series
    using tslearn.metrics.dtw.
    """
    N = len(series_ids)
    W = np.zeros((N, N))
    traj = dict(_series_arrays(raw, series_ids, group_idx, X_cols))

    for i, sid_i in enumerate(series_ids):
        Xi = traj[sid_i]
        for j, sid_j in enumerate(series_ids[i + 1 :], i + 1):
            Xj = traj[sid_j]
            dist = dtw(Xi, Xj)               # exact DTW
            # 近似版なら: dist, _ = fastdtw(Xi, Xj, dist=euclidean)
            W[i, j] = W[j, i] = dist
    return W


# ------------------------------------------------------------------
def compute_gw_matrix(raw, series_ids, group_idx, X_cols, sinkhorn_eps=None):
    """
    Gromov–Wasserstein distance^2  (square-loss) between series pairs.
    If `sinkhorn_eps` is given (e.g. 1e-1), entropic regularisation is used
    for speed; otherwise the exact solver is called.
    """
    N = len(series_ids)
    W = np.zeros((N, N))
    traj = dict(_series_arrays(raw, series_ids, group_idx, X_cols))

    for i, sid_i in enumerate(series_ids):
        Xi = traj[sid_i]
        Ci = cdist(Xi, Xi, metric="euclidean")
        pi = np.ones(len(Xi)) / len(Xi)
        for j, sid_j in enumerate(series_ids[i + 1 :], i + 1):
            Xj = traj[sid_j]
            Cj = cdist(Xj, Xj, metric="euclidean")
            pj = np.ones(len(Xj)) / len(Xj)

            if sinkhorn_eps is None:
                gw2 = ot.gromov.gromov_wasserstein2(
                    Ci, Cj, pi, pj, loss_fun="square_loss", armijo=True
                )
            else:
                gw2 = ot.gromov.entropic_gromov_wasserstein2(
                    Ci, Cj, pi, pj, loss_fun="square_loss",
                    epsilon=sinkhorn_eps, log=False
                )
            W[i, j] = W[j, i] = np.sqrt(gw2)   # √を取って“距離”に
    return W