import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib.patches import FancyArrowPatch
from itertools import combinations
from pathlib import Path
import ot                                      # POT: Python Optimal Transport
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

# ----------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 18

# Plot parameters
EDGE_LW      = 0.1    # edge line width
NODE_SIZE    = 3      # node scatter size
NODE_ALPHA   = 0.8    # node transparency
ARROW_SCALE  = 1      # arrowhead scale
ARROW_ALPHA  = 1      # arrow transparency
ARROW_LW     = 0.1    # arrow line width
n_clluster = 10
# Problem settings
problem_name = 'RWMOP23'
algo         = 'data'
base_dir     = Path('../../data09-20')
csv_path     = base_dir / f"{problem_name}_{algo}.csv"
assert csv_path.exists(), f"CSV file not found: {csv_path}"

# ----------------------------------------------------------------
# 1. Load data and compute CV
# ----------------------------------------------------------------
raw = pd.read_csv(csv_path)
X_cols = [c for c in raw.columns if c.startswith('X_')]
con_cols = [c for c in raw.columns if c.startswith('Con_')]
n_dim = len(X_cols)
assert n_dim >= 4, "This script requires dimension >= 4"

# compute total constraint violation (clip negatives)
raw['CV'] = raw[con_cols].clip(lower=0).sum(axis=1)
raw = raw.sort_values(['ID','Gen']).reset_index(drop=True)

# ----------------------------------------------------------------
# 2. Build graph and collect series data
# ----------------------------------------------------------------
G = nx.DiGraph()
for idx, row in raw.iterrows():
    G.add_node(idx, X=row[X_cols].values, CV=row['CV'])
    if idx > 0:
        prev = raw.iloc[idx-1]
        if (prev['ID']==row['ID']) and (row['Gen']==prev['Gen']+1):
            G.add_edge(idx-1, idx)
# group indices by series ID
group_idx = raw.groupby('ID').indices
series_ids  = list(group_idx.keys())

N = len(series_ids)
W = np.zeros((N, N))

for i, sid_i in enumerate(series_ids):
    Xi   = raw.loc[group_idx[sid_i], X_cols].values          # (T_i, n_dim)
    Ci   = cdist(Xi, Xi, metric='euclidean')                 # Ci: T_i × T_i
    pi   = np.ones(len(Xi)) / len(Xi)                        # 一様質量 μ_i

    for j, sid_j in enumerate(series_ids[i+1:], start=i+1):
        Xj   = raw.loc[group_idx[sid_j], X_cols].values      # (T_j, n_dim)
        Cj   = cdist(Xj, Xj, metric='euclidean')             # Cj: T_j × T_j
        pj   = np.ones(len(Xj)) / len(Xj)                    # ν_j

        # ---------- Gromov-Wasserstein 距離^2 を計算 ----------
        gw2 = ot.gromov.gromov_wasserstein2(
            C1=Ci, C2=Cj, p=pi, q=pj, loss_fun='square_loss', armijo=True
        )                          # armijo=True で収束性/速度Up
        W[i, j] = W[j, i] = gw2

# ----------------------------------------------------------------
# 4. Cluster series and select medoids
# ----------------------------------------------------------------
# use `metric` instead of deprecated `affinity`
cl = AgglomerativeClustering(
    n_clusters=n_clluster, metric='precomputed', linkage='average'
).fit(W)
labels = cl.labels_

medoids = []
for ci in range(n_clluster):
    members = np.where(labels==ci)[0]
    subW    = W[np.ix_(members, members)]
    med_idx = members[np.argmin(subW.sum(axis=1))]
    medoids.append(series_ids[med_idx])
print(f"Selected series IDs: {medoids}")

# filter raw and G to medoid series only
mask_medoid = raw['ID'].isin(medoids)
data = raw[mask_medoid].copy()

# recompute sinks on filtered data
sinks = {n for n in G.nodes() if G.out_degree(n)==0 and mask_medoid[n]}

# ----------------------------------------------------------------
# 5. Scatter-matrix visualization with filtered series
# ----------------------------------------------------------------
# prepare axes
axes = scatter_matrix(
    data[X_cols], figsize=(12,12), diagonal='hist', alpha=0.2, s=10
)
# categorize nodes
data['is_sink']  = data.index.isin(sinks)
data['feasible'] = data['CV']==0
masks = {
    'mid_infeasible':  (~data['is_sink']) & (~data['feasible']),
    'mid_feasible':    (~data['is_sink']) & ( data['feasible']),
    'final_feasible':   data['is_sink'] & ( data['feasible']),
    'final_infeasible': data['is_sink'] & (~data['feasible']),
}
colors = {
    'mid_infeasible':   'skyblue',
    'mid_feasible':     'salmon',
    'final_feasible':   'red',
    'final_infeasible': 'blue',
}
# plot points by category
for name, mask in masks.items():
    subset = data[mask]
    for i, j in combinations(range(n_dim), 2):
        ax = axes[j, i]
        ax.scatter(
            subset[X_cols[i]], subset[X_cols[j]],
            c=colors[name], edgecolor='black', linewidth=EDGE_LW,
            s=NODE_SIZE, alpha=NODE_ALPHA,
            label=name if (i,j)==(1,0) else None
        )
        ax2 = axes[i, j]
        ax2.scatter(
            subset[X_cols[j]], subset[X_cols[i]],
            c=colors[name], edgecolor='black', linewidth=EDGE_LW,
            s=NODE_SIZE, alpha=NODE_ALPHA
        )
# draw arrows only for medoid series
groups = data.groupby('ID')
for sid in medoids:
    group = groups.get_group(sid).sort_values('Gen')
    for u_idx, v_idx in zip(group.index[:-1], group.index[1:]):
        u = group.loc[u_idx, X_cols].values
        v = group.loc[v_idx, X_cols].values
        for i, j in combinations(range(n_dim), 2):
            ax = axes[j, i]
            arr = FancyArrowPatch(
                (u[i], u[j]), (v[i], v[j]),
                arrowstyle='-|>', mutation_scale=ARROW_SCALE,
                color='gray', alpha=ARROW_ALPHA, lw=ARROW_LW
            )
            ax.add_patch(arr)
            ax2 = axes[i, j]
            arr2 = FancyArrowPatch(
                (u[j], u[i]), (v[j], v[i]),
                arrowstyle='-|>', mutation_scale=ARROW_SCALE,
                color='gray', alpha=ARROW_ALPHA, lw=ARROW_LW
            )
            ax2.add_patch(arr2)
# finalize
for ax in axes.flatten():
    ax.tick_params(labelsize=8)
plt.show()
