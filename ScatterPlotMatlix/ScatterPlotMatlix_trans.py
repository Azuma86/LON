import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib.patches import FancyArrowPatch
from itertools import combinations
from pathlib import Path

# ----------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 18

# Plot parameters
EDGE_LW      = 0.1    # edge line width
NODE_SIZE    = 3      # node scatter size
FINAL_NODE_SIZE = 9
NODE_ALPHA   = 0.8    # node transparency
ARROW_SCALE  = 3      # arrowhead scale
ARROW_ALPHA  = 1      # arrow transparency
ARROW_LW     = 0.1    # arrow line width
# Problem settings
problem_name = 'RWMOP22'
algo = 'local11'
base_dir = Path('../data09-20-pre')
csv_path = base_dir / f"{problem_name}_{algo}.csv"
assert csv_path.exists(), f"CSV file not found: {csv_path}"

domain_df = pd.read_csv('../domain_info.csv')
row = domain_df.loc[domain_df['problem'] == problem_name].iloc[0]

lower = np.array([float(v) for v in row['lower'].split(",")])
upper = np.array([float(v) for v in row['upper'].split(",")])

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
data = raw.copy()
# recompute sinks on filtered data
sinks = {n for n in G.nodes() if G.out_degree(n)==0}
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
sizes = {
    'mid_infeasible':   NODE_SIZE,
    'mid_feasible':     NODE_SIZE,
    'final_feasible':   FINAL_NODE_SIZE,   # sink 可行
    'final_infeasible': FINAL_NODE_SIZE,   # sink 不可
}
# plot points by category
for name, mask in masks.items():
    subset = data[mask]
    point_size = sizes[name]
    for i, j in combinations(range(n_dim), 2):
        ax = axes[j, i]
        ax.scatter(
            subset[X_cols[i]], subset[X_cols[j]],
            c=colors[name], edgecolor='black', linewidth=EDGE_LW,
            s=point_size, alpha=NODE_ALPHA,
            label=name if (i,j)==(1,0) else None
        )
        ax2 = axes[i, j]
        ax2.scatter(
            subset[X_cols[j]], subset[X_cols[i]],
            c=colors[name], edgecolor='black', linewidth=EDGE_LW,
            s=point_size, alpha=NODE_ALPHA
        )
# draw arrows only for medoid series
groups = data.groupby('ID')
for sid in series_ids:
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

for i in range(len(X_cols)):
    for j in range(len(X_cols)):
        ax = axes[i, j]
        if i == j:
            d = (upper[i] - lower[i])/20
            ax.set_xlim(lower[i] - d, upper[i] + d)
        else:
            d1 = (upper[j] - lower[j]) / 20
            d2 = (upper[i] - lower[i]) / 20
            ax.set_xlim(lower[j] - d1, upper[j] + d1)
            ax.set_ylim(lower[i] - d2, upper[i] + d2)
# finalize
plt.subplots_adjust(wspace=0.1, hspace=0.1)
for ax in axes.flatten():
    ax.tick_params(labelsize=8)
plt.show()
