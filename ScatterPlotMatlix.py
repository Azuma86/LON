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

# Problem settings
problem_name = 'RWMOP23'
algo = 'data'
base_dir = Path('data09-20')
csv_path = base_dir / f"{problem_name}_{algo}.csv"
assert csv_path.exists(), f"CSV file not found: {csv_path}"

# ----------------------------------------------------------------
# 1. Load data and compute CV
# ----------------------------------------------------------------
data = pd.read_csv(csv_path)
X_cols = [c for c in data.columns if c.startswith('X_')]
con_cols = [c for c in data.columns if c.startswith('Con_')]
n_dim = len(X_cols)

# total constraint violation (clip negative to zero)
data['CV'] = data[con_cols].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)
# sort by ID and generation for graph edges order

data_sorted = data.sort_values(['ID', 'Gen']).reset_index(drop=True)
# グラフ構築

G = nx.DiGraph()
for idx, row in data_sorted.iterrows():
    G.add_node(idx, X=row[[f'X_{i+1}' for i in range(n_dim)]].values, CV=row['CV'])
    if idx>0:
        prev = data_sorted.iloc[idx-1]
        if (prev['ID']==row['ID']) and (row['Gen']==prev['Gen']+1):
            G.add_edge(idx-1, idx)
sinks = {n for n in G.nodes() if G.out_degree(n)==0}
# ---- 2. 散布図行列を作成 ----
features = [f'X_{i+1}' for i in range(n_dim)]
axes = scatter_matrix(data_sorted[features],
                     figsize=(12,12),
                     diagonal='hist',
                     alpha=0.2, s=10)

# --- カテゴリマスクと色定義 ---
data_sorted['is_sink']  = data_sorted.index.isin(sinks)
data_sorted['feasible'] = data_sorted['CV'] == 0

masks = {
    'mid_infeasible':   (~data_sorted['is_sink']) & (~data_sorted['feasible']),
    'mid_feasible':     (~data_sorted['is_sink']) & ( data_sorted['feasible']),
    'final_feasible':   ( data_sorted['is_sink']) & ( data_sorted['feasible']),
    'final_infeasible': ( data_sorted['is_sink']) & (~data_sorted['feasible']),
}

colors = {
    'mid_infeasible':   'skyblue',
    'mid_feasible':     'salmon',
    'final_feasible':   'red',
    'final_infeasible': 'blue',
}

# --- カラフルな点を上書き描画 ---
for name, mask in masks.items():
    subset = data_sorted[mask]
    for i, j in combinations(range(n_dim), 2):
        ax = axes[j, i]
        ax.scatter(
            subset[f'X_{i+1}'], subset[f'X_{j+1}'],
            c=colors[name], edgecolor = 'black',linewidth = 0.3,s=10, label=name if (i,j)==(1,0) else None, alpha=0.8
        )
        # 対称プロットにも
        ax2 = axes[i, j]
        ax2.scatter(
            subset[f'X_{j+1}'], subset[f'X_{i+1}'],
            c=colors[name], edgecolor = 'black',linewidth = 0.3,s=10, alpha=0.8
        )

# --- 矢印（世代遷移）を重ね描き ---
for id_val, group in data_sorted.groupby('ID'):
    group = group.sort_values('Gen')
    for u_idx, v_idx in zip(group.index[:-1], group.index[1:]):
        u = group.loc[u_idx, features].values
        v = group.loc[v_idx, features].values
        for i, j in combinations(range(n_dim), 2):
            ax = axes[j, i]
            arrow = FancyArrowPatch(
                (u[i], u[j]), (v[i], v[j]),
                arrowstyle='-|>', mutation_scale=3,
                color='gray', alpha=0.8, lw=0.2
            )
            ax.add_patch(arrow)
            # 対称側も
            ax2 = axes[i, j]
            arrow2 = FancyArrowPatch(
                (u[j], u[i]), (v[j], v[i]),
                arrowstyle='-|>', mutation_scale=3,
                color='gray', alpha=0.8, lw=0.2
            )
            ax2.add_patch(arrow2)

plt.show()