from collections import defaultdict

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
problem_name = 'RWMOP28'
algo = 'local11'
base_dir = Path('../data09-20-pre')
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
print(data['X_1'].values[0])
print(data['X_2'].iloc)
data_sorted = data.sort_values(['ID', 'Gen']).reset_index(drop=True)
# グラフ構築
G = nx.DiGraph()
for idx, row in data_sorted.iterrows():
    G.add_node(idx, X=row[[f'X_{i+1}' for i in range(n_dim)]].values, CV=row['CV'])
    #G.add_node(idx, X=row.iloc[2:n_dim+2].values, CV=row['CV'])
    if idx == 0:
        print("b")
    if idx>0:
        prev = data_sorted.iloc[idx-1]
        if (prev['ID']==row['ID']) and (row['Gen']==prev['Gen']+1):
            G.add_edge(idx-1, idx)


edge_lengths = []
for u, v in G.edges():
    x_u = G.nodes[u]['X']
    x_v = G.nodes[v]['X']
    # ユークリッド距離
    dist = np.linalg.norm(x_u - x_v,ord=2)

    edge_lengths.append(dist)
edge_lengths = np.array(edge_lengths)

print(np.max(edge_lengths))
#edge_lengths[edge_lengths<1e-12] = 1e-12
#log_lengths = np.log10(edge_lengths)
q1 = np.percentile(edge_lengths, 25)
q3 = np.percentile(edge_lengths, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# 2. 外れ値を除外したデータ
filtered_lengths = edge_lengths[(edge_lengths >= lower_bound) & (edge_lengths <= upper_bound)]
# 2) ヒストグラム描画
plt.figure(figsize=(8,5))
n, bins, patches = plt.hist(
    edge_lengths[(edge_lengths >=  0.1) ],
    #edge_lengths,
    #filtered_lengths,
    bins='auto',
    color='steelblue',
    edgecolor='black',
    alpha=0.8
)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()