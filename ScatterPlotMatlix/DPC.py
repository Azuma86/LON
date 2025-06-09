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
problem_name = 'RWMOP22'
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


edge_info = []          # [(CV, edge_len), …]

for u, v in G.edges():
    cv_u  = G.nodes[u]['CV']
    x_u   = G.nodes[u]['X']
    x_v   = G.nodes[v]['X']
    dist  = np.linalg.norm(x_u - x_v, ord=2)

    edge_info.append((cv_u, dist))

# numpy 配列にして扱いやすく
edge_info = np.array(edge_info)           # shape = (n_edge, 2)
CV_vals   = edge_info[:, 0]
edge_len  = edge_info[:, 1]


q1, q3 = np.percentile(edge_len, [25, 75])
iqr     = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
mask    = (edge_len >= lower) & (edge_len <= upper)

CV_plot  = CV_vals
len_plot = edge_len


plt.figure(figsize=(8, 6))
plt.scatter(len_plot, CV_plot, s=10, alpha=0.7)   # 点サイズや透明度はお好みで
plt.ylabel("CV")
plt.xlabel("Edge Length")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()