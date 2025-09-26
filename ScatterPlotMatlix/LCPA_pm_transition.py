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

def poly_inv(y, x, l, u, eta, e=1e-20):
    delta = (u - l)
    if y < x:
        t1 = ((u - x) / delta)**(eta + 1)
        t2 = ((delta - x + y) / delta)**(eta + 1)
        return (t1 - t2) / (2 * (t1 - 1) + e)
    elif y == x:
        return 0.5
    else:
        t1 = ((x - l) / delta)**(eta + 1)
        t2 = ((delta + x - y) / delta)**(eta + 1)
        return (t1 + t2 - 2) / (2 * (t1 - 1) + e)

def extremeness(y, x, l, u, eta):
    return 2 * abs(poly_inv(y, x, l, u, eta) - 0.5)

def transition_confidence(x, y, lower, upper, eta, pi):
    D = len(x)
    p = []
    for d in range(D):
        ex = extremeness(y[d], x[d], lower[d], upper[d], eta)
        if x[d] == y[d]:
            p.append(1.0)
        else:
            p.append(pi * (1 - ex))
    return np.prod(p)




plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 18

# Problem settings
problem_name = 'RWMOP20'
algo = 'local31'
base_dir = Path('../data09-20-pre')
csv_path = base_dir / f"{problem_name}_{algo}.csv"
assert csv_path.exists(), f"CSV file not found: {csv_path}"
domain_df = pd.read_csv('domain_info.csv')
row = domain_df.loc[domain_df['problem'] == problem_name].iloc[0]

lower = np.array([float(v) for v in row['lower'].split(",")])
upper = np.array([float(v) for v in row['upper'].split(",")])
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

for idx in G.nodes:
    G.nodes[idx]['Xn'] = (G.nodes[idx]['X'] - lower) / (upper - lower)

key_of = {n: tuple(G.nodes[n]['X']) for n in G.nodes}
groups = defaultdict(list)
for n, k in key_of.items():
    groups[k].append(n)
dup_count = {n: len(groups[key_of[n]]) for n in G.nodes}
nx.set_node_attributes(G, dup_count, name='count')

edge_info = []          # [(CV, edge_len), …]

for u, v in G.edges():
    cv_u  = G.nodes[u]['CV']
    count = G.nodes[u]['count']
    x_u   = G.nodes[u]['X']
    x_v   = G.nodes[v]['X']
    dist  = np.linalg.norm(x_u - x_v, ord=2)

    edge_info.append((cv_u, dist, count))

# numpy 配列にして扱いやすく
edge_info = np.array(edge_info)           # shape = (n_edge, 2)
CV_vals   = edge_info[:, 0]
edge_len  = edge_info[:, 1]
counts = edge_info[:, 2]
# PMパラメータ
n_dim = len(lower)
eta_m = 20
pm    = 0.9
threshold = 0.01
conf_vals = []
for u, v in G.edges():
    x = G.nodes[u]['X']
    y = G.nodes[v]['X']
    conf_vals.append(transition_confidence(
        x, y, lower, upper, eta_m, pm
    ))

conf_vals = np.array(conf_vals)
print(np.max(conf_vals))
print(np.min(conf_vals))
# ----------------------------------------------------------------
# 4. 散布図を色分けしてプロット
# ----------------------------------------------------------------

is_neighbor = (conf_vals >= threshold)

plt.figure(figsize=(8, 6))

# 非近傍（赤）
plt.scatter(edge_len[~is_neighbor],
            CV_vals[~is_neighbor],
            #counts[~is_neighbor],
            c='red',  s=10, alpha=0.6,
            label=f'Non-neighbor (C < {threshold})')
# 近傍（青）
plt.scatter(edge_len[is_neighbor],
            CV_vals[is_neighbor],
            #counts[is_neighbor],
            c='blue', s=10, alpha=0.6,
            label=f'Neighbor (C > {threshold})')

plt.xlabel("Edge Length")
plt.ylabel("CV")
plt.legend()
plt.grid(alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.show()