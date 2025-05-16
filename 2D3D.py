import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 for 3D projection
from collections import defaultdict
import matplotlib.ticker as ticker
from pathlib import Path

# ----------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 18

# Problem settings
problem_name = 'RWMOP25'
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
assert n_dim in (2, 3), f"Supports only 2D/3D problems, got {n_dim}D"

# total constraint violation (clip negative to zero)
data['CV'] = data[con_cols].clip(lower=0).sum(axis=1)
# sort by ID and generation for graph edges order
data_sorted = data.sort_values(['ID', 'Gen']).reset_index(drop=True)

# ----------------------------------------------------------------
# 2. Build directed graph
# ----------------------------------------------------------------
G = nx.DiGraph()
for idx, row in data_sorted.iterrows():
    G.add_node(idx,
               Gen=row['Gen'],
               ID=row['ID'],
               X=row[X_cols].values,
               CV=row['CV'])

prev_row = None
prev_idx = None

for idx, row in data_sorted.iterrows():
    if prev_row is not None and prev_row['ID'] == row['ID'] and row['Gen'] == prev_row['Gen'] + 1:
        G.add_edge(prev_idx, idx)
    prev_row = row
    prev_idx = idx
sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

# 重複ノード群の特定
vec2nodes = defaultdict(list)
for n in G.nodes():
    vec = tuple(G.nodes[n]['X'])
    vec2nodes[vec].append(n)

# 重複ノード統合処理
for vec, node_list in vec2nodes.items():
    if len(node_list) > 1:
        # 最初のノードを代表ノードとする
        representative = node_list[0]
        duplicates = node_list[1:]

        for dup in duplicates:
            # dupノードへの入出エッジを代表ノードに付け替え
            # 入ってくるエッジ( pred -> dup )を ( pred -> representative )へ
            for pred in list(G.predecessors(dup)):
                if pred != representative:  # 同一ノードへのループは避ける
                    # エッジがまだ存在しなければ追加
                    if not G.has_edge(pred, representative):
                        G.add_edge(pred, representative)

            # dupノードから出るエッジ(dup -> succ)を (representative -> succ)へ
            for succ in list(G.successors(dup)):
                if succ != representative:
                    if not G.has_edge(representative, succ):
                        G.add_edge(representative, succ)

            # dupノード削除
            G.remove_node(dup)

# ----------------------------------------------------------------
# 4. Determine positions from X coordinates
# ----------------------------------------------------------------
pos2d = {}
pos3d = {} if n_dim == 3 else None
for n, data_attr in G.nodes(data=True):
    xvals = data_attr['X']
    if n_dim == 2:
        pos2d[n] = (xvals[0], xvals[1])
    else:
        pos3d[n] = (xvals[0], xvals[1], xvals[2])

# ----------------------------------------------------------------
# 5. Classify nodes by sink/middle and feasibility
# ----------------------------------------------------------------
sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
middle = [n for n in G.nodes() if n not in sinks]

final_feasible = [n for n in sinks if G.nodes[n]['CV'] == 0]
final_infeasible = [n for n in sinks if G.nodes[n]['CV'] > 0]
mid_feasible = [n for n in middle if G.nodes[n]['CV'] == 0]
mid_infeasible = [n for n in middle if G.nodes[n]['CV'] > 0]

# color mapping
color_map = {
    'mid_infeasible': 'skyblue',
    'mid_feasible': 'salmon',
    'final_feasible': 'red',
    'final_infeasible': 'blue'
}

# ----------------------------------------------------------------
# 6. Plotting
# ----------------------------------------------------------------
if n_dim == 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    # edges
    nx.draw_networkx_edges(
        G, pos2d, ax=ax, arrowstyle='-|>', arrowsize=12,
        edge_color='gray', alpha=0.4
    )
    # nodes
    for name, nodeset in zip(
        ['mid_infeasible','mid_feasible','final_feasible','final_infeasible'],
        [mid_infeasible, mid_feasible, final_feasible, final_infeasible]
    ):
        nx.draw_networkx_nodes(
            G, pos2d, nodelist=nodeset,
            node_color=color_map[name], node_size=10,
        )

    plt.show()
else:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # edges
    for u, v in G.edges():
        xs, ys, zs = zip(pos3d[u], pos3d[v])
        ax.plot(xs, ys, zs, color='gray', alpha=0.4)
    # nodes
    for name, nodeset, size in [
        ('mid_infeasible', mid_infeasible, 10),
        ('mid_feasible', mid_feasible, 10),
        ('final_feasible', final_feasible, 50),
        ('final_infeasible', final_infeasible, 50),
    ]:
        coords = np.array([pos3d[n] for n in nodeset])
        if coords.size:
            ax.scatter(
                coords[:,0], coords[:,1], coords[:,2],
                s=size, c=color_map[name], edgecolors='k', label=name.replace('_',' ').title()
            )
    plt.show()
