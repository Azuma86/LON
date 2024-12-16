import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import matplotlib.ticker as ticker

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20

data = pd.read_csv('data/RWMOP22_data2.csv')
con_cols = [c for c in data.columns if c.startswith('Con_')]
data['CV'] = data[con_cols].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)
#print(data['CV'])
data_sorted = data.sort_values(by=['ID', 'Gen'])
G = nx.DiGraph()
X_cols = [c for c in data.columns if c.startswith('X_')]

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

# ここでGに重複ノードはなくなった
nodes = list(G.nodes())
X_all = np.array([G.nodes[n]['X'] for n in nodes])
N = len(nodes)

dist_matrix = pairwise_distances(X_all, metric='euclidean')
epsilon = 1e-10
dist_matrix[dist_matrix < epsilon] = epsilon

dist_dict = {
    ni: {nj: dist_matrix[i, j] for j, nj in enumerate(nodes)}
    for i, ni in enumerate(nodes)
}

# Kamada-Kawaiでdim=1
pos_1d = nx.kamada_kawai_layout(G, dist=dist_dict, dim=1)

# posを (x座標=1D埋め込み, y座標=CV) に再配置
pos = {}
for n in nodes:
    x_coord = pos_1d[n][0]
    cv_val = G.nodes[n]['CV']
    pos[n] = (x_coord, cv_val)

# 最終ノード（シンクノード）の特定
sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
print(len(sink_nodes))
# シンクノードの分類
final_feasible = [n for n in sink_nodes if G.nodes[n]['CV'] <= 0]
final_infeasible = [n for n in sink_nodes if G.nodes[n]['CV'] > 0]
other_nodes = [n for n in nodes if n not in sink_nodes]

# プロットの作成
plt.figure(figsize=(15, 12))

# エッジの描画
nx.draw_networkx_edges(
    G, pos, arrowstyle='->', arrowsize=15, edge_color='gray', alpha=0.5
)

# 他の世代のノードを描画
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=other_nodes,
    node_size=50,
    node_color='skyblue',
    edgecolors='black',

)

# 最終世代の実行可能ノード（塗りつぶし）
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=final_feasible,
    node_size=50,
    node_color='red',
    edgecolors='black',
)

# 最終世代の実行不可能ノード（穴あき）
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=final_infeasible,
    node_size=50,
    node_color='blue',
    edgecolors='black',
    node_shape='o',  # デフォルトの円形
    linewidths=2,
)

ax = plt.gca()
ax.yaxis.set_major_locator(ticker.AutoLocator())
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.tick_params(axis='y', which='both', labelleft=True)
#plt.yticks(np.arange(0,1.6,0.2))
plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)

#ax.axis('on')

plt.show()