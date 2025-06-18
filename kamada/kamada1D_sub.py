from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ot.gromov import gromov_wasserstein2
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import matplotlib.ticker as ticker
import ot
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20
problem_name = 'RWMOP28'
name = 'RWMOP7'
algo = 'data'
n_clusters = 30
if problem_name != name:
    domain_df = pd.read_csv('../domain_info.csv')

    # 指定した問題名の行を取得
    row = domain_df.loc[domain_df['problem'] == problem_name].iloc[0]

    # lower, upper を配列化 (スペース区切りを float に変換)
    lower = np.array([float(v) for v in row['lower'].split(",")])
    upper = np.array([float(v) for v in row['upper'].split(",")])
    diff = upper - lower
# =============================
# 1. CSVファイルの読み込みと前処理
# =============================
data = pd.read_csv(f'data09-20-pre/{problem_name}_{algo}.csv')
con_cols = [c for c in data.columns if c.startswith('Con_')]
total = data[con_cols].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)
target_constraints = ['Con_1']
sub = data[target_constraints].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)

data['CV'] = total
data_sorted = data.sort_values(by=['ID', 'Gen'])
G = nx.DiGraph()
X_cols = [c for c in data.columns if c.startswith('X_')]

for idx, row in data_sorted.iterrows():
    G.add_node(idx,
               Gen=row['Gen'],
               ID=row['ID'],
               X=row[X_cols].values,
               CV=row['CV'])


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

# -----------------------------
# 2. Extract subgraphs per series ID
# -----------------------------
id_values = sorted({G.nodes[n]['ID'] for n in G.nodes()})
subgraphs = []
for id_val in id_values:
    nodes_i = [n for n in G.nodes() if G.nodes[n]['ID']==id_val]
    subgraphs.append(G.subgraph(nodes_i).copy())

# -----------------------------
# 3. Compute distance matrices with inf handling
# -----------------------------
dist_mats = []
for H in subgraphs:
    D = nx.floyd_warshall_numpy(H)         # yields inf for unreachable
    D = np.array(D, dtype=float)
    if np.isinf(D).any():
        max_fin = np.nanmax(D[np.isfinite(D)])
        D[np.isinf(D)] = max_fin * 10       # large finite value
    dist_mats.append(D)

# -----------------------------
# 4. Compute pairwise Gromov--Wasserstein distances
# -----------------------------
n = len(dist_mats)
W = np.zeros((n,n))
for i in range(n):
    for j in range(i+1, n):
        C1 = dist_mats[i]
        C2 = dist_mats[j]
        p  = np.ones(C1.shape[0]) / C1.shape[0]
        q  = np.ones(C2.shape[0]) / C2.shape[0]
        # handle any NaNs
        C1 = np.nan_to_num(C1, nan=max_fin*10)
        C2 = np.nan_to_num(C2, nan=max_fin*10)
        gw = gromov_wasserstein2(C1, C2, p, q, 'square_loss')
        W[i, j] = W[j, i] = gw

# -----------------------------
# 5. Cluster series by GW distance and select medoids
# -----------------------------
cl = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric='precomputed',
    linkage='average'
).fit(W)
labels = cl.labels_

rep_ids = []
for c in range(n_clusters):
    idxs = np.where(labels==c)[0]
    subW  = W[np.ix_(idxs, idxs)]
    med   = idxs[np.argmin(subW.sum(axis=1))]
    rep_ids.append(id_values[med])
print("Representative series IDs:", rep_ids)

# -----------------------------
# 6. Filter G to representative subgraphs and visualize
# -----------------------------
keep = [n for n in G.nodes() if G.nodes[n]['ID'] in rep_ids]
G = G.subgraph(keep).copy()
prev_row = None
prev_idx = None

# --- 以下は元の可視化コードを G_filtered で再利用 ---
nodes = list(G.nodes())

X_all = np.array([G.nodes[n]['X'] for n in nodes])
N = len(nodes)

#正規化
if problem_name != name:
    X_all = (X_all - lower) / diff

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

# =============================
# 6. シンクノードの分類 (可行/不可行) と可視化
# =============================
sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
final_feasible = [n for n in sink_nodes if G.nodes[n]['CV'] == 0]
final_infeasible = [n for n in sink_nodes if G.nodes[n]['CV'] > 0]
other_nodes = [n for n in nodes if n not in sink_nodes]
midle_feasible = [n for n in other_nodes if G.nodes[n]['CV'] == 0]
midle_infeasible = [n for n in other_nodes if G.nodes[n]['CV'] > 0]

plt.figure(figsize=(15, 12))

# エッジの描画
nx.draw_networkx_edges(
    G, pos, arrowstyle='->', arrowsize=15, edge_color='gray', alpha=0.5
)

# 中間ノード（その他）
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=midle_infeasible,
    node_size=50,
    node_color='skyblue',
    edgecolors='black',
)

nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=midle_feasible,
    node_size=50,
    node_color='salmon',
    edgecolors='black',
)

# 最終世代・実行可能ノード
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=final_feasible,
    node_size=50,
    node_color='red',
    edgecolors='black',
)

# 最終世代・実行不可能ノード
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=final_infeasible,
    node_size=50,
    node_color='blue',
    edgecolors='black',
    node_shape='o',
    linewidths=2,
)

ax = plt.gca()
ax.set_yscale('symlog', linthresh=1e-5)
# 「指数表記」で軸を表示したい場合は LogFormatterSciNotation などを使う
log_formatter = ticker.LogFormatterSciNotation(base=10)
ax.yaxis.set_major_formatter(log_formatter)
ax.tick_params(axis='y', which='both', labelleft=True)

ax.axis('on')
plt.ylim(bottom=-1e-5)
plt.ylim(top=1e4)
plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)

plt.show()
