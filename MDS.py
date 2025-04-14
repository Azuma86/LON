import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import matplotlib.ticker as ticker
from sklearn.manifold import MDS

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20

problem_name = 'RWMOP22'
algo = 'data'

domain_df = pd.read_csv('domain_info.csv')
# 指定した問題名の行を取得
row = domain_df.loc[domain_df['problem'] == problem_name].iloc[0]

lower = np.array([float(v) for v in row['lower'].split(",")])
upper = np.array([float(v) for v in row['upper'].split(",")])
diff = upper - lower
# =============================
# 1. CSVファイルの読み込みと前処理
# =============================
#data = pd.read_csv(f'data09-20-pre/{problem_name}_{algo}.csv')
data = pd.read_csv(f'data09-20-pre/local_search{problem_name}.csv')
con_cols = [c for c in data.columns if c.startswith('Con_')]
total = data[con_cols].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)
target_constraints = ['Con_1']
sub = data[target_constraints].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)

data['CV'] = total
data_sorted = data.sort_values(by=['ID', 'Gen'])
# =============================
# 2. グラフ構築（ノードとエッジ）
# =============================
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

# 同じ個体IDで世代が連続している場合にエッジを張る
for idx, row in data_sorted.iterrows():
    if prev_row is not None and prev_row['ID'] == row['ID'] and row['Gen'] == prev_row['Gen'] + 1:
        G.add_edge(prev_idx, idx)
    prev_row = row
    prev_idx = idx

# =============================
# 3. シンクノードの特定
# =============================
sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

# =============================
# 4. 重複ノードの統合
# =============================
vec2nodes = defaultdict(list)
for n in G.nodes():
    vec = tuple(G.nodes[n]['X'])
    vec2nodes[vec].append(n)

for vec, node_list in vec2nodes.items():
    if len(node_list) > 1:
        representative = node_list[0]
        duplicates = node_list[1:]
        for dup in duplicates:
            # dupノードへ入ってくるエッジを代表ノードへ付け替え
            for pred in list(G.predecessors(dup)):
                if pred != representative:
                    if not G.has_edge(pred, representative):
                        G.add_edge(pred, representative)
            # dupノードから出ていくエッジを代表ノードへ付け替え
            for succ in list(G.successors(dup)):
                if succ != representative:
                    if not G.has_edge(representative, succ):
                        G.add_edge(representative, succ)
            # dupノードを削除
            G.remove_node(dup)
a = nx.number_of_nodes(G)
print(a)
# =============================
# 5. MDSによる1次元配置
# =============================
nodes = list(G.nodes())
X_all = np.array([G.nodes[n]['X'] for n in nodes])
N = len(nodes)

#正規化
X_all_norm = (X_all - lower) / diff

# ノード間のユークリッド距離行列を作成
dist_matrix = pairwise_distances(X_all_norm, metric='euclidean')

epsilon = 1e-10
dist_matrix[dist_matrix < epsilon] = epsilon  # ゼロ除算回避のため

# MDS(1次元)を適用
mds = MDS(n_components=1, dissimilarity='precomputed', random_state=0,eps=1e-5,max_iter=1000)
embedding_1d = mds.fit_transform(dist_matrix)

# MDS結果(1次元)をx座標、CVをy座標とする
pos = {}
for i, n in enumerate(nodes):
    x_coord = embedding_1d[i, 0]  # 1次元MDSの座標
    cv_val = G.nodes[n]['CV']     # 制約違反量をy軸に
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
ax.set_yscale('symlog', linthresh=1e-3)
# 「指数表記」で軸を表示したい場合は LogFormatterSciNotation などを使う
log_formatter = ticker.LogFormatterSciNotation(base=10)
ax.yaxis.set_major_formatter(log_formatter)

ax.tick_params(axis='y', which='both', labelleft=True)
ax.axis('on')
plt.ylim(bottom=-1e-3)
plt.ylim(top=1e6)
plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)

plt.show()
