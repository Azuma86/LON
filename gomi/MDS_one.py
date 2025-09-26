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

problem_name = 'RWMOP24'

domain_df = pd.read_csv('../domain_info.csv')
# 指定した問題名の行を取得
row = domain_df.loc[domain_df['problem'] == problem_name].iloc[0]

lower = np.array([float(v) for v in row['lower'].split(",")])
upper = np.array([float(v) for v in row['upper'].split(",")])
diff = upper - lower

# =============================
# 1. CSVファイルの読み込みと前処理
# =============================
data = pd.read_csv(f'data09-20-pre/{problem_name}_data.csv')
con_cols = [c for c in data.columns if c.startswith('Con_')]

# 全制約違反量 (合計)
total = data[con_cols].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)

# 指定した制約だけ合計 (例: Con_2)
target_constraints = ['Con_1']
sub = data[target_constraints].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)

data['CV'] = total  # 全制約合計
data['CV_sub'] = sub    # 指定制約(Con_2) の合計

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
               CV=row['CV'],
               CV_sub=row['CV_sub'])

prev_row = None
prev_idx = None

# 同じ個体IDで世代が連続している場合にエッジを張る
for idx, row in data_sorted.iterrows():
    if prev_row is not None and prev_row['ID'] == row['ID'] and row['Gen'] == prev_row['Gen'] + 1:
        G.add_edge(prev_idx, idx)
    prev_row = row
    prev_idx = idx

# =============================
# 3. 重複ノードの統合
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
            G.remove_node(dup)

# =============================
# 4. MDSによる1次元配置
# =============================
nodes = list(G.nodes())
X_all = np.array([G.nodes[n]['X'] for n in nodes])
cv_total = [G.nodes[n]['CV'] for n in nodes]     # 縦軸に使う値
cv_sub   = [G.nodes[n]['CV_sub'] for n in nodes] # カラー表示用
N = len(nodes)

# 正規化
X_all_norm = (X_all - lower) / diff

dist_matrix = pairwise_distances(X_all_norm, metric='euclidean')
dist_matrix[dist_matrix < 1e-10] = 1e-10

mds = MDS(n_components=1, dissimilarity='precomputed', random_state=0)
embedding_1d = mds.fit_transform(dist_matrix)

pos = {}
for i, n in enumerate(nodes):
    x_coord = embedding_1d[i, 0]
    y_coord = cv_total[i]  # 全体CVをY軸
    pos[n] = (x_coord, y_coord)

zero_sub_nodes   = [n for n in nodes if G.nodes[n]['CV_sub'] == 0]
nonzero_sub_nodes= [n for n in nodes if G.nodes[n]['CV_sub'] > 0]

cv_sub_nonzero   = [G.nodes[n]['CV_sub'] for n in nonzero_sub_nodes]

fig, ax = plt.subplots(figsize=(15,12))

# 1) CV_sub==0 のノードを赤で描画
nx.draw_networkx_nodes(
    G, pos,
    nodelist=zero_sub_nodes,
    node_size=50,
    node_color='red',
    edgecolors='black',
    linewidths=1.0,
    ax=ax
)

# 2) CV_sub>0 のノードをBluesカラーマップで描画
if len(nonzero_sub_nodes) > 0:
    # vmin/vmaxを決める (自動でもよいが明示例)
    vmin = min(cv_sub_nonzero)
    vmax = max(cv_sub_nonzero)

    pc = nx.draw_networkx_nodes(
        G, pos,
        nodelist=nonzero_sub_nodes,
        node_size=50,
        node_color=cv_sub_nonzero,
        cmap='Blues',
        edgecolors='black',
        linewidths=1.0,
        alpha=0.8,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    # カラーバーを付ける
    cb = fig.colorbar(pc, ax=ax, orientation='vertical')

# エッジ描画
nx.draw_networkx_edges(
    G, pos,
    ax=ax,
    arrowstyle='->', arrowsize=15, edge_color='gray', alpha=0.5
)

# y軸をログスケールにする
ax.set_yscale('symlog', linthresh=1e-5)
ax.set_ylim(bottom=-1e-6, top=1e5)
plt.show()