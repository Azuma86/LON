import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import matplotlib.ticker as ticker

def smacof_1d(dist,
              max_iter=300,
              tol=1e-6,
              init=None,
              verbose=False):
    """
    Weighted SMACOF for 1D embedding.
    dist : (N,N) array of target distances d_ij (>0 on i!=j).
    max_iter : 最大反復回数
    tol : ストレス変化の収束閾値
    init : 初期座標 (N,) shape array. None の場合ランダム初期化。
    returns : (N,) array of 1D コーディネート x_i
    """
    N = dist.shape[0]
    # 重み行列 W_ij = 1/d_ij^2, 対角は 0
    W = 1.0 / (dist**2)
    np.fill_diagonal(W, 0.0)

    # V = diag(v_i) where v_i = sum_j w_ij
    v = W.sum(axis=1)

    # 初期配置
    if init is None:
        X = np.random.RandomState(0).randn(N)
    else:
        X = init.copy()

    old_stress = None
    for it in range(1, max_iter+1):
        # 現在の埋め込み間距離
        diff = X[:, None] - X[None, :]
        dhat = np.abs(diff)
        # ゼロ割回避
        dhat[dhat < 1e-8] = 1e-8

        # B_ij = - w_ij * d_ij / d̂_ij  (i≠j)
        B = - W * dist / dhat
        np.fill_diagonal(B, -B.sum(axis=1))

        # 更新 x_i = (1/v_i) * sum_j B_ij * x_j
        X = (B @ X) / v

        # ストレスを計算して収束判定
        stress = np.sum(W * (dist - dhat)**2)
        if verbose:
            print(f"iter {it}, stress={stress:.6e}")
        if old_stress is not None and abs(old_stress - stress) < tol:
            break
        old_stress = stress

    return X


plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20
problem_name = 'RWMOP3'
name = 'RWMOP7'
algo = 'data'
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
data = pd.read_csv(f'data09-20/{problem_name}_{algo}.csv')
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
coords_1d = smacof_1d(dist_matrix,
                      max_iter=500,
                      tol=1e-7,
                      init=None,    # または init=init_coords
                      verbose=True)

# 2) pos 辞書を再構築
pos = {
    node: (coords_1d[i], G.nodes[node]['CV'])
    for i, node in enumerate(nodes)
}

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
