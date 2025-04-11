import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.ticker as ticker
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import cg  # 共役勾配法をインポート
import time
# プロット用のフォント設定
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20


def stress_majorization_1d_sparse(dist_matrix, pairs, max_iter=100, tol=1e-4, x0=None):
    """
    1次元における疎なStress Majorizationアルゴリズム（論文 "Graph Drawing by Stress Majorization" に基づく）
    ・入力:
      - dist_matrix: 各ノード間の理想距離行列（ここでは正規化済みのユークリッド距離など）
      - pairs: 疎なペア集合（ピボット＋k近傍に基づくペア）
      - max_iter, tol: 反復の最大回数，収束判定の閾値
      - x0: 初期解（与えなければ乱数初期化）
    """
    N = dist_matrix.shape[0]
    epsilon = 1e-10
    # ゼロ除算防止のための下限
    dist_matrix = np.maximum(dist_matrix, epsilon)
    if x0 is None:
        x = np.random.rand(N)
    else:
        x = x0.copy()

    # --- 重みと理想距離の辞書作成 (論文の式(1)に対応) ---
    W = {}
    D = {}
    for i, j in pairs:
        dij = dist_matrix[i, j]
        wij = 1.0 / (dij ** 1 + epsilon)
        W[(i, j)] = wij
        D[(i, j)] = dij

    # --- 定数ラプラシアン Lw の事前計算 (論文の式(4)) ---
    Lw = np.zeros((N, N))
    for (i, j), w in W.items():
        Lw[i, j] -= w
        Lw[j, i] -= w
        Lw[i, i] += w
        Lw[j, j] += w

    # --- 反復によるストレス最小化 ---
    for iteration in range(max_iter):
        # 現在の配置 x に基づきB行列（微分項に対応）を更新
        B = np.zeros((N, N))
        for i, j in pairs:
            diff = x[i] - x[j]
            norm_val = np.abs(diff) + epsilon
            w = W[(i, j)]
            d = D[(i, j)]
            # 各ペアの勾配項：ノード j からの "投票"として b_ij = w * d * sign(diff)
            b_ij = w * d * diff / norm_val
            B[i, j] -= b_ij
            B[j, i] += b_ij
            B[i, i] += b_ij
            B[j, j] -= b_ij

        # 線形システムの解法：Lw は定数，B は x に依存するため，式 Lw * x_new = B * x を解く
        #x_new = spsolve(csr_matrix(Lw), B @ x)
        x_new, info = cg(csr_matrix(Lw), B @ x, atol=tol)
        # 翻訳不変性除去のため，新しい配置から平均値を引く
        x_new = x_new - np.mean(x_new)

        # stress の計算（論文の式(1)）
        stress = 0.0
        for i, j in pairs:
            stress += 0.5 * W[(i, j)] * (np.abs(x_new[i] - x_new[j]) - D[(i, j)]) ** 2

        if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + epsilon) < tol:
            print(f"Converged after {iteration + 1} iterations, stress = {stress:.4e}")
            return x_new
        x = x_new

    print(f"Max iterations reached, final stress = {stress:.4e}")
    return x


def make_sparse_pairs(X, num_pivots=10, k_neigh=5):
    """
    ピボットベース＋k近傍により疎なノードペア集合を作成する関数
    （論文セクション4 "Sparse Stress Functions" に準拠）
    """
    N = X.shape[0]
    pivots_idx = np.random.choice(N, num_pivots, replace=False)
    pairs = set()

    # ① ピボットとその他の全ノードとのペア
    for pi in pivots_idx:
        for j in range(N):
            if pi != j:
                pairs.add((min(pi, j), max(pi, j)))

    # ② 各ノードの k-近傍同士のペアを追加
    nn = NearestNeighbors(n_neighbors=k_neigh + 1).fit(X)
    distances, indices = nn.kneighbors(X)
    for i in range(N):
        for j in indices[i][1:]:
            pairs.add((min(i, j), max(i, j)))

    return list(pairs)


def stress_majorization_1d(dist_matrix, max_iter=1000, tol=1e-4):
    """
    全ペアを使用する従来型の1次元Stress Majorization（比較用）
    """
    N = dist_matrix.shape[0]
    epsilon = 1e-10
    dist_matrix = np.maximum(dist_matrix, epsilon)
    W = 1.0 / dist_matrix ** 1
    #W = np.ones(dist_matrix.shape)
    Lw = np.diag(W.sum(axis=1)) - W
    x = np.random.rand(N)
    x = x - np.mean(x)
    stress_prev = np.inf
    for iteration in range(max_iter):
        delta = W * dist_matrix
        diff = np.subtract.outer(x, x)
        norm_val = np.maximum(np.abs(diff), epsilon)
        B = delta * diff / norm_val
        row_sum = B.sum(axis=1)
        Lz = np.diag(row_sum) - B
        #x_new = spsolve(csr_matrix(Lw), Lz @ x)
        x_new, info = cg(csr_matrix(Lw), Lz @ x, atol=tol)
        x_new = x_new - np.mean(x_new)
        norm_diff = np.abs(np.subtract.outer(x_new, x_new))
        stress = np.sum(W * (norm_diff - dist_matrix) ** 2) / 2
        if (stress_prev - stress) / stress_prev < tol:
            print(f"Converged after {iteration + 1} iterations, stress = {stress:.4e}")
            return x_new
        x = x_new
        stress_prev = stress
    print(f"Max iterations reached, final stress = {stress:.4e}")
    return x_new

start = time.time()
# --- データ読み込みと前処理 ---
problem_name = 'RWMOP22'
algo = 'data'
domain_df = pd.read_csv('domain_info.csv')
row = domain_df.loc[domain_df['problem'] == problem_name].iloc[0]
lower = np.array([float(v) for v in row['lower'].split(",")])
upper = np.array([float(v) for v in row['upper'].split(",")])
diff = upper - lower
#data = pd.read_csv(f'data09-20/{problem_name}_{algo}.csv')
data = pd.read_csv(f'data09-20-pre/local_search{problem_name}.csv')

# 各制約違反値の計算（複数のCon_がある場合は0以上の和をとる）
con_cols = [c for c in data.columns if c.startswith('Con_')]
total = data[con_cols].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)
data['CV'] = total
data_sorted = data.sort_values(by=['ID', 'Gen'])

# --- グラフ構築 ---
G = nx.DiGraph()
X_cols = [c for c in data.columns if c.startswith('X_')]
for idx, row in data_sorted.iterrows():
    G.add_node(idx, Gen=row['Gen'], ID=row['ID'], X=row[X_cols].values, CV=row['CV'])
prev_row = None
prev_idx = None
for idx, row in data_sorted.iterrows():
    if prev_row is not None and prev_row['ID'] == row['ID'] and row['Gen'] == prev_row['Gen'] + 1:
        G.add_edge(prev_idx, idx)
    prev_row = row
    prev_idx = idx

# --- 重複ノードの統合 ---
from collections import defaultdict

vec2nodes = defaultdict(list)
for n in G.nodes():
    vec = tuple(G.nodes[n]['X'])
    vec2nodes[vec].append(n)
for vec, node_list in vec2nodes.items():
    if len(node_list) > 1:
        representative = node_list[0]
        for dup in node_list[1:]:
            for pred in list(G.predecessors(dup)):
                if pred != representative and not G.has_edge(pred, representative):
                    G.add_edge(pred, representative)
            for succ in list(G.successors(dup)):
                if succ != representative and not G.has_edge(representative, succ):
                    G.add_edge(representative, succ)
            G.remove_node(dup)

nodes = list(G.nodes())
X_all = np.array([G.nodes[n]['X'] for n in nodes])
X_all_norm = (X_all - lower) / diff
# --- 設計変数に基づくユークリッド距離行列 ---
dist_matrix = pairwise_distances(X_all_norm, metric='euclidean')
'''
# --- グラフ構造に基づく隣接集合の作成 ---
# ここでは、グラフ G の全ノードをリストにして一貫して扱います
nodes = list(G.nodes())
neighbors = {node: set(G.predecessors(node)) | set(G.successors(node)) for node in nodes}

# --- 重み付き距離行列の計算 ---
num_nodes = len(nodes)
weighted_dist = np.zeros((num_nodes, num_nodes))
for i, node_i in enumerate(nodes):
    for j in range(i + 1, num_nodes):
        # ノードi, jそれぞれの隣接集合を取得
        ni = neighbors[node_i]
        nj = neighbors[nodes[j]]
        # 和集合と積集合のサイズを計算
        union_size = len(ni | nj)
        inter_size = len(ni & nj)
        # 距離は和集合の要素数から積集合の要素数を引いた値
        d = union_size - inter_size
        # ゼロ距離になった場合は小さな値に置換（数値計算上の安定性のため）
        if d == 0:
            d = 1e-6
        # 対称性を持たせて距離行列に代入
        weighted_dist[i, j] = d
        weighted_dist[j, i] = d# （同じ計算が重複していた箇所は整理済み）
'''
# --- 疎なペアリストの作成 ---
#sparse_pairs = make_sparse_pairs(X_all_norm, num_pivots=20, k_neigh=5)

# --- Stress Majorization で1次元座標の決定 ---
# ここではユークリッド距離行列 dist_matrix を用いているが，
# 必要に応じ weighted_dist への切替も可能
#x_coords = stress_majorization_1d_sparse(dist_matrix, sparse_pairs)
x_coords = stress_majorization_1d(dist_matrix)

# --- 可視化 ---
pos = {n: (x_coords[i], G.nodes[n]['CV']) for i, n in enumerate(nodes)}

# ノードの分類
sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
final_feasible = [n for n in sink_nodes if G.nodes[n]['CV'] == 0]
final_infeasible = [n for n in sink_nodes if G.nodes[n]['CV'] > 0]
other_nodes = [n for n in nodes if n not in sink_nodes]
midle_feasible = [n for n in other_nodes if G.nodes[n]['CV'] == 0]
midle_infeasible = [n for n in other_nodes if G.nodes[n]['CV'] > 0]

plt.figure(figsize=(15, 12))
nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray', alpha=0.5)
nx.draw_networkx_nodes(G, pos, nodelist=midle_infeasible, node_size=50, node_color='skyblue', edgecolors='black')
nx.draw_networkx_nodes(G, pos, nodelist=midle_feasible, node_size=50, node_color='salmon', edgecolors='black')
nx.draw_networkx_nodes(G, pos, nodelist=final_feasible, node_size=100, node_color='red', edgecolors='black')
nx.draw_networkx_nodes(G, pos, nodelist=final_infeasible, node_size=100, node_color='blue', edgecolors='black',
                       node_shape='o', linewidths=2)

ax = plt.gca()
ax.set_yscale('symlog', linthresh=1e-6)
log_formatter = ticker.LogFormatterSciNotation(base=10)
ax.yaxis.set_major_formatter(log_formatter)
ax.tick_params(axis='y', which='both', labelleft=True)
ax.axis('on')
plt.ylim(bottom=-1e-6, top=1e5)
plt.show()
end = time.time()  # 現在時刻（処理完了後）を取得
time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
print(time_diff)  # 処理にかかった時間データを使用