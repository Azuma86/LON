import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import spsolve
import matplotlib.ticker as ticker
from sklearn.neighbors import NearestNeighbors
import time
from random import random
from networkx import floyd_warshall_numpy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
# プロット用のフォント設定
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20


def inv(x):
    if x < 1e-5:
        return 0.0
    return 1 / x


def stress(X, D):
    """
    X: (n, 1) 配列、各ノードの1次元座標
    D: (n, n) ターゲット距離行列
    """
    n = len(X)
    s = 0.0
    for i in range(n):
        x_i = X[i, 0]  # 1次元なのでスカラー
        for j in range(i):
            x_j = X[j, 0]
            d = abs(x_i - x_j) - D[i, j]
            s += d * d
    return s


def stress_majorization(graph):
    epsilon = 1e-10
    n = graph.number_of_nodes()

    # ノード間の最短経路距離行列 D を算出
    D = floyd_warshall_numpy(graph)

    # 重み行列 w, および補正項 delta を計算
    w = np.zeros((n, n))
    delta = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            w[i, j] = w[j, i] = D[i, j] ** -2
            delta[i, j] = delta[j, i] = w[i, j] * D[i, j]

    # 初期配置 Z (1次元配列: shape (n, 1))
    Z = np.random.rand(n, 1)
    # ノード0を (0,) に固定（平行移動不変性の除去）
    Z[0, 0] = 0.0

    # 重み付きラプラシアン L_w の計算
    L_w = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            L_w[i, j] = L_w[j, i] = -w[i, j]
    for i in range(n):
        L_w[i, i] = -sum(L_w[i, :])

    # 初期ストレスの評価
    e0 = stress(Z, D)

    # 反復更新
    while True:
        # 補助ラプラシアン L_Z の構築
        L_Z = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                # 1次元の場合、Z[i]-Z[j] はスカラーなので abs() で距離を計算
                L_Z[i, j] = L_Z[j, i] = -delta[i, j] * inv(abs(Z[i, 0] - Z[j, 0]))
        for i in range(n):
            L_Z[i, i] = -sum(L_Z[i, :])

        # 更新：ノード0は固定なので、ノード1以降の座標のみ更新
        # 連立方程式 L_w_reduced * x_new = (L_Z @ Z) から解く
        rhs = (L_Z @ Z[:, 0])[1:]
        x_new_reduced, info = cg(L_w[1:, 1:], rhs, atol=epsilon)
        if info != 0:
            print(f"CG did not converge, info = {info}")
        Z[1:, 0] = x_new_reduced

        # 新たなストレスを計算して収束判定
        e = stress(Z, D)
        if (e0 - e) / e0 < epsilon:
            break
        e0 = e

    return Z

def stress_majorization_1d(dist_matrix, max_iter=1000, tol=1e-4):
    """
    全ペアを使用する従来型の1次元Stress Majorization（比較用）
    """
    N = dist_matrix.shape[0]
    epsilon = 1e-20
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
problem_name = 'RWMOP6'
algo = 'data'
domain_df = pd.read_csv('domain_info.csv')
row = domain_df.loc[domain_df['problem'] == problem_name].iloc[0]
lower = np.array([float(v) for v in row['lower'].split(",")])
upper = np.array([float(v) for v in row['upper'].split(",")])
diff = upper - lower
data = pd.read_csv(f'data09-20/{problem_name}_{algo}.csv')
#data = pd.read_csv(f'data09-20-pre/local_search{problem_name}.csv')

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
a = nx.number_of_nodes(G)
print(a)

X_all = np.array([G.nodes[n]['X'] for n in nodes])
X_all_norm = (X_all - lower) / diff
# --- 設計変数に基づくユークリッド距離行列 ---
dist_matrix = pairwise_distances(X_all_norm, metric='euclidean')
#x_coords = stress_majorization_1d(dist_matrix)
x_coords = stress_majorization(G)

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