import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20

problem_name = 'RWMOP22'
domain_df = pd.read_csv('domain_info.csv')

# 指定した問題名の行を取得
row = domain_df.loc[domain_df['problem'] == problem_name].iloc[0]

# lower, upper を配列化 (スペース区切りを float に変換)
lower = np.array([float(v) for v in row['lower'].split(",")])
upper = np.array([float(v) for v in row['upper'].split(",")])
diff = upper - lower
# --- CSVファイルのパスを合わせてください ---
filenameX = '/path/to/X_RWMOP23.csv'
filenameC = '/path/to/Cons_RWMOP23.csv'

X = np.genfromtxt(filenameX, delimiter=',')   # shape (N, D)
Cons = np.genfromtxt(filenameC, delimiter=',')# shape (N, C)
N, D = X.shape

# 制約違反量 g_{i,c} を \max(0, g_{i,c}) で切り取り、合計
consVal = np.sum(np.maximum(0, Cons), axis=1)  # shape (N,)

###############################################################################
# 2. 多項式突然変異 (Polynomial Mutation) に基づく遷移確信度の計算
###############################################################################

# --- パラメータ設定 (適宜調整してください) ---
delta = 3              # 平均で何次元が変異するか (例)
pi_ = delta / float(D) # 突然変異確率 π
mu_ = 20.0             # 多項式突然変異の形状パラメータ μ
epsilon = 0.95         # 近傍とみなす遷移確信度のしきい値

# --- 下限・上限を適切に定義してください ---
#   RWMOP問題の厳密な lb, ub があるなら  CSV等で読み込むのがおすすめです。
#   ここでは簡易にサンプル X の最小・最大を利用しています。
lb = X.min(axis=0)
ub = X.max(axis=0)

def poly_inverse(y_d, x_d, l_d, u_d, mu):
    """
    資料にある poly_d^{-1}(y_d) を計算する補助関数.
    y_d < x_d, = x_d, > x_d で場合分け
    """
    # 数値誤差でほぼ同じ場合は等しいとみなす
    if np.isclose(y_d, x_d):
        return 0.5
    elif y_d < x_d:
        num = ( ((u_d - x_d)/(u_d - l_d))**(mu+1)
               - ((u_d - l_d - x_d + y_d)/(u_d - l_d))**(mu+1) )
        den = 2.0*( ((u_d - x_d)/(u_d - l_d))**(mu+1) - 1.0 )
        return num/den if den != 0 else 0.5
    else:  # y_d > x_d
        num = ( ((x_d - l_d)/(u_d - l_d))**(mu+1)
               + ((u_d - l_d + x_d - y_d)/(u_d - l_d))**(mu+1)
               - 2.0 )
        den = 2.0*( ((x_d - l_d)/(u_d - l_d))**(mu+1) - 1.0 )
        return num/den if den != 0 else 0.5

def extremeness(y_d, x_d, l_d, u_d, mu):
    """
    ex_d(y_d) = 2 * | poly_d^{-1}(y_d) - 0.5 |
    x_d は poly_d^{-1}(x_d) = 0.5 を仮定
    """
    val_inv = poly_inverse(y_d, x_d, l_d, u_d, mu)
    return 2.0 * abs(val_inv - 0.5)

def transition_confidence(x_i, x_j, pi_, mu_, lb, ub, eps=1e-14):
    """
    C(x_i -> x_j) を計算する.
    x_i, x_j: 長さDのnumpy配列
    pi_, mu_: PMのパラメータ (π, μ)
    lb, ub  : 各次元の下限・上限 (配列)
    eps     : 数値的に極小値に達したら打ち切る用のしきい値
    """
    p = 1.0
    D_ = len(x_i)
    for d in range(D_):
        if np.isclose(x_i[d], x_j[d]):
            # Pr( ex_d(y_d) <= ex_d(z_d) ) = 1
            continue
        else:
            e = extremeness(x_j[d], x_i[d], lb[d], ub[d], mu_)
            # 資料式(4)の下側:  π * (1 - ex_d(y_d))
            p_d = pi_ * (1.0 - e)
            # 負になる場合は確率0とみなす (e>1.0 なら 1-e<0)
            if p_d < 0.0:
                p_d = 0.0
            p *= p_d
            if p < eps:
                break
    return p

###############################################################################
# 3. グラフ構築
###############################################################################

G = nx.DiGraph()
G.add_nodes_from(range(N))  # ノードはサンプル番号 0..N-1

# 例: 「制約違反量が小さくなる方向」にのみ辺を張る
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        c_ij = transition_confidence(X[i], X[j], pi_, mu_, lb, ub)
        if c_ij >= epsilon:
            if consVal[j] < consVal[i]:
                G.add_edge(i, j)

###############################################################################
# 4. 1次元 Kamada-Kawai レイアウト計算 → (x座標, 制約違反量) での可視化
###############################################################################

pos_1d = nx.kamada_kawai_layout(G, dim=1)  # 1次元埋め込み

# 2次元のdictに変換: x軸に 1次元埋め込み, y軸に consVal
pos_2d = {}
for i in range(N):
    x_coord = pos_1d[i][0]
    y_coord = consVal[i]
    pos_2d[i] = (x_coord, y_coord)

###############################################################################
# 5. グラフ描画
###############################################################################

plt.figure(figsize=(8, 6))

# ノードの描画
nx.draw_networkx_nodes(
    G, pos_2d,
    node_size=30,
    node_color='blue',
    alpha=0.7
)

# エッジの描画 (件数が多い場合は非常に煩雑になるので注意)
nx.draw_networkx_edges(
    G, pos_2d,
    arrowstyle='->',
    arrows=True,
    arrowsize=10,
    alpha=0.3
)

plt.title("Directed Graph for RWMOP#23 (1D K-K layout, y= sum of max(0, g_i))")
plt.xlabel("Kamada-Kawai 1D embedding")
plt.ylabel("Constraint Violation (sum of max(0, g_i))")
plt.tight_layout()
plt.show()