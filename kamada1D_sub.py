import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import matplotlib.ticker as ticker

# 描画設定
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20

# ========================================
# 1. 問題情報の読み込み
# ========================================
problem_name = 'RWMOP26'
algo = 'data'
domain_df = pd.read_csv('domain_info.csv')
row = domain_df.loc[domain_df['problem'] == problem_name].iloc[0]

# lower, upper をカンマ区切りの文字列から数値配列に変換
lower = np.array([float(v) for v in row['lower'].split(",")])
upper = np.array([float(v) for v in row['upper'].split(",")])
diff = upper - lower

# ========================================
# 2. データ読み込みと前処理
# ========================================
data = pd.read_csv(f'data09-20/{problem_name}_{algo}.csv')
con_cols = [c for c in data.columns if c.startswith('Con_')]
X_cols = [c for c in data.columns if c.startswith('X_')]


# ========================================
# 関数定義
# ========================================

def create_graph(df, X_cols):
    """
    ソート済みDataFrameから、各行の情報（Gen, ID, X, CV）を持つ有向グラフを作成し、
    同一個体の連続する世代間にエッジを追加する。
    """
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        G.add_node(idx,
                   Gen=row['Gen'],
                   ID=row['ID'],
                   X=row[X_cols].values,
                   CV=row['CV'])
    prev_row, prev_idx = None, None
    for idx, row in df.iterrows():
        if prev_row is not None and prev_row['ID'] == row['ID'] and row['Gen'] == prev_row['Gen'] + 1:
            G.add_edge(prev_idx, idx)
        prev_row, prev_idx = row, idx
    return G


def merge_duplicate_nodes(G):
    """
    ノード属性 'X' が同じノード群を特定し、最初のノードを代表として重複ノードを統合する。
    重複ノードへの入出エッジは代表ノードに付け替え、重複ノードは削除する。
    """
    vec2nodes = defaultdict(list)
    for n in G.nodes():
        vec = tuple(G.nodes[n]['X'])
        vec2nodes[vec].append(n)

    for vec, nodes in vec2nodes.items():
        if len(nodes) > 1:
            rep = nodes[0]
            for dup in nodes[1:]:
                # 入るエッジを付け替え
                for pred in list(G.predecessors(dup)):
                    if pred != rep and not G.has_edge(pred, rep):
                        G.add_edge(pred, rep)
                # 出るエッジを付け替え
                for succ in list(G.successors(dup)):
                    if succ != rep and not G.has_edge(rep, succ):
                        G.add_edge(rep, succ)
                G.remove_node(dup)
    return G


def compute_layout(G, lower, diff):
    """
    各ノードの意思決定変数 (X) を正規化し、
    ノード間のユークリッド距離行列を計算の上、Kamada–Kawai レイアウト（1次元）を得る。
    得られた x座標と、各ノードのCV値を用いて (x, y) の座標を返す。
    """
    nodes = list(G.nodes())
    X_all = np.array([G.nodes[n]['X'] for n in nodes])
    X_norm = (X_all - lower) / diff

    dist_matrix = pairwise_distances(X_norm, metric='euclidean')
    epsilon = 1e-10
    dist_matrix[dist_matrix < epsilon] = epsilon

    # 辞書形式に変換
    dist_dict = {ni: {nj: dist_matrix[i, j] for j, nj in enumerate(nodes)}
                 for i, ni in enumerate(nodes)}
    pos_1d = nx.kamada_kawai_layout(G, dist=dist_dict, dim=1)
    # x座標はレイアウト結果、y座標は制約違反CV
    pos = {n: (pos_1d[n][0], G.nodes[n]['CV']) for n in nodes}
    return pos


def classify_nodes(G):
    """
    グラフ内のノードを、sinkノード（後続エッジがない）と中間ノードに分類し、
    さらにCVの値（0：可行、>0：不可行）で分類する。
    """
    sink_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    others = [n for n in G.nodes() if n not in sink_nodes]

    final_feasible = [n for n in sink_nodes if G.nodes[n]['CV'] == 0]
    final_infeasible = [n for n in sink_nodes if G.nodes[n]['CV'] > 0]
    midle_feasible = [n for n in others if G.nodes[n]['CV'] == 0]
    midle_infeasible = [n for n in others if G.nodes[n]['CV'] > 0]

    return final_feasible, final_infeasible, midle_feasible, midle_infeasible


def plot_graph(G, pos, final_feasible, final_infeasible, midle_feasible, midle_infeasible):
    """
    ノードとエッジを描画する。y軸は対数スケール（symlog）で表示する。
    """
    plt.figure(figsize=(15, 12))

    # エッジ描画
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray', alpha=0.5)

    # 中間ノード
    nx.draw_networkx_nodes(G, pos, nodelist=midle_infeasible,
                           node_size=50, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=midle_feasible,
                           node_size=50, node_color='salmon', edgecolors='black')
    # 最終世代ノード
    nx.draw_networkx_nodes(G, pos, nodelist=final_feasible,
                           node_size=50, node_color='red', edgecolors='black')
    nx.draw_networkx_nodes(G, pos, nodelist=final_infeasible,
                           node_size=50, node_color='blue', edgecolors='black',
                           node_shape='o', linewidths=2)

    ax = plt.gca()
    ax.set_yscale('symlog', linthresh=1e-6)
    log_formatter = ticker.LogFormatterSciNotation(base=10)
    ax.yaxis.set_major_formatter(log_formatter)
    ax.tick_params(axis='y', which='both', labelleft=True)
    ax.axis('on')
    plt.ylim(bottom=-1e-7, top=1e1)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.05)
    plt.show()


# ========================================
# 3. 各制約に対して処理実行
# ========================================
for con_num in range(1, len(con_cols) + 1):
    print(f"Processing constraint: Con_{con_num}")

    # 特定の制約値を用いる場合は、ここで CV を上書きする
    target_constraint = f'Con_{con_num}'
    # 各行について、target_constraintの違反値（0以上のみの和）を計算
    data['CV'] = data[target_constraint].apply(lambda x: np.sum(np.maximum(0, x)))

    # ※ 全制約の合計を使う場合は以下のように変更可能：
    # data['CV'] = data[con_cols].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)

    # データをID, Genでソートしてグラフ作成
    data_sorted = data.sort_values(by=['ID', 'Gen'])
    G = create_graph(data_sorted, X_cols)
    G = merge_duplicate_nodes(G)

    pos = compute_layout(G, lower, diff)
    final_feasible, final_infeasible, midle_feasible, midle_infeasible = classify_nodes(G)

    plot_graph(G, pos, final_feasible, final_infeasible, midle_feasible, midle_infeasible)