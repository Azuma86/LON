
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from matplotlib.ticker import LinearLocator, MultipleLocator

from KMedoids import kmedoids
from dbi import davies
from distmatlix import (
    compute_ot_matrix,
    compute_dtw_matrix,
    compute_gw_matrix,
)

# ----------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 18

sns.set(font_scale=1.2)
sns.set_style('whitegrid', {
    'grid.color': '#eeeeee',
    'grid.linewidth': 0.5
})

# Plot parameters
EDGE_LW          = 0.1  # edge line width
NODE_SIZE        = 25   # node scatter size
FINAL_NODE_SIZE  = 60   # sink node size
NODE_ALPHA       = 1  # node transparency
ARROW_LW         = 0.8  # arrow line width
ARROW_ALPHA      = 1  # arrow transparency
MARGIN           = 0.05 # axis margin fraction
EDGE_LEN_THRESH = 80
CV_THRESH_up       = 1e5
CV_THRESH_low      = 0
# Distance and clustering settings
DIST_METHOD = 'dtw'  # 'ot' | 'dtw' | 'gw'
SINKHORN_EPS = None  # for GW
MAX_CLUSTERS = 20

# Problem settings
PROBLEM_NAME = 'RWMOP22'
ALGO         = 'local31'
BASE_DIR     = Path('../data09-20-pre')
DOMAIN_PATH  = Path('domain_info.csv')


def load_problem_data(problem: str,
                      algo: str,
                      base_dir: Path) -> pd.DataFrame:
    """
    Load CSV, compute total constraint violation, and sort by ID/Gen.
    """
    csv_file = base_dir / f"{problem}_{algo}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f'File not found: {csv_file}')
    df = pd.read_csv(csv_file)
    # compute CV
    con_cols = [c for c in df.columns if c.startswith('Con_')]
    df['CV'] = df[con_cols].clip(lower=0).sum(axis=1)
    df = df.sort_values(['ID','Gen']).reset_index(drop=True)
    return df

def load_problem_final(problem: str) -> pd.DataFrame:
    """
    Load CSV, compute total constraint violation, and sort by ID/Gen.
    """
    final_csv = Path(f'/Users/azumayuki/Downloads/RWMOP/{PROBLEM_NAME}_CDP_final.csv')
    df = pd.read_csv(final_csv)
    Con_cols = [c for c in df.columns if c.startswith('Con')]
    df = df.rename(columns={df.columns[0]: 'Gen'})
    df['CV'] = df[Con_cols].clip(lower=0).sum(axis=1)

    return df


def load_domain_bounds(domain_csv: Path,
                       problem: str) -> (np.ndarray, np.ndarray):
    """
    Read domain_info.csv and return arrays of lower/upper bounds for X dimensions.
    """
    df = pd.read_csv(domain_csv)
    row = df.loc[df['problem'] == problem]
    if row.empty:
        raise KeyError(f'Problem {problem} not in domain file')
    lower = np.fromstring(row['lower'].iloc[0], sep=',')
    upper = np.fromstring(row['upper'].iloc[0], sep=',')
    return lower, upper


def compute_distance_matrix(df: pd.DataFrame,
                            method: str,
                            sinkhorn_eps=None) -> (np.ndarray, list):
    """
    Compute series distance matrix W and series IDs list.
    """
    idx_map = df.groupby('ID').indices
    series_ids = list(idx_map.keys())
    print(len(series_ids))
    X_cols = [c for c in df.columns if c.startswith('X_')]
    if method == 'ot':
        W = compute_ot_matrix(df, series_ids, idx_map, X_cols)
    elif method == 'dtw':
        W = compute_dtw_matrix(df, series_ids, idx_map, X_cols)
    elif method == 'gw':
        W = compute_gw_matrix(df, series_ids, idx_map, X_cols,
                               sinkhorn_eps=sinkhorn_eps)
    else:
        raise ValueError(f'Unknown distance method: {method}')

    return W, series_ids


def select_optimal_k(W: np.ndarray,
                     k_range: range) -> int:
    """
    Evaluate Davies–Bouldin index over k_range and return best k.
    """
    dbi_scores = []
    for k in k_range:
        labels = kmedoids(n_clusters=k, random_state=42).fit_predict(W)
        dbi = davies(W, labels)
        dbi_scores.append(dbi)
        print(f'k={k:2d} -> DBI={dbi:.4f}')
    best = k_range[np.argmin(dbi_scores)]
    print(f'Optimal k = {best}')
    plt.figure(figsize=(8, 6))
    plt.plot(list(k_range), dbi_scores, marker='o')
    plt.grid(alpha=0.3)
    li = np.linspace(2,MAX_CLUSTERS,10)
    plt.xticks(li)
    plt.ylim(1.2,2.4)
    plt.tight_layout()
    plt.show()
    return best


def select_medoids(W: np.ndarray,
                   series_ids: list,
                   k: int) -> list:
    """
    Perform k-medoids and return list of medoid series IDs.
    """
    labels = kmedoids(n_clusters=k, random_state=42).fit_predict(W)
    medoids = []
    for ci in range(k):
        members = np.where(labels == ci)[0]
        subW = W[np.ix_(members, members)]
        medoid_idx = members[np.argmin(subW.sum(axis=1))]
        medoids.append(series_ids[medoid_idx])
    print(f'Selected medoids: {medoids}')
    return medoids


def plot_transitions(x, y, **kwargs):
    ax = plt.gca()
    df_all = kwargs.pop('data')
    for sid, grp in df_all.groupby('ID'):
        grp = grp.sort_values('Gen')
        xs = grp[x.name].values
        ys = grp[y.name].values
        for i in range(len(xs)-1):
            ax.annotate('',
                        xy=(xs[i+1], ys[i+1]),
                        xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle='->',mutation_scale=5, lw=ARROW_LW, alpha=ARROW_ALPHA, color="gray"),)


def visualize_pairgrid(data: pd.DataFrame,
                       data2: pd.DataFrame,
                       lower: np.ndarray,
                       upper: np.ndarray,
                       colors_final,
                       size_final):
    """
    Visualize medoid series via seaborn PairGrid with arrows.
    """
    X_cols = [c for c in data.columns if c.startswith('X_')]

    # 1) 描画用にプロットサイズと色を用意
    data['plot_size'] = np.where(data['is_sink'], FINAL_NODE_SIZE, np.where(data['long_edge'],FINAL_NODE_SIZE,NODE_SIZE))
    data['plot_color'] = np.where(
        data['feasible'],
        'red',
        np.where(data['is_sink'], 'blue',
                 np.where(data['long_edge'],'green','skyblue')
                 )
    )

    # 2) PairGrid を組み直し
    g = sns.PairGrid(data, vars=X_cols, diag_sharey=False, height=2)

    # 対角：ヒストグラム（色だけ feasibility 反映してもOK）
    g.map_diag(sns.histplot, kde=False, bins=10, color='royalblue', edgecolor='black')

    # 下三角：scatter + 矢印
    g.map_lower(plt.scatter,
                s=data['plot_size'],
                color=data['plot_color'],
                alpha=NODE_ALPHA,
                edgecolor='black',
                linewidths=0.1)
    g.map_lower(plot_transitions, data=data)

    g.set(xlabel=None, ylabel=None)
    X_cols = [c for c in data2.columns if c.startswith('X')]
    #上三角
    n = len(X_cols)
    for i in range(n):
        for j in range(n):
            if i > j:
                ax = g.axes[i, j]
                xcol = X_cols[j]
                ycol = X_cols[i]
                ax.scatter(
                    data2[xcol],
                    data2[ycol],
                    c=colors_final,
                    s=size_final, edgecolor='black', linewidth=0.1, alpha=0.8
                )
            ax = g.axes[i, j]
            dx = (upper[j] - lower[j]) * MARGIN
            dy = (upper[i] - lower[i]) * MARGIN
            dX = np.ceil(upper[j] - lower[j]) / 4
            dY = np.ceil(upper[i] - lower[i]) / 4
            ax.set_xlim(lower[j] - dx, upper[j] + dx)
            ax.set_ylim(lower[i] - dy, upper[i] + dy)
            ax.xaxis.set_major_locator(MultipleLocator(dX))
            ax.yaxis.set_major_locator(MultipleLocator(dY))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # データロード
    df_ = load_problem_data(PROBLEM_NAME, ALGO, BASE_DIR)
    print(df_.shape)
    fi_df = load_problem_final(PROBLEM_NAME)

    X_cols = [c for c in df_.columns if c.startswith('X_')]
    diff = df_.groupby('ID')[X_cols].diff(-1)  # (次行 － 現行)
    df_['edge_len'] = np.sqrt((diff ** 2).sum(axis=1))  # 末尾行は NaN
    # 閾値より大きいものを long_edge フラグとして立てる
    # 1) 行フラグを作る
    df_['long_edge'] = (df_['edge_len'] > EDGE_LEN_THRESH).fillna(False)
    cv_in_range = df_['CV'].between(CV_THRESH_low, CV_THRESH_up, inclusive='both')
    df_['trigger'] = (df_['long_edge'] & cv_in_range).fillna(False)

    # 2) 「各IDで少なくとも1回 triggerがTrue」を満たすIDを取得
    ids = df_.groupby('ID')['trigger'].any()
    ids = ids[ids].index  # True の ID だけ

    # 3) そのIDに属する全行を抽出（系列ごと丸ごと）
    raw_df = df_[df_['ID'].isin(ids)].copy().reset_index(drop=True)
    print(raw_df.shape)

    size_final = [
        FINAL_NODE_SIZE if c == 0 else
        NODE_SIZE
        for c in fi_df['CV'].values
    ]
    lower_bounds, upper_bounds = load_domain_bounds(DOMAIN_PATH, PROBLEM_NAME)
    print(lower_bounds)
    print(upper_bounds)
    # 距離行列と系列 ID
    W, series_list = compute_distance_matrix(raw_df, DIST_METHOD, SINKHORN_EPS)

    # 最適クラスタ数 k の選定
    k_values = range(2, MAX_CLUSTERS+1)
    best_k = select_optimal_k(W, k_values)

    # メドイド選択
    medoid_ids = select_medoids(W, series_list, best_k)
    medoid_df = raw_df[raw_df['ID'].isin(medoid_ids)].copy()

    # Sink 判定
    G = nx.DiGraph()
    for idx, row in raw_df.iterrows():
        G.add_node(idx)
    for idx, row in raw_df.iterrows():
        prev_idx = idx - 1
        if prev_idx >= 0 and raw_df.at[prev_idx, 'ID'] == row['ID']:
            G.add_edge(prev_idx, idx)
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    medoid_df['is_sink'] = medoid_df.index.isin(sinks)
    medoid_df['feasible'] = medoid_df['CV'] == 0

    dec = 10
    X_cols2 = [c for c in fi_df.columns if c.startswith('X')]
    long_keys = (
        df_.loc[df_['long_edge'], X_cols]
        .round(dec)
        .astype(str)
        .agg('|'.join, axis=1)
        .unique()
    )

    # 2) fi_df でも同じキーを生成
    fi_keys = (
        fi_df[X_cols2]
        .round(dec)
        .astype(str)
        .agg('|'.join, axis=1)
    )

    # 3) fi_df['long_edge'] フラグを立てる
    fi_df['long_edge'] = fi_keys.isin(long_keys)

    colors_final = np.where(
        fi_df['long_edge'], 'green',
        np.where(fi_df['CV'] == 0, 'red', 'skyblue')
    )
    size_final = np.where(
        fi_df['long_edge'] | (fi_df['CV'] == 0),
        FINAL_NODE_SIZE, NODE_SIZE
    )
    visualize_pairgrid(medoid_df,fi_df, lower_bounds, upper_bounds, colors_final, size_final)

