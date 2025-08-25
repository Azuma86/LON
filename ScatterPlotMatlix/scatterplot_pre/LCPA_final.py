import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 18

problem_name = 'RWMOP30'
algo         = 'local31'

# 既存：全世代 CSV
csv_path  = f"/Users/azumayuki/Documents/LONs/data09-20-pre/{problem_name}_{algo}.csv"
# 追加：最終世代 CSV（例： *_final.csv という命名と仮定）
final_csv = f"/Users/azumayuki/Downloads/RWMOP/{problem_name}_CDP_final.csv"

# ============ 1. 全世代データ読み込み ============
data = pd.read_csv(csv_path)
X_cols   = [c for c in data.columns if c.startswith('X_')]
con_cols = [c for c in data.columns if c.startswith('Con_')]
n_dim    = len(X_cols)
# CV 計算
data['CV'] = data[con_cols].apply(lambda r: np.maximum(0, r).sum(), axis=1)
data_sorted = data.sort_values(['ID', 'Gen']).reset_index(drop=True)

# ============ 2. 最終世代データ読み込み ============
final_data = pd.read_csv(final_csv)
X_cols2   = [c for c in final_data.columns if c.startswith('X')]
final_X = final_data[X_cols2].values        # (N_final × D)
# 文字列比較を避けるため、少数誤差に備えて丸め込み
final_keys = {tuple(np.round(row, 8)) for row in final_X}

# ============ 3. グラフ構築（同じ） ============
G = nx.DiGraph()
for idx, row in data_sorted.iterrows():
    G.add_node(idx,
               X=row[X_cols].values.astype(float),
               CV=row['CV'],
               is_final = tuple(np.round(row[X_cols].values,8)) in final_keys)

    if idx>0:
        prev = data_sorted.iloc[idx-1]
        if (prev['ID']==row['ID']) and (row['Gen']==prev['Gen']+1):
            G.add_edge(idx-1, idx)

# ============ 4. エッジ情報抽出 ============
edge_info = []
edge_color = []   # ← 追加：色格納
edge_size = []
for u,v in G.edges():
    x_u = G.nodes[u]['X']
    cv  = G.nodes[u]['CV']
    dist = np.linalg.norm(x_u - G.nodes[v]['X'])
    edge_info.append((cv, dist))

    # プロット色を決定（始点が最終世代の個体なら赤、それ以外は青）
    edge_color.append('red' if G.nodes[u]['is_final'] else 'tab:blue')
    edge_size.append(30 if G.nodes[u]['is_final'] else 15)

edge_info = np.asarray(edge_info)
CV_plot   = edge_info[:,0]
len_plot  = edge_info[:,1]

# ============ 5. プロット ============
plt.figure(figsize=(8,6))
plt.scatter(len_plot, CV_plot,
            s=edge_size, alpha=0.8, c=edge_color, edgecolors='k', linewidths=0.2)

plt.xlabel("Edge Length")
plt.ylabel("CV")
plt.yscale('log'); #plt.xlim(-0.5,150); #plt.ylim(1e-6,1e3)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()