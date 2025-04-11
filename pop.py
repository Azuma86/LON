import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D描画に必要

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20
problem_name = 'RWMOP26'
algo = 'data'
# 1) CSVファイル読み込み
data = pd.read_csv(f'data09-20/{problem_name}_{algo}.csv')

# 2) 制約違反量CVを計算 (Con_で始まる列を合計)
con_cols = [c for c in data.columns if c.startswith('Con_')]
data['CV'] = data[con_cols].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)

# 3) 決定変数の列名を取得 (ここでは X_0, X_1, X_2 の3変数を想定)
X_cols = [c for c in data.columns if c.startswith('X_')]
# shape = (N, 3) になっている想定
X = data[X_cols].values

# 4) 実行可能/不可行 の判定
feasible_mask   = (data['CV'] == 0)
infeasible_mask = (data['CV'] >  0)

# 5) 3Dプロットの準備
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 5-1) 実行可能解 (CV=0) を赤で表示
ax.scatter(
    X[feasible_mask, 0],
    X[feasible_mask, 1],
    X[feasible_mask, 2],
    c='red',
    marker='o',
    edgecolors='black',
    s=40,
    label='Feasible (CV=0)'
)

# 5-2) 不可行解 (CV>0) をCVに応じて青のグラデーションで表示
scatter_infeasible = ax.scatter(
    X[infeasible_mask, 0],
    X[infeasible_mask, 1],
    X[infeasible_mask, 2],
    c=data.loc[infeasible_mask, 'CV'],
    cmap='Blues',
    marker='o',
    edgecolors='black',
    s=40,
    label='Infeasible (CV>0)'
)

# 5-3) カラーバー (不可行解のCV値用)
#   ※fig.colorbar() を使うと2Dのカラーバーが表示されます
cbar = fig.colorbar(scatter_infeasible, ax=ax, shrink=0.8, pad=0.1)


#ax.view_init(elev=90, azim=30)
plt.tight_layout()
plt.show()