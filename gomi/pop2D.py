import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20

problem_name = 'RWMOP0'
algo = 'data'

# 1) CSVファイル読み込み
data = pd.read_csv(f'data09-20-pre/{problem_name}_{algo}.csv')

# 2) 制約違反量CVを計算 (Con_で始まる列を合計)
con_cols = [c for c in data.columns if c.startswith('Con_')]
data['CV'] = data[con_cols].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)

# 3) 決定変数の列名を取得（X_0, X_1）
X_cols = [c for c in data.columns if c.startswith('X_')]
if len(X_cols) != 2:
    raise ValueError(f"2次元の決定変数が必要ですが、{len(X_cols)}列見つかりました: {X_cols}")
X = data[X_cols].values

# 4) 実行可能 / 不可行 のマスク
feasible_mask   = (data['CV'] == 0)
infeasible_mask = (data['CV'] >  0)

# 5) 2Dプロットの準備
fig, ax = plt.subplots(figsize=(10, 8))

# 5-1) 実行可能解 (CV=0) を赤で表示
ax.scatter(
    X[feasible_mask, 0],
    X[feasible_mask, 1],
    c='red',
    marker='o',
    edgecolors='black',
    s=100,
    label='Feasible (CV=0)'
)

# 5-2) 不可行解 (CV>0) をCVに応じて青のグラデーションで表示
scatter_infeasible = ax.scatter(
    X[infeasible_mask, 0],
    X[infeasible_mask, 1],
    c=data.loc[infeasible_mask, 'CV'],
    cmap='Blues',
    marker='o',
    edgecolors='black',
    s=40,
    label='Infeasible (CV>0)'
)

# 5-3) カラーバー
cbar = fig.colorbar(scatter_infeasible, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Constraint Violation (CV)')

# 軸ラベルと凡例

ax.grid(True)

plt.tight_layout()
plt.show()