import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20
problem_name = 'RWMOP28'
csv_path = Path(f'/Users/azumayuki/Downloads/RWMOP/{problem_name}.csv')
df = pd.read_csv(csv_path)
use_log_cmap = False
# 決定変数 X_*, 制約 Con_* を自動検出
# 1列目が世代番号だと仮定
df = df.rename(columns={df.columns[0]: 'Gen'})
X_cols   = [c for c in df.columns if c.startswith('X')]
Con_cols = [c for c in df.columns if c.startswith('Con')]
# -------------------------------------
# 2. 制約違反量 CV の計算
# -------------------------------------
df['CV'] = df[Con_cols].apply(lambda row: np.sum(np.maximum(0, row)), axis=1)
print("a")
# プロットしたい世代リスト
selected_gens = [1000]
df_sel = df[df['Gen'].isin(selected_gens)].copy()

# Normalize を用意
norm = mcolors.Normalize(vmin=min(selected_gens), vmax=max(selected_gens))
reds  = plt.cm.Reds
blues = plt.cm.Blues

# 世代ベースの色配列（可行：赤系、不可行：青系）
colors_gen = [
    reds(norm(g))  if cv==0 else
    blues(norm(g))
    for g, cv in zip(df_sel['Gen'], df_sel['CV'])
]
# -------------------------------------
# 3. 色マップの準備
# -------------------------------------
cv = df['CV'].values
max_cv = cv.max()
# 違反量がすべて 0 の場合には blues_map の使用を避ける
if max_cv == 0:
    colors = ['red'] * len(df)
else:
    # choose normalization
    if use_log_cmap:
        # only positives for log scale
        pos = cv[cv > 0]
        vmin = pos.min() if pos.size else 1e-3
        vmax = pos.max()
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=max_cv)
    cmap = plt.colormaps['Blues']
    colors = [
        'red' if c == 0 else cmap(norm(c))
        for c in cv
    ]

print("a")
# -------------------------------------
# 4. seaborn で Scatter Plot Matrix
# -------------------------------------
sns.set(style='whitegrid', font='sans-serif', font_scale=1.2)

# PairGrid を作成
g = sns.PairGrid(df[X_cols], diag_sharey=False,height=2.5)

# 対角：ヒストグラム
g.map_diag(sns.histplot, kde=False, bins = 10, color='royalblue')

# 下三角／上三角：散布図
g.map_lower(plt.scatter, color=colors, s=1,  alpha=0.8)
# 上三角＝選択世代のみ、世代ベース色
n = len(X_cols)
for i in range(n):
    for j in range(n):
        if i < j:
            ax = g.axes[i, j]
            xcol = X_cols[j]
            ycol = X_cols[i]
            ax.scatter(
                df_sel[xcol],
                df_sel[ycol],
                c=colors_gen,
                s=1,
                alpha=0.8
            )
print("a")
"""
# 軸ラベルと目盛りの調整
for ax in g.axes.flatten():
    ax.tick_params(labelsize=8)


cbar = g.fig.colorbar(sm, ax=g.axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('Constraint Violation', fontsize=10)
# 赤（CV=0）のラベルを自分で追加
cbar.ax.plot([0.5], [0.0], color='red', marker='s', markersize=8)
cbar.ax.text(0.6, 0.0, 'Feasible (CV=0)', va='center', fontsize=8)
"""
plt.show()