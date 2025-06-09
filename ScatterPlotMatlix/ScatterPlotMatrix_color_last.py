import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20

# -------------------------------------
# 1. データ読み込み & 前処理
# -------------------------------------
# （1）すべての世代が入ったファイル
problem_name = 'RWMOP28'
csv_all = Path(f'/Users/azumayuki/Downloads/RWMOP/{problem_name}_CVM.csv')
df_all = pd.read_csv(csv_all)
use_log_cmap = False
# 1列目を 'Gen' と仮定してリネーム
df_all = df_all.rename(columns={df_all.columns[0]: 'Gen'})
X_cols = [c for c in df_all.columns if c.startswith('X')]
Con_cols = [c for c in df_all.columns if c.startswith('Con')]

# CV（制約違反量）を計算
df_all['CV'] = df_all[Con_cols].clip(lower=0).sum(axis=1)

# -------------------------------------
# （2）最終世代のみが入ったファイル
final_csv = Path(f'/Users/azumayuki/Downloads/RWMOP/{problem_name}_CDP_final.csv')
df_final = pd.read_csv(final_csv)

# もし最終世代ファイルにも Gen 列があるなら同様に取り扱っておく
if df_final.columns[0] != 'Gen':
    df_final = df_final.rename(columns={df_final.columns[0]: 'Gen'})
# 同じく CV を計算しておく（もしまだ無ければ）
if 'CV' not in df_final.columns:
    df_final['CV'] = df_final[Con_cols].clip(lower=0).sum(axis=1)

# -------------------------------------
# 2. 下三角用の色マップ（全データ：CVベース）
# -------------------------------------
cv = df_all['CV'].values
max_cv = cv.max()
# 違反量がすべて 0 の場合には blues_map の使用を避ける
if max_cv == 0:
    colors = ['red'] * len(df_all)
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

# -------------------------------------
# 3. 上三角用の色マップ（最終世代のみ：可行/不可行）
# -------------------------------------
# df_final の CV を見て、可行なら赤、不可行なら青とする
colors_final = [
    'red' if c == 0 else
    'blue'
    for c in df_final['CV'].values
]

# -------------------------------------
# 4. Seaborn で Scatter Plot Matrix
# -------------------------------------
sns.set(style='whitegrid', font_scale=1.2)
g = sns.PairGrid(df_all[X_cols], diag_sharey=False, height=2.5)

# 対角：全データのヒストグラム
g.map_diag(sns.histplot, kde=False, bins=10, color='royalblue', edgecolor='black')

# 下三角：全データを CV ベースの色で散布
g.map_lower(plt.scatter, color=colors, s=1, alpha=0.8)

# 上三角：最終世代ファイル df_final だけを“可行＝赤／違反＝青”で重ね描き
n = len(X_cols)
for i in range(n):
    for j in range(n):
        if i < j:
            ax = g.axes[i, j]
            xcol = X_cols[j]
            ycol = X_cols[i]
            ax.scatter(
                df_final[xcol],
                df_final[ycol],
                c=colors_final,
                s=20, edgecolor='k', linewidth=0.2, alpha=0.8
            )

# 軸ラベルや目盛りの調整（必要なら）
for ax in g.axes.flatten():
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.show()