import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

# --- データ読み込み ---
csv_path = Path('/Users/azumayuki/Downloads/RWMOP/RWMOP20.csv')
df = pd.read_csv(csv_path)

# 制約列を自動検出
Con_cols = [c for c in df.columns if c.startswith('Con')]
data = df[Con_cols]

# --- 相関行列の事前計算 ---
corr = data.corr()

# --- カラーマップ準備 ---
vmin, vmax = -1, 1
norm  = Normalize(vmin=vmin, vmax=vmax)
cmap  = plt.colormaps['RdBu']  # 赤⇔青

# --- 上三角セルを塗りつぶして数字を描く関数 ---
def corr_tile(x, y, **kws):
    ax = plt.gca()
    # x.name, y.name はそれぞれ列名 (Con_1, Con_2, …)
    r = corr.loc[y.name, x.name]
    if r >= 0:
        cmap = plt.colormaps['Reds']
        color = cmap(r)
    else:
        cmap = plt.colormaps['Blues']
        color = cmap(-r)
    # 背景色として塗りつぶし
    ax.set_facecolor(color)
    # 中央に係数を描画
    ax.text(0.5, 0.5, f"{r:.2f}",
            ha='center', va='center',
            color='white' if abs(r)>0.5 else 'black',
            fontsize=30)
    # 軸目盛りと枠線を消す
    """
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    """
# --- PairGrid の組み立て ---
sns.set(style='whitegrid', font_scale=1.2)
g = sns.PairGrid(data, diag_sharey=False, height=3)

# 対角：各変数のヒストグラム
g.map_diag(sns.histplot,bins = 10, color='royalblue')

# 下三角：散布図
g.map_lower(sns.scatterplot, s=20, edgecolor='k', alpha=0.7)

# 上三角：相関タイル
g.map_upper(corr_tile)

# レイアウト調整
for ax in g.axes.flatten():
    ax.tick_params(labelsize=8)
plt.show()