import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# .matファイルの読み込み
mat_data = sio.loadmat('your_result_file.mat')  # 例: 'result.mat'
# PlatEMOが出力するresultという変数を読み出す
# 通常、mat_data['result'] は1x1の構造体配列になっており、その中に
# Dec, Obj, etc.が含まれています。
result = mat_data['result']

# resultが構造体配列の場合（例: shape=(1,1) のStructured array）、
# Python側では配列の要素を添字でアクセスしてさらにフィールドを取得する形になります。
# Dec(設計変数)やObj(目的変数)の読み出しイメージ
Dec = result['Dec'][0,0]  # 決定変数を取り出し
Obj = result['Obj'][0,0]  # 目的変数を取り出し (必要に応じて)

# Decが2次元の設計空間だと仮定して散布図をプロット
# (例えば変数が2つの場合)
plt.figure()
plt.scatter(Dec[:, 0], Dec[:, 1], marker='o')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Solution in the Search Space (2D)')

plt.show()

# -----------------------------------------
# もし探索空間が3次元の場合のプロット例
# Decが3次元 (変数が3つ) であるとき
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Dec[:, 0], Dec[:, 1], Dec[:, 2], marker='o')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
plt.title('Solution in the Search Space (3D)')

plt.show()