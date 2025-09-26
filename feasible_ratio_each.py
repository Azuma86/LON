import pandas as pd
import matplotlib.pyplot as plt
pro_name = "RWMOP20"
algo = "CDP"
# === 1. CSVファイルを読み込み ===
df = pd.read_csv(f"feasible_ratio_each/{pro_name}_{algo}.csv")  # MATLAB 側で追記していくファイル

# === 2. 世代ごとに平均を取る ===
df_avg = df.groupby("gen", as_index=False).mean()

# 制約ごとの列名を抽出
con_cols = [c for c in df.columns if c.startswith("con")]

# 平均化済み df_avg を利用
plt.figure(figsize=(8, 5))
#plt.plot(df_avg["gen"], df_avg["all"], label="All", linewidth=2, color='red')

# 各制約ごとに描画
for col in con_cols:
    plt.plot(df_avg["gen"], df_avg[col], linestyle="--", label=col)


plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()