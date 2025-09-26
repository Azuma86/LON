import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20

problem_name = 'RWMOP16'
csv_all = Path(f'/Users/azumayuki/Downloads/RWMOP/{problem_name}.csv')
df_all = pd.read_csv(csv_all)

df_all = df_all.rename(columns={df_all.columns[0]: 'Gen'})
X_cols = [c for c in df_all.columns if c.startswith('X')]
Con_cols = [c for c in df_all.columns if c.startswith('Con')]

df_all['CV'] = df_all[Con_cols].clip(lower=0).sum(axis=1)


# 1. データ整形
X = df_all[X_cols]
y = df_all['CV'].clip(lower=0)

# 2. ノイズ除去
z = (y - y.mean()) / y.std()
mask = np.abs(z) < 3  # Zスコアで外れ値を除外（|z|<3）
X, y = X[mask], y[mask]

# 3. train-test 分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. ランダムフォレスト回帰
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 5. 予測と評価
y_pred = rf.predict(X_test)
print("R² :", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

feature_importances = pd.Series(rf.feature_importances_, index=X_cols)
feature_importances.sort_values(ascending=False).plot(kind='barh', figsize=(8,6))
sorted_importances = feature_importances.sort_values(ascending=True)
print("Feature Importances:")
for feature, importance in sorted_importances.items():
    print(f"{feature:30s}: {importance:.4f}")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

plt.figure(figsize=(50,50))
plt.scatter(y_test, y_pred, alpha=0.6, s=90)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--')
plt.xlabel('Actual CV'), plt.ylabel('Predicted CV')
plt.grid()
plt.show()