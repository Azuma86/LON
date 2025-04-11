import pandas as pd
import os

# 結合対象のディレクトリリスト
directories = ["/Users/azumayuki/Documents/LONs/feasible_ratio", "/Users/azumayuki/Documents/LONs/feasible_ratio_data"]  # ここに各ディレクトリのパスを指定

for i in range(1,36):
    name = "RWMOP{}".format(i)
    filenames = [
        f"{name}_CDP.csv",
        f"{name}_SR.csv",
        f"{name}_SP.csv",
        f"{name}_Mo.csv",
        f"{name}_eps.csv"
    ]
    for csv_filename in filenames:
        # CSVファイルの名前

        dfs = []  # DataFrameのリスト

        for directory in directories:
            file_path = os.path.join(directory, csv_filename)
            if os.path.exists(file_path):  # ファイルが存在するか確認
                df = pd.read_csv(file_path, header=None)  # ヘッダーなしで読み込む
                df.columns = ["ID", "Value"]  # 列名を明示的に設定
                dfs.append(df)

        # 全てのデータを結合（縦方向に連結）
        merged_df = pd.concat(dfs, ignore_index=True)

        # 結果を保存（必要なら）
        merged_df.to_csv(f"/Users/azumayuki/Documents/LONs/feasible_ratio31/{csv_filename}", index=False, header=False)  # ヘッダーなしで保存
