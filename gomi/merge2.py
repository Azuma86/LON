import pandas as pd
import os

# 結合対象のディレクトリリスト
directories = ["/Users/azumayuki/Documents/LONs/feasible_first", "/Users/azumayuki/Documents/LONs/fesible_first_data"]  # ここに各ディレクトリのパスを指定

for i in range(24,25):
    name = "RWMOP{}".format(i)
    filenames = [
        #f"{name}_CDP_first.csv",
        #f"{name}_SR_first.csv",
        #f"{name}_SP_first.csv",
        f"{name}_Mo_first.csv",
        #f"{name}_eps_first.csv"
    ]
    for csv_filename in filenames:
        # CSVファイルの名前

        dfs = []  # DataFrameのリスト

        for directory in directories:
            file_path = os.path.join(directory, csv_filename)
            if os.path.exists(file_path):  # ファイルが存在するか確認
                df = pd.read_csv(file_path, header=None)  # ヘッダーなしで読み込む
                dfs.append(df)

        # 全てのデータを結合（縦方向に連結）
        merged_df = pd.concat(dfs, axis=1,ignore_index=True)

        # 結果を保存（必要なら）
        merged_df.to_csv(f"/Users/azumayuki/Documents/LONs/feasible_first31/{csv_filename}", index=False, header=False)  # ヘッダーなしで保存
