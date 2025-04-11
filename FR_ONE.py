import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_feasible_ratio(filename, num_trials=21):
    """
    同じファイルにまとめて書かれた num_trials 回分の実行可能解割合を，
    世代番号をキーとして集計し，平均値をグラフ表示する.

    Parameters
    ----------
    filename : str
        CSVファイル名 (世代番号, 実行可能解割合) の形式
    num_trials : int, optional
        試行回数 (同じ世代番号で何回分のデータがあるか)
    """

    # 世代番号をキーとし，実行可能解割合をリストで格納する辞書
    data_dict = {}

    # CSVファイルを読み込み，辞書に格納する
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue  # 空行などスキップ
            gen_str, ratio_str = row[0], row[1]
            try:
                gen = int(gen_str)
                ratio = float(ratio_str)
            except ValueError:
                # 数値変換できない場合はスキップ
                continue

            if gen not in data_dict:
                data_dict[gen] = []
            data_dict[gen].append(ratio)

    # 世代番号をソートして平均値を計算
    gens = sorted(data_dict.keys())
    avg_ratios = []
    for g in gens:
        ratios_g = data_dict[g]
        # (念のため) num_trials 回分でない世代があっても平均はとる
        avg_ratios.append(np.mean(ratios_g))

    # グラフを描画
    plt.figure(figsize=(12, 10))
    plt.plot(gens, avg_ratios, marker='o', color='b', label='Average feasible ratio')
    plt.xlabel('Generation')
    plt.ylabel('Feasible Ratio')
    plt.title('Average Feasible Ratio over {} Trials'.format(num_trials))
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 例として "RWMOP_SR.csv" というファイルを読み込み，
    # 21回試行分の平均をプロットする
    plot_feasible_ratio("gomi/feasible_ratio/RWMOP20_SP.csv", num_trials=21)