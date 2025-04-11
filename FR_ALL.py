import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 20

def read_generation_data_pandas(filename):
    """
    CSVをPandasで読み込み、'Generation','Ratio' 列をもとに
    世代番号ごとに平均を計算して返す

    Parameters
    ----------
    filename : str
        読み込むCSVファイル名

    Returns
    -------
    gens : list of int
        世代番号をソートしたリスト
    avg_ratios : list of float
        gens[i]に対応する実行可能解割合の平均
    """
    # ヘッダ行が無い場合は "names=..." で列名を指定
    # ヘッダがあるなら 'header=0' などを指定して適宜調整
    df = pd.read_csv(filename, header=None, names=["Generation", "Ratio"])

    # 世代番号ごとに平均を計算
    df_group = df.groupby("Generation", as_index=False)["Ratio"].median()

    gens = df_group["Generation"].to_list()
    avg_ratios = df_group["Ratio"].to_list()
    return gens, avg_ratios


def plot_feasible_ratio_multi_pandas(filenames, labels=None, start_gen=None, end_gen=None):
    """
    複数ファイルに格納された世代番号と実行可能解割合を同じグラフ上にプロットし、比較表示する。
    指定された世代範囲内のデータのみをプロットします。

    Parameters
    ----------
    filenames : list of str
        読み込む CSV ファイル名のリスト
    labels : list of str, optional
        グラフの凡例に表示するラベル。ファイル名と1対1に対応。
        指定がなければ filename をそのまま表示する。
    start_gen : int, optional
        プロットを開始する世代番号。指定しない場合は最小世代から。
    end_gen : int, optional
        プロットを終了する世代番号。指定しない場合は最大世代まで。
    """
    plt.figure(figsize=(12, 10))

    if labels is None:
        # ラベルが指定されていなければファイル名をそのまま使う
        labels = filenames

    # 描画に使う色とラインスタイルのリスト
    # 必要に応じて色やスタイルを追加・変更してください
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']


    for i, (fname, label) in enumerate(zip(filenames, labels)):
        # ファイル読み込み -> groupby平均
        gens, avg_ratios = read_generation_data_pandas(fname)

        # 世代範囲でフィルタリング
        if start_gen is not None:
            filtered_data = [(g, r) for g, r in zip(gens, avg_ratios) if g >= start_gen]
        else:
            filtered_data = list(zip(gens, avg_ratios))

        if end_gen is not None:
            filtered_data = [(g, r) for g, r in filtered_data if g <= end_gen]

        if not filtered_data:
            print(f"Warning: No data to plot for file '{fname}' within the specified generation range.")
            continue

        # 分離
        filtered_gens, filtered_ratios = zip(*filtered_data)

        # 色とラインスタイルを選択
        color = color_list[i % len(color_list)]

        # プロット
        plt.plot(filtered_gens, filtered_ratios, color=color, linewidth=2,label=labels[i])

    plt.ylim(-0.01,1.01)
    plt.xlim(start_gen, end_gen)
    plt.grid(True, which='both', linewidth=0.5)
    #plt.legend(title="Methods", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 例: それぞれ異なる制約手法のファイルを読み込んで比較
    name = "feasible_ratio_CV1/RWMOP22"
    """
    filenames = [
        f"{name}_CDP.csv",
        f"{name}_SR.csv",
        f"{name}_SP.csv",
        f"{name}_Mo.csv",
        f"{name}_eps.csv"
    ]
    """
    filenames = [
        f"{name}_CV.csv",
        f"{name}_chev.csv",
        f"{name}_CVM.csv",
        f"{name}_count.csv",
    ]
    # 凡例として表示したい名前をファイルと同じ順序で用意
    #labels = ["CDP", "SR", "SP", "MO", "EPS"]
    labels = ["CV", "chev", "CVM", "count"]

    # プロットしたい世代範囲を指定（例: 世代50から世代200まで）
    start_generation = 1
    end_generation = 50

    plot_feasible_ratio_multi_pandas(
        filenames,
        labels=labels,
        start_gen=start_generation,
        end_gen=end_generation
    )