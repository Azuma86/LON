import csv

def compute_average_from_csv(filename):
    """
    filename という1行のCSVファイルを読み込み，
    カンマ区切りで並んだ整数を取得して平均を返す。
    空または数値でない要素は除外しつつ整数変換して合計。
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # 最初の1行を読み取る
        row = next(reader, None)
        if row is None:
            print(f"Warning: File '{filename}' is empty.")
            return None

    # row は文字列リスト
    # 空文字などは除外し、整数変換
    numbers = []
    for item in row:
        item = item.strip()
        if item != '':
            numbers.append(int(item))

    if len(numbers) == 0:
        print(f"Warning: File '{filename}' contains no valid numbers.")
        return None

    avg_value = sum(numbers) / len(numbers)
    return avg_value

if __name__=="__main__":
    # 対象となる複数ファイルのリスト
    name = "feasible_first_CHT/RWMOP6"
    filenames = [
        f"{name}_CDP_first.csv",
        f"{name}_SR_first.csv",
        f"{name}_SP_first.csv",
        f"{name}_Mo_first.csv",
        f"{name}_eps_first.csv"
    ]

    # 各ファイルごとに平均を計算・出力
    for fname in filenames:
        avg = compute_average_from_csv(fname)
        if avg is not None:
            print(f"File: {fname}, Average: {avg:.2f}")
        else:
            print(f"File: {fname}, no average computed.")