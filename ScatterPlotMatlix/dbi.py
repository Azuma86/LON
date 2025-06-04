import numpy as np

def davies(W, labels):
    """
    距離行列 W (N×N) と、クラスタラベル labels (長さ N) から
    Davies–Bouldin 指数を返す。

    W[i,j] = i番目と j番目の距離（対称行列）
    labels[i] = i番目の系列が属するクラスタ番号 (0～K-1)

    メドイドを使って S_i, M_ij を定義し、
    DBI = (1/K) * sum_i max_{j != i} ((S_i + S_j) / M_ij) を計算。
    """
    labels = np.asarray(labels)
    N = W.shape[0]
    cluster_ids = np.unique(labels)
    K = len(cluster_ids)

    # 1) 各クラスタごとに「メンバーの index」を集める
    clusters = {c: np.where(labels == c)[0] for c in cluster_ids}

    # 2) 各クラスタのメドイド index を求める
    #    メドイド = クラスタ内で「総距離和 (distance to all other members)」が最小のサンプル
    medoid_of = {}
    for c in cluster_ids:
        members = clusters[c]
        # クラスタ内の距離行列を切り出し
        subW = W[np.ix_(members, members)]  # shape = (|members|, |members|)
        # 各行の和を計算
        dist_sums = subW.sum(axis=1)
        # 和が最小のインデックスを取り、そのサンプルの global index をメドイドとする
        medoid_idx = members[np.argmin(dist_sums)]
        medoid_of[c] = medoid_idx

    # 3) 各クラスタの S_i を計算
    #    S_i = (1/|C_i|) * sum_{x in C_i} d(x, medoid_i)
    S = np.zeros(K)
    for idx_c, c in enumerate(cluster_ids):
        members = clusters[c]
        med = medoid_of[c]
        # W[med, members] は「メドイド ↔ 各メンバー」の距離ベクトル
        S[idx_c] = W[med, members].mean()

    # 4) クラスタ間の M_ij を計算（メドイド同士の距離）
    #    M_ij = d(medoid_i, medoid_j) = W[medoid_of[i], medoid_of[j]]
    M = np.zeros((K, K))
    for i, ci in enumerate(cluster_ids):
        for j, cj in enumerate(cluster_ids):
            if i == j:
                M[i, j] = 0.0
            else:
                M[i, j] = W[medoid_of[ci], medoid_of[cj]]

    # 5) Davies–Bouldin 指数を計算
    DBi = 0.0
    for i in range(K):
        max_val = -np.inf
        for j in range(K):
            if i == j:
                continue
            # (S_i + S_j) / M_ij を計算
            if M[i, j] > 0:
                val = (S[i] + S[j]) / M[i, j]
                if val > max_val:
                    max_val = val
        DBi += max_val

    DBi = DBi / K
    return DBi
