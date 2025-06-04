import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class kmedoids(BaseEstimator, ClusterMixin):


    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, D):

        D = np.asarray(D)
        n_samples = D.shape[0]
        rng = np.random.default_rng(self.random_state)

        # 1) ランダムに初期メドイドを選択
        medoids = rng.choice(n_samples, size=self.n_clusters, replace=False)
        labels = np.argmin(D[:, medoids], axis=1)

        for _ in range(self.max_iter):
            old_medoids = medoids.copy()

            # 2) 各クラスタごとにメドイドを更新
            for k in range(self.n_clusters):
                members = np.where(labels == k)[0]
                if len(members) == 0:
                    continue
                # クラスタ内の距離和が最小のノードを新メドイドに
                intra = D[np.ix_(members, members)]
                costs = intra.sum(axis=1)
                medoids[k] = members[np.argmin(costs)]

            # 3) ラベル再計算
            labels = np.argmin(D[:, medoids], axis=1)

            # 収束チェック
            if np.array_equal(medoids, old_medoids):
                break

        self.medoid_indices_ = medoids
        self.labels_ = labels
        return self

    def fit_predict(self, D):

        return self.fit(D).labels_