import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


def load_data(name: str = "imu"):
    data = []
    if name == "imu":
        data_path = Path("./sequencesimu.txt")
        with open(data_path, "r", encoding="utf-8") as d:
            lines = d.readlines()[1:]
        for l in lines:
            point = [float(e) for e in l.strip("\n").split()]
            data.append(point)

    elif name == "aiguillage":
        data_path = Path("./aiguillage.txt")
        with open(data_path, "r", encoding="utf-8") as d:
            lines = d.readlines()
        for l in lines:
            point = [float(e) for e in l.strip("\n").split(",")]
            data.append(point)

    return np.array(data, dtype=np.float32)


class FisherCluster:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.inertia_ = None

    @staticmethod
    def diam(data: np.ndarray, a: int, b: int) -> float:
        """
        一段区间内的
        :param data:
        :param a:
        :param b:
        :return:
        """
        d_seg = data[a:b + 1, :]  # x_a, ..., x_b (b inclus)
        mu = d_seg.mean(axis=0)
        return ((d_seg - mu) ** 2).sum()

    def fit_predict(self, X: np.ndarray):
        n, p = X.shape

        # Initialiser les matrices et vecteurs D, M, A, t, cluster
        D = np.zeros((n, n))
        M, A = np.zeros((n, self.n_clusters)), np.zeros((n, self.n_clusters), dtype=int)
        t = np.zeros(self.n_clusters - 1, dtype=int)
        cluster = np.zeros(n, dtype=int)

        #  calcul de la matrice triangulaire supérieure D
        for a in range(n):
            for b in range(a, n):
                if a == b:
                    D[a, b] = 0.0
                    continue
                D[a, b] = self.diam(X, a, b)

        # calcul recursive des critères optimaux
        M[:, 0] = D[0, :]
        for k in range(1, self.n_clusters):  # 第k个窗口
            for i in range(k, n):
                costs = [M[time - 1, k - 1] + D[time, i] for time in range(k, i + 1)]
                M[i, k] = np.min(costs)
                A[i, k] = np.argmin(costs) + k

        # calcul récursif des points de rupture optimaux
        # t[k-1] = indice (0-based) du début du (k+1)-ème segment
        k = self.n_clusters - 1
        m = n - 1
        while k >= 1:
            t[k - 1] = A[m, k]
            m = t[k - 1] - 1
            k -= 1

        # critère d'inertie C et labels des classes (1, ..., K)
        self.inertia_ = M[n - 1, self.n_clusters - 1]
        bornes = [0, *t, n]
        for k in range(self.n_clusters):
            cluster[bornes[k] : bornes[k + 1]] = k

        return t, cluster



def plot_scatter(X: np.ndarray, K: int):
    """
    画降维到2维的点，经过三个模型（Fisher, K-means, CAH）聚类K类之后得到的散点图
    :param X: 从原始维度的点降维到维度为2的点
    :param K: 聚类个数
    :return:
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    fisher = FisherCluster(n_clusters=K)
    k_means = KMeans(n_clusters=K)
    cah = AgglomerativeClustering(n_clusters=K)

    _, cluster_f = fisher.fit_predict(X)
    cluster_k = k_means.fit_predict(X)
    cluster_c = cah.fit_predict(X)

    clusters = [cluster_f, cluster_k, cluster_c]
    titles = ["Fisher", "K-means", "CAH"]
    colors = plt.get_cmap("tab10").colors

    # 对每个聚类模型生成的聚类画散点图
    for clus, ax, title in zip(clusters, axes, titles):
        unique_labels = np.unique(clus)  # [0, 1, ..., K-1]
        for i, label in enumerate(unique_labels):
            mask = clus == label
            ax.scatter(X[mask, 0], X[mask, 1], color=colors[i % len(colors)], s = 15, label = f"classe {label + 1}")
            ax.set_title(title)
            ax.legend()

    plt.tight_layout()
    plt.show()




def compute_intra_inertia(X: np.ndarray, labels: np.ndarray) -> float:
    """
    计算一个聚类的类内惯性
    :param X: 数据
    :param labels: 模型聚类后的标签
    :return: 类内惯性
    """
    inertia = 0.0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        inertia += ((cluster_points - centroid) ** 2).sum()
    return inertia


def plot_inertia(max_k: int):
    X = load_data()
    plt.figure(figsize=(8, 6))

    inertia_f, inertia_k, inertia_CAH = [], [], []
    for k in range(1, max_k + 1):
        # fisher
        fisher = FisherCluster(k)
        cuts, clusters = fisher.fit_predict(X)
        inertia_f.append(fisher.inertia_)

        # K-means
        k_means = KMeans(n_clusters=k)
        k_means_labels = k_means.fit_predict(X)
        inertia_k.append(compute_intra_inertia(X, k_means_labels))

        # CAH
        cah = AgglomerativeClustering(n_clusters=k)
        cah_labels = cah.fit_predict(X)
        inertia_CAH.append(compute_intra_inertia(X, cah_labels))

    x_ticks = np.arange(1, max_k + 1)

    plt.plot(x_ticks, inertia_f, label="Fisher")
    plt.plot(x_ticks, inertia_k, label="K-means")
    plt.plot(x_ticks, inertia_CAH, label="CAH")

    plt.title("Intra inertia trending of three clustering methods")
    plt.xlabel("Number of clusters")
    plt.ylabel("Intra inertia")

    plt.legend()
    plt.show()
    plt.close()


def main():
    X = load_data("imu")
    plot_scatter(X, 4)

if __name__ == "__main__":
    main()