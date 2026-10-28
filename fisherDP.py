import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import argparse

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path.cwd()


def load_data(file_name: str) -> np.ndarray:
    """
    Load a file based on its filename and return the training data
    Args:
        file_name (str): name of the training file 

    Raises:
        FileNotFoundError: If the file does not exist raise the error

    Returns:
        np.ndarray: training data in np.ndarray
    """
    data = []
    data_path = PROJECT_ROOT / file_name
    if not data_path.exists():
        raise FileNotFoundError("File not found")
    
    if "sequencesimu" in str(data_path):
        with open(data_path, "r", encoding="utf-8") as d:
            lines = d.readlines()[1:]
        for l in lines:
            point = [float(e) for e in l.strip("\n").split()]
            data.append(point)
            
    elif "aiguillage" in str(data_path):
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
        """Calculate the intra-group inertia of the data between windows a and b
        
        Args:
            data (np.ndarray): Training data in np.ndarray format
            a (int): Start Time of the Window
            b (int): End Time of the Window

        Returns:
            float: 
        """
        d_seg = data[a:b + 1, :]  # x_a, ..., x_b (b inclus)
        mu = d_seg.mean(axis=0)
        return ((d_seg - mu) ** 2).sum()

    def fit_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Train the Fisher model and return the clustering results
        
        Args:
            X (np.ndarray): Training data

        Returns:
            tuple[np.ndarray, np.ndarray]: Cluster boundaries and clustering results
        """
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



def plot_space_scatter(X: np.ndarray, clus_fisher: np.ndarray, 
                       clus_kmeans: np.ndarray, clus_cah: np.ndarray):
    """A scatter plot showing points reduced to two dimensions 
    after clustering into K classes using three models (Fisher, K-means, CAH)

    Args:
        clus_fisher (np.ndarray): training data reduced (2 dims)
        clus_kmeans
        clus_cah
        K (int): Number of clusters
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    clusters = [clus_fisher, clus_kmeans, clus_cah]
    titles = ["Fisher", "K-means", "CAH"]
    colors = plt.get_cmap("tab10").colors

    for clus, ax, title in zip(clusters, axes, titles):
        unique_labels = np.unique(clus)  # [0, 1, ..., K-1]
        for i, label in enumerate(unique_labels):
            mask = (clus == label)
            ax.scatter(x=X[mask, 0], y=X[mask, 1], color=colors[i % len(colors)], 
                       s = 15, label = f"class {label + 1}")
            ax.set_title(title)
            ax.legend()

    fig.suptitle("Clustering in Space")
    plt.tight_layout()
    plt.show()
    

def plot_time_scatter(X: np.ndarray, cuts_fisher: np.ndarray, clus_fisher: np.ndarray, 
                      clus_kmeans: np.ndarray, clus_cah: np.ndarray):
    """Plot the clustering results for the three models 
    on a timeline, with time on the x-axis and the first 
    and second dimensions of the dimension-reduced data 
    points on the y-axis.

    Args:
        X (np.ndarray):  training data reduced (2 dims)
        K (int): Number of clusters
    """
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))

    clusters = [clus_fisher, clus_kmeans, clus_cah]
    titles = ["Fisher", "K-means", "CAH"]
    colors = plt.get_cmap("tab10").colors

    for clus, ax, title in zip(clusters, axes, titles):
        unique_labels = np.unique(clus)
        for i, label in enumerate(unique_labels):
            mask = (clus == label)
            mask_ids = np.arange(X.shape[0])[mask]
            ax[0].scatter(x=mask_ids, y=X[mask, 0], color=colors[i % len(colors)], 
                          s = 15, label = f"class {label + 1}")
            ax[1].scatter(x=mask_ids, y=X[mask, 1], color=colors[i % len(colors)], 
                          s = 15, label = f"class {label + 1}")
            
            ax[0].set_title(f"{title} (dim 1)")
            ax[1].set_title(f"{title} (dim 2)")
            
            ax[0].set_xlabel("time")
            ax[1].set_xlabel("time")
            
            ax[0].set_ylabel("position")
            
            ax[0].legend()
            ax[1].legend()
    
    # draw vertical lines on Fisher clusters
    for cut in cuts_fisher:
        axes[0, 0].axvline(x=cut, ls="--", c="yellow", lw=1.5)
        axes[0, 1].axvline(x=cut, ls="--", c="yellow", lw=1.5)
    
    fig.suptitle("Clustering Along the Time Dimension")
    plt.tight_layout()
    plt.show()


def compute_intra_inertia(X: np.ndarray, labels: np.ndarray) -> float:
    """Calculate the intra-cluster inertia for a cluster

    Args:
        X (np.ndarray): Training data
        labels (np.ndarray): Labels after clustering

    Returns:
        float: Intra-inertia
    """
    inertia = 0.0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        inertia += ((cluster_points - centroid) ** 2).sum()
    return inertia


def plot_inertia(X: np.ndarray, max_k: int):
    """Plot the intra-cluster inertia for three models 
    ranging from 1 cluster to max_k clusters; the optimal 
    number of clusters can be found using the elbow method.

    Args:
        X (np.ndarray): Traning data
        max_k (int): Maximum number of the clusters
    """
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

    plt.title("Intra inertia curve of three clustering methods")
    plt.xlabel("Number of clusters")
    plt.ylabel("Intra inertia")

    plt.legend()
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_name", type=str, default="aiguillage.txt", 
                        choices=["sequencesimu.txt", "aiguillage.txt"])
    parser.add_argument("--optim_k", type=int, default=4, help="The optimal number of clusters for Fisher obtaining by elbow method")
    parser.add_argument("--max_k", type=int, default=10, help="Set the maximum number of clusters")
    
    args = parser.parse_args()
    
    if args.data_name == "aiguillage.txt":
        X = load_data(args.data_name)[:, :552]
    elif args.data_name == "sequencesimu.txt":
        X = load_data(args.data_name)
    
    if X.shape[1] > 2:
        scalar = StandardScaler()
        pca = PCA(n_components=2)
        X_scaled = scalar.fit_transform(X)
        X_reduced = pca.fit_transform(X_scaled)
    else:
        X_reduced = X
        
    fisher = FisherCluster(n_clusters=args.optim_k)
    k_means = KMeans(n_clusters=args.optim_k)
    cah = AgglomerativeClustering(n_clusters=args.optim_k)

    cuts, clus_f = fisher.fit_predict(X)
    clus_k = k_means.fit_predict(X)
    clus_c = cah.fit_predict(X)
        
    plot_inertia(X, args.max_k)    
    plot_space_scatter(X_reduced, clus_f, clus_k, clus_c)   
    plot_time_scatter(X_reduced, cuts, clus_f, clus_k, clus_c)
    

if __name__ == "__main__":
    main()