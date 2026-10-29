"""Clustering under order constraint: Fisher's dynamic programming algorithm

This script implements Fisher's dynamic programming algorithm, which splits a
sequence (x_1, ..., x_n) into K contiguous segments by minimizing the
intra-segment inertia criterion C (exact global optimum). It then compares the
resulting partition with K-Means and hierarchical clustering (CAH, Ward).

Data files (must be in the current working directory):
    - sequencesimu.txt : simulated 2D sequence, 160 points (header line skipped)
    - aiguillage.txt   : real data, 140 points in R^552, the 553rd column is
                         the true class (1, ..., 4)

Dependencies:
    numpy, scipy, scikit-learn, matplotlib, seaborn

Command line usage (run from the project directory):
    python fisherDP.py [--data_name FILE] [--optim_k K] [--max_k MAX_K] [--save_dir DIR]

    --data_name : "sequencesimu.txt" (default) or "aiguillage.txt"
    --optim_k   : number of clusters K used by the three models (default 4),
                  to be chosen with the elbow method on the inertia curve
    --max_k     : largest K shown on the inertia curve (default 10)
    --save_dir  : directory where the figures are saved (default "figures"),
                  each dataset gets its own sub-folder, e.g. figures/aiguillage/

    Examples:
        python fisherDP.py                                  # simulated data, K=4
        python fisherDP.py --data_name aiguillage.txt --optim_k 4

What the script produces:
    1. Intra-inertia curve vs K for Fisher, K-Means and CAH (elbow method)
    2. Scatter plot of the clusters in space (PCA to 2D if p > 2)
    3. Clusters along the time axis, with Fisher's cut points as dashed lines
    4. aiguillage only: ARI of each model, true vs Fisher cut points, and
       confusion matrices (cluster labels aligned with the Hungarian algorithm)
    5. Fisher's cut points printed to the console
    Figures 1-4 are shown and saved as PNG (300 dpi) in <save_dir>/<data name>/:
    inertia.png, space_scatter.png, time_scatter.png, confusion_matrix.png

Usage as a module:
    >>> from fisherDP import load_data, clustfisher
    >>> X = load_data("sequencesimu.txt")
    >>> fisher, cuts, labels = clustfisher(4, X)
    >>> cuts             # 0-based index of the first point of segments 2..K
    >>> fisher.inertia_  # optimal criterion C
"""
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import argparse

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path.cwd()


def load_data(file_name: str) -> np.ndarray:
    """Load a file based on its filename and return the training data
    
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
        for k in range(1, self.n_clusters):  # k-th window
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


def clustfisher(n_clusters: int, X: np.ndarray) -> tuple[FisherCluster, np.ndarray, np.ndarray]:
    """Wrapper function of Fisher cluster to align with the requirements

    Args:
        n_clusters (int): number of clusters 
        X (np.ndarray): train data

    Returns:
        tuple[FisherCluster, np.ndarray, np.ndarray]: Fisher cluster, cluster boundaries and clustering results
    """
    fisher = FisherCluster(n_clusters)
    t, cluster = fisher.fit_predict(X)
    return fisher, t, cluster
    


def show_and_save(fig: plt.Figure, save_path: Path | None = None):
    """Save the figure (if a path is given) and then display it

    Args:
        fig (plt.Figure): figure to save and show
        save_path (Path | None): output file (e.g. figures/inertia.png), None to only show
    """
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()
    plt.close(fig)


def plot_space_scatter(X: np.ndarray, clus_fisher: np.ndarray,
                       clus_kmeans: np.ndarray, clus_cah: np.ndarray,
                       save_path: Path | None = None):
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
    show_and_save(fig, save_path)
    

def plot_time_scatter(X: np.ndarray, cuts_fisher: np.ndarray, clus_fisher: np.ndarray,
                      clus_kmeans: np.ndarray, clus_cah: np.ndarray,
                      save_path: Path | None = None):
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
        axes[0, 0].axvline(x=cut, ls="--", c="red", lw=1.5)
        axes[0, 1].axvline(x=cut, ls="--", c="red", lw=1.5)
    
    fig.suptitle("Clustering Along the Time Dimension")
    plt.tight_layout()
    show_and_save(fig, save_path)


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


def plot_inertia(X: np.ndarray, max_k: int, save_path: Path | None = None):
    """Plot the intra-cluster inertia for three models 
    ranging from 1 cluster to max_k clusters; the optimal 
    number of clusters can be found using the elbow method.

    Args:
        X (np.ndarray): Traning data
        max_k (int): Maximum number of the clusters
    """
    fig = plt.figure(figsize=(8, 6))

    inertia_f, inertia_k, inertia_CAH = [], [], []
    for k in range(1, max_k + 1):
        # fisher
        fisher, _, _ = clustfisher(k, X)
        inertia_f.append(fisher.inertia_)

        # K-means
        k_means = KMeans(n_clusters=k ,random_state=42)
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
    show_and_save(fig, save_path)


def align_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Relabel the clusters so that they best match the true classes
    (Hungarian algorithm on the confusion matrix)

    Args:
        y_true (np.ndarray): True labels (0, ..., K-1)
        y_pred (np.ndarray): Cluster labels (0, ..., K-1)

    Returns:
        np.ndarray: Relabeled clusters
    """
    y_true = y_true.astype(int)
    cm = confusion_matrix(y_true, y_pred)
    # maximize the diagonal <=> minimize -cm
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[l] for l in y_pred])


def draw_cm(y_true: np.ndarray, clus_fisher: np.ndarray,
            clus_kmeans: np.ndarray, clus_cah: np.ndarray,
            save_path: Path | None = None):
    """Draw confusion matrix showing the predictions of the 
    three models and the true labels

    Args:
        y_true (np.ndarray): true labels
        clus_fisher (np.ndarray): predict labels of Fisher
        clus_kmeans (np.ndarray): predict labels of K-Means
        clus_cah (np.ndarray): predict labels of CAH
    """
    y_true = y_true.astype(int)
    
    cm_fisher = confusion_matrix(y_true, align_labels(y_true, clus_fisher))
    cm_kmeans = confusion_matrix(y_true, align_labels(y_true, clus_kmeans))
    cm_cah = confusion_matrix(y_true, align_labels(y_true, clus_cah))

    cms = [cm_fisher, cm_kmeans, cm_cah]
    labels = ["class 1", "class 2", "class 3", "class 4"]
    titles = ["Fisher", "K-Means", "CAH"]
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, cm, title in zip(axes, cms, titles):
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            square=True
        )
        ax.set_title(title)
        ax.set_xlabel("y_pred")
        ax.set_ylabel("y_true")
    
    fig.suptitle("Confusion matrix of 3 models")
    plt.tight_layout()
    show_and_save(fig, save_path)


def main():
    """Main function
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_name", type=str, default="sequencesimu.txt", 
                        choices=["sequencesimu.txt", "aiguillage.txt"])
    parser.add_argument("--optim_k", type=int, default=4, help="The optimal number of clusters for Fisher obtaining by elbow method")
    parser.add_argument("--max_k", type=int, default=10, help="Set the maximum number of clusters")
    parser.add_argument("--save_dir", type=str, default="figures", help="Directory where the figures are saved")

    args = parser.parse_args()
    
    if args.data_name == "aiguillage.txt":
        data = load_data(args.data_name)
        X, y = data[:, :552], data[:, 552] - 1
    elif args.data_name == "sequencesimu.txt":
        X, y = load_data(args.data_name), None
    
    # reduce the dimentsion of data in aiguillage.txt
    if X.shape[1] > 2:
        scalar = StandardScaler()
        pca = PCA(n_components=2)
        X_scaled = scalar.fit_transform(X)
        X_reduced = pca.fit_transform(X_scaled)
    else:
        X_reduced = X
        
    # fisher = FisherCluster(n_clusters=args.optim_k)
    k_means = KMeans(n_clusters=args.optim_k, random_state=42)
    cah = AgglomerativeClustering(n_clusters=args.optim_k)

    _, cuts, clus_f = clustfisher(args.optim_k, X)
    clus_k = k_means.fit_predict(X)
    clus_c = cah.fit_predict(X)
    
    # figures/<data name>/xxx.png
    save_dir = PROJECT_ROOT / args.save_dir / Path(args.data_name).stem

    plot_inertia(X, args.max_k, save_dir / "inertia.png")
    plot_space_scatter(X_reduced, clus_f, clus_k, clus_c, save_dir / "space_scatter.png")
    plot_time_scatter(X_reduced, cuts, clus_f, clus_k, clus_c, save_dir / "time_scatter.png")
    
    # Compare results of clustering of 3 models and true label on real data
    if y is not None:
        ari_fisher = adjusted_rand_score(clus_f, y)
        ari_kmeans = adjusted_rand_score(clus_k, y)
        ari_cah = adjusted_rand_score(clus_c, y)
    
        print(f"\tFisher\tK-Means\tCAH")
        print(f"ARI\t{ari_fisher:.3f}\t{ari_kmeans:.3f}\t{ari_cah:.3f}")
        print(f"Real cut points: [35, 75, 120]")
          
        draw_cm(y, clus_f, clus_k, clus_c, save_dir / "confusion_matrix.png")
    
    print(f"Fisher predict cut points: {cuts.tolist()}")
    

if __name__ == "__main__":
    main()