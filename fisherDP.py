"""Clustering under order constraint: Fisher's dynamic programming algorithm

This script implements Fisher's dynamic programming algorithm, which splits a
sequence (x_1, ..., x_n) into K contiguous segments by minimizing the
intra-segment inertia criterion C (exact global optimum). It then compares the
resulting partition with K-Means and hierarchical clustering (CAH, Ward).

Optimization guarantees:
    - Fisher  : global optimum of C over all partitions into K contiguous,
                nonempty segments (exact dynamic programming)
    - K-Means : local optimum of the intra-class inertia, depending on the
                initialization (hence several starts, see --n_init)
    - CAH     : greedy (Ward merges the two closest classes at each step),
                no optimality guarantee for a given K
    K-Means and CAH ignore the order of the observations.

Data files (must be in the current working directory):
    - sequencesimu.txt : simulated 2D sequence, 160 points (header line skipped)
    - aiguillage.txt   : real data, 140 points in R^552, the 553rd column is
                         the true class (1, ..., 4), never used for fitting

Dependencies:
    numpy, scipy, scikit-learn, matplotlib, seaborn

Command line usage (run from the project directory):
    python fisherDP.py [--data_name FILE] [--optim_k K] [--max_k MAX_K]
                       [--n_init N] [--seed SEED] [--save_dir DIR]

    --data_name : "sequencesimu.txt" (default) or "aiguillage.txt"
    --optim_k   : number of clusters K used by the three models (default 4),
                  to be chosen with the elbow method on the criterion C
    --max_k     : largest K of the elbow curve (default 10)
    --n_init    : number of K-Means initializations, the run with the lowest
                  inertia is kept (default 20)
    --seed      : random_state of K-Means (default 42)
    --save_dir  : directory where the figures are saved (default "figures"),
                  each dataset gets its own sub-folder, e.g. figures/aiguillage/
    Both K values must satisfy 1 <= K <= n.

    Examples:
        python fisherDP.py                                  # simulated data, K=4
        python fisherDP.py --data_name aiguillage.txt --optim_k 4

Conventions:
    - labels are zero-based: 0, ..., K-1
    - cut points are the zero-based indices of the first observation of
      segments 2..K; the course's one-based t_k is cut + 1
      (e.g. cuts [50, 80, 130] <=> t = [51, 81, 131])

What the script produces:
    1. Fisher's criterion C(K) and its decrease for K = 1..max_k (printed), and
       the inertia curves of the three methods (elbow method)
    2. Fisher cut points and segment lengths, cluster sizes and intra-class
       inertia of the three methods
    3. aiguillage only: ARI, correct classification rate (TBC) and true cut
       points, and confusion matrices against the true classes
    4. Scatter plot of the clusters in space (standardized PCA to 2D if p > 2;
       display only, all methods are fitted on the raw data)
    5. Clusters along the observation index, with Fisher's cut points (dashed)
       and the fitted segment means
    Cluster labels are aligned with the Hungarian algorithm on the true classes
    (aiguillage) or on Fisher's segments (sequencesimu), so a color means the
    same group in every panel.
    Figures are shown and saved as PNG (300 dpi) in <save_dir>/<data name>/:
    inertia.png, space_scatter.png, time_scatter.png, confusion_matrix.png

Usage as a module:
    >>> from fisherDP import load_data, clustfisher, diam
    >>> X = load_data("sequencesimu.txt")
    >>> fisher, cuts, labels = clustfisher(4, X)
    >>> cuts             # zero-based index of the first point of segments 2..K
    >>> fisher.inertia_  # optimal criterion C
    >>> diam(X, 0, 49)   # D(a, b) of the first segment
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
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path.cwd()
DATASETS = ("sequencesimu.txt", "aiguillage.txt")
COLORS = plt.get_cmap("tab10").colors


def load_data(file_name: str) -> np.ndarray:
    """Load a file based on its filename and return the training data

    Args:
        file_name (str): name of the training file, one of DATASETS

    Raises:
        ValueError: If the file is not one of the supported datasets
        FileNotFoundError: If the file does not exist raise the error

    Returns:
        np.ndarray: training data in np.ndarray
    """
    data = []
    data_path = PROJECT_ROOT / file_name
    if data_path.name not in DATASETS:
        raise ValueError(f"Unsupported data file '{file_name}', expected one of {DATASETS}")
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    if data_path.name == "sequencesimu.txt":
        with open(data_path, "r", encoding="utf-8") as d:
            lines = d.readlines()[1:]
        for l in lines:
            point = [float(e) for e in l.strip("\n").split()]
            data.append(point)

    elif data_path.name == "aiguillage.txt":
        with open(data_path, "r", encoding="utf-8") as d:
            lines = d.readlines()
        for l in lines:
            point = [float(e) for e in l.strip("\n").split(",")]
            data.append(point)

    return np.array(data, dtype=np.float32)


def check_n_clusters(n_clusters: int, n_samples: int | None = None) -> int:
    """Check that the number of clusters is an integer with 1 <= K (<= n)

    Args:
        n_clusters (int): number of clusters K
        n_samples (int | None): number of observations n, None to skip K <= n

    Raises:
        ValueError: If K is not an integer, K < 1 or K > n

    Returns:
        int: K
    """
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
        raise ValueError(f"The number of clusters must be an integer, got {n_clusters!r}")
    if n_clusters < 1:
        raise ValueError(f"The number of clusters must be >= 1, got {n_clusters}")
    if n_samples is not None and n_clusters > n_samples:
        raise ValueError(f"The number of clusters ({n_clusters}) cannot exceed "
                         f"the number of observations ({n_samples})")
    return int(n_clusters)


def check_data(X: np.ndarray) -> np.ndarray:
    """Check that the data is a nonempty 2D array of finite numbers

    Args:
        X (np.ndarray): data (n, p)

    Raises:
        ValueError: If X is not 2D, is empty or contains non-finite values

    Returns:
        np.ndarray: X as a numpy array
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"X must be a nonempty 2D array (n, p), got shape {X.shape}")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError("X must contain only finite numerical values")
    return X


def diam(X: np.ndarray, a: int, b: int) -> float:
    """Diameter D(a, b) of the segment (x_a, ..., x_b), formula (2) of the
    project: sum over i = a..b and j = 1..p of (x_i^j - mu_ab^j)^2, i.e. the
    intra-class inertia of the segment

    Args:
        X (np.ndarray): data (n, p)
        a (int): zero-based index of the first observation of the segment
        b (int): zero-based index of the last observation (included), a <= b

    Raises:
        ValueError: If the segment is not inside the data

    Returns:
        float: D(a, b)
    """
    if not 0 <= a <= b < X.shape[0]:
        raise ValueError(f"Invalid segment [{a}, {b}] for {X.shape[0]} observations")
    d_seg = X[a:b + 1, :]  # x_a, ..., x_b (b inclus)
    mu = d_seg.mean(axis=0)
    return float(((d_seg - mu) ** 2).sum())


class FisherCluster:
    """Fisher's dynamic programming for clustering under order constraint

    Splits (x_0, ..., x_{n-1}) into K contiguous, nonempty segments minimizing
    C = sum of the diameters D of the segments (global optimum over all such
    partitions).

    Attributes:
        n_clusters (int): number of segments K
        inertia_ (float): optimal criterion C, available after fit_predict
    """

    def __init__(self, n_clusters: int):
        self.n_clusters = check_n_clusters(n_clusters)
        self.inertia_ = None

    def fit_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Train the Fisher model and return the optimal segmentation

        Zero-based version of the course algorithm:
            - D[a, b] = diam(X, a, b) for a <= b (upper triangular)
            - forward pass: M[i, k] is the optimal criterion of (x_0, ..., x_i)
              split into k + 1 segments
                  M[i, 0] = D[0, i]
                  M[i, k] = min_{t = k..i} M[t - 1, k - 1] + D[t, i]
              (t >= k and t <= i keep every segment nonempty)
            - A[i, k] is the argmin t, i.e. the zero-based index of the first
              observation of the last segment
            - backward pass: the cut points are read in A from (n - 1, K - 1)

        Args:
            X (np.ndarray): data (n, p), nonempty and finite, with K <= n

        Raises:
            ValueError: If X is not valid or K > n

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - cuts (K - 1,): zero-based index of the first observation of
                  segments 2..K (course t_k = cuts + 1)
                - labels (n,): zero-based segment of each observation
                  (0, ..., K-1 in time order)
        """
        X = check_data(X)
        n, p = X.shape
        check_n_clusters(self.n_clusters, n)

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
                D[a, b] = diam(X, a, b)

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

        # critère d'inertie C et labels des classes (0, ..., K-1)
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
                       axis_labels: tuple[str, str] = ("dim 1", "dim 2"),
                       save_path: Path | None = None):
    """A scatter plot showing points reduced to two dimensions
    after clustering into K classes using three models (Fisher, K-means, CAH)

    Args:
        X (np.ndarray): data displayed (2 dims)
        clus_fisher (np.ndarray): clustering results of Fisher
        clus_kmeans (np.ndarray): clustering results of k-means
        clus_cah (np.ndarray): clustering results of CAH
        axis_labels (tuple[str, str]): names of the two displayed dimensions
        save_path (str): path of image saving
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    clusters = [clus_fisher, clus_kmeans, clus_cah]
    titles = ["Fisher", "K-means", "CAH"]

    for clus, ax, title in zip(clusters, axes, titles):
        for label in np.unique(clus):
            mask = (clus == label)
            # color given by the (aligned) label, so it is the same in every panel
            ax.scatter(x=X[mask, 0], y=X[mask, 1], color=COLORS[label % len(COLORS)],
                       s = 15, label = f"class {label + 1}")
        ax.set_title(title)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.legend()

    fig.suptitle("Clustering in Space")
    plt.tight_layout()
    show_and_save(fig, save_path)


def plot_time_scatter(X: np.ndarray, cuts_fisher: np.ndarray, clus_fisher: np.ndarray,
                      clus_kmeans: np.ndarray, clus_cah: np.ndarray,
                      axis_labels: tuple[str, str] = ("dim 1", "dim 2"),
                      save_path: Path | None = None):
    """Plot the clustering results for the three models
    along the observation index (x-axis), with the first
    and second displayed dimensions on the y-axis. Fisher's
    cut points and fitted segment means are drawn on its row.

    Args:
        X (np.ndarray): data displayed (2 dims)
        cuts_fisher (np.ndarray): cluster boundaries of Fisher
        clus_fisher (np.ndarray): clustering results of Fisher
        clus_kmeans (np.ndarray): clustering results of k-means
        clus_cah (np.ndarray): clustering results of CAH
        axis_labels (tuple[str, str]): names of the two displayed dimensions
        save_path (str): path of image saving
    """
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))

    n = X.shape[0]
    ids = np.arange(n)
    bornes = [0, *cuts_fisher, n]
    clusters = [clus_fisher, clus_kmeans, clus_cah]
    titles = ["Fisher", "K-means", "CAH"]

    for row, (clus, ax, title) in enumerate(zip(clusters, axes, titles)):
        for label in np.unique(clus):
            mask = (clus == label)
            for j in range(2):
                ax[j].scatter(x=ids[mask], y=X[mask, j], color=COLORS[label % len(COLORS)],
                              s = 15, label = f"class {label + 1}")

        for j in range(2):
            if row == 0:
                # cut between x_{cut-1} and x_cut, and mean of each segment
                # (the display is an affine map of X, so this is the projection
                # of the fitted segment mean)
                for cut in cuts_fisher:
                    ax[j].axvline(x=cut - 0.5, ls="--", c="red", lw=1.5)
                for k in range(len(bornes) - 1):
                    seg_mean = X[bornes[k]:bornes[k + 1], j].mean()
                    ax[j].hlines(seg_mean, bornes[k] - 0.5, bornes[k + 1] - 0.5, colors="black",
                                 lw=2, label="segment mean" if k == 0 else None)

            ax[j].set_title(title)
            ax[j].set_xlabel("observation index")
            ax[j].set_ylabel(axis_labels[j])
            ax[j].legend()

    fig.suptitle("Clustering Along the Observation Index")
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


def compute_inertia_curves(X: np.ndarray, max_k: int, n_init: int,
                           seed: int) -> dict[str, np.ndarray]:
    """Intra-cluster inertia of the three models for K = 1, ..., max_k

    Args:
        X (np.ndarray): Training data
        max_k (int): Maximum number of the clusters (<= n)
        n_init (int): number of K-Means initializations
        seed (int): random_state of K-Means

    Returns:
        dict[str, np.ndarray]: inertia for K = 1..max_k of each model
            ("Fisher" is the criterion C(K))
    """
    check_n_clusters(max_k, X.shape[0])

    curves = {"Fisher": [], "K-means": [], "CAH": []}
    for k in range(1, max_k + 1):
        # fisher
        fisher, _, _ = clustfisher(k, X)
        curves["Fisher"].append(fisher.inertia_)

        # K-means
        k_means = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        k_means_labels = k_means.fit_predict(X)
        curves["K-means"].append(compute_intra_inertia(X, k_means_labels))

        # CAH
        cah = AgglomerativeClustering(n_clusters=k)
        cah_labels = cah.fit_predict(X)
        curves["CAH"].append(compute_intra_inertia(X, cah_labels))

    return {name: np.array(values) for name, values in curves.items()}


def print_fisher_criterion(C: np.ndarray):
    """Print Fisher's criterion C(K) and its decrease from K-1 to K
    (elbow method)

    Args:
        C (np.ndarray): C(K) for K = 1, ..., max_k
    """
    print("Fisher criterion C(K) (elbow method)")
    print(f"{'K':>3} {'C(K)':>12} {'C(K-1)-C(K)':>13} {'decrease':>9}")
    for k, c in enumerate(C, start=1):
        if k == 1:
            print(f"{k:>3} {c:>12.2f} {'-':>13} {'-':>9}")
            continue
        delta = C[k - 2] - c
        rel = delta / C[k - 2] if C[k - 2] > 0 else 0.0
        print(f"{k:>3} {c:>12.2f} {delta:>13.2f} {rel:>9.1%}")


def plot_inertia(curves: dict[str, np.ndarray], save_path: Path | None = None):
    """Plot the intra-cluster inertia for three models
    ranging from 1 cluster to max_k clusters; the optimal
    number of clusters can be found using the elbow method.

    Args:
        curves (dict[str, np.ndarray]): output of compute_inertia_curves
        save_path (str): path of image saving
    """
    fig = plt.figure(figsize=(8, 6))

    for name, values in curves.items():
        x_ticks = np.arange(1, len(values) + 1)
        plt.plot(x_ticks, values, marker="o", label=name)
    plt.xticks(x_ticks)

    plt.title("Intra inertia curve of three clustering methods")
    plt.xlabel("Number of clusters")
    plt.ylabel("Intra inertia")

    plt.legend()
    show_and_save(fig, save_path)


def align_labels(y_ref: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Relabel the clusters so that they best match the reference classes
    (Hungarian algorithm on the contingency table). Clusters left unmatched,
    when y_pred has more groups than y_ref, get new labels after the
    reference ones.

    Args:
        y_ref (np.ndarray): Reference labels (true classes or another partition)
        y_pred (np.ndarray): Cluster labels

    Returns:
        np.ndarray: Relabeled clusters
    """
    ref_cats = np.unique(y_ref)
    pred_cats = np.unique(y_pred)
    table = contingency_matrix(y_ref, y_pred)  # rows: ref_cats, columns: pred_cats
    # maximize the matched counts <=> minimize -table
    row_ind, col_ind = linear_sum_assignment(-table)
    mapping = {int(pred_cats[c]): int(ref_cats[r]) for r, c in zip(row_ind, col_ind)}

    next_label = int(ref_cats.max()) + 1
    for c in pred_cats:
        if int(c) not in mapping:
            mapping[int(c)] = next_label
            next_label += 1
    return np.array([mapping[int(l)] for l in y_pred])


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
        save_path (str): path of image saving
    """
    y_true = y_true.astype(int)
    true_cats = np.unique(y_true)
    titles = ["Fisher", "K-Means", "CAH"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, clus, title in zip(axes, [clus_fisher, clus_kmeans, clus_cah], titles):
        clus = align_labels(y_true, clus)
        sns.heatmap(
            contingency_matrix(y_true, clus),
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[f"class {c + 1}" for c in np.unique(clus)],
            yticklabels=[f"class {c + 1}" for c in true_cats],
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
                        choices=list(DATASETS))
    parser.add_argument("--optim_k", type=int, default=4, help="The optimal number of clusters for Fisher obtaining by elbow method")
    parser.add_argument("--max_k", type=int, default=10, help="Set the maximum number of clusters")
    parser.add_argument("--n_init", type=int, default=20, help="Number of K-Means initializations (the best inertia is kept)")
    parser.add_argument("--seed", type=int, default=42, help="random_state of K-Means")
    parser.add_argument("--save_dir", type=str, default="figures", help="Directory where the figures are saved")

    args = parser.parse_args()
    if args.n_init < 1:
        parser.error(f"--n_init must be >= 1, got {args.n_init}")

    data = load_data(args.data_name)
    if args.data_name == "aiguillage.txt":
        # the 553rd column holds the true classes (1, ..., 4), excluded from fitting
        X, y = data[:, :552], data[:, 552].astype(int) - 1
    else:
        X, y = data, None

    n = X.shape[0]
    for name in ("optim_k", "max_k"):
        value = getattr(args, name)
        if not 1 <= value <= n:
            parser.error(f"--{name} must be between 1 and n = {n}, got {value}")

    # 2D display only: the three models are always fitted on the raw data X
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_display = pca.fit_transform(StandardScaler().fit_transform(X))
        axis_labels = tuple(f"PC{i + 1} ({r:.1%} of variance)"
                            for i, r in enumerate(pca.explained_variance_ratio_))
    else:
        X_display = X
        axis_labels = ("var1", "var2")

    # elbow method on Fisher's criterion C
    curves = compute_inertia_curves(X, args.max_k, args.n_init, args.seed)
    print_fisher_criterion(curves["Fisher"])

    print(f"\nK = {args.optim_k}, K-Means: n_init = {args.n_init}, random_state = {args.seed}")
    k_means = KMeans(n_clusters=args.optim_k, n_init=args.n_init, random_state=args.seed)
    cah = AgglomerativeClustering(n_clusters=args.optim_k)

    _, cuts, clus_f = clustfisher(args.optim_k, X)
    clus_k = k_means.fit_predict(X)
    clus_c = cah.fit_predict(X)

    # same label (and color) = same group in every output: align on the true
    # classes if available, else on Fisher's segments
    reference = y if y is not None else clus_f
    clus_f, clus_k, clus_c = (align_labels(reference, c) for c in (clus_f, clus_k, clus_c))
    methods = {"Fisher": clus_f, "K-Means": clus_k, "CAH": clus_c}

    print("Cut points: zero-based index of the first observation of segments 2..K "
          "(course t = cut + 1)")
    print(f"Fisher cut points: {cuts.tolist()} (t = {(cuts + 1).tolist()}), "
          f"segment lengths: {np.diff([0, *cuts, n]).tolist()}")

    rows = {"Inertia": ([compute_intra_inertia(X, c) for c in methods.values()], ".2f")}
    # Compare results of clustering of 3 models and true label on real data
    if y is not None:
        true_cuts = np.flatnonzero(np.diff(y)) + 1
        print(f"True cut points:   {true_cuts.tolist()} (t = {(true_cuts + 1).tolist()}), "
              f"segment lengths: {np.diff([0, *true_cuts, n]).tolist()}")
        # ARI does not depend on the labels, TBC needs the aligned labels
        rows["ARI"] = ([adjusted_rand_score(y, c) for c in methods.values()], ".3f")
        rows["TBC (%)"] = ([100 * np.mean(y == c) for c in methods.values()], ".2f")

    print(f"\n{'':<10}" + "".join(f"{name:>12}" for name in methods))
    for row_name, (values, fmt) in rows.items():
        print(f"{row_name:<10}" + "".join(f"{v:>12{fmt}}" for v in values))
    print("Cluster sizes (class 1, 2, ...):")
    for name, c in methods.items():
        print(f"  {name:<8} {np.bincount(c).tolist()}")

    # figures/<data name>/xxx.png
    save_dir = PROJECT_ROOT / args.save_dir / Path(args.data_name).stem

    plot_inertia(curves, save_dir / "inertia.png")
    plot_space_scatter(X_display, clus_f, clus_k, clus_c, axis_labels, save_dir / "space_scatter.png")
    plot_time_scatter(X_display, cuts, clus_f, clus_k, clus_c, axis_labels, save_dir / "time_scatter.png")
    if y is not None:
        draw_cm(y, clus_f, clus_k, clus_c, save_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
