# Clustering under order constraint — Fisher's dynamic programming

Master MLSD project (2026/2027). The goal is to split a sequence of observations
(x_1, ..., x_n) into K **contiguous** segments, as homogeneous as possible.

## What was done

- **Fisher's algorithm** implemented from scratch in `fisherDP.py` (`FisherCluster`,
  `clustfisher`, `diam`). It minimizes the intra-segment inertia criterion C exactly
  (global optimum), which was checked against a brute-force search on small inputs.
- **Comparison** with two unconstrained methods: K-Means and hierarchical clustering
  (CAH, Ward linkage).
- **Evaluation**: elbow method on C to choose K, ARI and confusion matrices against
  the true labels (cluster labels aligned with the Hungarian algorithm), and plots
  in space (PCA when p > 2) and along the time axis.

## Results

**`sequencesimu`** (160 points in R², simulated): the elbow of C is at **K = 4**,
and Fisher finds the cut points at indices **[50, 80, 130]**.

| K | 1 | 2 | 3 | **4** | 5 | 6 |
|---|---|---|---|---|---|---|
| C | 884.1 | 503.6 | 462.0 | **300.6** | 292.4 | 285.5 |

**`aiguillage`** (140 points in R⁵⁵², 4 true classes):

|                  | Fisher | K-Means | CAH   |
|------------------|--------|---------|-------|
| ARI              | **1.000** | 0.558 | 0.859 |
| Accuracy (aligned) | **100%** | 64%  | 94%   |

Fisher recovers the true cut points [35, 75, 120] exactly. K-Means merges
classes 1 and 2 and splits class 3. CAH mixes a few points between classes 1 and 2.
Using the time order lets Fisher separate neighbouring segments that are close in
feature space. The methods that ignore the order struggle there.

## Usage

Requirements: Python 3.10+, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`.

```bash
pip install -r requirements.txt
```

Run from the project directory (the data files are read from the current directory):

```bash
python fisherDP.py                                   # sequencesimu, K = 4
python fisherDP.py --data_name aiguillage.txt        # real data, K = 4
```

| Option        | Default            | Description                                   |
|---------------|--------------------|-----------------------------------------------|
| `--data_name` | `sequencesimu.txt` | `sequencesimu.txt` or `aiguillage.txt`        |
| `--optim_k`   | `4`                | number of clusters K for the three models     |
| `--max_k`     | `10`               | largest K on the inertia (elbow) curve        |
| `--save_dir`  | `figures`          | output directory for the figures              |

The figures are shown and saved in `figures/<dataset>/`: `inertia.png`,
`space_scatter.png`, `time_scatter.png`, and `confusion_matrix.png` (aiguillage only).
ARI scores and cut points are printed to the console.
