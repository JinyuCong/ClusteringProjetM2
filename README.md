# Clustering under order constraint — Fisher's dynamic programming

Master MLSD project (2026/2027). The goal is to split a sequence of observations
(x_1, ..., x_n) into K **contiguous** segments, as homogeneous as possible.

## Report

- [Final report (PDF)](rapport/rapport.pdf)
- [Editable Overleaf project (ZIP)](rapport/overleaf.zip)
- [Editing, figure regeneration and verification guide](rapport/README.md)

The French report presents the methodology, results and sensitivity to K-Means
initialization and scaling, with **8 figures and 5 tables**. It has **7 numbered
content pages, including references**, plus 5 preliminary pages.

For Overleaf, import the ZIP as an existing project, select `main.tex` and use
**pdfLaTeX**. To rebuild the PDF and ZIP locally with Python and pdfLaTeX installed:

```bash
python rapport/build.py
```

The report's vector figures are checked against the script's outputs. The linked
guide explains how to regenerate them and verify the numerical results.

## What was done

- **Fisher's algorithm** implemented from scratch in `fisherDP.py` (`FisherCluster`,
  `clustfisher`, `diam`). It minimizes the intra-segment inertia criterion C exactly
  (global optimum), which was checked against a brute-force search on small inputs.
- **Comparison** with two unconstrained methods: K-Means (20 initializations,
  local optimum) and hierarchical clustering (CAH, Ward linkage, greedy).
- **Evaluation**: elbow method on C to choose K, ARI, correct classification rate
  (TBC) and confusion matrices against the true labels (cluster labels aligned with
  the Hungarian algorithm), and plots in space (standardized PCA when p > 2, display
  only) and along the observation index.

## Results

Cut points are zero-based indices of the first observation of segments 2..K
(the course's one-based t is cut + 1).

**`sequencesimu`** (160 points in R², simulated): the elbow of C is at **K = 4**,
and Fisher finds the cut points **[50, 80, 130]** (t = [51, 81, 131]).

| K           | 1     | 2     | 3     | **4**     | 5     | 6     |
|-------------|-------|-------|-------|-----------|-------|-------|
| C(K)        | 884.1 | 503.6 | 462.0 | **300.6** | 292.4 | 285.5 |
| decrease    | -     | 43.0% | 8.3%  | **34.9%** | 2.7%  | 2.4%  |

**`aiguillage`** (140 points in R⁵⁵², 4 true classes), K = 4:

|     | Fisher    | K-Means | CAH    |
|-----|-----------|---------|--------|
| ARI | **1.000** | 0.943   | 0.859  |
| TBC | **100%**  | 97.86%  | 94.29% |

Fisher recovers the true cut points [35, 75, 120] exactly. K-Means misplaces 3
points between classes 1 and 2, and CAH misplaces 8. Classes 3 and 4 are
recovered by all three methods. Only classes 1 and 2, which are adjacent in time
and overlap in feature space, are mixed by the methods that ignore the order.
With a single K-Means initialization the ARI drops to 0.558, so the gap between
the methods depends on K-Means reaching a good local optimum.

## Usage

Requirements: Python 3.11+, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux/macOS: source .venv/bin/activate)
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
| `--n_init`    | `20`               | number of K-Means initializations             |
| `--seed`      | `42`               | random_state of K-Means                       |
| `--save_dir`  | `figures`          | output directory for the figures              |

`--optim_k` and `--max_k` must be between 1 and n, and `--n_init` at least 1.
Invalid inputs (K out of range, non-finite or empty data, unsupported file)
raise an error instead of returning an invalid segmentation.

Used as a module:

```python
from fisherDP import load_data, clustfisher, diam

X = load_data("sequencesimu.txt")
fisher, cuts, labels = clustfisher(4, X)  # cuts = [50, 80, 130], labels in 0..3
fisher.inertia_                           # optimal criterion C
diam(X, 0, 49)                            # D(a, b) of the first segment
```

The console shows Fisher's criterion C(K) and its decrease (elbow method), the
cut points and segment lengths, the cluster sizes and inertia of each method,
and for aiguillage the ARI, the TBC and the true cut points.
The figures are shown and saved in `figures/<dataset>/`: `inertia.png`,
`space_scatter.png`, `time_scatter.png`, and `confusion_matrix.png` (aiguillage only).

## Files

- `fisherDP.py`: implementation, plots and experiments
- `sequencesimu.txt`, `aiguillage.txt`: datasets
- `requirements.txt`: Python dependencies
- `figures/`: saved figures
- [`rapport/`](rapport/): report PDF, editable LaTeX sources, Overleaf ZIP and reproduction tools
- `projet.pdf`: project statement
