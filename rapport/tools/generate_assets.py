"""Recompute report figures and tables without changing the submitted solution.

Run from any directory: python rapport/tools/generate_assets.py
Requires numpy, scipy, scikit-learn and matplotlib. The Fisher definitions are
loaded unchanged from fisherDP.py using AST so its display-only imports and CLI
do not run. Original data, source code and figures are never written.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "clustering-report-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection, PathCollection, QuadMesh
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parent
FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"
for folder in (FIGURES, TABLES):
    folder.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "pdf.fonttype": 42,
    "text.color": "#19334D", "axes.labelcolor": "#334155",
    "axes.titlecolor": "#19334D", "axes.titleweight": "semibold",
    "axes.edgecolor": "#CBD5E1", "axes.linewidth": .7,
    "axes.facecolor": "#F8FAFC", "axes.axisbelow": True,
    "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.grid": True, "grid.color": "#CBD5E1", "grid.alpha": .45,
    "grid.linewidth": .5,
    "savefig.bbox": "tight", "savefig.pad_inches": .06,
})
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
MARKERS = ["o", "s", "^", "D"]
METHODS = ("Fisher", "K-Means", "CAH (Ward)")
PLOT_DATA = {}


def fisher_function():
    tree = ast.parse((ROOT / "fisherDP.py").read_text(encoding="utf-8"))
    names = {"check_n_clusters", "check_data", "diam", "FisherCluster", "clustfisher",
             "load_data", "align_labels", "compute_intra_inertia", "compute_inertia_curves"}
    selected = ast.Module(body=[node for node in tree.body
                               if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                               and node.name in names], type_ignores=[])
    scope = {"np": np, "Path": Path, "PROJECT_ROOT": ROOT,
             "DATASETS": ("sequencesimu.txt", "aiguillage.txt"),
             "KMeans": KMeans, "AgglomerativeClustering": AgglomerativeClustering,
             "contingency_matrix": contingency_matrix, "linear_sum_assignment": linear_sum_assignment}
    exec(compile(selected, str(ROOT / "fisherDP.py"), "exec"), scope)
    # Read CLI defaults from the implementation rather than duplicate them.
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                for kw in node.keywords:
                    if kw.arg == "default":
                        defaults[node.args[0].value.lstrip("-")] = ast.literal_eval(kw.value)
    return scope, defaults


def inertia(X, labels):
    # Match the solution's direct float32 segment/cluster calculation.
    return float(sum(((X[labels == c] - X[labels == c].mean(axis=0)) ** 2).sum()
                     for c in np.unique(labels)))


def figure_marks(fig):
    # Record the actual plotted marks for comparison with a fresh script run.
    axes_data = []
    for ax in fig.axes:
        axes_data.append({
            "title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
            "lines": [{"label": line.get_label(), "xy": line.get_xydata().tolist()}
                      for line in ax.lines],
            "scatter": [{"label": c.get_label(), "xy": np.asarray(c.get_offsets()).tolist(),
                         "colors": c.get_facecolors().tolist()}
                        for c in ax.collections if isinstance(c, PathCollection)],
            "segments": [segment.tolist() for c in ax.collections if isinstance(c, LineCollection)
                         for segment in c.get_segments()],
            "images": [{"values": np.asarray(im.get_array()).tolist(),
                        "extent": list(im.get_extent()), "range": list(im.get_clim())}
                       for im in ax.images],
            "heatmaps": [np.asarray(c.get_array()).tolist() for c in ax.collections
                         if isinstance(c, QuadMesh)]})
    return {"axes": axes_data}


def save(fig, name):
    PLOT_DATA[name] = figure_marks(fig)
    fig.savefig(FIGURES / (name + ".pdf"), metadata={"Creator": "generate_assets.py"})
    plt.close(fig)


def table(name, headings, rows, spec):
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule",
             " & ".join(headings) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLES / (name + ".tex")).write_text("\n".join(lines), encoding="utf-8")


def french(value, decimals=3):
    return f"{value:.{decimals}f}".replace(".", "{,}")


def space_plot(X, predictions, name, axis_names, true_labels=False):
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55), sharex=True, sharey=True,
                             layout="constrained")
    for ax, labels, method in zip(axes, predictions, METHODS):
        for c in range(4):
            mask = labels == c
            ax.scatter(*X[mask].T, s=17, c=COLORS[c], marker=MARKERS[c], alpha=.8,
                       edgecolors="white", linewidths=.25, label=str(c + 1))
        ax.set_title(method)
        ax.set_xlabel(axis_names[0])
    axes[0].set_ylabel(axis_names[1])
    axes[-1].legend(title="Classe alignée" if true_labels else "Aligné sur Fisher",
                    loc="best", fontsize=7, title_fontsize=7, facecolor="white",
                    edgecolor="#E2E8F0", framealpha=.95)
    save(fig, name)


def inertia_plot(curves, name):
    fig, ax = plt.subplots(figsize=(7.05, 2.35), layout="constrained")
    for index, (method, color, marker) in enumerate(zip(METHODS, COLORS, MARKERS)):
        values = curves[method]
        ax.plot(range(1, len(values) + 1), values, label=method, color=color, marker=marker,
                markersize=4, markeredgecolor="white", markeredgewidth=.55,
                linewidth=1.9 if index == 0 else 1.35)
    ax.axvline(4, color="#64748B", linestyle="--", linewidth=.85)
    ax.set(xlabel="Nombre de groupes K", ylabel="Inertie intra-groupe",
           xticks=range(1, len(curves["Fisher"]) + 1), ylim=(0, None))
    ax.text(4.15, .92, "K = 4", transform=ax.get_xaxis_transform(), fontsize=8,
            fontweight="semibold", va="top",
            bbox={"boxstyle": "round,pad=.3", "facecolor": "#E8EFF6", "edgecolor": "none"})
    if name == "aiguillage_coude":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:.0f}"))
        ax.set_ylabel(r"Inertie intra-groupe ($10^6$)")
    ax.legend(frameon=False, ncol=3)
    save(fig, name)


def time_plot(X, predictions, cuts, name, axis_names):
    fig, axes = plt.subplots(3, 2, figsize=(7.05, 3.65), sharex=True, sharey="col",
                             layout="constrained")
    indices = np.arange(1, len(X) + 1)
    for row, (method, labels) in enumerate(zip(METHODS, predictions)):
        for dim in range(2):
            ax = axes[row, dim]
            ax.grid(visible=False, axis="x")
            for c in range(4):
                mask = labels == c
                ax.scatter(indices[mask], X[mask, dim], s=8, color=COLORS[c],
                           marker=MARKERS[c], alpha=.8, edgecolors="none", label=str(c + 1))
            if row == 0:
                bounds = [0, *cuts, len(X)]
                for c, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
                    ax.axvspan(a + .5, b + .5, color=COLORS[c], alpha=.055, zorder=0)
                for cut in cuts:
                    ax.axvline(cut + .5, color="#64748B", linestyle="--", linewidth=.85)
                for a, b in zip(bounds[:-1], bounds[1:]):
                    ax.hlines(X[a:b, dim].mean(), a + .5, b + .5, color="#19334D", linewidth=1.5)
                ax.set_title(axis_names[dim])
            if dim == 0:
                ax.set_ylabel(method)
            if row == 2:
                ax.set_xlabel("Indice de l'observation (base 1)")
    save(fig, name)


def main():
    scope, defaults = fisher_function()
    clustfisher = scope["clustfisher"]
    aligned = scope["align_labels"]
    n_init, seed = defaults["n_init"], defaults["seed"]
    results = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "parameters": defaults,
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib")},
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in ("fisherDP.py", "sequencesimu.txt", "aiguillage.txt")},
        "datasets": {},
    }
    labels_by_dataset = {}
    for name in ("sequencesimu", "aiguillage"):
        data = scope["load_data"](name + ".txt")
        X, y = (data[:, :552], data[:, 552].astype(int) - 1) if name == "aiguillage" else (data, None)
        model, cuts, fisher_labels = clustfisher(4, X)
        km = KMeans(n_clusters=4, random_state=seed, n_init=n_init).fit(X)
        ward_labels = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X)
        reference = fisher_labels if y is None else y
        predictions = [aligned(reference, labels) for labels in (fisher_labels, km.labels_, ward_labels)]
        d = {"shape": list(X.shape), "cuts_zero_based": cuts.tolist(),
             "cuts_one_based": (cuts + 1).tolist(), "fisher_criterion": float(model.inertia_),
             "models": {}}
        for method, labels in zip(METHODS, predictions):
            item = {"inertia": inertia(X, labels),
                    "adjacent_changes": int(np.count_nonzero(np.diff(labels)))}
            if y is not None:
                matched = aligned(y, labels)
                item.update(ari=float(adjusted_rand_score(y, labels)),
                            correct=int(np.count_nonzero(y == matched)),
                            accuracy=float(np.mean(y == matched)),
                            confusion=confusion_matrix(y, matched).tolist())
            d["models"][method] = item
        labels_by_dataset[name] = (X, y, predictions, cuts)
        results["datasets"][name] = d

    X, _, pred, cuts = labels_by_dataset["sequencesimu"]
    actual_curves = scope["compute_inertia_curves"](X, 10, n_init, seed)
    curves = {method: actual_curves[key].tolist()
              for method, key in zip(METHODS, ("Fisher", "K-means", "CAH"))}
    results["datasets"]["sequencesimu"]["curves"] = curves
    inertia_plot(curves, "simulation_coude")
    space_plot(X, pred, "simulation_espace", ("Variable 1", "Variable 2"))
    time_plot(X, pred, cuts, "simulation_temps", ("Variable 1", "Variable 2"))
    table("simulation_cout", ["K", "1", "2", "3", r"\textbf{4}", "5", "6"],
          [["C"] + [french(v, 2) for v in curves["Fisher"][:6]]], "lrrrrrr")
    table("simulation_comparaison", ["Méthode", "Inertie", "Changements voisins"],
          [[method, french(results["datasets"]["sequencesimu"]["models"][method]["inertia"], 2),
            str(results["datasets"]["sequencesimu"]["models"][method]["adjacent_changes"])]
           for method in METHODS], "lrr")

    X, y, pred, cuts = labels_by_dataset["aiguillage"]
    actual_curves = scope["compute_inertia_curves"](X, 10, n_init, seed)
    real_curves = {method: actual_curves[key].tolist()
                   for method, key in zip(METHODS, ("Fisher", "K-means", "CAH"))}
    results["datasets"]["aiguillage"]["curves"] = real_curves
    inertia_plot(real_curves, "aiguillage_coude")
    matched = [aligned(y, labels) for labels in pred]
    pca = PCA(n_components=2, svd_solver="full")
    reduced = pca.fit_transform(StandardScaler().fit_transform(X))
    explained = pca.explained_variance_ratio_ * 100
    results["datasets"]["aiguillage"]["pca_explained_percent"] = explained.tolist()
    space_plot(reduced, matched, "aiguillage_espace",
               (f"PC1 ({explained[0]:.1f} %)", f"PC2 ({explained[1]:.1f} %)"), True)
    time_plot(reduced, matched, cuts, "aiguillage_temps",
              (f"PC1 ({explained[0]:.1f} %)", f"PC2 ({explained[1]:.1f} %)"))
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35), layout="constrained")
    for ax, labels, method in zip(axes, matched, METHODS):
        cm = confusion_matrix(y, labels)
        image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=45)
        for (i, j), value in np.ndenumerate(cm):
            error = i != j and value > 0
            if error:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor="#B86A13", linewidth=1.35))
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > 22 else ("#9A4F07" if error else "#19334D"),
                    fontsize=10, fontweight="semibold" if error else "normal")
        ax.set(title=method, xlabel="Classe prédite (alignée)",
               xticks=range(4), yticks=range(4), xticklabels=range(1, 5), yticklabels=range(1, 5))
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Classe de référence")
    bar = fig.colorbar(image, ax=axes, fraction=.025, pad=.025, shrink=.82, ticks=[0, 15, 30, 45])
    bar.ax.set_ylabel("Effectif", fontsize=8)
    bar.ax.tick_params(labelsize=7, length=2)
    bar.outline.set_visible(False)
    save(fig, "aiguillage_confusion")
    fig, ax = plt.subplots(figsize=(7.05, 1.65), layout="constrained")
    ax.imshow(np.stack([y, *matched]), aspect="auto", interpolation="nearest",
              cmap=ListedColormap(COLORS), vmin=0, vmax=3,
              extent=(.5, len(y) + .5, 3.5, -.5))
    ax.set(yticks=range(4), yticklabels=["Référence", *METHODS],
           xlabel="Indice de l'observation (base 1)")
    ax.grid(False)
    for boundary in (.5, 1.5, 2.5):
        ax.axhline(boundary, color="white", linewidth=1.1)
    ax.set_xticks([1, 20, 40, 60, 80, 100, 120, 140])
    ax.tick_params(axis="y", length=0)
    for cut in cuts:
        ax.axvline(cut + .5, color="white", linewidth=.8, linestyle="--")
    save(fig, "aiguillage_ordre")
    table("aiguillage_comparaison", ["Méthode", "ARI", "Corrects", "TBC", "Changements"],
          [[method, french(item["ari"]), f"{item['correct']}/140",
            french(item["accuracy"] * 100, 2) + r"\,\%", str(item["adjacent_changes"])]
           for method, item in results["datasets"]["aiguillage"]["models"].items()], "lrrrr")
    sensitivity = []
    for starts in (1, 10, 20, 50):
        km = KMeans(n_clusters=4, random_state=seed, n_init=starts).fit(X)
        labels = aligned(y, km.labels_)
        sensitivity.append({"n_init": starts, "ari": float(adjusted_rand_score(y, km.labels_)),
                            "accuracy": float(np.mean(y == labels)), "inertia": float(km.inertia_)})
    results["kmeans_sensitivity"] = sensitivity
    table("kmeans_sensibilite", [r"\texttt{n\_init}", "ARI", "TBC", r"Inertie ($10^6$)"],
          [[str(r["n_init"]), french(r["ari"]), french(r["accuracy"] * 100, 2) + r"\,\%",
            french(r["inertia"] / 1e6, 4)] for r in sensitivity], "rrrr")
    supplementary = {"source_commit": results["source_commit"], "sha256": results["sha256"],
                     "parameters": {"n_init": n_init, "seed": seed},
                     "profile": {}, "scaling": {}, "seed_runs": []}
    for name, (data, target, _, _) in labels_by_dataset.items():
        _, counts = np.unique(data, axis=0, return_counts=True)
        supplementary["profile"][name] = {
            "rows": len(data), "features": data.shape[1],
            "nonfinite": int(np.count_nonzero(~np.isfinite(data))),
            "distinct_rows": len(counts), "duplicate_excess": int(np.sum(counts - 1)),
            "duplicate_groups": int(np.count_nonzero(counts > 1)),
            "constant_features": int(np.count_nonzero(np.ptp(data, axis=0) == 0))}
    for label, data in (("Brutes", X), ("Standardisées", StandardScaler().fit_transform(X))):
        _, boundaries, labels = clustfisher(4, data)
        supplementary["scaling"][label] = {
            "fisher_cuts_zero": boundaries.tolist(), "fisher_ari": float(adjusted_rand_score(y, labels)),
            "kmeans_ari": float(adjusted_rand_score(y, KMeans(n_clusters=4, n_init=n_init, random_state=seed).fit_predict(data))),
            "ward_ari": float(adjusted_rand_score(y, AgglomerativeClustering(n_clusters=4).fit_predict(data)))}
    for run_seed in range(10):
        km = KMeans(n_clusters=4, n_init=n_init, random_state=run_seed).fit(X)
        supplementary["seed_runs"].append({"seed": run_seed, "ari": float(adjusted_rand_score(y, km.labels_)),
                                            "inertia": float(km.inertia_)})
    table("scaling", ["Données", "Fisher", f"K-Means ({n_init} départs)", "Ward"],
          [[label, french(row["fisher_ari"]), french(row["kmeans_ari"]), french(row["ward_ari"])]
           for label, row in supplementary["scaling"].items()], "lrrr")
    (REPORT / "analyses-complementaires.json").write_text(
        json.dumps(supplementary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (REPORT / "resultats.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance = {"source_commit": results["source_commit"], "sha256": results["sha256"],
                  "parameters": {"K": 4, "max_k": 10, "n_init": n_init, "seed": seed},
                  "display_index_base": 1, "pca_solver": "full", "plots": PLOT_DATA}
    (REPORT / "figure-provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (TABLES / "metrics.tex").write_text(
        r"\newcommand{\PcaOne}{" + french(explained[0], 1) + "}\n" +
        r"\newcommand{\PcaTwo}{" + french(explained[1], 1) + "}\n", encoding="utf-8")
    print("Regenerated report figures, tables, main results and supplementary evidence.")
    print("PCA explained variance (%):", explained.tolist())


if __name__ == "__main__":
    main()
