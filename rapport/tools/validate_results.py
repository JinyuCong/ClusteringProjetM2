"""Verify the numerical claims in the report without editing the solution.

Run after generate_assets.py: python rapport/tools/validate_results.py
"""
import sys
sys.dont_write_bytecode = True
import hashlib
from itertools import combinations
import json
import numpy as np
from generate_assets import REPORT, ROOT, fisher_function


def main():
    scope, defaults = fisher_function()
    fisher = scope["clustfisher"]
    rng = np.random.default_rng(1024)
    cases = 0
    for n in range(1, 10):
        for p in (1, 2, 4):
            for dtype in (np.float32, np.float64):
                X = rng.normal(size=(n, p)).astype(dtype)
                for k in range(1, n + 1):
                    model, cuts, labels = fisher(k, X)
                    costs = []
                    for candidate in combinations(range(1, n), k - 1):
                        bounds = (0, *candidate, n)
                        costs.append(sum(float(((X[a:b] - X[a:b].mean(axis=0)) ** 2).sum())
                                         for a, b in zip(bounds[:-1], bounds[1:])))
                    bounds = [0, *cuts, n]
                    assert np.all(np.diff(bounds) > 0)
                    assert len(cuts) == k - 1 and len(labels) == n
                    expected_labels = np.repeat(np.arange(k), np.diff(bounds))
                    np.testing.assert_array_equal(labels, expected_labels)
                    recovered = sum(float(((X[labels == c] - X[labels == c].mean(axis=0)) ** 2).sum())
                                    for c in range(k))
                    np.testing.assert_allclose(model.inertia_, min(costs), rtol=2e-6, atol=1e-6)
                    np.testing.assert_allclose(model.inertia_, recovered, rtol=2e-6, atol=1e-6)
                    cases += 1
    invalid_cases = [(k, np.ones((3, 2))) for k in (True, 1.5, 0, -1, 4)]
    invalid_cases += [(1, X) for X in (np.ones(3), np.empty((0, 2)), np.empty((2, 0)),
                                     np.array([[np.nan]]), np.array([[np.inf]]), np.array([["x"]]))]
    for k, X in invalid_cases:
        try:
            fisher(k, X)
        except ValueError:
            continue
        raise AssertionError(f"Invalid input accepted: K={k}, shape={X.shape}")
    np.testing.assert_array_equal(scope["align_labels"](np.array([10, 10, 30, 30]),
                                                       np.array([-2, -2, 7, 7])), [10, 10, 30, 30])
    aligned = scope["align_labels"](np.array([0, 0, 1, 1]), np.array([5, 6, 7, 7]))
    assert len(np.unique(aligned)) == 3
    results = json.loads((REPORT / "resultats.json").read_text(encoding="utf-8"))
    assert results["parameters"] == defaults
    for name, digest in results["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    residuals = {}
    for name in ("sequencesimu", "aiguillage"):
        data = scope["load_data"](name + ".txt")
        X = data[:, :552] if name == "aiguillage" else data
        model, cuts, labels = fisher(4, X)
        assert cuts.tolist() == results["datasets"][name]["cuts_zero_based"]
        assert model.inertia_ == results["datasets"][name]["fisher_criterion"]
        km = scope["KMeans"](n_clusters=4, n_init=defaults["n_init"], random_state=defaults["seed"]).fit_predict(X)
        ward = scope["AgglomerativeClustering"](n_clusters=4).fit_predict(X)
        residuals[name] = {}
        for method, partition in zip(("Fisher", "K-Means", "CAH (Ward)"), (labels, km, ward)):
            Y = X.astype(np.float64)
            mean = Y.mean(axis=0)
            total = ((Y - mean) ** 2).sum()
            within = sum(((Y[partition == c] - Y[partition == c].mean(axis=0)) ** 2).sum()
                         for c in np.unique(partition))
            between = sum(np.sum(partition == c) * ((Y[partition == c].mean(axis=0) - mean) ** 2).sum()
                          for c in np.unique(partition))
            residual = float(abs(total - within - between) / total)
            assert residual < 1e-10
            residuals[name][method] = residual
            stored = results["datasets"][name]["models"][method]
            assert stored["adjacent_changes"] == np.count_nonzero(np.diff(partition))
            assert stored["inertia"] == float(scope["compute_intra_inertia"](X, partition))
    evidence = {"source_commit": results["source_commit"], "exhaustive_cases": cases,
                "invalid_inputs_rejected": len(invalid_cases), "alignment_checks": 2,
                "huygens_relative_residuals": residuals, "sha256": results["sha256"]}
    (REPORT / "build").mkdir(exist_ok=True)
    (REPORT / "build" / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Passed: {cases} exhaustive cases, {len(invalid_cases)} invalid inputs, label alignment, source hashes and Huygens.")


if __name__ == "__main__":
    main()
