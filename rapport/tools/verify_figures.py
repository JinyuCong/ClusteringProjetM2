"""Compare the report's plotted marks with fresh runs of the submitted script.

Run after generate_assets.py. Requires the solution's plotting dependencies,
including seaborn. All script output is confined to rapport/build/.
"""
import sys
sys.dont_write_bytecode = True
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
from generate_assets import COLORS, METHODS, REPORT, ROOT, figure_marks


def scatter_comparison(actual, report, time_axis=False, projected=False):
    assert len(actual["scatter"]) == len(report["scatter"]) == 4
    max_error = 0.0
    for c, (left, right) in enumerate(zip(actual["scatter"], report["scatter"])):
        assert left["label"] == f"class {c + 1}" and right["label"] == str(c + 1)
        expected = np.array(left["xy"])
        if time_axis:
            expected[:, 0] += 1  # report labels observations from 1, script from 0
        if projected:
            # The script's auto solver uses randomized SVD on this 140x552
            # dataset; the report uses deterministic full SVD. Allow only
            # subpixel numerical differences, with labels and indices exact.
            if time_axis:
                np.testing.assert_array_equal(np.array(right["xy"])[:, 0], expected[:, 0])
            np.testing.assert_allclose(right["xy"], expected, rtol=1e-5, atol=1e-4)
        else:
            np.testing.assert_array_equal(right["xy"], expected)
        max_error = max(max_error, float(np.max(np.abs(np.array(right["xy"]) - expected))))
        np.testing.assert_allclose(right["colors"][0], to_rgba(COLORS[c], alpha=.8))
    return max_error


def main():
    evidence = json.loads((REPORT / "figure-provenance.json").read_text(encoding="utf-8"))
    for name, digest in evidence["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    plots = evidence["plots"]
    expected_names = {"simulation_coude", "simulation_espace", "simulation_temps",
                      "aiguillage_coude", "aiguillage_espace", "aiguillage_temps",
                      "aiguillage_confusion", "aiguillage_ordre"}
    assert set(plots) == expected_names
    used = set()
    for path in (REPORT / "sections").glob("*.tex"):
        used.update(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\.pdf\}",
                               path.read_text(encoding="utf-8")))
    assert used == expected_names, (used, expected_names)
    spec = importlib.util.spec_from_file_location("submitted_fisher", ROOT / "fisherDP.py")
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)
    solution.PROJECT_ROOT = ROOT
    plt.rcdefaults()
    plt.show = lambda: None
    destination = REPORT / "build" / "script-figure-check"
    destination.mkdir(parents=True, exist_ok=True)
    original_save = solution.show_and_save
    original_pca = solution.PCA
    pca_models = []

    def record_pca(*args, **kwargs):
        model = original_pca(*args, **kwargs)
        pca_models.append(model)
        return model

    solution.PCA = record_pca
    checks = {}
    argv = sys.argv[:]
    try:
        for dataset, prefix in (("sequencesimu", "simulation"), ("aiguillage", "aiguillage")):
            captured = {}
            projected = dataset == "aiguillage"
            max_coordinate_error = 0.0

            def capture(fig, save_path=None):
                captured[Path(save_path).stem] = figure_marks(fig)
                original_save(fig, save_path)

            solution.show_and_save = capture
            sys.argv = ["fisherDP.py", "--data_name", dataset + ".txt",
                        "--save_dir", str(destination)]
            console = io.StringIO()
            with redirect_stdout(console):
                solution.main()
            (destination / dataset / "console.txt").write_text(console.getvalue(), encoding="utf-8")
            (destination / dataset / "plotted-marks.json").write_text(
                json.dumps(captured, indent=2) + "\n", encoding="utf-8")
            actual = captured["inertia"]["axes"][0]
            report = plots[prefix + "_coude"]["axes"][0]
            assert len(actual["lines"]) == 3 and len(report["lines"]) == 4
            for left, right in zip(actual["lines"], report["lines"][:3]):
                np.testing.assert_allclose(right["xy"], left["xy"], rtol=1e-7, atol=1e-6)
            assert all(point[0] == 4 for point in report["lines"][3]["xy"])
            for left, right in zip(captured["space_scatter"]["axes"], plots[prefix + "_espace"]["axes"]):
                max_coordinate_error = max(max_coordinate_error, scatter_comparison(left, right, projected=projected))
            for left, right in zip(captured["time_scatter"]["axes"], plots[prefix + "_temps"]["axes"]):
                max_coordinate_error = max(max_coordinate_error, scatter_comparison(left, right, time_axis=True, projected=projected))
                assert len(left["lines"]) == len(right["lines"])
                for line_actual, line_report in zip(left["lines"], right["lines"]):
                    np.testing.assert_allclose(np.array(line_report["xy"])[:, 0],
                                               np.array(line_actual["xy"])[:, 0] + 1)
                means = np.asarray(left["segments"])
                if len(means):
                    means[:, :, 0] += 1
                    np.testing.assert_allclose(right["segments"], means,
                                               rtol=1e-5 if projected else 1e-6,
                                               atol=1e-4 if projected else 1e-6)
                else:
                    assert not right["segments"]
            if dataset == "aiguillage":
                heatmaps = [ax for ax in captured["confusion_matrix"]["axes"]
                            if ax["heatmaps"] and np.asarray(ax["heatmaps"][0]).shape == (4, 4)]
                assert len(heatmaps) == 3
                for actual_cm, report_cm in zip(heatmaps, plots["aiguillage_confusion"]["axes"]):
                    np.testing.assert_array_equal(report_cm["images"][0]["values"], actual_cm["heatmaps"][0])
                    assert report_cm["images"][0]["range"] == [0, 45]
                data = solution.load_data("aiguillage.txt")
                expected_rows = [data[:, 552].astype(int) - 1]
                for row in range(3):
                    labels = np.full(len(data), -1, dtype=int)
                    axis = captured["time_scatter"]["axes"][2 * row]
                    for c, collection in enumerate(axis["scatter"]):
                        ids = np.asarray(collection["xy"])[:, 0].astype(int)
                        assert np.all(labels[ids] == -1)
                        labels[ids] = c
                    assert np.all(labels >= 0)
                    expected_rows.append(labels)
                image = plots["aiguillage_ordre"]["axes"][0]["images"][0]
                np.testing.assert_array_equal(image["values"], expected_rows)
                assert image["extent"] == [.5, 140.5, 3.5, -.5]
            checks[dataset] = {"script_figures_checked": len(captured),
                               "sample_count": sum(len(c["xy"]) for c in captured["space_scatter"]["axes"][0]["scatter"]),
                               "curves": "match", "scatter_coordinates_and_labels": "match",
                               "boundaries_and_segment_means": "match",
                               "max_absolute_display_coordinate_difference": max_coordinate_error}
            if projected:
                assert len(pca_models) == 1
                model = pca_models[0]
                reference = json.loads((REPORT / "resultats.json").read_text(encoding="utf-8"))
                explained = reference["datasets"]["aiguillage"]["pca_explained_percent"]
                assert [f"{float(value):.1%}" for value in model.explained_variance_ratio_] == [
                    f"{value:.1f}%" for value in explained]
                checks[dataset]["pca_comparison"] = {
                    "script_solver": model._fit_svd_solver, "report_solver": evidence["pca_solver"],
                    "relative_tolerance": 1e-5, "absolute_tolerance": 1e-4,
                    "rounded_explained_variance_matches": True}
    finally:
        sys.argv = argv
        solution.show_and_save = original_save
        solution.PCA = original_pca
    output = {"source_commit": evidence["source_commit"], "sha256": evidence["sha256"],
              "parameters": evidence["parameters"], "datasets": checks,
              "primary_figures_checked": 7, "supplementary_figures_checked": 1,
              "all_eight_figures_included_in_report": True}
    (REPORT / "build" / "figure-validation.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Passed: all seven script figure types match report plots; the supplementary label timeline also matches.")


if __name__ == "__main__":
    main()
