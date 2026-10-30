"""Plot saved GCN validation associations; no inference or experiment execution.

Run with plotting packages supplied on sys.path by the launcher. This script
does not install packages or alter the machine-learning environment.
"""

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/results_first_diagnosis_v1/training_derived_bins.csv"
DESTINATION = ROOT / "docs/figures/results_first"
CONDITIONS = ("learned", "random")
FEATURES = ("nodes_per_target", "group_size")
COLORS = {"learned": "#236B8E", "random": "#D07B36"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bin_label(feature, lower, upper):
    if feature == "group_size":
        if not lower:
            return "< " + str(int(float(upper)))
        if not upper:
            return "\u2265 " + str(int(float(lower)))
        return str(int(float(lower))) + "\u2013" + str(int(float(upper)) - 1)
    if not lower:
        return "< %.3f" % float(upper)
    if not upper:
        return "\u2265 %.3f" % float(lower)
    return "%.3f\u2013< %.3f" % (float(lower), float(upper))


def percentage_label(value):
    # Three decimals keep the tiny nonzero residual rates visible.
    return ("%.3f%%" if value < 1 else "%.2f%%") % value


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        source_fields = list(reader.fieldnames)
        rows = [dict(row) for row in reader
                if row["cohort"] == "full_baseline"
                and row["condition"] in CONDITIONS
                and row["feature"] in FEATURES]
    assert len(rows) == 16, "Expected exactly four bins for each condition/feature."
    index = {(row["feature"], row["condition"], int(row["bin"])): row
             for row in rows}
    assert len(index) == 16, "Duplicate selected source rows."
    totals = {}
    for feature in FEATURES:
        for condition in CONDITIONS:
            selected = [index[feature, condition, b] for b in range(4)]
            totals[feature, condition] = (
                sum(int(row["target_occurrences"]) for row in selected),
                sum(int(row["gcn_correct_after_attack"]) for row in selected))
            for row in selected:
                count = int(row["gcn_correct_after_attack"])
                denominator = int(row["target_occurrences"])
                assert denominator > 0 and 0 <= count <= denominator
                assert abs(count / denominator - float(row["residual_rate"])) < 1e-12
                row["residual_percent"] = format(100 * count / denominator, ".17g")
                row["bar_value_label"] = percentage_label(float(row["residual_percent"]))
                row["exposure_label"] = "%s / %s" % (format(count, ","), format(denominator, ","))
                row["bin_label"] = bin_label(feature, row["lower_inclusive"], row["upper_exclusive"])
    for condition in CONDITIONS:
        assert totals[FEATURES[0], condition] == totals[FEATURES[1], condition], "Panel totals disagree."
    for feature in FEATURES:
        for b in range(4):
            learned, random = index[feature, "learned", b], index[feature, "random", b]
            assert learned["target_occurrences"] == random["target_occurrences"]
            assert learned["lower_inclusive"] == random["lower_inclusive"]
            assert learned["upper_exclusive"] == random["upper_exclusive"]

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#60717B",
        "axes.labelcolor": "#253641",
        "xtick.color": "#253641",
        "ytick.color": "#253641",
        "text.color": "#253641",
        "svg.fonttype": "none",
        "svg.hashsalt": "monti-results-first-v1",
    })
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 7.5))
    figure.subplots_adjust(left=0.070, right=0.985, bottom=0.325, top=0.735, wspace=0.21)
    figure.suptitle("Where GCN validation residuals remain", x=0.07, y=0.966,
                   ha="left", fontsize=19, fontweight="bold")
    figure.text(0.07, 0.909,
                "Full baseline \u00b7 selected checkpoints \u00b7 seeds 0/1/2 \u00b7 fixed split and GCN detector",
                ha="left", fontsize=11)
    figure.text(0.07, 0.868,
                "Residual = a target occurrence still classified correctly by the detector after attack.",
                ha="left", fontsize=10.5)
    figure.legend(handles=[Patch(facecolor=COLORS[c], edgecolor=COLORS[c],
                                 hatch="" if c == "learned" else "///", label=c.title())
                           for c in CONDITIONS],
                  loc="upper left", bbox_to_anchor=(0.065, 0.843), frameon=False,
                  ncol=2, columnspacing=2.2)

    for axis, feature, title, y_max in zip(
            axes, FEATURES,
            ("A  Allocated injected nodes per target", "B  Target-group size"),
            (18.8, 4.8)):
        x = np.arange(4)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#DCE3E8", linewidth=0.7)
        for condition, offset in (("learned", -0.205), ("random", 0.205)):
            values = [float(index[feature, condition, b]["residual_percent"]) for b in range(4)]
            bars = axis.bar(x + offset, values, width=0.36, color=COLORS[condition],
                            edgecolor=COLORS[condition], linewidth=0.8,
                            hatch="" if condition == "learned" else "///", zorder=3)
            for b, bar in enumerate(bars):
                axis.annotate(index[feature, condition, b]["bar_value_label"],
                              xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                              xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                              fontsize=9, fontweight="normal", color=COLORS[condition])
        labels = []
        for b in range(4):
            learned = index[feature, "learned", b]
            random = index[feature, "random", b]
            labels.append(learned["bin_label"] + "\nL " + learned["exposure_label"]
                          + "\nR " + random["exposure_label"])
        axis.set_xticks(x, labels, fontsize=9, linespacing=1.55)
        axis.tick_params(axis="x", length=0, pad=12)
        axis.set_ylim(0, y_max)
        axis.set_xlim(-0.65, 3.65)
        axis.set_title(title, loc="left", pad=16, fontweight="bold")
        axis.set_ylabel("Residual rate (%)")
        axis.set_yticks([0, 5, 10, 15] if feature == "nodes_per_target" else [0, 1, 2, 3, 4])

    figure.text(0.07, 0.174,
                "Below each bin: L = learned, R = random; residual occurrences / evaluated occurrences.",
                ha="left", fontsize=10)
    figure.text(0.07, 0.137,
                "Bin boundaries come from training metadata; displayed decimals are rounded. Axes use different scales.",
                ha="left", fontsize=9.5)
    figure.text(0.07, 0.094,
                "Exploratory association: degree and budget are confounded. Checkpoints were selected on these validation groups.",
                ha="left", fontsize=9.5)
    figure.text(0.07, 0.057,
                "Counts pool three seed passes; occurrences repeat across groups and seeds. Denominators are exposures, not independent sample sizes.",
                ha="left", fontsize=9.2)

    png = DESTINATION / "validation_residual_associations.png"
    svg = DESTINATION / "validation_residual_associations.svg"
    figure.savefig(png, dpi=240, facecolor="white", metadata={"Software": "MonTi results-first diagnosis plot"})
    figure.savefig(svg, facecolor="white", metadata={"Date": None, "Creator": "MonTi results-first diagnosis plot"})
    plt.close(figure)
    svg.write_bytes(b"\n".join(line.rstrip() for line in svg.read_bytes().splitlines()) + b"\n")

    plotted = DESTINATION / "plotted_data.csv"
    with plotted.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=source_fields + [
            "residual_percent", "bar_value_label", "exposure_label", "bin_label"], lineterminator="\n")
        writer.writeheader()
        for feature in FEATURES:
            for condition in CONDITIONS:
                for b in range(4):
                    writer.writerow(index[feature, condition, b])

    provenance = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "Static descriptive scientific figure; no inference, training, test data, or significance testing.",
        "source": {"path": SOURCE.relative_to(ROOT).as_posix(), "sha256": sha256(SOURCE)},
        "script": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha256(__file__)},
        "selection": {"cohort": "full_baseline", "conditions": list(CONDITIONS),
                      "features": list(FEATURES), "selected_rows": len(rows)},
        "metric": "100 * gcn_correct_after_attack / target_occurrences",
        "denominator": "Group-node occurrence exposures pooled across three seed passes; not independent sample sizes.",
        "limitations": [
            "Exploratory validation association, not causal evidence.",
            "Best checkpoints were selected using the same validation groups.",
            "Degree, budget, and group structure are confounded.",
            "Panel axes use different scales; exact cutoffs are retained in plotted_data.csv.",
            "No fabricated uncertainty intervals, p-values, or causal claims.",
        ],
        "software": {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
                     "environment": "Isolated plotting dependencies supplied by the launcher; ML environment unchanged."},
        "files": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                   "sha256": sha256(path)} for path in (png, svg, plotted)],
        "text_line_endings": "LF; SVG line endings and trailing whitespace normalized before hashing.",
    }
    with (DESTINATION / "provenance.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "created", "destination": str(DESTINATION),
                      "selected_rows": len(rows), "totals_by_condition": {
                          condition: {"exposures": totals[FEATURES[0], condition][0],
                                      "residuals": totals[FEATURES[0], condition][1]}
                          for condition in CONDITIONS}}, indent=2))


if __name__ == "__main__":
    main()
