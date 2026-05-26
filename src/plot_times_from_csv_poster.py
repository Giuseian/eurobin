from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


# ============================================================
# FONT CONFIGURATION
# ============================================================

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans"],

    # Mantiene il testo modificabile/vettoriale nel PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot inference times from metrics_master.csv"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to metrics_master.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "scenario",
        "scene_time_mean",
        "plan_time_mean",
        "sim_time_mean",
        "validator_time_mean",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    time_cols = [
        "scene_time_mean",
        "plan_time_mean",
        "sim_time_mean",
        "validator_time_mean",
    ]

    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    scenario_means = df.groupby("scenario", as_index=False)[time_cols].mean()

    plot_df = scenario_means.melt(
        id_vars="scenario",
        value_vars=time_cols,
        var_name="module",
        value_name="time",
    )

    module_names = {
        "scene_time_mean": "Scene\nPerception",
        "plan_time_mean": "Manipulation\nPlanning",
        "sim_time_mean": "Execution\nScheduling",
        "validator_time_mean": "Validator",
    }

    plot_df["module"] = plot_df["module"].map(module_names)

    modules = list(module_names.values())
    scenarios = scenario_means["scenario"].unique()

    colors = [
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#E45756",
        "#72B7B2",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 27,
        "xtick.labelsize": 27,
        "ytick.labelsize": 27,
        "legend.fontsize": 25,
    })

    fig, ax = plt.subplots(figsize=(30, 5.5), constrained_layout=False)

    x = list(range(len(modules)))

    for i, scenario in enumerate(scenarios):
        scenario_data = plot_df[plot_df["scenario"] == scenario]

        values = [
            scenario_data.loc[
                scenario_data["module"] == module,
                "time"
            ].values[0]
            for module in modules
        ]

        color = colors[i % len(colors)]

        ax.plot(
            x,
            values,
            marker="o",
            markersize=24,
            linewidth=8,
            color=color,
            label=scenario,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(modules, rotation=0, ha="center")

    ax.set_ylabel("Mean Inference Time [s]")

    # Righe orizzontali stile immagine
    ax.yaxis.grid(True, linestyle="-", linewidth=0.8, alpha=0.35)
    ax.xaxis.grid(False)

    # Asse Y leggermente più pulito
    ax.set_ylim(bottom=0)

    # Rimuove bordi superiori e destri
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            markersize=24,
            linewidth=8,
            color=colors[i % len(colors)],
            markerfacecolor=colors[i % len(colors)],
            markeredgecolor=colors[i % len(colors)],
            label=scenario,
        )
        for i, scenario in enumerate(scenarios)
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        labelspacing=1.2,
    )

    fig.subplots_adjust(
        left=0.06,    # margine sinistro
        right=0.78,   # spazio per la legenda a destra
        bottom=0.22,  # spazio per i nomi sull'asse X
        top=0.95
    )

    output_path = csv_path.parent / "plot_inference_times3.pdf"

    plt.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight"
    )

    plt.show()

    print(f"Plot saved in: {output_path}")


if __name__ == "__main__":
    main()