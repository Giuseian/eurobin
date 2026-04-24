from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

    # Colonne necessarie
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

    # Conversione numerica
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Media per scenario
    scenario_means = df.groupby("scenario", as_index=False)[time_cols].mean()

    # Formato lungo
    plot_df = scenario_means.melt(
        id_vars="scenario",
        value_vars=time_cols,
        var_name="module",
        value_name="time",
    )

    # Nomi moduli richiesti
    module_names = {
        "scene_time_mean": "Scene\nPerception",
        "plan_time_mean": "Manipulation\nPlanning",
        "sim_time_mean": "Execution\nScheduling",
        "validator_time_mean": "Validator",
    }
    plot_df["module"] = plot_df["module"].map(module_names)

    modules = list(module_names.values())
    scenarios = plot_df["scenario"].unique()

    # Palette colori
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

    x = range(len(modules))
    width = 0.8 / len(scenarios)

    # Font
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 25,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "legend.fontsize": 23,
    })

    plt.figure(figsize=(18, 5.5))

    for i, scenario in enumerate(scenarios):
        scenario_data = plot_df[plot_df["scenario"] == scenario]

        values = [
            scenario_data[scenario_data["module"] == module]["time"].values[0]
            for module in modules
        ]

        positions = [p + i * width - 0.4 + width / 2 for p in x]

        plt.bar(
            positions,
            values,
            width=width,
            label=scenario,
            color=colors[i % len(colors)],
            edgecolor="none",
            linewidth=0,
        )

    #plt.xticks(list(x), modules, rotation=20, ha="right")
    plt.xticks(list(x), modules, rotation=0, ha="center")
    #plt.xlabel("Module")
    plt.ylabel("Mean Inference Time [s]")

    # 👉 titolo rimosso qui

    plt.legend(frameon=False)
    plt.tight_layout()

    output_path = csv_path.parent / "plot_inference_times1.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Plot saved in: {output_path}")


if __name__ == "__main__":
    main()