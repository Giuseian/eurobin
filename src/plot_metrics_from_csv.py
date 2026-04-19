from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


GT_ACTIONS = {
    "2boxesaside": 4,
    "stacked_boxes": 3,
    "two_glasses_on_box": 3,
    "two_two_tennis": 3
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read metrics_master.csv and generate aggregate plots."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to metrics_master.csv",
    )
    return parser.parse_args()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_overall_from_scenario_means(scenario_df: pd.DataFrame, value_col: str) -> float:
    return float(scenario_df[value_col].mean())


def ensure_success_rate_columns_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "success_rate_general",
        "success_rate_scene",
        "success_rate_plan",
        "success_rate_sim",
        "success_rate_validator",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_efficiency_column(df: pd.DataFrame) -> pd.DataFrame:
    def compute_eff(row: pd.Series) -> float | None:
        scenario = row["scenario"]
        gt = GT_ACTIONS.get(scenario)
        if gt is None or gt == 0:
            return None
        return row["executed_actions_count"] / gt

    df["efficiency"] = df.apply(compute_eff, axis=1)
    return df


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def plot_inference_times(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []

    scenario_means = (
        df.groupby("scenario", as_index=False)[
            ["scene_time_mean", "plan_time_mean", "sim_time_mean", "validator_time_mean"]
        ]
        .mean()
    )

    for _, row in scenario_means.iterrows():
        scenario = row["scenario"]
        rows.extend(
            [
                {"group": scenario, "module": "scene", "value": row["scene_time_mean"]},
                {"group": scenario, "module": "plan", "value": row["plan_time_mean"]},
                {"group": scenario, "module": "sim", "value": row["sim_time_mean"]},
                {"group": scenario, "module": "validator", "value": row["validator_time_mean"]},
            ]
        )

    overall_row = {
        "scene_time_mean": compute_overall_from_scenario_means(scenario_means, "scene_time_mean"),
        "plan_time_mean": compute_overall_from_scenario_means(scenario_means, "plan_time_mean"),
        "sim_time_mean": compute_overall_from_scenario_means(scenario_means, "sim_time_mean"),
        "validator_time_mean": compute_overall_from_scenario_means(scenario_means, "validator_time_mean"),
    }
    rows.extend(
        [
            {"group": "overall", "module": "scene", "value": overall_row["scene_time_mean"]},
            {"group": "overall", "module": "plan", "value": overall_row["plan_time_mean"]},
            {"group": "overall", "module": "sim", "value": overall_row["sim_time_mean"]},
            {"group": "overall", "module": "validator", "value": overall_row["validator_time_mean"]},
        ]
    )

    plot_df = pd.DataFrame(rows)

    print_section("PLOT 1 - INFERENCE TIMES")
    print(plot_df.to_string(index=False))

    modules = ["scene", "plan", "sim", "validator"]
    groups = plot_df["group"].unique().tolist()
    x = range(len(modules))
    width = 0.8 / len(groups)

    plt.figure(figsize=(10, 6))
    for i, group in enumerate(groups):
        vals = []
        group_df = plot_df[plot_df["group"] == group]
        for module in modules:
            vals.append(group_df[group_df["module"] == module]["value"].iloc[0])
        positions = [p + i * width - 0.4 + width / 2 for p in x]
        plt.bar(positions, vals, width=width, label=group)

    plt.xticks(list(x), modules)
    plt.ylabel("Mean inference time (s)")
    plt.xlabel("Module")
    plt.title("Inference times by module")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_1_inference_times.png", dpi=300)
    plt.close()


def plot_success_rate_general(df: pd.DataFrame, output_dir: Path) -> None:
    scenario_means = (
        df.groupby("scenario", as_index=False)["success_rate_general"]
        .mean()
    )
    overall = float(scenario_means["success_rate_general"].mean())

    plot_df = pd.concat(
        [
            scenario_means,
            pd.DataFrame([{"scenario": "overall", "success_rate_general": overall}]),
        ],
        ignore_index=True,
    )

    print_section("PLOT 2 - GENERAL SUCCESS RATE")
    print(plot_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["scenario"], plot_df["success_rate_general"])
    plt.ylabel("Success rate")
    plt.xlabel("Scenario")
    plt.title("General success rate by scenario")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "plot_2_success_rate_general.png", dpi=300)
    plt.close()


def plot_success_rate_module_model(df: pd.DataFrame, output_dir: Path) -> None:
    expanded_rows = []

    for _, row in df.iterrows():
        expanded_rows.extend(
            [
                {
                    "scenario": row["scenario"],
                    "module": "scene",
                    "model": row["scene_model"],
                    "success_rate": row["success_rate_scene"],
                },
                {
                    "scenario": row["scenario"],
                    "module": "plan",
                    "model": row["plan_model"],
                    "success_rate": row["success_rate_plan"],
                },
                {
                    "scenario": row["scenario"],
                    "module": "sim",
                    "model": row["sim_model"],
                    "success_rate": row["success_rate_sim"],
                },
                {
                    "scenario": row["scenario"],
                    "module": "validator",
                    "model": row["validator_model"],
                    "success_rate": row["success_rate_validator"],
                },
            ]
        )

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df = expanded_df.dropna(subset=["success_rate"])

    scenario_module_model = (
        expanded_df.groupby(["scenario", "module", "model"], as_index=False)["success_rate"]
        .mean()
    )

    overall_df = (
        scenario_module_model.groupby(["module", "model"], as_index=False)["success_rate"]
        .mean()
    )
    overall_df["module_model"] = overall_df["module"] + " × " + overall_df["model"]

    print_section("PLOT 3 - SUCCESS RATE BY MODULE × MODEL (OVERALL)")
    print(overall_df[["module_model", "success_rate"]].to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.bar(overall_df["module_model"], overall_df["success_rate"])
    plt.ylabel("Success rate")
    plt.xlabel("Module × Model")
    plt.title("Success rate by module × model (overall)")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_3_success_rate_module_model_overall.png", dpi=300)
    plt.close()


def plot_efficiency(df: pd.DataFrame, output_dir: Path) -> None:
    eff_df = df.dropna(subset=["efficiency"]).copy()

    scenario_means = eff_df.groupby("scenario", as_index=False)["efficiency"].mean()
    overall = float(scenario_means["efficiency"].mean())

    plot_df = pd.concat(
        [
            scenario_means,
            pd.DataFrame([{"scenario": "overall", "efficiency": overall}]),
        ],
        ignore_index=True,
    )

    print_section("PLOT 4 - EFFICIENCY")
    print(plot_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["scenario"], plot_df["efficiency"])
    plt.ylabel("Efficiency")
    plt.xlabel("Scenario")
    plt.title("Efficiency by scenario")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_4_efficiency.png", dpi=300)
    plt.close()


def plot_replans(df: pd.DataFrame, output_dir: Path) -> None:
    scenario_means = df.groupby("scenario", as_index=False)["replans_done"].mean()
    overall = float(scenario_means["replans_done"].mean())

    plot_df = pd.concat(
        [
            scenario_means,
            pd.DataFrame([{"scenario": "overall", "replans_done": overall}]),
        ],
        ignore_index=True,
    )

    print_section("PLOT 5 - REPLANS")
    print(plot_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["scenario"], plot_df["replans_done"])
    plt.ylabel("Mean number of replans")
    plt.xlabel("Scenario")
    plt.title("Replanning by scenario")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_5_replans.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = csv_path.parent
    df = pd.read_csv(csv_path)

    required_cols = [
        "scenario",
        "task_completed",
        "replans_done",
        "scene_model",
        "plan_model",
        "sim_model",
        "validator_model",
        "scene_time_mean",
        "plan_time_mean",
        "sim_time_mean",
        "validator_time_mean",
        "n_pre_matching",
        "executed_actions_count",
        "success_rate_general",
        "success_rate_scene",
        "success_rate_plan",
        "success_rate_sim",
        "success_rate_validator",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    # Numeric cleanup
    for col in [
        "replans_done",
        "scene_time_mean",
        "plan_time_mean",
        "sim_time_mean",
        "validator_time_mean",
        "n_pre_matching",
        "executed_actions_count",
    ]:
        df[col] = safe_numeric(df[col])

    df = ensure_success_rate_columns_numeric(df)
    df = add_efficiency_column(df)

    print_section("INPUT CSV")
    print(f"CSV path: {csv_path}")
    print(f"Rows:     {len(df)}")
    print(f"Scenarios: {sorted(df['scenario'].dropna().unique().tolist())}")

    plot_inference_times(df, output_dir)
    plot_success_rate_general(df, output_dir)
    plot_success_rate_module_model(df, output_dir)
    plot_efficiency(df, output_dir)
    plot_replans(df, output_dir)

    print_section("PLOTS SAVED")
    for name in [
        "plot_1_inference_times.png",
        "plot_2_success_rate_general.png",
        "plot_3_success_rate_module_model_overall.png",
        "plot_4_efficiency.png",
        "plot_5_replans.png",
    ]:
        print(output_dir / name)


if __name__ == "__main__":
    main()