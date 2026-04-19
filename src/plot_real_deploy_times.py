# working, merged simultaneous and vlm_planning 
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def to_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def add_stage_order(stage_df: pd.DataFrame) -> pd.DataFrame:
    df = stage_df.copy()

    if "run_id" not in df.columns or "cycle_index" not in df.columns:
        raise ValueError("stage_summary.csv must contain run_id and cycle_index.")

    sort_cols = ["run_id", "cycle_index"]
    if "stage_id" in df.columns:
        df["_stage_sort_key"] = pd.to_numeric(df["stage_id"], errors="coerce")
        sort_cols.append("_stage_sort_key")
    else:
        df["_stage_sort_key"] = range(len(df))
        sort_cols.append("_stage_sort_key")

    df = df.sort_values(sort_cols).copy()
    df["stage_order"] = df.groupby(["run_id", "cycle_index"]).cumcount() + 1
    return df.drop(columns=["_stage_sort_key"])


def aggregate_manipulation_time_per_stage(
    events_df: pd.DataFrame,
    stage_with_order_df: pd.DataFrame,
) -> pd.DataFrame:
    events = events_df.copy()
    events = events[events["event_type"] == "manipulation_script"].copy()
    events = events[events["duration_sec"].notna()].copy()

    if events.empty:
        return pd.DataFrame(
            columns=["run_id", "cycle_index", "stage_id", "stage_order", "manipulation_time"]
        )

    manip = (
        events.groupby(["run_id", "cycle_index", "stage_id"], as_index=False)["duration_sec"]
        .sum()
        .rename(columns={"duration_sec": "manipulation_time"})
    )

    stage_keys = stage_with_order_df[["run_id", "cycle_index", "stage_id", "stage_order"]].drop_duplicates()
    manip = manip.merge(
        stage_keys,
        on=["run_id", "cycle_index", "stage_id"],
        how="left",
    )
    return manip


def plot_high_level_reasoning_means(
    cycle_df: pd.DataFrame,
    events_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    cycle = cycle_df.copy()
    events = events_df.copy()

    to_numeric_inplace(
        cycle,
        [
            "scene_description_time",
            "scene_enrichment_time",
            "planning_time",
            "simultaneous_time",
        ],
    )
    to_numeric_inplace(events, ["duration_sec"])

    cycle["scene_perception_time"] = (
        cycle["scene_description_time"].fillna(0.0)
        + cycle["scene_enrichment_time"].fillna(0.0)
    )

    cycle["manipulation_planning_time"] = (
        cycle["planning_time"].fillna(0.0)
        + cycle["simultaneous_time"].fillna(0.0)
    )

    scene_perception_mean = cycle["scene_perception_time"].dropna().mean()
    manipulation_planning_mean = cycle["manipulation_planning_time"].dropna().mean()

    validator_events = events[
        events["event_type"].isin(["validator_pre", "validator_post"])
    ].copy()
    validator_mean = validator_events["duration_sec"].dropna().mean()

    summary = pd.DataFrame(
        {
            "component": [
                "Scene perception",
                "Manipulation planning",
                "Validator (per call)",
            ],
            "mean_time_sec": [
                scene_perception_mean,
                manipulation_planning_mean,
                validator_mean,
            ],
        }
    )

    summary = summary[summary["mean_time_sec"].notna()].copy()
    if summary.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(summary["component"], summary["mean_time_sec"])
    plt.ylabel("Mean time [s]")
    plt.title("Mean execution time of high-level reasoning modules")
    plt.xticks(rotation=20, ha="right")
    save_plot(output_dir / "01_high_level_reasoning_mean_times.png")


def build_pipeline_summary_df(
    cycle_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    cycle = cycle_df.copy()
    stage = add_stage_order(stage_df)
    events = events_df.copy()

    to_numeric_inplace(
        cycle,
        [
            "scene_description_time",
            "scene_enrichment_time",
            "planning_time",
            "simultaneous_time",
        ],
    )
    to_numeric_inplace(
        stage,
        [
            "pre_validation_time",
            "post_validation_time",
            "deploy_time",
            "stage_total_time",
        ],
    )
    to_numeric_inplace(events, ["duration_sec"])

    cycle["scene_perception_time"] = (
        cycle["scene_description_time"].fillna(0.0)
        + cycle["scene_enrichment_time"].fillna(0.0)
    )

    cycle["manipulation_planning_time"] = (
        cycle["planning_time"].fillna(0.0)
        + cycle["simultaneous_time"].fillna(0.0)
    )

    manip = aggregate_manipulation_time_per_stage(events, stage)

    rows: list[dict[str, float | str]] = []

    def add_row(label: str, value: float | None) -> None:
        if value is None or pd.isna(value):
            return
        rows.append({"step": label, "mean_time_sec": float(value)})

    add_row("Scene perception", cycle["scene_perception_time"].dropna().mean())
    add_row("Manipulation planning", cycle["manipulation_planning_time"].dropna().mean())

    stage1 = stage[stage["stage_order"] == 1]
    stage2 = stage[stage["stage_order"] == 2]
    stage3 = stage[stage["stage_order"] == 3]

    manip1 = manip[manip["stage_order"] == 1]["manipulation_time"].dropna().mean()
    manip2 = manip[manip["stage_order"] == 2]["manipulation_time"].dropna().mean()
    manip3 = manip[manip["stage_order"] == 3]["manipulation_time"].dropna().mean()

    add_row("Validation before grasp", stage1["pre_validation_time"].dropna().mean())
    add_row("Grasp red box", manip1)
    add_row("Validation after grasp", stage1["post_validation_time"].dropna().mean())

    add_row("Validation before place", stage2["pre_validation_time"].dropna().mean())
    add_row("Place object", manip2)
    add_row("Validation after place", stage2["post_validation_time"].dropna().mean())

    add_row("Validation before green grasp", stage3["pre_validation_time"].dropna().mean())
    add_row("Grasp green box", manip3)
    add_row("Final validation", stage3["post_validation_time"].dropna().mean())

    return pd.DataFrame(rows)


def plot_pipeline_mean_times(
    cycle_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    events_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary = build_pipeline_summary_df(cycle_df, stage_df, events_df)
    if summary.empty:
        return

    plt.figure(figsize=(12, 5))
    plt.bar(summary["step"], summary["mean_time_sec"])
    plt.ylabel("Mean time [s]")
    plt.title("Mean execution time across the deployment pipeline")
    plt.xticks(rotation=35, ha="right")
    save_plot(output_dir / "02_pipeline_mean_times.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-oriented plots from real deploy timing CSVs."
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        required=True,
        help="Directory containing events.csv, stage_summary.csv, cycle_summary.csv, run_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where plots will be saved. Default: <csv-dir>/plots",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir).resolve()
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else csv_dir / "plots"
    ensure_dir(output_dir)

    events_df = load_csv(csv_dir / "events.csv")
    stage_df = load_csv(csv_dir / "stage_summary.csv")
    cycle_df = load_csv(csv_dir / "cycle_summary.csv")
    _run_df = load_csv(csv_dir / "run_summary.csv")

    plot_high_level_reasoning_means(cycle_df, events_df, output_dir)
    plot_pipeline_mean_times(cycle_df, stage_df, events_df, output_dir)

    print("Plots generated successfully.")
    print(f"CSV directory:   {csv_dir}")
    print(f"Output plots:    {output_dir}")
    print("Generated files:")
    print(" - 01_high_level_reasoning_mean_times.png")
    print(" - 02_pipeline_mean_times.png")


if __name__ == "__main__":
    main()




# ## also working, focusing on two plots 
# # from __future__ import annotations

# # import argparse
# # from pathlib import Path

# # import matplotlib.pyplot as plt
# # import pandas as pd


# # def ensure_dir(path: Path) -> Path:
# #     path.mkdir(parents=True, exist_ok=True)
# #     return path


# # def load_csv(path: Path) -> pd.DataFrame:
# #     if not path.exists():
# #         raise FileNotFoundError(f"CSV not found: {path}")
# #     return pd.read_csv(path)


# # def save_plot(path: Path) -> None:
# #     path.parent.mkdir(parents=True, exist_ok=True)
# #     plt.tight_layout()
# #     plt.savefig(path, dpi=300, bbox_inches="tight")
# #     plt.close()


# # def to_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> None:
# #     for col in cols:
# #         if col in df.columns:
# #             df[col] = pd.to_numeric(df[col], errors="coerce")


# # def add_stage_order(stage_df: pd.DataFrame) -> pd.DataFrame:
# #     df = stage_df.copy()

# #     if "run_id" not in df.columns or "cycle_index" not in df.columns:
# #         raise ValueError("stage_summary.csv must contain run_id and cycle_index.")

# #     sort_cols = ["run_id", "cycle_index"]
# #     if "stage_id" in df.columns:
# #         df["_stage_sort_key"] = pd.to_numeric(df["stage_id"], errors="coerce")
# #         sort_cols.append("_stage_sort_key")
# #     else:
# #         df["_stage_sort_key"] = range(len(df))
# #         sort_cols.append("_stage_sort_key")

# #     df = df.sort_values(sort_cols).copy()
# #     df["stage_order"] = df.groupby(["run_id", "cycle_index"]).cumcount() + 1
# #     return df.drop(columns=["_stage_sort_key"])


# # def aggregate_manipulation_time_per_stage(
# #     events_df: pd.DataFrame,
# #     stage_with_order_df: pd.DataFrame,
# # ) -> pd.DataFrame:
# #     events = events_df.copy()
# #     events = events[events["event_type"] == "manipulation_script"].copy()
# #     events = events[events["duration_sec"].notna()].copy()

# #     if events.empty:
# #         return pd.DataFrame(
# #             columns=["run_id", "cycle_index", "stage_id", "stage_order", "manipulation_time"]
# #         )

# #     manip = (
# #         events.groupby(["run_id", "cycle_index", "stage_id"], as_index=False)["duration_sec"]
# #         .sum()
# #         .rename(columns={"duration_sec": "manipulation_time"})
# #     )

# #     stage_keys = stage_with_order_df[["run_id", "cycle_index", "stage_id", "stage_order"]].drop_duplicates()
# #     manip = manip.merge(
# #         stage_keys,
# #         on=["run_id", "cycle_index", "stage_id"],
# #         how="left",
# #     )
# #     return manip


# # def plot_high_level_reasoning_means(
# #     cycle_df: pd.DataFrame,
# #     events_df: pd.DataFrame,
# #     output_dir: Path,
# # ) -> None:
# #     cycle = cycle_df.copy()
# #     events = events_df.copy()

# #     to_numeric_inplace(
# #         cycle,
# #         [
# #             "scene_description_time",
# #             "scene_enrichment_time",
# #             "planning_time",
# #             "simultaneous_time",
# #         ],
# #     )
# #     to_numeric_inplace(events, ["duration_sec"])

# #     cycle["scene_perception_time"] = (
# #         cycle["scene_description_time"].fillna(0.0)
# #         + cycle["scene_enrichment_time"].fillna(0.0)
# #     )

# #     scene_perception_mean = cycle["scene_perception_time"].dropna().mean()
# #     planning_mean = cycle["planning_time"].dropna().mean()
# #     simultaneous_mean = cycle["simultaneous_time"].dropna().mean()

# #     validator_events = events[
# #         events["event_type"].isin(["validator_pre", "validator_post"])
# #     ].copy()
# #     validator_mean = validator_events["duration_sec"].dropna().mean()

# #     summary = pd.DataFrame(
# #         {
# #             "component": [
# #                 "Scene perception",
# #                 "VLM planning",
# #                 "Simultaneous actions",
# #                 "Validator (per call)",
# #             ],
# #             "mean_time_sec": [
# #                 scene_perception_mean,
# #                 planning_mean,
# #                 simultaneous_mean,
# #                 validator_mean,
# #             ],
# #         }
# #     )

# #     summary = summary[summary["mean_time_sec"].notna()].copy()
# #     if summary.empty:
# #         return

# #     plt.figure(figsize=(9, 5))
# #     plt.bar(summary["component"], summary["mean_time_sec"])
# #     plt.ylabel("Mean time [s]")
# #     plt.title("Mean execution time of high-level reasoning modules")
# #     plt.xticks(rotation=20, ha="right")
# #     save_plot(output_dir / "01_high_level_reasoning_mean_times.png")


# # def build_pipeline_summary_df(
# #     cycle_df: pd.DataFrame,
# #     stage_df: pd.DataFrame,
# #     events_df: pd.DataFrame,
# # ) -> pd.DataFrame:
# #     cycle = cycle_df.copy()
# #     stage = add_stage_order(stage_df)
# #     events = events_df.copy()

# #     to_numeric_inplace(
# #         cycle,
# #         [
# #             "scene_description_time",
# #             "scene_enrichment_time",
# #             "planning_time",
# #             "simultaneous_time",
# #         ],
# #     )
# #     to_numeric_inplace(
# #         stage,
# #         [
# #             "pre_validation_time",
# #             "post_validation_time",
# #             "deploy_time",
# #             "stage_total_time",
# #         ],
# #     )
# #     to_numeric_inplace(events, ["duration_sec"])

# #     cycle["scene_perception_time"] = (
# #         cycle["scene_description_time"].fillna(0.0)
# #         + cycle["scene_enrichment_time"].fillna(0.0)
# #     )

# #     manip = aggregate_manipulation_time_per_stage(events, stage)

# #     rows: list[dict[str, float | str]] = []

# #     def add_row(label: str, value: float | None) -> None:
# #         if value is None or pd.isna(value):
# #             return
# #         rows.append({"step": label, "mean_time_sec": float(value)})

# #     add_row("Scene perception", cycle["scene_perception_time"].dropna().mean())
# #     add_row("VLM planning", cycle["planning_time"].dropna().mean())
# #     add_row("Simultaneous actions", cycle["simultaneous_time"].dropna().mean())

# #     stage1 = stage[stage["stage_order"] == 1]
# #     stage2 = stage[stage["stage_order"] == 2]
# #     stage3 = stage[stage["stage_order"] == 3]

# #     manip1 = manip[manip["stage_order"] == 1]["manipulation_time"].dropna().mean()
# #     manip2 = manip[manip["stage_order"] == 2]["manipulation_time"].dropna().mean()
# #     manip3 = manip[manip["stage_order"] == 3]["manipulation_time"].dropna().mean()

# #     add_row("Validation before grasp", stage1["pre_validation_time"].dropna().mean())
# #     add_row("Grasp red box", manip1)
# #     add_row("Validation after grasp", stage1["post_validation_time"].dropna().mean())

# #     add_row("Validation before place", stage2["pre_validation_time"].dropna().mean())
# #     add_row("Place object", manip2)
# #     add_row("Validation after place", stage2["post_validation_time"].dropna().mean())

# #     add_row("Validation before green grasp", stage3["pre_validation_time"].dropna().mean())
# #     add_row("Grasp green box", manip3)
# #     add_row("Final validation", stage3["post_validation_time"].dropna().mean())

# #     return pd.DataFrame(rows)


# # def plot_pipeline_mean_times(
# #     cycle_df: pd.DataFrame,
# #     stage_df: pd.DataFrame,
# #     events_df: pd.DataFrame,
# #     output_dir: Path,
# # ) -> None:
# #     summary = build_pipeline_summary_df(cycle_df, stage_df, events_df)
# #     if summary.empty:
# #         return

# #     plt.figure(figsize=(12, 5))
# #     plt.bar(summary["step"], summary["mean_time_sec"])
# #     plt.ylabel("Mean time [s]")
# #     plt.title("Mean execution time across the deployment pipeline")
# #     plt.xticks(rotation=35, ha="right")
# #     save_plot(output_dir / "02_pipeline_mean_times.png")


# # def build_parser() -> argparse.ArgumentParser:
# #     parser = argparse.ArgumentParser(
# #         description="Generate paper-oriented plots from real deploy timing CSVs."
# #     )
# #     parser.add_argument(
# #         "--csv-dir",
# #         type=str,
# #         required=True,
# #         help="Directory containing events.csv, stage_summary.csv, cycle_summary.csv, run_summary.csv",
# #     )
# #     parser.add_argument(
# #         "--output-dir",
# #         type=str,
# #         default=None,
# #         help="Directory where plots will be saved. Default: <csv-dir>/plots",
# #     )
# #     return parser


# # def main() -> None:
# #     parser = build_parser()
# #     args = parser.parse_args()

# #     csv_dir = Path(args.csv_dir).resolve()
# #     if not csv_dir.exists():
# #         raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

# #     output_dir = Path(args.output_dir).resolve() if args.output_dir else csv_dir / "plots"
# #     ensure_dir(output_dir)

# #     events_df = load_csv(csv_dir / "events.csv")
# #     stage_df = load_csv(csv_dir / "stage_summary.csv")
# #     cycle_df = load_csv(csv_dir / "cycle_summary.csv")
# #     _run_df = load_csv(csv_dir / "run_summary.csv")

# #     plot_high_level_reasoning_means(cycle_df, events_df, output_dir)
# #     plot_pipeline_mean_times(cycle_df, stage_df, events_df, output_dir)

# #     print("Plots generated successfully.")
# #     print(f"CSV directory:   {csv_dir}")
# #     print(f"Output plots:    {output_dir}")
# #     print("Generated files:")
# #     print(" - 01_high_level_reasoning_mean_times.png")
# #     print(" - 02_pipeline_mean_times.png")


# # if __name__ == "__main__":
# #     main()



# # #### working previously 
# # # from __future__ import annotations

# # # import argparse
# # # from pathlib import Path

# # # import matplotlib.pyplot as plt
# # # import pandas as pd


# # # def ensure_dir(path: Path) -> Path:
# # #     path.mkdir(parents=True, exist_ok=True)
# # #     return path


# # # def load_csv(path: Path) -> pd.DataFrame:
# # #     if not path.exists():
# # #         raise FileNotFoundError(f"CSV not found: {path}")
# # #     return pd.read_csv(path)


# # # def save_plot(path: Path) -> None:
# # #     path.parent.mkdir(parents=True, exist_ok=True)
# # #     plt.tight_layout()
# # #     plt.savefig(path, dpi=300, bbox_inches="tight")
# # #     plt.close()


# # # def plot_event_means(events_df: pd.DataFrame, output_dir: Path) -> None:
# # #     df = events_df.copy()
# # #     df = df[df["duration_sec"].notna()]

# # #     interesting_events = [
# # #         "scene_description",
# # #         "scene_enrichment",
# # #         "vlm_planning",
# # #         "simultaneous_actions",
# # #         "validator_pre",
# # #         "validator_post",
# # #         "manipulation_script",
# # #         "screenshot",
# # #         "deploy",
# # #     ]
# # #     df = df[df["event_type"].isin(interesting_events)]

# # #     grouped = (
# # #         df.groupby("event_type", as_index=False)["duration_sec"]
# # #         .mean()
# # #         .sort_values("duration_sec", ascending=False)
# # #     )

# # #     plt.figure(figsize=(10, 5))
# # #     plt.bar(grouped["event_type"], grouped["duration_sec"])
# # #     plt.xticks(rotation=30, ha="right")
# # #     plt.ylabel("Mean time [s]")
# # #     plt.title("Mean execution time by event type")
# # #     save_plot(output_dir / "01_event_mean_times.png")


# # # def plot_cycle_breakdown(cycle_df: pd.DataFrame, output_dir: Path) -> None:
# # #     df = cycle_df.copy().sort_values(["cycle_index"])

# # #     components = [
# # #         "scene_description_time",
# # #         "scene_enrichment_time",
# # #         "planning_time",
# # #         "simultaneous_time",
# # #         "validators_total_time",
# # #         "deploy_total_time",
# # #     ]

# # #     for c in components:
# # #         if c not in df.columns:
# # #             df[c] = 0.0
# # #     df[components] = df[components].fillna(0.0)

# # #     x = range(len(df))
# # #     bottom = pd.Series([0.0] * len(df))

# # #     plt.figure(figsize=(11, 6))
# # #     for comp in components:
# # #         plt.bar(df["cycle_name"], df[comp], bottom=bottom, label=comp)
# # #         bottom += df[comp]

# # #     plt.xticks(rotation=30, ha="right")
# # #     plt.ylabel("Time [s]")
# # #     plt.title("Cycle time breakdown")
# # #     plt.legend()
# # #     save_plot(output_dir / "02_cycle_breakdown.png")


# # # def plot_stage_total_boxplot(stage_df: pd.DataFrame, output_dir: Path) -> None:
# # #     df = stage_df.copy()
# # #     df = df[df["stage_total_time"].notna()]

# # #     stage_ids = sorted(df["stage_id"].dropna().unique())
# # #     data = [df[df["stage_id"] == sid]["stage_total_time"].dropna().values for sid in stage_ids]

# # #     if not data:
# # #         return

# # #     plt.figure(figsize=(8, 5))
# # #     plt.boxplot(data, labels=[str(int(s)) for s in stage_ids])
# # #     plt.xlabel("Stage ID")
# # #     plt.ylabel("Stage total time [s]")
# # #     plt.title("Distribution of total stage time")
# # #     save_plot(output_dir / "03_stage_total_boxplot.png")


# # # def plot_stage_component_means(stage_df: pd.DataFrame, output_dir: Path) -> None:
# # #     df = stage_df.copy()

# # #     cols = [
# # #         "pre_validation_time",
# # #         "deploy_time",
# # #         "post_validation_time",
# # #     ]
# # #     for c in cols:
# # #         if c not in df.columns:
# # #             df[c] = 0.0

# # #     grouped = (
# # #         df.groupby("stage_id", as_index=False)[cols]
# # #         .mean(numeric_only=True)
# # #         .sort_values("stage_id")
# # #         .fillna(0.0)
# # #     )

# # #     if grouped.empty:
# # #         return

# # #     plt.figure(figsize=(9, 5))
# # #     bottom = pd.Series([0.0] * len(grouped))

# # #     for comp in cols:
# # #         plt.bar(grouped["stage_id"].astype(str), grouped[comp], bottom=bottom, label=comp)
# # #         bottom += grouped[comp]

# # #     plt.xlabel("Stage ID")
# # #     plt.ylabel("Mean time [s]")
# # #     plt.title("Mean stage time breakdown")
# # #     plt.legend()
# # #     save_plot(output_dir / "04_stage_component_means.png")


# # # def plot_run_total_vs_replans(run_df: pd.DataFrame, output_dir: Path) -> None:
# # #     df = run_df.copy()
# # #     if "replans_done" not in df.columns or "total_execution_time_seconds" not in df.columns:
# # #         return

# # #     df = df[df["total_execution_time_seconds"].notna()]

# # #     if df.empty:
# # #         return

# # #     plt.figure(figsize=(7, 5))
# # #     plt.scatter(df["replans_done"], df["total_execution_time_seconds"])
# # #     plt.xlabel("Number of replans")
# # #     plt.ylabel("Run total time [s]")
# # #     plt.title("Run total time vs replans")
# # #     save_plot(output_dir / "05_run_time_vs_replans.png")


# # # def plot_macro_category_breakdown(events_df: pd.DataFrame, output_dir: Path) -> None:
# # #     df = events_df.copy()
# # #     df = df[df["duration_sec"].notna()]

# # #     reasoning_events = {
# # #         "scene_description",
# # #         "scene_enrichment",
# # #         "vlm_planning",
# # #         "simultaneous_actions",
# # #     }
# # #     validation_events = {
# # #         "validator_pre",
# # #         "validator_post",
# # #     }
# # #     manipulation_events = {
# # #         "manipulation_script",
# # #         "screenshot",
# # #         "deploy",
# # #         "initial_homing",
# # #         "initial_screenshot",
# # #     }

# # #     reasoning = df[df["event_type"].isin(reasoning_events)]["duration_sec"].sum()
# # #     validation = df[df["event_type"].isin(validation_events)]["duration_sec"].sum()
# # #     manipulation = df[df["event_type"].isin(manipulation_events)]["duration_sec"].sum()

# # #     summary = pd.DataFrame(
# # #         {
# # #             "category": ["reasoning", "validation", "manipulation"],
# # #             "time_sec": [reasoning, validation, manipulation],
# # #         }
# # #     )

# # #     plt.figure(figsize=(7, 5))
# # #     plt.bar(summary["category"], summary["time_sec"])
# # #     plt.ylabel("Total time [s]")
# # #     plt.title("Macro-category time breakdown")
# # #     save_plot(output_dir / "06_macro_category_breakdown.png")


# # # def plot_validator_pre_post_comparison(events_df: pd.DataFrame, output_dir: Path) -> None:
# # #     df = events_df.copy()
# # #     df = df[df["event_type"].isin(["validator_pre", "validator_post"])]
# # #     df = df[df["duration_sec"].notna()]

# # #     if df.empty:
# # #         return

# # #     grouped = (
# # #         df.groupby("event_type", as_index=False)["duration_sec"]
# # #         .mean()
# # #         .sort_values("event_type")
# # #     )

# # #     plt.figure(figsize=(6, 4))
# # #     plt.bar(grouped["event_type"], grouped["duration_sec"])
# # #     plt.ylabel("Mean time [s]")
# # #     plt.title("Validator pre vs post")
# # #     save_plot(output_dir / "07_validator_pre_post_comparison.png")


# # # def plot_manipulation_script_means(events_df: pd.DataFrame, output_dir: Path) -> None:
# # #     df = events_df.copy()
# # #     df = df[df["event_type"] == "manipulation_script"]
# # #     df = df[df["duration_sec"].notna()]
# # #     df = df[df["submodule_name"].notna()]

# # #     if df.empty:
# # #         return

# # #     grouped = (
# # #         df.groupby("submodule_name", as_index=False)["duration_sec"]
# # #         .mean()
# # #         .sort_values("duration_sec", ascending=False)
# # #     )

# # #     plt.figure(figsize=(10, 5))
# # #     plt.bar(grouped["submodule_name"], grouped["duration_sec"])
# # #     plt.xticks(rotation=30, ha="right")
# # #     plt.ylabel("Mean time [s]")
# # #     plt.title("Mean manipulation time by script")
# # #     save_plot(output_dir / "08_manipulation_script_means.png")


# # # def build_parser() -> argparse.ArgumentParser:
# # #     parser = argparse.ArgumentParser(
# # #         description="Generate plots from real deploy timing CSVs."
# # #     )
# # #     parser.add_argument(
# # #         "--csv-dir",
# # #         type=str,
# # #         required=True,
# # #         help="Directory containing events.csv, stage_summary.csv, cycle_summary.csv, run_summary.csv",
# # #     )
# # #     parser.add_argument(
# # #         "--output-dir",
# # #         type=str,
# # #         default=None,
# # #         help="Directory where plots will be saved. Default: <csv-dir>/plots",
# # #     )
# # #     return parser


# # # def main() -> None:
# # #     parser = build_parser()
# # #     args = parser.parse_args()

# # #     csv_dir = Path(args.csv_dir).resolve()
# # #     if not csv_dir.exists():
# # #         raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

# # #     output_dir = Path(args.output_dir).resolve() if args.output_dir else csv_dir / "plots"
# # #     ensure_dir(output_dir)

# # #     events_df = load_csv(csv_dir / "events.csv")
# # #     stage_df = load_csv(csv_dir / "stage_summary.csv")
# # #     cycle_df = load_csv(csv_dir / "cycle_summary.csv")
# # #     run_df = load_csv(csv_dir / "run_summary.csv")

# # #     plot_event_means(events_df, output_dir)
# # #     plot_cycle_breakdown(cycle_df, output_dir)
# # #     plot_stage_total_boxplot(stage_df, output_dir)
# # #     plot_stage_component_means(stage_df, output_dir)
# # #     plot_run_total_vs_replans(run_df, output_dir)
# # #     plot_macro_category_breakdown(events_df, output_dir)
# # #     plot_validator_pre_post_comparison(events_df, output_dir)
# # #     plot_manipulation_script_means(events_df, output_dir)

# # #     print("Plots generated successfully.")
# # #     print(f"CSV directory:   {csv_dir}")
# # #     print(f"Output plots:    {output_dir}")


# # # if __name__ == "__main__":
# # #     main()