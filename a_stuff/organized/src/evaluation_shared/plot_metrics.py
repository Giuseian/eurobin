# # from __future__ import annotations

# # from pathlib import Path
# # import pandas as pd
# # import matplotlib.pyplot as plt


# # PROJECT_ROOT = Path(__file__).resolve().parents[2]
# # FINAL_EVAL_DIR = PROJECT_ROOT / "final_evaluation"
# # ANALYSIS_DIR = FINAL_EVAL_DIR / "analysis"

# # MERGED_CSV = ANALYSIS_DIR / "merged_results.csv"
# # PLOTS_DIR = FINAL_EVAL_DIR / "plots"


# # def load_data() -> pd.DataFrame:
# #     if not MERGED_CSV.exists():
# #         raise FileNotFoundError(f"Missing merged results file: {MERGED_CSV}")

# #     df = pd.read_csv(MERGED_CSV)

# #     df["repeat_idx"] = pd.to_numeric(df["repeat_idx"], errors="coerce")
# #     df["inference_time_sec"] = pd.to_numeric(df["inference_time_sec"], errors="coerce")
# #     df["plan_correctness"] = pd.to_numeric(df["plan_correctness"], errors="coerce")

# #     return df


# # def plot_accuracy_by_prompt_and_model(df: pd.DataFrame) -> None:
# #     plot_df = df.dropna(subset=["plan_correctness"]).copy()

# #     grouped = (
# #         plot_df.groupby(["prompt_name", "model_name"])["plan_correctness"]
# #         .mean()
# #         .reset_index()
# #     )

# #     pivot = grouped.pivot(index="prompt_name", columns="model_name", values="plan_correctness")
# #     ax = pivot.plot(kind="bar", figsize=(10, 6))

# #     ax.set_title("Accuratezza media del piano per prompt e modello")
# #     ax.set_xlabel("Prompt")
# #     ax.set_ylabel("Accuratezza media")
# #     ax.set_ylim(0, 1)
# #     plt.xticks(rotation=30, ha="right")
# #     plt.tight_layout()

# #     out_path = PLOTS_DIR / "accuracy_by_prompt_model.png"
# #     plt.savefig(out_path, dpi=200)
# #     plt.close()
# #     print(f"Saved: {out_path}")


# # def plot_mean_time_by_prompt_and_model(df: pd.DataFrame) -> None:
# #     plot_df = df.dropna(subset=["inference_time_sec"]).copy()

# #     grouped = (
# #         plot_df.groupby(["prompt_name", "model_name"])["inference_time_sec"]
# #         .mean()
# #         .reset_index()
# #     )

# #     pivot = grouped.pivot(index="prompt_name", columns="model_name", values="inference_time_sec")
# #     ax = pivot.plot(kind="bar", figsize=(10, 6))

# #     ax.set_title("Tempo medio di inferenza per prompt e modello")
# #     ax.set_xlabel("Prompt")
# #     ax.set_ylabel("Secondi")
# #     plt.xticks(rotation=30, ha="right")
# #     plt.tight_layout()

# #     out_path = PLOTS_DIR / "mean_time_by_prompt_model.png"
# #     plt.savefig(out_path, dpi=200)
# #     plt.close()
# #     print(f"Saved: {out_path}")


# # def plot_timing_boxplot_by_model(df: pd.DataFrame) -> None:
# #     plot_df = df.dropna(subset=["inference_time_sec"]).copy()

# #     models = sorted(plot_df["model_name"].dropna().unique())
# #     data = [
# #         plot_df.loc[plot_df["model_name"] == model, "inference_time_sec"].values
# #         for model in models
# #     ]

# #     plt.figure(figsize=(8, 6))
# #     plt.boxplot(data, tick_labels=models)
# #     plt.title("Distribuzione dei tempi di inferenza per modello")
# #     plt.xlabel("Modello")
# #     plt.ylabel("Secondi")
# #     plt.tight_layout()

# #     out_path = PLOTS_DIR / "timing_boxplot_by_model.png"
# #     plt.savefig(out_path, dpi=200)
# #     plt.close()
# #     print(f"Saved: {out_path}")


# # def plot_timing_boxplot_by_prompt_and_model(df: pd.DataFrame) -> None:
# #     plot_df = df.dropna(subset=["inference_time_sec"]).copy()
# #     plot_df["group"] = plot_df["prompt_name"] + " | " + plot_df["model_name"]

# #     groups = sorted(plot_df["group"].unique())
# #     data = [
# #         plot_df.loc[plot_df["group"] == group, "inference_time_sec"].values
# #         for group in groups
# #     ]

# #     plt.figure(figsize=(14, 6))
# #     plt.boxplot(data, tick_labels=groups)
# #     plt.title("Distribuzione dei tempi per prompt e modello")
# #     plt.xlabel("Prompt | Modello")
# #     plt.ylabel("Secondi")
# #     plt.xticks(rotation=45, ha="right")
# #     plt.tight_layout()

# #     out_path = PLOTS_DIR / "timing_boxplot_by_prompt_model.png"
# #     plt.savefig(out_path, dpi=200)
# #     plt.close()
# #     print(f"Saved: {out_path}")


# # def main() -> None:
# #     PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# #     df = load_data()

# #     plot_accuracy_by_prompt_and_model(df)
# #     plot_mean_time_by_prompt_and_model(df)
# #     plot_timing_boxplot_by_model(df)
# #     plot_timing_boxplot_by_prompt_and_model(df)

# #     print("All plots generated.")


# # if __name__ == "__main__":
# #     main()



# from __future__ import annotations

# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt


# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# FINAL_EVAL_DIR = PROJECT_ROOT / "final_evaluation"
# ANALYSIS_DIR = FINAL_EVAL_DIR / "analysis"

# MERGED_CSV = ANALYSIS_DIR / "merged_results.csv"
# PLOTS_DIR = FINAL_EVAL_DIR / "plots"

# PROMPT_ORDER = [
#     "only_image",
#     "image_graph",
#     "graph_spatial",
#     "graph_sides",
# ]


# def load_data() -> pd.DataFrame:
#     if not MERGED_CSV.exists():
#         raise FileNotFoundError(f"Missing merged results file: {MERGED_CSV}")

#     df = pd.read_csv(MERGED_CSV)

#     df["repeat_idx"] = pd.to_numeric(df["repeat_idx"], errors="coerce")
#     df["inference_time_sec"] = pd.to_numeric(df["inference_time_sec"], errors="coerce")
#     df["plan_correctness"] = pd.to_numeric(df["plan_correctness"], errors="coerce")

#     return df


# def get_model_order(df: pd.DataFrame) -> list[str]:
#     return sorted(df["model_name"].dropna().unique())


# def reindex_prompt_rows(pivot: pd.DataFrame) -> pd.DataFrame:
#     ordered_index = [p for p in PROMPT_ORDER if p in pivot.index]
#     remaining = [p for p in pivot.index if p not in ordered_index]
#     return pivot.reindex(ordered_index + remaining)


# def filter_macro_scenario(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
#     return df[df["scenario_id"].astype(str).str.startswith(prefix)].copy()


# def plot_accuracy_by_prompt_and_model(
#     df: pd.DataFrame,
#     out_name: str,
#     title: str,
# ) -> None:
#     plot_df = df.dropna(subset=["plan_correctness"]).copy()

#     grouped = (
#         plot_df.groupby(["prompt_name", "model_name"])["plan_correctness"]
#         .mean()
#         .reset_index()
#     )

#     pivot = grouped.pivot(index="prompt_name", columns="model_name", values="plan_correctness")
#     pivot = reindex_prompt_rows(pivot)

#     model_order = [m for m in get_model_order(df) if m in pivot.columns]
#     pivot = pivot[model_order]

#     ax = pivot.plot(kind="bar", figsize=(10, 6))

#     ax.set_title(title)
#     ax.set_xlabel("Prompt")
#     ax.set_ylabel("Mean plan correctness")
#     ax.set_ylim(0, 1)

#     plt.xticks(rotation=30, ha="right")
#     plt.tight_layout()

#     out_path = PLOTS_DIR / out_name
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     print(f"Saved: {out_path}")


# def plot_mean_time_by_prompt_and_model(df: pd.DataFrame) -> None:
#     plot_df = df.dropna(subset=["inference_time_sec"]).copy()

#     grouped = (
#         plot_df.groupby(["prompt_name", "model_name"])["inference_time_sec"]
#         .mean()
#         .reset_index()
#     )

#     pivot = grouped.pivot(index="prompt_name", columns="model_name", values="inference_time_sec")
#     pivot = reindex_prompt_rows(pivot)

#     model_order = [m for m in get_model_order(df) if m in pivot.columns]
#     pivot = pivot[model_order]

#     ax = pivot.plot(kind="bar", figsize=(10, 6))

#     ax.set_title("Mean Inference Time by Prompt and Model")
#     ax.set_xlabel("Prompt")
#     ax.set_ylabel("Time (s)")

#     plt.xticks(rotation=30, ha="right")
#     plt.tight_layout()

#     out_path = PLOTS_DIR / "mean_time_by_prompt_model.png"
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     print(f"Saved: {out_path}")


# def plot_timing_boxplot_by_model(
#     df: pd.DataFrame,
#     out_name: str,
#     title: str,
# ) -> None:
#     plot_df = df.dropna(subset=["inference_time_sec"]).copy()

#     models = get_model_order(plot_df)
#     data = [
#         plot_df.loc[plot_df["model_name"] == model, "inference_time_sec"].values
#         for model in models
#     ]

#     plt.figure(figsize=(8, 6))
#     plt.boxplot(data, tick_labels=models)

#     plt.title(title)
#     plt.xlabel("Model")
#     plt.ylabel("Time (s)")
#     plt.tight_layout()

#     out_path = PLOTS_DIR / out_name
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     print(f"Saved: {out_path}")


# def plot_timing_boxplot_by_prompt_and_model(df: pd.DataFrame) -> None:
#     plot_df = df.dropna(subset=["inference_time_sec"]).copy()

#     model_order = get_model_order(plot_df)

#     groups = []
#     data = []

#     for prompt in PROMPT_ORDER:
#         for model in model_order:
#             subset = plot_df[
#                 (plot_df["prompt_name"] == prompt) &
#                 (plot_df["model_name"] == model)
#             ]["inference_time_sec"].values

#             if len(subset) == 0:
#                 continue

#             groups.append(f"{prompt} | {model}")
#             data.append(subset)

#     remaining_prompts = sorted([p for p in plot_df["prompt_name"].dropna().unique() if p not in PROMPT_ORDER])
#     for prompt in remaining_prompts:
#         for model in model_order:
#             subset = plot_df[
#                 (plot_df["prompt_name"] == prompt) &
#                 (plot_df["model_name"] == model)
#             ]["inference_time_sec"].values

#             if len(subset) == 0:
#                 continue

#             groups.append(f"{prompt} | {model}")
#             data.append(subset)

#     plt.figure(figsize=(14, 6))
#     plt.boxplot(data, tick_labels=groups)

#     plt.title("Inference Time Distribution by Prompt and Model")
#     plt.xlabel("Prompt | Model")
#     plt.ylabel("Time (s)")
#     plt.xticks(rotation=45, ha="right")
#     plt.tight_layout()

#     out_path = PLOTS_DIR / "timing_boxplot_by_prompt_model.png"
#     plt.savefig(out_path, dpi=200)
#     plt.close()
#     print(f"Saved: {out_path}")


# def main() -> None:
#     PLOTS_DIR.mkdir(parents=True, exist_ok=True)

#     df = load_data()

#     # Global / cumulative plots
#     plot_accuracy_by_prompt_and_model(
#         df=df,
#         out_name="accuracy_by_prompt_model.png",
#         title="Mean Plan Correctness by Prompt and Model",
#     )
#     plot_mean_time_by_prompt_and_model(df)
#     plot_timing_boxplot_by_model(
#         df=df,
#         out_name="timing_boxplot_by_model.png",
#         title="Inference Time Distribution by Model",
#     )
#     plot_timing_boxplot_by_prompt_and_model(df)

#     # Macro-scenario plots: SB and BC
#     sb_df = filter_macro_scenario(df, "sb")
#     bc_df = filter_macro_scenario(df, "bc")

#     if not sb_df.empty:
#         plot_accuracy_by_prompt_and_model(
#             df=sb_df,
#             out_name="accuracy_by_prompt_model__sb.png",
#             title="Mean Plan Correctness by Prompt and Model — Stacked Boxes",
#         )
#         plot_timing_boxplot_by_model(
#             df=sb_df,
#             out_name="timing_boxplot_by_model__sb.png",
#             title="Inference Time Distribution by Model — Stacked Boxes",
#         )

#     if not bc_df.empty:
#         plot_accuracy_by_prompt_and_model(
#             df=bc_df,
#             out_name="accuracy_by_prompt_model__bc.png",
#             title="Mean Plan Correctness by Prompt and Model — Boxes and Cup",
#         )
#         plot_timing_boxplot_by_model(
#             df=bc_df,
#             out_name="timing_boxplot_by_model__bc.png",
#             title="Inference Time Distribution by Model — Boxes and Cup",
#         )

#     print("All plots generated.")


# if __name__ == "__main__":
#     main()



from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_EVAL_DIR = PROJECT_ROOT / "final_evaluation"
ANALYSIS_DIR = FINAL_EVAL_DIR / "analysis"

MERGED_CSV = ANALYSIS_DIR / "merged_results.csv"
PLOTS_DIR = FINAL_EVAL_DIR / "plots"

PROMPT_ORDER = [
    "only_image",
    "image_graph",
    "graph_spatial",
    "graph_sides",
]


def load_data() -> pd.DataFrame:
    if not MERGED_CSV.exists():
        raise FileNotFoundError(f"Missing merged results file: {MERGED_CSV}")

    df = pd.read_csv(MERGED_CSV)

    df["repeat_idx"] = pd.to_numeric(df["repeat_idx"], errors="coerce")
    df["inference_time_sec"] = pd.to_numeric(df["inference_time_sec"], errors="coerce")
    df["plan_correctness"] = pd.to_numeric(df["plan_correctness"], errors="coerce")

    return df


def get_model_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["model_name"].dropna().unique())


def reindex_prompt_rows(pivot: pd.DataFrame) -> pd.DataFrame:
    ordered_index = [p for p in PROMPT_ORDER if p in pivot.index]
    remaining = [p for p in pivot.index if p not in ordered_index]
    return pivot.reindex(ordered_index + remaining)


def filter_macro_scenario(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df[df["scenario_id"].astype(str).str.startswith(prefix)].copy()


def plot_accuracy_by_prompt_and_model(
    df: pd.DataFrame,
    out_dir: Path,
    out_name: str,
    title: str,
) -> None:
    plot_df = df.dropna(subset=["plan_correctness"]).copy()

    grouped = (
        plot_df.groupby(["prompt_name", "model_name"])["plan_correctness"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(index="prompt_name", columns="model_name", values="plan_correctness")
    pivot = reindex_prompt_rows(pivot)

    model_order = [m for m in get_model_order(df) if m in pivot.columns]
    pivot = pivot[model_order]

    # Convert to percentage for display
    pivot = pivot * 100.0

    ax = pivot.plot(kind="bar", figsize=(10, 6))

    ax.set_title(title)
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Mean plan correctness (%)")
    ax.set_ylim(0, 100)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_mean_time_by_prompt_and_model(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["inference_time_sec"]).copy()

    grouped = (
        plot_df.groupby(["prompt_name", "model_name"])["inference_time_sec"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(index="prompt_name", columns="model_name", values="inference_time_sec")
    pivot = reindex_prompt_rows(pivot)

    model_order = [m for m in get_model_order(df) if m in pivot.columns]
    pivot = pivot[model_order]

    ax = pivot.plot(kind="bar", figsize=(10, 6))

    ax.set_title("Mean Inference Time by Prompt and Model")
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Time (s)")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "mean_time_by_prompt_model.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_timing_boxplot_by_model(
    df: pd.DataFrame,
    out_dir: Path,
    out_name: str,
    title: str,
) -> None:
    plot_df = df.dropna(subset=["inference_time_sec"]).copy()

    models = get_model_order(plot_df)
    data = [
        plot_df.loc[plot_df["model_name"] == model, "inference_time_sec"].values
        for model in models
    ]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, tick_labels=models)

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Time (s)")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def plot_timing_boxplot_by_prompt_and_model(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["inference_time_sec"]).copy()

    model_order = get_model_order(plot_df)

    groups = []
    data = []

    for prompt in PROMPT_ORDER:
        for model in model_order:
            subset = plot_df[
                (plot_df["prompt_name"] == prompt) &
                (plot_df["model_name"] == model)
            ]["inference_time_sec"].values

            if len(subset) == 0:
                continue

            groups.append(f"{prompt} | {model}")
            data.append(subset)

    remaining_prompts = sorted([p for p in plot_df["prompt_name"].dropna().unique() if p not in PROMPT_ORDER])
    for prompt in remaining_prompts:
        for model in model_order:
            subset = plot_df[
                (plot_df["prompt_name"] == prompt) &
                (plot_df["model_name"] == model)
            ]["inference_time_sec"].values

            if len(subset) == 0:
                continue

            groups.append(f"{prompt} | {model}")
            data.append(subset)

    plt.figure(figsize=(14, 6))
    plt.boxplot(data, tick_labels=groups)

    plt.title("Inference Time Distribution by Prompt and Model")
    plt.xlabel("Prompt | Model")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "timing_boxplot_by_prompt_model.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    sb_plots_dir = PLOTS_DIR / "sb"
    bc_plots_dir = PLOTS_DIR / "bc"

    df = load_data()

    # Global / cumulative plots
    plot_accuracy_by_prompt_and_model(
        df=df,
        out_dir=PLOTS_DIR,
        out_name="accuracy_by_prompt_model.png",
        title="Mean Plan Correctness by Prompt and Model",
    )
    plot_mean_time_by_prompt_and_model(df)
    plot_timing_boxplot_by_model(
        df=df,
        out_dir=PLOTS_DIR,
        out_name="timing_boxplot_by_model.png",
        title="Inference Time Distribution by Model",
    )
    plot_timing_boxplot_by_prompt_and_model(df)

    # Macro-scenario plots: SB = Stacked Boxes, BC = Boxes and Cup
    sb_df = filter_macro_scenario(df, "sb")
    bc_df = filter_macro_scenario(df, "bc")

    if not sb_df.empty:
        plot_accuracy_by_prompt_and_model(
            df=sb_df,
            out_dir=sb_plots_dir,
            out_name="accuracy_by_prompt_model__sb.png",
            title="Mean Plan Correctness by Prompt and Model — SB (Stacked Boxes)",
        )
        plot_timing_boxplot_by_model(
            df=sb_df,
            out_dir=sb_plots_dir,
            out_name="timing_boxplot_by_model__sb.png",
            title="Inference Time Distribution by Model — SB (Stacked Boxes)",
        )

    if not bc_df.empty:
        plot_accuracy_by_prompt_and_model(
            df=bc_df,
            out_dir=bc_plots_dir,
            out_name="accuracy_by_prompt_model__bc.png",
            title="Mean Plan Correctness by Prompt and Model — BC (Boxes and Cup)",
        )
        plot_timing_boxplot_by_model(
            df=bc_df,
            out_dir=bc_plots_dir,
            out_name="timing_boxplot_by_model__bc.png",
            title="Inference Time Distribution by Model — BC (Boxes and Cup)",
        )

    print("All plots generated.")


if __name__ == "__main__":
    main()