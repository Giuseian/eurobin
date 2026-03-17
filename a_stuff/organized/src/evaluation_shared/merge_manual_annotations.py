from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "final_evaluation" / "analysis"
RESULTS_CSV = ANALYSIS_DIR / "results_summary.csv"
ANNOTATIONS_CSV = ANALYSIS_DIR / "manual_annotations.csv"
OUT_CSV = ANALYSIS_DIR / "merged_results.csv"


MERGE_KEYS = [
    "scenario_id",
    "prompt_name",
    "run_id",
    "model_name",
    "repeat_idx",
]


def main() -> None:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_CSV}")

    if not ANNOTATIONS_CSV.exists():
        raise FileNotFoundError(f"Missing annotations file: {ANNOTATIONS_CSV}")

    results_df = pd.read_csv(RESULTS_CSV)
    annotations_df = pd.read_csv(ANNOTATIONS_CSV)

    results_df["repeat_idx"] = results_df["repeat_idx"].astype(int)
    annotations_df["repeat_idx"] = annotations_df["repeat_idx"].astype(int)

    merged_df = results_df.merge(
        annotations_df,
        on=MERGE_KEYS,
        how="left",
        validate="one_to_one",
    )

    merged_df = merged_df.sort_values(
        by=["scenario_id", "prompt_name", "run_id", "model_name", "repeat_idx"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    merged_df.to_csv(OUT_CSV, index=False)

    total_rows = len(merged_df)
    annotated_rows = merged_df["plan_correctness"].notna().sum()

    print(f"Saved merged CSV to: {OUT_CSV}")
    print(f"Total rows: {total_rows}")
    print(f"Rows with manual annotation: {annotated_rows}")
    print(f"Rows still missing annotation: {total_rows - annotated_rows}")


if __name__ == "__main__":
    main()