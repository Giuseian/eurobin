# # from __future__ import annotations

# # from pathlib import Path
# # import pandas as pd


# # PROJECT_ROOT = Path(__file__).resolve().parents[2]
# # ANALYSIS_DIR = PROJECT_ROOT / "final_evaluation" / "analysis"
# # RESULTS_CSV = ANALYSIS_DIR / "results_summary.csv"
# # OUT_CSV = ANALYSIS_DIR / "manual_annotations.csv"


# # def main() -> None:
# #     if not RESULTS_CSV.exists():
# #         raise FileNotFoundError(f"Missing results file: {RESULTS_CSV}")

# #     df = pd.read_csv(RESULTS_CSV)

# #     template_df = df[
# #         ["scenario_id", "prompt_name", "run_id", "model_name", "repeat_idx"]
# #     ].copy()

# #     template_df["plan_correctness"] = ""
# #     template_df["notes"] = ""

# #     template_df.to_csv(OUT_CSV, index=False)

# #     print(f"Saved manual annotation template to: {OUT_CSV}")
# #     print(f"Rows: {len(template_df)}")


# # if __name__ == "__main__":
# #     main()



# from __future__ import annotations

# from pathlib import Path
# import pandas as pd


# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# ANALYSIS_DIR = PROJECT_ROOT / "final_evaluation" / "analysis"
# RESULTS_CSV = ANALYSIS_DIR / "results_summary.csv"
# OUT_CSV = ANALYSIS_DIR / "manual_annotations.csv"


# def main() -> None:
#     if not RESULTS_CSV.exists():
#         raise FileNotFoundError(f"Missing results file: {RESULTS_CSV}")

#     ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(RESULTS_CSV)

#     template_df = df[
#         ["scenario_id", "prompt_name", "run_id", "model_name", "repeat_idx"]
#     ].copy()

#     template_df["repeat_idx"] = template_df["repeat_idx"].astype(int)

#     template_df = template_df.sort_values(
#         by=["scenario_id", "prompt_name", "run_id", "model_name", "repeat_idx"],
#         ascending=[True, True, True, True, True],
#     ).reset_index(drop=True)

#     template_df["plan_correctness"] = ""
#     template_df["notes"] = ""

#     template_df.to_csv(OUT_CSV, index=False)

#     print(f"Saved manual annotation template to: {OUT_CSV}")
#     print(f"Rows: {len(template_df)}")


# if __name__ == "__main__":
#     main()


from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "final_evaluation" / "analysis"
RESULTS_CSV = ANALYSIS_DIR / "results_summary.csv"
OUT_CSV = ANALYSIS_DIR / "manual_annotations.csv"


def main() -> None:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_CSV}")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_CSV)

    template_df = df[
        ["scenario_id", "prompt_name", "run_id", "model_name", "repeat_idx"]
    ].copy()

    template_df["repeat_idx"] = template_df["repeat_idx"].astype(int)

    template_df = template_df.sort_values(
        by=["scenario_id", "prompt_name", "run_id", "model_name", "repeat_idx"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    template_df["plan_correctness"] = ""
    template_df["notes"] = ""

    template_df.to_csv(OUT_CSV, index=False)

    print(f"Saved manual annotation template to: {OUT_CSV}")
    print(f"Rows: {len(template_df)}")


if __name__ == "__main__":
    main()