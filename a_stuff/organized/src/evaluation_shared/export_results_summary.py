# from __future__ import annotations

# import csv
# import json
# from pathlib import Path


# SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = Path(__file__).resolve().parents[2]   # eurobin root

# OUTPUTS_ROOT = PROJECT_ROOT / "outputs_shared"
# OUT_CSV = PROJECT_ROOT / "final_evaluation" / "analysis" / "results_summary.csv"

# print("SCRIPT_DIR:", SCRIPT_DIR)
# print("PROJECT_ROOT:", PROJECT_ROOT)
# print("OUTPUTS_ROOT:", OUTPUTS_ROOT)
# print("OUT_CSV:", OUT_CSV)


# def iter_json_files(root: Path):
#     yield from root.rglob("*.json")


# def load_json(path: Path) -> dict:
#     with path.open("r", encoding="utf-8") as f:
#         return json.load(f)


# def main() -> None:
#     OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

#     json_files = list(iter_json_files(OUTPUTS_ROOT))
#     print(f"Found JSON files: {len(json_files)}")

#     rows = []

#     for json_path in json_files:
#         try:
#             payload = load_json(json_path)
#         except Exception as exc:
#             print(f"[WARN] Failed to load {json_path}: {exc}")
#             continue

#         row = {
#             "file_path": str(json_path.relative_to(OUTPUTS_ROOT)),
#             "scenario_id": payload.get("scenario_id"),
#             "prompt_name": payload.get("prompt_name"),
#             "run_id": payload.get("run_id"),
#             "model_name": payload.get("model_name"),
#             "repeat_idx": payload.get("repeat_idx"),
#             "image_path": payload.get("image_path"),
#             "task_text": payload.get("task_text"),
#             "inference_time_sec": payload.get("inference_time_sec"),
#             "json_parse_ok": payload.get("json_parse_ok"),
#             "error": payload.get("error"),
#         }
#         rows.append(row)

#     fieldnames = [
#         "file_path",
#         "scenario_id",
#         "prompt_name",
#         "run_id",
#         "model_name",
#         "repeat_idx",
#         "image_path",
#         "task_text",
#         "inference_time_sec",
#         "json_parse_ok",
#         "error",
#     ]

#     with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)

#     print(f"Saved summary CSV to: {OUT_CSV}")
#     print(f"Total rows: {len(rows)}")


# if __name__ == "__main__":
#     main()




from __future__ import annotations

import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUTS_ROOT = PROJECT_ROOT / "outputs_shared"
OUT_CSV = PROJECT_ROOT / "final_evaluation" / "analysis" / "results_summary.csv"


def iter_json_files(root: Path):
    yield from root.rglob("*.json")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("SCRIPT_DIR:", SCRIPT_DIR)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("OUTPUTS_ROOT:", OUTPUTS_ROOT)
    print("OUT_CSV:", OUT_CSV)

    if not OUTPUTS_ROOT.exists():
        raise FileNotFoundError(f"Outputs root not found: {OUTPUTS_ROOT}")

    json_files = list(iter_json_files(OUTPUTS_ROOT))
    print(f"Found JSON files: {len(json_files)}")

    rows = []

    for json_path in json_files:
        try:
            payload = load_json(json_path)
        except Exception as exc:
            print(f"[WARN] Failed to load {json_path}: {exc}")
            continue

        row = {
            "file_path": str(json_path.relative_to(OUTPUTS_ROOT)),
            "scenario_id": payload.get("scenario_id"),
            "prompt_name": payload.get("prompt_name"),
            "run_id": payload.get("run_id"),
            "model_name": payload.get("model_name"),
            "repeat_idx": payload.get("repeat_idx"),
            "image_path": payload.get("image_path"),
            "task_text": payload.get("task_text"),
            "inference_time_sec": payload.get("inference_time_sec"),
            "json_parse_ok": payload.get("json_parse_ok"),
            "timestamp_utc": payload.get("timestamp_utc"),
            "error": payload.get("error"),
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            str(r.get("scenario_id") or ""),
            str(r.get("prompt_name") or ""),
            str(r.get("run_id") or ""),
            str(r.get("model_name") or ""),
            int(r.get("repeat_idx") or 0),
        )
    )

    fieldnames = [
        "file_path",
        "scenario_id",
        "prompt_name",
        "run_id",
        "model_name",
        "repeat_idx",
        "image_path",
        "task_text",
        "inference_time_sec",
        "json_parse_ok",
        "timestamp_utc",
        "error",
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary CSV to: {OUT_CSV}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()