from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "scenario",
    "loop_timestamp",
    "task_completed",
    "replans_done",
    "n_cycles",
    "scene_model",
    "plan_model",
    "sim_model",
    "validator_model",
    "scene_time_mean",
    "plan_time_mean",
    "sim_time_mean",
    "validator_time_mean",
    "n_total_stages",
    "n_pre_matching",
    "n_pre_non_matching",
    "n_post_matching",
    "n_post_non_matching",
    "executed_actions_count",
    "success_rate_general",
    "success_rate_scene",
    "success_rate_plan",
    "success_rate_sim",
    "success_rate_validator",
]


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_full_pipeline_summary_files(validation_loop_root: Path) -> list[Path]:
    return sorted(validation_loop_root.glob("*/*/full_pipeline_summary.json"))


def extract_run_row(summary: dict[str, Any]) -> dict[str, Any]:
    scenario = summary.get("scenario_name")
    loop_timestamp = summary.get("loop_timestamp")
    task_completed = summary.get("task_completed")
    replans_done = summary.get("replans_done")
    cycles = summary.get("cycles", [])
    n_cycles = len(cycles)

    config = summary.get("config", {})
    scene_model = config.get("scene_description", {}).get("model")
    plan_model = config.get("vlm_planning", {}).get("model")
    sim_model = config.get("simultaneous_actions", {}).get("model")
    validator_model = config.get("validator", {}).get("model")

    scene_cycle_times: list[float] = []
    plan_times: list[float] = []
    sim_times: list[float] = []
    validator_times: list[float] = []

    n_total_stages = 0
    n_pre_matching = 0
    n_pre_non_matching = 0
    n_post_matching = 0
    n_post_non_matching = 0

    for cycle in cycles:
        scene_description = cycle.get("scene_description")
        scene_description_full = cycle.get("scene_description_full")
        vlm_planning = cycle.get("vlm_planning")
        simultaneous_actions = cycle.get("simultaneous_actions")
        stages = cycle.get("stages", [])

        scene_time = 0.0
        has_scene_time = False

        if isinstance(scene_description, dict):
            t = scene_description.get("execution_time_seconds")
            if isinstance(t, (int, float)):
                scene_time += float(t)
                has_scene_time = True

        if isinstance(scene_description_full, dict):
            t = scene_description_full.get("execution_time_seconds")
            if isinstance(t, (int, float)):
                scene_time += float(t)
                has_scene_time = True

        if has_scene_time:
            scene_cycle_times.append(scene_time)

        if isinstance(vlm_planning, dict):
            t = vlm_planning.get("execution_time_seconds")
            if isinstance(t, (int, float)):
                plan_times.append(float(t))

        if isinstance(simultaneous_actions, dict):
            t = simultaneous_actions.get("execution_time_seconds")
            if isinstance(t, (int, float)):
                sim_times.append(float(t))

        for stage in stages:
            n_total_stages += 1

            pre_validation = stage.get("pre_validation")
            post_validation = stage.get("post_validation")

            if isinstance(pre_validation, dict):
                pre_result = pre_validation.get("result")
                if pre_result == "matching":
                    n_pre_matching += 1
                elif pre_result == "non_matching":
                    n_pre_non_matching += 1

            if isinstance(post_validation, dict):
                post_result = post_validation.get("result")
                if post_result == "matching":
                    n_post_matching += 1
                elif post_result == "non_matching":
                    n_post_non_matching += 1

            pre_time = stage.get("pre_validation_time_seconds")
            post_time = stage.get("post_validation_time_seconds")

            if isinstance(pre_time, (int, float)):
                validator_times.append(float(pre_time))
            if isinstance(post_time, (int, float)):
                validator_times.append(float(post_time))

    row = {
        "scenario": scenario,
        "loop_timestamp": loop_timestamp,
        "task_completed": task_completed,
        "replans_done": replans_done,
        "n_cycles": n_cycles,
        "scene_model": scene_model,
        "plan_model": plan_model,
        "sim_model": sim_model,
        "validator_model": validator_model,
        "scene_time_mean": safe_mean(scene_cycle_times),
        "plan_time_mean": safe_mean(plan_times),
        "sim_time_mean": safe_mean(sim_times),
        "validator_time_mean": safe_mean(validator_times),
        "n_total_stages": n_total_stages,
        "n_pre_matching": n_pre_matching,
        "n_pre_non_matching": n_pre_non_matching,
        "n_post_matching": n_post_matching,
        "n_post_non_matching": n_post_non_matching,
        "executed_actions_count": n_pre_matching,
        "success_rate_general": "",
        "success_rate_scene": "",
        "success_rate_plan": "",
        "success_rate_sim": "",
        "success_rate_validator": "",
    }

    return row


def make_row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row["scenario"]), str(row["loop_timestamp"]))


def read_existing_csv(output_csv: Path) -> list[dict[str, Any]]:
    if not output_csv.exists():
        return []

    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_append_only(rows: list[dict[str, Any]], output_csv: Path) -> tuple[int, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = read_existing_csv(output_csv)
    existing_keys = {make_row_key(row) for row in existing_rows}

    rows_to_add: list[dict[str, Any]] = []
    skipped_count = 0

    for row in rows:
        key = make_row_key(row)
        if key in existing_keys:
            skipped_count += 1
            continue

        normalized_row = {col: row.get(col, "") for col in CSV_COLUMNS}
        rows_to_add.append(normalized_row)

    all_rows = existing_rows + rows_to_add

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    return len(rows_to_add), skipped_count


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    validation_loop_root = project_root / "outputs_final" / "validation_loop"
    output_csv = project_root / "outputs_final" / "metrics" / "metrics_master.csv"

    if not validation_loop_root.exists():
        raise FileNotFoundError(
            f"Validation loop root not found: {validation_loop_root}"
        )

    summary_files = find_full_pipeline_summary_files(validation_loop_root)

    if not summary_files:
        raise FileNotFoundError(
            f"No full_pipeline_summary.json files found under: {validation_loop_root}"
        )

    rows: list[dict[str, Any]] = []

    for summary_path in summary_files:
        try:
            summary = load_json(summary_path)
            row = extract_run_row(summary)
            rows.append(row)
            print(f"[OK] Processed: {summary_path}")
        except Exception as exc:
            print(f"[ERROR] Failed to process {summary_path}: {exc}")

    added_count, skipped_count = write_csv_append_only(rows, output_csv)

    print("\n======================================================")
    print("METRICS CSV UPDATED")
    print(f"Input root:        {validation_loop_root}")
    print(f"Runs found:        {len(summary_files)}")
    print(f"New rows added:    {added_count}")
    print(f"Existing rows kept:{skipped_count}")
    print(f"Output CSV:        {output_csv}")
    print("======================================================")


if __name__ == "__main__":
    main()

