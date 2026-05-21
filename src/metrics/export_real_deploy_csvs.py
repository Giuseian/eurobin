from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_get(d: dict[str, Any] | None, *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def maybe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_json_if_exists(path_str: str | None) -> dict[str, Any] | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def normalize_outcome_to_replan_flag(outcome: str | None) -> bool:
    if not outcome:
        return False
    return outcome.startswith("replan_on_")


def make_run_id(full_summary: dict[str, Any], full_summary_path: Path) -> str:
    scenario = str(full_summary.get("scenario_name", "unknown_scenario"))
    loop_timestamp = str(full_summary.get("loop_timestamp", "unknown_timestamp"))
    return f"{scenario}__{loop_timestamp}__{full_summary_path.parent.name}"


def add_event_row(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    source_summary_path: str,
    scenario: str,
    loop_timestamp: str,
    cycle_index: int | None,
    cycle_name: str | None,
    stage_id: int | None,
    stage_name: str | None,
    event_type: str,
    module_name: str,
    submodule_name: str | None,
    model_name: str | None,
    duration_sec: float | None,
    outcome: str | None,
    replan_triggered: bool,
    image_before: str | None,
    image_after: str | None,
) -> None:
    rows.append(
        {
            "run_id": run_id,
            "source_summary_path": source_summary_path,
            "scenario": scenario,
            "loop_timestamp": loop_timestamp,
            "cycle_index": cycle_index,
            "cycle_name": cycle_name,
            "stage_id": stage_id,
            "stage_name": stage_name,
            "event_type": event_type,
            "module_name": module_name,
            "submodule_name": submodule_name,
            "model_name": model_name,
            "duration_sec": duration_sec,
            "outcome": outcome,
            "replan_triggered": replan_triggered,
            "image_before": image_before,
            "image_after": image_after,
        }
    )


def build_events_rows(
    full_summary: dict[str, Any],
    full_summary_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    scenario = str(full_summary["scenario_name"])
    loop_timestamp = str(full_summary["loop_timestamp"])
    run_id = make_run_id(full_summary, full_summary_path)
    source_summary_path = str(full_summary_path)

    initial_image = full_summary.get("initial_image_path")

    initial_homing_time = maybe_float(full_summary.get("initial_homing_time_seconds"))
    if initial_homing_time is not None:
        add_event_row(
            rows,
            run_id=run_id,
            source_summary_path=source_summary_path,
            scenario=scenario,
            loop_timestamp=loop_timestamp,
            cycle_index=None,
            cycle_name=None,
            stage_id=None,
            stage_name=None,
            event_type="initial_homing",
            module_name="deploy",
            submodule_name="homing.py",
            model_name=None,
            duration_sec=initial_homing_time,
            outcome="success",
            replan_triggered=False,
            image_before=None,
            image_after=initial_image,
        )

    initial_screenshot_time = maybe_float(full_summary.get("initial_screenshot_time_seconds"))
    if initial_screenshot_time is not None:
        add_event_row(
            rows,
            run_id=run_id,
            source_summary_path=source_summary_path,
            scenario=scenario,
            loop_timestamp=loop_timestamp,
            cycle_index=None,
            cycle_name=None,
            stage_id=None,
            stage_name=None,
            event_type="initial_screenshot",
            module_name="deploy",
            submodule_name=None,
            model_name=None,
            duration_sec=initial_screenshot_time,
            outcome="success",
            replan_triggered=False,
            image_before=None,
            image_after=initial_image,
        )

    cycles = full_summary.get("cycles", [])
    for cycle in cycles:
        cycle_index = cycle["cycle_index"]
        cycle_name = cycle["cycle_name"]
        cycle_outcome = cycle.get("outcome")
        cycle_replan = normalize_outcome_to_replan_flag(cycle_outcome)
        start_image = cycle.get("start_image_path")
        end_image = cycle.get("end_image_path")

        scene_desc = cycle.get("scene_description")
        if isinstance(scene_desc, dict):
            add_event_row(
                rows,
                run_id=run_id,
                source_summary_path=source_summary_path,
                scenario=scenario,
                loop_timestamp=loop_timestamp,
                cycle_index=cycle_index,
                cycle_name=cycle_name,
                stage_id=None,
                stage_name=None,
                event_type="scene_description",
                module_name="scene_description",
                submodule_name=None,
                model_name=scene_desc.get("model_name"),
                duration_sec=maybe_float(scene_desc.get("execution_time_seconds")),
                outcome=None,
                replan_triggered=cycle_replan,
                image_before=start_image,
                image_after=start_image,
            )

        scene_full = cycle.get("scene_description_full")
        if isinstance(scene_full, dict):
            add_event_row(
                rows,
                run_id=run_id,
                source_summary_path=source_summary_path,
                scenario=scenario,
                loop_timestamp=loop_timestamp,
                cycle_index=cycle_index,
                cycle_name=cycle_name,
                stage_id=None,
                stage_name=None,
                event_type="scene_enrichment",
                module_name="scene_description_full",
                submodule_name="enrich_scene",
                model_name=None,
                duration_sec=maybe_float(scene_full.get("execution_time_seconds")),
                outcome=None,
                replan_triggered=cycle_replan,
                image_before=start_image,
                image_after=start_image,
            )

        planning = cycle.get("vlm_planning")
        if isinstance(planning, dict):
            add_event_row(
                rows,
                run_id=run_id,
                source_summary_path=source_summary_path,
                scenario=scenario,
                loop_timestamp=loop_timestamp,
                cycle_index=cycle_index,
                cycle_name=cycle_name,
                stage_id=None,
                stage_name=None,
                event_type="vlm_planning",
                module_name="vlm_planning",
                submodule_name=None,
                model_name=planning.get("model_name"),
                duration_sec=maybe_float(planning.get("execution_time_seconds")),
                outcome=None,
                replan_triggered=cycle_replan,
                image_before=start_image,
                image_after=start_image,
            )

        sim = cycle.get("simultaneous_actions")
        if isinstance(sim, dict):
            add_event_row(
                rows,
                run_id=run_id,
                source_summary_path=source_summary_path,
                scenario=scenario,
                loop_timestamp=loop_timestamp,
                cycle_index=cycle_index,
                cycle_name=cycle_name,
                stage_id=None,
                stage_name=None,
                event_type="simultaneous_actions",
                module_name="simultaneous_actions",
                submodule_name=None,
                model_name=sim.get("model_name"),
                duration_sec=maybe_float(sim.get("execution_time_seconds")),
                outcome=None,
                replan_triggered=cycle_replan,
                image_before=start_image,
                image_after=start_image,
            )

        cycle_total = maybe_float(safe_get(cycle, "timing", "cycle_total"))
        if cycle_total is not None:
            add_event_row(
                rows,
                run_id=run_id,
                source_summary_path=source_summary_path,
                scenario=scenario,
                loop_timestamp=loop_timestamp,
                cycle_index=cycle_index,
                cycle_name=cycle_name,
                stage_id=None,
                stage_name=None,
                event_type="cycle_total",
                module_name="cycle",
                submodule_name=None,
                model_name=None,
                duration_sec=cycle_total,
                outcome=cycle_outcome,
                replan_triggered=cycle_replan,
                image_before=start_image,
                image_after=end_image,
            )

        for stage in cycle.get("stages", []):
            stage_id = stage.get("stage_id")
            stage_name = stage.get("stage_name")
            pre_image = stage.get("pre_image_path")
            post_image = stage.get("post_image_path") or stage.get("next_image_path")

            pre_result = safe_get(stage, "pre_validation", "result")
            post_result = safe_get(stage, "post_validation", "result")

            pre_run_info_path = safe_get(stage, "validator_paths", "pre", "run_info")
            post_run_info_path = safe_get(stage, "validator_paths", "post", "run_info")

            pre_run_info = load_json_if_exists(pre_run_info_path)
            post_run_info = load_json_if_exists(post_run_info_path)

            if pre_run_info is not None:
                add_event_row(
                    rows,
                    run_id=run_id,
                    source_summary_path=source_summary_path,
                    scenario=scenario,
                    loop_timestamp=loop_timestamp,
                    cycle_index=cycle_index,
                    cycle_name=cycle_name,
                    stage_id=stage_id,
                    stage_name=stage_name,
                    event_type="validator_pre",
                    module_name="validator",
                    submodule_name="pre",
                    model_name=pre_run_info.get("model"),
                    duration_sec=maybe_float(pre_run_info.get("execution_time_seconds")),
                    outcome=pre_result,
                    replan_triggered=cycle_replan and pre_result == "non_matching",
                    image_before=pre_image,
                    image_after=pre_image,
                )

            deploy_time = maybe_float(safe_get(stage, "timing", "deploy"))
            if deploy_time is not None:
                add_event_row(
                    rows,
                    run_id=run_id,
                    source_summary_path=source_summary_path,
                    scenario=scenario,
                    loop_timestamp=loop_timestamp,
                    cycle_index=cycle_index,
                    cycle_name=cycle_name,
                    stage_id=stage_id,
                    stage_name=stage_name,
                    event_type="deploy",
                    module_name="deploy",
                    submodule_name=None,
                    model_name=None,
                    duration_sec=deploy_time,
                    outcome="success",
                    replan_triggered=False,
                    image_before=pre_image,
                    image_after=post_image,
                )

            deploy_scripts = safe_get(stage, "timing", "deploy_scripts", default=[])
            if isinstance(deploy_scripts, list):
                for item in deploy_scripts:
                    if not isinstance(item, dict):
                        continue
                    add_event_row(
                        rows,
                        run_id=run_id,
                        source_summary_path=source_summary_path,
                        scenario=scenario,
                        loop_timestamp=loop_timestamp,
                        cycle_index=cycle_index,
                        cycle_name=cycle_name,
                        stage_id=stage_id,
                        stage_name=stage_name,
                        event_type=item.get("event_type", "manipulation_script"),
                        module_name=item.get("module_name", "deploy"),
                        submodule_name=item.get("script_name"),
                        model_name=None,
                        duration_sec=maybe_float(item.get("duration_sec")),
                        outcome=item.get("outcome", "success"),
                        replan_triggered=False,
                        image_before=item.get("image_before", pre_image),
                        image_after=item.get("image_after", post_image),
                    )

            screenshot_info = safe_get(stage, "timing", "screenshot")
            if isinstance(screenshot_info, dict):
                add_event_row(
                    rows,
                    run_id=run_id,
                    source_summary_path=source_summary_path,
                    scenario=scenario,
                    loop_timestamp=loop_timestamp,
                    cycle_index=cycle_index,
                    cycle_name=cycle_name,
                    stage_id=stage_id,
                    stage_name=stage_name,
                    event_type=screenshot_info.get("event_type", "screenshot"),
                    module_name=screenshot_info.get("module_name", "deploy"),
                    submodule_name=screenshot_info.get("script_name"),
                    model_name=None,
                    duration_sec=maybe_float(screenshot_info.get("duration_sec")),
                    outcome=screenshot_info.get("outcome", "success"),
                    replan_triggered=False,
                    image_before=pre_image,
                    image_after=post_image,
                )

            stage_total = maybe_float(safe_get(stage, "timing", "total"))
            if stage_total is not None:
                add_event_row(
                    rows,
                    run_id=run_id,
                    source_summary_path=source_summary_path,
                    scenario=scenario,
                    loop_timestamp=loop_timestamp,
                    cycle_index=cycle_index,
                    cycle_name=cycle_name,
                    stage_id=stage_id,
                    stage_name=stage_name,
                    event_type="stage_total",
                    module_name="stage",
                    submodule_name=None,
                    model_name=None,
                    duration_sec=stage_total,
                    outcome=post_result if post_result is not None else pre_result,
                    replan_triggered=False,
                    image_before=pre_image,
                    image_after=post_image,
                )

            if post_run_info is not None:
                add_event_row(
                    rows,
                    run_id=run_id,
                    source_summary_path=source_summary_path,
                    scenario=scenario,
                    loop_timestamp=loop_timestamp,
                    cycle_index=cycle_index,
                    cycle_name=cycle_name,
                    stage_id=stage_id,
                    stage_name=stage_name,
                    event_type="validator_post",
                    module_name="validator",
                    submodule_name="post",
                    model_name=post_run_info.get("model"),
                    duration_sec=maybe_float(post_run_info.get("execution_time_seconds")),
                    outcome=post_result,
                    replan_triggered=cycle_replan and post_result == "non_matching",
                    image_before=post_image,
                    image_after=post_image,
                )

    return rows


def build_stage_summary_rows(
    full_summary: dict[str, Any],
    full_summary_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    scenario = str(full_summary["scenario_name"])
    loop_timestamp = str(full_summary["loop_timestamp"])
    run_id = make_run_id(full_summary, full_summary_path)
    source_summary_path = str(full_summary_path)

    for cycle in full_summary.get("cycles", []):
        cycle_index = cycle["cycle_index"]
        cycle_name = cycle["cycle_name"]
        cycle_outcome = cycle.get("outcome")
        cycle_replan = normalize_outcome_to_replan_flag(cycle_outcome)

        for stage in cycle.get("stages", []):
            stage_id = stage.get("stage_id")
            stage_name = stage.get("stage_name")

            pre_run_info_path = safe_get(stage, "validator_paths", "pre", "run_info")
            post_run_info_path = safe_get(stage, "validator_paths", "post", "run_info")

            pre_run_info = load_json_if_exists(pre_run_info_path)
            post_run_info = load_json_if_exists(post_run_info_path)

            pre_validation_time = maybe_float(safe_get(stage, "timing", "pre_validation"))
            if pre_validation_time is None:
                pre_validation_time = maybe_float(
                    pre_run_info.get("execution_time_seconds") if pre_run_info else None
                )

            post_validation_time = maybe_float(safe_get(stage, "timing", "post_validation"))
            if post_validation_time is None:
                post_validation_time = maybe_float(
                    post_run_info.get("execution_time_seconds") if post_run_info else None
                )

            deploy_time = maybe_float(safe_get(stage, "timing", "deploy"))
            screenshot_time = maybe_float(safe_get(stage, "timing", "screenshot", "duration_sec"))

            explicit_stage_total = maybe_float(safe_get(stage, "timing", "total"))
            if explicit_stage_total is not None:
                stage_total_time = explicit_stage_total
            else:
                parts = [
                    x
                    for x in [
                        pre_validation_time,
                        deploy_time,
                        post_validation_time,
                    ]
                    if x is not None
                ]
                stage_total_time = sum(parts) if parts else None

            pre_result = safe_get(stage, "pre_validation", "result")
            post_result = safe_get(stage, "post_validation", "result")

            replan_after_stage = False
            if cycle_replan and isinstance(cycle_outcome, str):
                if cycle_outcome == f"replan_on_pre_stage_{stage_id}":
                    replan_after_stage = True
                if cycle_outcome == f"replan_on_post_stage_{stage_id}":
                    replan_after_stage = True

            rows.append(
                {
                    "run_id": run_id,
                    "source_summary_path": source_summary_path,
                    "scenario": scenario,
                    "loop_timestamp": loop_timestamp,
                    "cycle_index": cycle_index,
                    "cycle_name": cycle_name,
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "pre_validation_time": pre_validation_time,
                    "deploy_time": deploy_time,
                    "screenshot_time": screenshot_time,
                    "post_validation_time": post_validation_time,
                    "stage_total_time": stage_total_time,
                    "pre_result": pre_result,
                    "post_result": post_result,
                    "replan_after_stage": replan_after_stage,
                    "image_before": stage.get("pre_image_path"),
                    "image_after": stage.get("post_image_path") or stage.get("next_image_path"),
                }
            )

    return rows


def build_cycle_summary_rows(
    full_summary: dict[str, Any],
    full_summary_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    scenario = str(full_summary["scenario_name"])
    loop_timestamp = str(full_summary["loop_timestamp"])
    run_id = make_run_id(full_summary, full_summary_path)
    source_summary_path = str(full_summary_path)

    for cycle in full_summary.get("cycles", []):
        cycle_index = cycle["cycle_index"]
        cycle_name = cycle["cycle_name"]
        outcome = cycle.get("outcome")
        replan_happened = normalize_outcome_to_replan_flag(outcome)

        cycle_timing = cycle.get("timing", {})

        scene_description_time = maybe_float(cycle_timing.get("scene_description"))
        if scene_description_time is None:
            scene_description_time = maybe_float(
                safe_get(cycle, "scene_description", "execution_time_seconds")
            )

        scene_enrichment_time = maybe_float(cycle_timing.get("scene_enrichment"))
        if scene_enrichment_time is None:
            scene_enrichment_time = maybe_float(
                safe_get(cycle, "scene_description_full", "execution_time_seconds")
            )

        planning_time = maybe_float(cycle_timing.get("planning"))
        if planning_time is None:
            planning_time = maybe_float(
                safe_get(cycle, "vlm_planning", "execution_time_seconds")
            )

        simultaneous_time = maybe_float(cycle_timing.get("simultaneous"))
        if simultaneous_time is None:
            simultaneous_time = maybe_float(
                safe_get(cycle, "simultaneous_actions", "execution_time_seconds")
            )

        validators_total_time = maybe_float(cycle_timing.get("validators_total"))
        deploy_total_time = maybe_float(cycle_timing.get("deploy_total"))
        stages_total_time = maybe_float(cycle_timing.get("stages_total"))
        cycle_total_time = maybe_float(cycle_timing.get("cycle_total"))

        if validators_total_time is None or deploy_total_time is None or stages_total_time is None:
            validators_acc = 0.0
            validators_found = False

            deploy_acc = 0.0
            deploy_found = False

            stage_total_acc = 0.0
            stage_total_found = False

            for stage in cycle.get("stages", []):
                pre_run_info = load_json_if_exists(safe_get(stage, "validator_paths", "pre", "run_info"))
                post_run_info = load_json_if_exists(safe_get(stage, "validator_paths", "post", "run_info"))

                pre_t = maybe_float(safe_get(stage, "timing", "pre_validation"))
                if pre_t is None:
                    pre_t = maybe_float(
                        pre_run_info.get("execution_time_seconds") if pre_run_info else None
                    )

                post_t = maybe_float(safe_get(stage, "timing", "post_validation"))
                if post_t is None:
                    post_t = maybe_float(
                        post_run_info.get("execution_time_seconds") if post_run_info else None
                    )

                deploy_t = maybe_float(safe_get(stage, "timing", "deploy"))
                stage_t = maybe_float(safe_get(stage, "timing", "total"))

                if pre_t is not None:
                    validators_acc += pre_t
                    validators_found = True
                if post_t is not None:
                    validators_acc += post_t
                    validators_found = True
                if deploy_t is not None:
                    deploy_acc += deploy_t
                    deploy_found = True
                if stage_t is not None:
                    stage_total_acc += stage_t
                    stage_total_found = True

            if validators_total_time is None and validators_found:
                validators_total_time = validators_acc
            if deploy_total_time is None and deploy_found:
                deploy_total_time = deploy_acc
            if stages_total_time is None and stage_total_found:
                stages_total_time = stage_total_acc

        if cycle_total_time is None:
            cycle_total_calc = 0.0
            cycle_total_known = False

            for t in [
                scene_description_time,
                scene_enrichment_time,
                planning_time,
                simultaneous_time,
            ]:
                if t is not None:
                    cycle_total_calc += t
                    cycle_total_known = True

            if stages_total_time is not None:
                cycle_total_calc += stages_total_time
                cycle_total_known = True

            cycle_total_time = cycle_total_calc if cycle_total_known else None

        rows.append(
            {
                "run_id": run_id,
                "source_summary_path": source_summary_path,
                "scenario": scenario,
                "loop_timestamp": loop_timestamp,
                "cycle_index": cycle_index,
                "cycle_name": cycle_name,
                "scene_description_time": scene_description_time,
                "scene_enrichment_time": scene_enrichment_time,
                "planning_time": planning_time,
                "simultaneous_time": simultaneous_time,
                "validators_total_time": validators_total_time,
                "deploy_total_time": deploy_total_time,
                "stages_total_time": stages_total_time,
                "cycle_total_time": cycle_total_time,
                "num_stages": len(cycle.get("stages", [])),
                "outcome": outcome,
                "replan_happened": replan_happened,
                "start_image": cycle.get("start_image_path"),
                "end_image": cycle.get("end_image_path"),
            }
        )

    return rows


def build_run_summary_rows(
    full_summary: dict[str, Any],
    full_summary_path: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": make_run_id(full_summary, full_summary_path),
            "source_summary_path": str(full_summary_path),
            "scenario": full_summary.get("scenario_name"),
            "loop_timestamp": full_summary.get("loop_timestamp"),
            "initial_homing_time_seconds": maybe_float(full_summary.get("initial_homing_time_seconds")),
            "initial_screenshot_time_seconds": maybe_float(full_summary.get("initial_screenshot_time_seconds")),
            "total_execution_time_seconds": maybe_float(full_summary.get("total_execution_time_seconds")),
            "replans_done": full_summary.get("replans_done"),
            "task_completed": full_summary.get("task_completed"),
            "total_cycles": len(full_summary.get("cycles", [])),
            "initial_image_path": full_summary.get("initial_image_path"),
            "final_image_path": full_summary.get("final_image_path"),
        }
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_default_output_dir(summary_paths: list[Path]) -> Path:
    if len(summary_paths) == 1:
        return summary_paths[0].parent / "csv_exports"
    return Path.cwd() / "csv_exports_aggregated"


def gather_summary_paths(
    full_summary_paths: list[str] | None,
    summary_dirs: list[str] | None,
    glob_pattern: str,
) -> list[Path]:
    collected: list[Path] = []

    if full_summary_paths:
        for item in full_summary_paths:
            path = Path(item).resolve()
            if path.is_file():
                collected.append(path)
            else:
                raise FileNotFoundError(f"full summary file not found: {path}")

    if summary_dirs:
        for d in summary_dirs:
            root = Path(d).resolve()
            if not root.exists():
                raise FileNotFoundError(f"summary directory not found: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"not a directory: {root}")
            collected.extend(sorted(root.rglob(glob_pattern)))

    unique_paths = sorted({p.resolve() for p in collected})
    return unique_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export aggregated events.csv, stage_summary.csv, cycle_summary.csv, "
            "and run_summary.csv from one or more full_pipeline_summary.json files."
        )
    )
    parser.add_argument(
        "--full-summary",
        type=str,
        nargs="*",
        default=None,
        help="One or more explicit paths to full_pipeline_summary.json files.",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        nargs="*",
        default=None,
        help=(
            "One or more root directories to scan recursively for summary files. "
            "Use together with --glob-pattern if needed."
        ),
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="full_pipeline_summary.json",
        help="Filename pattern used when scanning directories recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where CSV files will be written. "
            "Default: sibling 'csv_exports' for a single input, otherwise './csv_exports_aggregated'."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary_paths = gather_summary_paths(
        full_summary_paths=args.full_summary,
        summary_dirs=args.summary_dir,
        glob_pattern=args.glob_pattern,
    )

    if not summary_paths:
        raise ValueError(
            "No input summaries found. Use --full-summary and/or --summary-dir."
        )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else infer_default_output_dir(summary_paths)
    )
    ensure_dir(output_dir)

    events_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for full_summary_path in summary_paths:
        full_summary = read_json(full_summary_path)
        events_rows.extend(build_events_rows(full_summary, full_summary_path))
        stage_rows.extend(build_stage_summary_rows(full_summary, full_summary_path))
        cycle_rows.extend(build_cycle_summary_rows(full_summary, full_summary_path))
        run_rows.extend(build_run_summary_rows(full_summary, full_summary_path))

    write_csv(
        output_dir / "events.csv",
        events_rows,
        fieldnames=[
            "run_id",
            "source_summary_path",
            "scenario",
            "loop_timestamp",
            "cycle_index",
            "cycle_name",
            "stage_id",
            "stage_name",
            "event_type",
            "module_name",
            "submodule_name",
            "model_name",
            "duration_sec",
            "outcome",
            "replan_triggered",
            "image_before",
            "image_after",
        ],
    )

    write_csv(
        output_dir / "stage_summary.csv",
        stage_rows,
        fieldnames=[
            "run_id",
            "source_summary_path",
            "scenario",
            "loop_timestamp",
            "cycle_index",
            "cycle_name",
            "stage_id",
            "stage_name",
            "pre_validation_time",
            "deploy_time",
            "screenshot_time",
            "post_validation_time",
            "stage_total_time",
            "pre_result",
            "post_result",
            "replan_after_stage",
            "image_before",
            "image_after",
        ],
    )

    write_csv(
        output_dir / "cycle_summary.csv",
        cycle_rows,
        fieldnames=[
            "run_id",
            "source_summary_path",
            "scenario",
            "loop_timestamp",
            "cycle_index",
            "cycle_name",
            "scene_description_time",
            "scene_enrichment_time",
            "planning_time",
            "simultaneous_time",
            "validators_total_time",
            "deploy_total_time",
            "stages_total_time",
            "cycle_total_time",
            "num_stages",
            "outcome",
            "replan_happened",
            "start_image",
            "end_image",
        ],
    )

    write_csv(
        output_dir / "run_summary.csv",
        run_rows,
        fieldnames=[
            "run_id",
            "source_summary_path",
            "scenario",
            "loop_timestamp",
            "initial_homing_time_seconds",
            "initial_screenshot_time_seconds",
            "total_execution_time_seconds",
            "replans_done",
            "task_completed",
            "total_cycles",
            "initial_image_path",
            "final_image_path",
        ],
    )

    print("CSV export completed.")
    print(f"Number of summaries: {len(summary_paths)}")
    print(f"Output directory:    {output_dir}")
    print(f"events.csv rows:     {len(events_rows)}")
    print(f"stage_summary rows:  {len(stage_rows)}")
    print(f"cycle_summary rows:  {len(cycle_rows)}")
    print(f"run_summary rows:    {len(run_rows)}")


if __name__ == "__main__":
    main()






### working previously 
# from __future__ import annotations

# import argparse
# import csv
# import json
# from pathlib import Path
# from typing import Any


# def read_json(path: Path) -> dict[str, Any]:
#     with path.open("r", encoding="utf-8") as f:
#         return json.load(f)


# def ensure_dir(path: Path) -> Path:
#     path.mkdir(parents=True, exist_ok=True)
#     return path


# def safe_get(d: dict[str, Any] | None, *keys: str, default=None):
#     cur = d
#     for k in keys:
#         if not isinstance(cur, dict) or k not in cur:
#             return default
#         cur = cur[k]
#     return cur


# def maybe_float(x: Any) -> float | None:
#     if x is None:
#         return None
#     try:
#         return float(x)
#     except (TypeError, ValueError):
#         return None


# def load_json_if_exists(path_str: str | None) -> dict[str, Any] | None:
#     if not path_str:
#         return None
#     path = Path(path_str)
#     if not path.exists():
#         return None
#     try:
#         return read_json(path)
#     except Exception:
#         return None


# def normalize_outcome_to_replan_flag(outcome: str | None) -> bool:
#     if not outcome:
#         return False
#     return outcome.startswith("replan_on_")


# def add_event_row(
#     rows: list[dict[str, Any]],
#     *,
#     run_id: str,
#     scenario: str,
#     loop_timestamp: str,
#     cycle_index: int | None,
#     cycle_name: str | None,
#     stage_id: int | None,
#     stage_name: str | None,
#     event_type: str,
#     module_name: str,
#     submodule_name: str | None,
#     model_name: str | None,
#     duration_sec: float | None,
#     outcome: str | None,
#     replan_triggered: bool,
#     image_before: str | None,
#     image_after: str | None,
# ) -> None:
#     rows.append(
#         {
#             "run_id": run_id,
#             "scenario": scenario,
#             "loop_timestamp": loop_timestamp,
#             "cycle_index": cycle_index,
#             "cycle_name": cycle_name,
#             "stage_id": stage_id,
#             "stage_name": stage_name,
#             "event_type": event_type,
#             "module_name": module_name,
#             "submodule_name": submodule_name,
#             "model_name": model_name,
#             "duration_sec": duration_sec,
#             "outcome": outcome,
#             "replan_triggered": replan_triggered,
#             "image_before": image_before,
#             "image_after": image_after,
#         }
#     )


# def build_events_rows(full_summary: dict[str, Any]) -> list[dict[str, Any]]:
#     rows: list[dict[str, Any]] = []

#     scenario = full_summary["scenario_name"]
#     loop_timestamp = full_summary["loop_timestamp"]
#     run_id = loop_timestamp

#     # --------------------------------------------------
#     # Run-level initial events
#     # --------------------------------------------------
#     initial_image = full_summary.get("initial_image_path")

#     initial_homing_time = maybe_float(full_summary.get("initial_homing_time_seconds"))
#     if initial_homing_time is not None:
#         add_event_row(
#             rows,
#             run_id=run_id,
#             scenario=scenario,
#             loop_timestamp=loop_timestamp,
#             cycle_index=None,
#             cycle_name=None,
#             stage_id=None,
#             stage_name=None,
#             event_type="initial_homing",
#             module_name="deploy",
#             submodule_name="homing.py",
#             model_name=None,
#             duration_sec=initial_homing_time,
#             outcome="success",
#             replan_triggered=False,
#             image_before=None,
#             image_after=initial_image,
#         )

#     initial_screenshot_time = maybe_float(full_summary.get("initial_screenshot_time_seconds"))
#     if initial_screenshot_time is not None:
#         add_event_row(
#             rows,
#             run_id=run_id,
#             scenario=scenario,
#             loop_timestamp=loop_timestamp,
#             cycle_index=None,
#             cycle_name=None,
#             stage_id=None,
#             stage_name=None,
#             event_type="initial_screenshot",
#             module_name="deploy",
#             submodule_name=None,
#             model_name=None,
#             duration_sec=initial_screenshot_time,
#             outcome="success",
#             replan_triggered=False,
#             image_before=None,
#             image_after=initial_image,
#         )

#     cycles = full_summary.get("cycles", [])
#     for cycle in cycles:
#         cycle_index = cycle["cycle_index"]
#         cycle_name = cycle["cycle_name"]
#         cycle_outcome = cycle.get("outcome")
#         cycle_replan = normalize_outcome_to_replan_flag(cycle_outcome)
#         start_image = cycle.get("start_image_path")
#         end_image = cycle.get("end_image_path")

#         # --------------------------------------------------
#         # Main modules from cycle_record
#         # --------------------------------------------------
#         scene_desc = cycle.get("scene_description")
#         if isinstance(scene_desc, dict):
#             add_event_row(
#                 rows,
#                 run_id=run_id,
#                 scenario=scenario,
#                 loop_timestamp=loop_timestamp,
#                 cycle_index=cycle_index,
#                 cycle_name=cycle_name,
#                 stage_id=None,
#                 stage_name=None,
#                 event_type="scene_description",
#                 module_name="scene_description",
#                 submodule_name=None,
#                 model_name=scene_desc.get("model_name"),
#                 duration_sec=maybe_float(scene_desc.get("execution_time_seconds")),
#                 outcome=None,
#                 replan_triggered=cycle_replan,
#                 image_before=start_image,
#                 image_after=start_image,
#             )

#         scene_full = cycle.get("scene_description_full")
#         if isinstance(scene_full, dict):
#             add_event_row(
#                 rows,
#                 run_id=run_id,
#                 scenario=scenario,
#                 loop_timestamp=loop_timestamp,
#                 cycle_index=cycle_index,
#                 cycle_name=cycle_name,
#                 stage_id=None,
#                 stage_name=None,
#                 event_type="scene_enrichment",
#                 module_name="scene_description_full",
#                 submodule_name="enrich_scene",
#                 model_name=None,
#                 duration_sec=maybe_float(scene_full.get("execution_time_seconds")),
#                 outcome=None,
#                 replan_triggered=cycle_replan,
#                 image_before=start_image,
#                 image_after=start_image,
#             )

#         planning = cycle.get("vlm_planning")
#         if isinstance(planning, dict):
#             add_event_row(
#                 rows,
#                 run_id=run_id,
#                 scenario=scenario,
#                 loop_timestamp=loop_timestamp,
#                 cycle_index=cycle_index,
#                 cycle_name=cycle_name,
#                 stage_id=None,
#                 stage_name=None,
#                 event_type="vlm_planning",
#                 module_name="vlm_planning",
#                 submodule_name=None,
#                 model_name=planning.get("model_name"),
#                 duration_sec=maybe_float(planning.get("execution_time_seconds")),
#                 outcome=None,
#                 replan_triggered=cycle_replan,
#                 image_before=start_image,
#                 image_after=start_image,
#             )

#         sim = cycle.get("simultaneous_actions")
#         if isinstance(sim, dict):
#             add_event_row(
#                 rows,
#                 run_id=run_id,
#                 scenario=scenario,
#                 loop_timestamp=loop_timestamp,
#                 cycle_index=cycle_index,
#                 cycle_name=cycle_name,
#                 stage_id=None,
#                 stage_name=None,
#                 event_type="simultaneous_actions",
#                 module_name="simultaneous_actions",
#                 submodule_name=None,
#                 model_name=sim.get("model_name"),
#                 duration_sec=maybe_float(sim.get("execution_time_seconds")),
#                 outcome=None,
#                 replan_triggered=cycle_replan,
#                 image_before=start_image,
#                 image_after=start_image,
#             )

#         # Optional cycle_total as event
#         cycle_total = maybe_float(safe_get(cycle, "timing", "cycle_total"))
#         if cycle_total is not None:
#             add_event_row(
#                 rows,
#                 run_id=run_id,
#                 scenario=scenario,
#                 loop_timestamp=loop_timestamp,
#                 cycle_index=cycle_index,
#                 cycle_name=cycle_name,
#                 stage_id=None,
#                 stage_name=None,
#                 event_type="cycle_total",
#                 module_name="cycle",
#                 submodule_name=None,
#                 model_name=None,
#                 duration_sec=cycle_total,
#                 outcome=cycle_outcome,
#                 replan_triggered=cycle_replan,
#                 image_before=start_image,
#                 image_after=end_image,
#             )

#         # --------------------------------------------------
#         # Stage-level events
#         # --------------------------------------------------
#         for stage in cycle.get("stages", []):
#             stage_id = stage.get("stage_id")
#             stage_name = stage.get("stage_name")
#             pre_image = stage.get("pre_image_path")
#             post_image = stage.get("post_image_path") or stage.get("next_image_path")

#             pre_result = safe_get(stage, "pre_validation", "result")
#             post_result = safe_get(stage, "post_validation", "result")

#             pre_run_info_path = safe_get(stage, "validator_paths", "pre", "run_info")
#             post_run_info_path = safe_get(stage, "validator_paths", "post", "run_info")

#             pre_run_info = load_json_if_exists(pre_run_info_path)
#             post_run_info = load_json_if_exists(post_run_info_path)

#             if pre_run_info is not None:
#                 add_event_row(
#                     rows,
#                     run_id=run_id,
#                     scenario=scenario,
#                     loop_timestamp=loop_timestamp,
#                     cycle_index=cycle_index,
#                     cycle_name=cycle_name,
#                     stage_id=stage_id,
#                     stage_name=stage_name,
#                     event_type="validator_pre",
#                     module_name="validator",
#                     submodule_name="pre",
#                     model_name=pre_run_info.get("model"),
#                     duration_sec=maybe_float(pre_run_info.get("execution_time_seconds")),
#                     outcome=pre_result,
#                     replan_triggered=cycle_replan and pre_result == "non_matching",
#                     image_before=pre_image,
#                     image_after=pre_image,
#                 )

#             deploy_time = maybe_float(safe_get(stage, "timing", "deploy"))
#             if deploy_time is not None:
#                 add_event_row(
#                     rows,
#                     run_id=run_id,
#                     scenario=scenario,
#                     loop_timestamp=loop_timestamp,
#                     cycle_index=cycle_index,
#                     cycle_name=cycle_name,
#                     stage_id=stage_id,
#                     stage_name=stage_name,
#                     event_type="deploy",
#                     module_name="deploy",
#                     submodule_name=None,
#                     model_name=None,
#                     duration_sec=deploy_time,
#                     outcome="success",
#                     replan_triggered=False,
#                     image_before=pre_image,
#                     image_after=post_image,
#                 )

#             deploy_scripts = safe_get(stage, "timing", "deploy_scripts", default=[])
#             if isinstance(deploy_scripts, list):
#                 for item in deploy_scripts:
#                     if not isinstance(item, dict):
#                         continue
#                     add_event_row(
#                         rows,
#                         run_id=run_id,
#                         scenario=scenario,
#                         loop_timestamp=loop_timestamp,
#                         cycle_index=cycle_index,
#                         cycle_name=cycle_name,
#                         stage_id=stage_id,
#                         stage_name=stage_name,
#                         event_type=item.get("event_type", "manipulation_script"),
#                         module_name=item.get("module_name", "deploy"),
#                         submodule_name=item.get("script_name"),
#                         model_name=None,
#                         duration_sec=maybe_float(item.get("duration_sec")),
#                         outcome=item.get("outcome", "success"),
#                         replan_triggered=False,
#                         image_before=item.get("image_before", pre_image),
#                         image_after=item.get("image_after", post_image),
#                     )

#             screenshot_info = safe_get(stage, "timing", "screenshot")
#             if isinstance(screenshot_info, dict):
#                 add_event_row(
#                     rows,
#                     run_id=run_id,
#                     scenario=scenario,
#                     loop_timestamp=loop_timestamp,
#                     cycle_index=cycle_index,
#                     cycle_name=cycle_name,
#                     stage_id=stage_id,
#                     stage_name=stage_name,
#                     event_type=screenshot_info.get("event_type", "screenshot"),
#                     module_name=screenshot_info.get("module_name", "deploy"),
#                     submodule_name=screenshot_info.get("script_name"),
#                     model_name=None,
#                     duration_sec=maybe_float(screenshot_info.get("duration_sec")),
#                     outcome=screenshot_info.get("outcome", "success"),
#                     replan_triggered=False,
#                     image_before=pre_image,
#                     image_after=post_image,
#                 )

#             stage_total = maybe_float(safe_get(stage, "timing", "total"))
#             if stage_total is not None:
#                 add_event_row(
#                     rows,
#                     run_id=run_id,
#                     scenario=scenario,
#                     loop_timestamp=loop_timestamp,
#                     cycle_index=cycle_index,
#                     cycle_name=cycle_name,
#                     stage_id=stage_id,
#                     stage_name=stage_name,
#                     event_type="stage_total",
#                     module_name="stage",
#                     submodule_name=None,
#                     model_name=None,
#                     duration_sec=stage_total,
#                     outcome=post_result if post_result is not None else pre_result,
#                     replan_triggered=False,
#                     image_before=pre_image,
#                     image_after=post_image,
#                 )

#             if post_run_info is not None:
#                 add_event_row(
#                     rows,
#                     run_id=run_id,
#                     scenario=scenario,
#                     loop_timestamp=loop_timestamp,
#                     cycle_index=cycle_index,
#                     cycle_name=cycle_name,
#                     stage_id=stage_id,
#                     stage_name=stage_name,
#                     event_type="validator_post",
#                     module_name="validator",
#                     submodule_name="post",
#                     model_name=post_run_info.get("model"),
#                     duration_sec=maybe_float(post_run_info.get("execution_time_seconds")),
#                     outcome=post_result,
#                     replan_triggered=cycle_replan and post_result == "non_matching",
#                     image_before=post_image,
#                     image_after=post_image,
#                 )

#     return rows


# def build_stage_summary_rows(full_summary: dict[str, Any]) -> list[dict[str, Any]]:
#     rows: list[dict[str, Any]] = []

#     scenario = full_summary["scenario_name"]
#     loop_timestamp = full_summary["loop_timestamp"]

#     for cycle in full_summary.get("cycles", []):
#         cycle_index = cycle["cycle_index"]
#         cycle_name = cycle["cycle_name"]
#         cycle_outcome = cycle.get("outcome")
#         cycle_replan = normalize_outcome_to_replan_flag(cycle_outcome)

#         for stage in cycle.get("stages", []):
#             stage_id = stage.get("stage_id")
#             stage_name = stage.get("stage_name")

#             pre_run_info_path = safe_get(stage, "validator_paths", "pre", "run_info")
#             post_run_info_path = safe_get(stage, "validator_paths", "post", "run_info")

#             pre_run_info = load_json_if_exists(pre_run_info_path)
#             post_run_info = load_json_if_exists(post_run_info_path)

#             pre_validation_time = maybe_float(
#                 safe_get(stage, "timing", "pre_validation")
#             )
#             if pre_validation_time is None:
#                 pre_validation_time = maybe_float(
#                     pre_run_info.get("execution_time_seconds") if pre_run_info else None
#                 )

#             post_validation_time = maybe_float(
#                 safe_get(stage, "timing", "post_validation")
#             )
#             if post_validation_time is None:
#                 post_validation_time = maybe_float(
#                     post_run_info.get("execution_time_seconds") if post_run_info else None
#                 )

#             deploy_time = maybe_float(safe_get(stage, "timing", "deploy"))
#             screenshot_time = maybe_float(safe_get(stage, "timing", "screenshot", "duration_sec"))

#             explicit_stage_total = maybe_float(safe_get(stage, "timing", "total"))
#             if explicit_stage_total is not None:
#                 stage_total_time = explicit_stage_total
#             else:
#                 parts = [
#                     x for x in [
#                         pre_validation_time,
#                         deploy_time,
#                         post_validation_time,
#                     ] if x is not None
#                 ]
#                 stage_total_time = sum(parts) if parts else None

#             pre_result = safe_get(stage, "pre_validation", "result")
#             post_result = safe_get(stage, "post_validation", "result")

#             replan_after_stage = False
#             if cycle_replan and isinstance(cycle_outcome, str):
#                 if cycle_outcome == f"replan_on_pre_stage_{stage_id}":
#                     replan_after_stage = True
#                 if cycle_outcome == f"replan_on_post_stage_{stage_id}":
#                     replan_after_stage = True

#             rows.append(
#                 {
#                     "scenario": scenario,
#                     "loop_timestamp": loop_timestamp,
#                     "cycle_index": cycle_index,
#                     "cycle_name": cycle_name,
#                     "stage_id": stage_id,
#                     "stage_name": stage_name,
#                     "pre_validation_time": pre_validation_time,
#                     "deploy_time": deploy_time,
#                     "screenshot_time": screenshot_time,
#                     "post_validation_time": post_validation_time,
#                     "stage_total_time": stage_total_time,
#                     "pre_result": pre_result,
#                     "post_result": post_result,
#                     "replan_after_stage": replan_after_stage,
#                     "image_before": stage.get("pre_image_path"),
#                     "image_after": stage.get("post_image_path") or stage.get("next_image_path"),
#                 }
#             )

#     return rows


# def build_cycle_summary_rows(full_summary: dict[str, Any]) -> list[dict[str, Any]]:
#     rows: list[dict[str, Any]] = []

#     scenario = full_summary["scenario_name"]
#     loop_timestamp = full_summary["loop_timestamp"]

#     for cycle in full_summary.get("cycles", []):
#         cycle_index = cycle["cycle_index"]
#         cycle_name = cycle["cycle_name"]
#         outcome = cycle.get("outcome")
#         replan_happened = normalize_outcome_to_replan_flag(outcome)

#         cycle_timing = cycle.get("timing", {})

#         scene_description_time = maybe_float(
#             cycle_timing.get("scene_description")
#         )
#         if scene_description_time is None:
#             scene_description_time = maybe_float(
#                 safe_get(cycle, "scene_description", "execution_time_seconds")
#             )

#         scene_enrichment_time = maybe_float(
#             cycle_timing.get("scene_enrichment")
#         )
#         if scene_enrichment_time is None:
#             scene_enrichment_time = maybe_float(
#                 safe_get(cycle, "scene_description_full", "execution_time_seconds")
#             )

#         planning_time = maybe_float(
#             cycle_timing.get("planning")
#         )
#         if planning_time is None:
#             planning_time = maybe_float(
#                 safe_get(cycle, "vlm_planning", "execution_time_seconds")
#             )

#         simultaneous_time = maybe_float(
#             cycle_timing.get("simultaneous")
#         )
#         if simultaneous_time is None:
#             simultaneous_time = maybe_float(
#                 safe_get(cycle, "simultaneous_actions", "execution_time_seconds")
#             )

#         validators_total_time = maybe_float(
#             cycle_timing.get("validators_total")
#         )
#         deploy_total_time = maybe_float(
#             cycle_timing.get("deploy_total")
#         )
#         stages_total_time = maybe_float(
#             cycle_timing.get("stages_total")
#         )
#         cycle_total_time = maybe_float(
#             cycle_timing.get("cycle_total")
#         )

#         # Fallback for old summaries without cycle["timing"]
#         if validators_total_time is None or deploy_total_time is None or stages_total_time is None:
#             validators_acc = 0.0
#             validators_found = False

#             deploy_acc = 0.0
#             deploy_found = False

#             stage_total_acc = 0.0
#             stage_total_found = False

#             for stage in cycle.get("stages", []):
#                 pre_run_info = load_json_if_exists(
#                     safe_get(stage, "validator_paths", "pre", "run_info")
#                 )
#                 post_run_info = load_json_if_exists(
#                     safe_get(stage, "validator_paths", "post", "run_info")
#                 )

#                 pre_t = maybe_float(
#                     safe_get(stage, "timing", "pre_validation")
#                 )
#                 if pre_t is None:
#                     pre_t = maybe_float(pre_run_info.get("execution_time_seconds") if pre_run_info else None)

#                 post_t = maybe_float(
#                     safe_get(stage, "timing", "post_validation")
#                 )
#                 if post_t is None:
#                     post_t = maybe_float(post_run_info.get("execution_time_seconds") if post_run_info else None)

#                 deploy_t = maybe_float(safe_get(stage, "timing", "deploy"))
#                 stage_t = maybe_float(safe_get(stage, "timing", "total"))

#                 if pre_t is not None:
#                     validators_acc += pre_t
#                     validators_found = True
#                 if post_t is not None:
#                     validators_acc += post_t
#                     validators_found = True
#                 if deploy_t is not None:
#                     deploy_acc += deploy_t
#                     deploy_found = True
#                 if stage_t is not None:
#                     stage_total_acc += stage_t
#                     stage_total_found = True

#             if validators_total_time is None and validators_found:
#                 validators_total_time = validators_acc
#             if deploy_total_time is None and deploy_found:
#                 deploy_total_time = deploy_acc
#             if stages_total_time is None and stage_total_found:
#                 stages_total_time = stage_total_acc

#         if cycle_total_time is None:
#             cycle_total_calc = 0.0
#             cycle_total_known = False

#             for t in [
#                 scene_description_time,
#                 scene_enrichment_time,
#                 planning_time,
#                 simultaneous_time,
#             ]:
#                 if t is not None:
#                     cycle_total_calc += t
#                     cycle_total_known = True

#             if stages_total_time is not None:
#                 cycle_total_calc += stages_total_time
#                 cycle_total_known = True

#             cycle_total_time = cycle_total_calc if cycle_total_known else None

#         rows.append(
#             {
#                 "scenario": scenario,
#                 "loop_timestamp": loop_timestamp,
#                 "cycle_index": cycle_index,
#                 "cycle_name": cycle_name,
#                 "scene_description_time": scene_description_time,
#                 "scene_enrichment_time": scene_enrichment_time,
#                 "planning_time": planning_time,
#                 "simultaneous_time": simultaneous_time,
#                 "validators_total_time": validators_total_time,
#                 "deploy_total_time": deploy_total_time,
#                 "stages_total_time": stages_total_time,
#                 "cycle_total_time": cycle_total_time,
#                 "num_stages": len(cycle.get("stages", [])),
#                 "outcome": outcome,
#                 "replan_happened": replan_happened,
#                 "start_image": cycle.get("start_image_path"),
#                 "end_image": cycle.get("end_image_path"),
#             }
#         )

#     return rows


# def build_run_summary_rows(full_summary: dict[str, Any]) -> list[dict[str, Any]]:
#     return [
#         {
#             "scenario": full_summary.get("scenario_name"),
#             "loop_timestamp": full_summary.get("loop_timestamp"),
#             "initial_homing_time_seconds": maybe_float(full_summary.get("initial_homing_time_seconds")),
#             "initial_screenshot_time_seconds": maybe_float(full_summary.get("initial_screenshot_time_seconds")),
#             "total_execution_time_seconds": maybe_float(full_summary.get("total_execution_time_seconds")),
#             "replans_done": full_summary.get("replans_done"),
#             "task_completed": full_summary.get("task_completed"),
#             "total_cycles": len(full_summary.get("cycles", [])),
#             "initial_image_path": full_summary.get("initial_image_path"),
#             "final_image_path": full_summary.get("final_image_path"),
#         }
#     ]


# def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
#     ensure_dir(path.parent)
#     with path.open("w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         for row in rows:
#             writer.writerow(row)


# def infer_default_output_dir(full_summary_path: Path) -> Path:
#     return full_summary_path.parent / "csv_exports"


# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description="Export events.csv, stage_summary.csv, cycle_summary.csv, and run_summary.csv from full_pipeline_summary.json."
#     )
#     parser.add_argument(
#         "--full-summary",
#         type=str,
#         required=True,
#         help="Path to full_pipeline_summary.json",
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default=None,
#         help="Directory where CSV files will be written. Default: sibling folder 'csv_exports'.",
#     )
#     return parser


# def main() -> None:
#     parser = build_parser()
#     args = parser.parse_args()

#     full_summary_path = Path(args.full_summary).resolve()
#     if not full_summary_path.exists():
#         raise FileNotFoundError(f"full_summary file not found: {full_summary_path}")

#     full_summary = read_json(full_summary_path)

#     output_dir = (
#         Path(args.output_dir).resolve()
#         if args.output_dir is not None
#         else infer_default_output_dir(full_summary_path)
#     )
#     ensure_dir(output_dir)

#     events_rows = build_events_rows(full_summary)
#     stage_rows = build_stage_summary_rows(full_summary)
#     cycle_rows = build_cycle_summary_rows(full_summary)
#     run_rows = build_run_summary_rows(full_summary)

#     write_csv(
#         output_dir / "events.csv",
#         events_rows,
#         fieldnames=[
#             "run_id",
#             "scenario",
#             "loop_timestamp",
#             "cycle_index",
#             "cycle_name",
#             "stage_id",
#             "stage_name",
#             "event_type",
#             "module_name",
#             "submodule_name",
#             "model_name",
#             "duration_sec",
#             "outcome",
#             "replan_triggered",
#             "image_before",
#             "image_after",
#         ],
#     )

#     write_csv(
#         output_dir / "stage_summary.csv",
#         stage_rows,
#         fieldnames=[
#             "scenario",
#             "loop_timestamp",
#             "cycle_index",
#             "cycle_name",
#             "stage_id",
#             "stage_name",
#             "pre_validation_time",
#             "deploy_time",
#             "screenshot_time",
#             "post_validation_time",
#             "stage_total_time",
#             "pre_result",
#             "post_result",
#             "replan_after_stage",
#             "image_before",
#             "image_after",
#         ],
#     )

#     write_csv(
#         output_dir / "cycle_summary.csv",
#         cycle_rows,
#         fieldnames=[
#             "scenario",
#             "loop_timestamp",
#             "cycle_index",
#             "cycle_name",
#             "scene_description_time",
#             "scene_enrichment_time",
#             "planning_time",
#             "simultaneous_time",
#             "validators_total_time",
#             "deploy_total_time",
#             "stages_total_time",
#             "cycle_total_time",
#             "num_stages",
#             "outcome",
#             "replan_happened",
#             "start_image",
#             "end_image",
#         ],
#     )

#     write_csv(
#         output_dir / "run_summary.csv",
#         run_rows,
#         fieldnames=[
#             "scenario",
#             "loop_timestamp",
#             "initial_homing_time_seconds",
#             "initial_screenshot_time_seconds",
#             "total_execution_time_seconds",
#             "replans_done",
#             "task_completed",
#             "total_cycles",
#             "initial_image_path",
#             "final_image_path",
#         ],
#     )

#     print("CSV export completed.")
#     print(f"Input full summary:  {full_summary_path}")
#     print(f"Output directory:    {output_dir}")
#     print(f"events.csv rows:     {len(events_rows)}")
#     print(f"stage_summary rows:  {len(stage_rows)}")
#     print(f"cycle_summary rows:  {len(cycle_rows)}")
#     print(f"run_summary rows:    {len(run_rows)}")


# if __name__ == "__main__":
#     main()







# # from __future__ import annotations

# # import argparse
# # import csv
# # import json
# # from pathlib import Path
# # from typing import Any


# # def read_json(path: Path) -> dict[str, Any]:
# #     with path.open("r", encoding="utf-8") as f:
# #         return json.load(f)


# # def ensure_dir(path: Path) -> Path:
# #     path.mkdir(parents=True, exist_ok=True)
# #     return path


# # def safe_get(d: dict[str, Any] | None, *keys: str, default=None):
# #     cur = d
# #     for k in keys:
# #         if not isinstance(cur, dict) or k not in cur:
# #             return default
# #         cur = cur[k]
# #     return cur


# # def maybe_float(x: Any) -> float | None:
# #     if x is None:
# #         return None
# #     try:
# #         return float(x)
# #     except (TypeError, ValueError):
# #         return None


# # def load_json_if_exists(path_str: str | None) -> dict[str, Any] | None:
# #     if not path_str:
# #         return None
# #     path = Path(path_str)
# #     if not path.exists():
# #         return None
# #     try:
# #         return read_json(path)
# #     except Exception:
# #         return None


# # def normalize_outcome_to_replan_flag(outcome: str | None) -> bool:
# #     if not outcome:
# #         return False
# #     return outcome.startswith("replan_on_")


# # def add_event_row(
# #     rows: list[dict[str, Any]],
# #     *,
# #     run_id: str,
# #     scenario: str,
# #     loop_timestamp: str,
# #     cycle_index: int,
# #     cycle_name: str,
# #     stage_id: int | None,
# #     stage_name: str | None,
# #     event_type: str,
# #     module_name: str,
# #     submodule_name: str | None,
# #     model_name: str | None,
# #     duration_sec: float | None,
# #     outcome: str | None,
# #     replan_triggered: bool,
# #     image_before: str | None,
# #     image_after: str | None,
# # ) -> None:
# #     rows.append(
# #         {
# #             "run_id": run_id,
# #             "scenario": scenario,
# #             "loop_timestamp": loop_timestamp,
# #             "cycle_index": cycle_index,
# #             "cycle_name": cycle_name,
# #             "stage_id": stage_id,
# #             "stage_name": stage_name,
# #             "event_type": event_type,
# #             "module_name": module_name,
# #             "submodule_name": submodule_name,
# #             "model_name": model_name,
# #             "duration_sec": duration_sec,
# #             "outcome": outcome,
# #             "replan_triggered": replan_triggered,
# #             "image_before": image_before,
# #             "image_after": image_after,
# #         }
# #     )


# # def build_events_rows(full_summary: dict[str, Any]) -> list[dict[str, Any]]:
# #     rows: list[dict[str, Any]] = []

# #     scenario = full_summary["scenario_name"]
# #     loop_timestamp = full_summary["loop_timestamp"]
# #     run_id = loop_timestamp

# #     cycles = full_summary.get("cycles", [])
# #     for cycle in cycles:
# #         cycle_index = cycle["cycle_index"]
# #         cycle_name = cycle["cycle_name"]
# #         cycle_outcome = cycle.get("outcome")
# #         cycle_replan = normalize_outcome_to_replan_flag(cycle_outcome)
# #         start_image = cycle.get("start_image_path")
# #         end_image = cycle.get("end_image_path")

# #         # --------------------------------------------------
# #         # Main modules from cycle_record
# #         # --------------------------------------------------
# #         scene_desc = cycle.get("scene_description")
# #         if isinstance(scene_desc, dict):
# #             add_event_row(
# #                 rows,
# #                 run_id=run_id,
# #                 scenario=scenario,
# #                 loop_timestamp=loop_timestamp,
# #                 cycle_index=cycle_index,
# #                 cycle_name=cycle_name,
# #                 stage_id=None,
# #                 stage_name=None,
# #                 event_type="scene_description",
# #                 module_name="scene_description",
# #                 submodule_name=None,
# #                 model_name=scene_desc.get("model_name"),
# #                 duration_sec=maybe_float(scene_desc.get("execution_time_seconds")),
# #                 outcome=None,
# #                 replan_triggered=cycle_replan,
# #                 image_before=start_image,
# #                 image_after=start_image,
# #             )

# #         scene_full = cycle.get("scene_description_full")
# #         if isinstance(scene_full, dict):
# #             add_event_row(
# #                 rows,
# #                 run_id=run_id,
# #                 scenario=scenario,
# #                 loop_timestamp=loop_timestamp,
# #                 cycle_index=cycle_index,
# #                 cycle_name=cycle_name,
# #                 stage_id=None,
# #                 stage_name=None,
# #                 event_type="scene_enrichment",
# #                 module_name="scene_description_full",
# #                 submodule_name="enrich_scene",
# #                 model_name=None,
# #                 duration_sec=maybe_float(scene_full.get("execution_time_seconds")),
# #                 outcome=None,
# #                 replan_triggered=cycle_replan,
# #                 image_before=start_image,
# #                 image_after=start_image,
# #             )

# #         planning = cycle.get("vlm_planning")
# #         if isinstance(planning, dict):
# #             add_event_row(
# #                 rows,
# #                 run_id=run_id,
# #                 scenario=scenario,
# #                 loop_timestamp=loop_timestamp,
# #                 cycle_index=cycle_index,
# #                 cycle_name=cycle_name,
# #                 stage_id=None,
# #                 stage_name=None,
# #                 event_type="vlm_planning",
# #                 module_name="vlm_planning",
# #                 submodule_name=None,
# #                 model_name=planning.get("model_name"),
# #                 duration_sec=maybe_float(planning.get("execution_time_seconds")),
# #                 outcome=None,
# #                 replan_triggered=cycle_replan,
# #                 image_before=start_image,
# #                 image_after=start_image,
# #             )

# #         sim = cycle.get("simultaneous_actions")
# #         if isinstance(sim, dict):
# #             add_event_row(
# #                 rows,
# #                 run_id=run_id,
# #                 scenario=scenario,
# #                 loop_timestamp=loop_timestamp,
# #                 cycle_index=cycle_index,
# #                 cycle_name=cycle_name,
# #                 stage_id=None,
# #                 stage_name=None,
# #                 event_type="simultaneous_actions",
# #                 module_name="simultaneous_actions",
# #                 submodule_name=None,
# #                 model_name=sim.get("model_name"),
# #                 duration_sec=maybe_float(sim.get("execution_time_seconds")),
# #                 outcome=None,
# #                 replan_triggered=cycle_replan,
# #                 image_before=start_image,
# #                 image_after=start_image,
# #             )

# #         # --------------------------------------------------
# #         # Stage-level validator events
# #         # --------------------------------------------------
# #         for stage in cycle.get("stages", []):
# #             stage_id = stage.get("stage_id")
# #             stage_name = stage.get("stage_name")
# #             pre_image = stage.get("pre_image_path")
# #             post_image = stage.get("post_image_path") or stage.get("next_image_path")

# #             pre_result = safe_get(stage, "pre_validation", "result")
# #             post_result = safe_get(stage, "post_validation", "result")

# #             pre_run_info_path = safe_get(stage, "validator_paths", "pre", "run_info")
# #             post_run_info_path = safe_get(stage, "validator_paths", "post", "run_info")

# #             pre_run_info = load_json_if_exists(pre_run_info_path)
# #             post_run_info = load_json_if_exists(post_run_info_path)

# #             if pre_run_info is not None:
# #                 add_event_row(
# #                     rows,
# #                     run_id=run_id,
# #                     scenario=scenario,
# #                     loop_timestamp=loop_timestamp,
# #                     cycle_index=cycle_index,
# #                     cycle_name=cycle_name,
# #                     stage_id=stage_id,
# #                     stage_name=stage_name,
# #                     event_type="validator_pre",
# #                     module_name="validator",
# #                     submodule_name="pre",
# #                     model_name=pre_run_info.get("model"),
# #                     duration_sec=maybe_float(pre_run_info.get("execution_time_seconds")),
# #                     outcome=pre_result,
# #                     replan_triggered=cycle_replan and pre_result == "non_matching",
# #                     image_before=pre_image,
# #                     image_after=pre_image,
# #                 )

# #             # Optional future compatibility:
# #             # if you later add stage["timing"]["deploy"], it will be exported here.
# #             deploy_time = maybe_float(safe_get(stage, "timing", "deploy"))
# #             if deploy_time is not None:
# #                 add_event_row(
# #                     rows,
# #                     run_id=run_id,
# #                     scenario=scenario,
# #                     loop_timestamp=loop_timestamp,
# #                     cycle_index=cycle_index,
# #                     cycle_name=cycle_name,
# #                     stage_id=stage_id,
# #                     stage_name=stage_name,
# #                     event_type="deploy",
# #                     module_name="deploy",
# #                     submodule_name=None,
# #                     model_name=None,
# #                     duration_sec=deploy_time,
# #                     outcome="success",
# #                     replan_triggered=False,
# #                     image_before=pre_image,
# #                     image_after=post_image,
# #                 )

# #             # Optional future compatibility for script-level timing
# #             deploy_scripts = safe_get(stage, "timing", "deploy_scripts", default=[])
# #             if isinstance(deploy_scripts, list):
# #                 for item in deploy_scripts:
# #                     if not isinstance(item, dict):
# #                         continue
# #                     add_event_row(
# #                         rows,
# #                         run_id=run_id,
# #                         scenario=scenario,
# #                         loop_timestamp=loop_timestamp,
# #                         cycle_index=cycle_index,
# #                         cycle_name=cycle_name,
# #                         stage_id=stage_id,
# #                         stage_name=stage_name,
# #                         event_type=item.get("event_type", "manipulation_script"),
# #                         module_name=item.get("module_name", "deploy"),
# #                         submodule_name=item.get("script_name"),
# #                         model_name=None,
# #                         duration_sec=maybe_float(item.get("duration_sec")),
# #                         outcome=item.get("outcome", "success"),
# #                         replan_triggered=False,
# #                         image_before=item.get("image_before", pre_image),
# #                         image_after=item.get("image_after", post_image),
# #                     )

# #             if post_run_info is not None:
# #                 add_event_row(
# #                     rows,
# #                     run_id=run_id,
# #                     scenario=scenario,
# #                     loop_timestamp=loop_timestamp,
# #                     cycle_index=cycle_index,
# #                     cycle_name=cycle_name,
# #                     stage_id=stage_id,
# #                     stage_name=stage_name,
# #                     event_type="validator_post",
# #                     module_name="validator",
# #                     submodule_name="post",
# #                     model_name=post_run_info.get("model"),
# #                     duration_sec=maybe_float(post_run_info.get("execution_time_seconds")),
# #                     outcome=post_result,
# #                     replan_triggered=cycle_replan and post_result == "non_matching",
# #                     image_before=post_image,
# #                     image_after=post_image,
# #                 )

# #     return rows


# # def build_stage_summary_rows(full_summary: dict[str, Any]) -> list[dict[str, Any]]:
# #     rows: list[dict[str, Any]] = []

# #     scenario = full_summary["scenario_name"]
# #     loop_timestamp = full_summary["loop_timestamp"]

# #     for cycle in full_summary.get("cycles", []):
# #         cycle_index = cycle["cycle_index"]
# #         cycle_name = cycle["cycle_name"]
# #         cycle_outcome = cycle.get("outcome")
# #         cycle_replan = normalize_outcome_to_replan_flag(cycle_outcome)

# #         for stage in cycle.get("stages", []):
# #             stage_id = stage.get("stage_id")
# #             stage_name = stage.get("stage_name")

# #             pre_run_info_path = safe_get(stage, "validator_paths", "pre", "run_info")
# #             post_run_info_path = safe_get(stage, "validator_paths", "post", "run_info")

# #             pre_run_info = load_json_if_exists(pre_run_info_path)
# #             post_run_info = load_json_if_exists(post_run_info_path)

# #             pre_validation_time = maybe_float(
# #                 pre_run_info.get("execution_time_seconds") if pre_run_info else None
# #             )
# #             post_validation_time = maybe_float(
# #                 post_run_info.get("execution_time_seconds") if post_run_info else None
# #             )

# #             deploy_time = maybe_float(safe_get(stage, "timing", "deploy"))

# #             # stage_total_time:
# #             # - if future explicit field exists, use it
# #             # - otherwise sum only known components
# #             explicit_stage_total = maybe_float(safe_get(stage, "timing", "total"))
# #             if explicit_stage_total is not None:
# #                 stage_total_time = explicit_stage_total
# #             else:
# #                 parts = [
# #                     x for x in [pre_validation_time, deploy_time, post_validation_time]
# #                     if x is not None
# #                 ]
# #                 stage_total_time = sum(parts) if parts else None

# #             pre_result = safe_get(stage, "pre_validation", "result")
# #             post_result = safe_get(stage, "post_validation", "result")

# #             replan_after_stage = False
# #             if cycle_replan and isinstance(cycle_outcome, str):
# #                 if cycle_outcome == f"replan_on_pre_stage_{stage_id}":
# #                     replan_after_stage = True
# #                 if cycle_outcome == f"replan_on_post_stage_{stage_id}":
# #                     replan_after_stage = True

# #             rows.append(
# #                 {
# #                     "scenario": scenario,
# #                     "loop_timestamp": loop_timestamp,
# #                     "cycle_index": cycle_index,
# #                     "cycle_name": cycle_name,
# #                     "stage_id": stage_id,
# #                     "stage_name": stage_name,
# #                     "pre_validation_time": pre_validation_time,
# #                     "deploy_time": deploy_time,
# #                     "post_validation_time": post_validation_time,
# #                     "stage_total_time": stage_total_time,
# #                     "pre_result": pre_result,
# #                     "post_result": post_result,
# #                     "replan_after_stage": replan_after_stage,
# #                     "image_before": stage.get("pre_image_path"),
# #                     "image_after": stage.get("post_image_path") or stage.get("next_image_path"),
# #                 }
# #             )

# #     return rows


# # def build_cycle_summary_rows(full_summary: dict[str, Any]) -> list[dict[str, Any]]:
# #     rows: list[dict[str, Any]] = []

# #     scenario = full_summary["scenario_name"]
# #     loop_timestamp = full_summary["loop_timestamp"]

# #     for cycle in full_summary.get("cycles", []):
# #         cycle_index = cycle["cycle_index"]
# #         cycle_name = cycle["cycle_name"]
# #         outcome = cycle.get("outcome")
# #         replan_happened = normalize_outcome_to_replan_flag(outcome)

# #         scene_description_time = maybe_float(
# #             safe_get(cycle, "scene_description", "execution_time_seconds")
# #         )
# #         scene_enrichment_time = maybe_float(
# #             safe_get(cycle, "scene_description_full", "execution_time_seconds")
# #         )
# #         planning_time = maybe_float(
# #             safe_get(cycle, "vlm_planning", "execution_time_seconds")
# #         )
# #         simultaneous_time = maybe_float(
# #             safe_get(cycle, "simultaneous_actions", "execution_time_seconds")
# #         )

# #         validators_total_time = 0.0
# #         validators_found = False

# #         deploy_total_time = 0.0
# #         deploy_found = False

# #         for stage in cycle.get("stages", []):
# #             pre_run_info = load_json_if_exists(
# #                 safe_get(stage, "validator_paths", "pre", "run_info")
# #             )
# #             post_run_info = load_json_if_exists(
# #                 safe_get(stage, "validator_paths", "post", "run_info")
# #             )

# #             pre_t = maybe_float(pre_run_info.get("execution_time_seconds") if pre_run_info else None)
# #             post_t = maybe_float(post_run_info.get("execution_time_seconds") if post_run_info else None)

# #             if pre_t is not None:
# #                 validators_total_time += pre_t
# #                 validators_found = True
# #             if post_t is not None:
# #                 validators_total_time += post_t
# #                 validators_found = True

# #             d_t = maybe_float(safe_get(stage, "timing", "deploy"))
# #             if d_t is not None:
# #                 deploy_total_time += d_t
# #                 deploy_found = True

# #         stages_total_time = 0.0
# #         stages_found = False
# #         if validators_found:
# #             stages_total_time += validators_total_time
# #             stages_found = True
# #         if deploy_found:
# #             stages_total_time += deploy_total_time
# #             stages_found = True

# #         cycle_total_time = 0.0
# #         cycle_total_known = False
# #         for t in [
# #             scene_description_time,
# #             scene_enrichment_time,
# #             planning_time,
# #             simultaneous_time,
# #         ]:
# #             if t is not None:
# #                 cycle_total_time += t
# #                 cycle_total_known = True

# #         if stages_found:
# #             cycle_total_time += stages_total_time
# #             cycle_total_known = True

# #         rows.append(
# #             {
# #                 "scenario": scenario,
# #                 "loop_timestamp": loop_timestamp,
# #                 "cycle_index": cycle_index,
# #                 "cycle_name": cycle_name,
# #                 "scene_description_time": scene_description_time,
# #                 "scene_enrichment_time": scene_enrichment_time,
# #                 "planning_time": planning_time,
# #                 "simultaneous_time": simultaneous_time,
# #                 "validators_total_time": validators_total_time if validators_found else None,
# #                 "deploy_total_time": deploy_total_time if deploy_found else None,
# #                 "stages_total_time": stages_total_time if stages_found else None,
# #                 "cycle_total_time": cycle_total_time if cycle_total_known else None,
# #                 "num_stages": len(cycle.get("stages", [])),
# #                 "outcome": outcome,
# #                 "replan_happened": replan_happened,
# #                 "start_image": cycle.get("start_image_path"),
# #                 "end_image": cycle.get("end_image_path"),
# #             }
# #         )

# #     return rows


# # def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
# #     ensure_dir(path.parent)
# #     with path.open("w", newline="", encoding="utf-8") as f:
# #         writer = csv.DictWriter(f, fieldnames=fieldnames)
# #         writer.writeheader()
# #         for row in rows:
# #             writer.writerow(row)


# # def infer_default_output_dir(full_summary_path: Path) -> Path:
# #     # Save CSVs next to full_pipeline_summary.json by default
# #     return full_summary_path.parent / "csv_exports"


# # def build_parser() -> argparse.ArgumentParser:
# #     parser = argparse.ArgumentParser(
# #         description="Export events.csv, stage_summary.csv, and cycle_summary.csv from full_pipeline_summary.json."
# #     )
# #     parser.add_argument(
# #         "--full-summary",
# #         type=str,
# #         required=True,
# #         help="Path to full_pipeline_summary.json",
# #     )
# #     parser.add_argument(
# #         "--output-dir",
# #         type=str,
# #         default=None,
# #         help="Directory where CSV files will be written. Default: sibling folder 'csv_exports'.",
# #     )
# #     return parser


# # def main() -> None:
# #     parser = build_parser()
# #     args = parser.parse_args()

# #     full_summary_path = Path(args.full_summary).resolve()
# #     if not full_summary_path.exists():
# #         raise FileNotFoundError(f"full_summary file not found: {full_summary_path}")

# #     full_summary = read_json(full_summary_path)

# #     output_dir = (
# #         Path(args.output_dir).resolve()
# #         if args.output_dir is not None
# #         else infer_default_output_dir(full_summary_path)
# #     )
# #     ensure_dir(output_dir)

# #     events_rows = build_events_rows(full_summary)
# #     stage_rows = build_stage_summary_rows(full_summary)
# #     cycle_rows = build_cycle_summary_rows(full_summary)

# #     write_csv(
# #         output_dir / "events.csv",
# #         events_rows,
# #         fieldnames=[
# #             "run_id",
# #             "scenario",
# #             "loop_timestamp",
# #             "cycle_index",
# #             "cycle_name",
# #             "stage_id",
# #             "stage_name",
# #             "event_type",
# #             "module_name",
# #             "submodule_name",
# #             "model_name",
# #             "duration_sec",
# #             "outcome",
# #             "replan_triggered",
# #             "image_before",
# #             "image_after",
# #         ],
# #     )

# #     write_csv(
# #         output_dir / "stage_summary.csv",
# #         stage_rows,
# #         fieldnames=[
# #             "scenario",
# #             "loop_timestamp",
# #             "cycle_index",
# #             "cycle_name",
# #             "stage_id",
# #             "stage_name",
# #             "pre_validation_time",
# #             "deploy_time",
# #             "post_validation_time",
# #             "stage_total_time",
# #             "pre_result",
# #             "post_result",
# #             "replan_after_stage",
# #             "image_before",
# #             "image_after",
# #         ],
# #     )

# #     write_csv(
# #         output_dir / "cycle_summary.csv",
# #         cycle_rows,
# #         fieldnames=[
# #             "scenario",
# #             "loop_timestamp",
# #             "cycle_index",
# #             "cycle_name",
# #             "scene_description_time",
# #             "scene_enrichment_time",
# #             "planning_time",
# #             "simultaneous_time",
# #             "validators_total_time",
# #             "deploy_total_time",
# #             "stages_total_time",
# #             "cycle_total_time",
# #             "num_stages",
# #             "outcome",
# #             "replan_happened",
# #             "start_image",
# #             "end_image",
# #         ],
# #     )

# #     print("CSV export completed.")
# #     print(f"Input full summary:  {full_summary_path}")
# #     print(f"Output directory:    {output_dir}")
# #     print(f"events.csv rows:     {len(events_rows)}")
# #     print(f"stage_summary rows:  {len(stage_rows)}")
# #     print(f"cycle_summary rows:  {len(cycle_rows)}")


# # if __name__ == "__main__":
# #     main()