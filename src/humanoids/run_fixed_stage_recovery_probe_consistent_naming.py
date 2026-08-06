"""
run_fixed_stage_recovery_probe_consistent_naming.py — same probe tool as
run_fixed_stage_recovery_probe.py, wired to the two naming-consistency fixes
instead of the original scene-perception / scene-transition functions.

WHY THIS IS A SEPARATE FILE
----------------------------
run_fixed_stage_recovery_probe.py stays exactly as it was: a faithful probe
of the recovery engine using the pipeline's original, unmodified functions.
This file exists so both versions can be run side by side and compared,
instead of overwriting the original tool. It duplicates only the orchestration
loop; every validator/recovery/modification-proposal building block is still
imported unchanged from run_validation_loop_fixed_on_demand.py and its
sibling modules. The only two swapped-in pieces are:

  - execute_scene_perception_for_state_consistent_naming
    (scene_perception_consistent_naming.py) instead of
    execute_scene_perception_for_state: tells the recovery-state
    scene-perception call to reuse the object identities already established
    earlier in the same cycle, instead of treating each call as a
    context-free "fresh observation" free to invent new names.

  - analyze_scene_transition_robust (scene_transition_analysis_robust.py)
    instead of analyze_scene_transition: a defense-in-depth layer that also
    accepts a color/category match as evidence a target is still present,
    not only an exact name match.

Artifacts are written under outputs/fixed_stage_probe_consistent_naming/ so
they never collide with outputs/fixed_stage_probe/ produced by the original
tool.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from src.settings import load_settings
from src.scenario_loader import load_scenario

from src.humanoids.failure_reporting import (
    assert_failure_report,
    build_failure_report,
    build_uncertainty_exhausted_report,
)
from src.humanoids.recovery_and_history import (
    SUPPORTED_SYMBOLIC_FIELDS,
    check_recovery_limits,
    extract_relevant_history,
    interpret_failure,
    plan_recovery_evidence_based,
    repeat_assessment,
    schedule_recovery,
)
from src.humanoids.scene_transition_analysis_robust import (
    analyze_scene_transition_robust,
)
from src.humanoids.scene_perception_consistent_naming import (
    execute_scene_perception_for_state_consistent_naming,
)

from src.humanoids.run_validation_loop_fixed_on_demand import (
    SUPPORTED_MODELS,
    open_attempt,
    set_attempt_status,
    close_attempt,
    make_stage_name,
    make_cycle_name,
    make_experiment_timestamp,
    make_scenario_context,
    build_planned_stage_context,
    print_pose_dict_for_image,
    resolve_poses_by_image_path,
    load_poses_by_image_map,
    list_frame_paths,
    prompt_for_post_image,
    resolve_validator_prompt_versions,
    execute_scene_description_step,
    execute_scene_description_full_step,
    execute_validator_step,
    execute_goal_baseline_validator_step,
    execute_postcondition_validator_step,
    execute_modification_proposal_step,
    collect_all_attempts,
    extract_remaining_task_goal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the repeat/modify/replan recovery decision using a "
            "hand-authored stage, bypassing vlm_planning/simultaneous_actions. "
            "Uses the naming-consistency fixes for scene-perception and "
            "scene-transition analysis."
        )
    )
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--frames-dir", type=str, required=True)
    parser.add_argument("--poses-by-image-path", type=str, default=None)
    parser.add_argument(
        "--fixed-stage-path",
        type=str,
        required=True,
        help="Path to a JSON file with {'stage': {...}, 'actions': [...]}.",
    )

    parser.add_argument("--scene-v", type=str, required=True)
    parser.add_argument("--scene-model", type=str, required=True, choices=SUPPORTED_MODELS)

    parser.add_argument("--validator-v", type=str, default=None)
    parser.add_argument("--validator-pre-v", type=str, default=None)
    parser.add_argument("--validator-post-v", type=str, default=None)
    parser.add_argument("--validator-baseline-v", type=str, default=None)
    parser.add_argument("--validator-model", type=str, required=True, choices=SUPPORTED_MODELS)

    parser.add_argument("--mod-v", type=str, default="v1")
    parser.add_argument("--mod-model", type=str, default="gpt-5.2", choices=SUPPORTED_MODELS)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--grounding-safety-threshold", type=float, default=0.21)
    parser.add_argument("--grounding-debug-mapping", action="store_true")

    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--max-repeats", type=int, default=1)
    parser.add_argument("--max-modifications", type=int, default=2)
    parser.add_argument("--max-replans", type=int, default=3)
    parser.add_argument("--max-total-actions", type=int, default=20)

    return parser


def load_fixed_stage(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stage = data["stage"]
    actions = data["actions"]
    for key in ("Stage_id", "Step_id", "Local_goal", "Preconditions", "Postconditions"):
        if key not in stage:
            raise ValueError(f"Fixed stage file is missing required key: {key}")
    if not isinstance(actions, list) or not actions:
        raise ValueError("Fixed stage file must define a non-empty 'actions' list.")
    return deepcopy(stage), deepcopy(actions)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    resolve_validator_prompt_versions(args)

    settings = load_settings()
    scenario_data = load_scenario(settings, args.scenario)

    poses_by_image_path = resolve_poses_by_image_path(
        settings=settings,
        scenario_name=args.scenario,
        explicit_path=args.poses_by_image_path,
    )
    poses_by_image = load_poses_by_image_map(poses_by_image_path)
    frame_paths = list_frame_paths(args.frames_dir)

    stage, actions = load_fixed_stage(args.fixed_stage_path)
    stage_id = stage["Stage_id"]

    loop_timestamp = make_experiment_timestamp()
    limits = {
        "max_attempts_per_stage": args.max_attempts,
        "max_repeats": args.max_repeats,
        "max_modifications": args.max_modifications,
        "max_replans": args.max_replans,
        "max_total_actions": args.max_total_actions,
    }
    full_summary: dict[str, Any] = {
        "cycles": [],
        "recovery_counters": {"replans": 0, "total_actions": 0},
    }

    print("\n======================================================")
    print("FIXED-STAGE RECOVERY PROBE (consistent-naming fix)")
    print(f"Scenario:     {args.scenario}")
    print(f"Fixed stage:  {args.fixed_stage_path}")
    print(f"Local goal:   {stage['Local_goal']}")
    print(f"Preconditions:  {json.dumps(stage['Preconditions'], ensure_ascii=False)}")
    print(f"Postconditions: {json.dumps(stage['Postconditions'], ensure_ascii=False)}")
    print("======================================================")

    current_image = frame_paths[0]
    parent_attempt_id: str | None = None
    recovery_type: str | None = None
    recovery_changes: dict[str, Any] | None = None

    for attempt_number in range(1, args.max_attempts + 1):
        cycle_name = make_cycle_name(attempt_number)
        cycle_record: dict[str, Any] = {"cycle_name": cycle_name, "attempts": []}

        print(f"\n------ Attempt {attempt_number} | I_pre = {Path(current_image).name} ------")

        scenario_context = make_scenario_context(scenario_data=scenario_data, image_path=current_image)
        pipeline_config: dict[str, Any] = {"probe_tool": True, "cycle_name": cycle_name}

        scene_description_artifact = execute_scene_description_step(
            settings=settings,
            scenario_name=args.scenario,
            scenario_context=scenario_context,
            version=args.scene_v,
            model_name=args.scene_model,
            loop_timestamp=loop_timestamp,
            cycle_name=cycle_name,
            cycle_idx=attempt_number,
            cycle_timestamp=make_experiment_timestamp(),
            pipeline_config=pipeline_config,
            image_path=current_image,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # Established object identities (name/category/color) for this
        # attempt, used as the naming reference for both recovery-state
        # scene-perception calls below.
        reference_scene_description = scene_description_artifact["output"]

        scene_description_full_artifact = execute_scene_description_full_step(
            settings=settings,
            scenario_name=args.scenario,
            scenario_context=scenario_context,
            version=args.scene_v,
            model_name=args.scene_model,
            loop_timestamp=loop_timestamp,
            cycle_name=cycle_name,
            cycle_idx=attempt_number,
            cycle_timestamp=make_experiment_timestamp(),
            scene_description=scene_description_artifact["output"],
            pipeline_config=pipeline_config,
            image_path=current_image,
            poses_by_image=poses_by_image,
            safety_threshold=args.grounding_safety_threshold,
            include_debug_mapping=args.grounding_debug_mapping,
        )

        planned_stage_context = build_planned_stage_context(stage)

        attempt_record = open_attempt(
            cycle_idx=attempt_number,
            stage=stage,
            attempt_idx=attempt_number,
            pre_image_path=current_image,
            pre_scene_description_full_path=scene_description_full_artifact["paths"]["artifact"],
            parent_attempt_id=parent_attempt_id,
            recovery_type=recovery_type,
            recovery_changes=recovery_changes,
        )
        cycle_record["attempts"].append(attempt_record)
        set_attempt_status(attempt_record, "awaiting_pre_validation")

        print_pose_dict_for_image(
            poses_by_image=poses_by_image,
            image_path=current_image,
            label=f"probe-pre-stage-{stage_id}",
        )

        pre_artifact = execute_validator_step(
            settings=settings,
            scenario_name=args.scenario,
            validator_version=args.validator_pre_v,
            validator_model=args.validator_model,
            loop_timestamp=loop_timestamp,
            cycle_name=cycle_name,
            cycle_idx=attempt_number,
            cycle_timestamp=make_experiment_timestamp(),
            stage_id=stage_id,
            planned_stage_context=planned_stage_context,
            preconditions=stage["Preconditions"],
            image_path=current_image,
            scene_version=args.scene_v,
            scene_model=args.scene_model,
            plan_version="n/a",
            plan_model="n/a",
            sim_version="n/a",
            sim_model="n/a",
            temperature=args.temperature,
            top_p=args.top_p,
        )
        attempt_record["pre"]["validation"] = pre_artifact["output"]
        pre_status = pre_artifact["output"]["overall_status"]
        print(f"[PROBE] PRE overall status: {pre_status}")
        if pre_status != "satisfied":
            print(
                "[PROBE] PRE conditions are not satisfied on this image "
                "(no evidence-gathering loop in this probe tool). Stopping."
            )
            return
        set_attempt_status(attempt_record, "preconditions_satisfied")

        goal_baseline_artifact = execute_goal_baseline_validator_step(
            settings=settings,
            scenario_name=args.scenario,
            validator_version=args.validator_baseline_v,
            validator_model=args.validator_model,
            loop_timestamp=loop_timestamp,
            cycle_name=cycle_name,
            cycle_idx=attempt_number,
            cycle_timestamp=make_experiment_timestamp(),
            stage_id=stage_id,
            planned_stage_context=planned_stage_context,
            postconditions=stage["Postconditions"],
            image_path=current_image,
            scene_version=args.scene_v,
            scene_model=args.scene_model,
            plan_version="n/a",
            plan_model="n/a",
            sim_version="n/a",
            sim_model="n/a",
            temperature=args.temperature,
            top_p=args.top_p,
        )
        attempt_record["pre"]["goal_baseline_validation"] = goal_baseline_artifact["output"]
        print(f"[PROBE] Goal baseline overall status: {goal_baseline_artifact['output']['overall_status']}")

        set_attempt_status(attempt_record, "executing")
        frame_cursor = frame_paths.index(str(Path(current_image).resolve()))
        post_cursor = prompt_for_post_image(frame_paths, frame_cursor)
        post_image = frame_paths[post_cursor]

        attempt_record["execution"]["started"] = True
        attempt_record["execution"]["started_at"] = datetime.now().isoformat()
        attempt_record["execution"]["mode"] = "probe_manual_frame_choice"
        attempt_record["execution"]["completed"] = True
        attempt_record["execution"]["completed_at"] = datetime.now().isoformat()
        attempt_record["post"]["image_path"] = str(Path(post_image).resolve())
        attempt_record["post"]["image_name"] = Path(post_image).name
        set_attempt_status(attempt_record, "awaiting_post_validation")

        post_artifact = execute_postcondition_validator_step(
            settings=settings,
            scenario_name=args.scenario,
            validator_version=args.validator_post_v,
            validator_model=args.validator_model,
            loop_timestamp=loop_timestamp,
            cycle_name=cycle_name,
            cycle_idx=attempt_number,
            cycle_timestamp=make_experiment_timestamp(),
            stage_id=stage_id,
            planned_stage_context=planned_stage_context,
            actions=actions,
            postconditions=stage["Postconditions"],
            pre_image_path=current_image,
            post_image_path=post_image,
            scene_version=args.scene_v,
            scene_model=args.scene_model,
            plan_version="n/a",
            plan_model="n/a",
            sim_version="n/a",
            sim_model="n/a",
            temperature=args.temperature,
            top_p=args.top_p,
        )
        post_response = post_artifact["output"]
        attempt_record["post"]["validation"] = post_response
        post_status = post_response["overall_status"]
        print(f"[PROBE] POST overall status: {post_status}")

        if post_status == "satisfied":
            set_attempt_status(attempt_record, "postconditions_satisfied")
            close_attempt(attempt_record, status="closed_success")
            cycle_record["outcome"] = "success"
            full_summary["cycles"].append(cycle_record)
            print("\n[PROBE] Postconditions satisfied. Stage succeeded, nothing to recover.")
            return

        recovery_pre_scene_graph = execute_scene_perception_for_state_consistent_naming(
            settings=settings,
            scenario_name=args.scenario,
            scenario_data=scenario_data,
            image_path=current_image,
            poses_by_image=poses_by_image,
            scene_version=args.scene_v,
            scene_model=args.scene_model,
            temperature=args.temperature,
            top_p=args.top_p,
            safety_threshold=args.grounding_safety_threshold,
            include_debug_mapping=args.grounding_debug_mapping,
            output_dir=Path(settings.project_root) / "outputs" / "fixed_stage_probe_consistent_naming" / args.scenario / loop_timestamp / cycle_name / "pre_recovery",
            purpose=f"pre_recovery_state_stage_{stage_id}",
            reference_scene_description=reference_scene_description,
        )["scene_graph"]
        post_scene_graph = execute_scene_perception_for_state_consistent_naming(
            settings=settings,
            scenario_name=args.scenario,
            scenario_data=scenario_data,
            image_path=post_image,
            poses_by_image=poses_by_image,
            scene_version=args.scene_v,
            scene_model=args.scene_model,
            temperature=args.temperature,
            top_p=args.top_p,
            safety_threshold=args.grounding_safety_threshold,
            include_debug_mapping=args.grounding_debug_mapping,
            output_dir=Path(settings.project_root) / "outputs" / "fixed_stage_probe_consistent_naming" / args.scenario / loop_timestamp / cycle_name / "post_recovery",
            purpose=f"post_recovery_state_stage_{stage_id}",
            reference_scene_description=reference_scene_description,
        )["scene_graph"]

        if post_status == "violated":
            failure_report = build_failure_report(
                attempt=attempt_record,
                failure_phase="post",
                failure_type="postcondition_failure",
                validation=post_response,
                action=actions,
                scene_graph_before=recovery_pre_scene_graph,
                scene_graph_after=post_scene_graph,
                relevant_history=[],
                evidence_rounds=[],
                notes="One or more expected postconditions were violated.",
            )
        else:
            failure_report = build_uncertainty_exhausted_report(
                attempt=attempt_record,
                phase="post",
                validation=post_response,
                action=actions,
                scene_graph_before=recovery_pre_scene_graph,
                scene_graph_after=post_scene_graph,
                relevant_history=[],
            )
        assert_failure_report(failure_report)
        close_attempt(attempt_record, status="closed_failure", failure_report=failure_report)
        cycle_record["outcome"] = "failure"
        full_summary["cycles"].append(cycle_record)

        relevant_history = extract_relevant_history(
            attempts=collect_all_attempts(full_summary),
            stage_id=stage_id,
            current_failure_report=failure_report,
            latest_scene_graph=post_scene_graph,
        )
        scene_transition = analyze_scene_transition_robust(
            scene_graph_before=recovery_pre_scene_graph,
            scene_graph_after=post_scene_graph,
            failed_stage=stage,
            actions=actions,
            before_goal_validation=goal_baseline_artifact["output"],
            after_goal_validation=post_response,
            reference_scene_description=reference_scene_description,
        )
        failure_interpretation = interpret_failure(
            failure_report=failure_report,
            relevant_history=relevant_history,
            failed_stage=stage,
            actions=actions,
            scene_transition=scene_transition,
        )
        print(
            "\n[RECOVERY][INTERPRETATION] "
            f"evidence={failure_interpretation['evidence_status']} | "
            f"phase={failure_interpretation['failure_phase']} | "
            f"goal_progress={failure_interpretation['goal_progress']} | "
            f"target_state_changed={failure_interpretation['target_state_changed']} | "
            f"post_validator_progress_observed="
            f"{failure_interpretation['post_validator_progress_observed']} | "
            f"same_failure_count={failure_interpretation['same_failure_count']} | "
            f"stage_still_applicable={failure_interpretation['stage_still_applicable']}"
        )

        repeat_ok, repeat_reason = repeat_assessment(interpretation=failure_interpretation)
        if (
            not repeat_ok
            and not failure_interpretation["replan_required"]
            and failure_interpretation["evidence_status"] == "sufficient"
        ):
            already_tried_values = [
                {
                    "action_index": change.get("action_index"),
                    "field": change.get("field"),
                    "new_value": change.get("new_value"),
                }
                for past_attempt in relevant_history.get("strategies_already_tried", [])
                if past_attempt.get("recovery_type") == "modify"
                for change in past_attempt.get("changes", {}).get("symbolic_modifications", [])
            ]
            already_tried_values.extend(
                {"action_index": index, "field": field, "new_value": action[field]}
                for index, action in enumerate(actions)
                for field in SUPPORTED_SYMBOLIC_FIELDS
                if field in action
            )
            attempt_image_paths = [
                path
                for past_attempt in relevant_history.get("same_stage_attempts", [])
                for path in (
                    past_attempt.get("pre", {}).get("image_path"),
                    past_attempt.get("post", {}).get("image_path"),
                )
                if path
            ]
            modification_proposal = execute_modification_proposal_step(
                settings=settings,
                scenario_name=args.scenario,
                version=args.mod_v,
                model_name=args.mod_model,
                loop_timestamp=loop_timestamp,
                cycle_name=cycle_name,
                cycle_idx=attempt_number,
                cycle_timestamp=make_experiment_timestamp(),
                stage_id=stage_id,
                local_goal=stage.get("Local_goal", ""),
                failed_conditions=failure_report.get("failed_conditions", []),
                failed_actions=actions,
                already_tried_values=already_tried_values,
                attempt_image_paths=attempt_image_paths,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            if modification_proposal["output"]["modification_supported"]:
                failure_report["supported_symbolic_modifications"] = (
                    modification_proposal["output"]["modifications"]
                )
                failure_interpretation = interpret_failure(
                    failure_report=failure_report,
                    relevant_history=relevant_history,
                    failed_stage=stage,
                    actions=actions,
                    scene_transition=scene_transition,
                )
                print(
                    "[RECOVERY][MODIFICATION_PROPOSAL] supported_modifications="
                    f"{len(failure_interpretation['supported_symbolic_modifications'])}"
                )

        full_summary["recovery_counters"]["total_actions"] += max(1, len(actions))
        check_recovery_limits(limits=limits, counters=full_summary["recovery_counters"])

        recovery_plan = plan_recovery_evidence_based(
            failure_report=failure_report,
            relevant_history=relevant_history,
            failure_interpretation=failure_interpretation,
            failed_stage=stage,
            actions=actions,
            remaining_task_goal=extract_remaining_task_goal(scenario_data),
            limits=limits,
            counters=full_summary["recovery_counters"],
        )
        for candidate, assessment in recovery_plan.get("admissibility", {}).items():
            print(
                f"[RECOVERY][CANDIDATE] {candidate}: "
                f"admissible={assessment.get('admissible')} | {assessment.get('reason')}"
            )

        decision = recovery_plan["decision"]
        print(f"\n[RECOVERY] decision={decision} | {recovery_plan['reason']}")

        if decision == "abort":
            print("[PROBE] Recovery aborted. Stopping.")
            return

        if decision == "replan":
            print(
                "[PROBE] This probe tool has no planner to consult for a "
                "global replan (that is intentional: it only tests repeat/"
                "modify against a fixed stage). Stopping here."
            )
            return

        recovery_schedule = schedule_recovery(
            recovery_plan=recovery_plan,
            failed_stage=stage,
            failed_actions=actions,
            remaining_stages=[],
            parent_attempt_id=attempt_record["attempt_id"],
            next_attempt_number=attempt_number + 1,
        )
        stage = recovery_schedule["stages"][0]
        actions = recovery_schedule["actions"]
        parent_attempt_id = attempt_record["attempt_id"]
        recovery_type = recovery_schedule["recovery"]["recovery_type"]
        recovery_changes = recovery_schedule["recovery"]["changes"]
        current_image = post_image

    print("\n[PROBE] Reached --max-attempts without a resolved outcome. Stopping.")


if __name__ == "__main__":
    main()
