"""Standalone baseline-ablation runner for the Table 1 / Table 2 comparison.

This is a SEPARATE script, not a modification of
`run_validation_loop_replan_with_feedback_soft_guidance.py`. It reuses that
module's tested step functions (scene_description, vlm_planning,
simultaneous_actions, PRE/POST validators) via import, but replaces the
production recovery machinery (attempt state machine, evidence-gathering
rounds, modification proposals, soft-guidance replanning) with a small,
explicit repeat/replan loop. That machinery is what "ours" uses; the point of
this script is to run the two ablations described in the paper text:

    --ablate-scene
        Skip `scene_description_full` (the side-level accessibility /
        reachability enrichment). The planner receives the plain
        `scene_description` object list instead, i.e. it "operates purely at
        the symbolic level".

    --ablate-validator
        Changes what recovery is allowed to see, not just the prompt. Two
        ways to run it, selected with --validator-mode:

        --validator-mode batch (default)
            Still calls the real batch PRE/POST validator (v6/v7, same
            prompt, same model, one call per stage), so per-condition
            accuracy is unaffected. The recovery decision only reads
            `overall_status` ("satisfied"/"violated") -- it never reads
            `results[]`, `reason`, or `progress_trend`.

        --validator-mode binary
            Actually uses the old single-condition prompts (v2-v5,
            "matching"/"non_matching" schema), one LLM call per condition
            instead of one batched call per stage. The per-condition calls
            are still made (so each individual condition is genuinely
            checked), but their results are ANDed into one overall_status
            and the per-condition breakdown is discarded before recovery
            sees it -- recovery only ever learns "satisfied" or "violated"
            for the whole stage, never which condition caused a violation.
            Note: v2-v5 have no "uncertain" state (only matching/
            non_matching), so this mode never produces PRE/POST "uncertain".

        Either way, this is the "reduced to a binary success/failure check
        that does not identify the violated condition" behaviour from the
        paper text.

Neither flag changes accuracy of the validator by itself (which is why the
module-level Validator success rate can legitimately be identical between
baseline and ours). What changes is what recovery is ALLOWED to do about a
failure:
    - both flags off:  approximates "ours" (minus evidence rounds/modify).
    - --ablate-validator on: recovery can only "repeat" (blind retry, same
      stage) or "replan" (regenerate the whole plan). It can never target a
      specific violated condition the way the full validator supports,
      because it is never told which condition failed.
    - --ablate-scene on: the planner itself has to guess side accessibility
      from the object list + image description alone, so its plans are more
      likely to violate a lateral-clearance precondition in the first place.

Simplifications versus the production script (by design, to keep this fast
and self-contained -- call these out if you report numbers from it):
    - No PRE/POST "uncertain" evidence-gathering rounds. "uncertain" is
      treated the same as "violated".
    - No modification-proposal / soft-guidance step. Recovery is only
      repeat-then-replan.
    - No attempt state machine / failure-report invariants. Outcomes are
      logged as plain dicts in the run summary JSON.
    - No goal-baseline (pre-execution "is the goal already met") check.

Usage mirrors the production script:

    PYTHONPATH=.:src python -m src.humanoids.run_validation_loop_baseline_ablation \\
        --scenario pick_glass --frames-dir image_data/pick_glass \\
        --scene-v v6 --plan-v v5 --sim-v v6 \\
        --validator-pre-v v6/precondition --validator-post-v v7/postcondition \\
        --scene-model gemini-robotics-er-1.6-preview \\
        --plan-model gemini-robotics-er-1.6-preview \\
        --sim-model gemini-robotics-er-1.6-preview \\
        --validator-model gemini-robotics-er-1.6-preview \\
        --poses-by-image-path image_data/pick_glass/poses_by_image.json \\
        --ablate-scene --ablate-validator \\
        --max-replans 3 --max-repeats 1 --interactive

--sim-v v5 (old scheduler schema)
    v5's `simultaneous_actions` prompt predates the current pipeline: it
    emits one 'Precondition'/'Postcondition' string per stage (not
    'Preconditions'/'Postconditions' lists) and no 'Local_goal'. Pass
    --sim-legacy-single-condition alongside --sim-v v5 so this script
    normalizes each stage into the modern shape before handing it to the
    (unmodified) production step functions -- Local_goal is synthesized from
    the stage's own actions, and the single condition string becomes a
    one-item Preconditions/Postconditions list, checked as one holistic
    condition per stage (matching how the old pipeline validated it):

    PYTHONPATH=.:src python -m src.humanoids.run_validation_loop_baseline_ablation \\
        --scenario two_two_tennis --frames-dir image_data/two_two_tennis \\
        --scene-v v6 --plan-v v5 --sim-v v5 --sim-legacy-single-condition \\
        --validator-pre-v v5 --validator-post-v v5 --validator-mode binary \\
        --scene-model gemini-robotics-er-1.6-preview \\
        --plan-model gemini-robotics-er-1.6-preview \\
        --sim-model gemini-robotics-er-1.6-preview \\
        --validator-model gemini-robotics-er-1.6-preview \\
        --poses-by-image-path image_data/two_two_tennis/poses_by_image.json \\
        --ablate-scene --ablate-validator \\
        --max-replans 3 --max-repeats 1 --interactive
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from src.settings import load_settings
from src.scenario_loader import load_scenario
from src.llm_client import call_llm_completion
from src.utils import (
    load_base_prompt,
    make_experiment_timestamp,
    make_cycle_name,
    make_stage_name,
    try_parse_json,
)

from src.humanoids.run_validation_loop_replan_with_feedback_soft_guidance import (
    SUPPORTED_MODELS,
    build_planned_stage_context,
    execute_postcondition_validator_step,
    execute_scene_description_full_step,
    execute_scene_description_step,
    execute_simultaneous_actions_step,
    execute_validator_step,
    execute_vlm_planning_step,
    extract_stage_actions,
    extract_stages,
    list_frame_paths,
    load_poses_by_image_map,
    load_scene_object_list_from_cycle,
    make_scenario_context,
    prompt_for_post_image,
    resolve_poses_by_image_path,
)


def print_output_block(title: str, payload: Any) -> None:
    """Print a step's parsed JSON output to the console (not just its saved path)."""
    print(f"[OUTPUT] {title}:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def synthesize_local_goal(stage_actions: list[dict[str, Any]], step_ids: list[int]) -> str:
    """Build a Local_goal string for schemas (e.g. v5) that don't emit one."""
    if not stage_actions:
        return f"Execute step(s) {step_ids}."
    parts = []
    for action in stage_actions:
        concept = action.get("Action concept", "").strip()
        target = action.get("Target", "").strip()
        parts.append(" ".join(p for p in (concept, target) if p))
    return "; ".join(p for p in parts if p) or f"Execute step(s) {step_ids}."


def extract_stages_legacy_single_condition(
    compact_parallel_plan: Any,
    sequential_plan: Any,
) -> list[dict[str, Any]]:
    """
    Normalize an old-schema (e.g. v5) simultaneous_actions output -- one
    'Precondition'/'Postcondition' string per stage, no 'Local_goal' -- into
    the modern stage shape (Local_goal + Preconditions/Postconditions lists)
    that the rest of this loop (and the imported production step functions)
    expect. Each single condition string becomes a one-item list, so the
    downstream binary validator checks it as one holistic condition per
    stage, matching how the old pipeline validated stage-level conditions.
    """
    if not isinstance(compact_parallel_plan, list):
        raise ValueError("simultaneous_actions output must be a JSON array of stages.")

    stages: list[dict[str, Any]] = []
    for idx, stage in enumerate(compact_parallel_plan):
        if not isinstance(stage, dict):
            raise ValueError(f"Stage at index {idx} is not a JSON object.")

        stage_id = stage.get("Stage_id")
        step_ids = stage.get("Step_id")
        precondition = stage.get("Precondition")
        postcondition = stage.get("Postcondition")

        if not isinstance(stage_id, int):
            raise ValueError(f"Stage at index {idx} has invalid or missing 'Stage_id'.")
        if not isinstance(step_ids, list) or not step_ids or not all(isinstance(v, int) for v in step_ids):
            raise ValueError(f"Stage {stage_id} has invalid or missing 'Step_id'.")
        if not isinstance(precondition, str) or not precondition.strip():
            raise ValueError(f"Stage {stage_id} has invalid or missing 'Precondition'.")

        stage_actions = extract_stage_actions(sequential_plan=sequential_plan, step_ids=step_ids)

        stages.append(
            {
                "Stage_id": stage_id,
                "Step_id": step_ids,
                "Local_goal": synthesize_local_goal(stage_actions, step_ids),
                "Preconditions": [precondition.strip()],
                "Postconditions": [postcondition.strip()] if isinstance(postcondition, str) and postcondition.strip() else [],
            }
        )

    return stages


def render_binary_condition_prompt(
    base_prompt: str,
    condition: str,
    scene_object_list: Any,
) -> str:
    """Render a v2-v5 style single-condition prompt (<CONDITION>/<SCENE_OBJECT_LIST>)."""
    prompt = base_prompt
    prompt = prompt.replace("<CONDITION>", condition)
    prompt = prompt.replace(
        "<SCENE_OBJECT_LIST>",
        json.dumps(scene_object_list, indent=2, ensure_ascii=False),
    )
    unresolved = sorted(set(re.findall(r"<[A-Z][A-Z0-9_]*>", prompt)))
    if unresolved:
        raise ValueError(
            "Unresolved binary validator prompt placeholders: "
            + ", ".join(unresolved)
        )
    return prompt.strip()


def execute_binary_condition_validator_step(
    settings,
    scenario_name: str,
    validator_version: str,
    validator_model: str,
    loop_timestamp: str,
    cycle_name: str,
    stage_id: int,
    condition_kind: str,
    conditions: list[str],
    image_path: str,
    scene_object_list: Any,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """
    Call the old single-condition binary validator prompt (v2-v5,
    matching/non_matching) once per condition, on a single image, then AND
    the results into one overall_status. per_condition_results is kept in
    the return value for provenance/debugging only -- callers that want the
    "binary, no diagnosis" ablation must read only overall_status.
    """
    base_prompt = load_base_prompt(settings, "validator", validator_version)
    stage_name = make_stage_name(stage_id)

    per_condition_results: list[dict[str, Any]] = []
    for idx, condition in enumerate(conditions, start=1):
        prompt = render_binary_condition_prompt(base_prompt, condition, scene_object_list)

        output_dir = (
            settings.project_root
            / "outputs"
            / "validator_binary"
            / scenario_name
            / validator_version
            / loop_timestamp
            / validator_model
            / cycle_name
            / stage_name
            / condition_kind
            / f"condition_{idx:02d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        result = call_llm_completion(
            settings=settings,
            model_name=validator_model,
            system_prompt=prompt,
            user_text="Verify the condition and return valid JSON only.",
            image_path=image_path,
            temperature=temperature,
            top_p=top_p,
        )
        parse_ok, parsed = try_parse_json(result["raw_response"])
        if not parse_ok or "result" not in parsed:
            raise ValueError(
                f"[validator_binary:{condition_kind}_{stage_id}] condition {idx} "
                f"could not be parsed as valid JSON.\n\nRaw:\n{result['raw_response']}"
            )

        per_condition_results.append(
            {
                "condition": condition,
                "result": parsed["result"],
                "reason": parsed.get("reason", ""),
            }
        )
        (output_dir / "response_parsed.json").write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[validator_binary:{condition_kind}_{stage_id}] condition {idx}/{len(conditions)}: "
            f"{parsed['result']} ({result['execution_time_seconds']:.2f}s) -- {condition}"
        )

    overall_status = (
        "satisfied"
        if all(r["result"] == "matching" for r in per_condition_results)
        else "violated"
    )

    return {
        "output": {
            "overall_status": overall_status,
            "per_condition_results": per_condition_results,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline-ablation validation loop: same step functions as the "
            "production pipeline, simplified repeat/replan recovery, with "
            "two independent ablation toggles (--ablate-scene, "
            "--ablate-validator)."
        )
    )

    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--frames-dir", type=str, required=True)
    parser.add_argument("--poses-by-image-path", type=str, default=None)

    parser.add_argument("--scene-v", type=str, required=True)
    parser.add_argument("--plan-v", type=str, required=True)
    parser.add_argument("--sim-v", type=str, required=True)
    parser.add_argument("--validator-pre-v", type=str, required=True)
    parser.add_argument("--validator-post-v", type=str, required=True)

    parser.add_argument("--scene-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--plan-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--sim-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--validator-model", type=str, required=True, choices=SUPPORTED_MODELS)

    parser.add_argument(
        "--ablate-scene",
        action="store_true",
        help="Skip scene_description_full; planner sees the plain symbolic object list only.",
    )
    parser.add_argument(
        "--ablate-validator",
        action="store_true",
        help=(
            "Recovery only looks at overall_status (satisfied/violated), "
            "never at per-condition results/reason/progress_trend."
        ),
    )
    parser.add_argument(
        "--validator-mode",
        type=str,
        choices=["batch", "binary"],
        default="batch",
        help=(
            "batch: use the real v6/v7 batched per-stage validator (default). "
            "binary: use the old v2-v5 single-condition matching/non_matching "
            "prompt, one LLM call per condition, ANDed into one overall_status "
            "-- pass the same version to --validator-pre-v/--validator-post-v, "
            "e.g. v5."
        ),
    )
    parser.add_argument(
        "--sim-legacy-single-condition",
        action="store_true",
        help=(
            "Pass this when --sim-v points to an old-schema simultaneous_actions "
            "prompt (e.g. v5) that emits one 'Precondition'/'Postcondition' "
            "string per stage instead of the current 'Preconditions'/"
            "'Postconditions' lists, and no 'Local_goal'. Each stage is "
            "normalized to the modern shape (Local_goal synthesized from the "
            "stage's actions, Precondition/Postcondition each wrapped as a "
            "single-item list) so the rest of the loop is unaffected."
        ),
    )

    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--max-replans", type=int, default=3)
    parser.add_argument("--max-repeats", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=30, help="Hard safety cap on cycles.")

    parser.add_argument("--grounding-safety-threshold", type=float, default=0.21)
    parser.add_argument("--grounding-debug-mapping", action="store_true")

    return parser


def build_pipeline_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ablation": {
            "ablate_scene": args.ablate_scene,
            "ablate_validator": args.ablate_validator,
        },
        "sampling": {"temperature": args.temperature, "top_p": args.top_p},
        "scene_description": {"prompt_version": args.scene_v, "model": args.scene_model},
        "vlm_planning": {"prompt_version": args.plan_v, "model": args.plan_model},
        "simultaneous_actions": {
            "prompt_version": args.sim_v,
            "model": args.sim_model,
            "legacy_single_condition": args.sim_legacy_single_condition,
        },
        "validator": {
            "pre_prompt_version": args.validator_pre_v,
            "post_prompt_version": args.validator_post_v,
            "model": args.validator_model,
        },
        "max_replans": args.max_replans,
        "max_repeats": args.max_repeats,
    }


def collapse_if_ablated(response: dict[str, Any], ablate_validator: bool) -> dict[str, Any]:
    """Return the view of a validator response that recovery is allowed to use."""
    if not ablate_validator:
        return response
    return {"overall_status": response.get("overall_status")}


def advance_to_post_image(
    frame_paths: list[str],
    frame_cursor: int,
    interactive: bool,
) -> tuple[str, int]:
    if interactive:
        chosen = prompt_for_post_image(frame_paths, frame_cursor)
    else:
        chosen = frame_cursor + 1
        if chosen >= len(frame_paths):
            raise RuntimeError(
                "No more frames available in --frames-dir to use as I_post."
            )
    return frame_paths[chosen], chosen


def main() -> None:
    args = build_parser().parse_args()

    settings = load_settings()
    scenario_data = load_scenario(settings, args.scenario)

    poses_by_image_path = resolve_poses_by_image_path(
        settings=settings,
        scenario_name=args.scenario,
        explicit_path=args.poses_by_image_path,
    )
    poses_by_image = load_poses_by_image_map(poses_by_image_path)

    frame_paths = list_frame_paths(args.frames_dir)
    if len(frame_paths) < 2:
        raise ValueError("--frames-dir must contain at least two images.")

    frame_cursor = 0
    current_image = frame_paths[0]
    loop_timestamp = make_experiment_timestamp()

    print("\n======================================================")
    print("BASELINE-ABLATION VALIDATION LOOP")
    print(f"Scenario:          {args.scenario}")
    print(f"ablate-scene:      {args.ablate_scene}")
    print(f"ablate-validator:  {args.ablate_validator}")
    print(f"Max replans:       {args.max_replans}")
    print(f"Max repeats/stage: {args.max_repeats}")
    print("======================================================")

    run_summary: dict[str, Any] = {
        "module": "baseline_ablation_run_summary",
        "scenario_name": args.scenario,
        "loop_timestamp": loop_timestamp,
        "timestamp": datetime.now().isoformat(),
        "config": build_pipeline_config(args),
        "cycles": [],
        "task_completed": False,
        "outcome": None,
        "replans_done": 0,
    }

    pending_stages: list[dict[str, Any]] | None = None
    pending_stage_repeats = 0
    pending_recovery_context: dict[str, Any] | None = None
    replans_done = 0
    task_completed = False
    cycle_idx = 0

    while not task_completed:
        cycle_idx += 1
        if cycle_idx > args.max_cycles:
            run_summary["outcome"] = "max_cycles_exceeded"
            print(f"\n[STOP] max-cycles ({args.max_cycles}) exceeded.")
            break

        cycle_name = make_cycle_name(cycle_idx)
        cycle_timestamp = make_experiment_timestamp()
        pipeline_config = build_pipeline_config(args)

        print("\n------------------------------------------------------")
        print(f"CYCLE {cycle_idx} | current_image={Path(current_image).name}")
        print("------------------------------------------------------")

        scenario_context = make_scenario_context(scenario_data, current_image)
        cycle_record: dict[str, Any] = {
            "cycle_index": cycle_idx,
            "start_image": Path(current_image).name,
            "reused_plan": pending_stages is not None,
            "stages": [],
            "outcome": None,
        }

        scene_description_artifact = execute_scene_description_step(
            settings=settings,
            scenario_name=args.scenario,
            scenario_context=scenario_context,
            version=args.scene_v,
            model_name=args.scene_model,
            loop_timestamp=loop_timestamp,
            cycle_name=cycle_name,
            cycle_idx=cycle_idx,
            cycle_timestamp=cycle_timestamp,
            pipeline_config=pipeline_config,
            image_path=current_image,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print_output_block("scene_description", scene_description_artifact["output"])

        if args.ablate_scene:
            scene_full_output = scene_description_artifact["output"]
            print("[ABLATION] scene_description_full skipped (--ablate-scene).")
        else:
            scene_description_full_artifact = execute_scene_description_full_step(
                settings=settings,
                scenario_name=args.scenario,
                scenario_context=scenario_context,
                version=args.scene_v,
                model_name=args.scene_model,
                loop_timestamp=loop_timestamp,
                cycle_name=cycle_name,
                cycle_idx=cycle_idx,
                cycle_timestamp=cycle_timestamp,
                scene_description=scene_description_artifact["output"],
                pipeline_config=pipeline_config,
                image_path=current_image,
                poses_by_image=poses_by_image,
                safety_threshold=args.grounding_safety_threshold,
                include_debug_mapping=args.grounding_debug_mapping,
            )
            scene_full_output = scene_description_full_artifact["output"]
            print_output_block("scene_description_full", scene_full_output)

        if pending_stages is not None:
            stages = pending_stages
            sequential_plan_output: list[Any] = []
            print(
                f"[RECOVERY] Reusing stored stage(s) for a repeat "
                f"({len(stages)} pending)."
            )
        else:
            sequential_plan_artifact = execute_vlm_planning_step(
                settings=settings,
                scenario_name=args.scenario,
                scenario_context=scenario_context,
                version=args.plan_v,
                model_name=args.plan_model,
                loop_timestamp=loop_timestamp,
                cycle_name=cycle_name,
                cycle_idx=cycle_idx,
                cycle_timestamp=cycle_timestamp,
                scene_description_full=scene_full_output,
                scene_version=args.scene_v,
                scene_model=args.scene_model,
                pipeline_config=pipeline_config,
                temperature=args.temperature,
                top_p=args.top_p,
                recovery_context=pending_recovery_context,
            )
            pending_recovery_context = None
            sequential_plan_output = sequential_plan_artifact["output"]
            print_output_block("vlm_planning (sequential_plan)", sequential_plan_output)

            simultaneous_actions_artifact = execute_simultaneous_actions_step(
                settings=settings,
                scenario_name=args.scenario,
                scenario_context=scenario_context,
                version=args.sim_v,
                model_name=args.sim_model,
                loop_timestamp=loop_timestamp,
                cycle_name=cycle_name,
                cycle_idx=cycle_idx,
                cycle_timestamp=cycle_timestamp,
                scene_description_full=scene_full_output,
                sequential_plan=sequential_plan_output,
                scene_version=args.scene_v,
                scene_model=args.scene_model,
                plan_version=args.plan_v,
                plan_model=args.plan_model,
                pipeline_config=pipeline_config,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print_output_block(
                "simultaneous_actions (raw model output)",
                simultaneous_actions_artifact["output"],
            )
            if args.sim_legacy_single_condition:
                stages = extract_stages_legacy_single_condition(
                    simultaneous_actions_artifact["output"],
                    sequential_plan_output,
                )
            else:
                stages = extract_stages(simultaneous_actions_artifact["output"])
            print_output_block("stages (normalized, with Preconditions/Postconditions)", stages)

            if not stages:
                if sequential_plan_output == [] and simultaneous_actions_artifact["output"] == []:
                    run_summary["outcome"] = "completed_no_actions_required"
                    task_completed = True
                    print("[LOOP] Empty plan: goal already satisfied. Task completed.")
                    run_summary["cycles"].append(cycle_record)
                    break
                raise ValueError("Scheduler returned no stages for a non-empty plan.")

        pending_stages = None
        stage_broke_loop = False

        for stage_position, stage in enumerate(stages, start=1):
            stage_id = stage["Stage_id"]
            stage_name = make_stage_name(stage_id)
            planned_stage_context = build_planned_stage_context(stage)
            preconditions = stage["Preconditions"]
            postconditions = stage["Postconditions"]
            stage_actions = extract_stage_actions(
                sequential_plan=sequential_plan_output,
                step_ids=stage["Step_id"],
            ) if sequential_plan_output else stage.get("_actions", [])

            stage_record: dict[str, Any] = {
                "stage_id": stage_id,
                "local_goal": stage["Local_goal"],
                "pre_image": Path(current_image).name,
                "repeat_index": pending_stage_repeats,
            }

            print(f"\n[STAGE {stage_id}] PRE validation on {Path(current_image).name}")
            if args.validator_mode == "binary":
                scene_object_list = load_scene_object_list_from_cycle(
                    settings=settings,
                    scenario_name=args.scenario,
                    scene_version=args.scene_v,
                    loop_timestamp=loop_timestamp,
                    scene_model=args.scene_model,
                    cycle_name=cycle_name,
                )
                pre_response = execute_binary_condition_validator_step(
                    settings=settings,
                    scenario_name=args.scenario,
                    validator_version=args.validator_pre_v,
                    validator_model=args.validator_model,
                    loop_timestamp=loop_timestamp,
                    cycle_name=cycle_name,
                    stage_id=stage_id,
                    condition_kind="pre",
                    conditions=preconditions,
                    image_path=current_image,
                    scene_object_list=scene_object_list,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )["output"]
            else:
                pre_artifact = execute_validator_step(
                    settings=settings,
                    scenario_name=args.scenario,
                    validator_version=args.validator_pre_v,
                    validator_model=args.validator_model,
                    loop_timestamp=loop_timestamp,
                    cycle_name=cycle_name,
                    cycle_idx=cycle_idx,
                    cycle_timestamp=cycle_timestamp,
                    stage_id=stage_id,
                    planned_stage_context=planned_stage_context,
                    preconditions=preconditions,
                    image_path=current_image,
                    scene_version=args.scene_v,
                    scene_model=args.scene_model,
                    plan_version=args.plan_v,
                    plan_model=args.plan_model,
                    sim_version=args.sim_v,
                    sim_model=args.sim_model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    condition_kind="pre",
                    validation_phase="pre",
                    image_role="I_pre",
                )
                pre_response = pre_artifact["output"]
                print_output_block(f"validator PRE (stage {stage_id}, batch)", pre_response)
            pre_view = collapse_if_ablated(pre_response, args.ablate_validator)
            stage_record["pre_status"] = pre_view["overall_status"]
            print(f"[STAGE {stage_id}] PRE overall status: {pre_view['overall_status']}")

            if pre_view["overall_status"] != "satisfied":
                # No modify path in this simplified loop: an unmet
                # precondition always forces a full replan from here.
                print(f"[STAGE {stage_id}] PRE not satisfied -> forcing full replan.")
                replans_done += 1
                stage_record["outcome"] = "pre_failed_replan"
                cycle_record["stages"].append(stage_record)
                stage_broke_loop = True
                pending_stage_repeats = 0
                if replans_done > args.max_replans:
                    run_summary["outcome"] = "failed_max_replans_exceeded"
                    task_completed = True
                break

            pre_image = current_image
            print(f"[STAGE {stage_id}] Executing (advancing frame)...")
            post_cursor_override = (
                prompt_for_post_image(frame_paths, frame_cursor)
                if args.interactive
                else None
            )
            try:
                if post_cursor_override is not None:
                    chosen = post_cursor_override
                    post_image, frame_cursor = frame_paths[chosen], chosen
                else:
                    post_image, frame_cursor = advance_to_post_image(
                        frame_paths, frame_cursor, interactive=False
                    )
            except RuntimeError as exc:
                stage_record["outcome"] = f"execution_stopped:{exc}"
                cycle_record["stages"].append(stage_record)
                run_summary["outcome"] = "failed_no_more_frames"
                task_completed = True
                stage_broke_loop = True
                break

            stage_record["post_image"] = Path(post_image).name
            print(f"[STAGE {stage_id}] I_post: {Path(post_image).name}")

            if args.validator_mode == "binary":
                scene_object_list = load_scene_object_list_from_cycle(
                    settings=settings,
                    scenario_name=args.scenario,
                    scene_version=args.scene_v,
                    loop_timestamp=loop_timestamp,
                    scene_model=args.scene_model,
                    cycle_name=cycle_name,
                )
                post_response = execute_binary_condition_validator_step(
                    settings=settings,
                    scenario_name=args.scenario,
                    validator_version=args.validator_post_v,
                    validator_model=args.validator_model,
                    loop_timestamp=loop_timestamp,
                    cycle_name=cycle_name,
                    stage_id=stage_id,
                    condition_kind="post",
                    conditions=postconditions,
                    image_path=post_image,
                    scene_object_list=scene_object_list,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )["output"]
            else:
                post_artifact = execute_postcondition_validator_step(
                    settings=settings,
                    scenario_name=args.scenario,
                    validator_version=args.validator_post_v,
                    validator_model=args.validator_model,
                    loop_timestamp=loop_timestamp,
                    cycle_name=cycle_name,
                    cycle_idx=cycle_idx,
                    cycle_timestamp=cycle_timestamp,
                    stage_id=stage_id,
                    planned_stage_context=planned_stage_context,
                    actions=stage_actions,
                    postconditions=postconditions,
                    pre_image_path=pre_image,
                    post_image_path=post_image,
                    scene_version=args.scene_v,
                    scene_model=args.scene_model,
                    plan_version=args.plan_v,
                    plan_model=args.plan_model,
                    sim_version=args.sim_v,
                    sim_model=args.sim_model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                post_response = post_artifact["output"]
                print_output_block(f"validator POST (stage {stage_id}, batch)", post_response)
            post_view = collapse_if_ablated(post_response, args.ablate_validator)
            stage_record["post_status"] = post_view["overall_status"]
            print(f"[STAGE {stage_id}] POST overall status: {post_view['overall_status']}")

            if post_view["overall_status"] == "satisfied":
                stage_record["outcome"] = "success"
                cycle_record["stages"].append(stage_record)
                current_image = post_image
                pending_stage_repeats = 0
                continue  # next stage in the same plan, same cycle

            # POST not satisfied (violated, or uncertain treated as violated).
            stage_record["outcome"] = "post_failed"
            cycle_record["stages"].append(stage_record)
            current_image = post_image  # world state is whatever I_post shows
            stage_broke_loop = True

            if pending_stage_repeats < args.max_repeats:
                pending_stage_repeats += 1
                pending_stages = deepcopy(stages[stage_position - 1:])
                print(
                    f"[RECOVERY] repeat ({pending_stage_repeats}/{args.max_repeats}) "
                    f"stage {stage_id} from {Path(current_image).name}."
                )
            else:
                replans_done += 1
                pending_stage_repeats = 0
                print(
                    f"[RECOVERY] repeat budget exhausted -> full replan "
                    f"({replans_done}/{args.max_replans})."
                )
                if replans_done > args.max_replans:
                    run_summary["outcome"] = "failed_max_replans_exceeded"
                    task_completed = True
                elif not args.ablate_validator:
                    if args.validator_mode == "binary":
                        failed_conditions = [
                            r["condition"]
                            for r in post_response.get("per_condition_results", [])
                            if r.get("result") == "non_matching"
                        ]
                    else:
                        failed_conditions = [
                            r["condition"]
                            for r in post_response.get("results", [])
                            if r.get("status") == "violated"
                        ]
                    pending_recovery_context = {
                        "local_goal": stage["Local_goal"],
                        "failed_actions": stage_actions,
                        "reason_for_replan": "Postcondition failed after repeat budget was exhausted.",
                        "failed_conditions": failed_conditions,
                        "uncertain_conditions": [],
                        "local_strategies_already_tried": {"repeat": pending_stage_repeats},
                        "total_attempts_on_stage": pending_stage_repeats + 1,
                    }
                # else: --ablate-validator means we deliberately do NOT know
                # which condition failed, so no recovery_context is built --
                # the replan gets only "try again from this image", which is
                # exactly the information-poverty the paper claims for the
                # binary-validator baseline.
            break

        run_summary["cycles"].append(cycle_record)

        if task_completed:
            break

        if not stage_broke_loop:
            # All stages in this plan succeeded; loop again to let the
            # planner decide if more actions are needed from the new image.
            continue

    run_summary["task_completed"] = task_completed and run_summary["outcome"] in (
        "completed_no_actions_required",
    )
    run_summary["replans_done"] = replans_done
    run_summary["final_image"] = Path(current_image).name

    output_dir = (
        settings.project_root
        / "outputs"
        / "validation_loop_baseline_ablation"
        / args.scenario
        / (
            f"scene{'OFF' if args.ablate_scene else 'ON'}_"
            f"validator{'OFF' if args.ablate_validator else 'ON'}"
        )
        / loop_timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n======================================================")
    print("RUN SUMMARY")
    print(f"Outcome:        {run_summary['outcome']}")
    print(f"Task completed: {run_summary['task_completed']}")
    print(f"Replans done:   {replans_done}")
    print(f"Saved to:       {summary_path}")
    print("======================================================")


if __name__ == "__main__":
    main()
