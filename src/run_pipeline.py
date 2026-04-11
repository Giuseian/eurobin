from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from settings import load_settings
from scenario_loader import load_scenario
from azure_openai_client import call_azure_chat_completion
from build_scene_object_list import build_scene_object_list_from_run
from scene_enrichment import enrich_scene
from utils import (
    load_base_prompt,
    make_experiment_timestamp,
    make_run_name,
    render_prompt,
    save_module_outputs,
    save_rendered_prompt,
    save_scene_description_full_artifact,
    try_parse_json,
)

SUPPORTED_MODELS = ["o3", "gpt-5.2"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full robotic VLM pipeline: "
            "scene_description (+ scene_description_full artifact) -> "
            "vlm_planning -> simultaneous_actions"
        )
    )

    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--scene-v", type=str, required=True)
    parser.add_argument("--plan-v", type=str, required=True)
    parser.add_argument("--sim-v", type=str, required=True)

    parser.add_argument("--scene-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--plan-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--sim-model", type=str, required=True, choices=SUPPORTED_MODELS)

    parser.add_argument("--repeats", type=int, default=1)

    parser.add_argument(
        "--grounding-safety-threshold",
        type=float,
        default=0.21,
        help="Safety threshold used by scene grounding to compute accessibility.",
    )
    parser.add_argument(
        "--grounding-debug-mapping",
        action="store_true",
        help="Store the internal VLM-to-Gazebo mapping in scene_description_full.json under _debug.",
    )

    return parser


def get_static_pose_file(settings, scenario_name: str) -> Path:
    pose_file = settings.project_root / "scenarios" / scenario_name / "poses.json"
    if not pose_file.exists():
        raise FileNotFoundError(f"Static pose file not found: {pose_file}")
    return pose_file


def execute_scene_description(
    settings,
    scenario_name: str,
    scenario_data: dict[str, Any],
    version: str,
    model_name: str,
    experiment_timestamp: str,
    run_name: str,
    pipeline_config: dict[str, Any],
) -> Any:
    module_name = "scene_description"
    base_prompt = load_base_prompt(settings, module_name, version)

    image_path = scenario_data.get("image_path_abs")
    if not image_path:
        raise ValueError("scene_description requires an image in scenario.json")

    system_prompt = base_prompt
    user_text = "Analyze the scene and return the structured JSON output."

    save_rendered_prompt(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
        prompt_text=system_prompt,
    )

    result = call_azure_chat_completion(
        settings=settings,
        model_name=model_name,
        system_prompt=system_prompt,
        user_text=user_text,
        image_path=image_path,
    )

    parse_ok, parsed_response = try_parse_json(result["raw_response"])
    if not parse_ok:
        raise ValueError(
            f"[scene_description] Model response could not be parsed as valid JSON.\n\n"
            f"Raw response:\n{result['raw_response']}"
        )

    parsed_path, run_info_path = save_module_outputs(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=result["model_name"],
        run_name=run_name,
        deployment_name=result["deployment_name"],
        execution_time_seconds=result["execution_time_seconds"],
        scenario_data=scenario_data,
        parsed_response=parsed_response,
        execution_mode="whole_pipeline",
        dependencies=None,
        pipeline_config=pipeline_config,
    )

    scene_object_list_path = build_scene_object_list_from_run(
        scenario=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model=result["model_name"],
        run_name=run_name,
    )

    print(f"[OK][scene_description] Parsed output saved to: {parsed_path}")
    print(f"[OK][scene_description] Run info saved to:      {run_info_path}")
    print(f"[OK][scene_description] Scene object list saved to: {scene_object_list_path}")
    print(f"[OK][scene_description] Execution time:         {result['execution_time_seconds']:.3f}s")

    return parsed_response


def execute_scene_description_full(
    settings,
    scenario_name: str,
    scenario_data: dict[str, Any],
    version: str,
    model_name: str,
    experiment_timestamp: str,
    run_name: str,
    scene_description: Any,
    pipeline_config: dict[str, Any],
    safety_threshold: float,
    include_debug_mapping: bool,
) -> Any:
    pose_file = get_static_pose_file(settings, scenario_name)

    start_time = time.perf_counter()

    parsed_response = enrich_scene(
        input_data=scene_description,
        safety_threshold=safety_threshold,
        pose_source="static",
        pose_file=str(pose_file.resolve()),
        include_debug_mapping=include_debug_mapping,
    )

    execution_time_seconds = time.perf_counter() - start_time

    dependencies = {
        "scene_description": {
            "prompt_version": version,
            "experiment_timestamp": experiment_timestamp,
            "model": model_name,
            "run_name": run_name,
        }
    }

    parsed_path, run_info_path = save_scene_description_full_artifact(
        settings=settings,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
        parsed_response=parsed_response,
        scenario_data=scenario_data,
        execution_time_seconds=execution_time_seconds,
        dependencies=dependencies,
        pipeline_config=pipeline_config,
        pose_file=str(pose_file.resolve()),
        safety_threshold=safety_threshold,
        include_debug_mapping=include_debug_mapping,
        execution_mode="whole_pipeline_side_artifact",
    )

    print(f"[OK][scene_description_full] Pose file used:       {pose_file}")
    print(f"[OK][scene_description_full] Parsed output saved to: {parsed_path}")
    print(f"[OK][scene_description_full] Run info saved to:      {run_info_path}")
    print(f"[OK][scene_description_full] Execution time:         {execution_time_seconds:.3f}s")

    return parsed_response


def execute_vlm_planning(
    settings,
    scenario_name: str,
    scenario_data: dict[str, Any],
    version: str,
    model_name: str,
    experiment_timestamp: str,
    run_name: str,
    scene_description_full: Any,
    scene_version: str,
    scene_model: str,
    pipeline_config: dict[str, Any],
) -> Any:
    module_name = "vlm_planning"
    base_prompt = load_base_prompt(settings, module_name, version)

    system_prompt = render_prompt(
        module_name=module_name,
        base_prompt=base_prompt,
        scenario_data=scenario_data,
        scene_description=scene_description_full,
    )

    user_text = "Generate the manipulation plan in valid JSON only."

    save_rendered_prompt(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
        prompt_text=system_prompt,
    )

    result = call_azure_chat_completion(
        settings=settings,
        model_name=model_name,
        system_prompt=system_prompt,
        user_text=user_text,
        image_path=None,
    )

    parse_ok, parsed_response = try_parse_json(result["raw_response"])
    if not parse_ok:
        raise ValueError(
            f"[vlm_planning] Model response could not be parsed as valid JSON.\n\n"
            f"Raw response:\n{result['raw_response']}"
        )

    dependencies = {
        "scene_description_full": {
            "stored_under_module": "scene_description",
            "artifact_filename": "scene_description_full.json",
            "prompt_version": scene_version,
            "experiment_timestamp": experiment_timestamp,
            "model": scene_model,
            "run_name": run_name,
        }
    }

    parsed_path, run_info_path = save_module_outputs(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=result["model_name"],
        run_name=run_name,
        deployment_name=result["deployment_name"],
        execution_time_seconds=result["execution_time_seconds"],
        scenario_data=scenario_data,
        parsed_response=parsed_response,
        execution_mode="whole_pipeline",
        dependencies=dependencies,
        pipeline_config=pipeline_config,
    )

    print(f"[OK][vlm_planning] Parsed output saved to: {parsed_path}")
    print(f"[OK][vlm_planning] Run info saved to:      {run_info_path}")
    print(f"[OK][vlm_planning] Execution time:         {result['execution_time_seconds']:.3f}s")

    return parsed_response


def execute_simultaneous_actions(
    settings,
    scenario_name: str,
    scenario_data: dict[str, Any],
    version: str,
    model_name: str,
    experiment_timestamp: str,
    run_name: str,
    scene_description_full: Any,
    sequential_plan: Any,
    scene_version: str,
    scene_model: str,
    plan_version: str,
    plan_model: str,
    pipeline_config: dict[str, Any],
) -> Any:
    module_name = "simultaneous_actions"
    base_prompt = load_base_prompt(settings, module_name, version)

    system_prompt = render_prompt(
        module_name=module_name,
        base_prompt=base_prompt,
        scenario_data=scenario_data,
        scene_description=scene_description_full,
        sequential_plan=sequential_plan,
    )

    user_text = "Generate the compact parallel plan in valid JSON only."

    save_rendered_prompt(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
        prompt_text=system_prompt,
    )

    result = call_azure_chat_completion(
        settings=settings,
        model_name=model_name,
        system_prompt=system_prompt,
        user_text=user_text,
        image_path=None,
    )

    parse_ok, parsed_response = try_parse_json(result["raw_response"])
    if not parse_ok:
        raise ValueError(
            f"[simultaneous_actions] Model response could not be parsed as valid JSON.\n\n"
            f"Raw response:\n{result['raw_response']}"
        )

    dependencies = {
        "scene_description_full": {
            "stored_under_module": "scene_description",
            "artifact_filename": "scene_description_full.json",
            "prompt_version": scene_version,
            "experiment_timestamp": experiment_timestamp,
            "model": scene_model,
            "run_name": run_name,
        },
        "vlm_planning": {
            "prompt_version": plan_version,
            "experiment_timestamp": experiment_timestamp,
            "model": plan_model,
            "run_name": run_name,
        },
    }

    parsed_path, run_info_path = save_module_outputs(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=result["model_name"],
        run_name=run_name,
        deployment_name=result["deployment_name"],
        execution_time_seconds=result["execution_time_seconds"],
        scenario_data=scenario_data,
        parsed_response=parsed_response,
        execution_mode="whole_pipeline",
        dependencies=dependencies,
        pipeline_config=pipeline_config,
    )

    print(f"[OK][simultaneous_actions] Parsed output saved to: {parsed_path}")
    print(f"[OK][simultaneous_actions] Run info saved to:      {run_info_path}")
    print(f"[OK][simultaneous_actions] Execution time:         {result['execution_time_seconds']:.3f}s")

    return parsed_response


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    settings = load_settings()
    scenario_data = load_scenario(settings, args.scenario)

    experiment_timestamp = make_experiment_timestamp()

    total_runs = 0
    successful_runs = 0
    failed_runs = 0

    for repeat_idx in range(1, args.repeats + 1):
        run_name = make_run_name(repeat_idx)
        total_runs += 1

        pipeline_config = {
            "scene_description": {
                "prompt_version": args.scene_v,
                "experiment_timestamp": experiment_timestamp,
                "model": args.scene_model,
                "run_name": run_name,
            },
            "scene_description_full": {
                "stored_under_module": "scene_description",
                "artifact_filename": "scene_description_full.json",
                "prompt_version": args.scene_v,
                "experiment_timestamp": experiment_timestamp,
                "model": args.scene_model,
                "run_name": run_name,
                "mode": "deterministic_scene_enrichment_static_pose",
            },
            "vlm_planning": {
                "prompt_version": args.plan_v,
                "experiment_timestamp": experiment_timestamp,
                "model": args.plan_model,
                "run_name": run_name,
            },
            "simultaneous_actions": {
                "prompt_version": args.sim_v,
                "experiment_timestamp": experiment_timestamp,
                "model": args.sim_model,
                "run_name": run_name,
            },
        }

        print("\n======================================================")
        print(
            f"PIPELINE RUN STARTED | scenario={args.scenario} | "
            f"timestamp={experiment_timestamp} | run_name={run_name}"
        )
        print("======================================================")

        try:
            scene_description = execute_scene_description(
                settings=settings,
                scenario_name=args.scenario,
                scenario_data=scenario_data,
                version=args.scene_v,
                model_name=args.scene_model,
                experiment_timestamp=experiment_timestamp,
                run_name=run_name,
                pipeline_config=pipeline_config,
            )

            print("\n[scene_description] Parsed JSON:")
            print(json.dumps(scene_description, indent=2, ensure_ascii=False))

            scene_description_full = execute_scene_description_full(
                settings=settings,
                scenario_name=args.scenario,
                scenario_data=scenario_data,
                version=args.scene_v,
                model_name=args.scene_model,
                experiment_timestamp=experiment_timestamp,
                run_name=run_name,
                scene_description=scene_description,
                pipeline_config=pipeline_config,
                safety_threshold=args.grounding_safety_threshold,
                include_debug_mapping=args.grounding_debug_mapping,
            )

            print("\n[scene_description_full] Parsed JSON:")
            print(json.dumps(scene_description_full, indent=2, ensure_ascii=False))

            sequential_plan = execute_vlm_planning(
                settings=settings,
                scenario_name=args.scenario,
                scenario_data=scenario_data,
                version=args.plan_v,
                model_name=args.plan_model,
                experiment_timestamp=experiment_timestamp,
                run_name=run_name,
                scene_description_full=scene_description_full,
                scene_version=args.scene_v,
                scene_model=args.scene_model,
                pipeline_config=pipeline_config,
            )

            print("\n[vlm_planning] Parsed JSON:")
            print(json.dumps(sequential_plan, indent=2, ensure_ascii=False))

            compact_parallel_plan = execute_simultaneous_actions(
                settings=settings,
                scenario_name=args.scenario,
                scenario_data=scenario_data,
                version=args.sim_v,
                model_name=args.sim_model,
                experiment_timestamp=experiment_timestamp,
                run_name=run_name,
                scene_description_full=scene_description_full,
                sequential_plan=sequential_plan,
                scene_version=args.scene_v,
                scene_model=args.scene_model,
                plan_version=args.plan_v,
                plan_model=args.plan_model,
                pipeline_config=pipeline_config,
            )

            print("\n[simultaneous_actions] Parsed JSON:")
            print(json.dumps(compact_parallel_plan, indent=2, ensure_ascii=False))

            successful_runs += 1

            print("\n======================================================")
            print(f"[OK] PIPELINE COMPLETED SUCCESSFULLY | run={run_name}")
            print("======================================================")

        except Exception as exc:
            failed_runs += 1
            print("\n======================================================")
            print(f"[ERROR] PIPELINE FAILED | run={run_name}")
            print(f"Reason: {exc}")
            print("======================================================")

    print("\n======================================================")
    print("FULL PIPELINE COMPLETED")
    print(f"Scenario:       {args.scenario}")
    print(f"Timestamp:      {experiment_timestamp}")
    print(f"Scene version:  {args.scene_v}")
    print(f"Plan version:   {args.plan_v}")
    print(f"Sim version:    {args.sim_v}")
    print(f"Scene model:    {args.scene_model}")
    print(f"Plan model:     {args.plan_model}")
    print(f"Sim model:      {args.sim_model}")
    print(f"Total runs:     {total_runs}")
    print(f"Successful:     {successful_runs}")
    print(f"Failed:         {failed_runs}")
    print("======================================================")


if __name__ == "__main__":
    main()
