""" This script allows running a single module in the robotic VLM pipeline with flexible configuration. 
It supports running the scene description, VLM planning, or simultaneous actions modules independently. """

from __future__ import annotations

import argparse
import json
from typing import Any

from settings import load_settings
from scenario_loader import load_scenario
from azure_openai_client import call_azure_chat_completion
from build_scene_object_list import build_scene_object_list_from_run
from utils import (
    load_base_prompt,
    load_previous_module_output,
    make_experiment_timestamp,
    make_run_name,
    render_prompt,
    save_module_outputs,
    save_rendered_prompt,
    try_parse_json,
    validate_module_name,
)


SUPPORTED_MODULES = {
    "scene_description",
    "vlm_planning",
    "simultaneous_actions",
}

SUPPORTED_MODELS = ["o3", "gpt-5.2"]


def parse_csv_arg(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single module for the robotic VLM pipeline."
    )

    parser.add_argument("--module", type=str, required=True, choices=sorted(SUPPORTED_MODULES))
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)

    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated models to use, e.g. o3,gpt-5.2",
    )

    parser.add_argument("--repeats", type=int, default=1)

    parser.add_argument("--scene-version", type=str, default=None)
    parser.add_argument("--scene-timestamp", type=str, default=None)
    parser.add_argument("--scene-model", type=str, default=None, choices=SUPPORTED_MODELS)
    parser.add_argument("--scene-run", type=str, default="run_001")

    parser.add_argument("--plan-version", type=str, default=None)
    parser.add_argument("--plan-timestamp", type=str, default=None)
    parser.add_argument("--plan-model", type=str, default=None, choices=SUPPORTED_MODELS)
    parser.add_argument("--plan-run", type=str, default="run_001")

    return parser


def resolve_experiment_timestamp(args: argparse.Namespace) -> str:
    if args.module == "scene_description":
        return make_experiment_timestamp()

    if args.module == "vlm_planning":
        if not args.scene_timestamp:
            raise ValueError("--scene-timestamp is required for vlm_planning")
        return args.scene_timestamp

    if args.module == "simultaneous_actions":
        if not args.scene_timestamp:
            raise ValueError("--scene-timestamp is required for simultaneous_actions")
        return args.scene_timestamp

    raise ValueError(f"Unsupported module for timestamp resolution: {args.module}")


def resolve_output_run_name(args: argparse.Namespace, repeat_idx: int) -> str:
    if args.module == "scene_description":
        return make_run_name(repeat_idx)

    if args.module == "vlm_planning":
        if not args.scene_run:
            raise ValueError("--scene-run is required for vlm_planning")
        return args.scene_run

    if args.module == "simultaneous_actions":
        if not args.scene_run:
            raise ValueError("--scene-run is required for simultaneous_actions")
        return args.scene_run

    raise ValueError(f"Unsupported module for run resolution: {args.module}")


def validate_downstream_run_alignment(args: argparse.Namespace) -> None:
    if args.module == "simultaneous_actions":
        if args.scene_run != args.plan_run:
            raise ValueError(
                "For simultaneous_actions, --scene-run and --plan-run must match "
                "to preserve 1:1 pipeline alignment."
            )

        if args.scene_timestamp != args.plan_timestamp:
            raise ValueError(
                "For simultaneous_actions, --scene-timestamp and --plan-timestamp must match "
                "to preserve a single root experiment timestamp."
            )


def run_scene_description(
    scenario_data: dict[str, Any],
    base_prompt: str,
) -> tuple[str, str | None, str]:
    image_path = scenario_data.get("image_path_abs")
    if not image_path:
        raise ValueError("scene_description requires an image in scenario.json")

    system_prompt = base_prompt
    user_text = "Analyze the scene and return the structured JSON output."
    return system_prompt, image_path, user_text


def run_vlm_planning(
    scenario_data: dict[str, Any],
    base_prompt: str,
    scene_description: Any,
) -> tuple[str, str | None, str]:
    system_prompt = render_prompt(
        module_name="vlm_planning",
        base_prompt=base_prompt,
        scenario_data=scenario_data,
        scene_description=scene_description,
    )
    user_text = "Generate the manipulation plan in valid JSON only."
    return system_prompt, None, user_text


def run_simultaneous_actions(
    scenario_data: dict[str, Any],
    base_prompt: str,
    scene_description: Any,
    sequential_plan: Any,
) -> tuple[str, str | None, str]:
    system_prompt = render_prompt(
        module_name="simultaneous_actions",
        base_prompt=base_prompt,
        scenario_data=scenario_data,
        scene_description=scene_description,
        sequential_plan=sequential_plan,
    )
    user_text = "Generate the compact parallel plan in valid JSON only."
    return system_prompt, None, user_text


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_module_name(args.module)

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    models = parse_csv_arg(args.models)
    if not models:
        raise ValueError("No models provided. Use --models o3,gpt-5.2")

    invalid_models = [m for m in models if m not in SUPPORTED_MODELS]
    if invalid_models:
        raise ValueError(
            f"Invalid models: {invalid_models}. Allowed values: {SUPPORTED_MODELS}"
        )

    if args.module != "scene_description" and args.repeats != 1:
        raise ValueError(
            "For downstream modules, --repeats must be 1 because output run is inherited "
            "from the upstream run."
        )

    validate_downstream_run_alignment(args)

    settings = load_settings()
    scenario_data = load_scenario(settings, args.scenario)
    base_prompt = load_base_prompt(settings, args.module, args.version)

    experiment_timestamp = resolve_experiment_timestamp(args)

    total_runs = 0
    successful_runs = 0
    failed_runs = 0

    for model_name in models:
        for repeat_idx in range(1, args.repeats + 1):
            run_name = resolve_output_run_name(args, repeat_idx)
            total_runs += 1

            print(f"\n=== Running {args.module} ===")
            print(f"Scenario:        {args.scenario}")
            print(f"Version:         {args.version}")
            print(f"Timestamp:       {experiment_timestamp}")
            print(f"Model:           {model_name}")
            print(f"Output run:      {run_name}")

            try:
                dependencies: dict[str, Any] | None = None

                if args.module == "scene_description":
                    system_prompt, image_path, user_text = run_scene_description(
                        scenario_data=scenario_data,
                        base_prompt=base_prompt,
                    )

                elif args.module == "vlm_planning":
                    if not args.scene_version:
                        raise ValueError("--scene-version is required for vlm_planning")
                    if not args.scene_timestamp:
                        raise ValueError("--scene-timestamp is required for vlm_planning")
                    if not args.scene_model:
                        raise ValueError("--scene-model is required for vlm_planning")

                    print(
                        "[INPUT] scene_description -> "
                        f"version={args.scene_version}, "
                        f"timestamp={args.scene_timestamp}, "
                        f"model={args.scene_model}, "
                        f"run={args.scene_run}"
                    )

                    scene_description = load_previous_module_output(
                        settings=settings,
                        module_name="scene_description",
                        scenario_name=args.scenario,
                        version=args.scene_version,
                        experiment_timestamp=args.scene_timestamp,
                        model_name=args.scene_model,
                        run_name=args.scene_run,
                    )

                    dependencies = {
                        "scene_description": {
                            "prompt_version": args.scene_version,
                            "experiment_timestamp": args.scene_timestamp,
                            "model": args.scene_model,
                            "run_name": args.scene_run,
                        }
                    }

                    system_prompt, image_path, user_text = run_vlm_planning(
                        scenario_data=scenario_data,
                        base_prompt=base_prompt,
                        scene_description=scene_description,
                    )

                elif args.module == "simultaneous_actions":
                    if not args.scene_version:
                        raise ValueError("--scene-version is required for simultaneous_actions")
                    if not args.scene_timestamp:
                        raise ValueError("--scene-timestamp is required for simultaneous_actions")
                    if not args.scene_model:
                        raise ValueError("--scene-model is required for simultaneous_actions")
                    if not args.plan_version:
                        raise ValueError("--plan-version is required for simultaneous_actions")
                    if not args.plan_timestamp:
                        raise ValueError("--plan-timestamp is required for simultaneous_actions")
                    if not args.plan_model:
                        raise ValueError("--plan-model is required for simultaneous_actions")

                    print(
                        "[INPUT] scene_description -> "
                        f"version={args.scene_version}, "
                        f"timestamp={args.scene_timestamp}, "
                        f"model={args.scene_model}, "
                        f"run={args.scene_run}"
                    )
                    print(
                        "[INPUT] vlm_planning -> "
                        f"version={args.plan_version}, "
                        f"timestamp={args.plan_timestamp}, "
                        f"model={args.plan_model}, "
                        f"run={args.plan_run}"
                    )

                    scene_description = load_previous_module_output(
                        settings=settings,
                        module_name="scene_description",
                        scenario_name=args.scenario,
                        version=args.scene_version,
                        experiment_timestamp=args.scene_timestamp,
                        model_name=args.scene_model,
                        run_name=args.scene_run,
                    )

                    sequential_plan = load_previous_module_output(
                        settings=settings,
                        module_name="vlm_planning",
                        scenario_name=args.scenario,
                        version=args.plan_version,
                        experiment_timestamp=args.plan_timestamp,
                        model_name=args.plan_model,
                        run_name=args.plan_run,
                    )

                    dependencies = {
                        "scene_description": {
                            "prompt_version": args.scene_version,
                            "experiment_timestamp": args.scene_timestamp,
                            "model": args.scene_model,
                            "run_name": args.scene_run,
                        },
                        "vlm_planning": {
                            "prompt_version": args.plan_version,
                            "experiment_timestamp": args.plan_timestamp,
                            "model": args.plan_model,
                            "run_name": args.plan_run,
                        },
                    }

                    system_prompt, image_path, user_text = run_simultaneous_actions(
                        scenario_data=scenario_data,
                        base_prompt=base_prompt,
                        scene_description=scene_description,
                        sequential_plan=sequential_plan,
                    )

                else:
                    raise ValueError(f"Unsupported module: {args.module}")

                save_rendered_prompt(
                    settings=settings,
                    module_name=args.module,
                    scenario_name=args.scenario,
                    version=args.version,
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

                raw_response = result["raw_response"]
                parse_ok, parsed_response = try_parse_json(raw_response)

                if not parse_ok:
                    raise ValueError(
                        f"Model response could not be parsed as valid JSON.\n\nRaw response:\n{raw_response}"
                    )

                parsed_path, run_info_path = save_module_outputs(
                    settings=settings,
                    module_name=args.module,
                    scenario_name=args.scenario,
                    version=args.version,
                    experiment_timestamp=experiment_timestamp,
                    model_name=result["model_name"],
                    run_name=run_name,
                    deployment_name=result["deployment_name"],
                    execution_time_seconds=result["execution_time_seconds"],
                    scenario_data=scenario_data,
                    parsed_response=parsed_response,
                    execution_mode="single_module",
                    dependencies=dependencies,
                    pipeline_config=None,
                )

                if args.module == "scene_description":
                    scene_object_list_path = build_scene_object_list_from_run(
                        scenario=args.scenario,
                        version=args.version,
                        experiment_timestamp=experiment_timestamp,
                        model=result["model_name"],
                        run_name=run_name,
                    )
                    print(f"[OK] Scene object list saved to: {scene_object_list_path}")

                successful_runs += 1

                print(f"[OK] Parsed output saved to: {parsed_path}")
                print(f"[OK] Run info saved to:      {run_info_path}")
                print(f"[OK] Execution time:         {result['execution_time_seconds']:.3f}s")
                print("\nParsed JSON:")
                print(json.dumps(parsed_response, indent=2, ensure_ascii=False))

            except Exception as exc:
                failed_runs += 1
                print(f"[ERROR] {args.module} | model={model_name} | {run_name} -> {exc}")

    print("\n==============================================")
    print("MODULE RUN COMPLETED")
    print(f"Module:        {args.module}")
    print(f"Scenario:      {args.scenario}")
    print(f"Version:       {args.version}")
    print(f"Timestamp:     {experiment_timestamp}")
    print(f"Models:        {', '.join(models)}")
    print(f"Total runs:    {total_runs}")
    print(f"Successful:    {successful_runs}")
    print(f"Failed:        {failed_runs}")
    print("==============================================")


if __name__ == "__main__":
    main()

