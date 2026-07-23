"""Standalone batch pre-condition validator.

The validator receives one image, the current planned-stage context, the complete
list of atomic preconditions, and scene_description_full.json. It performs one
VLM call and returns one result per precondition plus an overall_status.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.settings import load_settings
from src.azure_openai_client import call_azure_chat_completion
from src.utils import load_base_prompt, read_json, try_parse_json, write_json

SUPPORTED_MODELS = ["o3", "gpt-5.2"]
VALIDATOR_STATUSES = {"satisfied", "violated", "uncertain"}


def parse_csv_arg(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_step_ids(value: str) -> list[int]:
    items = parse_csv_arg(value)
    if not items:
        raise ValueError("--step-ids must contain at least one integer.")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise ValueError("--step-ids must be a comma-separated list of integers.") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one batch pre-condition validator call for a planned stage."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--models", required=True, help="Comma-separated models, e.g. o3,gpt-5.2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--upstream-timestamp", required=True)
    parser.add_argument("--run-name", required=True)

    parser.add_argument("--stage-id", type=int, required=True)
    parser.add_argument("--step-ids", required=True, help="Comma-separated step IDs, e.g. 1,3")
    parser.add_argument("--local-goal", required=True)
    parser.add_argument(
        "--precondition",
        action="append",
        dest="preconditions",
        required=True,
        help="Atomic precondition. Repeat this argument once per precondition.",
    )

    parser.add_argument("--image-path", required=True)

    parser.add_argument("--scene-description-full-path", default=None)
    parser.add_argument("--scene-version", default=None)
    parser.add_argument("--scene-model", choices=SUPPORTED_MODELS, default=None)

    parser.add_argument("--plan-version", required=True)
    parser.add_argument("--plan-model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--sim-version", required=True)
    parser.add_argument("--sim-model", required=True, choices=SUPPORTED_MODELS)
    return parser


def validate_sampling_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.temperature <= 1.0:
        raise ValueError("--temperature must be between 0.0 and 1.0")
    if not 0.0 <= args.top_p <= 1.0:
        raise ValueError("--top-p must be between 0.0 and 1.0")
    if args.temperature != 0.0 and args.top_p != 1.0:
        raise ValueError("Use either temperature or top_p, not both at the same time.")


def validate_args(args: argparse.Namespace, models: list[str]) -> None:
    if not models:
        raise ValueError("No models provided.")
    invalid_models = [model for model in models if model not in SUPPORTED_MODELS]
    if invalid_models:
        raise ValueError(f"Invalid models: {invalid_models}. Allowed: {SUPPORTED_MODELS}")

    has_direct_scene = args.scene_description_full_path is not None
    has_scene_reference = args.scene_version is not None and args.scene_model is not None
    if has_direct_scene == has_scene_reference:
        raise ValueError(
            "Provide either --scene-description-full-path or both --scene-version and "
            "--scene-model."
        )

    if args.stage_id < 1:
        raise ValueError("--stage-id must be >= 1")
    if not args.local_goal.strip():
        raise ValueError("--local-goal must be non-empty")
    if not all(isinstance(value, str) and value.strip() for value in args.preconditions):
        raise ValueError("Every --precondition must be non-empty")

    image_path = Path(args.image_path)
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")


def load_scene_description_full(
    settings,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.scene_description_full_path is not None:
        path = Path(args.scene_description_full_path)
        source_info = {"type": "direct_path", "path": str(path.resolve())}
    else:
        assert args.scene_version is not None
        assert args.scene_model is not None
        path = (
            settings.project_root
            / "outputs"
            / "scene_description"
            / args.scenario
            / args.scene_version
            / args.upstream_timestamp
            / args.scene_model
            / args.run_name
            / "scene_description_full.json"
        )
        source_info = {
            "type": "scene_description_full_output",
            "scenario_name": args.scenario,
            "prompt_version": args.scene_version,
            "experiment_timestamp": args.upstream_timestamp,
            "model": args.scene_model,
            "run_name": args.run_name,
            "path": str(path.resolve()),
        }

    if not path.exists():
        raise FileNotFoundError(f"scene_description_full.json not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("scene_description_full.json must contain a JSON object.")
    return data, source_info


def render_validator_prompt(
    base_prompt: str,
    planned_stage_context: dict[str, Any],
    preconditions: list[str],
    scene_description_full: dict[str, Any],
) -> str:
    prompt = base_prompt
    prompt = prompt.replace(
        "<PLANNED_STAGE_CONTEXT>",
        json.dumps(planned_stage_context, indent=2, ensure_ascii=False),
    )
    prompt = prompt.replace(
        "<PRECONDITIONS>",
        json.dumps(preconditions, indent=2, ensure_ascii=False),
    )
    prompt = prompt.replace(
        "<SCENE_DESCRIPTION_FULL>",
        json.dumps(scene_description_full, indent=2, ensure_ascii=False),
    )
    return prompt.strip()


def compute_overall_status(results: list[dict[str, Any]]) -> str:
    statuses = [item["status"] for item in results]
    if "violated" in statuses:
        return "violated"
    if "uncertain" in statuses:
        return "uncertain"
    return "satisfied"


def validate_validator_response(parsed_response: Any, expected_preconditions: list[str]) -> None:
    if not isinstance(parsed_response, dict):
        raise ValueError("Validator output must be a JSON object.")

    overall_status = parsed_response.get("overall_status")
    results = parsed_response.get("results")
    if overall_status not in VALIDATOR_STATUSES:
        raise ValueError("Invalid validator overall_status.")
    if not isinstance(results, list) or len(results) != len(expected_preconditions):
        raise ValueError("Validator results must match the number of input preconditions.")

    for index, (item, expected_condition) in enumerate(zip(results, expected_preconditions)):
        if not isinstance(item, dict):
            raise ValueError(f"Validator result {index} must be an object.")
        if item.get("condition") != expected_condition:
            raise ValueError(f"Validator result {index} changed condition text/order.")
        if item.get("status") not in VALIDATOR_STATUSES:
            raise ValueError(f"Validator result {index} has invalid status.")
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Validator result {index} has invalid reason.")

    computed = compute_overall_status(results)
    if overall_status != computed:
        raise ValueError(
            f"Inconsistent overall_status: expected {computed!r}, got {overall_status!r}."
        )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_validator_dir(
    settings,
    root_name: str,
    scenario: str,
    version: str,
    upstream_timestamp: str,
    model_name: str,
    run_name: str,
    stage_id: int,
) -> Path:
    return (
        settings.project_root
        / root_name
        / "validator"
        / scenario
        / version
        / upstream_timestamp
        / model_name
        / run_name
        / f"stage_{stage_id:03d}"
        / "pre"
    )


def save_outputs(
    settings,
    args: argparse.Namespace,
    model_name: str,
    deployment_name: str,
    execution_time_seconds: float,
    planned_stage_context: dict[str, Any],
    scene_source: dict[str, Any],
    parsed_response: dict[str, Any],
    prompt_text: str,
) -> tuple[Path, Path, Path]:
    prompt_dir = ensure_dir(
        get_validator_dir(
            settings, "prompts_scenarios", args.scenario, args.version,
            args.upstream_timestamp, model_name, args.run_name, args.stage_id,
        )
    )
    output_dir = ensure_dir(
        get_validator_dir(
            settings, "outputs", args.scenario, args.version,
            args.upstream_timestamp, model_name, args.run_name, args.stage_id,
        )
    )

    prompt_path = prompt_dir / "prompt.txt"
    parsed_path = output_dir / "response_parsed.json"
    run_info_path = output_dir / "run_info.json"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    write_json(parsed_path, parsed_response)
    write_json(
        run_info_path,
        {
            "module": "validator",
            "execution_mode": "single_module_precondition_batch",
            "scenario_name": args.scenario,
            "prompt_version": args.version,
            "experiment_timestamp": args.upstream_timestamp,
            "run_name": args.run_name,
            "stage_context": planned_stage_context,
            "preconditions": args.preconditions,
            "model": model_name,
            "deployment_name": deployment_name,
            "execution_time_seconds": execution_time_seconds,
            "timestamp": datetime.now().isoformat(),
            "image_path": str(Path(args.image_path).resolve()),
            "scene_description_full_source": scene_source,
            "dependencies": {
                "vlm_planning": {
                    "prompt_version": args.plan_version,
                    "model": args.plan_model,
                },
                "simultaneous_actions": {
                    "prompt_version": args.sim_version,
                    "model": args.sim_model,
                },
            },
            "sampling_config": {
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            "response_parsed": parsed_response,
        },
    )
    return prompt_path, parsed_path, run_info_path


def main() -> None:
    args = build_parser().parse_args()
    validate_sampling_args(args)
    models = parse_csv_arg(args.models)
    validate_args(args, models)

    settings = load_settings()
    base_prompt = load_base_prompt(settings, "validator", args.version)
    scene_description_full, scene_source = load_scene_description_full(settings, args)
    planned_stage_context = {
        "Stage_id": args.stage_id,
        "Step_id": parse_step_ids(args.step_ids),
        "Local_goal": args.local_goal,
    }

    successful_runs = 0
    failed_runs = 0
    for model_name in models:
        try:
            system_prompt = render_validator_prompt(
                base_prompt,
                planned_stage_context,
                args.preconditions,
                scene_description_full,
            )
            result = call_azure_chat_completion(
                settings=settings,
                model_name=model_name,
                system_prompt=system_prompt,
                user_text="Validate all stage preconditions and return valid JSON only.",
                image_path=args.image_path,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            parse_ok, parsed_response = try_parse_json(result["raw_response"])
            if not parse_ok:
                raise ValueError(f"Response is not valid JSON:\n{result['raw_response']}")

            validate_validator_response(parsed_response, args.preconditions)
            parsed_response["overall_status"] = compute_overall_status(parsed_response["results"])

            prompt_path, parsed_path, run_info_path = save_outputs(
                settings, args, result["model_name"], result["deployment_name"],
                result["execution_time_seconds"], planned_stage_context, scene_source,
                parsed_response, system_prompt,
            )
            successful_runs += 1
            print(f"[OK] Prompt saved to:        {prompt_path}")
            print(f"[OK] Parsed output saved to: {parsed_path}")
            print(f"[OK] Run info saved to:      {run_info_path}")
            print(json.dumps(parsed_response, indent=2, ensure_ascii=False))
        except Exception as exc:
            failed_runs += 1
            print(f"[ERROR] validator | model={model_name} | stage={args.stage_id} -> {exc}")

    print("\n==============================================")
    print("VALIDATOR RUN COMPLETED")
    print(f"Stage:       {args.stage_id}")
    print(f"Successful:  {successful_runs}")
    print(f"Failed:      {failed_runs}")
    print("==============================================")


if __name__ == "__main__":
    main()
