""" `run_validator.py` is the standalone script for running only the validator.
It takes one image, one condition, and a `scene_object_list`, then asks Azure OpenAI whether that condition matches the current scene. The condition can be something like a precondition or postcondition, for example `pre_1` or `post_1`.
The validator prompt is built from the validator base prompt, the condition text, and the `scene_object_list`. That `scene_object_list` can either be passed directly with `--scene-object-list-path`, or loaded automatically from a previous `scene_description` run using `--scene-version`, `--scene-model`, `--upstream-timestamp`, and `--run-name`.
It then calls Azure OpenAI with the selected image and expects a JSON response containing:
```json "result": "matching" ``` or ```json "result": "non_matching"``` plus a `reason`.
In short: `run_validator.py` is useful when you want to test a single validation check independently, without running the full validation loop or the full pipeline. It saves the rendered validator prompt, the parsed validator response, and metadata linking the check back to the upstream planning outputs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


from settings import load_settings
from azure_openai_client import call_azure_chat_completion
from utils import (
    load_base_prompt,
    read_json,
    try_parse_json,
    write_json,
)

SUPPORTED_MODELS = ["o3", "gpt-5.2"]


def parse_csv_arg(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a single validator call for a given image, condition, and scene_object_list."
    )

    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated models to use, e.g. o3,gpt-5.2",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for models that support it.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter for models that support it.",
    )

    # Upstream pipeline reference
    parser.add_argument(
        "--upstream-timestamp",
        type=str,
        required=True,
        help="Timestamp of the upstream pipeline run this validator refers to.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Upstream run name, e.g. run_001",
    )

    # Condition
    parser.add_argument(
        "--condition-name",
        type=str,
        required=True,
        help="Logical name of the condition, e.g. pre_1, post_1, pre_2",
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        help="Condition text to validate.",
    )

    # Image
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the current image to validate.",
    )

    # Scene object list source: either direct path or derived from scene_description run
    parser.add_argument("--scene-object-list-path", type=str, default=None)

    parser.add_argument("--scene-version", type=str, default=None)
    parser.add_argument("--scene-model", type=str, default=None, choices=SUPPORTED_MODELS)

    # Keep explicit dependencies to upstream planning blocks
    parser.add_argument("--plan-version", type=str, default=None)
    parser.add_argument("--plan-model", type=str, default=None, choices=SUPPORTED_MODELS)

    parser.add_argument("--sim-version", type=str, default=None)
    parser.add_argument("--sim-model", type=str, default=None, choices=SUPPORTED_MODELS)

    return parser


def validate_sampling_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.temperature <= 1.0:
        raise ValueError("--temperature must be between 0.0 and 1.0")

    if not 0.0 <= args.top_p <= 1.0:
        raise ValueError("--top-p must be between 0.0 and 1.0")

    if args.temperature != 0.0 and args.top_p != 1.0:
        raise ValueError(
            "Use either temperature or top_p for sampling control, not both at the same time."
        )


def validate_args(args: argparse.Namespace, models: list[str]) -> None:
    if not models:
        raise ValueError("No models provided. Use --models o3,gpt-5.2")

    invalid_models = [m for m in models if m not in SUPPORTED_MODELS]
    if invalid_models:
        raise ValueError(
            f"Invalid models: {invalid_models}. Allowed values: {SUPPORTED_MODELS}"
        )

    has_direct_object_list = args.scene_object_list_path is not None
    has_scene_ref = args.scene_version is not None and args.scene_model is not None

    if not has_direct_object_list and not has_scene_ref:
        raise ValueError(
            "You must provide either --scene-object-list-path "
            "or both --scene-version and --scene-model."
        )

    if has_direct_object_list and has_scene_ref:
        raise ValueError(
            "Provide either --scene-object-list-path OR "
            "(--scene-version and --scene-model), not both."
        )

    if args.plan_version is None or args.plan_model is None:
        raise ValueError(
            "--plan-version and --plan-model are required so validator metadata remains linked "
            "to vlm_planning."
        )

    if args.sim_version is None or args.sim_model is None:
        raise ValueError(
            "--sim-version and --sim-model are required so validator metadata remains linked "
            "to simultaneous_actions."
        )

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")


def load_scene_object_list(
    settings,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Returns:
    - scene_object_list data
    - metadata block describing the source
    """
    if args.scene_object_list_path is not None:
        path = Path(args.scene_object_list_path)
        if not path.exists():
            raise FileNotFoundError(f"scene_object_list file not found: {path}")

        data = read_json(path)
        source_info = {
            "type": "direct_path",
            "path": str(path.resolve()),
        }
        return data, source_info

    # Derived from scene_description run
    assert args.scene_version is not None
    assert args.scene_model is not None

    scene_dir = (
        settings.project_root
        / "outputs"
        / "scene_description"
        / args.scenario
        / args.scene_version
        / args.upstream_timestamp
        / args.scene_model
        / args.run_name
    )

    scene_object_list_path = scene_dir / "scene_object_list.json"
    if not scene_object_list_path.exists():
        raise FileNotFoundError(
            f"scene_object_list.json not found: {scene_object_list_path}"
        )

    data = read_json(scene_object_list_path)
    source_info = {
        "type": "scene_description_output",
        "scenario_name": args.scenario,
        "prompt_version": args.scene_version,
        "experiment_timestamp": args.upstream_timestamp,
        "model": args.scene_model,
        "run_name": args.run_name,
        "path": str(scene_object_list_path.resolve()),
    }
    return data, source_info


def render_validator_prompt(
    base_prompt: str,
    condition: str,
    scene_object_list: dict[str, Any],
) -> str:
    scene_object_list_str = json.dumps(scene_object_list, indent=2, ensure_ascii=False)

    prompt = base_prompt
    prompt = prompt.replace("<CONDITION>", condition)
    prompt = prompt.replace("<SCENE_OBJECT_LIST>", scene_object_list_str)

    return prompt.strip()



def validate_validator_response(parsed_response: Any) -> None:
    if not isinstance(parsed_response, dict):
        raise ValueError("Validator output must be a JSON object.")

    result = parsed_response.get("result")
    reason = parsed_response.get("reason")

    if result not in {"matching", "non_matching"}:
        raise ValueError(
            "Validator output field 'result' must be either 'matching' or 'non_matching'."
        )

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Validator output field 'reason' must be a non-empty string.")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_validator_prompt_dir(
    settings,
    scenario: str,
    version: str,
    upstream_timestamp: str,
    model_name: str,
    run_name: str,
    condition_name: str,
) -> Path:
    return (
        settings.project_root
        / "prompts_scenarios"
        / "validator"
        / scenario
        / version
        / upstream_timestamp
        / model_name
        / run_name
        / condition_name
    )


def get_validator_output_dir(
    settings,
    scenario: str,
    version: str,
    upstream_timestamp: str,
    model_name: str,
    run_name: str,
    condition_name: str,
) -> Path:
    return (
        settings.project_root
        / "outputs"
        / "validator"
        / scenario
        / version
        / upstream_timestamp
        / model_name
        / run_name
        / condition_name
    )


def save_validator_prompt(
    settings,
    scenario: str,
    version: str,
    upstream_timestamp: str,
    model_name: str,
    run_name: str,
    condition_name: str,
    prompt_text: str,
) -> Path:
    prompt_dir = get_validator_prompt_dir(
        settings=settings,
        scenario=scenario,
        version=version,
        upstream_timestamp=upstream_timestamp,
        model_name=model_name,
        run_name=run_name,
        condition_name=condition_name,
    )
    ensure_dir(prompt_dir)

    prompt_path = prompt_dir / "prompt.txt"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    return prompt_path


def save_validator_outputs(
    settings,
    scenario: str,
    version: str,
    upstream_timestamp: str,
    model_name: str,
    run_name: str,
    condition_name: str,
    deployment_name: str,
    execution_time_seconds: float,
    image_path: str,
    condition_text: str,
    scene_object_list_source: dict[str, Any],
    parsed_response: dict[str, Any],
    dependencies: dict[str, Any],
    sampling_config: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir = get_validator_output_dir(
        settings=settings,
        scenario=scenario,
        version=version,
        upstream_timestamp=upstream_timestamp,
        model_name=model_name,
        run_name=run_name,
        condition_name=condition_name,
    )
    ensure_dir(output_dir)

    parsed_path = output_dir / "response_parsed.json"
    run_info_path = output_dir / "run_info.json"

    write_json(parsed_path, parsed_response)

    run_info = {
        "module": "validator",
        "execution_mode": "single_module",
        "scenario_name": scenario,
        "prompt_version": version,
        "experiment_timestamp": upstream_timestamp,
        "run_name": run_name,
        "condition_name": condition_name,
        "condition_text": condition_text,
        "model": model_name,
        "deployment_name": deployment_name,
        "execution_time_seconds": execution_time_seconds,
        "timestamp": datetime.now().isoformat(),
        "image_path": str(Path(image_path).resolve()),
        "scene_object_list_source": scene_object_list_source,
        "dependencies": dependencies,
        "sampling_config": sampling_config,
        "response_parsed": parsed_response,
    }

    write_json(run_info_path, run_info)

    return parsed_path, run_info_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_sampling_args(args)

    models = parse_csv_arg(args.models)
    validate_args(args, models)

    settings = load_settings()
    base_prompt = load_base_prompt(settings, "validator", args.version)

    scene_object_list, scene_object_list_source = load_scene_object_list(
        settings=settings,
        args=args,
    )

    dependencies: dict[str, Any] = {
        "vlm_planning": {
            "prompt_version": args.plan_version,
            "experiment_timestamp": args.upstream_timestamp,
            "model": args.plan_model,
            "run_name": args.run_name,
        },
        "simultaneous_actions": {
            "prompt_version": args.sim_version,
            "experiment_timestamp": args.upstream_timestamp,
            "model": args.sim_model,
            "run_name": args.run_name,
        },
    }

    if args.scene_version is not None and args.scene_model is not None:
        dependencies["scene_description"] = {
            "prompt_version": args.scene_version,
            "experiment_timestamp": args.upstream_timestamp,
            "model": args.scene_model,
            "run_name": args.run_name,
        }

    sampling_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    total_runs = 0
    successful_runs = 0
    failed_runs = 0

    for model_name in models:
        total_runs += 1

        print("\n=== Running validator ===")
        print(f"Scenario:        {args.scenario}")
        print(f"Version:         {args.version}")
        print(f"Upstream ts:     {args.upstream_timestamp}")
        print(f"Model:           {model_name}")
        print(f"Run name:        {args.run_name}")
        print(f"Condition name:  {args.condition_name}")
        print(f"Temperature:     {args.temperature}")
        print(f"Top-p:           {args.top_p}")

        try:
            system_prompt = render_validator_prompt(
                base_prompt=base_prompt,
                condition=args.condition,
                scene_object_list=scene_object_list,
            )


            prompt_path = save_validator_prompt(
                settings=settings,
                scenario=args.scenario,
                version=args.version,
                upstream_timestamp=args.upstream_timestamp,
                model_name=model_name,
                run_name=args.run_name,
                condition_name=args.condition_name,
                prompt_text=system_prompt,
            )

            result = call_azure_chat_completion(
                settings=settings,
                model_name=model_name,
                system_prompt=system_prompt,
                user_text="Validate the condition and return valid JSON only.",
                image_path=args.image_path,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            parse_ok, parsed_response = try_parse_json(result["raw_response"])
            if not parse_ok:
                raise ValueError(
                    f"Model response could not be parsed as valid JSON.\n\nRaw response:\n{result['raw_response']}"
                )

            validate_validator_response(parsed_response)

            parsed_path, run_info_path = save_validator_outputs(
                settings=settings,
                scenario=args.scenario,
                version=args.version,
                upstream_timestamp=args.upstream_timestamp,
                model_name=result["model_name"],
                run_name=args.run_name,
                condition_name=args.condition_name,
                deployment_name=result["deployment_name"],
                execution_time_seconds=result["execution_time_seconds"],
                image_path=args.image_path,
                condition_text=args.condition,
                scene_object_list_source=scene_object_list_source,
                parsed_response=parsed_response,
                dependencies=dependencies,
                sampling_config=sampling_config,
            )

            successful_runs += 1

            print(f"[OK] Prompt saved to:        {prompt_path}")
            print(f"[OK] Parsed output saved to: {parsed_path}")
            print(f"[OK] Run info saved to:      {run_info_path}")
            print(f"[OK] Execution time:         {result['execution_time_seconds']:.3f}s")
            print("\nParsed JSON:")
            print(json.dumps(parsed_response, indent=2, ensure_ascii=False))

        except Exception as exc:
            failed_runs += 1
            print(
                f"[ERROR] validator | model={model_name} | run={args.run_name} | "
                f"condition={args.condition_name} -> {exc}"
            )

    print("\n==============================================")
    print("VALIDATOR RUN COMPLETED")
    print(f"Scenario:        {args.scenario}")
    print(f"Version:         {args.version}")
    print(f"Upstream ts:     {args.upstream_timestamp}")
    print(f"Run name:        {args.run_name}")
    print(f"Condition name:  {args.condition_name}")
    print(f"Models:          {', '.join(models)}")
    print(f"Temperature:     {args.temperature}")
    print(f"Top-p:           {args.top_p}")
    print(f"Total runs:      {total_runs}")
    print(f"Successful:      {successful_runs}")
    print(f"Failed:          {failed_runs}")
    print("==============================================")


if __name__ == "__main__":
    main()




# from __future__ import annotations

# import argparse
# import json
# from datetime import datetime
# from pathlib import Path
# from typing import Any

# from settings import load_settings
# from azure_openai_client import call_azure_chat_completion
# from utils import (
#     load_base_prompt,
#     load_previous_module_output,
#     read_json,
#     try_parse_json,
#     write_json,
# )

# SUPPORTED_MODELS = ["o3", "gpt-5.2"]


# def parse_csv_arg(value: str | None) -> list[str]:
#     if value is None or not value.strip():
#         return []
#     return [item.strip() for item in value.split(",") if item.strip()]


# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description="Run a single validator call for a given image, condition, and scene_object_list."
#     )

#     parser.add_argument("--scenario", type=str, required=True)
#     parser.add_argument("--version", type=str, required=True)
#     parser.add_argument(
#         "--models",
#         type=str,
#         required=True,
#         help="Comma-separated models to use, e.g. o3,gpt-5.2",
#     )

#     # Upstream pipeline reference
#     parser.add_argument(
#         "--upstream-timestamp",
#         type=str,
#         required=True,
#         help="Timestamp of the upstream pipeline run this validator refers to.",
#     )
#     parser.add_argument(
#         "--run-name",
#         type=str,
#         required=True,
#         help="Upstream run name, e.g. run_001",
#     )

#     # Condition
#     parser.add_argument(
#         "--condition-name",
#         type=str,
#         required=True,
#         help="Logical name of the condition, e.g. pre_1, post_1, pre_2",
#     )
#     parser.add_argument(
#         "--condition",
#         type=str,
#         required=True,
#         help="Condition text to validate.",
#     )

#     # Image
#     parser.add_argument(
#         "--image-path",
#         type=str,
#         required=True,
#         help="Path to the current image to validate.",
#     )

#     # Scene object list source: either direct path or derived from scene_description run
#     parser.add_argument("--scene-object-list-path", type=str, default=None)

#     parser.add_argument("--scene-version", type=str, default=None)
#     parser.add_argument("--scene-model", type=str, default=None, choices=SUPPORTED_MODELS)

#     # Keep explicit dependencies to upstream planning blocks
#     parser.add_argument("--plan-version", type=str, default=None)
#     parser.add_argument("--plan-model", type=str, default=None, choices=SUPPORTED_MODELS)

#     parser.add_argument("--sim-version", type=str, default=None)
#     parser.add_argument("--sim-model", type=str, default=None, choices=SUPPORTED_MODELS)

#     return parser


# def validate_args(args: argparse.Namespace, models: list[str]) -> None:
#     if not models:
#         raise ValueError("No models provided. Use --models o3,gpt-5.2")

#     invalid_models = [m for m in models if m not in SUPPORTED_MODELS]
#     if invalid_models:
#         raise ValueError(
#             f"Invalid models: {invalid_models}. Allowed values: {SUPPORTED_MODELS}"
#         )

#     has_direct_object_list = args.scene_object_list_path is not None
#     has_scene_ref = args.scene_version is not None and args.scene_model is not None

#     if not has_direct_object_list and not has_scene_ref:
#         raise ValueError(
#             "You must provide either --scene-object-list-path "
#             "or both --scene-version and --scene-model."
#         )

#     if has_direct_object_list and has_scene_ref:
#         raise ValueError(
#             "Provide either --scene-object-list-path OR "
#             "(--scene-version and --scene-model), not both."
#         )

#     if args.plan_version is None or args.plan_model is None:
#         raise ValueError(
#             "--plan-version and --plan-model are required so validator metadata remains linked "
#             "to vlm_planning."
#         )

#     if args.sim_version is None or args.sim_model is None:
#         raise ValueError(
#             "--sim-version and --sim-model are required so validator metadata remains linked "
#             "to simultaneous_actions."
#         )

#     image_path = Path(args.image_path)
#     if not image_path.exists():
#         raise FileNotFoundError(f"Image file not found: {image_path}")


# def load_scene_object_list(
#     settings,
#     args: argparse.Namespace,
# ) -> tuple[dict[str, Any], dict[str, Any]]:
#     """
#     Returns:
#     - scene_object_list data
#     - metadata block describing the source
#     """
#     if args.scene_object_list_path is not None:
#         path = Path(args.scene_object_list_path)
#         if not path.exists():
#             raise FileNotFoundError(f"scene_object_list file not found: {path}")

#         data = read_json(path)
#         source_info = {
#             "type": "direct_path",
#             "path": str(path.resolve()),
#         }
#         return data, source_info

#     # Derived from scene_description run
#     assert args.scene_version is not None
#     assert args.scene_model is not None

#     scene_dir = (
#         settings.project_root
#         / "outputs"
#         / "scene_description"
#         / args.scenario
#         / args.scene_version
#         / args.upstream_timestamp
#         / args.scene_model
#         / args.run_name
#     )

#     scene_object_list_path = scene_dir / "scene_object_list.json"
#     if not scene_object_list_path.exists():
#         raise FileNotFoundError(
#             f"scene_object_list.json not found: {scene_object_list_path}"
#         )

#     data = read_json(scene_object_list_path)
#     source_info = {
#         "type": "scene_description_output",
#         "scenario_name": args.scenario,
#         "prompt_version": args.scene_version,
#         "experiment_timestamp": args.upstream_timestamp,
#         "model": args.scene_model,
#         "run_name": args.run_name,
#         "path": str(scene_object_list_path.resolve()),
#     }
#     return data, source_info


# def render_validator_prompt(
#     base_prompt: str,
#     condition: str,
#     scene_object_list: dict[str, Any],
# ) -> str:
#     scene_object_list_str = json.dumps(scene_object_list, indent=2, ensure_ascii=False)

#     prompt = base_prompt
#     prompt = prompt.replace("<CONDITION>", condition)
#     prompt = prompt.replace("<SCENE_OBJECT_LIST>", scene_object_list_str)

#     return prompt.strip()


# def validate_validator_response(parsed_response: Any) -> None:
#     if not isinstance(parsed_response, dict):
#         raise ValueError("Validator output must be a JSON object.")

#     result = parsed_response.get("result")
#     reason = parsed_response.get("reason")

#     if result not in {"matching", "non_matching"}:
#         raise ValueError(
#             "Validator output field 'result' must be either 'matching' or 'non_matching'."
#         )

#     if not isinstance(reason, str) or not reason.strip():
#         raise ValueError("Validator output field 'reason' must be a non-empty string.")


# def ensure_dir(path: Path) -> Path:
#     path.mkdir(parents=True, exist_ok=True)
#     return path


# def get_validator_prompt_dir(
#     settings,
#     scenario: str,
#     version: str,
#     upstream_timestamp: str,
#     model_name: str,
#     run_name: str,
#     condition_name: str,
# ) -> Path:
#     return (
#         settings.project_root
#         / "prompts_scenarios"
#         / "validator"
#         / scenario
#         / version
#         / upstream_timestamp
#         / model_name
#         / run_name
#         / condition_name
#     )


# def get_validator_output_dir(
#     settings,
#     scenario: str,
#     version: str,
#     upstream_timestamp: str,
#     model_name: str,
#     run_name: str,
#     condition_name: str,
# ) -> Path:
#     return (
#         settings.project_root
#         / "outputs"
#         / "validator"
#         / scenario
#         / version
#         / upstream_timestamp
#         / model_name
#         / run_name
#         / condition_name
#     )


# def save_validator_prompt(
#     settings,
#     scenario: str,
#     version: str,
#     upstream_timestamp: str,
#     model_name: str,
#     run_name: str,
#     condition_name: str,
#     prompt_text: str,
# ) -> Path:
#     prompt_dir = get_validator_prompt_dir(
#         settings=settings,
#         scenario=scenario,
#         version=version,
#         upstream_timestamp=upstream_timestamp,
#         model_name=model_name,
#         run_name=run_name,
#         condition_name=condition_name,
#     )
#     ensure_dir(prompt_dir)

#     prompt_path = prompt_dir / "prompt.txt"
#     prompt_path.write_text(prompt_text, encoding="utf-8")
#     return prompt_path


# def save_validator_outputs(
#     settings,
#     scenario: str,
#     version: str,
#     upstream_timestamp: str,
#     model_name: str,
#     run_name: str,
#     condition_name: str,
#     deployment_name: str,
#     execution_time_seconds: float,
#     image_path: str,
#     condition_text: str,
#     scene_object_list_source: dict[str, Any],
#     parsed_response: dict[str, Any],
#     dependencies: dict[str, Any],
# ) -> tuple[Path, Path]:
#     output_dir = get_validator_output_dir(
#         settings=settings,
#         scenario=scenario,
#         version=version,
#         upstream_timestamp=upstream_timestamp,
#         model_name=model_name,
#         run_name=run_name,
#         condition_name=condition_name,
#     )
#     ensure_dir(output_dir)

#     parsed_path = output_dir / "response_parsed.json"
#     run_info_path = output_dir / "run_info.json"

#     write_json(parsed_path, parsed_response)

#     run_info = {
#         "module": "validator",
#         "execution_mode": "single_module",
#         "scenario_name": scenario,
#         "prompt_version": version,
#         "experiment_timestamp": upstream_timestamp,
#         "run_name": run_name,
#         "condition_name": condition_name,
#         "condition_text": condition_text,
#         "model": model_name,
#         "deployment_name": deployment_name,
#         "execution_time_seconds": execution_time_seconds,
#         "timestamp": datetime.now().isoformat(),
#         "image_path": str(Path(image_path).resolve()),
#         "scene_object_list_source": scene_object_list_source,
#         "dependencies": dependencies,
#         "response_parsed": parsed_response,
#     }

#     write_json(run_info_path, run_info)

#     return parsed_path, run_info_path


# def main() -> None:
#     parser = build_parser()
#     args = parser.parse_args()

#     models = parse_csv_arg(args.models)
#     validate_args(args, models)

#     settings = load_settings()
#     base_prompt = load_base_prompt(settings, "validator", args.version)

#     scene_object_list, scene_object_list_source = load_scene_object_list(
#         settings=settings,
#         args=args,
#     )

#     dependencies: dict[str, Any] = {
#         "vlm_planning": {
#             "prompt_version": args.plan_version,
#             "experiment_timestamp": args.upstream_timestamp,
#             "model": args.plan_model,
#             "run_name": args.run_name,
#         },
#         "simultaneous_actions": {
#             "prompt_version": args.sim_version,
#             "experiment_timestamp": args.upstream_timestamp,
#             "model": args.sim_model,
#             "run_name": args.run_name,
#         },
#     }

#     if args.scene_version is not None and args.scene_model is not None:
#         dependencies["scene_description"] = {
#             "prompt_version": args.scene_version,
#             "experiment_timestamp": args.upstream_timestamp,
#             "model": args.scene_model,
#             "run_name": args.run_name,
#         }

#     total_runs = 0
#     successful_runs = 0
#     failed_runs = 0

#     for model_name in models:
#         total_runs += 1

#         print("\n=== Running validator ===")
#         print(f"Scenario:        {args.scenario}")
#         print(f"Version:         {args.version}")
#         print(f"Upstream ts:     {args.upstream_timestamp}")
#         print(f"Model:           {model_name}")
#         print(f"Run name:        {args.run_name}")
#         print(f"Condition name:  {args.condition_name}")

#         try:
#             system_prompt = render_validator_prompt(
#                 base_prompt=base_prompt,
#                 condition=args.condition,
#                 scene_object_list=scene_object_list,
#             )

#             prompt_path = save_validator_prompt(
#                 settings=settings,
#                 scenario=args.scenario,
#                 version=args.version,
#                 upstream_timestamp=args.upstream_timestamp,
#                 model_name=model_name,
#                 run_name=args.run_name,
#                 condition_name=args.condition_name,
#                 prompt_text=system_prompt,
#             )

#             result = call_azure_chat_completion(
#                 settings=settings,
#                 model_name=model_name,
#                 system_prompt=system_prompt,
#                 user_text="Validate the condition and return valid JSON only.",
#                 image_path=args.image_path,
#             )

#             parse_ok, parsed_response = try_parse_json(result["raw_response"])
#             if not parse_ok:
#                 raise ValueError(
#                     f"Model response could not be parsed as valid JSON.\n\nRaw response:\n{result['raw_response']}"
#                 )

#             validate_validator_response(parsed_response)

#             parsed_path, run_info_path = save_validator_outputs(
#                 settings=settings,
#                 scenario=args.scenario,
#                 version=args.version,
#                 upstream_timestamp=args.upstream_timestamp,
#                 model_name=result["model_name"],
#                 run_name=args.run_name,
#                 condition_name=args.condition_name,
#                 deployment_name=result["deployment_name"],
#                 execution_time_seconds=result["execution_time_seconds"],
#                 image_path=args.image_path,
#                 condition_text=args.condition,
#                 scene_object_list_source=scene_object_list_source,
#                 parsed_response=parsed_response,
#                 dependencies=dependencies,
#             )

#             successful_runs += 1

#             print(f"[OK] Prompt saved to:        {prompt_path}")
#             print(f"[OK] Parsed output saved to: {parsed_path}")
#             print(f"[OK] Run info saved to:      {run_info_path}")
#             print(f"[OK] Execution time:         {result['execution_time_seconds']:.3f}s")
#             print("\nParsed JSON:")
#             print(json.dumps(parsed_response, indent=2, ensure_ascii=False))

#         except Exception as exc:
#             failed_runs += 1
#             print(
#                 f"[ERROR] validator | model={model_name} | run={args.run_name} | "
#                 f"condition={args.condition_name} -> {exc}"
#             )

#     print("\n==============================================")
#     print("VALIDATOR RUN COMPLETED")
#     print(f"Scenario:        {args.scenario}")
#     print(f"Version:         {args.version}")
#     print(f"Upstream ts:     {args.upstream_timestamp}")
#     print(f"Run name:        {args.run_name}")
#     print(f"Condition name:  {args.condition_name}")
#     print(f"Models:          {', '.join(models)}")
#     print(f"Total runs:      {total_runs}")
#     print(f"Successful:      {successful_runs}")
#     print(f"Failed:          {failed_runs}")
#     print("==============================================")


# if __name__ == "__main__":
#     main()