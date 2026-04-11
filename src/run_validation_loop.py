from __future__ import annotations

import argparse
import json
import tempfile
import time
from copy import deepcopy
from datetime import datetime
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
    render_prompt,
    save_module_outputs,
    save_rendered_prompt,
    save_scene_description_full_artifact,
    try_parse_json,
    write_json,
    read_json,
)

SUPPORTED_MODELS = ["o3", "gpt-5.2"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ============================================================
# PARSER
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline validation loop: pipeline -> stage pre/post validation -> "
            "replanning on failure."
        )
    )

    parser.add_argument("--scenario", type=str, required=True)

    parser.add_argument(
        "--initial-image-path",
        type=str,
        default=None,
        help="Optional explicit initial image path. If omitted, uses scenario.json image.",
    )

    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help=(
            "Directory containing the sequence of post-deploy images in chronological order. "
            "These images are consumed one-by-one when a stage is executed."
        ),
    )

    parser.add_argument(
        "--poses-by-image-path",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON mapping image filename -> pose dictionary. "
            "If omitted, defaults to scenarios/<scenario>/poses_by_image.json"
        ),
    )

    parser.add_argument("--scene-v", type=str, required=True)
    parser.add_argument("--plan-v", type=str, required=True)
    parser.add_argument("--sim-v", type=str, required=True)
    parser.add_argument("--validator-v", type=str, required=True)

    parser.add_argument("--scene-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--plan-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--sim-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--validator-model", type=str, required=True, choices=SUPPORTED_MODELS)

    parser.add_argument(
        "--max-replans",
        type=int,
        default=10,
        help="Maximum number of replanning cycles allowed before stopping.",
    )

    parser.add_argument(
        "--grounding-safety-threshold",
        type=float,
        default=0.21,
        help="Safety threshold used by scene enrichment to compute accessibility.",
    )
    parser.add_argument(
        "--grounding-debug-mapping",
        action="store_true",
        help="Store the internal VLM-to-Gazebo mapping inside scene_description_full.json under _debug.",
    )

    return parser


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_args(args: argparse.Namespace) -> None:
    if args.max_replans < 0:
        raise ValueError("--max-replans must be >= 0")

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames-dir not found: {frames_dir}")
    if not frames_dir.is_dir():
        raise ValueError(f"--frames-dir must be a directory: {frames_dir}")

    if args.poses_by_image_path is not None:
        poses_path = Path(args.poses_by_image_path)
        if not poses_path.exists():
            raise FileNotFoundError(f"poses-by-image-path not found: {poses_path}")


def list_frame_paths(frames_dir: str | Path) -> list[str]:
    frames_dir = Path(frames_dir)
    frames = sorted(
        [
            p
            for p in frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    if not frames:
        raise ValueError(f"No image files found inside frames-dir: {frames_dir}")

    return [str(p.resolve()) for p in frames]



def print_pose_dict_for_image(
    poses_by_image: dict[str, dict[str, list[float]]],
    image_path: str,
    label: str,
) -> None:
    image_name = Path(image_path).name

    if image_name not in poses_by_image:
        print(f"\n[DEBUG][{label}] No poses found for image: {image_name}")
        return

    pose_dict = poses_by_image[image_name]

    print(f"\n[DEBUG][{label}] Image path: {image_path}")
    print(f"[DEBUG][{label}] Image key:  {image_name}")
    print(f"[DEBUG][{label}] Pose entries:")

    for obj_name, pose in pose_dict.items():
        print(f"  - {obj_name}: {pose}")


def make_scenario_context(
    scenario_data: dict[str, Any],
    image_path: str,
) -> dict[str, Any]:
    """
    Copy scenario_data and override image fields so saved run_info reflects
    the actual image used for this cycle.
    """
    ctx = deepcopy(scenario_data)
    ctx["image"] = Path(image_path).name
    ctx["image_path_abs"] = str(Path(image_path).resolve())
    return ctx


def resolve_poses_by_image_path(
    settings,
    scenario_name: str,
    explicit_path: str | None,
) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).resolve()
    else:
        path = (
            settings.project_root
            / "scenarios"
            / scenario_name
            / "poses_by_image.json"
        ).resolve()

    if not path.exists():
        raise FileNotFoundError(f"poses_by_image.json not found: {path}")

    return path


def load_poses_by_image_map(path: str | Path) -> dict[str, dict[str, list[float]]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(
            f"poses_by_image mapping must be a JSON object. Found: {type(data).__name__}"
        )

    validated: dict[str, dict[str, list[float]]] = {}

    for image_name, pose_dict in data.items():
        if not isinstance(image_name, str):
            raise ValueError("Each poses_by_image key must be an image filename string.")

        if not isinstance(pose_dict, dict):
            raise ValueError(
                f"poses_by_image['{image_name}'] must be an object mapping object names to [x, y, z]."
            )

        cleaned_pose_dict: dict[str, list[float]] = {}
        for obj_name, pose in pose_dict.items():
            if not isinstance(obj_name, str):
                raise ValueError(
                    f"poses_by_image['{image_name}'] contains a non-string object name."
                )
            if not isinstance(pose, list) or len(pose) != 3:
                raise ValueError(
                    f"poses_by_image['{image_name}']['{obj_name}'] must be a list of 3 numeric values."
                )
            if not all(isinstance(v, (int, float)) for v in pose):
                raise ValueError(
                    f"poses_by_image['{image_name}']['{obj_name}'] must contain only numeric values."
                )
            cleaned_pose_dict[obj_name] = [float(v) for v in pose]

        validated[image_name] = cleaned_pose_dict

    return validated


def get_pose_dict_for_image(
    poses_by_image: dict[str, dict[str, list[float]]],
    image_path: str,
) -> dict[str, list[float]]:
    image_name = Path(image_path).name

    if image_name not in poses_by_image:
        available = ", ".join(sorted(poses_by_image.keys())[:10])
        raise KeyError(
            f"No pose entry found for image '{image_name}' in poses_by_image mapping. "
            f"Available examples: {available}"
        )

    return poses_by_image[image_name]


def write_temp_pose_file(pose_dict: dict[str, list[float]]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as tmp:
        json.dump(pose_dict, tmp, indent=2, ensure_ascii=False)
        return tmp.name


def load_scene_object_list_from_cycle(
    settings,
    scenario_name: str,
    scene_version: str,
    pipeline_timestamp: str,
    scene_model: str,
    run_name: str,
) -> dict[str, Any]:
    path = (
        settings.project_root
        / "outputs"
        / "scene_description"
        / scenario_name
        / scene_version
        / pipeline_timestamp
        / scene_model
        / run_name
        / "scene_object_list.json"
    )

    if not path.exists():
        raise FileNotFoundError(f"scene_object_list.json not found: {path}")

    return read_json(path)


def extract_stages(compact_parallel_plan: Any) -> list[dict[str, Any]]:
    if not isinstance(compact_parallel_plan, list):
        raise ValueError("simultaneous_actions output must be a JSON array of stages.")

    stages: list[dict[str, Any]] = []
    for idx, stage in enumerate(compact_parallel_plan):
        if not isinstance(stage, dict):
            raise ValueError(f"Stage at index {idx} is not a JSON object.")

        stage_id = stage.get("Stage_id")
        precondition = stage.get("Precondition")
        postcondition = stage.get("Postcondition")

        if not isinstance(stage_id, int):
            raise ValueError(f"Stage at index {idx} has invalid or missing 'Stage_id'.")
        if not isinstance(precondition, str) or not precondition.strip():
            raise ValueError(f"Stage {stage_id} has invalid or missing 'Precondition'.")
        if not isinstance(postcondition, str) or not postcondition.strip():
            raise ValueError(f"Stage {stage_id} has invalid or missing 'Postcondition'.")

        stages.append(
            {
                "Stage_id": stage_id,
                "Precondition": precondition,
                "Postcondition": postcondition,
            }
        )

    return stages


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
    parsed_response: dict[str, Any],
    dependencies: dict[str, Any],
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
        "execution_mode": "validation_loop",
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
        "dependencies": dependencies,
        "response_parsed": parsed_response,
    }

    write_json(run_info_path, run_info)

    return parsed_path, run_info_path


# ============================================================
# MODULE EXECUTION HELPERS (pipeline blocks)
# ============================================================

def execute_scene_description_step(
    settings,
    scenario_name: str,
    scenario_context: dict[str, Any],
    version: str,
    model_name: str,
    experiment_timestamp: str,
    run_name: str,
    pipeline_config: dict[str, Any],
    image_path: str,
) -> dict[str, Any]:
    module_name = "scene_description"
    base_prompt = load_base_prompt(settings, module_name, version)

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
        scenario_data=scenario_context,
        parsed_response=parsed_response,
        execution_mode="validation_loop",
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


def execute_scene_description_full_step(
    settings,
    scenario_name: str,
    scenario_context: dict[str, Any],
    version: str,
    model_name: str,
    experiment_timestamp: str,
    run_name: str,
    scene_description: Any,
    pipeline_config: dict[str, Any],
    image_path: str,
    poses_by_image: dict[str, dict[str, list[float]]],
    safety_threshold: float,
    include_debug_mapping: bool,
) -> dict[str, Any]:
    pose_dict = get_pose_dict_for_image(poses_by_image, image_path)
    temp_pose_file = write_temp_pose_file(pose_dict)

    try:
        start_time = time.perf_counter()

        parsed_response = enrich_scene(
            input_data=scene_description,
            safety_threshold=safety_threshold,
            pose_source="static",
            pose_file=temp_pose_file,
            include_debug_mapping=include_debug_mapping,
        )

        execution_time_seconds = time.perf_counter() - start_time

    finally:
        temp_path = Path(temp_pose_file)
        if temp_path.exists():
            temp_path.unlink()

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
        scenario_data=scenario_context,
        execution_time_seconds=execution_time_seconds,
        dependencies=dependencies,
        pipeline_config=pipeline_config,
        pose_file=temp_pose_file,
        safety_threshold=safety_threshold,
        include_debug_mapping=include_debug_mapping,
        execution_mode="validation_loop_side_artifact",
    )

    print(f"[OK][scene_description_full] Image key used:       {Path(image_path).name}")
    print(f"[OK][scene_description_full] Parsed output saved to: {parsed_path}")
    print(f"[OK][scene_description_full] Run info saved to:      {run_info_path}")
    print(f"[OK][scene_description_full] Execution time:         {execution_time_seconds:.3f}s")

    return parsed_response


def execute_vlm_planning_step(
    settings,
    scenario_name: str,
    scenario_context: dict[str, Any],
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
        scenario_data=scenario_context,
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
        scenario_data=scenario_context,
        parsed_response=parsed_response,
        execution_mode="validation_loop",
        dependencies=dependencies,
        pipeline_config=pipeline_config,
    )

    print(f"[OK][vlm_planning] Parsed output saved to: {parsed_path}")
    print(f"[OK][vlm_planning] Run info saved to:      {run_info_path}")
    print(f"[OK][vlm_planning] Execution time:         {result['execution_time_seconds']:.3f}s")

    return parsed_response


def execute_simultaneous_actions_step(
    settings,
    scenario_name: str,
    scenario_context: dict[str, Any],
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
        scenario_data=scenario_context,
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
        scenario_data=scenario_context,
        parsed_response=parsed_response,
        execution_mode="validation_loop",
        dependencies=dependencies,
        pipeline_config=pipeline_config,
    )

    print(f"[OK][simultaneous_actions] Parsed output saved to: {parsed_path}")
    print(f"[OK][simultaneous_actions] Run info saved to:      {run_info_path}")
    print(f"[OK][simultaneous_actions] Execution time:         {result['execution_time_seconds']:.3f}s")

    return parsed_response


def execute_validator_step(
    settings,
    scenario_name: str,
    validator_version: str,
    validator_model: str,
    upstream_timestamp: str,
    run_name: str,
    condition_name: str,
    condition_text: str,
    image_path: str,
    scene_version: str,
    scene_model: str,
    plan_version: str,
    plan_model: str,
    sim_version: str,
    sim_model: str,
) -> dict[str, Any]:
    scene_object_list = load_scene_object_list_from_cycle(
        settings=settings,
        scenario_name=scenario_name,
        scene_version=scene_version,
        pipeline_timestamp=upstream_timestamp,
        scene_model=scene_model,
        run_name=run_name,
    )

    base_prompt = load_base_prompt(settings, "validator", validator_version)
    system_prompt = render_validator_prompt(
        base_prompt=base_prompt,
        condition=condition_text,
        scene_object_list=scene_object_list,
    )

    prompt_path = save_validator_prompt(
        settings=settings,
        scenario=scenario_name,
        version=validator_version,
        upstream_timestamp=upstream_timestamp,
        model_name=validator_model,
        run_name=run_name,
        condition_name=condition_name,
        prompt_text=system_prompt,
    )

    result = call_azure_chat_completion(
        settings=settings,
        model_name=validator_model,
        system_prompt=system_prompt,
        user_text="Validate the condition and return valid JSON only.",
        image_path=image_path,
    )

    parse_ok, parsed_response = try_parse_json(result["raw_response"])
    if not parse_ok:
        raise ValueError(
            f"[validator:{condition_name}] Model response could not be parsed as valid JSON.\n\n"
            f"Raw response:\n{result['raw_response']}"
        )

    validate_validator_response(parsed_response)

    dependencies = {
        "scene_description_full": {
            "stored_under_module": "scene_description",
            "artifact_filename": "scene_description_full.json",
            "prompt_version": scene_version,
            "experiment_timestamp": upstream_timestamp,
            "model": scene_model,
            "run_name": run_name,
        },
        "vlm_planning": {
            "prompt_version": plan_version,
            "experiment_timestamp": upstream_timestamp,
            "model": plan_model,
            "run_name": run_name,
        },
        "simultaneous_actions": {
            "prompt_version": sim_version,
            "experiment_timestamp": upstream_timestamp,
            "model": sim_model,
            "run_name": run_name,
        },
    }

    parsed_path, run_info_path = save_validator_outputs(
        settings=settings,
        scenario=scenario_name,
        version=validator_version,
        upstream_timestamp=upstream_timestamp,
        model_name=result["model_name"],
        run_name=run_name,
        condition_name=condition_name,
        deployment_name=result["deployment_name"],
        execution_time_seconds=result["execution_time_seconds"],
        image_path=image_path,
        condition_text=condition_text,
        parsed_response=parsed_response,
        dependencies=dependencies,
    )

    print(f"[OK][validator:{condition_name}] Prompt saved to:        {prompt_path}")
    print(f"[OK][validator:{condition_name}] Parsed output saved to: {parsed_path}")
    print(f"[OK][validator:{condition_name}] Run info saved to:      {run_info_path}")
    print(f"[OK][validator:{condition_name}] Execution time:         {result['execution_time_seconds']:.3f}s")

    return parsed_response


# ============================================================
# SUMMARY HELPERS
# ============================================================

def save_loop_summary(
    settings,
    scenario_name: str,
    loop_timestamp: str,
    summary: dict[str, Any],
) -> Path:
    output_dir = (
        settings.project_root
        / "outputs"
        / "validation_loop"
        / scenario_name
        / loop_timestamp
    )
    ensure_dir(output_dir)

    out_path = output_dir / "loop_summary.json"
    write_json(out_path, summary)
    return out_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_args(args)

    settings = load_settings()
    scenario_data = load_scenario(settings, args.scenario)

    poses_by_image_path = resolve_poses_by_image_path(
        settings=settings,
        scenario_name=args.scenario,
        explicit_path=args.poses_by_image_path,
    )
    poses_by_image = load_poses_by_image_map(poses_by_image_path)

    initial_image_path = (
        str(Path(args.initial_image_path).resolve())
        if args.initial_image_path is not None
        else scenario_data.get("image_path_abs")
    )
    if not initial_image_path:
        raise ValueError(
            "No initial image available. Provide --initial-image-path or set 'image' in scenario.json."
        )
    if not Path(initial_image_path).exists():
        raise FileNotFoundError(f"Initial image not found: {initial_image_path}")

    frame_paths = list_frame_paths(args.frames_dir)
    initial_image_resolved = str(Path(initial_image_path).resolve())
    frame_paths = [p for p in frame_paths if str(Path(p).resolve()) != initial_image_resolved]
    frame_cursor = 0

    loop_timestamp = make_experiment_timestamp()
    run_name = "run_001"

    current_image = initial_image_path
    replans_done = 0
    task_completed = False
    cycle_idx = 0

    summary: dict[str, Any] = {
        "module": "validation_loop",
        "scenario_name": args.scenario,
        "loop_timestamp": loop_timestamp,
        "timestamp": datetime.now().isoformat(),
        "initial_image_path": str(Path(initial_image_path).resolve()),
        "frames_dir": str(Path(args.frames_dir).resolve()),
        "poses_by_image_path": str(poses_by_image_path),
        "config": {
            "scene_description": {
                "prompt_version": args.scene_v,
                "model": args.scene_model,
            },
            "scene_description_full": {
                "stored_under_module": "scene_description",
                "artifact_filename": "scene_description_full.json",
                "prompt_version": args.scene_v,
                "model": args.scene_model,
                "mode": "deterministic_scene_enrichment_per_image",
            },
            "vlm_planning": {
                "prompt_version": args.plan_v,
                "model": args.plan_model,
            },
            "simultaneous_actions": {
                "prompt_version": args.sim_v,
                "model": args.sim_model,
            },
            "validator": {
                "prompt_version": args.validator_v,
                "model": args.validator_model,
            },
            "max_replans": args.max_replans,
            "grounding_safety_threshold": args.grounding_safety_threshold,
            "grounding_debug_mapping": args.grounding_debug_mapping,
        },
        "replans_done": 0,
        "task_completed": False,
        "cycles": [],
    }

    while not task_completed:
        cycle_idx += 1
        pipeline_timestamp = make_experiment_timestamp()

        print("\n======================================================")
        print(f"VALIDATION LOOP CYCLE STARTED | cycle={cycle_idx}")
        print(f"Current image:   {current_image}")
        print(f"Pipeline ts:     {pipeline_timestamp}")
        print("======================================================")

        scenario_context = make_scenario_context(
            scenario_data=scenario_data,
            image_path=current_image,
        )

        pipeline_config = {
            "scene_description": {
                "prompt_version": args.scene_v,
                "experiment_timestamp": pipeline_timestamp,
                "model": args.scene_model,
                "run_name": run_name,
            },
            "scene_description_full": {
                "stored_under_module": "scene_description",
                "artifact_filename": "scene_description_full.json",
                "prompt_version": args.scene_v,
                "experiment_timestamp": pipeline_timestamp,
                "model": args.scene_model,
                "run_name": run_name,
                "mode": "deterministic_scene_enrichment_per_image",
            },
            "vlm_planning": {
                "prompt_version": args.plan_v,
                "experiment_timestamp": pipeline_timestamp,
                "model": args.plan_model,
                "run_name": run_name,
            },
            "simultaneous_actions": {
                "prompt_version": args.sim_v,
                "experiment_timestamp": pipeline_timestamp,
                "model": args.sim_model,
                "run_name": run_name,
            },
        }

        cycle_record: dict[str, Any] = {
            "cycle_idx": cycle_idx,
            "pipeline_timestamp": pipeline_timestamp,
            "run_name": run_name,
            "start_image_path": str(Path(current_image).resolve()),
            "start_image_name": Path(current_image).name,
            "stages": [],
            "outcome": None,
        }

        cycle_error = False

        try:
            scene_description = execute_scene_description_step(
                settings=settings,
                scenario_name=args.scenario,
                scenario_context=scenario_context,
                version=args.scene_v,
                model_name=args.scene_model,
                experiment_timestamp=pipeline_timestamp,
                run_name=run_name,
                pipeline_config=pipeline_config,
                image_path=current_image,
            )
            print("\n[scene_description] Parsed JSON:")
            print(json.dumps(scene_description, indent=2, ensure_ascii=False))

            scene_description_full = execute_scene_description_full_step(
                settings=settings,
                scenario_name=args.scenario,
                scenario_context=scenario_context,
                version=args.scene_v,
                model_name=args.scene_model,
                experiment_timestamp=pipeline_timestamp,
                run_name=run_name,
                scene_description=scene_description,
                pipeline_config=pipeline_config,
                image_path=current_image,
                poses_by_image=poses_by_image,
                safety_threshold=args.grounding_safety_threshold,
                include_debug_mapping=args.grounding_debug_mapping,
            )
            print("\n[scene_description_full] Parsed JSON:")
            print(json.dumps(scene_description_full, indent=2, ensure_ascii=False))

            sequential_plan = execute_vlm_planning_step(
                settings=settings,
                scenario_name=args.scenario,
                scenario_context=scenario_context,
                version=args.plan_v,
                model_name=args.plan_model,
                experiment_timestamp=pipeline_timestamp,
                run_name=run_name,
                scene_description_full=scene_description_full,
                scene_version=args.scene_v,
                scene_model=args.scene_model,
                pipeline_config=pipeline_config,
            )
            print("\n[vlm_planning] Parsed JSON:")
            print(json.dumps(sequential_plan, indent=2, ensure_ascii=False))

            compact_parallel_plan = execute_simultaneous_actions_step(
                settings=settings,
                scenario_name=args.scenario,
                scenario_context=scenario_context,
                version=args.sim_v,
                model_name=args.sim_model,
                experiment_timestamp=pipeline_timestamp,
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

            stages = extract_stages(compact_parallel_plan)

            all_stages_succeeded = True

            for stage in stages:
                stage_id = stage["Stage_id"]
                pre_condition = stage["Precondition"]
                post_condition = stage["Postcondition"]

                stage_record: dict[str, Any] = {
                    "stage_id": stage_id,
                    "precondition": pre_condition,
                    "postcondition": post_condition,
                    "pre_image_path": str(Path(current_image).resolve()),
                    "pre_image_name": Path(current_image).name,
                    "post_image_path": None,
                    "post_image_name": None,
                    "pre_validation": None,
                    "post_validation": None,
                    "next_image_path": None,
                    "next_image_name": None,
                }

                # --------------------------------------------------
                # PRE VALIDATION
                # --------------------------------------------------
                print(f"\n[LOOP] Stage {stage_id} PRE")
                print(f"[LOOP] PRE image:      {current_image}")
                print(f"[LOOP] PRE condition:  {pre_condition}")

                print_pose_dict_for_image(
                    poses_by_image=poses_by_image,
                    image_path=current_image,
                    label=f"validator-pre-stage-{stage_id}",
                )

                pre_name = f"pre_{stage_id}"
                pre_response = execute_validator_step(
                    settings=settings,
                    scenario_name=args.scenario,
                    validator_version=args.validator_v,
                    validator_model=args.validator_model,
                    upstream_timestamp=pipeline_timestamp,
                    run_name=run_name,
                    condition_name=pre_name,
                    condition_text=pre_condition,
                    image_path=current_image,
                    scene_version=args.scene_v,
                    scene_model=args.scene_model,
                    plan_version=args.plan_v,
                    plan_model=args.plan_model,
                    sim_version=args.sim_v,
                    sim_model=args.sim_model,
                )
                print(f"\n[PRE validator:{pre_name}] Parsed JSON:")
                print(json.dumps(pre_response, indent=2, ensure_ascii=False))
                print(f"[LOOP] PRE result:     {pre_response['result']}")
                print(f"[LOOP] PRE reason:     {pre_response['reason']}")

                stage_record["pre_validation"] = pre_response

                if pre_response["result"] == "non_matching":
                    print(f"[LOOP] Precondition failed at stage {stage_id}. Replanning from same image.")
                    cycle_record["stages"].append(stage_record)
                    cycle_record["outcome"] = f"replan_on_pre_stage_{stage_id}"

                    if replans_done >= args.max_replans:
                        raise RuntimeError(
                            f"Maximum number of replans reached ({args.max_replans})."
                        )

                    replans_done += 1
                    summary["replans_done"] = replans_done

                    all_stages_succeeded = False
                    break

                # --------------------------------------------------
                # DEPLOY PLACEHOLDER / OFFLINE NEXT IMAGE
                # --------------------------------------------------
                if frame_cursor >= len(frame_paths):
                    raise RuntimeError(
                        "No more images available in frames-dir for simulated deploy progression."
                    )

                next_image = frame_paths[frame_cursor]
                frame_cursor += 1

                stage_record["next_image_path"] = str(Path(next_image).resolve())
                stage_record["next_image_name"] = Path(next_image).name
                stage_record["post_image_path"] = str(Path(next_image).resolve())
                stage_record["post_image_name"] = Path(next_image).name

                print(f"\n[LOOP] Stage {stage_id} simulated deploy")
                print(f"[LOOP] NEXT image:     {next_image}")

                # --------------------------------------------------
                # POST VALIDATION
                # --------------------------------------------------
                print(f"\n[LOOP] Stage {stage_id} POST")
                print(f"[LOOP] POST image:     {next_image}")
                print(f"[LOOP] POST condition: {post_condition}")

                print_pose_dict_for_image(
                    poses_by_image=poses_by_image,
                    image_path=next_image,
                    label=f"validator-post-stage-{stage_id}",
                )

                post_name = f"post_{stage_id}"
                post_response = execute_validator_step(
                    settings=settings,
                    scenario_name=args.scenario,
                    validator_version=args.validator_v,
                    validator_model=args.validator_model,
                    upstream_timestamp=pipeline_timestamp,
                    run_name=run_name,
                    condition_name=post_name,
                    condition_text=post_condition,
                    image_path=next_image,
                    scene_version=args.scene_v,
                    scene_model=args.scene_model,
                    plan_version=args.plan_v,
                    plan_model=args.plan_model,
                    sim_version=args.sim_v,
                    sim_model=args.sim_model,
                )
                print(f"\n[POST validator:{post_name}] Parsed JSON:")
                print(json.dumps(post_response, indent=2, ensure_ascii=False))
                print(f"[LOOP] POST result:    {post_response['result']}")
                print(f"[LOOP] POST reason:    {post_response['reason']}")

                stage_record["post_validation"] = post_response
                cycle_record["stages"].append(stage_record)

                if post_response["result"] == "non_matching":
                    print(f"[LOOP] Postcondition failed at stage {stage_id}. Replanning from next image.")
                    cycle_record["outcome"] = f"replan_on_post_stage_{stage_id}"
                    current_image = next_image

                    if replans_done >= args.max_replans:
                        raise RuntimeError(
                            f"Maximum number of replans reached ({args.max_replans})."
                        )

                    replans_done += 1
                    summary["replans_done"] = replans_done

                    all_stages_succeeded = False
                    break

                current_image = next_image

            if all_stages_succeeded:
                cycle_record["outcome"] = "task_completed"
                task_completed = True
                summary["task_completed"] = True
                summary["final_image_path"] = str(Path(current_image).resolve())

                print("\n======================================================")
                print("[LOOP] TASK COMPLETED SUCCESSFULLY")
                print("======================================================")

        except Exception as exc:
            cycle_record["outcome"] = f"cycle_error: {exc}"
            summary["task_completed"] = False
            summary["error"] = str(exc)
            cycle_error = True

        summary["cycles"].append(cycle_record)

        if cycle_error:
            break

    summary_path = save_loop_summary(
        settings=settings,
        scenario_name=args.scenario,
        loop_timestamp=loop_timestamp,
        summary=summary,
    )

    print("\n======================================================")
    print("VALIDATION LOOP COMPLETED")
    print(f"Scenario:        {args.scenario}")
    print(f"Loop timestamp:  {loop_timestamp}")
    print(f"Task completed:  {summary['task_completed']}")
    print(f"Replans done:    {summary['replans_done']}")
    print(f"Summary saved:   {summary_path}")
    print("======================================================")


if __name__ == "__main__":
    main()

