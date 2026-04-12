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
from build_scene_object_list import build_scene_object_list_from_cycle
from scene_enrichment import enrich_scene
from utils import (
    load_base_prompt,
    make_experiment_timestamp,
    make_cycle_name,
    make_stage_name,
    render_prompt,
    save_rendered_prompt_for_cycle,
    save_module_outputs_for_cycle,
    save_scene_description_full_artifact_for_cycle,
    get_validator_prompt_cycle_dir,
    get_validator_output_cycle_dir,
    get_validation_loop_output_dir,
    get_validation_loop_cycle_dir,
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


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def save_json_file(path: Path, data: Any) -> Path:
    ensure_dir(path.parent)
    write_json(path, data)
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
    loop_timestamp: str,
    scene_model: str,
    cycle_name: str,
) -> dict[str, Any]:
    path = (
        settings.project_root
        / "outputs"
        / "scene_description"
        / scenario_name
        / scene_version
        / loop_timestamp
        / scene_model
        / cycle_name
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


def build_global_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
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
            "grounding_safety_threshold": args.grounding_safety_threshold,
            "grounding_debug_mapping": args.grounding_debug_mapping,
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
    }


def build_cycle_config(
    args: argparse.Namespace,
    cycle_timestamp: str,
    cycle_name: str,
    cycle_idx: int,
    loop_timestamp: str,
) -> dict[str, Any]:
    return {
        "cycle_name": cycle_name,
        "cycle_index": cycle_idx,
        "cycle_timestamp": cycle_timestamp,
        "scene_description": {
            "prompt_version": args.scene_v,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": args.scene_model,
        },
        "scene_description_full": {
            "stored_under_module": "scene_description",
            "artifact_filename": "scene_description_full.json",
            "prompt_version": args.scene_v,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": args.scene_model,
            "mode": "deterministic_scene_enrichment_per_image",
            "grounding_safety_threshold": args.grounding_safety_threshold,
            "grounding_debug_mapping": args.grounding_debug_mapping,
        },
        "vlm_planning": {
            "prompt_version": args.plan_v,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": args.plan_model,
        },
        "simultaneous_actions": {
            "prompt_version": args.sim_v,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": args.sim_model,
        },
    }


# ============================================================
# MODULE EXECUTION HELPERS
# ============================================================

def execute_scene_description_step(
    settings,
    scenario_name: str,
    scenario_context: dict[str, Any],
    version: str,
    model_name: str,
    loop_timestamp: str,
    cycle_name: str,
    cycle_idx: int,
    cycle_timestamp: str,
    pipeline_config: dict[str, Any],
    image_path: str,
) -> dict[str, Any]:
    module_name = "scene_description"
    base_prompt = load_base_prompt(settings, module_name, version)

    system_prompt = base_prompt
    user_text = "Analyze the scene and return the structured JSON output."

    prompt_path = save_rendered_prompt_for_cycle(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=model_name,
        cycle_name=cycle_name,
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

    parsed_path, run_info_path = save_module_outputs_for_cycle(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=result["model_name"],
        cycle_name=cycle_name,
        cycle_index=cycle_idx,
        cycle_timestamp=cycle_timestamp,
        deployment_name=result["deployment_name"],
        execution_time_seconds=result["execution_time_seconds"],
        scenario_data=scenario_context,
        parsed_response=parsed_response,
        execution_mode="validation_loop",
        dependencies=None,
        pipeline_config=pipeline_config,
    )

    scene_object_list_path = build_scene_object_list_from_cycle(
        scenario=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model=result["model_name"],
        cycle_name=cycle_name,
    )

    print(f"[OK][scene_description] Prompt saved to:         {prompt_path}")
    print(f"[OK][scene_description] Parsed output saved to:  {parsed_path}")
    print(f"[OK][scene_description] Run info saved to:       {run_info_path}")
    print(f"[OK][scene_description] Scene object list saved: {scene_object_list_path}")
    print(f"[OK][scene_description] Execution time:          {result['execution_time_seconds']:.3f}s")

    return {
        "output": parsed_response,
        "paths": {
            "prompt": str(prompt_path),
            "response_parsed": str(parsed_path),
            "run_info": str(run_info_path),
            "scene_object_list": str(scene_object_list_path),
        },
        "model_name": result["model_name"],
        "deployment_name": result["deployment_name"],
        "execution_time_seconds": result["execution_time_seconds"],
    }


def execute_scene_description_full_step(
    settings,
    scenario_name: str,
    scenario_context: dict[str, Any],
    version: str,
    model_name: str,
    loop_timestamp: str,
    cycle_name: str,
    cycle_idx: int,
    cycle_timestamp: str,
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

        dependencies = {
            "scene_description": {
                "prompt_version": version,
                "loop_timestamp": loop_timestamp,
                "cycle_name": cycle_name,
                "model": model_name,
            }
        }

        parsed_path, run_info_path = save_scene_description_full_artifact_for_cycle(
            settings=settings,
            scenario_name=scenario_name,
            version=version,
            loop_timestamp=loop_timestamp,
            model_name=model_name,
            cycle_name=cycle_name,
            cycle_index=cycle_idx,
            cycle_timestamp=cycle_timestamp,
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

    finally:
        temp_path = Path(temp_pose_file)
        if temp_path.exists():
            temp_path.unlink()

    print(f"[OK][scene_description_full] Image key used:       {Path(image_path).name}")
    print(f"[OK][scene_description_full] Parsed output saved to: {parsed_path}")
    print(f"[OK][scene_description_full] Run info saved to:      {run_info_path}")
    print(f"[OK][scene_description_full] Execution time:         {execution_time_seconds:.3f}s")

    return {
        "output": parsed_response,
        "paths": {
            "artifact": str(parsed_path),
            "run_info": str(run_info_path),
        },
        "execution_time_seconds": execution_time_seconds,
    }


def execute_vlm_planning_step(
    settings,
    scenario_name: str,
    scenario_context: dict[str, Any],
    version: str,
    model_name: str,
    loop_timestamp: str,
    cycle_name: str,
    cycle_idx: int,
    cycle_timestamp: str,
    scene_description_full: Any,
    scene_version: str,
    scene_model: str,
    pipeline_config: dict[str, Any],
) -> dict[str, Any]:
    module_name = "vlm_planning"
    base_prompt = load_base_prompt(settings, module_name, version)

    system_prompt = render_prompt(
        module_name=module_name,
        base_prompt=base_prompt,
        scenario_data=scenario_context,
        scene_description=scene_description_full,
    )

    user_text = "Generate the manipulation plan in valid JSON only."

    prompt_path = save_rendered_prompt_for_cycle(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=model_name,
        cycle_name=cycle_name,
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
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": scene_model,
        }
    }

    parsed_path, run_info_path = save_module_outputs_for_cycle(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=result["model_name"],
        cycle_name=cycle_name,
        cycle_index=cycle_idx,
        cycle_timestamp=cycle_timestamp,
        deployment_name=result["deployment_name"],
        execution_time_seconds=result["execution_time_seconds"],
        scenario_data=scenario_context,
        parsed_response=parsed_response,
        execution_mode="validation_loop",
        dependencies=dependencies,
        pipeline_config=pipeline_config,
    )

    print(f"[OK][vlm_planning] Prompt saved to:        {prompt_path}")
    print(f"[OK][vlm_planning] Parsed output saved to: {parsed_path}")
    print(f"[OK][vlm_planning] Run info saved to:      {run_info_path}")
    print(f"[OK][vlm_planning] Execution time:         {result['execution_time_seconds']:.3f}s")

    return {
        "output": parsed_response,
        "paths": {
            "prompt": str(prompt_path),
            "response_parsed": str(parsed_path),
            "run_info": str(run_info_path),
        },
        "model_name": result["model_name"],
        "execution_time_seconds": result["execution_time_seconds"],
    }


def execute_simultaneous_actions_step(
    settings,
    scenario_name: str,
    scenario_context: dict[str, Any],
    version: str,
    model_name: str,
    loop_timestamp: str,
    cycle_name: str,
    cycle_idx: int,
    cycle_timestamp: str,
    scene_description_full: Any,
    sequential_plan: Any,
    scene_version: str,
    scene_model: str,
    plan_version: str,
    plan_model: str,
    pipeline_config: dict[str, Any],
) -> dict[str, Any]:
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

    prompt_path = save_rendered_prompt_for_cycle(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=model_name,
        cycle_name=cycle_name,
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
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": scene_model,
        },
        "vlm_planning": {
            "prompt_version": plan_version,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": plan_model,
        },
    }

    parsed_path, run_info_path = save_module_outputs_for_cycle(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=result["model_name"],
        cycle_name=cycle_name,
        cycle_index=cycle_idx,
        cycle_timestamp=cycle_timestamp,
        deployment_name=result["deployment_name"],
        execution_time_seconds=result["execution_time_seconds"],
        scenario_data=scenario_context,
        parsed_response=parsed_response,
        execution_mode="validation_loop",
        dependencies=dependencies,
        pipeline_config=pipeline_config,
    )

    print(f"[OK][simultaneous_actions] Prompt saved to:        {prompt_path}")
    print(f"[OK][simultaneous_actions] Parsed output saved to: {parsed_path}")
    print(f"[OK][simultaneous_actions] Run info saved to:      {run_info_path}")
    print(f"[OK][simultaneous_actions] Execution time:         {result['execution_time_seconds']:.3f}s")

    return {
        "output": parsed_response,
        "paths": {
            "prompt": str(prompt_path),
            "response_parsed": str(parsed_path),
            "run_info": str(run_info_path),
        },
        "model_name": result["model_name"],
        "execution_time_seconds": result["execution_time_seconds"],
    }


def execute_validator_step(
    settings,
    scenario_name: str,
    validator_version: str,
    validator_model: str,
    loop_timestamp: str,
    cycle_name: str,
    cycle_idx: int,
    cycle_timestamp: str,
    stage_id: int,
    condition_kind: str,
    condition_text: str,
    image_path: str,
    scene_version: str,
    scene_model: str,
    plan_version: str,
    plan_model: str,
    sim_version: str,
    sim_model: str,
) -> dict[str, Any]:
    stage_name = make_stage_name(stage_id)

    scene_object_list = load_scene_object_list_from_cycle(
        settings=settings,
        scenario_name=scenario_name,
        scene_version=scene_version,
        loop_timestamp=loop_timestamp,
        scene_model=scene_model,
        cycle_name=cycle_name,
    )

    base_prompt = load_base_prompt(settings, "validator", validator_version)
    system_prompt = render_validator_prompt(
        base_prompt=base_prompt,
        condition=condition_text,
        scene_object_list=scene_object_list,
    )

    prompt_dir = get_validator_prompt_cycle_dir(
        settings=settings,
        scenario_name=scenario_name,
        version=validator_version,
        loop_timestamp=loop_timestamp,
        model_name=validator_model,
        cycle_name=cycle_name,
        stage_name=stage_name,
        condition_kind=condition_kind,
    )
    prompt_path = prompt_dir / "prompt.txt"
    write_text(prompt_path, system_prompt)

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
            f"[validator:{condition_kind}_{stage_id}] Model response could not be parsed as valid JSON.\n\n"
            f"Raw response:\n{result['raw_response']}"
        )

    validate_validator_response(parsed_response)

    dependencies = {
        "scene_description_full": {
            "stored_under_module": "scene_description",
            "artifact_filename": "scene_description_full.json",
            "prompt_version": scene_version,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": scene_model,
        },
        "vlm_planning": {
            "prompt_version": plan_version,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": plan_model,
        },
        "simultaneous_actions": {
            "prompt_version": sim_version,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": sim_model,
        },
    }

    output_dir = get_validator_output_cycle_dir(
        settings=settings,
        scenario_name=scenario_name,
        version=validator_version,
        loop_timestamp=loop_timestamp,
        model_name=result["model_name"],
        cycle_name=cycle_name,
        stage_name=stage_name,
        condition_kind=condition_kind,
    )
    ensure_dir(output_dir)

    parsed_path = save_json_file(output_dir / "response_parsed.json", parsed_response)

    run_info = {
        "module": "validator",
        "execution_mode": "validation_loop",
        "scenario_name": scenario_name,
        "prompt_version": validator_version,
        "loop_timestamp": loop_timestamp,
        "cycle_name": cycle_name,
        "cycle_index": cycle_idx,
        "cycle_timestamp": cycle_timestamp,
        "stage_id": stage_id,
        "stage_name": stage_name,
        "condition_kind": condition_kind,
        "condition_text": condition_text,
        "model": result["model_name"],
        "deployment_name": result["deployment_name"],
        "execution_time_seconds": result["execution_time_seconds"],
        "timestamp": datetime.now().isoformat(),
        "image_path": str(Path(image_path).resolve()),
        "dependencies": dependencies,
        "response_parsed": parsed_response,
    }
    run_info_path = save_json_file(output_dir / "run_info.json", run_info)

    print(f"[OK][validator:{condition_kind}_{stage_id}] Prompt saved to:        {prompt_path}")
    print(f"[OK][validator:{condition_kind}_{stage_id}] Parsed output saved to: {parsed_path}")
    print(f"[OK][validator:{condition_kind}_{stage_id}] Run info saved to:      {run_info_path}")
    print(f"[OK][validator:{condition_kind}_{stage_id}] Execution time:         {result['execution_time_seconds']:.3f}s")

    return {
        "output": parsed_response,
        "paths": {
            "prompt": str(prompt_path),
            "response_parsed": str(parsed_path),
            "run_info": str(run_info_path),
        },
        "model_name": result["model_name"],
        "execution_time_seconds": result["execution_time_seconds"],
    }


# ============================================================
# SUMMARY HELPERS
# ============================================================

def build_run_info(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "validation_loop",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": full_summary["timestamp"],
        "initial_image_path": full_summary["initial_image_path"],
        "frames_dir": full_summary["frames_dir"],
        "poses_by_image_path": full_summary["poses_by_image_path"],
        "config": full_summary["config"],
    }


def build_loop_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "validation_loop_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": full_summary["timestamp"],
        "config": full_summary["config"],
        "initial_image_path": full_summary["initial_image_path"],
        "final_image_path": full_summary.get("final_image_path"),
        "task_completed": full_summary["task_completed"],
        "replans_done": full_summary["replans_done"],
        "total_cycles": len(full_summary["cycles"]),
        "error": full_summary.get("error"),
        "cycles": [
            {
                "cycle_name": cycle["cycle_name"],
                "cycle_index": cycle["cycle_index"],
                "cycle_timestamp": cycle["cycle_timestamp"],
                "start_image_path": cycle["start_image_path"],
                "start_image_name": cycle["start_image_name"],
                "outcome": cycle["outcome"],
            }
            for cycle in full_summary["cycles"]
        ],
    }


def build_scene_description_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "scene_description_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": datetime.now().isoformat(),
        "config": {
            "scene_description": full_summary["config"]["scene_description"],
            "scene_description_full": full_summary["config"]["scene_description_full"],
        },
        "cycles": [
            {
                "cycle_name": cycle["cycle_name"],
                "cycle_index": cycle["cycle_index"],
                "cycle_timestamp": cycle["cycle_timestamp"],
                "image_path": cycle["start_image_path"],
                "image_name": cycle["start_image_name"],
                "scene_description_paths": {
                    "prompt": cycle["scene_description"]["paths"]["prompt"],
                    "response_parsed": cycle["scene_description"]["paths"]["response_parsed"],
                    "run_info": cycle["scene_description"]["paths"]["run_info"],
                    "scene_object_list": cycle["scene_description"]["paths"]["scene_object_list"],
                    "scene_description_full": cycle["scene_description_full"]["paths"]["artifact"],
                    "scene_description_full_run_info": cycle["scene_description_full"]["paths"]["run_info"],
                },
                "scene_description_output": cycle["scene_description"]["output"],
                "scene_description_full_output": cycle["scene_description_full"]["output"],
            }
            for cycle in full_summary["cycles"]
            if cycle.get("scene_description") is not None and cycle.get("scene_description_full") is not None
        ],
    }


def build_vlm_planning_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "vlm_planning_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": datetime.now().isoformat(),
        "config": {
            "vlm_planning": full_summary["config"]["vlm_planning"],
        },
        "cycles": [
            {
                "cycle_name": cycle["cycle_name"],
                "cycle_index": cycle["cycle_index"],
                "cycle_timestamp": cycle["cycle_timestamp"],
                "input_image_path": cycle["start_image_path"],
                "input_image_name": cycle["start_image_name"],
                "dependencies": {
                    "scene_description_cycle": cycle["cycle_name"],
                    "scene_description_full_path": cycle["scene_description_full"]["paths"]["artifact"],
                },
                "vlm_planning_paths": cycle["vlm_planning"]["paths"],
                "vlm_planning_output": cycle["vlm_planning"]["output"],
            }
            for cycle in full_summary["cycles"]
            if cycle.get("vlm_planning") is not None
        ],
    }


def build_simultaneous_actions_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "simultaneous_actions_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": datetime.now().isoformat(),
        "config": {
            "simultaneous_actions": full_summary["config"]["simultaneous_actions"],
        },
        "cycles": [
            {
                "cycle_name": cycle["cycle_name"],
                "cycle_index": cycle["cycle_index"],
                "cycle_timestamp": cycle["cycle_timestamp"],
                "input_image_path": cycle["start_image_path"],
                "input_image_name": cycle["start_image_name"],
                "dependencies": {
                    "scene_description_cycle": cycle["cycle_name"],
                    "scene_description_full_path": cycle["scene_description_full"]["paths"]["artifact"],
                    "vlm_planning_cycle": cycle["cycle_name"],
                    "vlm_planning_path": cycle["vlm_planning"]["paths"]["response_parsed"],
                },
                "simultaneous_actions_paths": cycle["simultaneous_actions"]["paths"],
                "simultaneous_actions_output": cycle["simultaneous_actions"]["output"],
            }
            for cycle in full_summary["cycles"]
            if cycle.get("simultaneous_actions") is not None
        ],
    }


def build_validator_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "validator_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": datetime.now().isoformat(),
        "config": {
            "validator": full_summary["config"]["validator"],
            "max_replans": full_summary["config"]["max_replans"],
        },
        "replans_done": full_summary["replans_done"],
        "task_completed": full_summary["task_completed"],
        "cycles": [
            {
                "cycle_name": cycle["cycle_name"],
                "cycle_index": cycle["cycle_index"],
                "cycle_timestamp": cycle["cycle_timestamp"],
                "start_image_path": cycle["start_image_path"],
                "start_image_name": cycle["start_image_name"],
                "outcome": cycle["outcome"],
                "stages": cycle["stages"],
            }
            for cycle in full_summary["cycles"]
        ],
    }


def build_full_pipeline_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(full_summary)


def build_cycle_summary(
    full_summary: dict[str, Any],
    cycle_record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "module": "cycle_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "cycle_name": cycle_record["cycle_name"],
        "cycle_index": cycle_record["cycle_index"],
        "cycle_timestamp": cycle_record["cycle_timestamp"],
        "start_image_path": cycle_record["start_image_path"],
        "start_image_name": cycle_record["start_image_name"],
        "scene_description": cycle_record.get("scene_description"),
        "scene_description_full": cycle_record.get("scene_description_full"),
        "vlm_planning": cycle_record.get("vlm_planning"),
        "simultaneous_actions": cycle_record.get("simultaneous_actions"),
        "stages": cycle_record["stages"],
        "outcome": cycle_record["outcome"],
        "end_image_path": cycle_record.get("end_image_path"),
        "end_image_name": cycle_record.get("end_image_name"),
    }


def save_validation_loop_artifacts(
    settings,
    scenario_name: str,
    loop_timestamp: str,
    run_info: dict[str, Any],
    loop_summary: dict[str, Any],
    scene_description_summary: dict[str, Any],
    vlm_planning_summary: dict[str, Any],
    simultaneous_actions_summary: dict[str, Any],
    validator_summary: dict[str, Any],
    full_pipeline_summary: dict[str, Any],
) -> dict[str, Path]:
    output_dir = get_validation_loop_output_dir(settings, scenario_name, loop_timestamp)
    ensure_dir(output_dir)

    paths = {
        "run_info": save_json_file(output_dir / "run_info.json", run_info),
        "loop_summary": save_json_file(output_dir / "loop_summary.json", loop_summary),
        "scene_description_summary": save_json_file(
            output_dir / "scene_description_summary.json",
            scene_description_summary,
        ),
        "vlm_planning_summary": save_json_file(
            output_dir / "vlm_planning_summary.json",
            vlm_planning_summary,
        ),
        "simultaneous_actions_summary": save_json_file(
            output_dir / "simultaneous_actions_summary.json",
            simultaneous_actions_summary,
        ),
        "validator_summary": save_json_file(
            output_dir / "validator_summary.json",
            validator_summary,
        ),
        "full_pipeline_summary": save_json_file(
            output_dir / "full_pipeline_summary.json",
            full_pipeline_summary,
        ),
    }
    return paths


def save_cycle_summary(
    settings,
    scenario_name: str,
    loop_timestamp: str,
    cycle_name: str,
    cycle_summary: dict[str, Any],
) -> Path:
    cycle_dir = get_validation_loop_cycle_dir(settings, scenario_name, loop_timestamp, cycle_name)
    ensure_dir(cycle_dir)
    return save_json_file(cycle_dir / "cycle_summary.json", cycle_summary)


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

    current_image = initial_image_path
    task_completed = False
    cycle_idx = 0

    full_summary: dict[str, Any] = {
        "module": "full_pipeline_summary",
        "scenario_name": args.scenario,
        "loop_timestamp": loop_timestamp,
        "timestamp": datetime.now().isoformat(),
        "initial_image_path": str(Path(initial_image_path).resolve()),
        "frames_dir": str(Path(args.frames_dir).resolve()),
        "poses_by_image_path": str(poses_by_image_path),
        "config": build_global_config(args),
        "replans_done": 0,
        "task_completed": False,
        "final_image_path": None,
        "cycles": [],
    }

    while not task_completed:
        cycle_idx += 1
        cycle_name = make_cycle_name(cycle_idx)
        cycle_timestamp = make_experiment_timestamp()

        print("\n======================================================")
        print(f"VALIDATION LOOP CYCLE STARTED | cycle={cycle_idx} | {cycle_name}")
        print(f"Current image:   {current_image}")
        print(f"Loop ts:         {loop_timestamp}")
        print(f"Cycle ts meta:   {cycle_timestamp}")
        print("======================================================")

        scenario_context = make_scenario_context(
            scenario_data=scenario_data,
            image_path=current_image,
        )

        pipeline_config = build_cycle_config(
            args=args,
            cycle_timestamp=cycle_timestamp,
            cycle_name=cycle_name,
            cycle_idx=cycle_idx,
            loop_timestamp=loop_timestamp,
        )

        cycle_record: dict[str, Any] = {
            "cycle_name": cycle_name,
            "cycle_index": cycle_idx,
            "cycle_timestamp": cycle_timestamp,
            "start_image_path": str(Path(current_image).resolve()),
            "start_image_name": Path(current_image).name,
            "scene_description": None,
            "scene_description_full": None,
            "vlm_planning": None,
            "simultaneous_actions": None,
            "stages": [],
            "outcome": None,
            "end_image_path": None,
            "end_image_name": None,
        }

        cycle_error = False

        try:
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
            )
            cycle_record["scene_description"] = scene_description_artifact

            print("\n[scene_description] Parsed JSON:")
            print(json.dumps(scene_description_artifact["output"], indent=2, ensure_ascii=False))

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
            cycle_record["scene_description_full"] = scene_description_full_artifact

            print("\n[scene_description_full] Parsed JSON:")
            print(json.dumps(scene_description_full_artifact["output"], indent=2, ensure_ascii=False))

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
                scene_description_full=scene_description_full_artifact["output"],
                scene_version=args.scene_v,
                scene_model=args.scene_model,
                pipeline_config=pipeline_config,
            )
            cycle_record["vlm_planning"] = sequential_plan_artifact

            print("\n[vlm_planning] Parsed JSON:")
            print(json.dumps(sequential_plan_artifact["output"], indent=2, ensure_ascii=False))

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
                scene_description_full=scene_description_full_artifact["output"],
                sequential_plan=sequential_plan_artifact["output"],
                scene_version=args.scene_v,
                scene_model=args.scene_model,
                plan_version=args.plan_v,
                plan_model=args.plan_model,
                pipeline_config=pipeline_config,
            )
            cycle_record["simultaneous_actions"] = simultaneous_actions_artifact

            print("\n[simultaneous_actions] Parsed JSON:")
            print(json.dumps(simultaneous_actions_artifact["output"], indent=2, ensure_ascii=False))

            stages = extract_stages(simultaneous_actions_artifact["output"])
            all_stages_succeeded = True

            for stage in stages:
                stage_id = stage["Stage_id"]
                stage_name = make_stage_name(stage_id)
                pre_condition = stage["Precondition"]
                post_condition = stage["Postcondition"]

                stage_record: dict[str, Any] = {
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "precondition": pre_condition,
                    "postcondition": post_condition,
                    "pre_image_path": str(Path(current_image).resolve()),
                    "pre_image_name": Path(current_image).name,
                    "post_image_path": None,
                    "post_image_name": None,
                    "pre_validation": None,
                    "post_validation": None,
                    "validator_paths": {
                        "pre": None,
                        "post": None,
                    },
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

                pre_artifact = execute_validator_step(
                    settings=settings,
                    scenario_name=args.scenario,
                    validator_version=args.validator_v,
                    validator_model=args.validator_model,
                    loop_timestamp=loop_timestamp,
                    cycle_name=cycle_name,
                    cycle_idx=cycle_idx,
                    cycle_timestamp=cycle_timestamp,
                    stage_id=stage_id,
                    condition_kind="pre",
                    condition_text=pre_condition,
                    image_path=current_image,
                    scene_version=args.scene_v,
                    scene_model=args.scene_model,
                    plan_version=args.plan_v,
                    plan_model=args.plan_model,
                    sim_version=args.sim_v,
                    sim_model=args.sim_model,
                )
                pre_response = pre_artifact["output"]

                print(f"\n[PRE validator:pre_{stage_id}] Parsed JSON:")
                print(json.dumps(pre_response, indent=2, ensure_ascii=False))
                print(f"[LOOP] PRE result:     {pre_response['result']}")
                print(f"[LOOP] PRE reason:     {pre_response['reason']}")

                stage_record["pre_validation"] = pre_response
                stage_record["validator_paths"]["pre"] = pre_artifact["paths"]

                if pre_response["result"] == "non_matching":
                    print(f"[LOOP] Precondition failed at stage {stage_id}. Replanning from same image.")
                    cycle_record["stages"].append(stage_record)
                    cycle_record["outcome"] = f"replan_on_pre_stage_{stage_id}"
                    cycle_record["end_image_path"] = str(Path(current_image).resolve())
                    cycle_record["end_image_name"] = Path(current_image).name

                    if full_summary["replans_done"] >= args.max_replans:
                        raise RuntimeError(
                            f"Maximum number of replans reached ({args.max_replans})."
                        )

                    full_summary["replans_done"] += 1
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

                post_artifact = execute_validator_step(
                    settings=settings,
                    scenario_name=args.scenario,
                    validator_version=args.validator_v,
                    validator_model=args.validator_model,
                    loop_timestamp=loop_timestamp,
                    cycle_name=cycle_name,
                    cycle_idx=cycle_idx,
                    cycle_timestamp=cycle_timestamp,
                    stage_id=stage_id,
                    condition_kind="post",
                    condition_text=post_condition,
                    image_path=next_image,
                    scene_version=args.scene_v,
                    scene_model=args.scene_model,
                    plan_version=args.plan_v,
                    plan_model=args.plan_model,
                    sim_version=args.sim_v,
                    sim_model=args.sim_model,
                )
                post_response = post_artifact["output"]

                print(f"\n[POST validator:post_{stage_id}] Parsed JSON:")
                print(json.dumps(post_response, indent=2, ensure_ascii=False))
                print(f"[LOOP] POST result:    {post_response['result']}")
                print(f"[LOOP] POST reason:    {post_response['reason']}")

                stage_record["post_validation"] = post_response
                stage_record["validator_paths"]["post"] = post_artifact["paths"]
                cycle_record["stages"].append(stage_record)

                if post_response["result"] == "non_matching":
                    print(f"[LOOP] Postcondition failed at stage {stage_id}. Replanning from next image.")
                    cycle_record["outcome"] = f"replan_on_post_stage_{stage_id}"
                    cycle_record["end_image_path"] = str(Path(next_image).resolve())
                    cycle_record["end_image_name"] = Path(next_image).name
                    current_image = next_image

                    if full_summary["replans_done"] >= args.max_replans:
                        raise RuntimeError(
                            f"Maximum number of replans reached ({args.max_replans})."
                        )

                    full_summary["replans_done"] += 1
                    all_stages_succeeded = False
                    break

                current_image = next_image

            if all_stages_succeeded:
                cycle_record["outcome"] = "task_completed"
                cycle_record["end_image_path"] = str(Path(current_image).resolve())
                cycle_record["end_image_name"] = Path(current_image).name
                task_completed = True
                full_summary["task_completed"] = True
                full_summary["final_image_path"] = str(Path(current_image).resolve())

                print("\n======================================================")
                print("[LOOP] TASK COMPLETED SUCCESSFULLY")
                print("======================================================")

        except Exception as exc:
            cycle_record["outcome"] = f"cycle_error: {exc}"
            cycle_record["end_image_path"] = str(Path(current_image).resolve())
            cycle_record["end_image_name"] = Path(current_image).name
            full_summary["task_completed"] = False
            full_summary["error"] = str(exc)
            cycle_error = True

        full_summary["cycles"].append(cycle_record)

        cycle_summary = build_cycle_summary(full_summary, cycle_record)
        cycle_summary_path = save_cycle_summary(
            settings=settings,
            scenario_name=args.scenario,
            loop_timestamp=loop_timestamp,
            cycle_name=cycle_name,
            cycle_summary=cycle_summary,
        )
        print(f"[OK][validation_loop] Cycle summary saved to: {cycle_summary_path}")

        if cycle_error:
            break

    run_info = build_run_info(full_summary)
    loop_summary = build_loop_summary(full_summary)
    scene_description_summary = build_scene_description_summary(full_summary)
    vlm_planning_summary = build_vlm_planning_summary(full_summary)
    simultaneous_actions_summary = build_simultaneous_actions_summary(full_summary)
    validator_summary = build_validator_summary(full_summary)
    full_pipeline_summary = build_full_pipeline_summary(full_summary)

    summary_paths = save_validation_loop_artifacts(
        settings=settings,
        scenario_name=args.scenario,
        loop_timestamp=loop_timestamp,
        run_info=run_info,
        loop_summary=loop_summary,
        scene_description_summary=scene_description_summary,
        vlm_planning_summary=vlm_planning_summary,
        simultaneous_actions_summary=simultaneous_actions_summary,
        validator_summary=validator_summary,
        full_pipeline_summary=full_pipeline_summary,
    )

    print("\n======================================================")
    print("VALIDATION LOOP COMPLETED")
    print(f"Scenario:                  {args.scenario}")
    print(f"Loop timestamp:            {loop_timestamp}")
    print(f"Task completed:            {full_summary['task_completed']}")
    print(f"Replans done:              {full_summary['replans_done']}")
    print(f"Run info saved:            {summary_paths['run_info']}")
    print(f"Loop summary saved:        {summary_paths['loop_summary']}")
    print(f"Scene summary saved:       {summary_paths['scene_description_summary']}")
    print(f"Planning summary saved:    {summary_paths['vlm_planning_summary']}")
    print(f"Sim-actions summary saved: {summary_paths['simultaneous_actions_summary']}")
    print(f"Validator summary saved:   {summary_paths['validator_summary']}")
    print(f"Full summary saved:        {summary_paths['full_pipeline_summary']}")
    print("======================================================")


if __name__ == "__main__":
    main()





# from __future__ import annotations

# import argparse
# import json
# import shutil
# import tempfile
# import time
# from copy import deepcopy
# from datetime import datetime
# from pathlib import Path
# from typing import Any

# from settings import load_settings
# from scenario_loader import load_scenario
# from azure_openai_client import call_azure_chat_completion
# from build_scene_object_list import build_scene_object_list_from_run
# from scene_enrichment import enrich_scene
# from utils import (
#     load_base_prompt,
#     make_experiment_timestamp,
#     render_prompt,
#     save_module_outputs,                 # kept only for legacy compatibility
#     save_rendered_prompt,               # kept only for legacy compatibility
#     save_scene_description_full_artifact,  # kept only for legacy compatibility
#     try_parse_json,
#     write_json,
#     read_json,
# )

# SUPPORTED_MODELS = ["o3", "gpt-5.2"]
# IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# # ============================================================
# # PARSER
# # ============================================================

# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Run the offline validation loop: pipeline -> stage pre/post validation -> "
#             "replanning on failure."
#         )
#     )

#     parser.add_argument("--scenario", type=str, required=True)

#     parser.add_argument(
#         "--initial-image-path",
#         type=str,
#         default=None,
#         help="Optional explicit initial image path. If omitted, uses scenario.json image.",
#     )

#     parser.add_argument(
#         "--frames-dir",
#         type=str,
#         required=True,
#         help=(
#             "Directory containing the sequence of post-deploy images in chronological order. "
#             "These images are consumed one-by-one when a stage is executed."
#         ),
#     )

#     parser.add_argument(
#         "--poses-by-image-path",
#         type=str,
#         default=None,
#         help=(
#             "Optional path to a JSON mapping image filename -> pose dictionary. "
#             "If omitted, defaults to scenarios/<scenario>/poses_by_image.json"
#         ),
#     )

#     parser.add_argument("--scene-v", type=str, required=True)
#     parser.add_argument("--plan-v", type=str, required=True)
#     parser.add_argument("--sim-v", type=str, required=True)
#     parser.add_argument("--validator-v", type=str, required=True)

#     parser.add_argument("--scene-model", type=str, required=True, choices=SUPPORTED_MODELS)
#     parser.add_argument("--plan-model", type=str, required=True, choices=SUPPORTED_MODELS)
#     parser.add_argument("--sim-model", type=str, required=True, choices=SUPPORTED_MODELS)
#     parser.add_argument("--validator-model", type=str, required=True, choices=SUPPORTED_MODELS)

#     parser.add_argument(
#         "--max-replans",
#         type=int,
#         default=10,
#         help="Maximum number of replanning cycles allowed before stopping.",
#     )

#     parser.add_argument(
#         "--grounding-safety-threshold",
#         type=float,
#         default=0.21,
#         help="Safety threshold used by scene enrichment to compute accessibility.",
#     )
#     parser.add_argument(
#         "--grounding-debug-mapping",
#         action="store_true",
#         help="Store the internal VLM-to-Gazebo mapping inside scene_description_full.json under _debug.",
#     )

#     return parser


# # ============================================================
# # HELPERS
# # ============================================================

# def ensure_dir(path: Path) -> Path:
#     path.mkdir(parents=True, exist_ok=True)
#     return path


# def validate_args(args: argparse.Namespace) -> None:
#     if args.max_replans < 0:
#         raise ValueError("--max-replans must be >= 0")

#     frames_dir = Path(args.frames_dir)
#     if not frames_dir.exists():
#         raise FileNotFoundError(f"frames-dir not found: {frames_dir}")
#     if not frames_dir.is_dir():
#         raise ValueError(f"--frames-dir must be a directory: {frames_dir}")

#     if args.poses_by_image_path is not None:
#         poses_path = Path(args.poses_by_image_path)
#         if not poses_path.exists():
#             raise FileNotFoundError(f"poses-by-image-path not found: {poses_path}")


# def list_frame_paths(frames_dir: str | Path) -> list[str]:
#     frames_dir = Path(frames_dir)
#     frames = sorted(
#         [
#             p
#             for p in frames_dir.iterdir()
#             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
#         ]
#     )

#     if not frames:
#         raise ValueError(f"No image files found inside frames-dir: {frames_dir}")

#     return [str(p.resolve()) for p in frames]


# def print_pose_dict_for_image(
#     poses_by_image: dict[str, dict[str, list[float]]],
#     image_path: str,
#     label: str,
# ) -> None:
#     image_name = Path(image_path).name

#     if image_name not in poses_by_image:
#         print(f"\n[DEBUG][{label}] No poses found for image: {image_name}")
#         return

#     pose_dict = poses_by_image[image_name]

#     print(f"\n[DEBUG][{label}] Image path: {image_path}")
#     print(f"[DEBUG][{label}] Image key:  {image_name}")
#     print(f"[DEBUG][{label}] Pose entries:")

#     for obj_name, pose in pose_dict.items():
#         print(f"  - {obj_name}: {pose}")


# def make_scenario_context(
#     scenario_data: dict[str, Any],
#     image_path: str,
# ) -> dict[str, Any]:
#     ctx = deepcopy(scenario_data)
#     ctx["image"] = Path(image_path).name
#     ctx["image_path_abs"] = str(Path(image_path).resolve())
#     return ctx


# def resolve_poses_by_image_path(
#     settings,
#     scenario_name: str,
#     explicit_path: str | None,
# ) -> Path:
#     if explicit_path is not None:
#         path = Path(explicit_path).resolve()
#     else:
#         path = (
#             settings.project_root
#             / "scenarios"
#             / scenario_name
#             / "poses_by_image.json"
#         ).resolve()

#     if not path.exists():
#         raise FileNotFoundError(f"poses_by_image.json not found: {path}")

#     return path


# def load_poses_by_image_map(path: str | Path) -> dict[str, dict[str, list[float]]]:
#     data = read_json(path)
#     if not isinstance(data, dict):
#         raise ValueError(
#             f"poses_by_image mapping must be a JSON object. Found: {type(data).__name__}"
#         )

#     validated: dict[str, dict[str, list[float]]] = {}

#     for image_name, pose_dict in data.items():
#         if not isinstance(image_name, str):
#             raise ValueError("Each poses_by_image key must be an image filename string.")

#         if not isinstance(pose_dict, dict):
#             raise ValueError(
#                 f"poses_by_image['{image_name}'] must be an object mapping object names to [x, y, z]."
#             )

#         cleaned_pose_dict: dict[str, list[float]] = {}
#         for obj_name, pose in pose_dict.items():
#             if not isinstance(obj_name, str):
#                 raise ValueError(
#                     f"poses_by_image['{image_name}'] contains a non-string object name."
#                 )
#             if not isinstance(pose, list) or len(pose) != 3:
#                 raise ValueError(
#                     f"poses_by_image['{image_name}']['{obj_name}'] must be a list of 3 numeric values."
#                 )
#             if not all(isinstance(v, (int, float)) for v in pose):
#                 raise ValueError(
#                     f"poses_by_image['{image_name}']['{obj_name}'] must contain only numeric values."
#                 )
#             cleaned_pose_dict[obj_name] = [float(v) for v in pose]

#         validated[image_name] = cleaned_pose_dict

#     return validated


# def get_pose_dict_for_image(
#     poses_by_image: dict[str, dict[str, list[float]]],
#     image_path: str,
# ) -> dict[str, list[float]]:
#     image_name = Path(image_path).name

#     if image_name not in poses_by_image:
#         available = ", ".join(sorted(poses_by_image.keys())[:10])
#         raise KeyError(
#             f"No pose entry found for image '{image_name}' in poses_by_image mapping. "
#             f"Available examples: {available}"
#         )

#     return poses_by_image[image_name]


# def write_temp_pose_file(pose_dict: dict[str, list[float]]) -> str:
#     with tempfile.NamedTemporaryFile(
#         mode="w",
#         suffix=".json",
#         encoding="utf-8",
#         delete=False,
#     ) as tmp:
#         json.dump(pose_dict, tmp, indent=2, ensure_ascii=False)
#         return tmp.name


# def make_cycle_name(cycle_idx: int) -> str:
#     return f"cycle_{cycle_idx:03d}"


# def make_stage_name(stage_id: int) -> str:
#     return f"stage_{stage_id:03d}"


# def build_global_config(args: argparse.Namespace) -> dict[str, Any]:
#     return {
#         "scene_description": {
#             "prompt_version": args.scene_v,
#             "model": args.scene_model,
#         },
#         "scene_description_full": {
#             "stored_under_module": "scene_description",
#             "artifact_filename": "scene_description_full.json",
#             "prompt_version": args.scene_v,
#             "model": args.scene_model,
#             "mode": "deterministic_scene_enrichment_per_image",
#             "grounding_safety_threshold": args.grounding_safety_threshold,
#             "grounding_debug_mapping": args.grounding_debug_mapping,
#         },
#         "vlm_planning": {
#             "prompt_version": args.plan_v,
#             "model": args.plan_model,
#         },
#         "simultaneous_actions": {
#             "prompt_version": args.sim_v,
#             "model": args.sim_model,
#         },
#         "validator": {
#             "prompt_version": args.validator_v,
#             "model": args.validator_model,
#         },
#         "max_replans": args.max_replans,
#     }


# def build_cycle_config(
#     args: argparse.Namespace,
#     cycle_timestamp: str,
#     cycle_name: str,
#     cycle_idx: int,
# ) -> dict[str, Any]:
#     return {
#         "cycle_name": cycle_name,
#         "cycle_index": cycle_idx,
#         "cycle_timestamp": cycle_timestamp,
#         "scene_description": {
#             "prompt_version": args.scene_v,
#             "loop_timestamp": None,  # filled only in global summaries
#             "cycle_name": cycle_name,
#             "model": args.scene_model,
#         },
#         "scene_description_full": {
#             "stored_under_module": "scene_description",
#             "artifact_filename": "scene_description_full.json",
#             "prompt_version": args.scene_v,
#             "cycle_name": cycle_name,
#             "model": args.scene_model,
#             "mode": "deterministic_scene_enrichment_per_image",
#             "grounding_safety_threshold": args.grounding_safety_threshold,
#             "grounding_debug_mapping": args.grounding_debug_mapping,
#         },
#         "vlm_planning": {
#             "prompt_version": args.plan_v,
#             "cycle_name": cycle_name,
#             "model": args.plan_model,
#         },
#         "simultaneous_actions": {
#             "prompt_version": args.sim_v,
#             "cycle_name": cycle_name,
#             "model": args.sim_model,
#         },
#     }


# def get_module_prompt_dir(
#     settings,
#     module_name: str,
#     scenario_name: str,
#     version: str,
#     loop_timestamp: str,
#     model_name: str,
#     cycle_name: str,
# ) -> Path:
#     return (
#         settings.project_root
#         / "prompts_scenarios"
#         / module_name
#         / scenario_name
#         / version
#         / loop_timestamp
#         / model_name
#         / cycle_name
#     )


# def get_module_output_dir(
#     settings,
#     module_name: str,
#     scenario_name: str,
#     version: str,
#     loop_timestamp: str,
#     model_name: str,
#     cycle_name: str,
# ) -> Path:
#     return (
#         settings.project_root
#         / "outputs"
#         / module_name
#         / scenario_name
#         / version
#         / loop_timestamp
#         / model_name
#         / cycle_name
#     )


# def get_validator_prompt_dir(
#     settings,
#     scenario: str,
#     version: str,
#     loop_timestamp: str,
#     model_name: str,
#     cycle_name: str,
#     stage_name: str,
#     condition_kind: str,
# ) -> Path:
#     return (
#         settings.project_root
#         / "prompts_scenarios"
#         / "validator"
#         / scenario
#         / version
#         / loop_timestamp
#         / model_name
#         / cycle_name
#         / stage_name
#         / condition_kind
#     )


# def get_validator_output_dir(
#     settings,
#     scenario: str,
#     version: str,
#     loop_timestamp: str,
#     model_name: str,
#     cycle_name: str,
#     stage_name: str,
#     condition_kind: str,
# ) -> Path:
#     return (
#         settings.project_root
#         / "outputs"
#         / "validator"
#         / scenario
#         / version
#         / loop_timestamp
#         / model_name
#         / cycle_name
#         / stage_name
#         / condition_kind
#     )


# def get_validation_loop_dir(
#     settings,
#     scenario_name: str,
#     loop_timestamp: str,
# ) -> Path:
#     return (
#         settings.project_root
#         / "outputs"
#         / "validation_loop"
#         / scenario_name
#         / loop_timestamp
#     )


# def get_validation_loop_cycle_dir(
#     settings,
#     scenario_name: str,
#     loop_timestamp: str,
#     cycle_name: str,
# ) -> Path:
#     return get_validation_loop_dir(settings, scenario_name, loop_timestamp) / "cycles" / cycle_name


# def write_text(path: Path, text: str) -> None:
#     ensure_dir(path.parent)
#     path.write_text(text, encoding="utf-8")


# def save_prompt_text(path: Path, prompt_text: str) -> Path:
#     write_text(path, prompt_text)
#     return path


# def save_json_file(path: Path, data: Any) -> Path:
#     ensure_dir(path.parent)
#     write_json(path, data)
#     return path


# def load_scene_object_list_from_cycle(
#     settings,
#     scenario_name: str,
#     scene_version: str,
#     loop_timestamp: str,
#     scene_model: str,
#     cycle_name: str,
# ) -> dict[str, Any]:
#     path = (
#         settings.project_root
#         / "outputs"
#         / "scene_description"
#         / scenario_name
#         / scene_version
#         / loop_timestamp
#         / scene_model
#         / cycle_name
#         / "scene_object_list.json"
#     )

#     if not path.exists():
#         raise FileNotFoundError(f"scene_object_list.json not found: {path}")

#     return read_json(path)


# def extract_stages(compact_parallel_plan: Any) -> list[dict[str, Any]]:
#     if not isinstance(compact_parallel_plan, list):
#         raise ValueError("simultaneous_actions output must be a JSON array of stages.")

#     stages: list[dict[str, Any]] = []
#     for idx, stage in enumerate(compact_parallel_plan):
#         if not isinstance(stage, dict):
#             raise ValueError(f"Stage at index {idx} is not a JSON object.")

#         stage_id = stage.get("Stage_id")
#         precondition = stage.get("Precondition")
#         postcondition = stage.get("Postcondition")

#         if not isinstance(stage_id, int):
#             raise ValueError(f"Stage at index {idx} has invalid or missing 'Stage_id'.")
#         if not isinstance(precondition, str) or not precondition.strip():
#             raise ValueError(f"Stage {stage_id} has invalid or missing 'Precondition'.")
#         if not isinstance(postcondition, str) or not postcondition.strip():
#             raise ValueError(f"Stage {stage_id} has invalid or missing 'Postcondition'.")

#         stages.append(
#             {
#                 "Stage_id": stage_id,
#                 "Precondition": precondition,
#                 "Postcondition": postcondition,
#             }
#         )

#     return stages


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


# # ============================================================
# # LEGACY COMPATIBILITY
# # ============================================================

# def create_legacy_scene_description_artifacts_for_object_list(
#     settings,
#     scenario_name: str,
#     scenario_context: dict[str, Any],
#     version: str,
#     model_name: str,
#     legacy_cycle_timestamp: str,
#     run_name: str,
#     prompt_text: str,
#     parsed_response: dict[str, Any],
#     deployment_name: str,
#     execution_time_seconds: float,
#     pipeline_config: dict[str, Any],
# ) -> Path:
#     """
#     This preserves compatibility with build_scene_object_list_from_run(...),
#     which still relies on the old directory convention.
#     """
#     save_rendered_prompt(
#         settings=settings,
#         module_name="scene_description",
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=legacy_cycle_timestamp,
#         model_name=model_name,
#         run_name=run_name,
#         prompt_text=prompt_text,
#     )

#     save_module_outputs(
#         settings=settings,
#         module_name="scene_description",
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=legacy_cycle_timestamp,
#         model_name=model_name,
#         run_name=run_name,
#         deployment_name=deployment_name,
#         execution_time_seconds=execution_time_seconds,
#         scenario_data=scenario_context,
#         parsed_response=parsed_response,
#         execution_mode="validation_loop_legacy_compat",
#         dependencies=None,
#         pipeline_config=pipeline_config,
#     )

#     scene_object_list_path = build_scene_object_list_from_run(
#         scenario=scenario_name,
#         version=version,
#         experiment_timestamp=legacy_cycle_timestamp,
#         model=model_name,
#         run_name=run_name,
#     )
#     return Path(scene_object_list_path)


# def create_legacy_scene_description_full_artifacts(
#     settings,
#     scenario_name: str,
#     scenario_context: dict[str, Any],
#     version: str,
#     model_name: str,
#     legacy_cycle_timestamp: str,
#     run_name: str,
#     parsed_response: dict[str, Any],
#     execution_time_seconds: float,
#     dependencies: dict[str, Any],
#     pipeline_config: dict[str, Any],
#     pose_file: str,
#     safety_threshold: float,
#     include_debug_mapping: bool,
# ) -> None:
#     save_scene_description_full_artifact(
#         settings=settings,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=legacy_cycle_timestamp,
#         model_name=model_name,
#         run_name=run_name,
#         parsed_response=parsed_response,
#         scenario_data=scenario_context,
#         execution_time_seconds=execution_time_seconds,
#         dependencies=dependencies,
#         pipeline_config=pipeline_config,
#         pose_file=pose_file,
#         safety_threshold=safety_threshold,
#         include_debug_mapping=include_debug_mapping,
#         execution_mode="validation_loop_legacy_compat",
#     )


# # ============================================================
# # MODULE EXECUTION HELPERS (NEW STRUCTURE)
# # ============================================================

# def execute_scene_description_step(
#     settings,
#     scenario_name: str,
#     scenario_context: dict[str, Any],
#     version: str,
#     model_name: str,
#     loop_timestamp: str,
#     cycle_name: str,
#     cycle_idx: int,
#     cycle_timestamp: str,
#     run_name: str,
#     pipeline_config: dict[str, Any],
#     image_path: str,
# ) -> dict[str, Any]:
#     module_name = "scene_description"
#     base_prompt = load_base_prompt(settings, module_name, version)

#     system_prompt = base_prompt
#     user_text = "Analyze the scene and return the structured JSON output."

#     prompt_dir = get_module_prompt_dir(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=model_name,
#         cycle_name=cycle_name,
#     )
#     prompt_path = save_prompt_text(prompt_dir / "prompt.txt", system_prompt)

#     result = call_azure_chat_completion(
#         settings=settings,
#         model_name=model_name,
#         system_prompt=system_prompt,
#         user_text=user_text,
#         image_path=image_path,
#     )

#     parse_ok, parsed_response = try_parse_json(result["raw_response"])
#     if not parse_ok:
#         raise ValueError(
#             f"[scene_description] Model response could not be parsed as valid JSON.\n\n"
#             f"Raw response:\n{result['raw_response']}"
#         )

#     output_dir = get_module_output_dir(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=result["model_name"],
#         cycle_name=cycle_name,
#     )
#     ensure_dir(output_dir)

#     parsed_path = save_json_file(output_dir / "response_parsed.json", parsed_response)

#     run_info = {
#         "module": "scene_description",
#         "execution_mode": "validation_loop",
#         "scenario_name": scenario_name,
#         "prompt_version": version,
#         "loop_timestamp": loop_timestamp,
#         "cycle_name": cycle_name,
#         "cycle_index": cycle_idx,
#         "cycle_timestamp": cycle_timestamp,
#         "run_name": run_name,
#         "model": result["model_name"],
#         "deployment_name": result["deployment_name"],
#         "execution_time_seconds": result["execution_time_seconds"],
#         "timestamp": datetime.now().isoformat(),
#         "image_path": str(Path(image_path).resolve()),
#         "scenario_data": scenario_context,
#         "pipeline_config": pipeline_config,
#         "response_parsed": parsed_response,
#     }
#     run_info_path = save_json_file(output_dir / "run_info.json", run_info)

#     # legacy compatibility only for build_scene_object_list_from_run(...)
#     scene_object_list_legacy_path = create_legacy_scene_description_artifacts_for_object_list(
#         settings=settings,
#         scenario_name=scenario_name,
#         scenario_context=scenario_context,
#         version=version,
#         model_name=result["model_name"],
#         legacy_cycle_timestamp=cycle_timestamp,
#         run_name=run_name,
#         prompt_text=system_prompt,
#         parsed_response=parsed_response,
#         deployment_name=result["deployment_name"],
#         execution_time_seconds=result["execution_time_seconds"],
#         pipeline_config=pipeline_config,
#     )

#     scene_object_list_new_path = output_dir / "scene_object_list.json"
#     ensure_dir(scene_object_list_new_path.parent)
#     shutil.copy2(scene_object_list_legacy_path, scene_object_list_new_path)

#     print(f"[OK][scene_description] Prompt saved to:         {prompt_path}")
#     print(f"[OK][scene_description] Parsed output saved to:  {parsed_path}")
#     print(f"[OK][scene_description] Run info saved to:       {run_info_path}")
#     print(f"[OK][scene_description] Scene object list saved: {scene_object_list_new_path}")
#     print(f"[OK][scene_description] Execution time:          {result['execution_time_seconds']:.3f}s")

#     return {
#         "output": parsed_response,
#         "paths": {
#             "prompt": str(prompt_path),
#             "response_parsed": str(parsed_path),
#             "run_info": str(run_info_path),
#             "scene_object_list": str(scene_object_list_new_path),
#         },
#         "model_name": result["model_name"],
#         "deployment_name": result["deployment_name"],
#         "execution_time_seconds": result["execution_time_seconds"],
#     }


# def execute_scene_description_full_step(
#     settings,
#     scenario_name: str,
#     scenario_context: dict[str, Any],
#     version: str,
#     model_name: str,
#     loop_timestamp: str,
#     cycle_name: str,
#     cycle_idx: int,
#     cycle_timestamp: str,
#     run_name: str,
#     scene_description: Any,
#     pipeline_config: dict[str, Any],
#     image_path: str,
#     poses_by_image: dict[str, dict[str, list[float]]],
#     safety_threshold: float,
#     include_debug_mapping: bool,
# ) -> dict[str, Any]:
#     output_dir = get_module_output_dir(
#         settings=settings,
#         module_name="scene_description",
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=model_name,
#         cycle_name=cycle_name,
#     )
#     ensure_dir(output_dir)

#     pose_dict = get_pose_dict_for_image(poses_by_image, image_path)
#     temp_pose_file = write_temp_pose_file(pose_dict)

#     try:
#         start_time = time.perf_counter()

#         parsed_response = enrich_scene(
#             input_data=scene_description,
#             safety_threshold=safety_threshold,
#             pose_source="static",
#             pose_file=temp_pose_file,
#             include_debug_mapping=include_debug_mapping,
#         )

#         execution_time_seconds = time.perf_counter() - start_time

#         dependencies = {
#             "scene_description": {
#                 "prompt_version": version,
#                 "loop_timestamp": loop_timestamp,
#                 "cycle_name": cycle_name,
#                 "model": model_name,
#                 "run_name": run_name,
#             }
#         }

#         parsed_path = save_json_file(output_dir / "scene_description_full.json", parsed_response)

#         run_info = {
#             "module": "scene_description_full",
#             "execution_mode": "validation_loop",
#             "scenario_name": scenario_name,
#             "prompt_version": version,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "cycle_index": cycle_idx,
#             "cycle_timestamp": cycle_timestamp,
#             "run_name": run_name,
#             "model": model_name,
#             "execution_time_seconds": execution_time_seconds,
#             "timestamp": datetime.now().isoformat(),
#             "image_path": str(Path(image_path).resolve()),
#             "scenario_data": scenario_context,
#             "dependencies": dependencies,
#             "pipeline_config": pipeline_config,
#             "pose_file": temp_pose_file,
#             "safety_threshold": safety_threshold,
#             "include_debug_mapping": include_debug_mapping,
#             "response_parsed": parsed_response,
#         }
#         run_info_path = save_json_file(output_dir / "scene_description_full_run_info.json", run_info)

#         # legacy compatibility only
#         create_legacy_scene_description_full_artifacts(
#             settings=settings,
#             scenario_name=scenario_name,
#             scenario_context=scenario_context,
#             version=version,
#             model_name=model_name,
#             legacy_cycle_timestamp=cycle_timestamp,
#             run_name=run_name,
#             parsed_response=parsed_response,
#             execution_time_seconds=execution_time_seconds,
#             dependencies=dependencies,
#             pipeline_config=pipeline_config,
#             pose_file=temp_pose_file,
#             safety_threshold=safety_threshold,
#             include_debug_mapping=include_debug_mapping,
#         )

#     finally:
#         temp_path = Path(temp_pose_file)
#         if temp_path.exists():
#             temp_path.unlink()

#     print(f"[OK][scene_description_full] Image key used:       {Path(image_path).name}")
#     print(f"[OK][scene_description_full] Parsed output saved to: {parsed_path}")
#     print(f"[OK][scene_description_full] Run info saved to:      {run_info_path}")
#     print(f"[OK][scene_description_full] Execution time:         {execution_time_seconds:.3f}s")

#     return {
#         "output": parsed_response,
#         "paths": {
#             "artifact": str(parsed_path),
#             "run_info": str(run_info_path),
#         },
#         "execution_time_seconds": execution_time_seconds,
#     }


# def execute_vlm_planning_step(
#     settings,
#     scenario_name: str,
#     scenario_context: dict[str, Any],
#     version: str,
#     model_name: str,
#     loop_timestamp: str,
#     cycle_name: str,
#     cycle_idx: int,
#     cycle_timestamp: str,
#     run_name: str,
#     scene_description_full: Any,
#     scene_version: str,
#     scene_model: str,
#     pipeline_config: dict[str, Any],
# ) -> dict[str, Any]:
#     module_name = "vlm_planning"
#     base_prompt = load_base_prompt(settings, module_name, version)

#     system_prompt = render_prompt(
#         module_name=module_name,
#         base_prompt=base_prompt,
#         scenario_data=scenario_context,
#         scene_description=scene_description_full,
#     )

#     user_text = "Generate the manipulation plan in valid JSON only."

#     prompt_dir = get_module_prompt_dir(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=model_name,
#         cycle_name=cycle_name,
#     )
#     prompt_path = save_prompt_text(prompt_dir / "prompt.txt", system_prompt)

#     result = call_azure_chat_completion(
#         settings=settings,
#         model_name=model_name,
#         system_prompt=system_prompt,
#         user_text=user_text,
#         image_path=None,
#     )

#     parse_ok, parsed_response = try_parse_json(result["raw_response"])
#     if not parse_ok:
#         raise ValueError(
#             f"[vlm_planning] Model response could not be parsed as valid JSON.\n\n"
#             f"Raw response:\n{result['raw_response']}"
#         )

#     dependencies = {
#         "scene_description_full": {
#             "stored_under_module": "scene_description",
#             "artifact_filename": "scene_description_full.json",
#             "prompt_version": scene_version,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": scene_model,
#             "run_name": run_name,
#         }
#     }

#     output_dir = get_module_output_dir(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=result["model_name"],
#         cycle_name=cycle_name,
#     )
#     ensure_dir(output_dir)

#     parsed_path = save_json_file(output_dir / "response_parsed.json", parsed_response)

#     run_info = {
#         "module": "vlm_planning",
#         "execution_mode": "validation_loop",
#         "scenario_name": scenario_name,
#         "prompt_version": version,
#         "loop_timestamp": loop_timestamp,
#         "cycle_name": cycle_name,
#         "cycle_index": cycle_idx,
#         "cycle_timestamp": cycle_timestamp,
#         "run_name": run_name,
#         "model": result["model_name"],
#         "deployment_name": result["deployment_name"],
#         "execution_time_seconds": result["execution_time_seconds"],
#         "timestamp": datetime.now().isoformat(),
#         "scenario_data": scenario_context,
#         "dependencies": dependencies,
#         "pipeline_config": pipeline_config,
#         "response_parsed": parsed_response,
#     }
#     run_info_path = save_json_file(output_dir / "run_info.json", run_info)

#     print(f"[OK][vlm_planning] Prompt saved to:        {prompt_path}")
#     print(f"[OK][vlm_planning] Parsed output saved to: {parsed_path}")
#     print(f"[OK][vlm_planning] Run info saved to:      {run_info_path}")
#     print(f"[OK][vlm_planning] Execution time:         {result['execution_time_seconds']:.3f}s")

#     return {
#         "output": parsed_response,
#         "paths": {
#             "prompt": str(prompt_path),
#             "response_parsed": str(parsed_path),
#             "run_info": str(run_info_path),
#         },
#         "model_name": result["model_name"],
#         "execution_time_seconds": result["execution_time_seconds"],
#     }


# def execute_simultaneous_actions_step(
#     settings,
#     scenario_name: str,
#     scenario_context: dict[str, Any],
#     version: str,
#     model_name: str,
#     loop_timestamp: str,
#     cycle_name: str,
#     cycle_idx: int,
#     cycle_timestamp: str,
#     run_name: str,
#     scene_description_full: Any,
#     sequential_plan: Any,
#     scene_version: str,
#     scene_model: str,
#     plan_version: str,
#     plan_model: str,
#     pipeline_config: dict[str, Any],
# ) -> dict[str, Any]:
#     module_name = "simultaneous_actions"
#     base_prompt = load_base_prompt(settings, module_name, version)

#     system_prompt = render_prompt(
#         module_name=module_name,
#         base_prompt=base_prompt,
#         scenario_data=scenario_context,
#         scene_description=scene_description_full,
#         sequential_plan=sequential_plan,
#     )

#     user_text = "Generate the compact parallel plan in valid JSON only."

#     prompt_dir = get_module_prompt_dir(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=model_name,
#         cycle_name=cycle_name,
#     )
#     prompt_path = save_prompt_text(prompt_dir / "prompt.txt", system_prompt)

#     result = call_azure_chat_completion(
#         settings=settings,
#         model_name=model_name,
#         system_prompt=system_prompt,
#         user_text=user_text,
#         image_path=None,
#     )

#     parse_ok, parsed_response = try_parse_json(result["raw_response"])
#     if not parse_ok:
#         raise ValueError(
#             f"[simultaneous_actions] Model response could not be parsed as valid JSON.\n\n"
#             f"Raw response:\n{result['raw_response']}"
#         )

#     dependencies = {
#         "scene_description_full": {
#             "stored_under_module": "scene_description",
#             "artifact_filename": "scene_description_full.json",
#             "prompt_version": scene_version,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": scene_model,
#             "run_name": run_name,
#         },
#         "vlm_planning": {
#             "prompt_version": plan_version,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": plan_model,
#             "run_name": run_name,
#         },
#     }

#     output_dir = get_module_output_dir(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=result["model_name"],
#         cycle_name=cycle_name,
#     )
#     ensure_dir(output_dir)

#     parsed_path = save_json_file(output_dir / "response_parsed.json", parsed_response)

#     run_info = {
#         "module": "simultaneous_actions",
#         "execution_mode": "validation_loop",
#         "scenario_name": scenario_name,
#         "prompt_version": version,
#         "loop_timestamp": loop_timestamp,
#         "cycle_name": cycle_name,
#         "cycle_index": cycle_idx,
#         "cycle_timestamp": cycle_timestamp,
#         "run_name": run_name,
#         "model": result["model_name"],
#         "deployment_name": result["deployment_name"],
#         "execution_time_seconds": result["execution_time_seconds"],
#         "timestamp": datetime.now().isoformat(),
#         "scenario_data": scenario_context,
#         "dependencies": dependencies,
#         "pipeline_config": pipeline_config,
#         "response_parsed": parsed_response,
#     }
#     run_info_path = save_json_file(output_dir / "run_info.json", run_info)

#     print(f"[OK][simultaneous_actions] Prompt saved to:        {prompt_path}")
#     print(f"[OK][simultaneous_actions] Parsed output saved to: {parsed_path}")
#     print(f"[OK][simultaneous_actions] Run info saved to:      {run_info_path}")
#     print(f"[OK][simultaneous_actions] Execution time:         {result['execution_time_seconds']:.3f}s")

#     return {
#         "output": parsed_response,
#         "paths": {
#             "prompt": str(prompt_path),
#             "response_parsed": str(parsed_path),
#             "run_info": str(run_info_path),
#         },
#         "model_name": result["model_name"],
#         "execution_time_seconds": result["execution_time_seconds"],
#     }


# def execute_validator_step(
#     settings,
#     scenario_name: str,
#     validator_version: str,
#     validator_model: str,
#     loop_timestamp: str,
#     cycle_name: str,
#     cycle_idx: int,
#     cycle_timestamp: str,
#     run_name: str,
#     stage_id: int,
#     condition_kind: str,
#     condition_text: str,
#     image_path: str,
#     scene_version: str,
#     scene_model: str,
#     plan_version: str,
#     plan_model: str,
#     sim_version: str,
#     sim_model: str,
# ) -> dict[str, Any]:
#     stage_name = make_stage_name(stage_id)

#     scene_object_list = load_scene_object_list_from_cycle(
#         settings=settings,
#         scenario_name=scenario_name,
#         scene_version=scene_version,
#         loop_timestamp=loop_timestamp,
#         scene_model=scene_model,
#         cycle_name=cycle_name,
#     )

#     base_prompt = load_base_prompt(settings, "validator", validator_version)
#     system_prompt = render_validator_prompt(
#         base_prompt=base_prompt,
#         condition=condition_text,
#         scene_object_list=scene_object_list,
#     )

#     prompt_dir = get_validator_prompt_dir(
#         settings=settings,
#         scenario=scenario_name,
#         version=validator_version,
#         loop_timestamp=loop_timestamp,
#         model_name=validator_model,
#         cycle_name=cycle_name,
#         stage_name=stage_name,
#         condition_kind=condition_kind,
#     )
#     prompt_path = save_prompt_text(prompt_dir / "prompt.txt", system_prompt)

#     result = call_azure_chat_completion(
#         settings=settings,
#         model_name=validator_model,
#         system_prompt=system_prompt,
#         user_text="Validate the condition and return valid JSON only.",
#         image_path=image_path,
#     )

#     parse_ok, parsed_response = try_parse_json(result["raw_response"])
#     if not parse_ok:
#         raise ValueError(
#             f"[validator:{condition_kind}_{stage_id}] Model response could not be parsed as valid JSON.\n\n"
#             f"Raw response:\n{result['raw_response']}"
#         )

#     validate_validator_response(parsed_response)

#     dependencies = {
#         "scene_description_full": {
#             "stored_under_module": "scene_description",
#             "artifact_filename": "scene_description_full.json",
#             "prompt_version": scene_version,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": scene_model,
#             "run_name": run_name,
#         },
#         "vlm_planning": {
#             "prompt_version": plan_version,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": plan_model,
#             "run_name": run_name,
#         },
#         "simultaneous_actions": {
#             "prompt_version": sim_version,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": sim_model,
#             "run_name": run_name,
#         },
#     }

#     output_dir = get_validator_output_dir(
#         settings=settings,
#         scenario=scenario_name,
#         version=validator_version,
#         loop_timestamp=loop_timestamp,
#         model_name=result["model_name"],
#         cycle_name=cycle_name,
#         stage_name=stage_name,
#         condition_kind=condition_kind,
#     )
#     ensure_dir(output_dir)

#     parsed_path = save_json_file(output_dir / "response_parsed.json", parsed_response)

#     run_info = {
#         "module": "validator",
#         "execution_mode": "validation_loop",
#         "scenario_name": scenario_name,
#         "prompt_version": validator_version,
#         "loop_timestamp": loop_timestamp,
#         "cycle_name": cycle_name,
#         "cycle_index": cycle_idx,
#         "cycle_timestamp": cycle_timestamp,
#         "run_name": run_name,
#         "stage_id": stage_id,
#         "stage_name": stage_name,
#         "condition_kind": condition_kind,
#         "condition_text": condition_text,
#         "model": result["model_name"],
#         "deployment_name": result["deployment_name"],
#         "execution_time_seconds": result["execution_time_seconds"],
#         "timestamp": datetime.now().isoformat(),
#         "image_path": str(Path(image_path).resolve()),
#         "dependencies": dependencies,
#         "response_parsed": parsed_response,
#     }
#     run_info_path = save_json_file(output_dir / "run_info.json", run_info)

#     print(f"[OK][validator:{condition_kind}_{stage_id}] Prompt saved to:        {prompt_path}")
#     print(f"[OK][validator:{condition_kind}_{stage_id}] Parsed output saved to: {parsed_path}")
#     print(f"[OK][validator:{condition_kind}_{stage_id}] Run info saved to:      {run_info_path}")
#     print(f"[OK][validator:{condition_kind}_{stage_id}] Execution time:         {result['execution_time_seconds']:.3f}s")

#     return {
#         "output": parsed_response,
#         "paths": {
#             "prompt": str(prompt_path),
#             "response_parsed": str(parsed_path),
#             "run_info": str(run_info_path),
#         },
#         "model_name": result["model_name"],
#         "execution_time_seconds": result["execution_time_seconds"],
#     }


# # ============================================================
# # SUMMARY HELPERS
# # ============================================================

# def build_run_info(full_summary: dict[str, Any]) -> dict[str, Any]:
#     return {
#         "module": "validation_loop",
#         "scenario_name": full_summary["scenario_name"],
#         "loop_timestamp": full_summary["loop_timestamp"],
#         "timestamp": full_summary["timestamp"],
#         "initial_image_path": full_summary["initial_image_path"],
#         "frames_dir": full_summary["frames_dir"],
#         "poses_by_image_path": full_summary["poses_by_image_path"],
#         "config": full_summary["config"],
#     }


# def build_loop_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
#     return {
#         "module": "validation_loop_summary",
#         "scenario_name": full_summary["scenario_name"],
#         "loop_timestamp": full_summary["loop_timestamp"],
#         "timestamp": full_summary["timestamp"],
#         "config": full_summary["config"],
#         "initial_image_path": full_summary["initial_image_path"],
#         "final_image_path": full_summary.get("final_image_path"),
#         "task_completed": full_summary["task_completed"],
#         "replans_done": full_summary["replans_done"],
#         "total_cycles": len(full_summary["cycles"]),
#         "error": full_summary.get("error"),
#         "cycles": [
#             {
#                 "cycle_name": cycle["cycle_name"],
#                 "cycle_index": cycle["cycle_index"],
#                 "cycle_timestamp": cycle["cycle_timestamp"],
#                 "start_image_path": cycle["start_image_path"],
#                 "start_image_name": cycle["start_image_name"],
#                 "outcome": cycle["outcome"],
#             }
#             for cycle in full_summary["cycles"]
#         ],
#     }


# def build_scene_description_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
#     return {
#         "module": "scene_description_summary",
#         "scenario_name": full_summary["scenario_name"],
#         "loop_timestamp": full_summary["loop_timestamp"],
#         "timestamp": datetime.now().isoformat(),
#         "config": {
#             "scene_description": full_summary["config"]["scene_description"],
#             "scene_description_full": full_summary["config"]["scene_description_full"],
#         },
#         "cycles": [
#             {
#                 "cycle_name": cycle["cycle_name"],
#                 "cycle_index": cycle["cycle_index"],
#                 "cycle_timestamp": cycle["cycle_timestamp"],
#                 "image_path": cycle["start_image_path"],
#                 "image_name": cycle["start_image_name"],
#                 "scene_description_paths": {
#                     "prompt": cycle["scene_description"]["paths"]["prompt"],
#                     "response_parsed": cycle["scene_description"]["paths"]["response_parsed"],
#                     "run_info": cycle["scene_description"]["paths"]["run_info"],
#                     "scene_object_list": cycle["scene_description"]["paths"]["scene_object_list"],
#                     "scene_description_full": cycle["scene_description_full"]["paths"]["artifact"],
#                     "scene_description_full_run_info": cycle["scene_description_full"]["paths"]["run_info"],
#                 },
#                 "scene_description_output": cycle["scene_description"]["output"],
#                 "scene_description_full_output": cycle["scene_description_full"]["output"],
#             }
#             for cycle in full_summary["cycles"]
#             if cycle.get("scene_description") is not None and cycle.get("scene_description_full") is not None
#         ],
#     }


# def build_vlm_planning_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
#     return {
#         "module": "vlm_planning_summary",
#         "scenario_name": full_summary["scenario_name"],
#         "loop_timestamp": full_summary["loop_timestamp"],
#         "timestamp": datetime.now().isoformat(),
#         "config": {
#             "vlm_planning": full_summary["config"]["vlm_planning"],
#         },
#         "cycles": [
#             {
#                 "cycle_name": cycle["cycle_name"],
#                 "cycle_index": cycle["cycle_index"],
#                 "cycle_timestamp": cycle["cycle_timestamp"],
#                 "input_image_path": cycle["start_image_path"],
#                 "input_image_name": cycle["start_image_name"],
#                 "dependencies": {
#                     "scene_description_cycle": cycle["cycle_name"],
#                     "scene_description_full_path": cycle["scene_description_full"]["paths"]["artifact"],
#                 },
#                 "vlm_planning_paths": cycle["vlm_planning"]["paths"],
#                 "vlm_planning_output": cycle["vlm_planning"]["output"],
#             }
#             for cycle in full_summary["cycles"]
#             if cycle.get("vlm_planning") is not None
#         ],
#     }


# def build_simultaneous_actions_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
#     return {
#         "module": "simultaneous_actions_summary",
#         "scenario_name": full_summary["scenario_name"],
#         "loop_timestamp": full_summary["loop_timestamp"],
#         "timestamp": datetime.now().isoformat(),
#         "config": {
#             "simultaneous_actions": full_summary["config"]["simultaneous_actions"],
#         },
#         "cycles": [
#             {
#                 "cycle_name": cycle["cycle_name"],
#                 "cycle_index": cycle["cycle_index"],
#                 "cycle_timestamp": cycle["cycle_timestamp"],
#                 "input_image_path": cycle["start_image_path"],
#                 "input_image_name": cycle["start_image_name"],
#                 "dependencies": {
#                     "scene_description_cycle": cycle["cycle_name"],
#                     "scene_description_full_path": cycle["scene_description_full"]["paths"]["artifact"],
#                     "vlm_planning_cycle": cycle["cycle_name"],
#                     "vlm_planning_path": cycle["vlm_planning"]["paths"]["response_parsed"],
#                 },
#                 "simultaneous_actions_paths": cycle["simultaneous_actions"]["paths"],
#                 "simultaneous_actions_output": cycle["simultaneous_actions"]["output"],
#             }
#             for cycle in full_summary["cycles"]
#             if cycle.get("simultaneous_actions") is not None
#         ],
#     }


# def build_validator_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
#     return {
#         "module": "validator_summary",
#         "scenario_name": full_summary["scenario_name"],
#         "loop_timestamp": full_summary["loop_timestamp"],
#         "timestamp": datetime.now().isoformat(),
#         "config": {
#             "validator": full_summary["config"]["validator"],
#             "max_replans": full_summary["config"]["max_replans"],
#         },
#         "replans_done": full_summary["replans_done"],
#         "task_completed": full_summary["task_completed"],
#         "cycles": [
#             {
#                 "cycle_name": cycle["cycle_name"],
#                 "cycle_index": cycle["cycle_index"],
#                 "cycle_timestamp": cycle["cycle_timestamp"],
#                 "start_image_path": cycle["start_image_path"],
#                 "start_image_name": cycle["start_image_name"],
#                 "outcome": cycle["outcome"],
#                 "stages": cycle["stages"],
#             }
#             for cycle in full_summary["cycles"]
#         ],
#     }


# def build_full_pipeline_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
#     return deepcopy(full_summary)


# def build_cycle_summary(
#     full_summary: dict[str, Any],
#     cycle_record: dict[str, Any],
# ) -> dict[str, Any]:
#     return {
#         "module": "cycle_summary",
#         "scenario_name": full_summary["scenario_name"],
#         "loop_timestamp": full_summary["loop_timestamp"],
#         "cycle_name": cycle_record["cycle_name"],
#         "cycle_index": cycle_record["cycle_index"],
#         "cycle_timestamp": cycle_record["cycle_timestamp"],
#         "start_image_path": cycle_record["start_image_path"],
#         "start_image_name": cycle_record["start_image_name"],
#         "scene_description": cycle_record.get("scene_description"),
#         "scene_description_full": cycle_record.get("scene_description_full"),
#         "vlm_planning": cycle_record.get("vlm_planning"),
#         "simultaneous_actions": cycle_record.get("simultaneous_actions"),
#         "stages": cycle_record["stages"],
#         "outcome": cycle_record["outcome"],
#         "end_image_path": cycle_record.get("end_image_path"),
#         "end_image_name": cycle_record.get("end_image_name"),
#     }


# def save_validation_loop_artifacts(
#     settings,
#     scenario_name: str,
#     loop_timestamp: str,
#     run_info: dict[str, Any],
#     loop_summary: dict[str, Any],
#     scene_description_summary: dict[str, Any],
#     vlm_planning_summary: dict[str, Any],
#     simultaneous_actions_summary: dict[str, Any],
#     validator_summary: dict[str, Any],
#     full_pipeline_summary: dict[str, Any],
# ) -> dict[str, Path]:
#     output_dir = get_validation_loop_dir(settings, scenario_name, loop_timestamp)
#     ensure_dir(output_dir)

#     paths = {
#         "run_info": save_json_file(output_dir / "run_info.json", run_info),
#         "loop_summary": save_json_file(output_dir / "loop_summary.json", loop_summary),
#         "scene_description_summary": save_json_file(
#             output_dir / "scene_description_summary.json",
#             scene_description_summary,
#         ),
#         "vlm_planning_summary": save_json_file(
#             output_dir / "vlm_planning_summary.json",
#             vlm_planning_summary,
#         ),
#         "simultaneous_actions_summary": save_json_file(
#             output_dir / "simultaneous_actions_summary.json",
#             simultaneous_actions_summary,
#         ),
#         "validator_summary": save_json_file(
#             output_dir / "validator_summary.json",
#             validator_summary,
#         ),
#         "full_pipeline_summary": save_json_file(
#             output_dir / "full_pipeline_summary.json",
#             full_pipeline_summary,
#         ),
#     }
#     return paths


# def save_cycle_summary(
#     settings,
#     scenario_name: str,
#     loop_timestamp: str,
#     cycle_name: str,
#     cycle_summary: dict[str, Any],
# ) -> Path:
#     cycle_dir = get_validation_loop_cycle_dir(settings, scenario_name, loop_timestamp, cycle_name)
#     ensure_dir(cycle_dir)
#     return save_json_file(cycle_dir / "cycle_summary.json", cycle_summary)


# # ============================================================
# # MAIN
# # ============================================================

# def main() -> None:
#     parser = build_parser()
#     args = parser.parse_args()

#     validate_args(args)

#     settings = load_settings()
#     scenario_data = load_scenario(settings, args.scenario)

#     poses_by_image_path = resolve_poses_by_image_path(
#         settings=settings,
#         scenario_name=args.scenario,
#         explicit_path=args.poses_by_image_path,
#     )
#     poses_by_image = load_poses_by_image_map(poses_by_image_path)

#     initial_image_path = (
#         str(Path(args.initial_image_path).resolve())
#         if args.initial_image_path is not None
#         else scenario_data.get("image_path_abs")
#     )
#     if not initial_image_path:
#         raise ValueError(
#             "No initial image available. Provide --initial-image-path or set 'image' in scenario.json."
#         )
#     if not Path(initial_image_path).exists():
#         raise FileNotFoundError(f"Initial image not found: {initial_image_path}")

#     frame_paths = list_frame_paths(args.frames_dir)
#     initial_image_resolved = str(Path(initial_image_path).resolve())
#     frame_paths = [p for p in frame_paths if str(Path(p).resolve()) != initial_image_resolved]
#     frame_cursor = 0

#     loop_timestamp = make_experiment_timestamp()
#     run_name = "run_001"

#     current_image = initial_image_path
#     replans_done = 0
#     task_completed = False
#     cycle_idx = 0

#     full_summary: dict[str, Any] = {
#         "module": "full_pipeline_summary",
#         "scenario_name": args.scenario,
#         "loop_timestamp": loop_timestamp,
#         "timestamp": datetime.now().isoformat(),
#         "initial_image_path": str(Path(initial_image_path).resolve()),
#         "frames_dir": str(Path(args.frames_dir).resolve()),
#         "poses_by_image_path": str(poses_by_image_path),
#         "config": build_global_config(args),
#         "replans_done": 0,
#         "task_completed": False,
#         "final_image_path": None,
#         "cycles": [],
#     }

#     while not task_completed:
#         cycle_idx += 1
#         cycle_name = make_cycle_name(cycle_idx)
#         cycle_timestamp = make_experiment_timestamp()

#         print("\n======================================================")
#         print(f"VALIDATION LOOP CYCLE STARTED | cycle={cycle_idx} | {cycle_name}")
#         print(f"Current image:   {current_image}")
#         print(f"Loop ts:         {loop_timestamp}")
#         print(f"Cycle ts meta:   {cycle_timestamp}")
#         print("======================================================")

#         scenario_context = make_scenario_context(
#             scenario_data=scenario_data,
#             image_path=current_image,
#         )

#         pipeline_config = build_cycle_config(
#             args=args,
#             cycle_timestamp=cycle_timestamp,
#             cycle_name=cycle_name,
#             cycle_idx=cycle_idx,
#         )

#         cycle_record: dict[str, Any] = {
#             "cycle_name": cycle_name,
#             "cycle_index": cycle_idx,
#             "cycle_timestamp": cycle_timestamp,
#             "run_name": run_name,
#             "start_image_path": str(Path(current_image).resolve()),
#             "start_image_name": Path(current_image).name,
#             "scene_description": None,
#             "scene_description_full": None,
#             "vlm_planning": None,
#             "simultaneous_actions": None,
#             "stages": [],
#             "outcome": None,
#             "end_image_path": None,
#             "end_image_name": None,
#         }

#         cycle_error = False

#         try:
#             scene_description_artifact = execute_scene_description_step(
#                 settings=settings,
#                 scenario_name=args.scenario,
#                 scenario_context=scenario_context,
#                 version=args.scene_v,
#                 model_name=args.scene_model,
#                 loop_timestamp=loop_timestamp,
#                 cycle_name=cycle_name,
#                 cycle_idx=cycle_idx,
#                 cycle_timestamp=cycle_timestamp,
#                 run_name=run_name,
#                 pipeline_config=pipeline_config,
#                 image_path=current_image,
#             )
#             cycle_record["scene_description"] = scene_description_artifact

#             print("\n[scene_description] Parsed JSON:")
#             print(json.dumps(scene_description_artifact["output"], indent=2, ensure_ascii=False))

#             scene_description_full_artifact = execute_scene_description_full_step(
#                 settings=settings,
#                 scenario_name=args.scenario,
#                 scenario_context=scenario_context,
#                 version=args.scene_v,
#                 model_name=args.scene_model,
#                 loop_timestamp=loop_timestamp,
#                 cycle_name=cycle_name,
#                 cycle_idx=cycle_idx,
#                 cycle_timestamp=cycle_timestamp,
#                 run_name=run_name,
#                 scene_description=scene_description_artifact["output"],
#                 pipeline_config=pipeline_config,
#                 image_path=current_image,
#                 poses_by_image=poses_by_image,
#                 safety_threshold=args.grounding_safety_threshold,
#                 include_debug_mapping=args.grounding_debug_mapping,
#             )
#             cycle_record["scene_description_full"] = scene_description_full_artifact

#             print("\n[scene_description_full] Parsed JSON:")
#             print(json.dumps(scene_description_full_artifact["output"], indent=2, ensure_ascii=False))

#             sequential_plan_artifact = execute_vlm_planning_step(
#                 settings=settings,
#                 scenario_name=args.scenario,
#                 scenario_context=scenario_context,
#                 version=args.plan_v,
#                 model_name=args.plan_model,
#                 loop_timestamp=loop_timestamp,
#                 cycle_name=cycle_name,
#                 cycle_idx=cycle_idx,
#                 cycle_timestamp=cycle_timestamp,
#                 run_name=run_name,
#                 scene_description_full=scene_description_full_artifact["output"],
#                 scene_version=args.scene_v,
#                 scene_model=args.scene_model,
#                 pipeline_config=pipeline_config,
#             )
#             cycle_record["vlm_planning"] = sequential_plan_artifact

#             print("\n[vlm_planning] Parsed JSON:")
#             print(json.dumps(sequential_plan_artifact["output"], indent=2, ensure_ascii=False))

#             simultaneous_actions_artifact = execute_simultaneous_actions_step(
#                 settings=settings,
#                 scenario_name=args.scenario,
#                 scenario_context=scenario_context,
#                 version=args.sim_v,
#                 model_name=args.sim_model,
#                 loop_timestamp=loop_timestamp,
#                 cycle_name=cycle_name,
#                 cycle_idx=cycle_idx,
#                 cycle_timestamp=cycle_timestamp,
#                 run_name=run_name,
#                 scene_description_full=scene_description_full_artifact["output"],
#                 sequential_plan=sequential_plan_artifact["output"],
#                 scene_version=args.scene_v,
#                 scene_model=args.scene_model,
#                 plan_version=args.plan_v,
#                 plan_model=args.plan_model,
#                 pipeline_config=pipeline_config,
#             )
#             cycle_record["simultaneous_actions"] = simultaneous_actions_artifact

#             print("\n[simultaneous_actions] Parsed JSON:")
#             print(json.dumps(simultaneous_actions_artifact["output"], indent=2, ensure_ascii=False))

#             stages = extract_stages(simultaneous_actions_artifact["output"])
#             all_stages_succeeded = True

#             for stage in stages:
#                 stage_id = stage["Stage_id"]
#                 stage_name = make_stage_name(stage_id)
#                 pre_condition = stage["Precondition"]
#                 post_condition = stage["Postcondition"]

#                 stage_record: dict[str, Any] = {
#                     "stage_id": stage_id,
#                     "stage_name": stage_name,
#                     "precondition": pre_condition,
#                     "postcondition": post_condition,
#                     "pre_image_path": str(Path(current_image).resolve()),
#                     "pre_image_name": Path(current_image).name,
#                     "post_image_path": None,
#                     "post_image_name": None,
#                     "pre_validation": None,
#                     "post_validation": None,
#                     "validator_paths": {
#                         "pre": None,
#                         "post": None,
#                     },
#                     "next_image_path": None,
#                     "next_image_name": None,
#                 }

#                 # --------------------------------------------------
#                 # PRE VALIDATION
#                 # --------------------------------------------------
#                 print(f"\n[LOOP] Stage {stage_id} PRE")
#                 print(f"[LOOP] PRE image:      {current_image}")
#                 print(f"[LOOP] PRE condition:  {pre_condition}")

#                 print_pose_dict_for_image(
#                     poses_by_image=poses_by_image,
#                     image_path=current_image,
#                     label=f"validator-pre-stage-{stage_id}",
#                 )

#                 pre_artifact = execute_validator_step(
#                     settings=settings,
#                     scenario_name=args.scenario,
#                     validator_version=args.validator_v,
#                     validator_model=args.validator_model,
#                     loop_timestamp=loop_timestamp,
#                     cycle_name=cycle_name,
#                     cycle_idx=cycle_idx,
#                     cycle_timestamp=cycle_timestamp,
#                     run_name=run_name,
#                     stage_id=stage_id,
#                     condition_kind="pre",
#                     condition_text=pre_condition,
#                     image_path=current_image,
#                     scene_version=args.scene_v,
#                     scene_model=args.scene_model,
#                     plan_version=args.plan_v,
#                     plan_model=args.plan_model,
#                     sim_version=args.sim_v,
#                     sim_model=args.sim_model,
#                 )
#                 pre_response = pre_artifact["output"]

#                 print(f"\n[PRE validator:pre_{stage_id}] Parsed JSON:")
#                 print(json.dumps(pre_response, indent=2, ensure_ascii=False))
#                 print(f"[LOOP] PRE result:     {pre_response['result']}")
#                 print(f"[LOOP] PRE reason:     {pre_response['reason']}")

#                 stage_record["pre_validation"] = pre_response
#                 stage_record["validator_paths"]["pre"] = pre_artifact["paths"]

#                 if pre_response["result"] == "non_matching":
#                     print(f"[LOOP] Precondition failed at stage {stage_id}. Replanning from same image.")
#                     cycle_record["stages"].append(stage_record)
#                     cycle_record["outcome"] = f"replan_on_pre_stage_{stage_id}"
#                     cycle_record["end_image_path"] = str(Path(current_image).resolve())
#                     cycle_record["end_image_name"] = Path(current_image).name

#                     if replans_done >= args.max_replans:
#                         raise RuntimeError(
#                             f"Maximum number of replans reached ({args.max_replans})."
#                         )

#                     replans_done += 1
#                     full_summary["replans_done"] = replans_done

#                     all_stages_succeeded = False
#                     break

#                 # --------------------------------------------------
#                 # DEPLOY PLACEHOLDER / OFFLINE NEXT IMAGE
#                 # --------------------------------------------------
#                 if frame_cursor >= len(frame_paths):
#                     raise RuntimeError(
#                         "No more images available in frames-dir for simulated deploy progression."
#                     )

#                 next_image = frame_paths[frame_cursor]
#                 frame_cursor += 1

#                 stage_record["next_image_path"] = str(Path(next_image).resolve())
#                 stage_record["next_image_name"] = Path(next_image).name
#                 stage_record["post_image_path"] = str(Path(next_image).resolve())
#                 stage_record["post_image_name"] = Path(next_image).name

#                 print(f"\n[LOOP] Stage {stage_id} simulated deploy")
#                 print(f"[LOOP] NEXT image:     {next_image}")

#                 # --------------------------------------------------
#                 # POST VALIDATION
#                 # --------------------------------------------------
#                 print(f"\n[LOOP] Stage {stage_id} POST")
#                 print(f"[LOOP] POST image:     {next_image}")
#                 print(f"[LOOP] POST condition: {post_condition}")

#                 print_pose_dict_for_image(
#                     poses_by_image=poses_by_image,
#                     image_path=next_image,
#                     label=f"validator-post-stage-{stage_id}",
#                 )

#                 post_artifact = execute_validator_step(
#                     settings=settings,
#                     scenario_name=args.scenario,
#                     validator_version=args.validator_v,
#                     validator_model=args.validator_model,
#                     loop_timestamp=loop_timestamp,
#                     cycle_name=cycle_name,
#                     cycle_idx=cycle_idx,
#                     cycle_timestamp=cycle_timestamp,
#                     run_name=run_name,
#                     stage_id=stage_id,
#                     condition_kind="post",
#                     condition_text=post_condition,
#                     image_path=next_image,
#                     scene_version=args.scene_v,
#                     scene_model=args.scene_model,
#                     plan_version=args.plan_v,
#                     plan_model=args.plan_model,
#                     sim_version=args.sim_v,
#                     sim_model=args.sim_model,
#                 )
#                 post_response = post_artifact["output"]

#                 print(f"\n[POST validator:post_{stage_id}] Parsed JSON:")
#                 print(json.dumps(post_response, indent=2, ensure_ascii=False))
#                 print(f"[LOOP] POST result:    {post_response['result']}")
#                 print(f"[LOOP] POST reason:    {post_response['reason']}")

#                 stage_record["post_validation"] = post_response
#                 stage_record["validator_paths"]["post"] = post_artifact["paths"]
#                 cycle_record["stages"].append(stage_record)

#                 if post_response["result"] == "non_matching":
#                     print(f"[LOOP] Postcondition failed at stage {stage_id}. Replanning from next image.")
#                     cycle_record["outcome"] = f"replan_on_post_stage_{stage_id}"
#                     cycle_record["end_image_path"] = str(Path(next_image).resolve())
#                     cycle_record["end_image_name"] = Path(next_image).name
#                     current_image = next_image

#                     if replans_done >= args.max_replans:
#                         raise RuntimeError(
#                             f"Maximum number of replans reached ({args.max_replans})."
#                         )

#                     replans_done += 1
#                     full_summary["replans_done"] = replans_done

#                     all_stages_succeeded = False
#                     break

#                 current_image = next_image

#             if all_stages_succeeded:
#                 cycle_record["outcome"] = "task_completed"
#                 cycle_record["end_image_path"] = str(Path(current_image).resolve())
#                 cycle_record["end_image_name"] = Path(current_image).name
#                 task_completed = True
#                 full_summary["task_completed"] = True
#                 full_summary["final_image_path"] = str(Path(current_image).resolve())

#                 print("\n======================================================")
#                 print("[LOOP] TASK COMPLETED SUCCESSFULLY")
#                 print("======================================================")

#         except Exception as exc:
#             cycle_record["outcome"] = f"cycle_error: {exc}"
#             cycle_record["end_image_path"] = str(Path(current_image).resolve())
#             cycle_record["end_image_name"] = Path(current_image).name
#             full_summary["task_completed"] = False
#             full_summary["error"] = str(exc)
#             cycle_error = True

#         full_summary["cycles"].append(cycle_record)

#         cycle_summary = build_cycle_summary(full_summary, cycle_record)
#         cycle_summary_path = save_cycle_summary(
#             settings=settings,
#             scenario_name=args.scenario,
#             loop_timestamp=loop_timestamp,
#             cycle_name=cycle_name,
#             cycle_summary=cycle_summary,
#         )
#         print(f"[OK][validation_loop] Cycle summary saved to: {cycle_summary_path}")

#         if cycle_error:
#             break

#     run_info = build_run_info(full_summary)
#     loop_summary = build_loop_summary(full_summary)
#     scene_description_summary = build_scene_description_summary(full_summary)
#     vlm_planning_summary = build_vlm_planning_summary(full_summary)
#     simultaneous_actions_summary = build_simultaneous_actions_summary(full_summary)
#     validator_summary = build_validator_summary(full_summary)
#     full_pipeline_summary = build_full_pipeline_summary(full_summary)

#     summary_paths = save_validation_loop_artifacts(
#         settings=settings,
#         scenario_name=args.scenario,
#         loop_timestamp=loop_timestamp,
#         run_info=run_info,
#         loop_summary=loop_summary,
#         scene_description_summary=scene_description_summary,
#         vlm_planning_summary=vlm_planning_summary,
#         simultaneous_actions_summary=simultaneous_actions_summary,
#         validator_summary=validator_summary,
#         full_pipeline_summary=full_pipeline_summary,
#     )

#     print("\n======================================================")
#     print("VALIDATION LOOP COMPLETED")
#     print(f"Scenario:                 {args.scenario}")
#     print(f"Loop timestamp:           {loop_timestamp}")
#     print(f"Task completed:           {full_summary['task_completed']}")
#     print(f"Replans done:             {full_summary['replans_done']}")
#     print(f"Run info saved:           {summary_paths['run_info']}")
#     print(f"Loop summary saved:       {summary_paths['loop_summary']}")
#     print(f"Scene summary saved:      {summary_paths['scene_description_summary']}")
#     print(f"Planning summary saved:   {summary_paths['vlm_planning_summary']}")
#     print(f"Sim-actions summary saved:{summary_paths['simultaneous_actions_summary']}")
#     print(f"Validator summary saved:  {summary_paths['validator_summary']}")
#     print(f"Full summary saved:       {summary_paths['full_pipeline_summary']}")
#     print("======================================================")


# if __name__ == "__main__":
#     main()

















# # from __future__ import annotations

# # import argparse
# # import json
# # import tempfile
# # import time
# # from copy import deepcopy
# # from datetime import datetime
# # from pathlib import Path
# # from typing import Any

# # from settings import load_settings
# # from scenario_loader import load_scenario
# # from azure_openai_client import call_azure_chat_completion
# # from build_scene_object_list import build_scene_object_list_from_run
# # from scene_enrichment import enrich_scene
# # from utils import (
# #     load_base_prompt,
# #     make_experiment_timestamp,
# #     render_prompt,
# #     save_module_outputs,
# #     save_rendered_prompt,
# #     save_scene_description_full_artifact,
# #     try_parse_json,
# #     write_json,
# #     read_json,
# # )

# # SUPPORTED_MODELS = ["o3", "gpt-5.2"]
# # IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# # # ============================================================
# # # PARSER
# # # ============================================================

# # def build_parser() -> argparse.ArgumentParser:
# #     parser = argparse.ArgumentParser(
# #         description=(
# #             "Run the offline validation loop: pipeline -> stage pre/post validation -> "
# #             "replanning on failure."
# #         )
# #     )

# #     parser.add_argument("--scenario", type=str, required=True)

# #     parser.add_argument(
# #         "--initial-image-path",
# #         type=str,
# #         default=None,
# #         help="Optional explicit initial image path. If omitted, uses scenario.json image.",
# #     )

# #     parser.add_argument(
# #         "--frames-dir",
# #         type=str,
# #         required=True,
# #         help=(
# #             "Directory containing the sequence of post-deploy images in chronological order. "
# #             "These images are consumed one-by-one when a stage is executed."
# #         ),
# #     )

# #     parser.add_argument(
# #         "--poses-by-image-path",
# #         type=str,
# #         default=None,
# #         help=(
# #             "Optional path to a JSON mapping image filename -> pose dictionary. "
# #             "If omitted, defaults to scenarios/<scenario>/poses_by_image.json"
# #         ),
# #     )

# #     parser.add_argument("--scene-v", type=str, required=True)
# #     parser.add_argument("--plan-v", type=str, required=True)
# #     parser.add_argument("--sim-v", type=str, required=True)
# #     parser.add_argument("--validator-v", type=str, required=True)

# #     parser.add_argument("--scene-model", type=str, required=True, choices=SUPPORTED_MODELS)
# #     parser.add_argument("--plan-model", type=str, required=True, choices=SUPPORTED_MODELS)
# #     parser.add_argument("--sim-model", type=str, required=True, choices=SUPPORTED_MODELS)
# #     parser.add_argument("--validator-model", type=str, required=True, choices=SUPPORTED_MODELS)

# #     parser.add_argument(
# #         "--max-replans",
# #         type=int,
# #         default=10,
# #         help="Maximum number of replanning cycles allowed before stopping.",
# #     )

# #     parser.add_argument(
# #         "--grounding-safety-threshold",
# #         type=float,
# #         default=0.21,
# #         help="Safety threshold used by scene enrichment to compute accessibility.",
# #     )
# #     parser.add_argument(
# #         "--grounding-debug-mapping",
# #         action="store_true",
# #         help="Store the internal VLM-to-Gazebo mapping inside scene_description_full.json under _debug.",
# #     )

# #     return parser


# # # ============================================================
# # # HELPERS
# # # ============================================================

# # def ensure_dir(path: Path) -> Path:
# #     path.mkdir(parents=True, exist_ok=True)
# #     return path


# # def validate_args(args: argparse.Namespace) -> None:
# #     if args.max_replans < 0:
# #         raise ValueError("--max-replans must be >= 0")

# #     frames_dir = Path(args.frames_dir)
# #     if not frames_dir.exists():
# #         raise FileNotFoundError(f"frames-dir not found: {frames_dir}")
# #     if not frames_dir.is_dir():
# #         raise ValueError(f"--frames-dir must be a directory: {frames_dir}")

# #     if args.poses_by_image_path is not None:
# #         poses_path = Path(args.poses_by_image_path)
# #         if not poses_path.exists():
# #             raise FileNotFoundError(f"poses-by-image-path not found: {poses_path}")


# # def list_frame_paths(frames_dir: str | Path) -> list[str]:
# #     frames_dir = Path(frames_dir)
# #     frames = sorted(
# #         [
# #             p
# #             for p in frames_dir.iterdir()
# #             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
# #         ]
# #     )

# #     if not frames:
# #         raise ValueError(f"No image files found inside frames-dir: {frames_dir}")

# #     return [str(p.resolve()) for p in frames]



# # def print_pose_dict_for_image(
# #     poses_by_image: dict[str, dict[str, list[float]]],
# #     image_path: str,
# #     label: str,
# # ) -> None:
# #     image_name = Path(image_path).name

# #     if image_name not in poses_by_image:
# #         print(f"\n[DEBUG][{label}] No poses found for image: {image_name}")
# #         return

# #     pose_dict = poses_by_image[image_name]

# #     print(f"\n[DEBUG][{label}] Image path: {image_path}")
# #     print(f"[DEBUG][{label}] Image key:  {image_name}")
# #     print(f"[DEBUG][{label}] Pose entries:")

# #     for obj_name, pose in pose_dict.items():
# #         print(f"  - {obj_name}: {pose}")


# # def make_scenario_context(
# #     scenario_data: dict[str, Any],
# #     image_path: str,
# # ) -> dict[str, Any]:
# #     """
# #     Copy scenario_data and override image fields so saved run_info reflects
# #     the actual image used for this cycle.
# #     """
# #     ctx = deepcopy(scenario_data)
# #     ctx["image"] = Path(image_path).name
# #     ctx["image_path_abs"] = str(Path(image_path).resolve())
# #     return ctx


# # def resolve_poses_by_image_path(
# #     settings,
# #     scenario_name: str,
# #     explicit_path: str | None,
# # ) -> Path:
# #     if explicit_path is not None:
# #         path = Path(explicit_path).resolve()
# #     else:
# #         path = (
# #             settings.project_root
# #             / "scenarios"
# #             / scenario_name
# #             / "poses_by_image.json"
# #         ).resolve()

# #     if not path.exists():
# #         raise FileNotFoundError(f"poses_by_image.json not found: {path}")

# #     return path


# # def load_poses_by_image_map(path: str | Path) -> dict[str, dict[str, list[float]]]:
# #     data = read_json(path)
# #     if not isinstance(data, dict):
# #         raise ValueError(
# #             f"poses_by_image mapping must be a JSON object. Found: {type(data).__name__}"
# #         )

# #     validated: dict[str, dict[str, list[float]]] = {}

# #     for image_name, pose_dict in data.items():
# #         if not isinstance(image_name, str):
# #             raise ValueError("Each poses_by_image key must be an image filename string.")

# #         if not isinstance(pose_dict, dict):
# #             raise ValueError(
# #                 f"poses_by_image['{image_name}'] must be an object mapping object names to [x, y, z]."
# #             )

# #         cleaned_pose_dict: dict[str, list[float]] = {}
# #         for obj_name, pose in pose_dict.items():
# #             if not isinstance(obj_name, str):
# #                 raise ValueError(
# #                     f"poses_by_image['{image_name}'] contains a non-string object name."
# #                 )
# #             if not isinstance(pose, list) or len(pose) != 3:
# #                 raise ValueError(
# #                     f"poses_by_image['{image_name}']['{obj_name}'] must be a list of 3 numeric values."
# #                 )
# #             if not all(isinstance(v, (int, float)) for v in pose):
# #                 raise ValueError(
# #                     f"poses_by_image['{image_name}']['{obj_name}'] must contain only numeric values."
# #                 )
# #             cleaned_pose_dict[obj_name] = [float(v) for v in pose]

# #         validated[image_name] = cleaned_pose_dict

# #     return validated


# # def get_pose_dict_for_image(
# #     poses_by_image: dict[str, dict[str, list[float]]],
# #     image_path: str,
# # ) -> dict[str, list[float]]:
# #     image_name = Path(image_path).name

# #     if image_name not in poses_by_image:
# #         available = ", ".join(sorted(poses_by_image.keys())[:10])
# #         raise KeyError(
# #             f"No pose entry found for image '{image_name}' in poses_by_image mapping. "
# #             f"Available examples: {available}"
# #         )

# #     return poses_by_image[image_name]


# # def write_temp_pose_file(pose_dict: dict[str, list[float]]) -> str:
# #     with tempfile.NamedTemporaryFile(
# #         mode="w",
# #         suffix=".json",
# #         encoding="utf-8",
# #         delete=False,
# #     ) as tmp:
# #         json.dump(pose_dict, tmp, indent=2, ensure_ascii=False)
# #         return tmp.name


# # def load_scene_object_list_from_cycle(
# #     settings,
# #     scenario_name: str,
# #     scene_version: str,
# #     pipeline_timestamp: str,
# #     scene_model: str,
# #     run_name: str,
# # ) -> dict[str, Any]:
# #     path = (
# #         settings.project_root
# #         / "outputs"
# #         / "scene_description"
# #         / scenario_name
# #         / scene_version
# #         / pipeline_timestamp
# #         / scene_model
# #         / run_name
# #         / "scene_object_list.json"
# #     )

# #     if not path.exists():
# #         raise FileNotFoundError(f"scene_object_list.json not found: {path}")

# #     return read_json(path)


# # def extract_stages(compact_parallel_plan: Any) -> list[dict[str, Any]]:
# #     if not isinstance(compact_parallel_plan, list):
# #         raise ValueError("simultaneous_actions output must be a JSON array of stages.")

# #     stages: list[dict[str, Any]] = []
# #     for idx, stage in enumerate(compact_parallel_plan):
# #         if not isinstance(stage, dict):
# #             raise ValueError(f"Stage at index {idx} is not a JSON object.")

# #         stage_id = stage.get("Stage_id")
# #         precondition = stage.get("Precondition")
# #         postcondition = stage.get("Postcondition")

# #         if not isinstance(stage_id, int):
# #             raise ValueError(f"Stage at index {idx} has invalid or missing 'Stage_id'.")
# #         if not isinstance(precondition, str) or not precondition.strip():
# #             raise ValueError(f"Stage {stage_id} has invalid or missing 'Precondition'.")
# #         if not isinstance(postcondition, str) or not postcondition.strip():
# #             raise ValueError(f"Stage {stage_id} has invalid or missing 'Postcondition'.")

# #         stages.append(
# #             {
# #                 "Stage_id": stage_id,
# #                 "Precondition": precondition,
# #                 "Postcondition": postcondition,
# #             }
# #         )

# #     return stages


# # def render_validator_prompt(
# #     base_prompt: str,
# #     condition: str,
# #     scene_object_list: dict[str, Any],
# # ) -> str:
# #     scene_object_list_str = json.dumps(scene_object_list, indent=2, ensure_ascii=False)

# #     prompt = base_prompt
# #     prompt = prompt.replace("<CONDITION>", condition)
# #     prompt = prompt.replace("<SCENE_OBJECT_LIST>", scene_object_list_str)

# #     return prompt.strip()


# # def validate_validator_response(parsed_response: Any) -> None:
# #     if not isinstance(parsed_response, dict):
# #         raise ValueError("Validator output must be a JSON object.")

# #     result = parsed_response.get("result")
# #     reason = parsed_response.get("reason")

# #     if result not in {"matching", "non_matching"}:
# #         raise ValueError(
# #             "Validator output field 'result' must be either 'matching' or 'non_matching'."
# #         )

# #     if not isinstance(reason, str) or not reason.strip():
# #         raise ValueError("Validator output field 'reason' must be a non-empty string.")


# # def get_validator_prompt_dir(
# #     settings,
# #     scenario: str,
# #     version: str,
# #     upstream_timestamp: str,
# #     model_name: str,
# #     run_name: str,
# #     condition_name: str,
# # ) -> Path:
# #     return (
# #         settings.project_root
# #         / "prompts_scenarios"
# #         / "validator"
# #         / scenario
# #         / version
# #         / upstream_timestamp
# #         / model_name
# #         / run_name
# #         / condition_name
# #     )


# # def get_validator_output_dir(
# #     settings,
# #     scenario: str,
# #     version: str,
# #     upstream_timestamp: str,
# #     model_name: str,
# #     run_name: str,
# #     condition_name: str,
# # ) -> Path:
# #     return (
# #         settings.project_root
# #         / "outputs"
# #         / "validator"
# #         / scenario
# #         / version
# #         / upstream_timestamp
# #         / model_name
# #         / run_name
# #         / condition_name
# #     )


# # def save_validator_prompt(
# #     settings,
# #     scenario: str,
# #     version: str,
# #     upstream_timestamp: str,
# #     model_name: str,
# #     run_name: str,
# #     condition_name: str,
# #     prompt_text: str,
# # ) -> Path:
# #     prompt_dir = get_validator_prompt_dir(
# #         settings=settings,
# #         scenario=scenario,
# #         version=version,
# #         upstream_timestamp=upstream_timestamp,
# #         model_name=model_name,
# #         run_name=run_name,
# #         condition_name=condition_name,
# #     )
# #     ensure_dir(prompt_dir)

# #     prompt_path = prompt_dir / "prompt.txt"
# #     prompt_path.write_text(prompt_text, encoding="utf-8")
# #     return prompt_path


# # def save_validator_outputs(
# #     settings,
# #     scenario: str,
# #     version: str,
# #     upstream_timestamp: str,
# #     model_name: str,
# #     run_name: str,
# #     condition_name: str,
# #     deployment_name: str,
# #     execution_time_seconds: float,
# #     image_path: str,
# #     condition_text: str,
# #     parsed_response: dict[str, Any],
# #     dependencies: dict[str, Any],
# # ) -> tuple[Path, Path]:
# #     output_dir = get_validator_output_dir(
# #         settings=settings,
# #         scenario=scenario,
# #         version=version,
# #         upstream_timestamp=upstream_timestamp,
# #         model_name=model_name,
# #         run_name=run_name,
# #         condition_name=condition_name,
# #     )
# #     ensure_dir(output_dir)

# #     parsed_path = output_dir / "response_parsed.json"
# #     run_info_path = output_dir / "run_info.json"

# #     write_json(parsed_path, parsed_response)

# #     run_info = {
# #         "module": "validator",
# #         "execution_mode": "validation_loop",
# #         "scenario_name": scenario,
# #         "prompt_version": version,
# #         "experiment_timestamp": upstream_timestamp,
# #         "run_name": run_name,
# #         "condition_name": condition_name,
# #         "condition_text": condition_text,
# #         "model": model_name,
# #         "deployment_name": deployment_name,
# #         "execution_time_seconds": execution_time_seconds,
# #         "timestamp": datetime.now().isoformat(),
# #         "image_path": str(Path(image_path).resolve()),
# #         "dependencies": dependencies,
# #         "response_parsed": parsed_response,
# #     }

# #     write_json(run_info_path, run_info)

# #     return parsed_path, run_info_path


# # # ============================================================
# # # MODULE EXECUTION HELPERS (pipeline blocks)
# # # ============================================================

# # def execute_scene_description_step(
# #     settings,
# #     scenario_name: str,
# #     scenario_context: dict[str, Any],
# #     version: str,
# #     model_name: str,
# #     experiment_timestamp: str,
# #     run_name: str,
# #     pipeline_config: dict[str, Any],
# #     image_path: str,
# # ) -> dict[str, Any]:
# #     module_name = "scene_description"
# #     base_prompt = load_base_prompt(settings, module_name, version)

# #     system_prompt = base_prompt
# #     user_text = "Analyze the scene and return the structured JSON output."

# #     save_rendered_prompt(
# #         settings=settings,
# #         module_name=module_name,
# #         scenario_name=scenario_name,
# #         version=version,
# #         experiment_timestamp=experiment_timestamp,
# #         model_name=model_name,
# #         run_name=run_name,
# #         prompt_text=system_prompt,
# #     )

# #     result = call_azure_chat_completion(
# #         settings=settings,
# #         model_name=model_name,
# #         system_prompt=system_prompt,
# #         user_text=user_text,
# #         image_path=image_path,
# #     )

# #     parse_ok, parsed_response = try_parse_json(result["raw_response"])
# #     if not parse_ok:
# #         raise ValueError(
# #             f"[scene_description] Model response could not be parsed as valid JSON.\n\n"
# #             f"Raw response:\n{result['raw_response']}"
# #         )

# #     parsed_path, run_info_path = save_module_outputs(
# #         settings=settings,
# #         module_name=module_name,
# #         scenario_name=scenario_name,
# #         version=version,
# #         experiment_timestamp=experiment_timestamp,
# #         model_name=result["model_name"],
# #         run_name=run_name,
# #         deployment_name=result["deployment_name"],
# #         execution_time_seconds=result["execution_time_seconds"],
# #         scenario_data=scenario_context,
# #         parsed_response=parsed_response,
# #         execution_mode="validation_loop",
# #         dependencies=None,
# #         pipeline_config=pipeline_config,
# #     )

# #     scene_object_list_path = build_scene_object_list_from_run(
# #         scenario=scenario_name,
# #         version=version,
# #         experiment_timestamp=experiment_timestamp,
# #         model=result["model_name"],
# #         run_name=run_name,
# #     )

# #     print(f"[OK][scene_description] Parsed output saved to: {parsed_path}")
# #     print(f"[OK][scene_description] Run info saved to:      {run_info_path}")
# #     print(f"[OK][scene_description] Scene object list saved to: {scene_object_list_path}")
# #     print(f"[OK][scene_description] Execution time:         {result['execution_time_seconds']:.3f}s")

# #     return parsed_response


# # def execute_scene_description_full_step(
# #     settings,
# #     scenario_name: str,
# #     scenario_context: dict[str, Any],
# #     version: str,
# #     model_name: str,
# #     experiment_timestamp: str,
# #     run_name: str,
# #     scene_description: Any,
# #     pipeline_config: dict[str, Any],
# #     image_path: str,
# #     poses_by_image: dict[str, dict[str, list[float]]],
# #     safety_threshold: float,
# #     include_debug_mapping: bool,
# # ) -> dict[str, Any]:
# #     pose_dict = get_pose_dict_for_image(poses_by_image, image_path)
# #     temp_pose_file = write_temp_pose_file(pose_dict)

# #     try:
# #         start_time = time.perf_counter()

# #         parsed_response = enrich_scene(
# #             input_data=scene_description,
# #             safety_threshold=safety_threshold,
# #             pose_source="static",
# #             pose_file=temp_pose_file,
# #             include_debug_mapping=include_debug_mapping,
# #         )

# #         execution_time_seconds = time.perf_counter() - start_time

# #     finally:
# #         temp_path = Path(temp_pose_file)
# #         if temp_path.exists():
# #             temp_path.unlink()

# #     dependencies = {
# #         "scene_description": {
# #             "prompt_version": version,
# #             "experiment_timestamp": experiment_timestamp,
# #             "model": model_name,
# #             "run_name": run_name,
# #         }
# #     }

# #     parsed_path, run_info_path = save_scene_description_full_artifact(
# #         settings=settings,
# #         scenario_name=scenario_name,
# #         version=version,
# #         experiment_timestamp=experiment_timestamp,
# #         model_name=model_name,
# #         run_name=run_name,
# #         parsed_response=parsed_response,
# #         scenario_data=scenario_context,
# #         execution_time_seconds=execution_time_seconds,
# #         dependencies=dependencies,
# #         pipeline_config=pipeline_config,
# #         pose_file=temp_pose_file,
# #         safety_threshold=safety_threshold,
# #         include_debug_mapping=include_debug_mapping,
# #         execution_mode="validation_loop_side_artifact",
# #     )

# #     print(f"[OK][scene_description_full] Image key used:       {Path(image_path).name}")
# #     print(f"[OK][scene_description_full] Parsed output saved to: {parsed_path}")
# #     print(f"[OK][scene_description_full] Run info saved to:      {run_info_path}")
# #     print(f"[OK][scene_description_full] Execution time:         {execution_time_seconds:.3f}s")

# #     return parsed_response


# # def execute_vlm_planning_step(
# #     settings,
# #     scenario_name: str,
# #     scenario_context: dict[str, Any],
# #     version: str,
# #     model_name: str,
# #     experiment_timestamp: str,
# #     run_name: str,
# #     scene_description_full: Any,
# #     scene_version: str,
# #     scene_model: str,
# #     pipeline_config: dict[str, Any],
# # ) -> Any:
# #     module_name = "vlm_planning"
# #     base_prompt = load_base_prompt(settings, module_name, version)

# #     system_prompt = render_prompt(
# #         module_name=module_name,
# #         base_prompt=base_prompt,
# #         scenario_data=scenario_context,
# #         scene_description=scene_description_full,
# #     )

# #     user_text = "Generate the manipulation plan in valid JSON only."

# #     save_rendered_prompt(
# #         settings=settings,
# #         module_name=module_name,
# #         scenario_name=scenario_name,
# #         version=version,
# #         experiment_timestamp=experiment_timestamp,
# #         model_name=model_name,
# #         run_name=run_name,
# #         prompt_text=system_prompt,
# #     )

# #     result = call_azure_chat_completion(
# #         settings=settings,
# #         model_name=model_name,
# #         system_prompt=system_prompt,
# #         user_text=user_text,
# #         image_path=None,
# #     )

# #     parse_ok, parsed_response = try_parse_json(result["raw_response"])
# #     if not parse_ok:
# #         raise ValueError(
# #             f"[vlm_planning] Model response could not be parsed as valid JSON.\n\n"
# #             f"Raw response:\n{result['raw_response']}"
# #         )

# #     dependencies = {
# #         "scene_description_full": {
# #             "stored_under_module": "scene_description",
# #             "artifact_filename": "scene_description_full.json",
# #             "prompt_version": scene_version,
# #             "experiment_timestamp": experiment_timestamp,
# #             "model": scene_model,
# #             "run_name": run_name,
# #         }
# #     }

# #     parsed_path, run_info_path = save_module_outputs(
# #         settings=settings,
# #         module_name=module_name,
# #         scenario_name=scenario_name,
# #         version=version,
# #         experiment_timestamp=experiment_timestamp,
# #         model_name=result["model_name"],
# #         run_name=run_name,
# #         deployment_name=result["deployment_name"],
# #         execution_time_seconds=result["execution_time_seconds"],
# #         scenario_data=scenario_context,
# #         parsed_response=parsed_response,
# #         execution_mode="validation_loop",
# #         dependencies=dependencies,
# #         pipeline_config=pipeline_config,
# #     )

# #     print(f"[OK][vlm_planning] Parsed output saved to: {parsed_path}")
# #     print(f"[OK][vlm_planning] Run info saved to:      {run_info_path}")
# #     print(f"[OK][vlm_planning] Execution time:         {result['execution_time_seconds']:.3f}s")

# #     return parsed_response


# # def execute_simultaneous_actions_step(
# #     settings,
# #     scenario_name: str,
# #     scenario_context: dict[str, Any],
# #     version: str,
# #     model_name: str,
# #     experiment_timestamp: str,
# #     run_name: str,
# #     scene_description_full: Any,
# #     sequential_plan: Any,
# #     scene_version: str,
# #     scene_model: str,
# #     plan_version: str,
# #     plan_model: str,
# #     pipeline_config: dict[str, Any],
# # ) -> Any:
# #     module_name = "simultaneous_actions"
# #     base_prompt = load_base_prompt(settings, module_name, version)

# #     system_prompt = render_prompt(
# #         module_name=module_name,
# #         base_prompt=base_prompt,
# #         scenario_data=scenario_context,
# #         scene_description=scene_description_full,
# #         sequential_plan=sequential_plan,
# #     )

# #     user_text = "Generate the compact parallel plan in valid JSON only."

# #     save_rendered_prompt(
# #         settings=settings,
# #         module_name=module_name,
# #         scenario_name=scenario_name,
# #         version=version,
# #         experiment_timestamp=experiment_timestamp,
# #         model_name=model_name,
# #         run_name=run_name,
# #         prompt_text=system_prompt,
# #     )

# #     result = call_azure_chat_completion(
# #         settings=settings,
# #         model_name=model_name,
# #         system_prompt=system_prompt,
# #         user_text=user_text,
# #         image_path=None,
# #     )

# #     parse_ok, parsed_response = try_parse_json(result["raw_response"])
# #     if not parse_ok:
# #         raise ValueError(
# #             f"[simultaneous_actions] Model response could not be parsed as valid JSON.\n\n"
# #             f"Raw response:\n{result['raw_response']}"
# #         )

# #     dependencies = {
# #         "scene_description_full": {
# #             "stored_under_module": "scene_description",
# #             "artifact_filename": "scene_description_full.json",
# #             "prompt_version": scene_version,
# #             "experiment_timestamp": experiment_timestamp,
# #             "model": scene_model,
# #             "run_name": run_name,
# #         },
# #         "vlm_planning": {
# #             "prompt_version": plan_version,
# #             "experiment_timestamp": experiment_timestamp,
# #             "model": plan_model,
# #             "run_name": run_name,
# #         },
# #     }

# #     parsed_path, run_info_path = save_module_outputs(
# #         settings=settings,
# #         module_name=module_name,
# #         scenario_name=scenario_name,
# #         version=version,
# #         experiment_timestamp=experiment_timestamp,
# #         model_name=result["model_name"],
# #         run_name=run_name,
# #         deployment_name=result["deployment_name"],
# #         execution_time_seconds=result["execution_time_seconds"],
# #         scenario_data=scenario_context,
# #         parsed_response=parsed_response,
# #         execution_mode="validation_loop",
# #         dependencies=dependencies,
# #         pipeline_config=pipeline_config,
# #     )

# #     print(f"[OK][simultaneous_actions] Parsed output saved to: {parsed_path}")
# #     print(f"[OK][simultaneous_actions] Run info saved to:      {run_info_path}")
# #     print(f"[OK][simultaneous_actions] Execution time:         {result['execution_time_seconds']:.3f}s")

# #     return parsed_response


# # def execute_validator_step(
# #     settings,
# #     scenario_name: str,
# #     validator_version: str,
# #     validator_model: str,
# #     upstream_timestamp: str,
# #     run_name: str,
# #     condition_name: str,
# #     condition_text: str,
# #     image_path: str,
# #     scene_version: str,
# #     scene_model: str,
# #     plan_version: str,
# #     plan_model: str,
# #     sim_version: str,
# #     sim_model: str,
# # ) -> dict[str, Any]:
# #     scene_object_list = load_scene_object_list_from_cycle(
# #         settings=settings,
# #         scenario_name=scenario_name,
# #         scene_version=scene_version,
# #         pipeline_timestamp=upstream_timestamp,
# #         scene_model=scene_model,
# #         run_name=run_name,
# #     )

# #     base_prompt = load_base_prompt(settings, "validator", validator_version)
# #     system_prompt = render_validator_prompt(
# #         base_prompt=base_prompt,
# #         condition=condition_text,
# #         scene_object_list=scene_object_list,
# #     )

# #     prompt_path = save_validator_prompt(
# #         settings=settings,
# #         scenario=scenario_name,
# #         version=validator_version,
# #         upstream_timestamp=upstream_timestamp,
# #         model_name=validator_model,
# #         run_name=run_name,
# #         condition_name=condition_name,
# #         prompt_text=system_prompt,
# #     )

# #     result = call_azure_chat_completion(
# #         settings=settings,
# #         model_name=validator_model,
# #         system_prompt=system_prompt,
# #         user_text="Validate the condition and return valid JSON only.",
# #         image_path=image_path,
# #     )

# #     parse_ok, parsed_response = try_parse_json(result["raw_response"])
# #     if not parse_ok:
# #         raise ValueError(
# #             f"[validator:{condition_name}] Model response could not be parsed as valid JSON.\n\n"
# #             f"Raw response:\n{result['raw_response']}"
# #         )

# #     validate_validator_response(parsed_response)

# #     dependencies = {
# #         "scene_description_full": {
# #             "stored_under_module": "scene_description",
# #             "artifact_filename": "scene_description_full.json",
# #             "prompt_version": scene_version,
# #             "experiment_timestamp": upstream_timestamp,
# #             "model": scene_model,
# #             "run_name": run_name,
# #         },
# #         "vlm_planning": {
# #             "prompt_version": plan_version,
# #             "experiment_timestamp": upstream_timestamp,
# #             "model": plan_model,
# #             "run_name": run_name,
# #         },
# #         "simultaneous_actions": {
# #             "prompt_version": sim_version,
# #             "experiment_timestamp": upstream_timestamp,
# #             "model": sim_model,
# #             "run_name": run_name,
# #         },
# #     }

# #     parsed_path, run_info_path = save_validator_outputs(
# #         settings=settings,
# #         scenario=scenario_name,
# #         version=validator_version,
# #         upstream_timestamp=upstream_timestamp,
# #         model_name=result["model_name"],
# #         run_name=run_name,
# #         condition_name=condition_name,
# #         deployment_name=result["deployment_name"],
# #         execution_time_seconds=result["execution_time_seconds"],
# #         image_path=image_path,
# #         condition_text=condition_text,
# #         parsed_response=parsed_response,
# #         dependencies=dependencies,
# #     )

# #     print(f"[OK][validator:{condition_name}] Prompt saved to:        {prompt_path}")
# #     print(f"[OK][validator:{condition_name}] Parsed output saved to: {parsed_path}")
# #     print(f"[OK][validator:{condition_name}] Run info saved to:      {run_info_path}")
# #     print(f"[OK][validator:{condition_name}] Execution time:         {result['execution_time_seconds']:.3f}s")

# #     return parsed_response


# # # ============================================================
# # # SUMMARY HELPERS
# # # ============================================================

# # def save_loop_summary(
# #     settings,
# #     scenario_name: str,
# #     loop_timestamp: str,
# #     summary: dict[str, Any],
# # ) -> Path:
# #     output_dir = (
# #         settings.project_root
# #         / "outputs"
# #         / "validation_loop"
# #         / scenario_name
# #         / loop_timestamp
# #     )
# #     ensure_dir(output_dir)

# #     out_path = output_dir / "loop_summary.json"
# #     write_json(out_path, summary)
# #     return out_path


# # # ============================================================
# # # MAIN
# # # ============================================================

# # def main() -> None:
# #     parser = build_parser()
# #     args = parser.parse_args()

# #     validate_args(args)

# #     settings = load_settings()
# #     scenario_data = load_scenario(settings, args.scenario)

# #     poses_by_image_path = resolve_poses_by_image_path(
# #         settings=settings,
# #         scenario_name=args.scenario,
# #         explicit_path=args.poses_by_image_path,
# #     )
# #     poses_by_image = load_poses_by_image_map(poses_by_image_path)

# #     initial_image_path = (
# #         str(Path(args.initial_image_path).resolve())
# #         if args.initial_image_path is not None
# #         else scenario_data.get("image_path_abs")
# #     )
# #     if not initial_image_path:
# #         raise ValueError(
# #             "No initial image available. Provide --initial-image-path or set 'image' in scenario.json."
# #         )
# #     if not Path(initial_image_path).exists():
# #         raise FileNotFoundError(f"Initial image not found: {initial_image_path}")

# #     frame_paths = list_frame_paths(args.frames_dir)
# #     initial_image_resolved = str(Path(initial_image_path).resolve())
# #     frame_paths = [p for p in frame_paths if str(Path(p).resolve()) != initial_image_resolved]
# #     frame_cursor = 0

# #     loop_timestamp = make_experiment_timestamp()
# #     run_name = "run_001"

# #     current_image = initial_image_path
# #     replans_done = 0
# #     task_completed = False
# #     cycle_idx = 0

# #     summary: dict[str, Any] = {
# #         "module": "validation_loop",
# #         "scenario_name": args.scenario,
# #         "loop_timestamp": loop_timestamp,
# #         "timestamp": datetime.now().isoformat(),
# #         "initial_image_path": str(Path(initial_image_path).resolve()),
# #         "frames_dir": str(Path(args.frames_dir).resolve()),
# #         "poses_by_image_path": str(poses_by_image_path),
# #         "config": {
# #             "scene_description": {
# #                 "prompt_version": args.scene_v,
# #                 "model": args.scene_model,
# #             },
# #             "scene_description_full": {
# #                 "stored_under_module": "scene_description",
# #                 "artifact_filename": "scene_description_full.json",
# #                 "prompt_version": args.scene_v,
# #                 "model": args.scene_model,
# #                 "mode": "deterministic_scene_enrichment_per_image",
# #             },
# #             "vlm_planning": {
# #                 "prompt_version": args.plan_v,
# #                 "model": args.plan_model,
# #             },
# #             "simultaneous_actions": {
# #                 "prompt_version": args.sim_v,
# #                 "model": args.sim_model,
# #             },
# #             "validator": {
# #                 "prompt_version": args.validator_v,
# #                 "model": args.validator_model,
# #             },
# #             "max_replans": args.max_replans,
# #             "grounding_safety_threshold": args.grounding_safety_threshold,
# #             "grounding_debug_mapping": args.grounding_debug_mapping,
# #         },
# #         "replans_done": 0,
# #         "task_completed": False,
# #         "cycles": [],
# #     }

# #     while not task_completed:
# #         cycle_idx += 1
# #         pipeline_timestamp = make_experiment_timestamp()

# #         print("\n======================================================")
# #         print(f"VALIDATION LOOP CYCLE STARTED | cycle={cycle_idx}")
# #         print(f"Current image:   {current_image}")
# #         print(f"Pipeline ts:     {pipeline_timestamp}")
# #         print("======================================================")

# #         scenario_context = make_scenario_context(
# #             scenario_data=scenario_data,
# #             image_path=current_image,
# #         )

# #         pipeline_config = {
# #             "scene_description": {
# #                 "prompt_version": args.scene_v,
# #                 "experiment_timestamp": pipeline_timestamp,
# #                 "model": args.scene_model,
# #                 "run_name": run_name,
# #             },
# #             "scene_description_full": {
# #                 "stored_under_module": "scene_description",
# #                 "artifact_filename": "scene_description_full.json",
# #                 "prompt_version": args.scene_v,
# #                 "experiment_timestamp": pipeline_timestamp,
# #                 "model": args.scene_model,
# #                 "run_name": run_name,
# #                 "mode": "deterministic_scene_enrichment_per_image",
# #             },
# #             "vlm_planning": {
# #                 "prompt_version": args.plan_v,
# #                 "experiment_timestamp": pipeline_timestamp,
# #                 "model": args.plan_model,
# #                 "run_name": run_name,
# #             },
# #             "simultaneous_actions": {
# #                 "prompt_version": args.sim_v,
# #                 "experiment_timestamp": pipeline_timestamp,
# #                 "model": args.sim_model,
# #                 "run_name": run_name,
# #             },
# #         }

# #         cycle_record: dict[str, Any] = {
# #             "cycle_idx": cycle_idx,
# #             "pipeline_timestamp": pipeline_timestamp,
# #             "run_name": run_name,
# #             "start_image_path": str(Path(current_image).resolve()),
# #             "start_image_name": Path(current_image).name,
# #             "stages": [],
# #             "outcome": None,
# #         }

# #         cycle_error = False

# #         try:
# #             scene_description = execute_scene_description_step(
# #                 settings=settings,
# #                 scenario_name=args.scenario,
# #                 scenario_context=scenario_context,
# #                 version=args.scene_v,
# #                 model_name=args.scene_model,
# #                 experiment_timestamp=pipeline_timestamp,
# #                 run_name=run_name,
# #                 pipeline_config=pipeline_config,
# #                 image_path=current_image,
# #             )
# #             print("\n[scene_description] Parsed JSON:")
# #             print(json.dumps(scene_description, indent=2, ensure_ascii=False))

# #             scene_description_full = execute_scene_description_full_step(
# #                 settings=settings,
# #                 scenario_name=args.scenario,
# #                 scenario_context=scenario_context,
# #                 version=args.scene_v,
# #                 model_name=args.scene_model,
# #                 experiment_timestamp=pipeline_timestamp,
# #                 run_name=run_name,
# #                 scene_description=scene_description,
# #                 pipeline_config=pipeline_config,
# #                 image_path=current_image,
# #                 poses_by_image=poses_by_image,
# #                 safety_threshold=args.grounding_safety_threshold,
# #                 include_debug_mapping=args.grounding_debug_mapping,
# #             )
# #             print("\n[scene_description_full] Parsed JSON:")
# #             print(json.dumps(scene_description_full, indent=2, ensure_ascii=False))

# #             sequential_plan = execute_vlm_planning_step(
# #                 settings=settings,
# #                 scenario_name=args.scenario,
# #                 scenario_context=scenario_context,
# #                 version=args.plan_v,
# #                 model_name=args.plan_model,
# #                 experiment_timestamp=pipeline_timestamp,
# #                 run_name=run_name,
# #                 scene_description_full=scene_description_full,
# #                 scene_version=args.scene_v,
# #                 scene_model=args.scene_model,
# #                 pipeline_config=pipeline_config,
# #             )
# #             print("\n[vlm_planning] Parsed JSON:")
# #             print(json.dumps(sequential_plan, indent=2, ensure_ascii=False))

# #             compact_parallel_plan = execute_simultaneous_actions_step(
# #                 settings=settings,
# #                 scenario_name=args.scenario,
# #                 scenario_context=scenario_context,
# #                 version=args.sim_v,
# #                 model_name=args.sim_model,
# #                 experiment_timestamp=pipeline_timestamp,
# #                 run_name=run_name,
# #                 scene_description_full=scene_description_full,
# #                 sequential_plan=sequential_plan,
# #                 scene_version=args.scene_v,
# #                 scene_model=args.scene_model,
# #                 plan_version=args.plan_v,
# #                 plan_model=args.plan_model,
# #                 pipeline_config=pipeline_config,
# #             )
# #             print("\n[simultaneous_actions] Parsed JSON:")
# #             print(json.dumps(compact_parallel_plan, indent=2, ensure_ascii=False))

# #             stages = extract_stages(compact_parallel_plan)

# #             all_stages_succeeded = True

# #             for stage in stages:
# #                 stage_id = stage["Stage_id"]
# #                 pre_condition = stage["Precondition"]
# #                 post_condition = stage["Postcondition"]

# #                 stage_record: dict[str, Any] = {
# #                     "stage_id": stage_id,
# #                     "precondition": pre_condition,
# #                     "postcondition": post_condition,
# #                     "pre_image_path": str(Path(current_image).resolve()),
# #                     "pre_image_name": Path(current_image).name,
# #                     "post_image_path": None,
# #                     "post_image_name": None,
# #                     "pre_validation": None,
# #                     "post_validation": None,
# #                     "next_image_path": None,
# #                     "next_image_name": None,
# #                 }

# #                 # --------------------------------------------------
# #                 # PRE VALIDATION
# #                 # --------------------------------------------------
# #                 print(f"\n[LOOP] Stage {stage_id} PRE")
# #                 print(f"[LOOP] PRE image:      {current_image}")
# #                 print(f"[LOOP] PRE condition:  {pre_condition}")

# #                 print_pose_dict_for_image(
# #                     poses_by_image=poses_by_image,
# #                     image_path=current_image,
# #                     label=f"validator-pre-stage-{stage_id}",
# #                 )

# #                 pre_name = f"pre_{stage_id}"
# #                 pre_response = execute_validator_step(
# #                     settings=settings,
# #                     scenario_name=args.scenario,
# #                     validator_version=args.validator_v,
# #                     validator_model=args.validator_model,
# #                     upstream_timestamp=pipeline_timestamp,
# #                     run_name=run_name,
# #                     condition_name=pre_name,
# #                     condition_text=pre_condition,
# #                     image_path=current_image,
# #                     scene_version=args.scene_v,
# #                     scene_model=args.scene_model,
# #                     plan_version=args.plan_v,
# #                     plan_model=args.plan_model,
# #                     sim_version=args.sim_v,
# #                     sim_model=args.sim_model,
# #                 )
# #                 print(f"\n[PRE validator:{pre_name}] Parsed JSON:")
# #                 print(json.dumps(pre_response, indent=2, ensure_ascii=False))
# #                 print(f"[LOOP] PRE result:     {pre_response['result']}")
# #                 print(f"[LOOP] PRE reason:     {pre_response['reason']}")

# #                 stage_record["pre_validation"] = pre_response

# #                 if pre_response["result"] == "non_matching":
# #                     print(f"[LOOP] Precondition failed at stage {stage_id}. Replanning from same image.")
# #                     cycle_record["stages"].append(stage_record)
# #                     cycle_record["outcome"] = f"replan_on_pre_stage_{stage_id}"

# #                     if replans_done >= args.max_replans:
# #                         raise RuntimeError(
# #                             f"Maximum number of replans reached ({args.max_replans})."
# #                         )

# #                     replans_done += 1
# #                     summary["replans_done"] = replans_done

# #                     all_stages_succeeded = False
# #                     break

# #                 # --------------------------------------------------
# #                 # DEPLOY PLACEHOLDER / OFFLINE NEXT IMAGE
# #                 # --------------------------------------------------
# #                 if frame_cursor >= len(frame_paths):
# #                     raise RuntimeError(
# #                         "No more images available in frames-dir for simulated deploy progression."
# #                     )

# #                 next_image = frame_paths[frame_cursor]
# #                 frame_cursor += 1

# #                 stage_record["next_image_path"] = str(Path(next_image).resolve())
# #                 stage_record["next_image_name"] = Path(next_image).name
# #                 stage_record["post_image_path"] = str(Path(next_image).resolve())
# #                 stage_record["post_image_name"] = Path(next_image).name

# #                 print(f"\n[LOOP] Stage {stage_id} simulated deploy")
# #                 print(f"[LOOP] NEXT image:     {next_image}")

# #                 # --------------------------------------------------
# #                 # POST VALIDATION
# #                 # --------------------------------------------------
# #                 print(f"\n[LOOP] Stage {stage_id} POST")
# #                 print(f"[LOOP] POST image:     {next_image}")
# #                 print(f"[LOOP] POST condition: {post_condition}")

# #                 print_pose_dict_for_image(
# #                     poses_by_image=poses_by_image,
# #                     image_path=next_image,
# #                     label=f"validator-post-stage-{stage_id}",
# #                 )

# #                 post_name = f"post_{stage_id}"
# #                 post_response = execute_validator_step(
# #                     settings=settings,
# #                     scenario_name=args.scenario,
# #                     validator_version=args.validator_v,
# #                     validator_model=args.validator_model,
# #                     upstream_timestamp=pipeline_timestamp,
# #                     run_name=run_name,
# #                     condition_name=post_name,
# #                     condition_text=post_condition,
# #                     image_path=next_image,
# #                     scene_version=args.scene_v,
# #                     scene_model=args.scene_model,
# #                     plan_version=args.plan_v,
# #                     plan_model=args.plan_model,
# #                     sim_version=args.sim_v,
# #                     sim_model=args.sim_model,
# #                 )
# #                 print(f"\n[POST validator:{post_name}] Parsed JSON:")
# #                 print(json.dumps(post_response, indent=2, ensure_ascii=False))
# #                 print(f"[LOOP] POST result:    {post_response['result']}")
# #                 print(f"[LOOP] POST reason:    {post_response['reason']}")

# #                 stage_record["post_validation"] = post_response
# #                 cycle_record["stages"].append(stage_record)

# #                 if post_response["result"] == "non_matching":
# #                     print(f"[LOOP] Postcondition failed at stage {stage_id}. Replanning from next image.")
# #                     cycle_record["outcome"] = f"replan_on_post_stage_{stage_id}"
# #                     current_image = next_image

# #                     if replans_done >= args.max_replans:
# #                         raise RuntimeError(
# #                             f"Maximum number of replans reached ({args.max_replans})."
# #                         )

# #                     replans_done += 1
# #                     summary["replans_done"] = replans_done

# #                     all_stages_succeeded = False
# #                     break

# #                 current_image = next_image

# #             if all_stages_succeeded:
# #                 cycle_record["outcome"] = "task_completed"
# #                 task_completed = True
# #                 summary["task_completed"] = True
# #                 summary["final_image_path"] = str(Path(current_image).resolve())

# #                 print("\n======================================================")
# #                 print("[LOOP] TASK COMPLETED SUCCESSFULLY")
# #                 print("======================================================")

# #         except Exception as exc:
# #             cycle_record["outcome"] = f"cycle_error: {exc}"
# #             summary["task_completed"] = False
# #             summary["error"] = str(exc)
# #             cycle_error = True

# #         summary["cycles"].append(cycle_record)

# #         if cycle_error:
# #             break

# #     summary_path = save_loop_summary(
# #         settings=settings,
# #         scenario_name=args.scenario,
# #         loop_timestamp=loop_timestamp,
# #         summary=summary,
# #     )

# #     print("\n======================================================")
# #     print("VALIDATION LOOP COMPLETED")
# #     print(f"Scenario:        {args.scenario}")
# #     print(f"Loop timestamp:  {loop_timestamp}")
# #     print(f"Task completed:  {summary['task_completed']}")
# #     print(f"Replans done:    {summary['replans_done']}")
# #     print(f"Summary saved:   {summary_path}")
# #     print("======================================================")


# # if __name__ == "__main__":
# #     main()

