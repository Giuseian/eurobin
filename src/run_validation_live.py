""" run_validation_image.py is the offline image-based validation loop.
Instead of using Gazebo screenshots or real robot execution, it works with images stored under image_data/<scenario>. At the start, it asks you in the terminal to manually choose the initial image. For every later stage, after the precondition validator passes, it asks you again to choose the “next” image manually, as if that image represented the result after deployment.
For each selected image, it runs the normal VLM pipeline: scene_description, scene_description_full, vlm_planning, and simultaneous_actions. The enrichment step uses poses_by_image.json, which maps each image filename to object poses, instead of reading poses from Gazebo.
Then it validates each planned stage. It calls the validator model on the current image and the precondition. If the precondition fails, it replans from the same image. If it passes, you manually select the next image, and the validator checks the postcondition on that new image. If the postcondition fails, it replans from the new image.
In short: run_validation_image.py is a manual, offline version of the validation loop. It lets you test planning, validation, and replanning logic using a folder of prepared images instead of live robot execution or Gazebo screenshots. """


from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from settings import load_settings
from scenario_loader import load_scenario
from llm_client import call_llm_completion
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

SUPPORTED_MODELS = ["o3", "gpt-5.2", "gemini-robotics-er-1.6-preview"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ============================================================
# PARSER
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline validation loop: pipeline -> stage pre/post validation -> "
            "replanning on failure, with manual image selection from image_data/<scenario>."
        )
    )

    parser.add_argument("--scenario", type=str, required=True)

    parser.add_argument(
        "--image-data-root",
        type=str,
        default=None,
        help=(
            "Optional root directory for scenario image folders. "
            "If omitted, defaults to <project_root>/image_data"
        ),
    )

    parser.add_argument(
        "--poses-by-image-path",
        type=str,
        default=None,
        help=(
            "Optional path to a JSON mapping image filename -> pose dictionary. "
            "If omitted, defaults to image_data/<scenario>/poses_by_image.json"
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

    parser.add_argument(
        "--max-replans",
        type=int,
        default=10,
        help="Maximum number of replanning cycles allowed before stopping.",
    )

    parser.add_argument(
        "--grounding-safety-threshold",
        type=float,
        default=0.11,
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


def validate_sampling_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.temperature <= 1.0:
        raise ValueError("--temperature must be between 0.0 and 1.0")

    if not 0.0 <= args.top_p <= 1.0:
        raise ValueError("--top-p must be between 0.0 and 1.0")

    if args.temperature != 0.0 and args.top_p != 1.0:
        raise ValueError(
            "Use either temperature or top_p for sampling control, not both at the same time."
        )


def resolve_image_data_root(settings, explicit_root: str | None) -> Path:
    if explicit_root is not None:
        path = Path(explicit_root).resolve()
    else:
        path = (settings.project_root / "image_data").resolve()

    if not path.exists():
        raise FileNotFoundError(f"image_data root not found: {path}")
    if not path.is_dir():
        raise ValueError(f"image_data root must be a directory: {path}")

    return path


def resolve_scenario_image_dir(image_data_root: Path, scenario_name: str) -> Path:
    scenario_dir = (image_data_root / scenario_name).resolve()

    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario image directory not found: {scenario_dir}")
    if not scenario_dir.is_dir():
        raise ValueError(f"Scenario image path is not a directory: {scenario_dir}")

    return scenario_dir


def validate_args(args: argparse.Namespace, settings) -> tuple[Path, Path]:
    if args.max_replans < 0:
        raise ValueError("--max-replans must be >= 0")

    image_data_root = resolve_image_data_root(settings, args.image_data_root)
    scenario_image_dir = resolve_scenario_image_dir(image_data_root, args.scenario)

    if args.poses_by_image_path is not None:
        poses_path = Path(args.poses_by_image_path).resolve()
        if not poses_path.exists():
            raise FileNotFoundError(f"poses-by-image-path not found: {poses_path}")

    return image_data_root, scenario_image_dir


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
            / "image_data"
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


def validate_relative_image_path(
    relative_input: str,
    scenario_image_dir: Path,
    poses_by_image: dict[str, dict[str, list[float]]],
) -> str:
    rel_path = Path(relative_input)

    if rel_path.is_absolute():
        raise ValueError(
            "Please insert a path relative to image_data/<scenario>, not an absolute path."
        )

    image_path = (scenario_image_dir / rel_path).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension: {image_path.suffix}. "
            f"Allowed: {sorted(IMAGE_EXTENSIONS)}"
        )

    try:
        image_path.relative_to(scenario_image_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Image must stay inside scenario directory: {scenario_image_dir}"
        ) from exc

    _ = get_pose_dict_for_image(poses_by_image, str(image_path))

    return str(image_path)


def prompt_user_for_relative_image_path(
    scenario_image_dir: Path,
    poses_by_image: dict[str, dict[str, list[float]]],
    prompt_label: str,
) -> str:
    while True:
        print(f"\n[{prompt_label}]")
        print(f"Scenario directory: {scenario_image_dir}")
        user_input = input("Insert relative image path: ").strip()

        if not user_input:
            print("[ERROR] Empty input. Please try again.")
            continue

        try:
            selected_image = validate_relative_image_path(
                relative_input=user_input,
                scenario_image_dir=scenario_image_dir,
                poses_by_image=poses_by_image,
            )
            print(f"[OK] Selected image: {selected_image}")
            print(f"[OK] Pose found for:  {Path(selected_image).name}")
            return selected_image
        except Exception as exc:
            print(f"[ERROR] {exc}")
            print("Please try again.")


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


def write_temp_pose_file(pose_dict: dict[str, list[float]]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        encoding="utf-8",
        delete=False,
    ) as tmp:
        json.dump(pose_dict, tmp, indent=2, ensure_ascii=False)
        return tmp.name


def make_scenario_context(
    scenario_data: dict[str, Any],
    image_path: str,
) -> dict[str, Any]:
    ctx = deepcopy(scenario_data)
    ctx["image"] = Path(image_path).name
    ctx["image_path_abs"] = str(Path(image_path).resolve())
    return ctx


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


def build_global_config(args: argparse.Namespace, scenario_image_dir: Path) -> dict[str, Any]:
    return {
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "image_input": {
            "mode": "manual_terminal_selection_relative_to_scenario_dir",
            "scenario_image_dir": str(scenario_image_dir.resolve()),
        },
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
    scenario_image_dir: Path,
) -> dict[str, Any]:
    return {
        "cycle_name": cycle_name,
        "cycle_index": cycle_idx,
        "cycle_timestamp": cycle_timestamp,
        "image_input": {
            "mode": "manual_terminal_selection_relative_to_scenario_dir",
            "scenario_image_dir": str(scenario_image_dir.resolve()),
        },
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
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
        "validator": {
            "prompt_version": args.validator_v,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": args.validator_model,
        },
    }


def copy_tree_if_exists(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    ensure_dir(dst_dir.parent)
    if dst_dir.exists():
        raise FileExistsError(f"Destination already exists: {dst_dir}")
    shutil.copytree(src_dir, dst_dir)


def copy_full_run_outputs_to_final(
    settings,
    scenario_name: str,
    loop_timestamp: str,
    scene_version: str,
    scene_model: str,
    plan_version: str,
    plan_model: str,
    sim_version: str,
    sim_model: str,
    validator_version: str,
    validator_model: str,
) -> dict[str, Path]:
    outputs_root = settings.project_root / "outputs"
    outputs_final_root = settings.project_root / "outputs_final"

    copied_paths = {
        "validation_loop": outputs_final_root / "validation_loop" / scenario_name / loop_timestamp,
        "scene_description": outputs_final_root / "scene_description" / scenario_name / scene_version / loop_timestamp / scene_model,
        "vlm_planning": outputs_final_root / "vlm_planning" / scenario_name / plan_version / loop_timestamp / plan_model,
        "simultaneous_actions": outputs_final_root / "simultaneous_actions" / scenario_name / sim_version / loop_timestamp / sim_model,
        "validator": outputs_final_root / "validator" / scenario_name / validator_version / loop_timestamp / validator_model,
    }

    copy_tree_if_exists(
        outputs_root / "validation_loop" / scenario_name / loop_timestamp,
        copied_paths["validation_loop"],
    )
    copy_tree_if_exists(
        outputs_root / "scene_description" / scenario_name / scene_version / loop_timestamp / scene_model,
        copied_paths["scene_description"],
    )
    copy_tree_if_exists(
        outputs_root / "vlm_planning" / scenario_name / plan_version / loop_timestamp / plan_model,
        copied_paths["vlm_planning"],
    )
    copy_tree_if_exists(
        outputs_root / "simultaneous_actions" / scenario_name / sim_version / loop_timestamp / sim_model,
        copied_paths["simultaneous_actions"],
    )
    copy_tree_if_exists(
        outputs_root / "validator" / scenario_name / validator_version / loop_timestamp / validator_model,
        copied_paths["validator"],
    )

    return copied_paths


def load_scenario_from_image_data(settings, scenario_name: str) -> dict[str, Any]:
    """
    Carica lo scenario da:
    image_data/<scenario_name>/scenario.json

    Usato SOLO da run_validation_image.py
    """
    scenario_dir = settings.project_root / "image_data" / scenario_name
    scenario_file = scenario_dir / "scenario.json"

    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    if not scenario_file.exists():
        raise FileNotFoundError(f"scenario.json not found: {scenario_file}")

    scenario_data = json.loads(scenario_file.read_text(encoding="utf-8"))

    if not isinstance(scenario_data, dict):
        raise ValueError(f"Scenario file must contain a JSON object: {scenario_file}")

    scenario_data["scenario_name"] = scenario_data.get("scenario_name", scenario_name)

    image_rel = scenario_data.get("image")
    if image_rel:
        image_abs = (scenario_dir / image_rel).resolve()
        if not image_abs.exists():
            raise FileNotFoundError(
                f"Image file declared in scenario.json not found: {image_abs}"
            )
        scenario_data["image_path_abs"] = str(image_abs)
    else:
        scenario_data["image_path_abs"] = None

    scenario_data["scenario_dir_abs"] = str(scenario_dir.resolve())

    return scenario_data


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
    temperature: float,
    top_p: float,
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

    result = call_llm_completion(
        settings=settings,
        model_name=model_name,
        system_prompt=system_prompt,
        user_text=user_text,
        image_path=image_path,
        temperature=temperature,
        top_p=top_p,
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
    direct_name_matching: bool = False,
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
            direct_name_matching=direct_name_matching,
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
    temperature: float,
    top_p: float,
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

    result = call_llm_completion(
        settings=settings,
        model_name=model_name,
        system_prompt=system_prompt,
        user_text=user_text,
        image_path=None,
        temperature=temperature,
        top_p=top_p,
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
    temperature: float,
    top_p: float,
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

    result = call_llm_completion(
        settings=settings,
        model_name=model_name,
        system_prompt=system_prompt,
        user_text=user_text,
        image_path=None,
        temperature=temperature,
        top_p=top_p,
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
    temperature: float,
    top_p: float,
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

    result = call_llm_completion(
        settings=settings,
        model_name=validator_model,
        system_prompt=system_prompt,
        user_text="Validate the condition and return valid JSON only.",
        image_path=image_path,
        temperature=temperature,
        top_p=top_p,
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
        "sampling_config": {
            "temperature": temperature,
            "top_p": top_p,
        },
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
        "scenario_image_dir": full_summary["scenario_image_dir"],
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
            "sampling": full_summary["config"]["sampling"],
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
            "sampling": full_summary["config"]["sampling"],
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
            "sampling": full_summary["config"]["sampling"],
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
            "sampling": full_summary["config"]["sampling"],
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

    validate_sampling_args(args)

    settings = load_settings()
    image_data_root, scenario_image_dir = validate_args(args, settings)
    #scenario_data = load_scenario(settings, args.scenario)
    scenario_data = load_scenario_from_image_data(settings, args.scenario)



    scenario_dir = Path(scenario_data["scenario_dir_abs"])
    scenario_file = scenario_dir / "scenario.json"

    print("\n[DEBUG][SCENARIO SOURCE]")
    print(f"Directory: {scenario_dir}")
    print(f"File:      {scenario_file}")
    print(f"Task:      {scenario_data.get('task')}")
    print("--------------------------------------------------")

    poses_by_image_path = resolve_poses_by_image_path(
        settings=settings,
        scenario_name=args.scenario,
        explicit_path=args.poses_by_image_path,
    )
    poses_by_image = load_poses_by_image_map(poses_by_image_path)

    initial_image_path = prompt_user_for_relative_image_path(
        scenario_image_dir=scenario_image_dir,
        poses_by_image=poses_by_image,
        prompt_label="SELECTING INITIAL IMAGE",
    )

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
        "scenario_image_dir": str(scenario_image_dir.resolve()),
        "poses_by_image_path": str(poses_by_image_path),
        "config": build_global_config(args, scenario_image_dir),
        "replans_done": 0,
        "task_completed": False,
        "final_image_path": None,
        "cycles": [],
    }

    print("\n======================================================")
    print("VALIDATION LOOP CONFIG")
    print(f"Scenario:                  {args.scenario}")
    print(f"Image data root:           {image_data_root}")
    print(f"Scenario image dir:        {scenario_image_dir}")
    print(f"Initial image:             {initial_image_path}")
    print(f"Poses file:                {poses_by_image_path}")
    print(f"Temperature:               {args.temperature}")
    print(f"Top-p:                     {args.top_p}")
    print(f"Max replans:               {args.max_replans}")
    print("Image input mode:          manual relative path selection")
    print("======================================================")

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
            scenario_image_dir=scenario_image_dir,
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
                temperature=args.temperature,
                top_p=args.top_p,
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
                temperature=args.temperature,
                top_p=args.top_p,
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
                temperature=args.temperature,
                top_p=args.top_p,
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
                    "pre_validation_time_seconds": None,
                    "post_validation_time_seconds": None,
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
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                pre_response = pre_artifact["output"]

                print(f"\n[PRE validator:pre_{stage_id}] Parsed JSON:")
                print(json.dumps(pre_response, indent=2, ensure_ascii=False))
                print(f"[LOOP] PRE result:     {pre_response['result']}")
                print(f"[LOOP] PRE reason:     {pre_response['reason']}")

                stage_record["pre_validation"] = pre_response
                stage_record["validator_paths"]["pre"] = pre_artifact["paths"]
                stage_record["pre_validation_time_seconds"] = pre_artifact["execution_time_seconds"]

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
                # MANUAL NEXT IMAGE AFTER SUCCESSFUL DEPLOY
                # --------------------------------------------------
                print(f"\n[LOOP] Stage {stage_id} deploy succeeded.")
                print("[LOOP] A new image is now required for post-validation.")

                next_image = prompt_user_for_relative_image_path(
                    scenario_image_dir=scenario_image_dir,
                    poses_by_image=poses_by_image,
                    prompt_label="SELECTING NEW IMAGE",
                )

                stage_record["next_image_path"] = str(Path(next_image).resolve())
                stage_record["next_image_name"] = Path(next_image).name
                stage_record["post_image_path"] = str(Path(next_image).resolve())
                stage_record["post_image_name"] = Path(next_image).name

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
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                post_response = post_artifact["output"]

                print(f"\n[POST validator:post_{stage_id}] Parsed JSON:")
                print(json.dumps(post_response, indent=2, ensure_ascii=False))
                print(f"[LOOP] POST result:    {post_response['result']}")
                print(f"[LOOP] POST reason:    {post_response['reason']}")

                stage_record["post_validation"] = post_response
                stage_record["validator_paths"]["post"] = post_artifact["paths"]
                stage_record["post_validation_time_seconds"] = post_artifact["execution_time_seconds"]
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

    copied_output_paths = copy_full_run_outputs_to_final(
        settings=settings,
        scenario_name=args.scenario,
        loop_timestamp=loop_timestamp,
        scene_version=args.scene_v,
        scene_model=args.scene_model,
        plan_version=args.plan_v,
        plan_model=args.plan_model,
        sim_version=args.sim_v,
        sim_model=args.sim_model,
        validator_version=args.validator_v,
        validator_model=args.validator_model,
    )

    print("\n======================================================")
    print("VALIDATION LOOP COMPLETED")
    print(f"Scenario:                  {args.scenario}")
    print(f"Loop timestamp:            {loop_timestamp}")
    print(f"Temperature:               {args.temperature}")
    print(f"Top-p:                     {args.top_p}")
    print(f"Task completed:            {full_summary['task_completed']}")
    print(f"Replans done:              {full_summary['replans_done']}")
    print(f"Run info saved:            {summary_paths['run_info']}")
    print(f"Loop summary saved:        {summary_paths['loop_summary']}")
    print(f"Scene summary saved:       {summary_paths['scene_description_summary']}")
    print(f"Planning summary saved:    {summary_paths['vlm_planning_summary']}")
    print(f"Sim-actions summary saved: {summary_paths['simultaneous_actions_summary']}")
    print(f"Validator summary saved:   {summary_paths['validator_summary']}")
    print(f"Full summary saved:        {summary_paths['full_pipeline_summary']}")
    print(f"Copied validation_loop to: {copied_output_paths['validation_loop']}")
    print(f"Copied scene_description:  {copied_output_paths['scene_description']}")
    print(f"Copied vlm_planning to:    {copied_output_paths['vlm_planning']}")
    print(f"Copied sim_actions to:     {copied_output_paths['simultaneous_actions']}")
    print(f"Copied validator to:       {copied_output_paths['validator']}")
    print("======================================================")


if __name__ == "__main__":
    main()

