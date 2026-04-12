from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    try_parse_json,
    write_json,
    read_json,
)

SUPPORTED_MODELS = ["o3", "gpt-5.2"]


# ============================================================
# PARSER
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real deploy validation loop: scene pipeline -> stage pre/post validation -> "
            "real deploy -> replanning on failure."
        )
    )

    parser.add_argument("--scenario", type=str, required=True)

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
        "--manip-dir",
        type=str,
        default=".",
        help="Directory containing manipulation scripts (grasp_box_yz.py, place_box.py, etc.).",
    )

    parser.add_argument(
        "--screens-subdir",
        type=str,
        default="screens",
        help="Subdirectory name used inside the real_deploy scenario folder for screenshots.",
    )

    parser.add_argument(
        "--grounding-topic",
        type=str,
        default="/world/default/dynamic_pose/info",
        help="Gazebo topic used by scene grounding to read dynamic poses.",
    )

    parser.add_argument(
        "--grounding-timeout-sec",
        type=float,
        default=3.0,
        help="Timeout for Gazebo pose read in the scene grounding step.",
    )

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


# ============================================================
# GENERIC HELPERS
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

    manip_dir = Path(args.manip_dir).resolve()
    if not manip_dir.exists():
        raise FileNotFoundError(f"manip-dir not found: {manip_dir}")
    if not manip_dir.is_dir():
        raise ValueError(f"--manip-dir must be a directory: {manip_dir}")

    required_scripts = [
        "grasp_box_yz.py",
        "place_box.py",
        "place_box_2.py",
        "homing.py",
        "grasp_box_xy.py",
    ]

    for script_name in required_scripts:
        script_path = manip_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Required manipulation script not found: {script_path}")


def make_scenario_context(
    scenario_data: dict[str, Any],
    image_path: str,
) -> dict[str, Any]:
    ctx = deepcopy(scenario_data)
    ctx["image"] = Path(image_path).name
    ctx["image_path_abs"] = str(Path(image_path).resolve())
    return ctx


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


def run_command(cmd: list[str], label: str) -> None:
    print(f"\n[CMD] {label}")
    print("[CMD] " + " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(f"[STDOUT][{label}]\n{result.stdout}")
    if result.stderr:
        print(f"[STDERR][{label}]\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed for '{label}' with return code {result.returncode}."
        )


def run_python_script(script_path: Path, label: str) -> None:
    run_command([sys.executable, str(script_path)], label=label)


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
            "mode": "deterministic_grounding_gazebo",
            "grounding_topic": args.grounding_topic,
            "grounding_timeout_sec": args.grounding_timeout_sec,
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
        "real_deploy": {
            "manip_dir": str(Path(args.manip_dir).resolve()),
            "screens_subdir": args.screens_subdir,
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
            "mode": "deterministic_grounding_gazebo",
            "grounding_topic": args.grounding_topic,
            "grounding_timeout_sec": args.grounding_timeout_sec,
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
        "real_deploy": {
            "manip_dir": str(Path(args.manip_dir).resolve()),
            "screens_subdir": args.screens_subdir,
        },
    }


# ============================================================
# SCREENSHOT HELPERS
# ============================================================

def get_screens_root(
    settings,
    scenario_name: str,
    loop_timestamp: str,
    screens_subdir: str,
) -> Path:
    return (
        settings.project_root
        / "scenarios"
        / "real_deploy_validation_loop"
        / scenario_name
        / loop_timestamp
        / screens_subdir
    )


def get_next_sequential_image_path(screens_dir: Path) -> Path:
    existing = sorted(
        p for p in screens_dir.glob("*.png")
        if p.stem.isdigit()
    )

    if not existing:
        next_index = 1
    else:
        next_index = max(int(p.stem) for p in existing) + 1

    return screens_dir / f"{next_index:03d}.png"


def take_screenshot(
    screens_dir: Path,
    wait_timeout: float = 5.0,
    poll_interval: float = 0.2,
) -> str:
    screens_dir = ensure_dir(screens_dir.resolve())

    before = {p.resolve() for p in screens_dir.glob("*.png")}

    cmd = [
        "gz", "service",
        "-s", "/gui/screenshot",
        "--reqtype", "gz.msgs.StringMsg",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req", f'data: "{str(screens_dir)}"'
    ]

    print(f"\n[SCREEN] Taking screenshot into directory: {screens_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(f"[SCREEN][STDOUT]\n{result.stdout}")
    if result.stderr:
        print(f"[SCREEN][STDERR]\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError("Screenshot failed.")

    deadline = time.time() + wait_timeout
    new_file: Path | None = None

    while time.time() < deadline:
        current = {p.resolve() for p in screens_dir.glob("*.png")}
        created = current - before
        if created:
            new_file = max(created, key=lambda p: p.stat().st_mtime)
            break
        time.sleep(poll_interval)

    if new_file is None or not new_file.exists():
        raise RuntimeError(
            f"Screenshot command returned success but no new PNG appeared in: {screens_dir}"
        )

    final_path = get_next_sequential_image_path(screens_dir)

    if final_path.exists():
        raise RuntimeError(f"Sequential screenshot path already exists: {final_path}")

    new_file.rename(final_path)

    print(f"[SCREEN] Screenshot saved as: {final_path}")
    return str(final_path.resolve())


# ============================================================
# REAL DEPLOY OUTPUT HELPERS
# ============================================================

def get_real_deploy_output_dir(settings, scenario_name: str, loop_timestamp: str) -> Path:
    return (
        settings.project_root
        / "outputs"
        / "real_deploy_validation_loop"
        / scenario_name
        / loop_timestamp
    )


def get_real_deploy_cycle_dir(
    settings,
    scenario_name: str,
    loop_timestamp: str,
    cycle_name: str,
) -> Path:
    return get_real_deploy_output_dir(settings, scenario_name, loop_timestamp) / cycle_name


def save_cycle_summary(
    settings,
    scenario_name: str,
    loop_timestamp: str,
    cycle_name: str,
    cycle_summary: dict[str, Any],
) -> Path:
    cycle_dir = get_real_deploy_cycle_dir(
        settings=settings,
        scenario_name=scenario_name,
        loop_timestamp=loop_timestamp,
        cycle_name=cycle_name,
    )
    ensure_dir(cycle_dir)
    return save_json_file(cycle_dir / "cycle_summary.json", cycle_summary)


def save_real_deploy_artifacts(
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
    output_dir = get_real_deploy_output_dir(settings, scenario_name, loop_timestamp)
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


# ============================================================
# SCENE OBJECT LIST / VALIDATOR HELPERS
# ============================================================

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
        execution_mode="real_deploy_validation_loop",
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
    topic: str,
    timeout_sec: float,
    safety_threshold: float,
    include_debug_mapping: bool,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    parsed_response = enrich_scene(
        input_data=scene_description,
        safety_threshold=safety_threshold,
        pose_source="gazebo",
        topic=topic,
        timeout_sec=timeout_sec,
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
        pose_file=None,
        safety_threshold=safety_threshold,
        include_debug_mapping=include_debug_mapping,
        execution_mode="real_deploy_validation_loop_side_artifact",
    )

    print(f"[OK][scene_description_full] Pose source:          gazebo")
    print(f"[OK][scene_description_full] Gazebo topic:         {topic}")
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
        execution_mode="real_deploy_validation_loop",
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
        execution_mode="real_deploy_validation_loop",
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
        "execution_mode": "real_deploy_validation_loop",
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
# DEPLOY HELPERS
# ============================================================

def deploy_stage_1(
    manip_dir: Path,
    screens_dir: Path,
    stage_id: int,
) -> str:
    run_python_script(manip_dir / "grasp_box_yz.py", label="stage_1/grasp_box_yz.py")
    next_image = take_screenshot(screens_dir)
    return next_image


def deploy_stage_2(
    manip_dir: Path,
    screens_dir: Path,
    stage_id: int,
) -> str:
    run_python_script(manip_dir / "place_box.py", label="stage_2/place_box.py")
    run_python_script(manip_dir / "place_box_2.py", label="stage_2/place_box_2.py")
    run_python_script(manip_dir / "homing.py", label="stage_2/homing.py")
    next_image = take_screenshot(screens_dir)
    return next_image


def deploy_stage_3(
    manip_dir: Path,
    screens_dir: Path,
    stage_id: int,
) -> str:
    run_python_script(manip_dir / "grasp_box_xy.py", label="stage_3/grasp_box_xy.py")
    next_image = take_screenshot(screens_dir)
    return next_image


def execute_stage_deploy(
    stage_id: int,
    manip_dir: Path,
    screens_dir: Path,
) -> str:
    if stage_id == 1:
        return deploy_stage_1(
            manip_dir=manip_dir,
            screens_dir=screens_dir,
            stage_id=stage_id,
        )

    if stage_id == 2:
        return deploy_stage_2(
            manip_dir=manip_dir,
            screens_dir=screens_dir,
            stage_id=stage_id,
        )

    if stage_id == 3:
        return deploy_stage_3(
            manip_dir=manip_dir,
            screens_dir=screens_dir,
            stage_id=stage_id,
        )

    raise ValueError(
        f"No deploy routine defined for Stage_id={stage_id}. "
        "This script currently supports stage ids 1, 2, 3."
    )


# ============================================================
# SUMMARY HELPERS
# ============================================================

def build_run_info(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "real_deploy_validation_loop",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": full_summary["timestamp"],
        "initial_image_path": full_summary["initial_image_path"],
        "screens_dir": full_summary["screens_dir"],
        "config": full_summary["config"],
    }


def build_loop_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "real_deploy_validation_loop_summary",
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_args(args)

    settings = load_settings()
    scenario_data = load_scenario(settings, args.scenario)

    loop_timestamp = make_experiment_timestamp()
    manip_dir = Path(args.manip_dir).resolve()

    screens_dir = get_screens_root(
        settings=settings,
        scenario_name=args.scenario,
        loop_timestamp=loop_timestamp,
        screens_subdir=args.screens_subdir,
    )
    ensure_dir(screens_dir)

    initial_image_path = take_screenshot(screens_dir)
    current_image = initial_image_path
    task_completed = False
    cycle_idx = 0

    full_summary: dict[str, Any] = {
        "module": "real_deploy_validation_loop_full_summary",
        "scenario_name": args.scenario,
        "loop_timestamp": loop_timestamp,
        "timestamp": datetime.now().isoformat(),
        "initial_image_path": str(Path(initial_image_path).resolve()),
        "screens_dir": str(screens_dir.resolve()),
        "manip_dir": str(manip_dir),
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
        print(f"REAL DEPLOY VALIDATION LOOP CYCLE STARTED | cycle={cycle_idx} | {cycle_name}")
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
                topic=args.grounding_topic,
                timeout_sec=args.grounding_timeout_sec,
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
                # REAL DEPLOY
                # --------------------------------------------------
                print(f"\n[LOOP] Stage {stage_id} REAL DEPLOY")
                next_image = execute_stage_deploy(
                    stage_id=stage_id,
                    manip_dir=manip_dir,
                    screens_dir=screens_dir,
                )

                stage_record["next_image_path"] = str(Path(next_image).resolve())
                stage_record["next_image_name"] = Path(next_image).name
                stage_record["post_image_path"] = str(Path(next_image).resolve())
                stage_record["post_image_name"] = Path(next_image).name

                print(f"[LOOP] NEXT image:     {next_image}")

                # --------------------------------------------------
                # POST VALIDATION
                # --------------------------------------------------
                print(f"\n[LOOP] Stage {stage_id} POST")
                print(f"[LOOP] POST image:     {next_image}")
                print(f"[LOOP] POST condition: {post_condition}")

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
        print(f"[OK][real_deploy_validation_loop] Cycle summary saved to: {cycle_summary_path}")

        if cycle_error:
            break

    run_info = build_run_info(full_summary)
    loop_summary = build_loop_summary(full_summary)
    scene_description_summary = build_scene_description_summary(full_summary)
    vlm_planning_summary = build_vlm_planning_summary(full_summary)
    simultaneous_actions_summary = build_simultaneous_actions_summary(full_summary)
    validator_summary = build_validator_summary(full_summary)
    full_pipeline_summary = build_full_pipeline_summary(full_summary)

    summary_paths = save_real_deploy_artifacts(
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
    print("REAL DEPLOY VALIDATION LOOP COMPLETED")
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
# import subprocess
# import sys
# import time
# from copy import deepcopy
# from datetime import datetime
# from pathlib import Path
# from typing import Any

# from settings import load_settings
# from scenario_loader import load_scenario
# from azure_openai_client import call_azure_chat_completion
# from build_scene_object_list import build_scene_object_list_from_cycle
# from scene_enrichment import enrich_scene
# from utils import (
#     load_base_prompt,
#     make_experiment_timestamp,
#     make_cycle_name,
#     make_stage_name,
#     render_prompt,
#     save_rendered_prompt_for_cycle,
#     save_module_outputs_for_cycle,
#     save_scene_description_full_artifact_for_cycle,
#     get_validator_prompt_cycle_dir,
#     get_validator_output_cycle_dir,
#     try_parse_json,
#     write_json,
#     read_json,
# )

# SUPPORTED_MODELS = ["o3", "gpt-5.2"]


# # ============================================================
# # PARSER
# # ============================================================

# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Run the real deploy validation loop: scene pipeline -> stage pre/post validation -> "
#             "real deploy -> replanning on failure."
#         )
#     )

#     parser.add_argument("--scenario", type=str, required=True)

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
#         "--manip-dir",
#         type=str,
#         default=".",
#         help="Directory containing manipulation scripts (grasp_box_yz.py, place_box.py, etc.).",
#     )

#     parser.add_argument(
#         "--screens-subdir",
#         type=str,
#         default="screens",
#         help="Subdirectory name used inside the real_deploy scenario folder for screenshots.",
#     )

#     parser.add_argument(
#         "--grounding-topic",
#         type=str,
#         default="/world/default/dynamic_pose/info",
#         help="Gazebo topic used by scene grounding to read dynamic poses.",
#     )

#     parser.add_argument(
#         "--grounding-timeout-sec",
#         type=float,
#         default=3.0,
#         help="Timeout for Gazebo pose read in the scene grounding step.",
#     )

#     parser.add_argument(
#         "--grounding-safety-threshold",
#         type=float,
#         default=0.21,
#         help="Safety threshold used by scene grounding to compute accessibility.",
#     )

#     parser.add_argument(
#         "--grounding-debug-mapping",
#         action="store_true",
#         help="Store the internal VLM-to-Gazebo mapping in scene_description_full.json under _debug.",
#     )

#     return parser


# # ============================================================
# # GENERIC HELPERS
# # ============================================================

# def ensure_dir(path: Path) -> Path:
#     path.mkdir(parents=True, exist_ok=True)
#     return path


# def write_text(path: Path, text: str) -> None:
#     ensure_dir(path.parent)
#     path.write_text(text, encoding="utf-8")


# def save_json_file(path: Path, data: Any) -> Path:
#     ensure_dir(path.parent)
#     write_json(path, data)
#     return path


# def validate_args(args: argparse.Namespace) -> None:
#     if args.max_replans < 0:
#         raise ValueError("--max-replans must be >= 0")

#     manip_dir = Path(args.manip_dir).resolve()
#     if not manip_dir.exists():
#         raise FileNotFoundError(f"manip-dir not found: {manip_dir}")
#     if not manip_dir.is_dir():
#         raise ValueError(f"--manip-dir must be a directory: {manip_dir}")

#     required_scripts = [
#         "grasp_box_yz.py",
#         "place_box.py",
#         "place_box_2.py",
#         "homing.py",
#         "grasp_box_xy.py",
#     ]

#     for script_name in required_scripts:
#         script_path = manip_dir / script_name
#         if not script_path.exists():
#             raise FileNotFoundError(f"Required manipulation script not found: {script_path}")


# def make_scenario_context(
#     scenario_data: dict[str, Any],
#     image_path: str,
# ) -> dict[str, Any]:
#     ctx = deepcopy(scenario_data)
#     ctx["image"] = Path(image_path).name
#     ctx["image_path_abs"] = str(Path(image_path).resolve())
#     return ctx


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


# def run_command(cmd: list[str], label: str) -> None:
#     print(f"\n[CMD] {label}")
#     print("[CMD] " + " ".join(cmd))

#     result = subprocess.run(cmd, capture_output=True, text=True)

#     if result.stdout:
#         print(f"[STDOUT][{label}]\n{result.stdout}")
#     if result.stderr:
#         print(f"[STDERR][{label}]\n{result.stderr}")

#     if result.returncode != 0:
#         raise RuntimeError(
#             f"Command failed for '{label}' with return code {result.returncode}."
#         )


# def run_python_script(script_path: Path, label: str) -> None:
#     run_command([sys.executable, str(script_path)], label=label)


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
#             "mode": "deterministic_grounding_gazebo",
#             "grounding_topic": args.grounding_topic,
#             "grounding_timeout_sec": args.grounding_timeout_sec,
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
#         "real_deploy": {
#             "manip_dir": str(Path(args.manip_dir).resolve()),
#             "screens_subdir": args.screens_subdir,
#         },
#         "max_replans": args.max_replans,
#     }


# def build_cycle_config(
#     args: argparse.Namespace,
#     cycle_timestamp: str,
#     cycle_name: str,
#     cycle_idx: int,
#     loop_timestamp: str,
# ) -> dict[str, Any]:
#     return {
#         "cycle_name": cycle_name,
#         "cycle_index": cycle_idx,
#         "cycle_timestamp": cycle_timestamp,
#         "scene_description": {
#             "prompt_version": args.scene_v,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": args.scene_model,
#         },
#         "scene_description_full": {
#             "stored_under_module": "scene_description",
#             "artifact_filename": "scene_description_full.json",
#             "prompt_version": args.scene_v,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": args.scene_model,
#             "mode": "deterministic_grounding_gazebo",
#             "grounding_topic": args.grounding_topic,
#             "grounding_timeout_sec": args.grounding_timeout_sec,
#             "grounding_safety_threshold": args.grounding_safety_threshold,
#             "grounding_debug_mapping": args.grounding_debug_mapping,
#         },
#         "vlm_planning": {
#             "prompt_version": args.plan_v,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": args.plan_model,
#         },
#         "simultaneous_actions": {
#             "prompt_version": args.sim_v,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": args.sim_model,
#         },
#         "validator": {
#             "prompt_version": args.validator_v,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": args.validator_model,
#         },
#         "real_deploy": {
#             "manip_dir": str(Path(args.manip_dir).resolve()),
#             "screens_subdir": args.screens_subdir,
#         },
#     }


# # ============================================================
# # SCREENSHOT HELPERS
# # ============================================================

# def get_screens_root(
#     settings,
#     scenario_name: str,
#     loop_timestamp: str,
#     screens_subdir: str,
# ) -> Path:
#     return (
#         settings.project_root
#         / "scenarios"
#         / "real_deploy_validation_loop"
#         / scenario_name
#         / loop_timestamp
#         / screens_subdir
#     )


# def get_next_sequential_image_path(screens_dir: Path) -> Path:
#     existing = sorted(
#         p for p in screens_dir.glob("*.png")
#         if p.stem.isdigit()
#     )

#     if not existing:
#         next_index = 1
#     else:
#         next_index = max(int(p.stem) for p in existing) + 1

#     return screens_dir / f"{next_index:03d}.png"


# def take_screenshot(
#     screens_dir: Path,
#     wait_timeout: float = 5.0,
#     poll_interval: float = 0.2,
# ) -> str:
#     screens_dir = ensure_dir(screens_dir.resolve())

#     before = {p.resolve() for p in screens_dir.glob("*.png")}

#     cmd = [
#         "gz", "service",
#         "-s", "/gui/screenshot",
#         "--reqtype", "gz.msgs.StringMsg",
#         "--reptype", "gz.msgs.Boolean",
#         "--timeout", "3000",
#         "--req", f'data: "{str(screens_dir)}"'
#     ]

#     print(f"\n[SCREEN] Taking screenshot into directory: {screens_dir}")
#     result = subprocess.run(cmd, capture_output=True, text=True)

#     if result.stdout:
#         print(f"[SCREEN][STDOUT]\n{result.stdout}")
#     if result.stderr:
#         print(f"[SCREEN][STDERR]\n{result.stderr}")

#     if result.returncode != 0:
#         raise RuntimeError("Screenshot failed.")

#     deadline = time.time() + wait_timeout
#     new_file: Path | None = None

#     while time.time() < deadline:
#         current = {p.resolve() for p in screens_dir.glob("*.png")}
#         created = current - before
#         if created:
#             new_file = max(created, key=lambda p: p.stat().st_mtime)
#             break
#         time.sleep(poll_interval)

#     if new_file is None or not new_file.exists():
#         raise RuntimeError(
#             f"Screenshot command returned success but no new PNG appeared in: {screens_dir}"
#         )

#     final_path = get_next_sequential_image_path(screens_dir)

#     if final_path.exists():
#         raise RuntimeError(f"Sequential screenshot path already exists: {final_path}")

#     new_file.rename(final_path)

#     print(f"[SCREEN] Screenshot saved as: {final_path}")
#     return str(final_path.resolve())


# # ============================================================
# # REAL DEPLOY OUTPUT HELPERS
# # ============================================================

# def get_real_deploy_output_dir(settings, scenario_name: str, loop_timestamp: str) -> Path:
#     return (
#         settings.project_root
#         / "outputs"
#         / "real_deploy_validation_loop"
#         / scenario_name
#         / loop_timestamp
#     )


# def get_real_deploy_cycle_dir(
#     settings,
#     scenario_name: str,
#     loop_timestamp: str,
#     cycle_name: str,
# ) -> Path:
#     return get_real_deploy_output_dir(settings, scenario_name, loop_timestamp) / cycle_name


# def save_cycle_summary(
#     settings,
#     scenario_name: str,
#     loop_timestamp: str,
#     cycle_name: str,
#     cycle_summary: dict[str, Any],
# ) -> Path:
#     cycle_dir = get_real_deploy_cycle_dir(
#         settings=settings,
#         scenario_name=scenario_name,
#         loop_timestamp=loop_timestamp,
#         cycle_name=cycle_name,
#     )
#     ensure_dir(cycle_dir)
#     return save_json_file(cycle_dir / "cycle_summary.json", cycle_summary)


# def save_real_deploy_artifacts(
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
#     output_dir = get_real_deploy_output_dir(settings, scenario_name, loop_timestamp)
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


# # ============================================================
# # SCENE OBJECT LIST / VALIDATOR HELPERS
# # ============================================================

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


# # ============================================================
# # MODULE EXECUTION HELPERS
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
#     pipeline_config: dict[str, Any],
#     image_path: str,
# ) -> dict[str, Any]:
#     module_name = "scene_description"
#     base_prompt = load_base_prompt(settings, module_name, version)

#     system_prompt = base_prompt
#     user_text = "Analyze the scene and return the structured JSON output."

#     prompt_path = save_rendered_prompt_for_cycle(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=model_name,
#         cycle_name=cycle_name,
#         prompt_text=system_prompt,
#     )

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

#     parsed_path, run_info_path = save_module_outputs_for_cycle(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=result["model_name"],
#         cycle_name=cycle_name,
#         cycle_index=cycle_idx,
#         cycle_timestamp=cycle_timestamp,
#         deployment_name=result["deployment_name"],
#         execution_time_seconds=result["execution_time_seconds"],
#         scenario_data=scenario_context,
#         parsed_response=parsed_response,
#         execution_mode="real_deploy_validation_loop",
#         dependencies=None,
#         pipeline_config=pipeline_config,
#     )

#     scene_object_list_path = build_scene_object_list_from_cycle(
#         scenario=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model=result["model_name"],
#         cycle_name=cycle_name,
#     )

#     print(f"[OK][scene_description] Prompt saved to:         {prompt_path}")
#     print(f"[OK][scene_description] Parsed output saved to:  {parsed_path}")
#     print(f"[OK][scene_description] Run info saved to:       {run_info_path}")
#     print(f"[OK][scene_description] Scene object list saved: {scene_object_list_path}")
#     print(f"[OK][scene_description] Execution time:          {result['execution_time_seconds']:.3f}s")

#     return {
#         "output": parsed_response,
#         "paths": {
#             "prompt": str(prompt_path),
#             "response_parsed": str(parsed_path),
#             "run_info": str(run_info_path),
#             "scene_object_list": str(scene_object_list_path),
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
#     scene_description: Any,
#     pipeline_config: dict[str, Any],
#     topic: str,
#     timeout_sec: float,
#     safety_threshold: float,
#     include_debug_mapping: bool,
# ) -> dict[str, Any]:
#     start_time = time.perf_counter()

#     parsed_response = enrich_scene(
#         input_data=scene_description,
#         safety_threshold=safety_threshold,
#         pose_source="gazebo",
#         topic=topic,
#         timeout_sec=timeout_sec,
#         include_debug_mapping=include_debug_mapping,
#     )

#     execution_time_seconds = time.perf_counter() - start_time

#     dependencies = {
#         "scene_description": {
#             "prompt_version": version,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": model_name,
#         }
#     }

#     parsed_path, run_info_path = save_scene_description_full_artifact_for_cycle(
#         settings=settings,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=model_name,
#         cycle_name=cycle_name,
#         cycle_index=cycle_idx,
#         cycle_timestamp=cycle_timestamp,
#         parsed_response=parsed_response,
#         scenario_data=scenario_context,
#         execution_time_seconds=execution_time_seconds,
#         dependencies=dependencies,
#         pipeline_config=pipeline_config,
#         pose_file=None,
#         safety_threshold=safety_threshold,
#         include_debug_mapping=include_debug_mapping,
#         execution_mode="real_deploy_validation_loop_side_artifact",
#     )

#     print(f"[OK][scene_description_full] Pose source:          gazebo")
#     print(f"[OK][scene_description_full] Gazebo topic:         {topic}")
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

#     prompt_path = save_rendered_prompt_for_cycle(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=model_name,
#         cycle_name=cycle_name,
#         prompt_text=system_prompt,
#     )

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
#         }
#     }

#     parsed_path, run_info_path = save_module_outputs_for_cycle(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=result["model_name"],
#         cycle_name=cycle_name,
#         cycle_index=cycle_idx,
#         cycle_timestamp=cycle_timestamp,
#         deployment_name=result["deployment_name"],
#         execution_time_seconds=result["execution_time_seconds"],
#         scenario_data=scenario_context,
#         parsed_response=parsed_response,
#         execution_mode="real_deploy_validation_loop",
#         dependencies=dependencies,
#         pipeline_config=pipeline_config,
#     )

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

#     prompt_path = save_rendered_prompt_for_cycle(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=model_name,
#         cycle_name=cycle_name,
#         prompt_text=system_prompt,
#     )

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
#         },
#         "vlm_planning": {
#             "prompt_version": plan_version,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": plan_model,
#         },
#     }

#     parsed_path, run_info_path = save_module_outputs_for_cycle(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         loop_timestamp=loop_timestamp,
#         model_name=result["model_name"],
#         cycle_name=cycle_name,
#         cycle_index=cycle_idx,
#         cycle_timestamp=cycle_timestamp,
#         deployment_name=result["deployment_name"],
#         execution_time_seconds=result["execution_time_seconds"],
#         scenario_data=scenario_context,
#         parsed_response=parsed_response,
#         execution_mode="real_deploy_validation_loop",
#         dependencies=dependencies,
#         pipeline_config=pipeline_config,
#     )

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

#     prompt_dir = get_validator_prompt_cycle_dir(
#         settings=settings,
#         scenario_name=scenario_name,
#         version=validator_version,
#         loop_timestamp=loop_timestamp,
#         model_name=validator_model,
#         cycle_name=cycle_name,
#         stage_name=stage_name,
#         condition_kind=condition_kind,
#     )
#     prompt_path = prompt_dir / "prompt.txt"
#     write_text(prompt_path, system_prompt)

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
#         },
#         "vlm_planning": {
#             "prompt_version": plan_version,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": plan_model,
#         },
#         "simultaneous_actions": {
#             "prompt_version": sim_version,
#             "loop_timestamp": loop_timestamp,
#             "cycle_name": cycle_name,
#             "model": sim_model,
#         },
#     }

#     output_dir = get_validator_output_cycle_dir(
#         settings=settings,
#         scenario_name=scenario_name,
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
#         "execution_mode": "real_deploy_validation_loop",
#         "scenario_name": scenario_name,
#         "prompt_version": validator_version,
#         "loop_timestamp": loop_timestamp,
#         "cycle_name": cycle_name,
#         "cycle_index": cycle_idx,
#         "cycle_timestamp": cycle_timestamp,
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
# # DEPLOY HELPERS
# # ============================================================

# def deploy_stage_1(
#     manip_dir: Path,
#     screens_dir: Path,
#     stage_id: int,
# ) -> str:
#     run_python_script(manip_dir / "grasp_box_yz.py", label="stage_1/grasp_box_yz.py")
#     next_image = take_screenshot(screens_dir)
#     return next_image


# def deploy_stage_2(
#     manip_dir: Path,
#     screens_dir: Path,
#     stage_id: int,
# ) -> str:
#     run_python_script(manip_dir / "place_box.py", label="stage_2/place_box.py")
#     run_python_script(manip_dir / "place_box_2.py", label="stage_2/place_box_2.py")
#     run_python_script(manip_dir / "homing.py", label="stage_2/homing.py")
#     next_image = take_screenshot(screens_dir)
#     return next_image


# def deploy_stage_3(
#     manip_dir: Path,
#     screens_dir: Path,
#     stage_id: int,
# ) -> str:
#     run_python_script(manip_dir / "grasp_box_xy.py", label="stage_3/grasp_box_xy.py")
#     next_image = take_screenshot(screens_dir)
#     return next_image


# def execute_stage_deploy(
#     stage_id: int,
#     manip_dir: Path,
#     screens_dir: Path,
# ) -> str:
#     if stage_id == 1:
#         return deploy_stage_1(
#             manip_dir=manip_dir,
#             screens_dir=screens_dir,
#             stage_id=stage_id,
#         )

#     if stage_id == 2:
#         return deploy_stage_2(
#             manip_dir=manip_dir,
#             screens_dir=screens_dir,
#             stage_id=stage_id,
#         )

#     if stage_id == 3:
#         return deploy_stage_3(
#             manip_dir=manip_dir,
#             screens_dir=screens_dir,
#             stage_id=stage_id,
#         )

#     raise ValueError(
#         f"No deploy routine defined for Stage_id={stage_id}. "
#         "This script currently supports stage ids 1, 2, 3."
#     )


# # ============================================================
# # SUMMARY HELPERS
# # ============================================================

# def build_run_info(full_summary: dict[str, Any]) -> dict[str, Any]:
#     return {
#         "module": "real_deploy_validation_loop",
#         "scenario_name": full_summary["scenario_name"],
#         "loop_timestamp": full_summary["loop_timestamp"],
#         "timestamp": full_summary["timestamp"],
#         "initial_image_path": full_summary["initial_image_path"],
#         "screens_dir": full_summary["screens_dir"],
#         "config": full_summary["config"],
#     }


# def build_loop_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
#     return {
#         "module": "real_deploy_validation_loop_summary",
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


# # ============================================================
# # MAIN
# # ============================================================

# def main() -> None:
#     parser = build_parser()
#     args = parser.parse_args()

#     validate_args(args)

#     settings = load_settings()
#     scenario_data = load_scenario(settings, args.scenario)

#     loop_timestamp = make_experiment_timestamp()
#     manip_dir = Path(args.manip_dir).resolve()

#     screens_dir = get_screens_root(
#         settings=settings,
#         scenario_name=args.scenario,
#         loop_timestamp=loop_timestamp,
#         screens_subdir=args.screens_subdir,
#     )
#     ensure_dir(screens_dir)

#     initial_image_path = take_screenshot(screens_dir)
#     current_image = initial_image_path
#     task_completed = False
#     cycle_idx = 0

#     full_summary: dict[str, Any] = {
#         "module": "full_pipeline_summary",
#         "scenario_name": args.scenario,
#         "loop_timestamp": loop_timestamp,
#         "timestamp": datetime.now().isoformat(),
#         "initial_image_path": str(Path(initial_image_path).resolve()),
#         "screens_dir": str(screens_dir.resolve()),
#         "manip_dir": str(manip_dir),
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
#         print(f"REAL DEPLOY CYCLE STARTED | cycle={cycle_idx} | {cycle_name}")
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
#             loop_timestamp=loop_timestamp,
#         )

#         cycle_record: dict[str, Any] = {
#             "cycle_name": cycle_name,
#             "cycle_index": cycle_idx,
#             "cycle_timestamp": cycle_timestamp,
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
#                 scene_description=scene_description_artifact["output"],
#                 pipeline_config=pipeline_config,
#                 topic=args.grounding_topic,
#                 timeout_sec=args.grounding_timeout_sec,
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

#                 pre_artifact = execute_validator_step(
#                     settings=settings,
#                     scenario_name=args.scenario,
#                     validator_version=args.validator_v,
#                     validator_model=args.validator_model,
#                     loop_timestamp=loop_timestamp,
#                     cycle_name=cycle_name,
#                     cycle_idx=cycle_idx,
#                     cycle_timestamp=cycle_timestamp,
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

#                     if full_summary["replans_done"] >= args.max_replans:
#                         raise RuntimeError(
#                             f"Maximum number of replans reached ({args.max_replans})."
#                         )

#                     full_summary["replans_done"] += 1
#                     all_stages_succeeded = False
#                     break

#                 # --------------------------------------------------
#                 # REAL DEPLOY
#                 # --------------------------------------------------
#                 print(f"\n[LOOP] Stage {stage_id} REAL DEPLOY")
#                 next_image = execute_stage_deploy(
#                     stage_id=stage_id,
#                     manip_dir=manip_dir,
#                     screens_dir=screens_dir,
#                 )

#                 stage_record["next_image_path"] = str(Path(next_image).resolve())
#                 stage_record["next_image_name"] = Path(next_image).name
#                 stage_record["post_image_path"] = str(Path(next_image).resolve())
#                 stage_record["post_image_name"] = Path(next_image).name

#                 print(f"[LOOP] NEXT image:     {next_image}")

#                 # --------------------------------------------------
#                 # POST VALIDATION
#                 # --------------------------------------------------
#                 print(f"\n[LOOP] Stage {stage_id} POST")
#                 print(f"[LOOP] POST image:     {next_image}")
#                 print(f"[LOOP] POST condition: {post_condition}")

#                 post_artifact = execute_validator_step(
#                     settings=settings,
#                     scenario_name=args.scenario,
#                     validator_version=args.validator_v,
#                     validator_model=args.validator_model,
#                     loop_timestamp=loop_timestamp,
#                     cycle_name=cycle_name,
#                     cycle_idx=cycle_idx,
#                     cycle_timestamp=cycle_timestamp,
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

#                     if full_summary["replans_done"] >= args.max_replans:
#                         raise RuntimeError(
#                             f"Maximum number of replans reached ({args.max_replans})."
#                         )

#                     full_summary["replans_done"] += 1
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
#         print(f"[OK][real_deploy_validation_loop] Cycle summary saved to: {cycle_summary_path}")

#         if cycle_error:
#             break

#     run_info = build_run_info(full_summary)
#     loop_summary = build_loop_summary(full_summary)
#     scene_description_summary = build_scene_description_summary(full_summary)
#     vlm_planning_summary = build_vlm_planning_summary(full_summary)
#     simultaneous_actions_summary = build_simultaneous_actions_summary(full_summary)
#     validator_summary = build_validator_summary(full_summary)
#     full_pipeline_summary = build_full_pipeline_summary(full_summary)

#     summary_paths = save_real_deploy_artifacts(
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
#     print("REAL DEPLOY VALIDATION LOOP COMPLETED")
#     print(f"Scenario:                  {args.scenario}")
#     print(f"Loop timestamp:            {loop_timestamp}")
#     print(f"Task completed:            {full_summary['task_completed']}")
#     print(f"Replans done:              {full_summary['replans_done']}")
#     print(f"Run info saved:            {summary_paths['run_info']}")
#     print(f"Loop summary saved:        {summary_paths['loop_summary']}")
#     print(f"Scene summary saved:       {summary_paths['scene_description_summary']}")
#     print(f"Planning summary saved:    {summary_paths['vlm_planning_summary']}")
#     print(f"Sim-actions summary saved: {summary_paths['simultaneous_actions_summary']}")
#     print(f"Validator summary saved:   {summary_paths['validator_summary']}")
#     print(f"Full summary saved:        {summary_paths['real_deploy_validation_loop_full_summary']}")
#     print("======================================================")


# if __name__ == "__main__":
#     main()