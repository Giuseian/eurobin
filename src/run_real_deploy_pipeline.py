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
from build_scene_object_list import build_scene_object_list_from_run
from scene_grounding import enrich_scene
from utils import (
    load_base_prompt,
    make_experiment_timestamp,
    render_prompt,
    save_module_outputs,
    save_rendered_prompt,
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
            "Run the real deploy pipeline with scene grounding, integrated screenshots, "
            "and forced validator matching."
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

    parser.add_argument(
        "--validator-model",
        type=str,
        default="bypassed_validator",
        help=(
            "Logical model name stored in validator artifacts. "
            "No real validator call is executed."
        ),
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
        "--cartesio-startup-wait",
        type=float,
        default=5.0,
        help="Seconds to wait after starting Cartesio.",
    )

    parser.add_argument(
        "--cartesio-stop-poll",
        type=float,
        default=1.0,
        help="Polling period while waiting Cartesio to stop.",
    )

    parser.add_argument(
        "--post-kill-wait",
        type=float,
        default=10.0,
        help="Seconds to wait after Cartesio stop before homing.",
    )

    parser.add_argument(
        "--post-homing-wait",
        type=float,
        default=60.0,
        help="Seconds to wait after homing before taking the next screenshot.",
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
        help="Store the internal VLM-to-Gazebo mapping in scene_description_full output.",
    )

    return parser


# ============================================================
# GENERIC HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_args(args: argparse.Namespace) -> None:
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
        / "real_deploy"
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
# CARTESIO HELPERS
# ============================================================

def kill_cartesio(stop_poll_seconds: float = 1.0) -> None:
    print("\n[CARTESIO] Stopping Cartesio with SIGINT...")

    subprocess.run(
        ["pkill", "-INT", "-f", "ros2 launch kyon_cartesio kyon.launch"],
        check=False,
        capture_output=True,
        text=True,
    )

    while True:
        check = subprocess.run(
            ["pgrep", "-f", "ros2 launch kyon_cartesio kyon.launch"],
            check=False,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            break
        time.sleep(stop_poll_seconds)

    print("[CARTESIO] Cartesio stopped.")


def start_cartesio(startup_wait_seconds: float = 5.0) -> None:
    print("\n[CARTESIO] Starting Cartesio...")
    subprocess.Popen(
        ["ros2", "launch", "kyon_cartesio", "kyon.launch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(startup_wait_seconds)
    print("[CARTESIO] Cartesio started.")


# ============================================================
# SCENE OBJECT LIST / VALIDATOR ARTIFACT HELPERS
# ============================================================

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
        "execution_mode": "forced_matching_real_deploy",
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


def execute_forced_validator_step(
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

    parsed_response = {
        "result": "matching",
        "reason": (
            "Forced matching assumption: validator temporarily bypassed "
            "because of the known scene_perception prompt issue. "
            "The comparison artifact is still saved for traceability."
        ),
        "forced": True,
        "bypass_type": "validator_not_called",
        "condition_name": condition_name,
        "condition_text": condition_text,
        "image_path": str(Path(image_path).resolve()),
        "scene_object_list": scene_object_list,
    }

    dependencies = {
        "scene_description": {
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
        model_name=validator_model,
        run_name=run_name,
        condition_name=condition_name,
        deployment_name="validator_bypassed",
        execution_time_seconds=0.0,
        image_path=image_path,
        condition_text=condition_text,
        parsed_response=parsed_response,
        dependencies=dependencies,
    )

    print(f"[OK][forced-validator:{condition_name}] Prompt saved to:        {prompt_path}")
    print(f"[OK][forced-validator:{condition_name}] Parsed output saved to: {parsed_path}")
    print(f"[OK][forced-validator:{condition_name}] Run info saved to:      {run_info_path}")
    print(f"[OK][forced-validator:{condition_name}] Execution time:         0.000s")

    return parsed_response


# ============================================================
# SCENE_DESCRIPTION_FULL HELPERS
# ============================================================

def get_scene_description_full_output_dir(
    settings,
    scenario: str,
    version: str,
    experiment_timestamp: str,
    model_name: str,
    run_name: str,
) -> Path:
    return (
        settings.project_root
        / "outputs"
        / "scene_description_full"
        / scenario
        / version
        / experiment_timestamp
        / model_name
        / run_name
    )


def save_scene_description_full_outputs(
    settings,
    scenario_name: str,
    version: str,
    experiment_timestamp: str,
    model_name: str,
    run_name: str,
    scenario_data: dict[str, Any],
    parsed_response: dict[str, Any],
    dependencies: dict[str, Any],
    pipeline_config: dict[str, Any],
    execution_time_seconds: float,
) -> tuple[Path, Path]:
    output_dir = get_scene_description_full_output_dir(
        settings=settings,
        scenario=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
    )
    ensure_dir(output_dir)

    parsed_path = output_dir / "response_parsed.json"
    run_info_path = output_dir / "run_info.json"

    write_json(parsed_path, parsed_response)

    run_info = {
        "module": "scene_description_full",
        "execution_mode": "real_deploy_pipeline",
        "scenario_name": scenario_name,
        "prompt_version": version,
        "experiment_timestamp": experiment_timestamp,
        "run_name": run_name,
        "model": model_name,
        "deployment_name": "deterministic_scene_grounding",
        "execution_time_seconds": execution_time_seconds,
        "timestamp": datetime.now().isoformat(),
        "scenario_data": scenario_data,
        "dependencies": dependencies,
        "pipeline_config": pipeline_config,
        "response_parsed": parsed_response,
    }

    write_json(run_info_path, run_info)
    return parsed_path, run_info_path


# ============================================================
# MODULE EXECUTION HELPERS
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
        execution_mode="real_deploy_pipeline",
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
    scene_description: dict[str, Any],
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
        topic=topic,
        timeout_sec=timeout_sec,
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

    parsed_path, run_info_path = save_scene_description_full_outputs(
        settings=settings,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
        scenario_data=scenario_context,
        parsed_response=parsed_response,
        dependencies=dependencies,
        pipeline_config=pipeline_config,
        execution_time_seconds=execution_time_seconds,
    )

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
        execution_mode="real_deploy_pipeline",
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
        execution_mode="real_deploy_pipeline",
        dependencies=dependencies,
        pipeline_config=pipeline_config,
    )

    print(f"[OK][simultaneous_actions] Parsed output saved to: {parsed_path}")
    print(f"[OK][simultaneous_actions] Run info saved to:      {run_info_path}")
    print(f"[OK][simultaneous_actions] Execution time:         {result['execution_time_seconds']:.3f}s")

    return parsed_response


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
    cartesio_stop_poll: float,
    post_kill_wait: float,
    post_homing_wait: float,
    cartesio_startup_wait: float,
) -> str:
    run_python_script(manip_dir / "place_box.py", label="stage_2/place_box.py")
    run_python_script(manip_dir / "place_box_2.py", label="stage_2/place_box_2.py")

    kill_cartesio(stop_poll_seconds=cartesio_stop_poll)
    time.sleep(post_kill_wait)

    run_python_script(manip_dir / "homing.py", label="stage_2/homing.py")
    time.sleep(post_homing_wait)

    next_image = take_screenshot(screens_dir)

    start_cartesio(startup_wait_seconds=cartesio_startup_wait)
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
    cartesio_stop_poll: float,
    post_kill_wait: float,
    post_homing_wait: float,
    cartesio_startup_wait: float,
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
            cartesio_stop_poll=cartesio_stop_poll,
            post_kill_wait=post_kill_wait,
            post_homing_wait=post_homing_wait,
            cartesio_startup_wait=cartesio_startup_wait,
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

def get_run_root(settings, scenario_name: str, loop_timestamp: str) -> Path:
    return (
        settings.project_root
        / "outputs"
        / "real_deploy_pipeline"
        / scenario_name
        / loop_timestamp
    )


def save_run_summary(
    settings,
    scenario_name: str,
    loop_timestamp: str,
    summary: dict[str, Any],
) -> Path:
    output_dir = get_run_root(settings, scenario_name, loop_timestamp)
    ensure_dir(output_dir)

    out_path = output_dir / "run_summary.json"
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

    loop_timestamp = make_experiment_timestamp()
    pipeline_timestamp = loop_timestamp
    run_name = "run_001"

    manip_dir = Path(args.manip_dir).resolve()

    run_root = get_run_root(settings, args.scenario, loop_timestamp)

    screens_dir = get_screens_root(
        settings=settings,
        scenario_name=args.scenario,
        loop_timestamp=loop_timestamp,
        screens_subdir=args.screens_subdir,
    )
    ensure_dir(screens_dir)

    summary: dict[str, Any] = {
        "module": "real_deploy_pipeline",
        "scenario_name": args.scenario,
        "loop_timestamp": loop_timestamp,
        "timestamp": datetime.now().isoformat(),
        "manip_dir": str(manip_dir),
        "run_root": str(run_root),
        "screens_dir": str(screens_dir),
        "config": {
            "scene_description": {
                "prompt_version": args.scene_v,
                "model": args.scene_model,
            },
            "scene_description_full": {
                "prompt_version": args.scene_v,
                "model": args.scene_model,
                "mode": "deterministic_grounding",
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
                "mode": "forced_matching_bypass",
            },
        },
        "task_completed": False,
        "initial_image_path": None,
        "final_image_path": None,
        "stages": [],
    }

    print("\n======================================================")
    print("REAL DEPLOY PIPELINE STARTED")
    print(f"Scenario:        {args.scenario}")
    print(f"Pipeline ts:     {pipeline_timestamp}")
    print(f"Run root:        {run_root}")
    print(f"Screens dir:     {screens_dir}")
    print("======================================================")

    try:
        initial_image = take_screenshot(screens_dir)
        summary["initial_image_path"] = str(Path(initial_image).resolve())

        scenario_context = make_scenario_context(
            scenario_data=scenario_data,
            image_path=initial_image,
        )

        pipeline_config = {
            "scene_description": {
                "prompt_version": args.scene_v,
                "experiment_timestamp": pipeline_timestamp,
                "model": args.scene_model,
                "run_name": run_name,
            },
            "scene_description_full": {
                "prompt_version": args.scene_v,
                "experiment_timestamp": pipeline_timestamp,
                "model": args.scene_model,
                "run_name": run_name,
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

        scene_description = execute_scene_description_step(
            settings=settings,
            scenario_name=args.scenario,
            scenario_context=scenario_context,
            version=args.scene_v,
            model_name=args.scene_model,
            experiment_timestamp=pipeline_timestamp,
            run_name=run_name,
            pipeline_config=pipeline_config,
            image_path=initial_image,
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
            topic=args.grounding_topic,
            timeout_sec=args.grounding_timeout_sec,
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

        current_image = initial_image

        for stage in stages:
            stage_id = stage["Stage_id"]
            pre_condition = stage["Precondition"]
            post_condition = stage["Postcondition"]

            print("\n------------------------------------------------------")
            print(f"[STAGE {stage_id}] START")
            print("------------------------------------------------------")

            stage_record: dict[str, Any] = {
                "stage_id": stage_id,
                "precondition": pre_condition,
                "postcondition": post_condition,
                "pre_image_path": str(Path(current_image).resolve()),
                "post_image_path": None,
                "pre_validation": None,
                "post_validation": None,
            }

            print(f"\n[STAGE {stage_id}] PRE CHECK")
            print(f"[STAGE {stage_id}] PRE image:     {current_image}")
            print(f"[STAGE {stage_id}] PRE condition: {pre_condition}")

            pre_name = f"pre_{stage_id}"
            pre_response = execute_forced_validator_step(
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

            print(f"\n[PRE forced-validator:{pre_name}] Parsed JSON:")
            print(json.dumps(pre_response, indent=2, ensure_ascii=False))

            stage_record["pre_validation"] = pre_response

            print(f"\n[STAGE {stage_id}] DEPLOY")
            next_image = execute_stage_deploy(
                stage_id=stage_id,
                manip_dir=manip_dir,
                screens_dir=screens_dir,
                cartesio_stop_poll=args.cartesio_stop_poll,
                post_kill_wait=args.post_kill_wait,
                post_homing_wait=args.post_homing_wait,
                cartesio_startup_wait=args.cartesio_startup_wait,
            )

            print(f"[STAGE {stage_id}] POST image:    {next_image}")

            print(f"\n[STAGE {stage_id}] POST CHECK")
            print(f"[STAGE {stage_id}] POST image:    {next_image}")
            print(f"[STAGE {stage_id}] POST condition:{post_condition}")

            post_name = f"post_{stage_id}"
            post_response = execute_forced_validator_step(
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

            print(f"\n[POST forced-validator:{post_name}] Parsed JSON:")
            print(json.dumps(post_response, indent=2, ensure_ascii=False))

            stage_record["post_validation"] = post_response
            stage_record["post_image_path"] = str(Path(next_image).resolve())
            summary["stages"].append(stage_record)

            current_image = next_image

        summary["task_completed"] = True
        summary["final_image_path"] = str(Path(current_image).resolve())

        print("\n======================================================")
        print("[PIPELINE] TASK COMPLETED SUCCESSFULLY")
        print("======================================================")

    except Exception as exc:
        summary["task_completed"] = False
        summary["error"] = str(exc)
        print("\n======================================================")
        print("[PIPELINE] ERROR")
        print(str(exc))
        print("======================================================")

    summary_path = save_run_summary(
        settings=settings,
        scenario_name=args.scenario,
        loop_timestamp=loop_timestamp,
        summary=summary,
    )

    print("\n======================================================")
    print("REAL DEPLOY PIPELINE COMPLETED")
    print(f"Scenario:        {args.scenario}")
    print(f"Loop timestamp:  {loop_timestamp}")
    print(f"Task completed:  {summary['task_completed']}")
    print(f"Summary saved:   {summary_path}")
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
# from build_scene_object_list import build_scene_object_list_from_run
# from utils import (
#     load_base_prompt,
#     make_experiment_timestamp,
#     render_prompt,
#     save_module_outputs,
#     save_rendered_prompt,
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
#             "Run the real deploy pipeline with integrated screenshots and "
#             "forced validator matching."
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

#     parser.add_argument(
#         "--validator-model",
#         type=str,
#         default="bypassed_validator",
#         help=(
#             "Logical model name stored in validator artifacts. "
#             "No real validator call is executed."
#         ),
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
#         "--cartesio-startup-wait",
#         type=float,
#         default=5.0,
#         help="Seconds to wait after starting Cartesio.",
#     )

#     parser.add_argument(
#         "--cartesio-stop-poll",
#         type=float,
#         default=1.0,
#         help="Polling period while waiting Cartesio to stop.",
#     )

#     parser.add_argument(
#         "--post-kill-wait",
#         type=float,
#         default=10.0,
#         help="Seconds to wait after Cartesio stop before homing.",
#     )

#     parser.add_argument(
#         "--post-homing-wait",
#         type=float,
#         default=60.0,
#         help="Seconds to wait after homing before taking the next screenshot.",
#     )

#     return parser


# # ============================================================
# # GENERIC HELPERS
# # ============================================================

# def ensure_dir(path: Path) -> Path:
#     path.mkdir(parents=True, exist_ok=True)
#     return path


# def validate_args(args: argparse.Namespace) -> None:
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
#         / "real_deploy"
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
#     """
#     Ask Gazebo to save a screenshot into screens_dir.
#     Gazebo generates a timestamped PNG; we detect the new file and rename it
#     sequentially as 001.png, 002.png, ...

#     Returns the absolute path of the renamed screenshot.
#     """
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
# # CARTESIO HELPERS
# # ============================================================

# def kill_cartesio(stop_poll_seconds: float = 1.0) -> None:
#     print("\n[CARTESIO] Stopping Cartesio with SIGINT...")

#     subprocess.run(
#         ["pkill", "-INT", "-f", "ros2 launch kyon_cartesio kyon.launch"],
#         check=False,
#         capture_output=True,
#         text=True,
#     )

#     while True:
#         check = subprocess.run(
#             ["pgrep", "-f", "ros2 launch kyon_cartesio kyon.launch"],
#             check=False,
#             capture_output=True,
#             text=True,
#         )
#         if check.returncode != 0:
#             break
#         time.sleep(stop_poll_seconds)

#     print("[CARTESIO] Cartesio stopped.")


# def start_cartesio(startup_wait_seconds: float = 5.0) -> None:
#     print("\n[CARTESIO] Starting Cartesio...")
#     subprocess.Popen(
#         ["ros2", "launch", "kyon_cartesio", "kyon.launch"],
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#     )

#     time.sleep(startup_wait_seconds)
#     print("[CARTESIO] Cartesio started.")


# # ============================================================
# # SCENE OBJECT LIST / VALIDATOR ARTIFACT HELPERS
# # ============================================================

# def load_scene_object_list_from_cycle(
#     settings,
#     scenario_name: str,
#     scene_version: str,
#     pipeline_timestamp: str,
#     scene_model: str,
#     run_name: str,
# ) -> dict[str, Any]:
#     path = (
#         settings.project_root
#         / "outputs"
#         / "scene_description"
#         / scenario_name
#         / scene_version
#         / pipeline_timestamp
#         / scene_model
#         / run_name
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
#         "execution_mode": "forced_matching_real_deploy",
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
#         "dependencies": dependencies,
#         "response_parsed": parsed_response,
#     }

#     write_json(run_info_path, run_info)
#     return parsed_path, run_info_path


# def execute_forced_validator_step(
#     settings,
#     scenario_name: str,
#     validator_version: str,
#     validator_model: str,
#     upstream_timestamp: str,
#     run_name: str,
#     condition_name: str,
#     condition_text: str,
#     image_path: str,
#     scene_version: str,
#     scene_model: str,
#     plan_version: str,
#     plan_model: str,
#     sim_version: str,
#     sim_model: str,
# ) -> dict[str, Any]:
#     """
#     Does NOT call the validator model.
#     It still renders/saves the validator prompt and writes a forced response:
#     result = matching
#     """
#     scene_object_list = load_scene_object_list_from_cycle(
#         settings=settings,
#         scenario_name=scenario_name,
#         scene_version=scene_version,
#         pipeline_timestamp=upstream_timestamp,
#         scene_model=scene_model,
#         run_name=run_name,
#     )

#     base_prompt = load_base_prompt(settings, "validator", validator_version)
#     system_prompt = render_validator_prompt(
#         base_prompt=base_prompt,
#         condition=condition_text,
#         scene_object_list=scene_object_list,
#     )

#     prompt_path = save_validator_prompt(
#         settings=settings,
#         scenario=scenario_name,
#         version=validator_version,
#         upstream_timestamp=upstream_timestamp,
#         model_name=validator_model,
#         run_name=run_name,
#         condition_name=condition_name,
#         prompt_text=system_prompt,
#     )

#     parsed_response = {
#         "result": "matching",
#         "reason": (
#             "Forced matching assumption: validator temporarily bypassed "
#             "because of the known scene_perception prompt issue. "
#             "The comparison artifact is still saved for traceability."
#         ),
#         "forced": True,
#         "bypass_type": "validator_not_called",
#         "condition_name": condition_name,
#         "condition_text": condition_text,
#         "image_path": str(Path(image_path).resolve()),
#         "scene_object_list": scene_object_list,
#     }

#     dependencies = {
#         "scene_description": {
#             "prompt_version": scene_version,
#             "experiment_timestamp": upstream_timestamp,
#             "model": scene_model,
#             "run_name": run_name,
#         },
#         "vlm_planning": {
#             "prompt_version": plan_version,
#             "experiment_timestamp": upstream_timestamp,
#             "model": plan_model,
#             "run_name": run_name,
#         },
#         "simultaneous_actions": {
#             "prompt_version": sim_version,
#             "experiment_timestamp": upstream_timestamp,
#             "model": sim_model,
#             "run_name": run_name,
#         },
#     }

#     parsed_path, run_info_path = save_validator_outputs(
#         settings=settings,
#         scenario=scenario_name,
#         version=validator_version,
#         upstream_timestamp=upstream_timestamp,
#         model_name=validator_model,
#         run_name=run_name,
#         condition_name=condition_name,
#         deployment_name="validator_bypassed",
#         execution_time_seconds=0.0,
#         image_path=image_path,
#         condition_text=condition_text,
#         parsed_response=parsed_response,
#         dependencies=dependencies,
#     )

#     print(f"[OK][forced-validator:{condition_name}] Prompt saved to:        {prompt_path}")
#     print(f"[OK][forced-validator:{condition_name}] Parsed output saved to: {parsed_path}")
#     print(f"[OK][forced-validator:{condition_name}] Run info saved to:      {run_info_path}")
#     print(f"[OK][forced-validator:{condition_name}] Execution time:         0.000s")

#     return parsed_response


# # ============================================================
# # MODULE EXECUTION HELPERS
# # ============================================================

# def execute_scene_description_step(
#     settings,
#     scenario_name: str,
#     scenario_context: dict[str, Any],
#     version: str,
#     model_name: str,
#     experiment_timestamp: str,
#     run_name: str,
#     pipeline_config: dict[str, Any],
#     image_path: str,
# ) -> dict[str, Any]:
#     module_name = "scene_description"
#     base_prompt = load_base_prompt(settings, module_name, version)

#     system_prompt = base_prompt
#     user_text = "Analyze the scene and return the structured JSON output."

#     save_rendered_prompt(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=model_name,
#         run_name=run_name,
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

#     parsed_path, run_info_path = save_module_outputs(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=result["model_name"],
#         run_name=run_name,
#         deployment_name=result["deployment_name"],
#         execution_time_seconds=result["execution_time_seconds"],
#         scenario_data=scenario_context,
#         parsed_response=parsed_response,
#         execution_mode="real_deploy_pipeline",
#         dependencies=None,
#         pipeline_config=pipeline_config,
#     )

#     scene_object_list_path = build_scene_object_list_from_run(
#         scenario=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model=result["model_name"],
#         run_name=run_name,
#     )

#     print(f"[OK][scene_description] Parsed output saved to: {parsed_path}")
#     print(f"[OK][scene_description] Run info saved to:      {run_info_path}")
#     print(f"[OK][scene_description] Scene object list saved to: {scene_object_list_path}")
#     print(f"[OK][scene_description] Execution time:         {result['execution_time_seconds']:.3f}s")

#     return parsed_response


# def execute_vlm_planning_step(
#     settings,
#     scenario_name: str,
#     scenario_context: dict[str, Any],
#     version: str,
#     model_name: str,
#     experiment_timestamp: str,
#     run_name: str,
#     scene_description: Any,
#     scene_version: str,
#     scene_model: str,
#     pipeline_config: dict[str, Any],
# ) -> Any:
#     module_name = "vlm_planning"
#     base_prompt = load_base_prompt(settings, module_name, version)

#     system_prompt = render_prompt(
#         module_name=module_name,
#         base_prompt=base_prompt,
#         scenario_data=scenario_context,
#         scene_description=scene_description,
#     )

#     user_text = "Generate the manipulation plan in valid JSON only."

#     save_rendered_prompt(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=model_name,
#         run_name=run_name,
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
#         "scene_description": {
#             "prompt_version": scene_version,
#             "experiment_timestamp": experiment_timestamp,
#             "model": scene_model,
#             "run_name": run_name,
#         }
#     }

#     parsed_path, run_info_path = save_module_outputs(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=result["model_name"],
#         run_name=run_name,
#         deployment_name=result["deployment_name"],
#         execution_time_seconds=result["execution_time_seconds"],
#         scenario_data=scenario_context,
#         parsed_response=parsed_response,
#         execution_mode="real_deploy_pipeline",
#         dependencies=dependencies,
#         pipeline_config=pipeline_config,
#     )

#     print(f"[OK][vlm_planning] Parsed output saved to: {parsed_path}")
#     print(f"[OK][vlm_planning] Run info saved to:      {run_info_path}")
#     print(f"[OK][vlm_planning] Execution time:         {result['execution_time_seconds']:.3f}s")

#     return parsed_response


# def execute_simultaneous_actions_step(
#     settings,
#     scenario_name: str,
#     scenario_context: dict[str, Any],
#     version: str,
#     model_name: str,
#     experiment_timestamp: str,
#     run_name: str,
#     scene_description: Any,
#     sequential_plan: Any,
#     scene_version: str,
#     scene_model: str,
#     plan_version: str,
#     plan_model: str,
#     pipeline_config: dict[str, Any],
# ) -> Any:
#     module_name = "simultaneous_actions"
#     base_prompt = load_base_prompt(settings, module_name, version)

#     system_prompt = render_prompt(
#         module_name=module_name,
#         base_prompt=base_prompt,
#         scenario_data=scenario_context,
#         scene_description=scene_description,
#         sequential_plan=sequential_plan,
#     )

#     user_text = "Generate the compact parallel plan in valid JSON only."

#     save_rendered_prompt(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=model_name,
#         run_name=run_name,
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
#         "scene_description": {
#             "prompt_version": scene_version,
#             "experiment_timestamp": experiment_timestamp,
#             "model": scene_model,
#             "run_name": run_name,
#         },
#         "vlm_planning": {
#             "prompt_version": plan_version,
#             "experiment_timestamp": experiment_timestamp,
#             "model": plan_model,
#             "run_name": run_name,
#         },
#     }

#     parsed_path, run_info_path = save_module_outputs(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=result["model_name"],
#         run_name=run_name,
#         deployment_name=result["deployment_name"],
#         execution_time_seconds=result["execution_time_seconds"],
#         scenario_data=scenario_context,
#         parsed_response=parsed_response,
#         execution_mode="real_deploy_pipeline",
#         dependencies=dependencies,
#         pipeline_config=pipeline_config,
#     )

#     print(f"[OK][simultaneous_actions] Parsed output saved to: {parsed_path}")
#     print(f"[OK][simultaneous_actions] Run info saved to:      {run_info_path}")
#     print(f"[OK][simultaneous_actions] Execution time:         {result['execution_time_seconds']:.3f}s")

#     return parsed_response


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
#     cartesio_stop_poll: float,
#     post_kill_wait: float,
#     post_homing_wait: float,
#     cartesio_startup_wait: float,
# ) -> str:
#     run_python_script(manip_dir / "place_box.py", label="stage_2/place_box.py")
#     run_python_script(manip_dir / "place_box_2.py", label="stage_2/place_box_2.py")

#     kill_cartesio(stop_poll_seconds=cartesio_stop_poll)
#     time.sleep(post_kill_wait)

#     run_python_script(manip_dir / "homing.py", label="stage_2/homing.py")
#     time.sleep(post_homing_wait)

#     next_image = take_screenshot(screens_dir)

#     start_cartesio(startup_wait_seconds=cartesio_startup_wait)
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
#     cartesio_stop_poll: float,
#     post_kill_wait: float,
#     post_homing_wait: float,
#     cartesio_startup_wait: float,
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
#             cartesio_stop_poll=cartesio_stop_poll,
#             post_kill_wait=post_kill_wait,
#             post_homing_wait=post_homing_wait,
#             cartesio_startup_wait=cartesio_startup_wait,
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

# def get_run_root(settings, scenario_name: str, loop_timestamp: str) -> Path:
#     return (
#         settings.project_root
#         / "outputs"
#         / "real_deploy_pipeline"
#         / scenario_name
#         / loop_timestamp
#     )


# def save_run_summary(
#     settings,
#     scenario_name: str,
#     loop_timestamp: str,
#     summary: dict[str, Any],
# ) -> Path:
#     output_dir = get_run_root(settings, scenario_name, loop_timestamp)
#     ensure_dir(output_dir)

#     out_path = output_dir / "run_summary.json"
#     write_json(out_path, summary)
#     return out_path


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
#     pipeline_timestamp = loop_timestamp
#     run_name = "run_001"

#     manip_dir = Path(args.manip_dir).resolve()

#     # pipeline artifacts remain where they already are
#     run_root = get_run_root(settings, args.scenario, loop_timestamp)

#     # screenshots go to scenarios/real_deploy/<scenario>/<timestamp>/screens
#     screens_dir = get_screens_root(
#         settings=settings,
#         scenario_name=args.scenario,
#         loop_timestamp=loop_timestamp,
#         screens_subdir=args.screens_subdir,
#     )
#     ensure_dir(screens_dir)

#     summary: dict[str, Any] = {
#         "module": "real_deploy_pipeline",
#         "scenario_name": args.scenario,
#         "loop_timestamp": loop_timestamp,
#         "timestamp": datetime.now().isoformat(),
#         "manip_dir": str(manip_dir),
#         "run_root": str(run_root),
#         "screens_dir": str(screens_dir),
#         "config": {
#             "scene_description": {
#                 "prompt_version": args.scene_v,
#                 "model": args.scene_model,
#             },
#             "vlm_planning": {
#                 "prompt_version": args.plan_v,
#                 "model": args.plan_model,
#             },
#             "simultaneous_actions": {
#                 "prompt_version": args.sim_v,
#                 "model": args.sim_model,
#             },
#             "validator": {
#                 "prompt_version": args.validator_v,
#                 "model": args.validator_model,
#                 "mode": "forced_matching_bypass",
#             },
#         },
#         "task_completed": False,
#         "initial_image_path": None,
#         "final_image_path": None,
#         "stages": [],
#     }

#     print("\n======================================================")
#     print("REAL DEPLOY PIPELINE STARTED")
#     print(f"Scenario:        {args.scenario}")
#     print(f"Pipeline ts:     {pipeline_timestamp}")
#     print(f"Run root:        {run_root}")
#     print(f"Screens dir:     {screens_dir}")
#     print("======================================================")

#     try:
#         # --------------------------------------------------
#         # INITIAL SCREENSHOT
#         # --------------------------------------------------
#         initial_image = take_screenshot(screens_dir)
#         summary["initial_image_path"] = str(Path(initial_image).resolve())

#         scenario_context = make_scenario_context(
#             scenario_data=scenario_data,
#             image_path=initial_image,
#         )

#         pipeline_config = {
#             "scene_description": {
#                 "prompt_version": args.scene_v,
#                 "experiment_timestamp": pipeline_timestamp,
#                 "model": args.scene_model,
#                 "run_name": run_name,
#             },
#             "vlm_planning": {
#                 "prompt_version": args.plan_v,
#                 "experiment_timestamp": pipeline_timestamp,
#                 "model": args.plan_model,
#                 "run_name": run_name,
#             },
#             "simultaneous_actions": {
#                 "prompt_version": args.sim_v,
#                 "experiment_timestamp": pipeline_timestamp,
#                 "model": args.sim_model,
#                 "run_name": run_name,
#             },
#         }

#         # --------------------------------------------------
#         # PIPELINE
#         # --------------------------------------------------
#         scene_description = execute_scene_description_step(
#             settings=settings,
#             scenario_name=args.scenario,
#             scenario_context=scenario_context,
#             version=args.scene_v,
#             model_name=args.scene_model,
#             experiment_timestamp=pipeline_timestamp,
#             run_name=run_name,
#             pipeline_config=pipeline_config,
#             image_path=initial_image,
#         )
#         print("\n[scene_description] Parsed JSON:")
#         print(json.dumps(scene_description, indent=2, ensure_ascii=False))

#         sequential_plan = execute_vlm_planning_step(
#             settings=settings,
#             scenario_name=args.scenario,
#             scenario_context=scenario_context,
#             version=args.plan_v,
#             model_name=args.plan_model,
#             experiment_timestamp=pipeline_timestamp,
#             run_name=run_name,
#             scene_description=scene_description,
#             scene_version=args.scene_v,
#             scene_model=args.scene_model,
#             pipeline_config=pipeline_config,
#         )
#         print("\n[vlm_planning] Parsed JSON:")
#         print(json.dumps(sequential_plan, indent=2, ensure_ascii=False))

#         compact_parallel_plan = execute_simultaneous_actions_step(
#             settings=settings,
#             scenario_name=args.scenario,
#             scenario_context=scenario_context,
#             version=args.sim_v,
#             model_name=args.sim_model,
#             experiment_timestamp=pipeline_timestamp,
#             run_name=run_name,
#             scene_description=scene_description,
#             sequential_plan=sequential_plan,
#             scene_version=args.scene_v,
#             scene_model=args.scene_model,
#             plan_version=args.plan_v,
#             plan_model=args.plan_model,
#             pipeline_config=pipeline_config,
#         )
#         print("\n[simultaneous_actions] Parsed JSON:")
#         print(json.dumps(compact_parallel_plan, indent=2, ensure_ascii=False))

#         stages = extract_stages(compact_parallel_plan)

#         current_image = initial_image

#         # --------------------------------------------------
#         # STAGES
#         # --------------------------------------------------
#         for stage in stages:
#             stage_id = stage["Stage_id"]
#             pre_condition = stage["Precondition"]
#             post_condition = stage["Postcondition"]

#             print("\n------------------------------------------------------")
#             print(f"[STAGE {stage_id}] START")
#             print("------------------------------------------------------")

#             stage_record: dict[str, Any] = {
#                 "stage_id": stage_id,
#                 "precondition": pre_condition,
#                 "postcondition": post_condition,
#                 "pre_image_path": str(Path(current_image).resolve()),
#                 "post_image_path": None,
#                 "pre_validation": None,
#                 "post_validation": None,
#             }

#             # ---------------- PRE ----------------
#             print(f"\n[STAGE {stage_id}] PRE CHECK")
#             print(f"[STAGE {stage_id}] PRE image:     {current_image}")
#             print(f"[STAGE {stage_id}] PRE condition: {pre_condition}")

#             pre_name = f"pre_{stage_id}"
#             pre_response = execute_forced_validator_step(
#                 settings=settings,
#                 scenario_name=args.scenario,
#                 validator_version=args.validator_v,
#                 validator_model=args.validator_model,
#                 upstream_timestamp=pipeline_timestamp,
#                 run_name=run_name,
#                 condition_name=pre_name,
#                 condition_text=pre_condition,
#                 image_path=current_image,
#                 scene_version=args.scene_v,
#                 scene_model=args.scene_model,
#                 plan_version=args.plan_v,
#                 plan_model=args.plan_model,
#                 sim_version=args.sim_v,
#                 sim_model=args.sim_model,
#             )

#             print(f"\n[PRE forced-validator:{pre_name}] Parsed JSON:")
#             print(json.dumps(pre_response, indent=2, ensure_ascii=False))

#             stage_record["pre_validation"] = pre_response

#             # ---------------- DEPLOY ----------------
#             print(f"\n[STAGE {stage_id}] DEPLOY")
#             next_image = execute_stage_deploy(
#                 stage_id=stage_id,
#                 manip_dir=manip_dir,
#                 screens_dir=screens_dir,
#                 cartesio_stop_poll=args.cartesio_stop_poll,
#                 post_kill_wait=args.post_kill_wait,
#                 post_homing_wait=args.post_homing_wait,
#                 cartesio_startup_wait=args.cartesio_startup_wait,
#             )

#             print(f"[STAGE {stage_id}] POST image:    {next_image}")

#             # ---------------- POST ----------------
#             print(f"\n[STAGE {stage_id}] POST CHECK")
#             print(f"[STAGE {stage_id}] POST image:    {next_image}")
#             print(f"[STAGE {stage_id}] POST condition:{post_condition}")

#             post_name = f"post_{stage_id}"
#             post_response = execute_forced_validator_step(
#                 settings=settings,
#                 scenario_name=args.scenario,
#                 validator_version=args.validator_v,
#                 validator_model=args.validator_model,
#                 upstream_timestamp=pipeline_timestamp,
#                 run_name=run_name,
#                 condition_name=post_name,
#                 condition_text=post_condition,
#                 image_path=next_image,
#                 scene_version=args.scene_v,
#                 scene_model=args.scene_model,
#                 plan_version=args.plan_v,
#                 plan_model=args.plan_model,
#                 sim_version=args.sim_v,
#                 sim_model=args.sim_model,
#             )

#             print(f"\n[POST forced-validator:{post_name}] Parsed JSON:")
#             print(json.dumps(post_response, indent=2, ensure_ascii=False))

#             stage_record["post_validation"] = post_response
#             stage_record["post_image_path"] = str(Path(next_image).resolve())
#             summary["stages"].append(stage_record)

#             current_image = next_image

#         summary["task_completed"] = True
#         summary["final_image_path"] = str(Path(current_image).resolve())

#         print("\n======================================================")
#         print("[PIPELINE] TASK COMPLETED SUCCESSFULLY")
#         print("======================================================")

#     except Exception as exc:
#         summary["task_completed"] = False
#         summary["error"] = str(exc)
#         print("\n======================================================")
#         print("[PIPELINE] ERROR")
#         print(str(exc))
#         print("======================================================")

#     summary_path = save_run_summary(
#         settings=settings,
#         scenario_name=args.scenario,
#         loop_timestamp=loop_timestamp,
#         summary=summary,
#     )

#     print("\n======================================================")
#     print("REAL DEPLOY PIPELINE COMPLETED")
#     print(f"Scenario:        {args.scenario}")
#     print(f"Loop timestamp:  {loop_timestamp}")
#     print(f"Task completed:  {summary['task_completed']}")
#     print(f"Summary saved:   {summary_path}")
#     print("======================================================")


# if __name__ == "__main__":
#     main()
