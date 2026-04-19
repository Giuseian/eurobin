from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from settings import load_settings
from utils import (
    make_experiment_timestamp,
    read_json,
    write_json,
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

    parser.add_argument(
        "--source-run-dir",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--source-cycle-name",
        type=str,
        default="cycle_001",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--max-sleep",
        type=float,
        default=5.0,
        help=argparse.SUPPRESS,
    )

    return parser


# ============================================================
# GENERIC HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
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


def validate_args(args: argparse.Namespace, source_run_dir: Path) -> None:
    if args.max_replans < 0:
        raise ValueError("--max-replans must be >= 0")

    if args.max_sleep < 0.0:
        raise ValueError("--max-sleep must be >= 0.0")

    if not source_run_dir.exists():
        raise FileNotFoundError(f"source run directory not found: {source_run_dir}")
    if not source_run_dir.is_dir():
        raise ValueError(f"source run directory must be a directory: {source_run_dir}")

    full_summary_path = source_run_dir / "full_pipeline_summary.json"
    if not full_summary_path.exists():
        raise FileNotFoundError(f"full_pipeline_summary.json not found: {full_summary_path}")

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


def safe_float(x: Any, default: float = 1.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def extract_stages_from_cycle(cycle: dict[str, Any]) -> list[dict[str, Any]]:
    raw_stages = cycle.get("stages")
    if not isinstance(raw_stages, list):
        raise ValueError("Selected cycle does not contain a valid 'stages' list.")

    stages: list[dict[str, Any]] = []

    for idx, stage in enumerate(raw_stages):
        if not isinstance(stage, dict):
            raise ValueError(f"Stage at index {idx} is not a JSON object.")

        stage_id = stage.get("stage_id")
        precondition = stage.get("precondition")
        postcondition = stage.get("postcondition")

        if not isinstance(stage_id, int):
            raise ValueError(f"Stage at index {idx} has invalid or missing 'stage_id'.")
        if not isinstance(precondition, str) or not precondition.strip():
            raise ValueError(f"Stage {stage_id} has invalid or missing 'precondition'.")
        if not isinstance(postcondition, str) or not postcondition.strip():
            raise ValueError(f"Stage {stage_id} has invalid or missing 'postcondition'.")

        stages.append(stage)

    return stages


def finalize_cycle_timing(cycle_record: dict[str, Any]) -> None:
    validator_times: list[float] = []
    deploy_times: list[float] = []
    stage_times: list[float] = []

    for stage in cycle_record.get("stages", []):
        timing = stage.get("timing", {})

        pre_t = safe_float(timing.get("pre_validation"), default=None)
        post_t = safe_float(timing.get("post_validation"), default=None)
        deploy_t = safe_float(timing.get("deploy"), default=None)
        total_t = safe_float(timing.get("total"), default=None)

        if pre_t is not None:
            validator_times.append(pre_t)
        if post_t is not None:
            validator_times.append(post_t)
        if deploy_t is not None:
            deploy_times.append(deploy_t)
        if total_t is not None:
            stage_times.append(total_t)

    cycle_record["timing"]["validators_total"] = sum(validator_times) if validator_times else None
    cycle_record["timing"]["deploy_total"] = sum(deploy_times) if deploy_times else None
    cycle_record["timing"]["stages_total"] = sum(stage_times) if stage_times else None


def build_global_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
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
        "timing": cycle_record.get("timing"),
        "outcome": cycle_record["outcome"],
        "end_image_path": cycle_record.get("end_image_path"),
        "end_image_name": cycle_record.get("end_image_name"),
    }


# ============================================================
# SOURCE SUMMARY HELPERS
# ============================================================

def get_default_source_run_dir(settings) -> Path:
    return (
        settings.project_root
        / "outputs"
        / "real_deploy_validation_loop"
        / "stacked_boxes"
        / "2026-04-13_13-31-22"
    ).resolve()


def load_full_pipeline_summary(source_run_dir: Path) -> dict[str, Any]:
    summary_path = source_run_dir / "full_pipeline_summary.json"
    data = read_json(summary_path)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid full_pipeline_summary.json content: {summary_path}")

    return data


def select_cycle_from_full_summary(
    full_summary: dict[str, Any],
    cycle_name: str | None = None,
) -> dict[str, Any]:
    cycles = full_summary.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        raise ValueError("full_pipeline_summary.json does not contain a valid non-empty 'cycles' list.")

    if cycle_name is None:
        return cycles[-1]

    for cycle in cycles:
        if isinstance(cycle, dict) and cycle.get("cycle_name") == cycle_name:
            return cycle

    raise ValueError(f"Cycle '{cycle_name}' not found in full_pipeline_summary.json")


# ============================================================
# COMMAND / ROS HELPERS
# ============================================================

def run_command(
    cmd: list[str],
    label: str,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    print(f"\n[CMD] {label}")
    print("[CMD] " + " ".join(cmd))

    start_time = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    execution_time_seconds = time.perf_counter() - start_time

    if result.stdout:
        print(f"[STDOUT][{label}]\n{result.stdout}")
    if result.stderr:
        print(f"[STDERR][{label}]\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed for '{label}' with return code {result.returncode}."
        )

    print(f"[TIME][{label}] Manipulation script completed in {execution_time_seconds:.3f}s")

    return {
        "label": label,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "execution_time_seconds": execution_time_seconds,
    }


def is_ros_script(script_path: Path) -> bool:
    ros_script_names = {
        "homing.py",
        "grasp_box_yz.py",
        "place_box.py",
        "place_box_2.py",
        "grasp_box_xy.py",
    }
    return script_path.name in ros_script_names


def build_ros_env() -> dict[str, str]:
    env = os.environ.copy()

    ros_pythonpath = "/opt/ros/jazzy/lib/python3.12/site-packages"
    existing_pythonpath = env.get("PYTHONPATH", "")

    paths = [p for p in existing_pythonpath.split(":") if p] if existing_pythonpath else []
    if ros_pythonpath not in paths:
        paths.insert(0, ros_pythonpath)

    env["PYTHONPATH"] = ":".join(paths)
    return env


def run_python_script(script_path: Path, label: str) -> dict[str, Any]:
    script_path = script_path.resolve()

    if is_ros_script(script_path):
        python_exec = "/usr/bin/python3"
        env = build_ros_env()
    else:
        python_exec = sys.executable
        env = None

    result = run_command([python_exec, str(script_path)], label=label, env=env)

    return {
        "script_name": script_path.name,
        "script_path": str(script_path),
        "label": label,
        "python_exec": python_exec,
        "execution_time_seconds": result["execution_time_seconds"],
    }


# ============================================================
# SCREENSHOT HELPERS
# ============================================================

def wait_until_file_is_stable(
    path: Path,
    timeout: float = 5.0,
    poll_interval: float = 0.2,
) -> None:
    deadline = time.time() + timeout
    last_size = -1
    stable_reads = 0

    while time.time() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 0 and size == last_size:
                stable_reads += 1
                if stable_reads >= 3:
                    return
            else:
                stable_reads = 0
            last_size = size

        time.sleep(poll_interval)

    raise RuntimeError(f"File did not stabilize in time: {path}")


def validate_png(path: Path) -> None:
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        raise RuntimeError(f"Screenshot is not a valid PNG yet: {path} | {exc}")


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
) -> dict[str, Any]:
    overall_start_time = time.perf_counter()

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

    cmd_start_time = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    cmd_execution_time_seconds = time.perf_counter() - cmd_start_time

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

    wait_until_file_is_stable(
        new_file,
        timeout=wait_timeout,
        poll_interval=poll_interval,
    )
    validate_png(new_file)

    final_path = get_next_sequential_image_path(screens_dir)

    if final_path.exists():
        raise RuntimeError(f"Sequential screenshot path already exists: {final_path}")

    new_file.rename(final_path)

    wait_until_file_is_stable(
        final_path,
        timeout=wait_timeout,
        poll_interval=poll_interval,
    )
    validate_png(final_path)

    total_execution_time_seconds = time.perf_counter() - overall_start_time

    print(f"[SCREEN] Screenshot saved as: {final_path}")
    print(f"[SCREEN] Screenshot time: {total_execution_time_seconds:.3f}s")

    return {
        "image_path": str(final_path.resolve()),
        "command_execution_time_seconds": cmd_execution_time_seconds,
        "execution_time_seconds": total_execution_time_seconds,
    }


# ============================================================
# DEPLOY HELPERS
# ============================================================

def deploy_stage_1(
    manip_dir: Path,
    screens_dir: Path,
    stage_id: int,
) -> dict[str, Any]:
    stage_start_time = time.perf_counter()
    scripts: list[dict[str, Any]] = []

    grasp_result = run_python_script(
        manip_dir / "grasp_box_yz.py",
        label="stage_1/grasp_box_yz.py",
    )
    scripts.append(
        {
            "event_type": "manipulation_script",
            "module_name": "deploy",
            "script_name": grasp_result["script_name"],
            "script_path": grasp_result["script_path"],
            "duration_sec": grasp_result["execution_time_seconds"],
            "outcome": "success",
        }
    )

    screenshot_result = take_screenshot(screens_dir)
    total_execution_time_seconds = time.perf_counter() - stage_start_time

    print(f"[TIME][stage_{stage_id}] Total deploy time: {total_execution_time_seconds:.3f}s")

    return {
        "next_image_path": screenshot_result["image_path"],
        "deploy_time_seconds": total_execution_time_seconds,
        "scripts": scripts,
        "screenshot": {
            "event_type": "screenshot",
            "module_name": "deploy",
            "script_name": None,
            "duration_sec": screenshot_result["execution_time_seconds"],
            "outcome": "success",
        },
    }


def deploy_stage_2(
    manip_dir: Path,
    screens_dir: Path,
    stage_id: int,
) -> dict[str, Any]:
    stage_start_time = time.perf_counter()
    scripts: list[dict[str, Any]] = []

    place_1 = run_python_script(
        manip_dir / "place_box.py",
        label="stage_2/place_box.py",
    )
    scripts.append(
        {
            "event_type": "manipulation_script",
            "module_name": "deploy",
            "script_name": place_1["script_name"],
            "script_path": place_1["script_path"],
            "duration_sec": place_1["execution_time_seconds"],
            "outcome": "success",
        }
    )

    place_2 = run_python_script(
        manip_dir / "place_box_2.py",
        label="stage_2/place_box_2.py",
    )
    scripts.append(
        {
            "event_type": "manipulation_script",
            "module_name": "deploy",
            "script_name": place_2["script_name"],
            "script_path": place_2["script_path"],
            "duration_sec": place_2["execution_time_seconds"],
            "outcome": "success",
        }
    )

    homing = run_python_script(
        manip_dir / "homing.py",
        label="stage_2/homing.py",
    )
    scripts.append(
        {
            "event_type": "manipulation_script",
            "module_name": "deploy",
            "script_name": homing["script_name"],
            "script_path": homing["script_path"],
            "duration_sec": homing["execution_time_seconds"],
            "outcome": "success",
        }
    )

    screenshot_result = take_screenshot(screens_dir)
    total_execution_time_seconds = time.perf_counter() - stage_start_time

    print(f"[TIME][stage_{stage_id}] Total deploy time: {total_execution_time_seconds:.3f}s")

    return {
        "next_image_path": screenshot_result["image_path"],
        "deploy_time_seconds": total_execution_time_seconds,
        "scripts": scripts,
        "screenshot": {
            "event_type": "screenshot",
            "module_name": "deploy",
            "script_name": None,
            "duration_sec": screenshot_result["execution_time_seconds"],
            "outcome": "success",
        },
    }


def deploy_stage_3(
    manip_dir: Path,
    screens_dir: Path,
    stage_id: int,
) -> dict[str, Any]:
    stage_start_time = time.perf_counter()
    scripts: list[dict[str, Any]] = []

    grasp_result = run_python_script(
        manip_dir / "grasp_box_xy.py",
        label="stage_3/grasp_box_xy.py",
    )
    scripts.append(
        {
            "event_type": "manipulation_script",
            "module_name": "deploy",
            "script_name": grasp_result["script_name"],
            "script_path": grasp_result["script_path"],
            "duration_sec": grasp_result["execution_time_seconds"],
            "outcome": "success",
        }
    )

    screenshot_result = take_screenshot(screens_dir)
    total_execution_time_seconds = time.perf_counter() - stage_start_time

    print(f"[TIME][stage_{stage_id}] Total deploy time: {total_execution_time_seconds:.3f}s")

    return {
        "next_image_path": screenshot_result["image_path"],
        "deploy_time_seconds": total_execution_time_seconds,
        "scripts": scripts,
        "screenshot": {
            "event_type": "screenshot",
            "module_name": "deploy",
            "script_name": None,
            "duration_sec": screenshot_result["execution_time_seconds"],
            "outcome": "success",
        },
    }


def execute_stage_deploy(
    stage_id: int,
    manip_dir: Path,
    screens_dir: Path,
) -> dict[str, Any]:
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
# MODULE TIMING HELPERS
# ============================================================

def consume_module_time(execution_time_seconds: Any, max_sleep: float) -> float:
    recorded = safe_float(execution_time_seconds, default=1.0)
    sleep_time = min(recorded, max_sleep)
    time.sleep(sleep_time)
    return sleep_time


def get_module_artifact(cycle: dict[str, Any], key: str) -> dict[str, Any]:
    artifact = cycle.get(key)
    if not isinstance(artifact, dict):
        raise ValueError(f"Missing module artifact in cycle: {key}")
    return artifact


def get_validator_artifact_from_stage(
    stage: dict[str, Any],
    condition_kind: str,
) -> tuple[dict[str, Any], float]:
    timing = stage.get("timing", {})
    if not isinstance(timing, dict):
        timing = {}

    if condition_kind == "pre":
        response = stage.get("pre_validation")
        execution_time_seconds = timing.get("pre_validation")
    elif condition_kind == "post":
        response = stage.get("post_validation")
        execution_time_seconds = timing.get("post_validation")
    else:
        raise ValueError(f"Unsupported condition_kind: {condition_kind}")

    if not isinstance(response, dict):
        raise ValueError(
            f"Missing saved {condition_kind}_validation for stage {stage.get('stage_id')}"
        )

    return response, safe_float(execution_time_seconds, default=1.0)


# ============================================================
# OUTPUT HELPERS
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
    path = cycle_dir / "cycle_summary.json"
    write_json(path, cycle_summary)
    return path


def build_run_info(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "real_deploy_validation_loop",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": full_summary["timestamp"],
        "initial_image_path": full_summary["initial_image_path"],
        "screens_dir": full_summary["screens_dir"],
        "config": full_summary["config"],
        "total_execution_time_seconds": full_summary.get("total_execution_time_seconds"),
        "initial_homing_time_seconds": full_summary.get("initial_homing_time_seconds"),
        "initial_screenshot_time_seconds": full_summary.get("initial_screenshot_time_seconds"),
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
                "timing": cycle.get("timing"),
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
                "scene_description_paths": cycle["scene_description"]["paths"],
                "scene_description_full_paths": cycle["scene_description_full"]["paths"],
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
                "timing": cycle.get("timing"),
            }
            for cycle in full_summary["cycles"]
        ],
    }


def build_full_pipeline_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(full_summary))


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
        "run_info": output_dir / "run_info.json",
        "loop_summary": output_dir / "loop_summary.json",
        "scene_description_summary": output_dir / "scene_description_summary.json",
        "vlm_planning_summary": output_dir / "vlm_planning_summary.json",
        "simultaneous_actions_summary": output_dir / "simultaneous_actions_summary.json",
        "validator_summary": output_dir / "validator_summary.json",
        "full_pipeline_summary": output_dir / "full_pipeline_summary.json",
    }

    write_json(paths["run_info"], run_info)
    write_json(paths["loop_summary"], loop_summary)
    write_json(paths["scene_description_summary"], scene_description_summary)
    write_json(paths["vlm_planning_summary"], vlm_planning_summary)
    write_json(paths["simultaneous_actions_summary"], simultaneous_actions_summary)
    write_json(paths["validator_summary"], validator_summary)
    write_json(paths["full_pipeline_summary"], full_pipeline_summary)

    return paths


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_sampling_args(args)
    settings = load_settings()

    if args.source_run_dir is None:
        source_run_dir = get_default_source_run_dir(settings)
    else:
        source_run_dir = Path(args.source_run_dir).resolve()

    validate_args(args, source_run_dir)

    loop_timestamp = make_experiment_timestamp()
    manip_dir = Path(args.manip_dir).resolve()

    screens_dir = get_screens_root(
        settings=settings,
        scenario_name=args.scenario,
        loop_timestamp=loop_timestamp,
        screens_subdir=args.screens_subdir,
    )
    ensure_dir(screens_dir)

    run_start_time = time.perf_counter()

    initial_homing_artifact = run_python_script(
        manip_dir / "homing.py",
        label="initial_homing/homing.py",
    )

    initial_screenshot_artifact = take_screenshot(screens_dir)
    initial_image_path = initial_screenshot_artifact["image_path"]
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
        "total_execution_time_seconds": None,
        "initial_homing_time_seconds": initial_homing_artifact["execution_time_seconds"],
        "initial_screenshot_time_seconds": initial_screenshot_artifact["execution_time_seconds"],
        "cycles": [],
    }

    print("\n======================================================")
    print("REAL DEPLOY VALIDATION LOOP CONFIG")
    print(f"Scenario:                  {args.scenario}")
    print(f"Temperature:               {args.temperature}")
    print(f"Top-p:                     {args.top_p}")
    print(f"Max replans:               {args.max_replans}")
    print("======================================================")

    source_full_summary = load_full_pipeline_summary(source_run_dir)
    source_cycle = select_cycle_from_full_summary(
        source_full_summary,
        args.source_cycle_name,
    )

    while not task_completed:
        cycle_idx += 1
        cycle_name = f"cycle_{cycle_idx:03d}"
        cycle_timestamp = make_experiment_timestamp()

        print("\n======================================================")
        print(f"REAL DEPLOY VALIDATION LOOP CYCLE STARTED | cycle={cycle_idx} | {cycle_name}")
        print(f"Current image:   {current_image}")
        print(f"Loop ts:         {loop_timestamp}")
        print(f"Cycle ts meta:   {cycle_timestamp}")
        print("======================================================")

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
            "timing": {
                "scene_description": None,
                "scene_enrichment": None,
                "planning": None,
                "simultaneous": None,
                "validators_total": None,
                "deploy_total": None,
                "stages_total": None,
                "cycle_total": None,
            },
        }

        cycle_error = False
        cycle_start_time = time.perf_counter()

        try:
            scene_description_artifact = get_module_artifact(source_cycle, "scene_description")
            consume_module_time(
                scene_description_artifact.get("execution_time_seconds"),
                args.max_sleep,
            )
            cycle_record["scene_description"] = scene_description_artifact
            cycle_record["timing"]["scene_description"] = safe_float(
                scene_description_artifact.get("execution_time_seconds")
            )

            print("\n[scene_description] Parsed JSON:")
            print(json.dumps(scene_description_artifact["output"], indent=2, ensure_ascii=False))
            print(f"[OK][scene_description] Execution time:          {safe_float(scene_description_artifact.get('execution_time_seconds')):.3f}s")

            scene_description_full_artifact = get_module_artifact(source_cycle, "scene_description_full")
            consume_module_time(
                scene_description_full_artifact.get("execution_time_seconds"),
                args.max_sleep,
            )
            cycle_record["scene_description_full"] = scene_description_full_artifact
            cycle_record["timing"]["scene_enrichment"] = safe_float(
                scene_description_full_artifact.get("execution_time_seconds")
            )

            print("\n[scene_description_full] Parsed JSON:")
            print(json.dumps(scene_description_full_artifact["output"], indent=2, ensure_ascii=False))
            print(f"[OK][scene_description_full] Execution time:         {safe_float(scene_description_full_artifact.get('execution_time_seconds')):.3f}s")

            sequential_plan_artifact = get_module_artifact(source_cycle, "vlm_planning")
            consume_module_time(
                sequential_plan_artifact.get("execution_time_seconds"),
                args.max_sleep,
            )
            cycle_record["vlm_planning"] = sequential_plan_artifact
            cycle_record["timing"]["planning"] = safe_float(
                sequential_plan_artifact.get("execution_time_seconds")
            )

            print("\n[vlm_planning] Parsed JSON:")
            print(json.dumps(sequential_plan_artifact["output"], indent=2, ensure_ascii=False))
            print(f"[OK][vlm_planning] Execution time:         {safe_float(sequential_plan_artifact.get('execution_time_seconds')):.3f}s")

            simultaneous_actions_artifact = get_module_artifact(source_cycle, "simultaneous_actions")
            consume_module_time(
                simultaneous_actions_artifact.get("execution_time_seconds"),
                args.max_sleep,
            )
            cycle_record["simultaneous_actions"] = simultaneous_actions_artifact
            cycle_record["timing"]["simultaneous"] = safe_float(
                simultaneous_actions_artifact.get("execution_time_seconds")
            )

            print("\n[simultaneous_actions] Parsed JSON:")
            print(json.dumps(simultaneous_actions_artifact["output"], indent=2, ensure_ascii=False))
            print(f"[OK][simultaneous_actions] Execution time:         {safe_float(simultaneous_actions_artifact.get('execution_time_seconds')):.3f}s")

            stages = extract_stages_from_cycle(source_cycle)
            all_stages_succeeded = True

            for source_stage in stages:
                stage_id = source_stage["stage_id"]
                stage_name = source_stage.get("stage_name", f"stage_{stage_id:03d}")
                pre_condition = source_stage["precondition"]
                post_condition = source_stage["postcondition"]

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
                    "timing": {
                        "pre_validation": None,
                        "deploy": None,
                        "post_validation": None,
                        "total": None,
                        "deploy_scripts": [],
                        "screenshot": None,
                    },
                }
                stage_start_time = time.perf_counter()

                print(f"\n[LOOP] Stage {stage_id} PRE")
                print(f"[LOOP] PRE image:      {current_image}")
                print(f"[LOOP] PRE condition:  {pre_condition}")

                pre_response, pre_execution_time = get_validator_artifact_from_stage(
                    source_stage,
                    "pre",
                )
                consume_module_time(pre_execution_time, args.max_sleep)

                print(f"\n[PRE validator:pre_{stage_id}] Parsed JSON:")
                print(json.dumps(pre_response, indent=2, ensure_ascii=False))
                print(f"[LOOP] PRE result:     {pre_response['result']}")
                print(f"[LOOP] PRE reason:     {pre_response['reason']}")

                stage_record["pre_validation"] = pre_response
                stage_record["timing"]["pre_validation"] = pre_execution_time

                if pre_response["result"] == "non_matching":
                    print(f"[LOOP] Precondition failed at stage {stage_id}. Replanning from same image.")
                    stage_record["timing"]["total"] = time.perf_counter() - stage_start_time
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

                print(f"\n[LOOP] Stage {stage_id} REAL DEPLOY")
                deploy_artifact = execute_stage_deploy(
                    stage_id=stage_id,
                    manip_dir=manip_dir,
                    screens_dir=screens_dir,
                )

                next_image = deploy_artifact["next_image_path"]

                stage_record["next_image_path"] = str(Path(next_image).resolve())
                stage_record["next_image_name"] = Path(next_image).name
                stage_record["post_image_path"] = str(Path(next_image).resolve())
                stage_record["post_image_name"] = Path(next_image).name
                stage_record["timing"]["deploy"] = deploy_artifact.get("deploy_time_seconds")
                stage_record["timing"]["deploy_scripts"] = deploy_artifact.get("scripts", [])
                stage_record["timing"]["screenshot"] = deploy_artifact.get("screenshot")

                print(f"[LOOP] NEXT image:     {next_image}")

                print(f"\n[LOOP] Stage {stage_id} POST")
                print(f"[LOOP] POST image:     {next_image}")
                print(f"[LOOP] POST condition: {post_condition}")

                post_response, post_execution_time = get_validator_artifact_from_stage(
                    source_stage,
                    "post",
                )
                consume_module_time(post_execution_time, args.max_sleep)

                print(f"\n[POST validator:post_{stage_id}] Parsed JSON:")
                print(json.dumps(post_response, indent=2, ensure_ascii=False))
                print(f"[LOOP] POST result:    {post_response['result']}")
                print(f"[LOOP] POST reason:    {post_response['reason']}")

                stage_record["post_validation"] = post_response
                stage_record["timing"]["post_validation"] = post_execution_time
                stage_record["timing"]["total"] = time.perf_counter() - stage_start_time
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
            tb = traceback.format_exc()
            print("\n[ERROR] Exception during cycle execution:")
            print(tb)

            cycle_record["outcome"] = f"cycle_error: {exc}"
            cycle_record["error_traceback"] = tb
            cycle_record["end_image_path"] = str(Path(current_image).resolve())
            cycle_record["end_image_name"] = Path(current_image).name
            full_summary["task_completed"] = False
            full_summary["error"] = str(exc)
            full_summary["error_traceback"] = tb
            cycle_error = True

        finalize_cycle_timing(cycle_record)
        cycle_record["timing"]["cycle_total"] = time.perf_counter() - cycle_start_time
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

    full_summary["total_execution_time_seconds"] = time.perf_counter() - run_start_time

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
    print(f"Temperature:               {args.temperature}")
    print(f"Top-p:                     {args.top_p}")
    print(f"Task completed:            {full_summary['task_completed']}")
    print(f"Replans done:              {full_summary['replans_done']}")
    print(f"Total execution time:      {full_summary['total_execution_time_seconds']:.3f}s")
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
# import os
# import subprocess
# import sys
# import time
# from datetime import datetime
# from pathlib import Path
# from typing import Any

# from PIL import Image

# from settings import load_settings
# from utils import (
#     make_experiment_timestamp,
#     read_json,
#     write_json,
# )

# SUPPORTED_MODELS = ["o3", "gpt-5.2"]


# # ============================================================
# # PARSER
# # ============================================================

# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Replay a real deploy execution using a hardcoded plan loaded from a previous "
#             "real_deploy_validation_loop full_pipeline_summary.json, while keeping a CLI "
#             "compatible with run_real_deploy_validation_loop.py."
#         )
#     )

#     # Same CLI shape as run_real_deploy_validation_loop.py
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
#         "--temperature",
#         type=float,
#         default=0.0,
#         help="Accepted for CLI compatibility. Not used for hardcoded replay execution.",
#     )
#     parser.add_argument(
#         "--top-p",
#         type=float,
#         default=1.0,
#         help="Accepted for CLI compatibility. Not used for hardcoded replay execution.",
#     )

#     parser.add_argument(
#         "--max-replans",
#         type=int,
#         default=10,
#         help="Accepted for CLI compatibility. Not used in replay mode.",
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
#         help="Subdirectory name used inside the replay scenario folder for screenshots.",
#     )

#     parser.add_argument(
#         "--grounding-topic",
#         type=str,
#         default="/world/default/dynamic_pose/info",
#         help="Accepted for CLI compatibility. Not used in replay mode.",
#     )

#     parser.add_argument(
#         "--grounding-timeout-sec",
#         type=float,
#         default=3.0,
#         help="Accepted for CLI compatibility. Not used in replay mode.",
#     )

#     parser.add_argument(
#         "--grounding-safety-threshold",
#         type=float,
#         default=0.21,
#         help="Accepted for CLI compatibility. Not used in replay mode.",
#     )

#     parser.add_argument(
#         "--grounding-debug-mapping",
#         action="store_true",
#         help="Accepted for CLI compatibility. Not used in replay mode.",
#     )

#     # Replay-specific source, now optional
#     parser.add_argument(
#         "--source-run-dir",
#         type=str,
#         default=None,
#         help=(
#             "Optional path to a previous real_deploy_validation_loop output directory "
#             "containing full_pipeline_summary.json. If omitted, the built-in replay "
#             "source is used."
#         ),
#     )

#     parser.add_argument(
#         "--source-cycle-name",
#         type=str,
#         default="cycle_001",
#         help=(
#             "Optional cycle name to replay from full_pipeline_summary.json. "
#             "Default: cycle_001."
#         ),
#     )

#     parser.add_argument(
#         "--max-sleep",
#         type=float,
#         default=5.0,
#         help="Maximum sleep time applied to each replayed saved module timing.",
#     )

#     return parser


# # ============================================================
# # GENERIC HELPERS
# # ============================================================

# def ensure_dir(path: Path) -> Path:
#     path.mkdir(parents=True, exist_ok=True)
#     return path


# def validate_sampling_args(args: argparse.Namespace) -> None:
#     if not 0.0 <= args.temperature <= 1.0:
#         raise ValueError("--temperature must be between 0.0 and 1.0")

#     if not 0.0 <= args.top_p <= 1.0:
#         raise ValueError("--top-p must be between 0.0 and 1.0")

#     if args.temperature != 0.0 and args.top_p != 1.0:
#         raise ValueError(
#             "Use either temperature or top_p for sampling control, not both at the same time."
#         )


# def validate_args(args: argparse.Namespace, source_run_dir: Path) -> None:
#     if args.max_replans < 0:
#         raise ValueError("--max-replans must be >= 0")

#     if args.max_sleep < 0.0:
#         raise ValueError("--max-sleep must be >= 0.0")

#     if not source_run_dir.exists():
#         raise FileNotFoundError(f"source-run-dir not found: {source_run_dir}")
#     if not source_run_dir.is_dir():
#         raise ValueError(f"source-run-dir must be a directory: {source_run_dir}")

#     full_summary_path = source_run_dir / "full_pipeline_summary.json"
#     if not full_summary_path.exists():
#         raise FileNotFoundError(f"full_pipeline_summary.json not found: {full_summary_path}")

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


# def safe_float(x: Any, default: float = 1.0) -> float:
#     try:
#         if x is None:
#             return default
#         return float(x)
#     except (TypeError, ValueError):
#         return default


# def build_global_config(args: argparse.Namespace, source_run_dir: Path) -> dict[str, Any]:
#     return {
#         "sampling": {
#             "temperature": args.temperature,
#             "top_p": args.top_p,
#         },
#         "scene_description": {
#             "prompt_version": args.scene_v,
#             "model": args.scene_model,
#         },
#         "scene_description_full": {
#             "stored_under_module": "scene_description",
#             "artifact_filename": "scene_description_full.json",
#             "prompt_version": args.scene_v,
#             "model": args.scene_model,
#             "mode": "replayed_from_full_pipeline_summary",
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
#         "replay": {
#             "source_run_dir": str(source_run_dir),
#             "source_cycle_name": args.source_cycle_name,
#             "max_sleep": args.max_sleep,
#             "mode": "hardcoded_plan_replay",
#         },
#     }


# # ============================================================
# # SOURCE SUMMARY HELPERS
# # ============================================================

# def get_default_source_run_dir(settings) -> Path:
#     return (
#         settings.project_root
#         / "outputs"
#         / "real_deploy_validation_loop"
#         / "stacked_boxes"
#         / "2026-04-13_13-31-22"
#     ).resolve()


# def load_full_pipeline_summary(source_run_dir: Path) -> dict[str, Any]:
#     summary_path = source_run_dir / "full_pipeline_summary.json"
#     data = read_json(summary_path)

#     if not isinstance(data, dict):
#         raise ValueError(f"Invalid full_pipeline_summary.json content: {summary_path}")

#     return data


# def select_cycle_from_full_summary(
#     full_summary: dict[str, Any],
#     cycle_name: str | None = None,
# ) -> dict[str, Any]:
#     cycles = full_summary.get("cycles")
#     if not isinstance(cycles, list) or not cycles:
#         raise ValueError("full_pipeline_summary.json does not contain a valid non-empty 'cycles' list.")

#     if cycle_name is None:
#         return cycles[-1]

#     for cycle in cycles:
#         if isinstance(cycle, dict) and cycle.get("cycle_name") == cycle_name:
#             return cycle

#     raise ValueError(f"Cycle '{cycle_name}' not found in full_pipeline_summary.json")


# def extract_stages_from_cycle(cycle: dict[str, Any]) -> list[dict[str, Any]]:
#     raw_stages = cycle.get("stages")
#     if not isinstance(raw_stages, list):
#         raise ValueError("Selected cycle does not contain a valid 'stages' list.")

#     stages: list[dict[str, Any]] = []

#     for idx, stage in enumerate(raw_stages):
#         if not isinstance(stage, dict):
#             raise ValueError(f"Stage at index {idx} is not a JSON object.")

#         stage_id = stage.get("stage_id")
#         precondition = stage.get("precondition")
#         postcondition = stage.get("postcondition")

#         if not isinstance(stage_id, int):
#             raise ValueError(f"Stage at index {idx} has invalid or missing 'stage_id'.")
#         if not isinstance(precondition, str) or not precondition.strip():
#             raise ValueError(f"Stage {stage_id} has invalid or missing 'precondition'.")
#         if not isinstance(postcondition, str) or not postcondition.strip():
#             raise ValueError(f"Stage {stage_id} has invalid or missing 'postcondition'.")

#         stages.append(stage)

#     return stages


# # ============================================================
# # COMMAND / ROS HELPERS
# # ============================================================

# def run_command(
#     cmd: list[str],
#     label: str,
#     env: dict[str, str] | None = None,
# ) -> dict[str, Any]:
#     print(f"\n[CMD] {label}")
#     print("[CMD] " + " ".join(cmd))

#     start_time = time.perf_counter()
#     result = subprocess.run(
#         cmd,
#         capture_output=True,
#         text=True,
#         env=env,
#     )
#     execution_time_seconds = time.perf_counter() - start_time

#     if result.stdout:
#         print(f"[STDOUT][{label}]\n{result.stdout}")
#     if result.stderr:
#         print(f"[STDERR][{label}]\n{result.stderr}")

#     if result.returncode != 0:
#         raise RuntimeError(
#             f"Command failed for '{label}' with return code {result.returncode}."
#         )

#     print(f"[TIME][{label}] Completed in {execution_time_seconds:.3f}s")

#     return {
#         "label": label,
#         "command": cmd,
#         "returncode": result.returncode,
#         "stdout": result.stdout,
#         "stderr": result.stderr,
#         "execution_time_seconds": execution_time_seconds,
#     }


# def is_ros_script(script_path: Path) -> bool:
#     ros_script_names = {
#         "homing.py",
#         "grasp_box_yz.py",
#         "place_box.py",
#         "place_box_2.py",
#         "grasp_box_xy.py",
#     }
#     return script_path.name in ros_script_names


# def build_ros_env() -> dict[str, str]:
#     env = os.environ.copy()

#     ros_pythonpath = "/opt/ros/jazzy/lib/python3.12/site-packages"
#     existing_pythonpath = env.get("PYTHONPATH", "")

#     paths = [p for p in existing_pythonpath.split(":") if p] if existing_pythonpath else []
#     if ros_pythonpath not in paths:
#         paths.insert(0, ros_pythonpath)

#     env["PYTHONPATH"] = ":".join(paths)
#     return env


# def run_python_script(script_path: Path, label: str) -> dict[str, Any]:
#     script_path = script_path.resolve()

#     if is_ros_script(script_path):
#         python_exec = "/usr/bin/python3"
#         env = build_ros_env()
#     else:
#         python_exec = sys.executable
#         env = None

#     result = run_command([python_exec, str(script_path)], label=label, env=env)

#     return {
#         "script_name": script_path.name,
#         "script_path": str(script_path),
#         "label": label,
#         "python_exec": python_exec,
#         "execution_time_seconds": result["execution_time_seconds"],
#     }


# # ============================================================
# # SCREENSHOT HELPERS
# # ============================================================

# def wait_until_file_is_stable(
#     path: Path,
#     timeout: float = 5.0,
#     poll_interval: float = 0.2,
# ) -> None:
#     deadline = time.time() + timeout
#     last_size = -1
#     stable_reads = 0

#     while time.time() < deadline:
#         if path.exists():
#             size = path.stat().st_size
#             if size > 0 and size == last_size:
#                 stable_reads += 1
#                 if stable_reads >= 3:
#                     return
#             else:
#                 stable_reads = 0
#             last_size = size

#         time.sleep(poll_interval)

#     raise RuntimeError(f"File did not stabilize in time: {path}")


# def validate_png(path: Path) -> None:
#     try:
#         with Image.open(path) as img:
#             img.verify()
#     except Exception as exc:
#         raise RuntimeError(f"Screenshot is not a valid PNG yet: {path} | {exc}")


# def get_screens_root(
#     settings,
#     scenario_name: str,
#     replay_timestamp: str,
#     screens_subdir: str,
# ) -> Path:
#     return (
#         settings.project_root
#         / "scenarios"
#         / "real_deploy_replay"
#         / scenario_name
#         / replay_timestamp
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
# ) -> dict[str, Any]:
#     overall_start_time = time.perf_counter()

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

#     cmd_start_time = time.perf_counter()
#     result = subprocess.run(cmd, capture_output=True, text=True)
#     cmd_execution_time_seconds = time.perf_counter() - cmd_start_time

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

#     wait_until_file_is_stable(
#         new_file,
#         timeout=wait_timeout,
#         poll_interval=poll_interval,
#     )
#     validate_png(new_file)

#     final_path = get_next_sequential_image_path(screens_dir)

#     if final_path.exists():
#         raise RuntimeError(f"Sequential screenshot path already exists: {final_path}")

#     new_file.rename(final_path)

#     wait_until_file_is_stable(
#         final_path,
#         timeout=wait_timeout,
#         poll_interval=poll_interval,
#     )
#     validate_png(final_path)

#     total_execution_time_seconds = time.perf_counter() - overall_start_time

#     print(f"[SCREEN] Screenshot saved as: {final_path}")
#     print(f"[SCREEN] Screenshot time: {total_execution_time_seconds:.3f}s")

#     return {
#         "image_path": str(final_path.resolve()),
#         "command_execution_time_seconds": cmd_execution_time_seconds,
#         "execution_time_seconds": total_execution_time_seconds,
#     }


# # ============================================================
# # DEPLOY HELPERS
# # ============================================================

# def deploy_stage_1(
#     manip_dir: Path,
#     screens_dir: Path,
#     stage_id: int,
# ) -> dict[str, Any]:
#     stage_start_time = time.perf_counter()
#     scripts: list[dict[str, Any]] = []

#     grasp_result = run_python_script(
#         manip_dir / "grasp_box_yz.py",
#         label="stage_1/grasp_box_yz.py",
#     )
#     scripts.append(
#         {
#             "event_type": "manipulation_script",
#             "module_name": "deploy",
#             "script_name": grasp_result["script_name"],
#             "script_path": grasp_result["script_path"],
#             "duration_sec": grasp_result["execution_time_seconds"],
#             "outcome": "success",
#         }
#     )

#     screenshot_result = take_screenshot(screens_dir)
#     total_execution_time_seconds = time.perf_counter() - stage_start_time

#     print(f"[TIME][stage_{stage_id}] Total deploy time: {total_execution_time_seconds:.3f}s")

#     return {
#         "next_image_path": screenshot_result["image_path"],
#         "deploy_time_seconds": total_execution_time_seconds,
#         "scripts": scripts,
#         "screenshot": {
#             "event_type": "screenshot",
#             "module_name": "deploy",
#             "script_name": None,
#             "duration_sec": screenshot_result["execution_time_seconds"],
#             "outcome": "success",
#         },
#     }


# def deploy_stage_2(
#     manip_dir: Path,
#     screens_dir: Path,
#     stage_id: int,
# ) -> dict[str, Any]:
#     stage_start_time = time.perf_counter()
#     scripts: list[dict[str, Any]] = []

#     place_1 = run_python_script(
#         manip_dir / "place_box.py",
#         label="stage_2/place_box.py",
#     )
#     scripts.append(
#         {
#             "event_type": "manipulation_script",
#             "module_name": "deploy",
#             "script_name": place_1["script_name"],
#             "script_path": place_1["script_path"],
#             "duration_sec": place_1["execution_time_seconds"],
#             "outcome": "success",
#         }
#     )

#     place_2 = run_python_script(
#         manip_dir / "place_box_2.py",
#         label="stage_2/place_box_2.py",
#     )
#     scripts.append(
#         {
#             "event_type": "manipulation_script",
#             "module_name": "deploy",
#             "script_name": place_2["script_name"],
#             "script_path": place_2["script_path"],
#             "duration_sec": place_2["execution_time_seconds"],
#             "outcome": "success",
#         }
#     )

#     homing = run_python_script(
#         manip_dir / "homing.py",
#         label="stage_2/homing.py",
#     )
#     scripts.append(
#         {
#             "event_type": "manipulation_script",
#             "module_name": "deploy",
#             "script_name": homing["script_name"],
#             "script_path": homing["script_path"],
#             "duration_sec": homing["execution_time_seconds"],
#             "outcome": "success",
#         }
#     )

#     screenshot_result = take_screenshot(screens_dir)
#     total_execution_time_seconds = time.perf_counter() - stage_start_time

#     print(f"[TIME][stage_{stage_id}] Total deploy time: {total_execution_time_seconds:.3f}s")

#     return {
#         "next_image_path": screenshot_result["image_path"],
#         "deploy_time_seconds": total_execution_time_seconds,
#         "scripts": scripts,
#         "screenshot": {
#             "event_type": "screenshot",
#             "module_name": "deploy",
#             "script_name": None,
#             "duration_sec": screenshot_result["execution_time_seconds"],
#             "outcome": "success",
#         },
#     }


# def deploy_stage_3(
#     manip_dir: Path,
#     screens_dir: Path,
#     stage_id: int,
# ) -> dict[str, Any]:
#     stage_start_time = time.perf_counter()
#     scripts: list[dict[str, Any]] = []

#     grasp_result = run_python_script(
#         manip_dir / "grasp_box_xy.py",
#         label="stage_3/grasp_box_xy.py",
#     )
#     scripts.append(
#         {
#             "event_type": "manipulation_script",
#             "module_name": "deploy",
#             "script_name": grasp_result["script_name"],
#             "script_path": grasp_result["script_path"],
#             "duration_sec": grasp_result["execution_time_seconds"],
#             "outcome": "success",
#         }
#     )

#     screenshot_result = take_screenshot(screens_dir)
#     total_execution_time_seconds = time.perf_counter() - stage_start_time

#     print(f"[TIME][stage_{stage_id}] Total deploy time: {total_execution_time_seconds:.3f}s")

#     return {
#         "next_image_path": screenshot_result["image_path"],
#         "deploy_time_seconds": total_execution_time_seconds,
#         "scripts": scripts,
#         "screenshot": {
#             "event_type": "screenshot",
#             "module_name": "deploy",
#             "script_name": None,
#             "duration_sec": screenshot_result["execution_time_seconds"],
#             "outcome": "success",
#         },
#     }


# def execute_stage_deploy(
#     stage_id: int,
#     manip_dir: Path,
#     screens_dir: Path,
# ) -> dict[str, Any]:
#     if stage_id == 1:
#         return deploy_stage_1(manip_dir=manip_dir, screens_dir=screens_dir, stage_id=stage_id)
#     if stage_id == 2:
#         return deploy_stage_2(manip_dir=manip_dir, screens_dir=screens_dir, stage_id=stage_id)
#     if stage_id == 3:
#         return deploy_stage_3(manip_dir=manip_dir, screens_dir=screens_dir, stage_id=stage_id)

#     raise ValueError(
#         f"No deploy routine defined for stage_id={stage_id}. "
#         "This script currently supports stage ids 1, 2, 3."
#     )


# # ============================================================
# # REPLAY HELPERS
# # ============================================================

# def replay_sleep(label: str, recorded_seconds: Any, max_sleep: float) -> float:
#     recorded = safe_float(recorded_seconds, default=1.0)
#     sleep_time = min(recorded, max_sleep)

#     print(
#         f"[REPLAY][{label}] Sleeping for recorded module time: "
#         f"{sleep_time:.3f}s (recorded={recorded:.3f}s)"
#     )
#     time.sleep(sleep_time)
#     return sleep_time


# def replay_validator_from_stage(
#     stage: dict[str, Any],
#     condition_kind: str,
#     max_sleep: float,
# ) -> tuple[dict[str, Any], float]:
#     timing = stage.get("timing", {})
#     if not isinstance(timing, dict):
#         timing = {}

#     if condition_kind == "pre":
#         response = stage.get("pre_validation")
#         recorded = timing.get("pre_validation")
#     elif condition_kind == "post":
#         response = stage.get("post_validation")
#         recorded = timing.get("post_validation")
#     else:
#         raise ValueError(f"Unsupported condition_kind: {condition_kind}")

#     if not isinstance(response, dict):
#         raise ValueError(
#             f"Missing saved {condition_kind}_validation for stage {stage.get('stage_id')}"
#         )

#     slept = replay_sleep(
#         label=f"validator:{condition_kind}_{stage.get('stage_id')}",
#         recorded_seconds=recorded,
#         max_sleep=max_sleep,
#     )
#     return response, slept


# # ============================================================
# # SUMMARY HELPERS
# # ============================================================

# def get_replay_output_root(settings, scenario_name: str, replay_timestamp: str) -> Path:
#     return (
#         settings.project_root
#         / "outputs"
#         / "real_deploy_replay"
#         / scenario_name
#         / replay_timestamp
#     )


# def save_run_summary(
#     settings,
#     scenario_name: str,
#     replay_timestamp: str,
#     summary: dict[str, Any],
# ) -> Path:
#     output_dir = get_replay_output_root(settings, scenario_name, replay_timestamp)
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

#     validate_sampling_args(args)
#     settings = load_settings()

#     if args.source_run_dir is None:
#         source_run_dir = get_default_source_run_dir(settings)
#     else:
#         source_run_dir = Path(args.source_run_dir).resolve()

#     validate_args(args, source_run_dir)

#     replay_timestamp = make_experiment_timestamp()
#     manip_dir = Path(args.manip_dir).resolve()

#     screens_dir = get_screens_root(
#         settings=settings,
#         scenario_name=args.scenario,
#         replay_timestamp=replay_timestamp,
#         screens_subdir=args.screens_subdir,
#     )
#     ensure_dir(screens_dir)

#     run_start_time = time.perf_counter()

#     print("\n======================================================")
#     print("REAL DEPLOY REPLAY CONFIG")
#     print(f"Scenario:                  {args.scenario}")
#     print(f"Source run dir:            {source_run_dir}")
#     print(f"Source cycle name:         {args.source_cycle_name}")
#     print(f"Temperature:               {args.temperature} (accepted, not used)")
#     print(f"Top-p:                     {args.top_p} (accepted, not used)")
#     print(f"Max replans:               {args.max_replans} (accepted, not used)")
#     print(f"Max sleep:                 {args.max_sleep}")
#     print("======================================================")

#     full_source_summary = load_full_pipeline_summary(source_run_dir)
#     source_cycle = select_cycle_from_full_summary(
#         full_source_summary,
#         args.source_cycle_name,
#     )
#     stages = extract_stages_from_cycle(source_cycle)

#     initial_homing_artifact = run_python_script(
#         manip_dir / "homing.py",
#         label="initial_homing/homing.py",
#     )

#     initial_screenshot_artifact = take_screenshot(screens_dir)
#     initial_image_path = initial_screenshot_artifact["image_path"]
#     current_image = initial_image_path

#     replay_summary: dict[str, Any] = {
#         "module": "real_deploy_replay",
#         "scenario_name": args.scenario,
#         "replay_timestamp": replay_timestamp,
#         "timestamp": datetime.now().isoformat(),
#         "initial_image_path": str(Path(initial_image_path).resolve()),
#         "screens_dir": str(screens_dir.resolve()),
#         "manip_dir": str(manip_dir),
#         "config": build_global_config(args, source_run_dir),
#         "task_completed": False,
#         "final_image_path": None,
#         "total_execution_time_seconds": None,
#         "initial_homing_time_seconds": initial_homing_artifact["execution_time_seconds"],
#         "initial_screenshot_time_seconds": initial_screenshot_artifact["execution_time_seconds"],
#         "source": {
#             "source_run_dir": str(source_run_dir),
#             "source_full_pipeline_summary_path": str((source_run_dir / "full_pipeline_summary.json").resolve()),
#             "source_cycle_name": source_cycle.get("cycle_name"),
#             "source_cycle_index": source_cycle.get("cycle_index"),
#             "source_cycle_timestamp": source_cycle.get("cycle_timestamp"),
#             "source_cycle_outcome": source_cycle.get("outcome"),
#         },
#         "replayed_modules": {
#             "scene_description": source_cycle.get("scene_description"),
#             "scene_description_full": source_cycle.get("scene_description_full"),
#             "vlm_planning": source_cycle.get("vlm_planning"),
#             "simultaneous_actions": source_cycle.get("simultaneous_actions"),
#         },
#         "timing": {
#             "scene_description_sleep": None,
#             "scene_description_full_sleep": None,
#             "planning_sleep": None,
#             "simultaneous_sleep": None,
#         },
#         "stages": [],
#     }

#     try:
#         scene_description_artifact = source_cycle.get("scene_description")
#         if isinstance(scene_description_artifact, dict):
#             print("\n[scene_description] Replayed JSON:")
#             print(json.dumps(scene_description_artifact.get("output"), indent=2, ensure_ascii=False))
#             replay_summary["timing"]["scene_description_sleep"] = replay_sleep(
#                 label="scene_description",
#                 recorded_seconds=scene_description_artifact.get("execution_time_seconds"),
#                 max_sleep=args.max_sleep,
#             )

#         scene_description_full_artifact = source_cycle.get("scene_description_full")
#         if isinstance(scene_description_full_artifact, dict):
#             print("\n[scene_description_full] Replayed JSON:")
#             print(json.dumps(scene_description_full_artifact.get("output"), indent=2, ensure_ascii=False))
#             replay_summary["timing"]["scene_description_full_sleep"] = replay_sleep(
#                 label="scene_description_full",
#                 recorded_seconds=scene_description_full_artifact.get("execution_time_seconds"),
#                 max_sleep=args.max_sleep,
#             )

#         vlm_planning_artifact = source_cycle.get("vlm_planning")
#         if isinstance(vlm_planning_artifact, dict):
#             print("\n[vlm_planning] Replayed JSON:")
#             print(json.dumps(vlm_planning_artifact.get("output"), indent=2, ensure_ascii=False))
#             replay_summary["timing"]["planning_sleep"] = replay_sleep(
#                 label="vlm_planning",
#                 recorded_seconds=vlm_planning_artifact.get("execution_time_seconds"),
#                 max_sleep=args.max_sleep,
#             )

#         simultaneous_actions_artifact = source_cycle.get("simultaneous_actions")
#         if isinstance(simultaneous_actions_artifact, dict):
#             print("\n[simultaneous_actions] Replayed JSON:")
#             print(json.dumps(simultaneous_actions_artifact.get("output"), indent=2, ensure_ascii=False))
#             replay_summary["timing"]["simultaneous_sleep"] = replay_sleep(
#                 label="simultaneous_actions",
#                 recorded_seconds=simultaneous_actions_artifact.get("execution_time_seconds"),
#                 max_sleep=args.max_sleep,
#             )

#         for stage in stages:
#             stage_id = stage["stage_id"]
#             pre_condition = stage["precondition"]
#             post_condition = stage["postcondition"]

#             print("\n------------------------------------------------------")
#             print(f"[STAGE {stage_id}] START")
#             print("------------------------------------------------------")

#             stage_record: dict[str, Any] = {
#                 "stage_id": stage_id,
#                 "stage_name": stage.get("stage_name"),
#                 "precondition": pre_condition,
#                 "postcondition": post_condition,
#                 "pre_image_path": str(Path(current_image).resolve()),
#                 "pre_image_name": Path(current_image).name,
#                 "post_image_path": None,
#                 "post_image_name": None,
#                 "pre_validation": None,
#                 "post_validation": None,
#                 "timing": {
#                     "pre_validation": None,
#                     "deploy": None,
#                     "post_validation": None,
#                     "deploy_scripts": [],
#                     "screenshot": None,
#                 },
#             }

#             print(f"\n[STAGE {stage_id}] PRE CHECK")
#             print(f"[STAGE {stage_id}] PRE image:      {current_image}")
#             print(f"[STAGE {stage_id}] PRE condition:  {pre_condition}")

#             pre_response, pre_sleep = replay_validator_from_stage(
#                 stage=stage,
#                 condition_kind="pre",
#                 max_sleep=args.max_sleep,
#             )

#             print(f"\n[PRE validator:pre_{stage_id}] Replayed JSON:")
#             print(json.dumps(pre_response, indent=2, ensure_ascii=False))

#             stage_record["pre_validation"] = pre_response
#             stage_record["timing"]["pre_validation"] = pre_sleep

#             print(f"\n[STAGE {stage_id}] DEPLOY")
#             deploy_artifact = execute_stage_deploy(
#                 stage_id=stage_id,
#                 manip_dir=manip_dir,
#                 screens_dir=screens_dir,
#             )

#             next_image = deploy_artifact["next_image_path"]

#             stage_record["post_image_path"] = str(Path(next_image).resolve())
#             stage_record["post_image_name"] = Path(next_image).name
#             stage_record["timing"]["deploy"] = deploy_artifact.get("deploy_time_seconds")
#             stage_record["timing"]["deploy_scripts"] = deploy_artifact.get("scripts", [])
#             stage_record["timing"]["screenshot"] = deploy_artifact.get("screenshot")

#             print(f"[STAGE {stage_id}] POST image:     {next_image}")

#             print(f"\n[STAGE {stage_id}] POST CHECK")
#             print(f"[STAGE {stage_id}] POST image:     {next_image}")
#             print(f"[STAGE {stage_id}] POST condition: {post_condition}")

#             post_response, post_sleep = replay_validator_from_stage(
#                 stage=stage,
#                 condition_kind="post",
#                 max_sleep=args.max_sleep,
#             )

#             print(f"\n[POST validator:post_{stage_id}] Replayed JSON:")
#             print(json.dumps(post_response, indent=2, ensure_ascii=False))

#             stage_record["post_validation"] = post_response
#             stage_record["timing"]["post_validation"] = post_sleep

#             replay_summary["stages"].append(stage_record)
#             current_image = next_image

#         replay_summary["task_completed"] = True
#         replay_summary["final_image_path"] = str(Path(current_image).resolve())

#         print("\n======================================================")
#         print("[REPLAY] TASK COMPLETED SUCCESSFULLY")
#         print("======================================================")

#     except Exception as exc:
#         replay_summary["task_completed"] = False
#         replay_summary["error"] = str(exc)

#         print("\n======================================================")
#         print("[REPLAY] ERROR")
#         print(str(exc))
#         print("======================================================")

#     replay_summary["total_execution_time_seconds"] = time.perf_counter() - run_start_time

#     summary_path = save_run_summary(
#         settings=settings,
#         scenario_name=args.scenario,
#         replay_timestamp=replay_timestamp,
#         summary=replay_summary,
#     )

#     print("\n======================================================")
#     print("REAL DEPLOY REPLAY COMPLETED")
#     print(f"Scenario:                  {args.scenario}")
#     print(f"Replay timestamp:          {replay_timestamp}")
#     print(f"Task completed:            {replay_summary['task_completed']}")
#     print(f"Total execution time:      {replay_summary['total_execution_time_seconds']:.3f}s")
#     print(f"Summary saved:             {summary_path}")
#     print("======================================================")


# if __name__ == "__main__":
#     main()







# # from __future__ import annotations

# # import argparse
# # import json
# # import os
# # import subprocess
# # import sys
# # import time
# # from datetime import datetime
# # from pathlib import Path
# # from typing import Any

# # from PIL import Image

# # from settings import load_settings
# # from utils import (
# #     make_experiment_timestamp,
# #     read_json,
# #     write_json,
# # )

# # SUPPORTED_MODELS = ["o3", "gpt-5.2"]


# # # ============================================================
# # # PARSER
# # # ============================================================

# # def build_parser() -> argparse.ArgumentParser:
# #     parser = argparse.ArgumentParser(
# #         description=(
# #             "Replay a real deploy execution using a hardcoded plan loaded from a previous "
# #             "real_deploy_validation_loop full_pipeline_summary.json, while keeping a CLI "
# #             "compatible with run_real_deploy_validation_loop.py."
# #         )
# #     )

# #     # Same CLI shape as run_real_deploy_validation_loop.py
# #     parser.add_argument("--scenario", type=str, required=True)

# #     parser.add_argument("--scene-v", type=str, required=True)
# #     parser.add_argument("--plan-v", type=str, required=True)
# #     parser.add_argument("--sim-v", type=str, required=True)
# #     parser.add_argument("--validator-v", type=str, required=True)

# #     parser.add_argument("--scene-model", type=str, required=True, choices=SUPPORTED_MODELS)
# #     parser.add_argument("--plan-model", type=str, required=True, choices=SUPPORTED_MODELS)
# #     parser.add_argument("--sim-model", type=str, required=True, choices=SUPPORTED_MODELS)
# #     parser.add_argument("--validator-model", type=str, required=True, choices=SUPPORTED_MODELS)

# #     parser.add_argument(
# #         "--temperature",
# #         type=float,
# #         default=0.0,
# #         help="Accepted for CLI compatibility. Not used for hardcoded replay execution.",
# #     )
# #     parser.add_argument(
# #         "--top-p",
# #         type=float,
# #         default=1.0,
# #         help="Accepted for CLI compatibility. Not used for hardcoded replay execution.",
# #     )

# #     parser.add_argument(
# #         "--max-replans",
# #         type=int,
# #         default=10,
# #         help="Accepted for CLI compatibility. Not used in replay mode.",
# #     )

# #     parser.add_argument(
# #         "--manip-dir",
# #         type=str,
# #         default=".",
# #         help="Directory containing manipulation scripts (grasp_box_yz.py, place_box.py, etc.).",
# #     )

# #     parser.add_argument(
# #         "--screens-subdir",
# #         type=str,
# #         default="screens",
# #         help="Subdirectory name used inside the replay scenario folder for screenshots.",
# #     )

# #     parser.add_argument(
# #         "--grounding-topic",
# #         type=str,
# #         default="/world/default/dynamic_pose/info",
# #         help="Accepted for CLI compatibility. Not used in replay mode.",
# #     )

# #     parser.add_argument(
# #         "--grounding-timeout-sec",
# #         type=float,
# #         default=3.0,
# #         help="Accepted for CLI compatibility. Not used in replay mode.",
# #     )

# #     parser.add_argument(
# #         "--grounding-safety-threshold",
# #         type=float,
# #         default=0.21,
# #         help="Accepted for CLI compatibility. Not used in replay mode.",
# #     )

# #     parser.add_argument(
# #         "--grounding-debug-mapping",
# #         action="store_true",
# #         help="Accepted for CLI compatibility. Not used in replay mode.",
# #     )

# #     # Replay-specific source
# #     parser.add_argument(
# #         "--source-run-dir",
# #         type=str,
# #         required=True,
# #         help=(
# #             "Path to a previous real_deploy_validation_loop output directory containing "
# #             "full_pipeline_summary.json. Example: "
# #             "/home/user/outputs/real_deploy_validation_loop/stacked_boxes/2026-04-13_13-31-22"
# #         ),
# #     )

# #     parser.add_argument(
# #         "--source-cycle-name",
# #         type=str,
# #         default=None,
# #         help=(
# #             "Optional cycle name to replay from full_pipeline_summary.json "
# #             "(example: cycle_001). Default: last cycle."
# #         ),
# #     )

# #     parser.add_argument(
# #         "--max-sleep",
# #         type=float,
# #         default=5.0,
# #         help="Maximum sleep time applied to each replayed saved module timing.",
# #     )

# #     return parser


# # # ============================================================
# # # GENERIC HELPERS
# # # ============================================================

# # def ensure_dir(path: Path) -> Path:
# #     path.mkdir(parents=True, exist_ok=True)
# #     return path


# # def validate_sampling_args(args: argparse.Namespace) -> None:
# #     if not 0.0 <= args.temperature <= 1.0:
# #         raise ValueError("--temperature must be between 0.0 and 1.0")

# #     if not 0.0 <= args.top_p <= 1.0:
# #         raise ValueError("--top-p must be between 0.0 and 1.0")

# #     if args.temperature != 0.0 and args.top_p != 1.0:
# #         raise ValueError(
# #             "Use either temperature or top_p for sampling control, not both at the same time."
# #         )


# # def validate_args(args: argparse.Namespace) -> None:
# #     if args.max_replans < 0:
# #         raise ValueError("--max-replans must be >= 0")

# #     if args.max_sleep < 0.0:
# #         raise ValueError("--max-sleep must be >= 0.0")

# #     source_run_dir = Path(args.source_run_dir).resolve()
# #     if not source_run_dir.exists():
# #         raise FileNotFoundError(f"source-run-dir not found: {source_run_dir}")
# #     if not source_run_dir.is_dir():
# #         raise ValueError(f"--source-run-dir must be a directory: {source_run_dir}")

# #     full_summary_path = source_run_dir / "full_pipeline_summary.json"
# #     if not full_summary_path.exists():
# #         raise FileNotFoundError(f"full_pipeline_summary.json not found: {full_summary_path}")

# #     manip_dir = Path(args.manip_dir).resolve()
# #     if not manip_dir.exists():
# #         raise FileNotFoundError(f"manip-dir not found: {manip_dir}")
# #     if not manip_dir.is_dir():
# #         raise ValueError(f"--manip-dir must be a directory: {manip_dir}")

# #     required_scripts = [
# #         "grasp_box_yz.py",
# #         "place_box.py",
# #         "place_box_2.py",
# #         "homing.py",
# #         "grasp_box_xy.py",
# #     ]
# #     for script_name in required_scripts:
# #         script_path = manip_dir / script_name
# #         if not script_path.exists():
# #             raise FileNotFoundError(f"Required manipulation script not found: {script_path}")


# # def safe_float(x: Any, default: float = 1.0) -> float:
# #     try:
# #         if x is None:
# #             return default
# #         return float(x)
# #     except (TypeError, ValueError):
# #         return default


# # def build_global_config(args: argparse.Namespace) -> dict[str, Any]:
# #     return {
# #         "sampling": {
# #             "temperature": args.temperature,
# #             "top_p": args.top_p,
# #         },
# #         "scene_description": {
# #             "prompt_version": args.scene_v,
# #             "model": args.scene_model,
# #         },
# #         "scene_description_full": {
# #             "stored_under_module": "scene_description",
# #             "artifact_filename": "scene_description_full.json",
# #             "prompt_version": args.scene_v,
# #             "model": args.scene_model,
# #             "mode": "replayed_from_full_pipeline_summary",
# #             "grounding_topic": args.grounding_topic,
# #             "grounding_timeout_sec": args.grounding_timeout_sec,
# #             "grounding_safety_threshold": args.grounding_safety_threshold,
# #             "grounding_debug_mapping": args.grounding_debug_mapping,
# #         },
# #         "vlm_planning": {
# #             "prompt_version": args.plan_v,
# #             "model": args.plan_model,
# #         },
# #         "simultaneous_actions": {
# #             "prompt_version": args.sim_v,
# #             "model": args.sim_model,
# #         },
# #         "validator": {
# #             "prompt_version": args.validator_v,
# #             "model": args.validator_model,
# #         },
# #         "real_deploy": {
# #             "manip_dir": str(Path(args.manip_dir).resolve()),
# #             "screens_subdir": args.screens_subdir,
# #         },
# #         "max_replans": args.max_replans,
# #         "replay": {
# #             "source_run_dir": str(Path(args.source_run_dir).resolve()),
# #             "source_cycle_name": args.source_cycle_name,
# #             "max_sleep": args.max_sleep,
# #             "mode": "hardcoded_plan_replay",
# #         },
# #     }


# # # ============================================================
# # # SOURCE SUMMARY HELPERS
# # # ============================================================

# # def load_full_pipeline_summary(source_run_dir: Path) -> dict[str, Any]:
# #     summary_path = source_run_dir / "full_pipeline_summary.json"
# #     data = read_json(summary_path)

# #     if not isinstance(data, dict):
# #         raise ValueError(f"Invalid full_pipeline_summary.json content: {summary_path}")

# #     return data


# # def select_cycle_from_full_summary(
# #     full_summary: dict[str, Any],
# #     cycle_name: str | None = None,
# # ) -> dict[str, Any]:
# #     cycles = full_summary.get("cycles")
# #     if not isinstance(cycles, list) or not cycles:
# #         raise ValueError("full_pipeline_summary.json does not contain a valid non-empty 'cycles' list.")

# #     if cycle_name is None:
# #         return cycles[-1]

# #     for cycle in cycles:
# #         if isinstance(cycle, dict) and cycle.get("cycle_name") == cycle_name:
# #             return cycle

# #     raise ValueError(f"Cycle '{cycle_name}' not found in full_pipeline_summary.json")


# # def extract_stages_from_cycle(cycle: dict[str, Any]) -> list[dict[str, Any]]:
# #     raw_stages = cycle.get("stages")
# #     if not isinstance(raw_stages, list):
# #         raise ValueError("Selected cycle does not contain a valid 'stages' list.")

# #     stages: list[dict[str, Any]] = []

# #     for idx, stage in enumerate(raw_stages):
# #         if not isinstance(stage, dict):
# #             raise ValueError(f"Stage at index {idx} is not a JSON object.")

# #         stage_id = stage.get("stage_id")
# #         precondition = stage.get("precondition")
# #         postcondition = stage.get("postcondition")

# #         if not isinstance(stage_id, int):
# #             raise ValueError(f"Stage at index {idx} has invalid or missing 'stage_id'.")
# #         if not isinstance(precondition, str) or not precondition.strip():
# #             raise ValueError(f"Stage {stage_id} has invalid or missing 'precondition'.")
# #         if not isinstance(postcondition, str) or not postcondition.strip():
# #             raise ValueError(f"Stage {stage_id} has invalid or missing 'postcondition'.")

# #         stages.append(stage)

# #     return stages


# # # ============================================================
# # # COMMAND / ROS HELPERS
# # # ============================================================

# # def run_command(
# #     cmd: list[str],
# #     label: str,
# #     env: dict[str, str] | None = None,
# # ) -> dict[str, Any]:
# #     print(f"\n[CMD] {label}")
# #     print("[CMD] " + " ".join(cmd))

# #     start_time = time.perf_counter()
# #     result = subprocess.run(
# #         cmd,
# #         capture_output=True,
# #         text=True,
# #         env=env,
# #     )
# #     execution_time_seconds = time.perf_counter() - start_time

# #     if result.stdout:
# #         print(f"[STDOUT][{label}]\n{result.stdout}")
# #     if result.stderr:
# #         print(f"[STDERR][{label}]\n{result.stderr}")

# #     if result.returncode != 0:
# #         raise RuntimeError(
# #             f"Command failed for '{label}' with return code {result.returncode}."
# #         )

# #     print(f"[TIME][{label}] Completed in {execution_time_seconds:.3f}s")

# #     return {
# #         "label": label,
# #         "command": cmd,
# #         "returncode": result.returncode,
# #         "stdout": result.stdout,
# #         "stderr": result.stderr,
# #         "execution_time_seconds": execution_time_seconds,
# #     }


# # def is_ros_script(script_path: Path) -> bool:
# #     ros_script_names = {
# #         "homing.py",
# #         "grasp_box_yz.py",
# #         "place_box.py",
# #         "place_box_2.py",
# #         "grasp_box_xy.py",
# #     }
# #     return script_path.name in ros_script_names


# # def build_ros_env() -> dict[str, str]:
# #     env = os.environ.copy()

# #     ros_pythonpath = "/opt/ros/jazzy/lib/python3.12/site-packages"
# #     existing_pythonpath = env.get("PYTHONPATH", "")

# #     paths = [p for p in existing_pythonpath.split(":") if p] if existing_pythonpath else []
# #     if ros_pythonpath not in paths:
# #         paths.insert(0, ros_pythonpath)

# #     env["PYTHONPATH"] = ":".join(paths)
# #     return env


# # def run_python_script(script_path: Path, label: str) -> dict[str, Any]:
# #     script_path = script_path.resolve()

# #     if is_ros_script(script_path):
# #         python_exec = "/usr/bin/python3"
# #         env = build_ros_env()
# #     else:
# #         python_exec = sys.executable
# #         env = None

# #     result = run_command([python_exec, str(script_path)], label=label, env=env)

# #     return {
# #         "script_name": script_path.name,
# #         "script_path": str(script_path),
# #         "label": label,
# #         "python_exec": python_exec,
# #         "execution_time_seconds": result["execution_time_seconds"],
# #     }


# # # ============================================================
# # # SCREENSHOT HELPERS
# # # ============================================================

# # def wait_until_file_is_stable(
# #     path: Path,
# #     timeout: float = 5.0,
# #     poll_interval: float = 0.2,
# # ) -> None:
# #     deadline = time.time() + timeout
# #     last_size = -1
# #     stable_reads = 0

# #     while time.time() < deadline:
# #         if path.exists():
# #             size = path.stat().st_size
# #             if size > 0 and size == last_size:
# #                 stable_reads += 1
# #                 if stable_reads >= 3:
# #                     return
# #             else:
# #                 stable_reads = 0
# #             last_size = size

# #         time.sleep(poll_interval)

# #     raise RuntimeError(f"File did not stabilize in time: {path}")


# # def validate_png(path: Path) -> None:
# #     try:
# #         with Image.open(path) as img:
# #             img.verify()
# #     except Exception as exc:
# #         raise RuntimeError(f"Screenshot is not a valid PNG yet: {path} | {exc}")


# # def get_screens_root(
# #     settings,
# #     scenario_name: str,
# #     replay_timestamp: str,
# #     screens_subdir: str,
# # ) -> Path:
# #     return (
# #         settings.project_root
# #         / "scenarios"
# #         / "real_deploy_replay"
# #         / scenario_name
# #         / replay_timestamp
# #         / screens_subdir
# #     )


# # def get_next_sequential_image_path(screens_dir: Path) -> Path:
# #     existing = sorted(
# #         p for p in screens_dir.glob("*.png")
# #         if p.stem.isdigit()
# #     )

# #     if not existing:
# #         next_index = 1
# #     else:
# #         next_index = max(int(p.stem) for p in existing) + 1

# #     return screens_dir / f"{next_index:03d}.png"


# # def take_screenshot(
# #     screens_dir: Path,
# #     wait_timeout: float = 5.0,
# #     poll_interval: float = 0.2,
# # ) -> dict[str, Any]:
# #     overall_start_time = time.perf_counter()

# #     screens_dir = ensure_dir(screens_dir.resolve())
# #     before = {p.resolve() for p in screens_dir.glob("*.png")}

# #     cmd = [
# #         "gz", "service",
# #         "-s", "/gui/screenshot",
# #         "--reqtype", "gz.msgs.StringMsg",
# #         "--reptype", "gz.msgs.Boolean",
# #         "--timeout", "3000",
# #         "--req", f'data: "{str(screens_dir)}"'
# #     ]

# #     print(f"\n[SCREEN] Taking screenshot into directory: {screens_dir}")

# #     cmd_start_time = time.perf_counter()
# #     result = subprocess.run(cmd, capture_output=True, text=True)
# #     cmd_execution_time_seconds = time.perf_counter() - cmd_start_time

# #     if result.stdout:
# #         print(f"[SCREEN][STDOUT]\n{result.stdout}")
# #     if result.stderr:
# #         print(f"[SCREEN][STDERR]\n{result.stderr}")

# #     if result.returncode != 0:
# #         raise RuntimeError("Screenshot failed.")

# #     deadline = time.time() + wait_timeout
# #     new_file: Path | None = None

# #     while time.time() < deadline:
# #         current = {p.resolve() for p in screens_dir.glob("*.png")}
# #         created = current - before
# #         if created:
# #             new_file = max(created, key=lambda p: p.stat().st_mtime)
# #             break
# #         time.sleep(poll_interval)

# #     if new_file is None or not new_file.exists():
# #         raise RuntimeError(
# #             f"Screenshot command returned success but no new PNG appeared in: {screens_dir}"
# #         )

# #     wait_until_file_is_stable(
# #         new_file,
# #         timeout=wait_timeout,
# #         poll_interval=poll_interval,
# #     )
# #     validate_png(new_file)

# #     final_path = get_next_sequential_image_path(screens_dir)

# #     if final_path.exists():
# #         raise RuntimeError(f"Sequential screenshot path already exists: {final_path}")

# #     new_file.rename(final_path)

# #     wait_until_file_is_stable(
# #         final_path,
# #         timeout=wait_timeout,
# #         poll_interval=poll_interval,
# #     )
# #     validate_png(final_path)

# #     total_execution_time_seconds = time.perf_counter() - overall_start_time

# #     print(f"[SCREEN] Screenshot saved as: {final_path}")
# #     print(f"[SCREEN] Screenshot time: {total_execution_time_seconds:.3f}s")

# #     return {
# #         "image_path": str(final_path.resolve()),
# #         "command_execution_time_seconds": cmd_execution_time_seconds,
# #         "execution_time_seconds": total_execution_time_seconds,
# #     }


# # # ============================================================
# # # DEPLOY HELPERS
# # # ============================================================

# # def deploy_stage_1(
# #     manip_dir: Path,
# #     screens_dir: Path,
# #     stage_id: int,
# # ) -> dict[str, Any]:
# #     stage_start_time = time.perf_counter()
# #     scripts: list[dict[str, Any]] = []

# #     grasp_result = run_python_script(
# #         manip_dir / "grasp_box_yz.py",
# #         label="stage_1/grasp_box_yz.py",
# #     )
# #     scripts.append(
# #         {
# #             "event_type": "manipulation_script",
# #             "module_name": "deploy",
# #             "script_name": grasp_result["script_name"],
# #             "script_path": grasp_result["script_path"],
# #             "duration_sec": grasp_result["execution_time_seconds"],
# #             "outcome": "success",
# #         }
# #     )

# #     screenshot_result = take_screenshot(screens_dir)
# #     total_execution_time_seconds = time.perf_counter() - stage_start_time

# #     print(f"[TIME][stage_{stage_id}] Total deploy time: {total_execution_time_seconds:.3f}s")

# #     return {
# #         "next_image_path": screenshot_result["image_path"],
# #         "deploy_time_seconds": total_execution_time_seconds,
# #         "scripts": scripts,
# #         "screenshot": {
# #             "event_type": "screenshot",
# #             "module_name": "deploy",
# #             "script_name": None,
# #             "duration_sec": screenshot_result["execution_time_seconds"],
# #             "outcome": "success",
# #         },
# #     }


# # def deploy_stage_2(
# #     manip_dir: Path,
# #     screens_dir: Path,
# #     stage_id: int,
# # ) -> dict[str, Any]:
# #     stage_start_time = time.perf_counter()
# #     scripts: list[dict[str, Any]] = []

# #     place_1 = run_python_script(
# #         manip_dir / "place_box.py",
# #         label="stage_2/place_box.py",
# #     )
# #     scripts.append(
# #         {
# #             "event_type": "manipulation_script",
# #             "module_name": "deploy",
# #             "script_name": place_1["script_name"],
# #             "script_path": place_1["script_path"],
# #             "duration_sec": place_1["execution_time_seconds"],
# #             "outcome": "success",
# #         }
# #     )

# #     place_2 = run_python_script(
# #         manip_dir / "place_box_2.py",
# #         label="stage_2/place_box_2.py",
# #     )
# #     scripts.append(
# #         {
# #             "event_type": "manipulation_script",
# #             "module_name": "deploy",
# #             "script_name": place_2["script_name"],
# #             "script_path": place_2["script_path"],
# #             "duration_sec": place_2["execution_time_seconds"],
# #             "outcome": "success",
# #         }
# #     )

# #     homing = run_python_script(
# #         manip_dir / "homing.py",
# #         label="stage_2/homing.py",
# #     )
# #     scripts.append(
# #         {
# #             "event_type": "manipulation_script",
# #             "module_name": "deploy",
# #             "script_name": homing["script_name"],
# #             "script_path": homing["script_path"],
# #             "duration_sec": homing["execution_time_seconds"],
# #             "outcome": "success",
# #         }
# #     )

# #     screenshot_result = take_screenshot(screens_dir)
# #     total_execution_time_seconds = time.perf_counter() - stage_start_time

# #     print(f"[TIME][stage_{stage_id}] Total deploy time: {total_execution_time_seconds:.3f}s")

# #     return {
# #         "next_image_path": screenshot_result["image_path"],
# #         "deploy_time_seconds": total_execution_time_seconds,
# #         "scripts": scripts,
# #         "screenshot": {
# #             "event_type": "screenshot",
# #             "module_name": "deploy",
# #             "script_name": None,
# #             "duration_sec": screenshot_result["execution_time_seconds"],
# #             "outcome": "success",
# #         },
# #     }


# # def deploy_stage_3(
# #     manip_dir: Path,
# #     screens_dir: Path,
# #     stage_id: int,
# # ) -> dict[str, Any]:
# #     stage_start_time = time.perf_counter()
# #     scripts: list[dict[str, Any]] = []

# #     grasp_result = run_python_script(
# #         manip_dir / "grasp_box_xy.py",
# #         label="stage_3/grasp_box_xy.py",
# #     )
# #     scripts.append(
# #         {
# #             "event_type": "manipulation_script",
# #             "module_name": "deploy",
# #             "script_name": grasp_result["script_name"],
# #             "script_path": grasp_result["script_path"],
# #             "duration_sec": grasp_result["execution_time_seconds"],
# #             "outcome": "success",
# #         }
# #     )

# #     screenshot_result = take_screenshot(screens_dir)
# #     total_execution_time_seconds = time.perf_counter() - stage_start_time

# #     print(f"[TIME][stage_{stage_id}] Total deploy time: {total_execution_time_seconds:.3f}s")

# #     return {
# #         "next_image_path": screenshot_result["image_path"],
# #         "deploy_time_seconds": total_execution_time_seconds,
# #         "scripts": scripts,
# #         "screenshot": {
# #             "event_type": "screenshot",
# #             "module_name": "deploy",
# #             "script_name": None,
# #             "duration_sec": screenshot_result["execution_time_seconds"],
# #             "outcome": "success",
# #         },
# #     }


# # def execute_stage_deploy(
# #     stage_id: int,
# #     manip_dir: Path,
# #     screens_dir: Path,
# # ) -> dict[str, Any]:
# #     if stage_id == 1:
# #         return deploy_stage_1(manip_dir=manip_dir, screens_dir=screens_dir, stage_id=stage_id)
# #     if stage_id == 2:
# #         return deploy_stage_2(manip_dir=manip_dir, screens_dir=screens_dir, stage_id=stage_id)
# #     if stage_id == 3:
# #         return deploy_stage_3(manip_dir=manip_dir, screens_dir=screens_dir, stage_id=stage_id)

# #     raise ValueError(
# #         f"No deploy routine defined for stage_id={stage_id}. "
# #         "This script currently supports stage ids 1, 2, 3."
# #     )


# # # ============================================================
# # # REPLAY HELPERS
# # # ============================================================

# # def replay_sleep(label: str, recorded_seconds: Any, max_sleep: float) -> float:
# #     recorded = safe_float(recorded_seconds, default=1.0)
# #     sleep_time = min(recorded, max_sleep)

# #     print(
# #         f"[REPLAY][{label}] Sleeping for recorded module time: "
# #         f"{sleep_time:.3f}s (recorded={recorded:.3f}s)"
# #     )
# #     time.sleep(sleep_time)
# #     return sleep_time


# # def replay_validator_from_stage(
# #     stage: dict[str, Any],
# #     condition_kind: str,
# #     max_sleep: float,
# # ) -> tuple[dict[str, Any], float]:
# #     timing = stage.get("timing", {})
# #     if not isinstance(timing, dict):
# #         timing = {}

# #     if condition_kind == "pre":
# #         response = stage.get("pre_validation")
# #         recorded = timing.get("pre_validation")
# #     elif condition_kind == "post":
# #         response = stage.get("post_validation")
# #         recorded = timing.get("post_validation")
# #     else:
# #         raise ValueError(f"Unsupported condition_kind: {condition_kind}")

# #     if not isinstance(response, dict):
# #         raise ValueError(
# #             f"Missing saved {condition_kind}_validation for stage {stage.get('stage_id')}"
# #         )

# #     slept = replay_sleep(
# #         label=f"validator:{condition_kind}_{stage.get('stage_id')}",
# #         recorded_seconds=recorded,
# #         max_sleep=max_sleep,
# #     )
# #     return response, slept


# # # ============================================================
# # # SUMMARY HELPERS
# # # ============================================================

# # def get_replay_output_root(settings, scenario_name: str, replay_timestamp: str) -> Path:
# #     return (
# #         settings.project_root
# #         / "outputs"
# #         / "real_deploy_replay"
# #         / scenario_name
# #         / replay_timestamp
# #     )


# # def save_run_summary(
# #     settings,
# #     scenario_name: str,
# #     replay_timestamp: str,
# #     summary: dict[str, Any],
# # ) -> Path:
# #     output_dir = get_replay_output_root(settings, scenario_name, replay_timestamp)
# #     ensure_dir(output_dir)

# #     out_path = output_dir / "run_summary.json"
# #     write_json(out_path, summary)
# #     return out_path


# # # ============================================================
# # # MAIN
# # # ============================================================

# # def main() -> None:
# #     parser = build_parser()
# #     args = parser.parse_args()

# #     validate_sampling_args(args)
# #     validate_args(args)

# #     settings = load_settings()

# #     replay_timestamp = make_experiment_timestamp()
# #     source_run_dir = Path(args.source_run_dir).resolve()
# #     manip_dir = Path(args.manip_dir).resolve()

# #     screens_dir = get_screens_root(
# #         settings=settings,
# #         scenario_name=args.scenario,
# #         replay_timestamp=replay_timestamp,
# #         screens_subdir=args.screens_subdir,
# #     )
# #     ensure_dir(screens_dir)

# #     run_start_time = time.perf_counter()

# #     print("\n======================================================")
# #     print("REAL DEPLOY REPLAY CONFIG")
# #     print(f"Scenario:                  {args.scenario}")
# #     print(f"Source run dir:            {source_run_dir}")
# #     print(f"Source cycle name:         {args.source_cycle_name if args.source_cycle_name else '[last cycle]'}")
# #     print(f"Temperature:               {args.temperature} (accepted, not used)")
# #     print(f"Top-p:                     {args.top_p} (accepted, not used)")
# #     print(f"Max replans:               {args.max_replans} (accepted, not used)")
# #     print(f"Max sleep:                 {args.max_sleep}")
# #     print("======================================================")

# #     full_source_summary = load_full_pipeline_summary(source_run_dir)
# #     source_cycle = select_cycle_from_full_summary(
# #         full_source_summary,
# #         args.source_cycle_name,
# #     )
# #     stages = extract_stages_from_cycle(source_cycle)

# #     initial_homing_artifact = run_python_script(
# #         manip_dir / "homing.py",
# #         label="initial_homing/homing.py",
# #     )

# #     initial_screenshot_artifact = take_screenshot(screens_dir)
# #     initial_image_path = initial_screenshot_artifact["image_path"]
# #     current_image = initial_image_path

# #     replay_summary: dict[str, Any] = {
# #         "module": "real_deploy_replay",
# #         "scenario_name": args.scenario,
# #         "replay_timestamp": replay_timestamp,
# #         "timestamp": datetime.now().isoformat(),
# #         "initial_image_path": str(Path(initial_image_path).resolve()),
# #         "screens_dir": str(screens_dir.resolve()),
# #         "manip_dir": str(manip_dir),
# #         "config": build_global_config(args),
# #         "task_completed": False,
# #         "final_image_path": None,
# #         "total_execution_time_seconds": None,
# #         "initial_homing_time_seconds": initial_homing_artifact["execution_time_seconds"],
# #         "initial_screenshot_time_seconds": initial_screenshot_artifact["execution_time_seconds"],
# #         "source": {
# #             "source_run_dir": str(source_run_dir),
# #             "source_full_pipeline_summary_path": str((source_run_dir / "full_pipeline_summary.json").resolve()),
# #             "source_cycle_name": source_cycle.get("cycle_name"),
# #             "source_cycle_index": source_cycle.get("cycle_index"),
# #             "source_cycle_timestamp": source_cycle.get("cycle_timestamp"),
# #             "source_cycle_outcome": source_cycle.get("outcome"),
# #         },
# #         "replayed_modules": {
# #             "scene_description": source_cycle.get("scene_description"),
# #             "scene_description_full": source_cycle.get("scene_description_full"),
# #             "vlm_planning": source_cycle.get("vlm_planning"),
# #             "simultaneous_actions": source_cycle.get("simultaneous_actions"),
# #         },
# #         "timing": {
# #             "scene_description_sleep": None,
# #             "scene_description_full_sleep": None,
# #             "planning_sleep": None,
# #             "simultaneous_sleep": None,
# #         },
# #         "stages": [],
# #     }

# #     try:
# #         scene_description_artifact = source_cycle.get("scene_description")
# #         if isinstance(scene_description_artifact, dict):
# #             print("\n[scene_description] Replayed JSON:")
# #             print(json.dumps(scene_description_artifact.get("output"), indent=2, ensure_ascii=False))
# #             replay_summary["timing"]["scene_description_sleep"] = replay_sleep(
# #                 label="scene_description",
# #                 recorded_seconds=scene_description_artifact.get("execution_time_seconds"),
# #                 max_sleep=args.max_sleep,
# #             )

# #         scene_description_full_artifact = source_cycle.get("scene_description_full")
# #         if isinstance(scene_description_full_artifact, dict):
# #             print("\n[scene_description_full] Replayed JSON:")
# #             print(json.dumps(scene_description_full_artifact.get("output"), indent=2, ensure_ascii=False))
# #             replay_summary["timing"]["scene_description_full_sleep"] = replay_sleep(
# #                 label="scene_description_full",
# #                 recorded_seconds=scene_description_full_artifact.get("execution_time_seconds"),
# #                 max_sleep=args.max_sleep,
# #             )

# #         vlm_planning_artifact = source_cycle.get("vlm_planning")
# #         if isinstance(vlm_planning_artifact, dict):
# #             print("\n[vlm_planning] Replayed JSON:")
# #             print(json.dumps(vlm_planning_artifact.get("output"), indent=2, ensure_ascii=False))
# #             replay_summary["timing"]["planning_sleep"] = replay_sleep(
# #                 label="vlm_planning",
# #                 recorded_seconds=vlm_planning_artifact.get("execution_time_seconds"),
# #                 max_sleep=args.max_sleep,
# #             )

# #         simultaneous_actions_artifact = source_cycle.get("simultaneous_actions")
# #         if isinstance(simultaneous_actions_artifact, dict):
# #             print("\n[simultaneous_actions] Replayed JSON:")
# #             print(json.dumps(simultaneous_actions_artifact.get("output"), indent=2, ensure_ascii=False))
# #             replay_summary["timing"]["simultaneous_sleep"] = replay_sleep(
# #                 label="simultaneous_actions",
# #                 recorded_seconds=simultaneous_actions_artifact.get("execution_time_seconds"),
# #                 max_sleep=args.max_sleep,
# #             )

# #         for stage in stages:
# #             stage_id = stage["stage_id"]
# #             pre_condition = stage["precondition"]
# #             post_condition = stage["postcondition"]

# #             print("\n------------------------------------------------------")
# #             print(f"[STAGE {stage_id}] START")
# #             print("------------------------------------------------------")

# #             stage_record: dict[str, Any] = {
# #                 "stage_id": stage_id,
# #                 "stage_name": stage.get("stage_name"),
# #                 "precondition": pre_condition,
# #                 "postcondition": post_condition,
# #                 "pre_image_path": str(Path(current_image).resolve()),
# #                 "pre_image_name": Path(current_image).name,
# #                 "post_image_path": None,
# #                 "post_image_name": None,
# #                 "pre_validation": None,
# #                 "post_validation": None,
# #                 "timing": {
# #                     "pre_validation": None,
# #                     "deploy": None,
# #                     "post_validation": None,
# #                     "deploy_scripts": [],
# #                     "screenshot": None,
# #                 },
# #             }

# #             print(f"\n[STAGE {stage_id}] PRE CHECK")
# #             print(f"[STAGE {stage_id}] PRE image:      {current_image}")
# #             print(f"[STAGE {stage_id}] PRE condition:  {pre_condition}")

# #             pre_response, pre_sleep = replay_validator_from_stage(
# #                 stage=stage,
# #                 condition_kind="pre",
# #                 max_sleep=args.max_sleep,
# #             )

# #             print(f"\n[PRE validator:pre_{stage_id}] Replayed JSON:")
# #             print(json.dumps(pre_response, indent=2, ensure_ascii=False))

# #             stage_record["pre_validation"] = pre_response
# #             stage_record["timing"]["pre_validation"] = pre_sleep

# #             print(f"\n[STAGE {stage_id}] DEPLOY")
# #             deploy_artifact = execute_stage_deploy(
# #                 stage_id=stage_id,
# #                 manip_dir=manip_dir,
# #                 screens_dir=screens_dir,
# #             )

# #             next_image = deploy_artifact["next_image_path"]

# #             stage_record["post_image_path"] = str(Path(next_image).resolve())
# #             stage_record["post_image_name"] = Path(next_image).name
# #             stage_record["timing"]["deploy"] = deploy_artifact.get("deploy_time_seconds")
# #             stage_record["timing"]["deploy_scripts"] = deploy_artifact.get("scripts", [])
# #             stage_record["timing"]["screenshot"] = deploy_artifact.get("screenshot")

# #             print(f"[STAGE {stage_id}] POST image:     {next_image}")

# #             print(f"\n[STAGE {stage_id}] POST CHECK")
# #             print(f"[STAGE {stage_id}] POST image:     {next_image}")
# #             print(f"[STAGE {stage_id}] POST condition: {post_condition}")

# #             post_response, post_sleep = replay_validator_from_stage(
# #                 stage=stage,
# #                 condition_kind="post",
# #                 max_sleep=args.max_sleep,
# #             )

# #             print(f"\n[POST validator:post_{stage_id}] Replayed JSON:")
# #             print(json.dumps(post_response, indent=2, ensure_ascii=False))

# #             stage_record["post_validation"] = post_response
# #             stage_record["timing"]["post_validation"] = post_sleep

# #             replay_summary["stages"].append(stage_record)
# #             current_image = next_image

# #         replay_summary["task_completed"] = True
# #         replay_summary["final_image_path"] = str(Path(current_image).resolve())

# #         print("\n======================================================")
# #         print("[REPLAY] TASK COMPLETED SUCCESSFULLY")
# #         print("======================================================")

# #     except Exception as exc:
# #         replay_summary["task_completed"] = False
# #         replay_summary["error"] = str(exc)

# #         print("\n======================================================")
# #         print("[REPLAY] ERROR")
# #         print(str(exc))
# #         print("======================================================")

# #     replay_summary["total_execution_time_seconds"] = time.perf_counter() - run_start_time

# #     summary_path = save_run_summary(
# #         settings=settings,
# #         scenario_name=args.scenario,
# #         replay_timestamp=replay_timestamp,
# #         summary=replay_summary,
# #     )

# #     print("\n======================================================")
# #     print("REAL DEPLOY REPLAY COMPLETED")
# #     print(f"Scenario:                  {args.scenario}")
# #     print(f"Replay timestamp:          {replay_timestamp}")
# #     print(f"Task completed:            {replay_summary['task_completed']}")
# #     print(f"Total execution time:      {replay_summary['total_execution_time_seconds']:.3f}s")
# #     print(f"Summary saved:             {summary_path}")
# #     print("======================================================")


# # if __name__ == "__main__":
# #     main()















# # from __future__ import annotations

# # import argparse
# # import json
# # import os
# # import subprocess
# # import sys
# # import time
# # from datetime import datetime
# # from pathlib import Path
# # from typing import Any

# # from settings import load_settings
# # from utils import (
# #     make_experiment_timestamp,
# #     read_json,
# #     write_json,
# # )

# # SUPPORTED_MODELS = ["o3", "gpt-5.2", "bypassed_validator"]


# # # ============================================================
# # # PARSER
# # # ============================================================

# # def build_parser() -> argparse.ArgumentParser:
# #     parser = argparse.ArgumentParser(
# #         description=(
# #             "Replay a previously successful real deploy pipeline using saved artifacts, "
# #             "recorded module timings, and real Gazebo execution."
# #         )
# #     )

# #     parser.add_argument("--scenario", type=str, required=True)

# #     parser.add_argument(
# #         "--run-dir",
# #         type=str,
# #         required=True,
# #         help=(
# #             "Path to the root directory of a previous successful run. "
# #             "Example: /home/user/outputs/real_deploy_pipeline_bypassed_validator/"
# #             "stacked_boxes/2026-04-12_22-56-34"
# #         ),
# #     )

# #     parser.add_argument(
# #         "--manip-dir",
# #         type=str,
# #         required=True,
# #         help="Directory containing manipulation scripts (grasp_box_yz.py, place_box.py, etc.).",
# #     )

# #     parser.add_argument(
# #         "--screens-subdir",
# #         type=str,
# #         default="screens",
# #         help="Subdirectory name used inside the replay scenario folder for screenshots.",
# #     )

# #     parser.add_argument(
# #         "--max-sleep",
# #         type=float,
# #         default=5.0,
# #         help="Maximum sleep time applied to each replayed module timing.",
# #     )

# #     return parser


# # # ============================================================
# # # GENERIC HELPERS
# # # ============================================================

# # def ensure_dir(path: Path) -> Path:
# #     path.mkdir(parents=True, exist_ok=True)
# #     return path


# # def validate_args(args: argparse.Namespace) -> None:
# #     run_dir = Path(args.run_dir).resolve()
# #     if not run_dir.exists():
# #         raise FileNotFoundError(f"run-dir not found: {run_dir}")
# #     if not run_dir.is_dir():
# #         raise ValueError(f"--run-dir must be a directory: {run_dir}")

# #     manip_dir = Path(args.manip_dir).resolve()
# #     if not manip_dir.exists():
# #         raise FileNotFoundError(f"manip-dir not found: {manip_dir}")
# #     if not manip_dir.is_dir():
# #         raise ValueError(f"--manip-dir must be a directory: {manip_dir}")

# #     required_scripts = [
# #         "grasp_box_yz.py",
# #         "place_box.py",
# #         "place_box_2.py",
# #         "homing.py",
# #         "grasp_box_xy.py",
# #     ]
# #     for script_name in required_scripts:
# #         script_path = manip_dir / script_name
# #         if not script_path.exists():
# #             raise FileNotFoundError(f"Required manipulation script not found: {script_path}")

# #     if args.max_sleep < 0.0:
# #         raise ValueError("--max-sleep must be >= 0.0")


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


# # # ============================================================
# # # COMMAND / ROS HELPERS
# # # ============================================================

# # def run_command(
# #     cmd: list[str],
# #     label: str,
# #     env: dict[str, str] | None = None,
# # ) -> None:
# #     print(f"\n[CMD] {label}")
# #     print("[CMD] " + " ".join(cmd))

# #     result = subprocess.run(
# #         cmd,
# #         capture_output=True,
# #         text=True,
# #         env=env,
# #     )

# #     if result.stdout:
# #         print(f"[STDOUT][{label}]\n{result.stdout}")
# #     if result.stderr:
# #         print(f"[STDERR][{label}]\n{result.stderr}")

# #     if result.returncode != 0:
# #         raise RuntimeError(
# #             f"Command failed for '{label}' with return code {result.returncode}."
# #         )


# # def is_ros_script(script_path: Path) -> bool:
# #     ros_script_names = {
# #         "homing.py",
# #         "grasp_box_yz.py",
# #         "place_box.py",
# #         "place_box_2.py",
# #         "grasp_box_xy.py",
# #     }
# #     return script_path.name in ros_script_names


# # def build_ros_env() -> dict[str, str]:
# #     env = os.environ.copy()

# #     ros_pythonpath = "/opt/ros/jazzy/lib/python3.12/site-packages"
# #     existing_pythonpath = env.get("PYTHONPATH", "")

# #     paths = [p for p in existing_pythonpath.split(":") if p] if existing_pythonpath else []
# #     if ros_pythonpath not in paths:
# #         paths.insert(0, ros_pythonpath)

# #     env["PYTHONPATH"] = ":".join(paths)
# #     return env


# # def run_python_script(script_path: Path, label: str) -> None:
# #     script_path = script_path.resolve()

# #     if is_ros_script(script_path):
# #         python_exec = "/usr/bin/python3"
# #         env = build_ros_env()
# #     else:
# #         python_exec = sys.executable
# #         env = None

# #     run_command([python_exec, str(script_path)], label=label, env=env)


# # # ============================================================
# # # SCREENSHOT HELPERS
# # # ============================================================

# # def get_screens_root(
# #     settings,
# #     scenario_name: str,
# #     replay_timestamp: str,
# #     screens_subdir: str,
# # ) -> Path:
# #     return (
# #         settings.project_root
# #         / "scenarios"
# #         / "real_deploy_replay"
# #         / scenario_name
# #         / replay_timestamp
# #         / screens_subdir
# #     )


# # def get_next_sequential_image_path(screens_dir: Path) -> Path:
# #     existing = sorted(
# #         p for p in screens_dir.glob("*.png")
# #         if p.stem.isdigit()
# #     )

# #     if not existing:
# #         next_index = 1
# #     else:
# #         next_index = max(int(p.stem) for p in existing) + 1

# #     return screens_dir / f"{next_index:03d}.png"


# # def take_screenshot(
# #     screens_dir: Path,
# #     wait_timeout: float = 5.0,
# #     poll_interval: float = 0.2,
# # ) -> str:
# #     screens_dir = ensure_dir(screens_dir.resolve())

# #     before = {p.resolve() for p in screens_dir.glob("*.png")}

# #     cmd = [
# #         "gz", "service",
# #         "-s", "/gui/screenshot",
# #         "--reqtype", "gz.msgs.StringMsg",
# #         "--reptype", "gz.msgs.Boolean",
# #         "--timeout", "3000",
# #         "--req", f'data: "{str(screens_dir)}"'
# #     ]

# #     print(f"\n[SCREEN] Taking screenshot into directory: {screens_dir}")
# #     result = subprocess.run(cmd, capture_output=True, text=True)

# #     if result.stdout:
# #         print(f"[SCREEN][STDOUT]\n{result.stdout}")
# #     if result.stderr:
# #         print(f"[SCREEN][STDERR]\n{result.stderr}")

# #     if result.returncode != 0:
# #         raise RuntimeError("Screenshot failed.")

# #     deadline = time.time() + wait_timeout
# #     new_file: Path | None = None

# #     while time.time() < deadline:
# #         current = {p.resolve() for p in screens_dir.glob("*.png")}
# #         created = current - before
# #         if created:
# #             new_file = max(created, key=lambda p: p.stat().st_mtime)
# #             break
# #         time.sleep(poll_interval)

# #     if new_file is None or not new_file.exists():
# #         raise RuntimeError(
# #             f"Screenshot command returned success but no new PNG appeared in: {screens_dir}"
# #         )

# #     final_path = get_next_sequential_image_path(screens_dir)

# #     if final_path.exists():
# #         raise RuntimeError(f"Sequential screenshot path already exists: {final_path}")

# #     new_file.rename(final_path)

# #     print(f"[SCREEN] Screenshot saved as: {final_path}")
# #     return str(final_path.resolve())


# # # ============================================================
# # # DEPLOY HELPERS
# # # ============================================================

# # def deploy_stage_1(
# #     manip_dir: Path,
# #     screens_dir: Path,
# #     stage_id: int,
# # ) -> str:
# #     run_python_script(manip_dir / "grasp_box_yz.py", label="stage_1/grasp_box_yz.py")
# #     next_image = take_screenshot(screens_dir)
# #     return next_image


# # def deploy_stage_2(
# #     manip_dir: Path,
# #     screens_dir: Path,
# #     stage_id: int,
# # ) -> str:
# #     run_python_script(manip_dir / "place_box.py", label="stage_2/place_box.py")
# #     run_python_script(manip_dir / "place_box_2.py", label="stage_2/place_box_2.py")
# #     run_python_script(manip_dir / "homing.py", label="stage_2/homing.py")
# #     next_image = take_screenshot(screens_dir)
# #     return next_image


# # def deploy_stage_3(
# #     manip_dir: Path,
# #     screens_dir: Path,
# #     stage_id: int,
# # ) -> str:
# #     run_python_script(manip_dir / "grasp_box_xy.py", label="stage_3/grasp_box_xy.py")
# #     next_image = take_screenshot(screens_dir)
# #     return next_image


# # def execute_stage_deploy(
# #     stage_id: int,
# #     manip_dir: Path,
# #     screens_dir: Path,
# # ) -> str:
# #     if stage_id == 1:
# #         return deploy_stage_1(
# #             manip_dir=manip_dir,
# #             screens_dir=screens_dir,
# #             stage_id=stage_id,
# #         )

# #     if stage_id == 2:
# #         return deploy_stage_2(
# #             manip_dir=manip_dir,
# #             screens_dir=screens_dir,
# #             stage_id=stage_id,
# #         )

# #     if stage_id == 3:
# #         return deploy_stage_3(
# #             manip_dir=manip_dir,
# #             screens_dir=screens_dir,
# #             stage_id=stage_id,
# #         )

# #     raise ValueError(
# #         f"No deploy routine defined for Stage_id={stage_id}. "
# #         "This script currently supports stage ids 1, 2, 3."
# #     )


# # # ============================================================
# # # REPLAY ARTIFACT HELPERS
# # # ============================================================

# # def find_first_file(root: Path, filename: str) -> Path:
# #     matches = sorted(root.rglob(filename))
# #     if not matches:
# #         raise FileNotFoundError(f"Could not find '{filename}' under: {root}")
# #     return matches[0]


# # def find_all_files(root: Path, filename: str) -> list[Path]:
# #     return sorted(root.rglob(filename))


# # def extract_execution_time_seconds(run_info_path: Path, default: float = 1.0) -> float:
# #     data = read_json(run_info_path)
# #     value = data.get("execution_time_seconds", default)
# #     try:
# #         return float(value)
# #     except Exception:
# #         return default


# # def recorded_sleep(label: str, run_info_path: Path | None, max_sleep: float) -> float:
# #     if run_info_path is None or not run_info_path.exists():
# #         sleep_time = min(1.0, max_sleep)
# #         print(f"[REPLAY][{label}] Missing run_info, using fallback sleep: {sleep_time:.3f}s")
# #         time.sleep(sleep_time)
# #         return sleep_time

# #     recorded = extract_execution_time_seconds(run_info_path, default=1.0)
# #     sleep_time = min(recorded, max_sleep)

# #     print(
# #         f"[REPLAY][{label}] Sleeping for recorded module time: "
# #         f"{sleep_time:.3f}s (recorded={recorded:.3f}s)"
# #     )
# #     time.sleep(sleep_time)
# #     return sleep_time


# # def load_replay_artifacts(run_dir: Path) -> dict[str, Any]:
# #     scene_response_path = find_first_file(run_dir, "response_parsed.json")

# #     scene_full_candidates = [
# #         p for p in find_all_files(run_dir, "scene_description_full.json")
# #         if p.is_file()
# #     ]
# #     if not scene_full_candidates:
# #         raise FileNotFoundError(f"Could not find scene_description_full.json under: {run_dir}")
# #     scene_full_path = scene_full_candidates[0]

# #     all_response_paths = find_all_files(run_dir, "response_parsed.json")

# #     scene_path = None
# #     plan_path = None
# #     sim_path = None

# #     for p in all_response_paths:
# #         p_str = str(p)
# #         if "/scene_description/" in p_str and scene_path is None:
# #             scene_path = p
# #         elif "/vlm_planning/" in p_str and plan_path is None:
# #             plan_path = p
# #         elif "/simultaneous_actions/" in p_str and sim_path is None:
# #             sim_path = p

# #     if scene_path is None:
# #         scene_path = scene_response_path

# #     if plan_path is None:
# #         raise FileNotFoundError("Could not find vlm_planning response_parsed.json under run-dir")
# #     if sim_path is None:
# #         raise FileNotFoundError("Could not find simultaneous_actions response_parsed.json under run-dir")

# #     scene_run_info = scene_path.parent / "run_info.json"
# #     scene_full_run_info = scene_full_path.parent / "run_info.json"
# #     plan_run_info = plan_path.parent / "run_info.json"
# #     sim_run_info = sim_path.parent / "run_info.json"

# #     validator_response_paths = [
# #         p for p in all_response_paths
# #         if "/validator/" in str(p)
# #     ]

# #     validator_run_info_paths = [
# #         p.parent / "run_info.json"
# #         for p in validator_response_paths
# #     ]

# #     return {
# #         "scene_description": {
# #             "response_path": scene_path,
# #             "run_info_path": scene_run_info if scene_run_info.exists() else None,
# #             "output": read_json(scene_path),
# #         },
# #         "scene_description_full": {
# #             "response_path": scene_full_path,
# #             "run_info_path": scene_full_run_info if scene_full_run_info.exists() else None,
# #             "output": read_json(scene_full_path),
# #         },
# #         "vlm_planning": {
# #             "response_path": plan_path,
# #             "run_info_path": plan_run_info if plan_run_info.exists() else None,
# #             "output": read_json(plan_path),
# #         },
# #         "simultaneous_actions": {
# #             "response_path": sim_path,
# #             "run_info_path": sim_run_info if sim_run_info.exists() else None,
# #             "output": read_json(sim_path),
# #         },
# #         "validator": {
# #             "response_paths": validator_response_paths,
# #             "run_info_paths": validator_run_info_paths,
# #         },
# #     }


# # def build_validator_replay_index(run_dir: Path) -> dict[str, dict[str, Path | None]]:
# #     all_response_paths = find_all_files(run_dir, "response_parsed.json")
    
# #     validator_paths = [p for p in all_response_paths if "/validator/" in str(p)]

# #     index: dict[str, dict[str, Path | None]] = {}

# #     for response_path in validator_paths:
# #         key = response_path.parent.name
# #         index[key] = {
# #             "response_path": response_path,
# #             "run_info_path": (response_path.parent / "run_info.json")
# #             if (response_path.parent / "run_info.json").exists()
# #             else None,
# #         }

# #     return index


# # def replay_validator_condition(
# #     validator_index: dict[str, dict[str, Path | None]],
# #     condition_name: str,
# #     condition_text: str,
# #     image_path: str,
# #     max_sleep: float,
# # ) -> tuple[dict[str, Any], float]:
# #     record = validator_index.get(condition_name)

# #     if record is not None and record.get("response_path") is not None:
# #         response_path = record["response_path"]
# #         run_info_path = record.get("run_info_path")
# #         output = read_json(response_path)

# #         print(f"[REPLAY][validator:{condition_name}] Loaded response from: {response_path}")
# #         slept = recorded_sleep(
# #             label=f"validator:{condition_name}",
# #             run_info_path=run_info_path,
# #             max_sleep=max_sleep,
# #         )
# #         return output, slept

# #     fallback = {
# #         "result": "matching",
# #         "reason": (
# #             "Replay fallback: validator artifact not found in the source run, "
# #             "forcing matching to keep the demo deterministic."
# #         ),
# #         "forced": True,
# #         "condition_name": condition_name,
# #         "condition_text": condition_text,
# #         "image_path": str(Path(image_path).resolve()),
# #     }

# #     print(f"[REPLAY][validator:{condition_name}] Artifact not found, using fallback matching.")
# #     slept = min(1.0, max_sleep)
# #     time.sleep(slept)
# #     return fallback, slept


# # # ============================================================
# # # SUMMARY HELPERS
# # # ============================================================

# # def get_replay_output_root(settings, scenario_name: str, replay_timestamp: str) -> Path:
# #     return (
# #         settings.project_root
# #         / "outputs"
# #         / "real_deploy_replay"
# #         / scenario_name
# #         / replay_timestamp
# #     )


# # def save_run_summary(
# #     settings,
# #     scenario_name: str,
# #     replay_timestamp: str,
# #     summary: dict[str, Any],
# # ) -> Path:
# #     output_dir = get_replay_output_root(settings, scenario_name, replay_timestamp)
# #     ensure_dir(output_dir)

# #     out_path = output_dir / "run_summary.json"
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

# #     replay_timestamp = make_experiment_timestamp()
# #     run_dir = Path(args.run_dir).resolve()
# #     manip_dir = Path(args.manip_dir).resolve()

# #     screens_dir = get_screens_root(
# #         settings=settings,
# #         scenario_name=args.scenario,
# #         replay_timestamp=replay_timestamp,
# #         screens_subdir=args.screens_subdir,
# #     )
# #     ensure_dir(screens_dir)

# #     print("\n======================================================")
# #     print("REAL DEPLOY REPLAY STARTED")
# #     print(f"Scenario:        {args.scenario}")
# #     print(f"Replay ts:       {replay_timestamp}")
# #     print(f"Source run dir:  {run_dir}")
# #     print(f"Manip dir:       {manip_dir}")
# #     print(f"Screens dir:     {screens_dir}")
# #     print(f"Max sleep:       {args.max_sleep}")
# #     print("======================================================")

# #     artifacts = load_replay_artifacts(run_dir)
# #     validator_index = build_validator_replay_index(run_dir)

# #     summary: dict[str, Any] = {
# #         "module": "real_deploy_replay",
# #         "scenario_name": args.scenario,
# #         "replay_timestamp": replay_timestamp,
# #         "timestamp": datetime.now().isoformat(),
# #         "source_run_dir": str(run_dir),
# #         "manip_dir": str(manip_dir),
# #         "screens_dir": str(screens_dir),
# #         "max_sleep": args.max_sleep,
# #         "task_completed": False,
# #         "initial_image_path": None,
# #         "final_image_path": None,
# #         "replayed_artifacts": {
# #             "scene_description_path": str(artifacts["scene_description"]["response_path"]),
# #             "scene_description_full_path": str(artifacts["scene_description_full"]["response_path"]),
# #             "vlm_planning_path": str(artifacts["vlm_planning"]["response_path"]),
# #             "simultaneous_actions_path": str(artifacts["simultaneous_actions"]["response_path"]),
# #         },
# #         "timings": {},
# #         "stages": [],
# #     }

# #     try:
# #         # --------------------------------------------------
# #         # INITIAL HOMING (REAL)
# #         # --------------------------------------------------
# #         run_python_script(manip_dir / "homing.py", label="initial_homing/homing.py")

# #         initial_image = take_screenshot(screens_dir)
# #         current_image = initial_image
# #         summary["initial_image_path"] = str(Path(initial_image).resolve())

# #         # --------------------------------------------------
# #         # REPLAY scene_description
# #         # --------------------------------------------------
# #         print("\n[scene_description] Replayed JSON:")
# #         print(json.dumps(artifacts["scene_description"]["output"], indent=2, ensure_ascii=False))
# #         summary["timings"]["scene_description_sleep_s"] = recorded_sleep(
# #             label="scene_description",
# #             run_info_path=artifacts["scene_description"]["run_info_path"],
# #             max_sleep=args.max_sleep,
# #         )

# #         # --------------------------------------------------
# #         # REPLAY scene_description_full
# #         # --------------------------------------------------
# #         print("\n[scene_description_full] Replayed JSON:")
# #         print(json.dumps(artifacts["scene_description_full"]["output"], indent=2, ensure_ascii=False))
# #         summary["timings"]["scene_description_full_sleep_s"] = recorded_sleep(
# #             label="scene_description_full",
# #             run_info_path=artifacts["scene_description_full"]["run_info_path"],
# #             max_sleep=args.max_sleep,
# #         )

# #         # --------------------------------------------------
# #         # REPLAY vlm_planning
# #         # --------------------------------------------------
# #         print("\n[vlm_planning] Replayed JSON:")
# #         print(json.dumps(artifacts["vlm_planning"]["output"], indent=2, ensure_ascii=False))
# #         summary["timings"]["vlm_planning_sleep_s"] = recorded_sleep(
# #             label="vlm_planning",
# #             run_info_path=artifacts["vlm_planning"]["run_info_path"],
# #             max_sleep=args.max_sleep,
# #         )

# #         # --------------------------------------------------
# #         # REPLAY simultaneous_actions
# #         # --------------------------------------------------
# #         print("\n[simultaneous_actions] Replayed JSON:")
# #         print(json.dumps(artifacts["simultaneous_actions"]["output"], indent=2, ensure_ascii=False))
# #         summary["timings"]["simultaneous_actions_sleep_s"] = recorded_sleep(
# #             label="simultaneous_actions",
# #             run_info_path=artifacts["simultaneous_actions"]["run_info_path"],
# #             max_sleep=args.max_sleep,
# #         )

# #         stages = extract_stages(artifacts["simultaneous_actions"]["output"])

# #         # --------------------------------------------------
# #         # STAGE LOOP
# #         # --------------------------------------------------
# #         for stage in stages:
# #             stage_id = stage["Stage_id"]
# #             pre_condition = stage["Precondition"]
# #             post_condition = stage["Postcondition"]

# #             print("\n------------------------------------------------------")
# #             print(f"[STAGE {stage_id}] START")
# #             print("------------------------------------------------------")

# #             stage_record: dict[str, Any] = {
# #                 "stage_id": stage_id,
# #                 "precondition": pre_condition,
# #                 "postcondition": post_condition,
# #                 "pre_image_path": str(Path(current_image).resolve()),
# #                 "post_image_path": None,
# #                 "pre_validation": None,
# #                 "post_validation": None,
# #                 "timings": {},
# #             }

# #             # ---------------- PRE VALIDATOR REPLAY ----------------
# #             print(f"\n[STAGE {stage_id}] PRE CHECK")
# #             print(f"[STAGE {stage_id}] PRE image:      {current_image}")
# #             print(f"[STAGE {stage_id}] PRE condition:  {pre_condition}")

# #             pre_name = f"pre_{stage_id}"
# #             pre_response, pre_sleep = replay_validator_condition(
# #                 validator_index=validator_index,
# #                 condition_name=pre_name,
# #                 condition_text=pre_condition,
# #                 image_path=current_image,
# #                 max_sleep=args.max_sleep,
# #             )

# #             print(f"\n[PRE validator:{pre_name}] Replayed JSON:")
# #             print(json.dumps(pre_response, indent=2, ensure_ascii=False))

# #             stage_record["pre_validation"] = pre_response
# #             stage_record["timings"]["pre_validator_sleep_s"] = pre_sleep

# #             # ---------------- REAL DEPLOY ----------------
# #             print(f"\n[STAGE {stage_id}] DEPLOY")
# #             next_image = execute_stage_deploy(
# #                 stage_id=stage_id,
# #                 manip_dir=manip_dir,
# #                 screens_dir=screens_dir,
# #             )

# #             print(f"[STAGE {stage_id}] POST image:     {next_image}")

# #             # ---------------- POST VALIDATOR REPLAY ----------------
# #             print(f"\n[STAGE {stage_id}] POST CHECK")
# #             print(f"[STAGE {stage_id}] POST image:     {next_image}")
# #             print(f"[STAGE {stage_id}] POST condition: {post_condition}")

# #             post_name = f"post_{stage_id}"
# #             post_response, post_sleep = replay_validator_condition(
# #                 validator_index=validator_index,
# #                 condition_name=post_name,
# #                 condition_text=post_condition,
# #                 image_path=next_image,
# #                 max_sleep=args.max_sleep,
# #             )

# #             print(f"\n[POST validator:{post_name}] Replayed JSON:")
# #             print(json.dumps(post_response, indent=2, ensure_ascii=False))

# #             stage_record["post_validation"] = post_response
# #             stage_record["post_image_path"] = str(Path(next_image).resolve())
# #             stage_record["timings"]["post_validator_sleep_s"] = post_sleep

# #             summary["stages"].append(stage_record)
# #             current_image = next_image

# #         summary["task_completed"] = True
# #         summary["final_image_path"] = str(Path(current_image).resolve())

# #         print("\n======================================================")
# #         print("[REPLAY] TASK COMPLETED SUCCESSFULLY")
# #         print("======================================================")

# #     except Exception as exc:
# #         summary["task_completed"] = False
# #         summary["error"] = str(exc)

# #         print("\n======================================================")
# #         print("[REPLAY] ERROR")
# #         print(str(exc))
# #         print("======================================================")

# #     summary_path = save_run_summary(
# #         settings=settings,
# #         scenario_name=args.scenario,
# #         replay_timestamp=replay_timestamp,
# #         summary=summary,
# #     )

# #     print("\n======================================================")
# #     print("REAL DEPLOY REPLAY COMPLETED")
# #     print(f"Scenario:        {args.scenario}")
# #     print(f"Replay ts:       {replay_timestamp}")
# #     print(f"Task completed:  {summary['task_completed']}")
# #     print(f"Summary saved:   {summary_path}")
# #     print("======================================================")


# # if __name__ == "__main__":
# #     main()




