"""Continue the live validation pipeline from the latest scene perception.

This script assumes run_validation_live1.py has already produced a
scene_description response for the latest Realsense frame. It loads the latest
response_parsed.json, takes the image from shared_data/realsense/rgb, takes
object poses from shared_data/realsense/outputs_by_object, runs scene enrichment,
planning, scheduling, and validation. If a validation failure would require
replanning, the script stops.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from settings import load_settings
from utils import make_experiment_timestamp, read_json
from run_validation_live import (
    IMAGE_EXTENSIONS,
    SUPPORTED_MODELS,
    build_full_pipeline_summary,
    build_loop_summary,
    build_run_info,
    build_scene_description_summary,
    build_simultaneous_actions_summary,
    build_validator_summary,
    build_vlm_planning_summary,
    build_cycle_summary,
    execute_scene_description_full_step,
    execute_simultaneous_actions_step,
    execute_validator_step,
    execute_vlm_planning_step,
    extract_stages,
    load_poses_by_image_map,
    print_pose_dict_for_image,
    save_cycle_summary,
    save_validation_loop_artifacts,
    validate_sampling_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continue validation from the latest live1 scene_description output."
    )
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--shared-data-root", type=str, default="/workspace/shared_data")
    parser.add_argument("--realsense-timestamp", type=str, default=None)
    parser.add_argument("--poses-timestamp", type=str, default=None)
    parser.add_argument("--poses-path", type=str, default=None)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task for planning/scheduling. live2 does not read scenario.json.",
    )

    parser.add_argument("--scene-v", type=str, required=True)
    parser.add_argument("--plan-v", type=str, required=True)
    parser.add_argument("--sim-v", type=str, required=True)
    parser.add_argument("--validator-v", type=str, required=True)

    parser.add_argument("--scene-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--plan-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--sim-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--validator-model", type=str, required=True, choices=SUPPORTED_MODELS)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--grounding-safety-threshold", type=float, default=0.11)
    parser.add_argument("--grounding-debug-mapping", action="store_true")
    parser.add_argument(
        "--new-frame-timeout",
        type=float,
        default=300.0,
        help=(
            "Seconds to wait after a matching precondition for a newer "
            "Realsense RGB timestamp to appear."
        ),
    )
    parser.add_argument(
        "--new-frame-poll-seconds",
        type=float,
        default=1.0,
        help="Polling interval while waiting for the next Realsense RGB frame.",
    )
    parser.add_argument(
        "--new-frame-file-timeout",
        type=float,
        default=30.0,
        help=(
            "Seconds to wait after a new timestamp is detected until "
            "rgb/<timestamp>/000000.png exists and is stable."
        ),
    )
    parser.add_argument(
        "--new-frame-file-poll-seconds",
        type=float,
        default=0.1,
        help="Polling interval while waiting for rgb/<timestamp>/000000.png to be ready.",
    )
    return parser


def get_latest_timestamp(root: Path) -> str:
    if not root.exists():
        raise FileNotFoundError(f"Timestamp root does not exist: {root}")

    timestamp_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )

    if not timestamp_dirs:
        raise RuntimeError(f"No timestamp folders found in: {root}")

    return timestamp_dirs[-1].name


def get_timestamp_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Timestamp root does not exist: {root}")

    return sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )


def wait_for_realsense_rgb_ready(
    shared_data_root: str | Path,
    realsense_timestamp: str,
    image_name: str = "000000.png",
    timeout_seconds: float = 30.0,
    poll_seconds: float = 0.1,
    stable_checks_required: int = 2,
) -> Path:
    """Wait until rgb/<timestamp>/000000.png exists, is non-empty, and is stable.

    This prevents the race condition where live2 detects a new timestamp folder
    before the image file is completely visible inside perception_docker.
    """
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be greater than 0.")
    if poll_seconds <= 0.0:
        raise ValueError("poll_seconds must be greater than 0.")
    if stable_checks_required < 1:
        raise ValueError("stable_checks_required must be at least 1.")

    image_path = (
        Path(shared_data_root)
        / "realsense"
        / "rgb"
        / realsense_timestamp
        / image_name
    ).resolve()

    deadline = time.monotonic() + timeout_seconds
    last_size: int | None = None
    stable_checks = 0

    while time.monotonic() < deadline:
        if image_path.is_file():
            size = image_path.stat().st_size

            if size > 0:
                if last_size is not None and size == last_size:
                    stable_checks += 1
                else:
                    stable_checks = 1
                    last_size = size

                if stable_checks >= stable_checks_required:
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        raise ValueError(f"Unsupported image extension: {image_path.suffix}")

                    return image_path

        time.sleep(poll_seconds)

    raise FileNotFoundError(
        "Expected Realsense RGB image was not ready before timeout: "
        f"{image_path} "
        f"(timeout={timeout_seconds:.1f}s, poll={poll_seconds:.3f}s)"
    )


def wait_for_new_realsense_frame(
    shared_data_root: str | Path,
    previous_timestamp: str,
    timeout_seconds: float,
    poll_seconds: float,
    file_timeout_seconds: float = 30.0,
    file_poll_seconds: float = 0.1,
) -> tuple[Path, str]:
    if timeout_seconds <= 0.0:
        raise ValueError("--new-frame-timeout must be greater than 0.")
    if poll_seconds <= 0.0:
        raise ValueError("--new-frame-poll-seconds must be greater than 0.")
    if file_timeout_seconds <= 0.0:
        raise ValueError("--new-frame-file-timeout must be greater than 0.")
    if file_poll_seconds <= 0.0:
        raise ValueError("--new-frame-file-poll-seconds must be greater than 0.")

    rgb_root = Path(shared_data_root) / "realsense" / "rgb"
    previous_dir = rgb_root / previous_timestamp

    if previous_dir.exists():
        previous_mtime = previous_dir.stat().st_mtime
    else:
        previous_mtime = 0.0

    deadline = time.monotonic() + timeout_seconds

    print(
        "[LOOP] Waiting for a new Realsense frame newer than "
        f"{previous_timestamp}..."
    )

    while time.monotonic() < deadline:
        timestamp_dirs = get_timestamp_dirs(rgb_root)

        newer_dirs = [
            path
            for path in timestamp_dirs
            if path.name != previous_timestamp and path.stat().st_mtime > previous_mtime
        ]

        if newer_dirs:
            latest_dir = newer_dirs[-1]
            new_timestamp = latest_dir.name

            print(f"[LOOP] New timestamp directory detected: {new_timestamp}")
            print("[LOOP] Waiting for RGB file to be ready...")

            image_path = wait_for_realsense_rgb_ready(
                shared_data_root=shared_data_root,
                realsense_timestamp=new_timestamp,
                image_name="000000.png",
                timeout_seconds=file_timeout_seconds,
                poll_seconds=file_poll_seconds,
            )

            return image_path, new_timestamp

        time.sleep(poll_seconds)

    raise TimeoutError(
        "Timed out waiting for a new Realsense RGB frame in "
        f"{rgb_root} after {timeout_seconds:.1f}s. "
        "Run the capture script after the robot action finishes."
    )


def resolve_realsense_timestamp(shared_data_root: str | Path, explicit_timestamp: str | None) -> str:
    if explicit_timestamp is not None and explicit_timestamp.strip():
        return explicit_timestamp.strip()
    return get_latest_timestamp(Path(shared_data_root) / "realsense" / "rgb")


def resolve_realsense_image_path(
    shared_data_root: str | Path,
    realsense_timestamp: str,
) -> Path:
    image_path = (
        Path(shared_data_root)
        / "realsense"
        / "rgb"
        / realsense_timestamp
        / "000000.png"
    ).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Expected Realsense RGB image not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Expected Realsense RGB image to be a file: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {image_path.suffix}")

    return image_path


def resolve_live_poses_path(
    shared_data_root: str | Path,
    explicit_timestamp: str | None,
    explicit_path: str | None,
) -> tuple[Path, str]:
    if explicit_path is not None:
        path = Path(explicit_path).resolve()
        poses_timestamp = path.parent.name
    else:
        output_root = Path(shared_data_root) / "realsense" / "outputs_by_object"
        poses_timestamp = (
            explicit_timestamp.strip()
            if explicit_timestamp is not None and explicit_timestamp.strip()
            else get_latest_timestamp(output_root)
        )
        path = (output_root / poses_timestamp / "poses.json").resolve()

    if not path.exists():
        raise FileNotFoundError(f"Live poses file not found: {path}")

    return path, poses_timestamp


def resolve_task(explicit_task: str | None) -> str:
    if explicit_task is not None and explicit_task.strip():
        return explicit_task.strip()

    raise ValueError("live2 needs a non-empty --task for vlm_planning and simultaneous_actions.")


def find_latest_scene_description_response(
    settings,
    scenario_name: str,
    scene_version: str,
    scene_model: str,
) -> tuple[Path, str, str]:
    search_root = (
        settings.project_root
        / "outputs"
        / "scene_description"
        / scenario_name
        / scene_version
    )

    if not search_root.exists():
        raise FileNotFoundError(f"No scene_description output directory found: {search_root}")

    candidates = [
        path
        for path in search_root.glob(f"*/{scene_model}/*/response_parsed.json")
        if path.is_file()
    ]

    if not candidates:
        raise FileNotFoundError(
            "No response_parsed.json found for "
            f"scenario={scenario_name}, scene_v={scene_version}, model={scene_model}"
        )

    latest_path = max(candidates, key=lambda path: path.stat().st_mtime)
    cycle_name = latest_path.parent.name
    loop_timestamp = latest_path.parent.parent.parent.name
    return latest_path, loop_timestamp, cycle_name


def build_scene_artifact_from_path(response_path: Path) -> dict[str, Any]:
    output_dir = response_path.parent
    return {
        "output": read_json(response_path),
        "paths": {
            "prompt": str(output_dir / "prompt.txt"),
            "response_parsed": str(response_path),
            "run_info": str(output_dir / "run_info.json"),
            "scene_object_list": str(output_dir / "scene_object_list.json"),
        },
    }


def build_live2_config(
    args: argparse.Namespace,
    image_path: Path,
    realsense_timestamp: str,
    poses_path: Path,
    poses_timestamp: str,
    task: str,
) -> dict[str, Any]:
    return {
        "module": "run_validation_live2",
        "timestamp": datetime.now().isoformat(),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "image_input": {
            "mode": "latest_realsense_rgb_frame",
            "image_path": str(image_path),
            "shared_data_root": str(Path(args.shared_data_root).resolve()),
            "realsense_timestamp": realsense_timestamp,
            "frame_id": "000000",
        },
        "poses_input": {
            "mode": "latest_outputs_by_object_poses",
            "poses_path": str(poses_path),
            "poses_timestamp": poses_timestamp,
        },
        "task": task,
        "scene_description": {
            "prompt_version": args.scene_v,
            "model": args.scene_model,
            "source": "latest_run_validation_live1_output",
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
        "live_frame_refresh": {
            "mode": "wait_for_new_realsense_rgb_timestamp_after_precondition",
            "timeout_seconds": args.new_frame_timeout,
            "poll_seconds": args.new_frame_poll_seconds,
            "file_timeout_seconds": args.new_frame_file_timeout,
            "file_poll_seconds": args.new_frame_file_poll_seconds,
        },
        "max_replans": 0,
        "replanning_policy": "stop_on_first_replan_request",
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_sampling_args(args)

    settings = load_settings()
    realsense_timestamp = resolve_realsense_timestamp(
        shared_data_root=args.shared_data_root,
        explicit_timestamp=args.realsense_timestamp,
    )
    image_path = resolve_realsense_image_path(
        shared_data_root=args.shared_data_root,
        realsense_timestamp=realsense_timestamp,
    )
    poses_by_image_path, poses_timestamp = resolve_live_poses_path(
        shared_data_root=args.shared_data_root,
        explicit_timestamp=args.poses_timestamp,
        explicit_path=args.poses_path,
    )
    poses_by_image = load_poses_by_image_map(poses_by_image_path)
    task = resolve_task(args.task)

    scene_response_path, loop_timestamp, cycle_name = find_latest_scene_description_response(
        settings=settings,
        scenario_name=args.scenario,
        scene_version=args.scene_v,
        scene_model=args.scene_model,
    )
    scene_description_artifact = build_scene_artifact_from_path(scene_response_path)

    cycle_idx = 1
    cycle_timestamp = make_experiment_timestamp()
    scenario_context = {
        "scenario_name": args.scenario,
        "task": task,
        "image": image_path.name,
        "image_path_abs": str(image_path),
        "scenario_dir_abs": None,
    }
    pipeline_config = build_live2_config(
        args=args,
        image_path=image_path,
        realsense_timestamp=realsense_timestamp,
        poses_path=poses_by_image_path,
        poses_timestamp=poses_timestamp,
        task=task,
    )
    realsense_rgb_dir = (
        Path(args.shared_data_root)
        / "realsense"
        / "rgb"
        / realsense_timestamp
    ).resolve()
    current_image_path = image_path
    current_realsense_timestamp = realsense_timestamp

    full_summary: dict[str, Any] = {
        "module": "full_pipeline_summary",
        "scenario_name": args.scenario,
        "loop_timestamp": loop_timestamp,
        "timestamp": datetime.now().isoformat(),
        "initial_image_path": str(image_path),
        "scenario_image_dir": str(realsense_rgb_dir),
        "poses_by_image_path": str(poses_by_image_path),
        "config": pipeline_config,
        "replans_done": 0,
        "task_completed": False,
        "final_image_path": None,
        "final_realsense_timestamp": None,
        "cycles": [],
    }

    cycle_record: dict[str, Any] = {
        "cycle_name": cycle_name,
        "cycle_index": cycle_idx,
        "cycle_timestamp": cycle_timestamp,
        "start_image_path": str(image_path),
        "start_image_name": image_path.name,
        "scene_description": scene_description_artifact,
        "scene_description_full": None,
        "vlm_planning": None,
        "simultaneous_actions": None,
        "stages": [],
        "outcome": None,
        "end_image_path": None,
        "end_image_name": None,
    }

    print("\n======================================================")
    print("LIVE2 | CONTINUE FROM LATEST SCENE PERCEPTION")
    print(f"Scenario:                    {args.scenario}")
    print(f"Image:                       {image_path}")
    print(f"Realsense timestamp:         {realsense_timestamp}")
    print(f"Poses file:                  {poses_by_image_path}")
    print(f"Poses timestamp:             {poses_timestamp}")
    print(f"Loaded scene response:       {scene_response_path}")
    print(f"Task:                        {task}")
    print(f"Loop timestamp from live1:   {loop_timestamp}")
    print(f"Cycle from live1:            {cycle_name}")
    print("Replanning policy:           stop on first validation failure")
    print("======================================================")

    try:
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
            image_path=str(image_path),
            poses_by_image=poses_by_image,
            safety_threshold=args.grounding_safety_threshold,
            include_debug_mapping=args.grounding_debug_mapping,
            direct_name_matching=True,
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
            pre_condition = stage["Precondition"]
            post_condition = stage["Postcondition"]

            stage_record: dict[str, Any] = {
                "stage_id": stage_id,
                "stage_name": f"stage_{stage_id:03d}",
                "precondition": pre_condition,
                "postcondition": post_condition,
                "pre_realsense_timestamp": current_realsense_timestamp,
                "post_realsense_timestamp": None,
                "pre_image_path": str(current_image_path),
                "pre_image_name": current_image_path.name,
                "post_image_path": None,
                "post_image_name": None,
                "pre_validation": None,
                "post_validation": None,
                "pre_validation_time_seconds": None,
                "post_validation_time_seconds": None,
                "new_frame_wait_time_seconds": None,
                "validator_paths": {
                    "pre": None,
                    "post": None,
                },
                "next_image_path": None,
                "next_image_name": None,
            }

            print(f"\n[LOOP] Stage {stage_id} PRE")
            print(f"[LOOP] PRE image:      {current_image_path}")
            print(f"[LOOP] PRE condition:  {pre_condition}")
            print_pose_dict_for_image(
                poses_by_image=poses_by_image,
                image_path=str(current_image_path),
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
                image_path=str(current_image_path),
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
                print(f"[LOOP] Precondition failed at stage {stage_id}. Stopping instead of replanning.")
                cycle_record["stages"].append(stage_record)
                cycle_record["outcome"] = f"stopped_replan_on_pre_stage_{stage_id}"
                all_stages_succeeded = False
                break

            print(f"\n[LOOP] Stage {stage_id} ACTION")
            print(
                "[LOOP] Execute the robot action now, then save a new frame from "
                "the other docker."
            )
            new_frame_wait_start = time.perf_counter()
            post_image_path, post_realsense_timestamp = wait_for_new_realsense_frame(
                shared_data_root=args.shared_data_root,
                previous_timestamp=current_realsense_timestamp,
                timeout_seconds=args.new_frame_timeout,
                poll_seconds=args.new_frame_poll_seconds,
                file_timeout_seconds=args.new_frame_file_timeout,
                file_poll_seconds=args.new_frame_file_poll_seconds,
            )
            stage_record["new_frame_wait_time_seconds"] = (
                time.perf_counter() - new_frame_wait_start
            )
            stage_record["post_realsense_timestamp"] = post_realsense_timestamp
            stage_record["post_image_path"] = str(post_image_path)
            stage_record["post_image_name"] = post_image_path.name
            stage_record["next_image_path"] = str(post_image_path)
            stage_record["next_image_name"] = post_image_path.name

            print(f"[LOOP] NEW frame:      {post_image_path}")
            print(f"[LOOP] NEW timestamp:  {post_realsense_timestamp}")

            print(f"\n[LOOP] Stage {stage_id} POST")
            print(f"[LOOP] POST image:     {post_image_path}")
            print(f"[LOOP] POST condition: {post_condition}")
            print_pose_dict_for_image(
                poses_by_image=poses_by_image,
                image_path=str(post_image_path),
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
                image_path=str(post_image_path),
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
                print(f"[LOOP] Postcondition failed at stage {stage_id}. Stopping instead of replanning.")
                cycle_record["outcome"] = f"stopped_replan_on_post_stage_{stage_id}"
                all_stages_succeeded = False
                break

            current_image_path = post_image_path
            current_realsense_timestamp = post_realsense_timestamp

        if all_stages_succeeded:
            cycle_record["outcome"] = "task_completed"
            full_summary["task_completed"] = True
            full_summary["final_image_path"] = str(current_image_path)
            full_summary["final_realsense_timestamp"] = current_realsense_timestamp

    except Exception as exc:
        cycle_record["outcome"] = f"cycle_error: {exc}"
        full_summary["error"] = str(exc)

    cycle_record["end_image_path"] = str(current_image_path)
    cycle_record["end_image_name"] = current_image_path.name
    full_summary["cycles"].append(cycle_record)

    cycle_summary = build_cycle_summary(full_summary, cycle_record)
    cycle_summary_path = save_cycle_summary(
        settings=settings,
        scenario_name=args.scenario,
        loop_timestamp=loop_timestamp,
        cycle_name=cycle_name,
        cycle_summary=cycle_summary,
    )

    summary_paths = save_validation_loop_artifacts(
        settings=settings,
        scenario_name=args.scenario,
        loop_timestamp=loop_timestamp,
        run_info=build_run_info(full_summary),
        loop_summary=build_loop_summary(full_summary),
        scene_description_summary=build_scene_description_summary(full_summary),
        vlm_planning_summary=build_vlm_planning_summary(full_summary),
        simultaneous_actions_summary=build_simultaneous_actions_summary(full_summary),
        validator_summary=build_validator_summary(full_summary),
        full_pipeline_summary=build_full_pipeline_summary(full_summary),
    )

    print("\n======================================================")
    print("LIVE2 COMPLETED")
    print(f"Task completed:          {full_summary['task_completed']}")
    print(f"Outcome:                 {cycle_record['outcome']}")
    print(f"Cycle summary saved:     {cycle_summary_path}")
    print(f"Full summary saved:      {summary_paths['full_pipeline_summary']}")
    print("======================================================")


if __name__ == "__main__":
    main()