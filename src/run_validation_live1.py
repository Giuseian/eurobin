"""Run only the first live validation step: scene perception.

This script reads the latest /workspace/shared_data/realsense/rgb/<timestamp>/000000.png
frame, calls the scene_description model, stores the usual scene_description
artifacts, and writes short SAM3 prompts for every perceived object.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from settings import load_settings
from utils import make_cycle_name, make_experiment_timestamp, write_json
from run_validation_live import (
    IMAGE_EXTENSIONS,
    SUPPORTED_MODELS,
    execute_scene_description_step,
    resolve_image_data_root,
    resolve_scenario_image_dir,
    validate_sampling_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run only scene perception on the latest Realsense RGB 000000.png frame."
    )
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--image-data-root", type=str, default=None)
    parser.add_argument("--shared-data-root", type=str, default="/workspace/shared_data")
    parser.add_argument(
        "--realsense-timestamp",
        type=str,
        default=None,
        help=(
            "Timestamp folder to use under shared_data/realsense/objects. "
            "If omitted, the latest folder under shared_data/realsense/rgb is used."
        ),
    )
    parser.add_argument("--scene-v", type=str, required=True)
    parser.add_argument("--scene-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser


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


def extract_scene_objects(scene_description_output: Any) -> list[dict[str, str]]:
    if not isinstance(scene_description_output, dict):
        raise ValueError("scene_description output must be a JSON object.")

    scene_description = scene_description_output.get("scene_description", scene_description_output)
    if not isinstance(scene_description, dict):
        raise ValueError("scene_description payload must be a JSON object.")

    objects = scene_description.get("objects")
    if not isinstance(objects, list):
        raise ValueError("scene_description must contain an 'objects' list.")

    scene_objects: list[dict[str, str]] = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"Object at index {index} is not a JSON object.")
        name = obj.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Object at index {index} has invalid or missing name.")
        category = obj.get("category")
        color = obj.get("color")
        scene_objects.append(
            {
                "name": name.strip(),
                "category": category.strip() if isinstance(category, str) else "",
                "color": color.strip() if isinstance(color, str) else "",
            }
        )

    return scene_objects


def extract_scene_object_names(scene_description_output: Any) -> list[str]:
    return [obj["name"] for obj in extract_scene_objects(scene_description_output)]


def resolve_realsense_timestamp(shared_data_root: str | Path, explicit_timestamp: str | None) -> str:
    if explicit_timestamp is not None and explicit_timestamp.strip():
        return explicit_timestamp.strip()

    rgb_root = Path(shared_data_root) / "realsense" / "rgb"
    if not rgb_root.exists():
        raise FileNotFoundError(f"Realsense RGB root not found: {rgb_root}")

    timestamp_dirs = sorted(
        [path for path in rgb_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    if not timestamp_dirs:
        raise RuntimeError(f"No Realsense timestamp folders found in: {rgb_root}")

    return timestamp_dirs[-1].name


def make_sam3_prompt(scene_object: dict[str, str]) -> str:
    color = scene_object.get("color", "").strip().lower()
    category = scene_object.get("category", "").strip().lower()
    words = [word for word in [color, category] if word and word not in {"unknown", "none"}]
    if words:
        return " ".join(words)
    return scene_object["name"].strip().lower()


def write_sam3_prompt_file(
    shared_data_root: str | Path,
    realsense_timestamp: str,
    scene_objects: list[dict[str, str]],
) -> Path:
    prompts = [
        {
            "object_id": scene_object["name"],
            "prompt": make_sam3_prompt(scene_object),
            "category": scene_object.get("category", ""),
            "color": scene_object.get("color", ""),
        }
        for scene_object in scene_objects
    ]
    output_path = (
        Path(shared_data_root)
        / "realsense"
        / "objects"
        / realsense_timestamp
        / "sam3_prompts.json"
    )
    write_json(output_path, prompts)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    validate_sampling_args(args)

    settings = load_settings()
    image_data_root = resolve_image_data_root(settings, args.image_data_root)
    scenario_image_dir = resolve_scenario_image_dir(image_data_root, args.scenario)
    realsense_timestamp = resolve_realsense_timestamp(
        shared_data_root=args.shared_data_root,
        explicit_timestamp=args.realsense_timestamp,
    )
    image_path = resolve_realsense_image_path(
        shared_data_root=args.shared_data_root,
        realsense_timestamp=realsense_timestamp,
    )

    loop_timestamp = make_experiment_timestamp()
    cycle_name = make_cycle_name(1)
    cycle_timestamp = make_experiment_timestamp()

    scenario_context = {
        "scenario_name": args.scenario,
        "task": None,
        "image": image_path.name,
        "image_path_abs": str(image_path),
        "scenario_dir_abs": str(scenario_image_dir),
    }
    pipeline_config = {
        "module": "run_validation_live1",
        "timestamp": datetime.now().isoformat(),
        "image_input": {
            "mode": "latest_realsense_rgb_frame",
            "image_path": str(image_path),
            "shared_data_root": str(Path(args.shared_data_root).resolve()),
            "realsense_timestamp": realsense_timestamp,
            "frame_id": "000000",
        },
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "scene_description": {
            "prompt_version": args.scene_v,
            "model": args.scene_model,
        },
    }

    print("\n======================================================")
    print("LIVE1 | SCENE PERCEPTION ONLY")
    print(f"Scenario:        {args.scenario}")
    print(f"Image:           {image_path}")
    print(f"Realsense ts:    {realsense_timestamp}")
    print(f"Loop timestamp:  {loop_timestamp}")
    print("======================================================")

    scene_artifact = execute_scene_description_step(
        settings=settings,
        scenario_name=args.scenario,
        scenario_context=scenario_context,
        version=args.scene_v,
        model_name=args.scene_model,
        loop_timestamp=loop_timestamp,
        cycle_name=cycle_name,
        cycle_idx=1,
        cycle_timestamp=cycle_timestamp,
        pipeline_config=pipeline_config,
        image_path=str(image_path),
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n[scene_description] Parsed JSON:")
    print(json.dumps(scene_artifact["output"], indent=2, ensure_ascii=False))

    scene_objects = extract_scene_objects(scene_artifact["output"])
    object_names = [scene_object["name"] for scene_object in scene_objects]
    sam3_prompt_path = write_sam3_prompt_file(
        shared_data_root=args.shared_data_root,
        realsense_timestamp=realsense_timestamp,
        scene_objects=scene_objects,
    )

    print("\n[OK][live1] Scene perception completed.")
    print(f"[OK][live1] Objects found:       {', '.join(object_names) if object_names else '(none)'}")
    print(f"[OK][live1] SAM3 prompts saved:  {sam3_prompt_path}")
    print(f"[OK][live1] Response parsed:     {scene_artifact['paths']['response_parsed']}")
    print(f"[OK][live1] Scene object list:   {scene_artifact['paths']['scene_object_list']}")


if __name__ == "__main__":
    main()
