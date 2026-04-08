from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from settings import load_settings
from utils import get_output_dir, read_json, write_json


def extract_scene_object_list(scene_description_data: dict[str, Any]) -> dict[str, Any]:
    """
    Converte l'output di scene_description in una scene_object_list
    contenente solo:
    - name
    - category
    - color
    """
    if "scene_description" not in scene_description_data:
        raise ValueError("Missing top-level key 'scene_description' in scene description output.")

    scene_block = scene_description_data["scene_description"]
    if not isinstance(scene_block, dict):
        raise ValueError("'scene_description' must be a JSON object.")

    objects = scene_block.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Missing or invalid 'objects' list in scene description output.")

    filtered_objects: list[dict[str, str]] = []

    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"Object at index {idx} is not a JSON object.")

        name = obj.get("name")
        category = obj.get("category")
        color = obj.get("color")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Object at index {idx} has invalid or missing 'name'.")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Object '{name}' has invalid or missing 'category'.")
        if not isinstance(color, str) or not color.strip():
            raise ValueError(f"Object '{name}' has invalid or missing 'color'.")

        filtered_objects.append(
            {
                "name": name,
                "category": category,
                "color": color,
            }
        )

    return {"objects": filtered_objects}


def build_scene_object_list(
    scene_description_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Legge response_parsed.json di scene_description e salva scene_object_list.json.
    """
    scene_description_data = read_json(scene_description_path)
    scene_object_list = extract_scene_object_list(scene_description_data)
    write_json(output_path, scene_object_list)
    return Path(output_path)


def build_scene_object_list_from_run(
    scenario: str,
    version: str,
    experiment_timestamp: str,
    model: str,
    run_name: str,
) -> Path:
    """
    Risolve automaticamente i path a partire da
    scenario/version/timestamp/model/run.
    """
    settings = load_settings()

    output_dir = get_output_dir(
        settings=settings,
        module_name="scene_description",
        scenario_name=scenario,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model,
        run_name=run_name,
    )

    scene_description_path = output_dir / "response_parsed.json"
    scene_object_list_path = output_dir / "scene_object_list.json"

    if not scene_description_path.exists():
        raise FileNotFoundError(
            f"Scene description parsed output not found: {scene_description_path}"
        )

    return build_scene_object_list(
        scene_description_path=scene_description_path,
        output_path=scene_object_list_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build scene_object_list.json from a scene_description response_parsed.json."
    )

    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--run", type=str, default="run_001")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    out_path = build_scene_object_list_from_run(
        scenario=args.scenario,
        version=args.version,
        experiment_timestamp=args.timestamp,
        model=args.model,
        run_name=args.run,
    )

    print(f"[OK] scene_object_list saved to: {out_path}")


if __name__ == "__main__":
    main()