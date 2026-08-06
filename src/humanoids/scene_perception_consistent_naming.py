"""
scene_perception_consistent_naming.py — fixed variant of
`execute_scene_perception_for_state` (defined in
run_validation_loop_fixed_on_demand.py) that prevents a specific, observed
failure mode of the recovery engine.

THE PROBLEM THIS FILE ADDRESSES
--------------------------------
The recovery engine decides whether a failed stage is still applicable
(`stage_still_applicable`) by checking whether the action's `Target` name
(e.g. "box_2") is still mentioned in a freshly generated scene description of
the post-execution image. That fresh scene description is produced by a
standalone VLM call whose prompt explicitly says:

    "Analyze the current scene again ... Treat this as a fresh observation."

Because the call has no memory of the object names already established
earlier in the same episode, it is free to invent a different but equally
valid description of the same physical objects (e.g. "red_box" instead of
"box_2") on any given call. When that happens, the target-presence check
concludes the target object has disappeared from the scene -- even though
nothing physically changed -- which forces `stage_still_applicable = False`
and `replan_required = True`, silently overriding repeat/modify regardless of
how good the rest of the evidence is. This was confirmed twice in practice by
comparing the two independent scene graphs generated for the same recovery
step: one used "box_1"/"box_2", the other used "green_box"/"red_box" for the
exact same image.

THE FIX
-------
This module does not touch the matching logic at all. Instead it fixes the
problem at its source: it passes the object identities already established
earlier in the same cycle (name + category + color, taken from the cycle's
own `scene_description` step) into the scene-perception prompt, and instructs
the model to reuse those exact names for the same physical objects instead of
inventing new ones.

It does not modify or duplicate any existing pipeline file beyond copying the
body of `execute_scene_perception_for_state` verbatim (all its dependencies
are imported unchanged from where they already live).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.llm_client import call_llm_completion
from src.scene_enrichment_simulation import enrich_scene
from src.utils import load_base_prompt, try_parse_json

from src.humanoids.run_validation_loop_fixed_on_demand import (
    ensure_dir,
    write_text,
    save_json_file,
    get_pose_dict_for_image,
    write_temp_pose_file,
    make_scenario_context,
)


def _extract_established_identities(
    reference_scene_description: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Pull {name, category, color} triples out of an already-generated
    scene_description, so they can be reused as a naming reference."""
    if not isinstance(reference_scene_description, dict):
        return []

    payload = reference_scene_description.get(
        "scene_description", reference_scene_description
    )
    if not isinstance(payload, dict):
        return []

    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        return []

    identities: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not name:
            continue
        identities.append(
            {
                "name": name,
                "category": obj.get("category"),
                "color": obj.get("color"),
            }
        )
    return identities


def _render_identity_hint(identities: list[dict[str, Any]]) -> str:
    if not identities:
        return ""

    lines = []
    for item in identities:
        descriptor_bits = [
            str(value)
            for value in (item.get("category"), item.get("color"))
            if value
        ]
        descriptor = f" ({', '.join(descriptor_bits)})" if descriptor_bits else ""
        lines.append(f'- "{item["name"]}"{descriptor}')

    return (
        "\n\nThe following object identities are already established for "
        "this episode:\n"
        + "\n".join(lines)
        + "\nReuse exactly these names for the same physical objects. Do "
        "not invent a new descriptive name (e.g. a color-based name) for an "
        "object that already has an established name above. Only assign a "
        "new name if you see an object that is not in this list."
    )


def execute_scene_perception_for_state_consistent_naming(
    *,
    settings,
    scenario_name: str,
    scenario_data: dict[str, Any],
    image_path: str,
    poses_by_image: dict[str, dict[str, list[float]]],
    scene_version: str,
    scene_model: str,
    temperature: float,
    top_p: float,
    safety_threshold: float,
    include_debug_mapping: bool,
    output_dir: Path,
    purpose: str,
    reference_scene_description: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Reconstruct a structured world state from one image, instructed to reuse
    object identities already established earlier in the same cycle.

    This is a naming-consistent drop-in replacement for
    `execute_scene_perception_for_state`. Everything else about it (enrichment,
    artifact layout, return shape) is unchanged.
    """
    ensure_dir(output_dir)

    identities = _extract_established_identities(reference_scene_description)
    identity_hint = _render_identity_hint(identities)

    base_prompt = load_base_prompt(settings, "scene_description", scene_version)
    result = call_llm_completion(
        settings=settings,
        model_name=scene_model,
        system_prompt=base_prompt,
        user_text=(
            "Analyze the current scene again and return the structured JSON "
            "output. Treat this as a fresh observation of object positions "
            "and states, but keep object identities consistent with what "
            "was already established." + identity_hint
        ),
        image_path=image_path,
        temperature=temperature,
        top_p=top_p,
    )

    parse_ok, scene_description = try_parse_json(result["raw_response"])
    if not parse_ok:
        raise ValueError(
            f"[scene_perception:{purpose}] Model response could not be parsed "
            f"as valid JSON.\n\nRaw response:\n{result['raw_response']}"
        )

    pose_dict = get_pose_dict_for_image(poses_by_image, image_path)
    temp_pose_file = write_temp_pose_file(pose_dict)
    try:
        enrichment_start = time.perf_counter()
        scene_graph = enrich_scene(
            input_data=scene_description,
            safety_threshold=safety_threshold,
            pose_source="static",
            pose_file=temp_pose_file,
            include_debug_mapping=include_debug_mapping,
        )
        enrichment_seconds = time.perf_counter() - enrichment_start
    finally:
        temp_path = Path(temp_pose_file)
        if temp_path.exists():
            temp_path.unlink()

    prompt_path = output_dir / "prompt.txt"
    scene_description_path = output_dir / "scene_description.json"
    scene_graph_path = output_dir / "scene_description_full.json"
    run_info_path = output_dir / "run_info.json"

    write_text(prompt_path, base_prompt + identity_hint)
    save_json_file(scene_description_path, scene_description)
    save_json_file(scene_graph_path, scene_graph)
    save_json_file(
        run_info_path,
        {
            "module": "scene_perception_consistent_naming",
            "purpose": purpose,
            "scenario_name": scenario_name,
            "image_path": str(Path(image_path).resolve()),
            "image_name": Path(image_path).name,
            "pose_key": Path(image_path).name,
            "scene_version": scene_version,
            "scene_model": result["model_name"],
            "deployment_name": result["deployment_name"],
            "vlm_execution_time_seconds": result["execution_time_seconds"],
            "enrichment_execution_time_seconds": enrichment_seconds,
            "reference_identities_used": identities,
            "sampling_config": {
                "temperature": temperature,
                "top_p": top_p,
            },
            "scenario_context": make_scenario_context(
                scenario_data=scenario_data,
                image_path=image_path,
            ),
            "created_at": datetime.now().isoformat(),
        },
    )

    print(
        f"[OK][scene_perception_consistent_naming:{purpose}] Updated scene "
        f"graph saved to: {scene_graph_path}"
    )
    return {
        "scene_description": scene_description,
        "scene_graph": scene_graph,
        "paths": {
            "prompt": str(prompt_path),
            "scene_description": str(scene_description_path),
            "scene_graph": str(scene_graph_path),
            "run_info": str(run_info_path),
        },
        "model_name": result["model_name"],
        "execution_time_seconds": (
            result["execution_time_seconds"] + enrichment_seconds
        ),
    }
