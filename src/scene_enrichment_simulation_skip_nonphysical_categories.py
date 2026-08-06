"""
Wrapper around `src.scene_enrichment_simulation.enrich_scene`.

The original `enrich_scene` requires a strict one-to-one mapping between every
VLM-detected scene object and a Gazebo entity, and raises a hard `ValueError`
("No Gazebo candidates found ...") when a detected object has no physical
counterpart in the simulator catalog. This happens for VLM categories that
describe a marking/annotation on another object rather than an independently
simulated entity (e.g. "label", "sticker", "tag") -- there is no Gazebo model
for those, so matching can never succeed and the whole validation cycle was
aborted.

This module keeps `scene_enrichment_simulation.py` untouched and instead
filters out scene objects (and any spatial relationship referencing them)
whose category is in `NON_PHYSICAL_CATEGORIES` before delegating to the
original matching/accessibility pipeline. Skipped objects are reported back
in the output instead of silently disappearing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.scene_enrichment_simulation import (
    GZ_POSE_TOPIC,
    POSE_SOURCE_GAZEBO,
    build_objects_with_geometry,
    compute_accessibility,
    normalize_text,
)

NON_PHYSICAL_CATEGORIES = {
    "label", "labels",
    "sticker", "stickers",
    "tag", "tags",
    "marking", "markings",
    "text",
    "logo", "logos",
    "decal", "decals",
}


def _is_non_physical_category(category: Any) -> bool:
    if category is None:
        return False
    return normalize_text(str(category)) in NON_PHYSICAL_CATEGORIES


def enrich_scene(
    input_data: Dict[str, Any],
    safety_threshold: float = 0.21,
    pose_source: str = POSE_SOURCE_GAZEBO,
    pose_file: Optional[str] = None,
    topic: str = GZ_POSE_TOPIC,
    timeout_sec: float = 3.0,
    include_debug_mapping: bool = False,
) -> Dict[str, Any]:
    if "scene_description" not in input_data:
        raise ValueError("Missing required field: 'scene_description'")

    scene_description = input_data["scene_description"]
    all_scene_objects = scene_description.get("objects", [])
    all_spatial_relationships = scene_description.get("spatial_relationships", [])

    skipped_objects = [
        obj for obj in all_scene_objects if _is_non_physical_category(obj.get("category"))
    ]
    skipped_names = {obj["name"] for obj in skipped_objects}

    if skipped_objects:
        print(
            "[WARN][scene_enrichment_skip_nonphysical] Skipping VLM objects with "
            f"no Gazebo representation (non-physical category): {sorted(skipped_names)}"
        )

    scene_objects = [obj for obj in all_scene_objects if obj["name"] not in skipped_names]
    spatial_relationships = [
        rel
        for rel in all_spatial_relationships
        if rel.get("subject") not in skipped_names and rel.get("object") not in skipped_names
    ]

    objects_with_geometry, vlm_to_gazebo, matching_warnings = build_objects_with_geometry(
        scene_objects=scene_objects,
        spatial_relationships=spatial_relationships,
        pose_source=pose_source,
        pose_file=pose_file,
        topic=topic,
        timeout_sec=timeout_sec,
    )

    computed_info = compute_accessibility(
        objects_with_geometry=objects_with_geometry,
        spatial_relationships=spatial_relationships,
        safety_threshold=safety_threshold,
    )

    enriched_objects: List[Dict[str, Any]] = []

    for obj in scene_objects:
        name = obj["name"]
        enriched_obj = dict(obj)
        enriched_obj["sides"] = computed_info[name]["sides"]
        enriched_objects.append(enriched_obj)

    output: Dict[str, Any] = {
        "scene_description": {
            "objects": enriched_objects,
            "end_effectors": scene_description.get("end_effectors", []),
            "spatial_relationships": spatial_relationships,
        }
    }

    if skipped_objects:
        output["scene_description"]["skipped_non_physical_objects"] = skipped_objects

    if include_debug_mapping:
        output["_debug"] = {
            "pose_source": pose_source,
            "pose_file": pose_file,
            "vlm_to_gazebo_mapping": vlm_to_gazebo,
            "matching_warnings": matching_warnings,
            "skipped_non_physical_objects": sorted(skipped_names),
        }

    return output
