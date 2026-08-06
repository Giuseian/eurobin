"""
scene_transition_analysis_robust.py — defense-in-depth fixed variant of
`analyze_scene_transition` (defined in scene_transition_analysis.py).

THE PROBLEM THIS FILE ADDRESSES
--------------------------------
`analyze_scene_transition` decides whether a failed stage is "still
applicable" by checking whether the action's `Target` name (e.g. "box_2") is
still present, as a literal substring, in a freshly generated scene
description of the post-execution image. Two independent scene-perception
calls on the same episode are not guaranteed to name the same physical
object identically (one may say "box_2", another "red_box" for the exact
same box) -- see scene_perception_consistent_naming.py for the fix at the
source of that inconsistency.

This file is a second, independent layer of protection: even if the source
call is fixed (or on the rare occasion it still drifts), this variant does
not give up on a target the moment its literal name is missing. It also
checks whether an object with the same color/category as the target (learned
from an authoritative reference scene description, established earlier in
the same cycle) is still present. Only if neither the name nor a matching
color/category fingerprint can be found is the target treated as missing.

It does not modify scene_transition_analysis.py. All the parts of the
original logic that do not need to change (flattening, target extraction,
relevance filtering, goal-progress inference, the conservative
scene-to-action modification bridge) are imported and reused unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.humanoids.scene_transition_analysis import (
    _derive_scene_supported_modifications,
    _extract_target_names,
    _flatten_scene,
    _infer_goal_progress,
    _is_target_relevant,
    _normalize_token,
)


def _build_identity_fingerprints(
    reference_scene_description: dict[str, Any] | None,
) -> dict[str, dict[str, str | None]]:
    """Map normalized target name -> {color, category} learned from an
    authoritative scene_description generated earlier in the same cycle."""
    if not isinstance(reference_scene_description, dict):
        return {}

    payload = reference_scene_description.get(
        "scene_description", reference_scene_description
    )
    if not isinstance(payload, dict):
        return {}

    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        return {}

    fingerprints: dict[str, dict[str, str | None]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not name:
            continue
        color = obj.get("color")
        category = obj.get("category")
        fingerprints[_normalize_token(name)] = {
            "color": _normalize_token(color) if color else None,
            "category": _normalize_token(category) if category else None,
        }
    return fingerprints


def _presence_by_target_robust(
    flat_scene: dict[str, Any],
    targets: list[str],
    fingerprints: dict[str, dict[str, str | None]],
) -> dict[str, bool]:
    searchable_facts = [
        _normalize_token(f"{path} {value}")
        for path, value in flat_scene.items()
    ]

    presence: dict[str, bool] = {}
    for target in targets:
        norm_target = _normalize_token(target)
        name_hit = any(norm_target in fact for fact in searchable_facts)
        if name_hit:
            presence[target] = True
            continue

        fingerprint = fingerprints.get(norm_target)
        color = fingerprint.get("color") if fingerprint else None
        category = fingerprint.get("category") if fingerprint else None

        color_hit = bool(color) and any(color in fact for fact in searchable_facts)
        category_hit = bool(category) and any(
            category in fact for fact in searchable_facts
        )
        # Color is the more discriminating signal (two boxes of the same
        # category but different colors are common); category alone is only
        # trusted as a fallback when no color fingerprint is available.
        presence[target] = color_hit or (color is None and category_hit)

    return presence


def analyze_scene_transition_robust(
    *,
    scene_graph_before: dict[str, Any],
    scene_graph_after: dict[str, Any],
    failed_stage: dict[str, Any],
    actions: list[dict[str, Any]],
    before_goal_validation: dict[str, Any] | None = None,
    after_goal_validation: dict[str, Any] | None = None,
    reference_scene_description: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before_flat = _flatten_scene(scene_graph_before or {})
    after_flat = _flatten_scene(scene_graph_after or {})

    targets = _extract_target_names(
        failed_stage=failed_stage,
        actions=actions,
    )

    added_facts: list[dict[str, Any]] = []
    removed_facts: list[dict[str, Any]] = []
    changed_properties: list[dict[str, Any]] = []

    before_paths = set(before_flat)
    after_paths = set(after_flat)

    for path in sorted(after_paths - before_paths):
        added_facts.append(
            {"path": path, "value": deepcopy(after_flat[path])}
        )

    for path in sorted(before_paths - after_paths):
        removed_facts.append(
            {"path": path, "value": deepcopy(before_flat[path])}
        )

    for path in sorted(before_paths & after_paths):
        old_value = before_flat[path]
        new_value = after_flat[path]
        if old_value != new_value:
            changed_properties.append(
                {
                    "path": path,
                    "before": deepcopy(old_value),
                    "after": deepcopy(new_value),
                }
            )

    relevant_changes: list[dict[str, Any]] = []
    for item in changed_properties:
        if _is_target_relevant(
            path=item["path"],
            old_value=item["before"],
            new_value=item["after"],
            targets=targets,
        ):
            relevant_changes.append(
                {"change_type": "property_changed", **deepcopy(item)}
            )

    for item in added_facts:
        if _is_target_relevant(
            path=item["path"],
            old_value=None,
            new_value=item["value"],
            targets=targets,
        ):
            relevant_changes.append(
                {"change_type": "fact_added", **deepcopy(item)}
            )

    for item in removed_facts:
        if _is_target_relevant(
            path=item["path"],
            old_value=item["value"],
            new_value=None,
            targets=targets,
        ):
            relevant_changes.append(
                {"change_type": "fact_removed", **deepcopy(item)}
            )

    goal_progress, condition_transitions = _infer_goal_progress(
        before_goal_validation=before_goal_validation,
        after_goal_validation=after_goal_validation,
    )

    fingerprints = _build_identity_fingerprints(reference_scene_description)
    before_presence = _presence_by_target_robust(before_flat, targets, fingerprints)
    after_presence = _presence_by_target_robust(after_flat, targets, fingerprints)
    targets_missing = [
        target
        for target in targets
        if before_presence.get(target) and not after_presence.get(target)
    ]

    stage_still_applicable = not targets_missing
    modifications = _derive_scene_supported_modifications(
        relevant_changes=relevant_changes,
        actions=actions,
    )

    return {
        "targets": targets,
        "observable_change": bool(
            added_facts or removed_facts or changed_properties
        ),
        "target_state_changed": bool(relevant_changes),
        "goal_progress": goal_progress,
        "condition_transitions": condition_transitions,
        "stage_still_applicable": stage_still_applicable,
        "targets_missing": targets_missing,
        "target_presence_before": before_presence,
        "target_presence_after": after_presence,
        "added_facts": added_facts,
        "removed_facts": removed_facts,
        "changed_properties": changed_properties,
        "relevant_changes": relevant_changes,
        "supported_symbolic_modifications": modifications,
        "analysis_confidence": (
            "high"
            if condition_transitions
            else "medium"
            if targets and relevant_changes
            else "low"
        ),
        "identity_fingerprints_used": fingerprints,
    }
