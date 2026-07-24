from __future__ import annotations

from copy import deepcopy
from typing import Any
import json
import re


IGNORED_KEYS = {
    "created_at",
    "updated_at",
    "timestamp",
    "frame_id",
    "image_path",
    "source_image",
    "run_info",
    "_debug",
}

IDENTITY_KEYS = (
    "id",
    "object_id",
    "name",
    "label",
    "entity",
    "object",
    "region_id",
)


def _normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", " ", str(value).lower()).strip()


def _item_identity(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    for key in IDENTITY_KEYS:
        value = item.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return f"{key}={_normalize_token(value)}"
    return None


def _flatten_scene(
    value: Any,
    *,
    path: str = "scene",
) -> dict[str, Any]:
    """
    Convert a nested scene graph into stable path -> scalar facts.

    Lists of dictionaries are keyed by semantic identity when possible, so
    changes in list ordering do not appear as world-state changes.
    """
    result: dict[str, Any] = {}

    if isinstance(value, dict):
        for key in sorted(value):
            if key in IGNORED_KEYS:
                continue
            result.update(
                _flatten_scene(
                    value[key],
                    path=f"{path}.{key}",
                )
            )
        return result

    if isinstance(value, list):
        identities = [_item_identity(item) for item in value]
        identities_are_usable = (
            bool(value)
            and all(identity is not None for identity in identities)
            and len(set(identities)) == len(identities)
        )

        if identities_are_usable:
            ordered = sorted(
                zip(identities, value),
                key=lambda pair: str(pair[0]),
            )
            for identity, child in ordered:
                result.update(
                    _flatten_scene(
                        child,
                        path=f"{path}[{identity}]",
                    )
                )
        else:
            for index, child in enumerate(value):
                result.update(
                    _flatten_scene(
                        child,
                        path=f"{path}[{index}]",
                    )
                )
        return result

    result[path] = deepcopy(value)
    return result


def _extract_target_names(
    *,
    failed_stage: dict[str, Any],
    actions: list[dict[str, Any]],
) -> list[str]:
    candidate_keys = {
        "target",
        "object",
        "object_id",
        "target_object",
        "Target",
        "Object",
        "Target object",
        "Object name",
    }

    targets: list[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        for key, value in action.items():
            if key not in candidate_keys or value is None:
                continue
            if isinstance(value, list):
                targets.extend(str(item) for item in value)
            else:
                targets.append(str(value))

    # Fall back to explicit target-like fields in the stage, but never parse
    # arbitrary prose to invent object identifiers.
    for key in candidate_keys:
        value = failed_stage.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            targets.extend(str(item) for item in value)
        else:
            targets.append(str(value))

    normalized: list[str] = []
    for target in targets:
        token = _normalize_token(target)
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _is_target_relevant(
    *,
    path: str,
    old_value: Any,
    new_value: Any,
    targets: list[str],
) -> bool:
    if not targets:
        return True
    searchable = _normalize_token(f"{path} {old_value} {new_value}")
    return any(target in searchable for target in targets)


def _validation_status_map(
    validation: dict[str, Any] | None,
) -> dict[str, str]:
    if not isinstance(validation, dict):
        return {}

    results = validation.get("results")
    if not isinstance(results, list):
        return {}

    status_map: dict[str, str] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        condition = result.get("condition")
        status = result.get("status")
        if condition is None or status is None:
            continue
        status_map[_normalize_token(condition)] = str(status).lower()
    return status_map


def _infer_goal_progress(
    *,
    before_goal_validation: dict[str, Any] | None,
    after_goal_validation: dict[str, Any] | None,
) -> tuple[str, list[dict[str, str]]]:
    """
    Infer goal progress only from explicit condition-status transitions.

    A scene change alone is not treated as progress toward the goal.
    """
    before = _validation_status_map(before_goal_validation)
    after = _validation_status_map(after_goal_validation)

    transitions: list[dict[str, str]] = []
    positive = 0
    negative = 0

    for condition in sorted(set(before) | set(after)):
        old_status = before.get(condition, "missing")
        new_status = after.get(condition, "missing")
        if old_status == new_status:
            continue

        transitions.append(
            {
                "condition": condition,
                "before": old_status,
                "after": new_status,
            }
        )

        if (
            old_status in {"violated", "uncertain", "missing"}
            and new_status == "satisfied"
        ):
            positive += 1
        elif (
            old_status == "satisfied"
            and new_status in {"violated", "uncertain", "missing"}
        ):
            negative += 1

    if positive and not negative:
        return "positive", transitions
    if negative and not positive:
        return "negative", transitions
    if positive and negative:
        return "mixed", transitions
    if before and after:
        return "none", transitions
    return "unknown", transitions


def _presence_by_target(
    flat_scene: dict[str, Any],
    targets: list[str],
) -> dict[str, bool]:
    searchable_facts = [
        _normalize_token(f"{path} {value}")
        for path, value in flat_scene.items()
    ]
    return {
        target: any(target in fact for fact in searchable_facts)
        for target in targets
    }


def _derive_scene_supported_modifications(
    *,
    relevant_changes: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Conservative scene-to-action bridge.

    It proposes a contact-side modification only when the scene graph
    explicitly exposes a property whose path contains 'accessible_side' or
    'available_side'. No other symbolic choice is invented.
    """
    candidates: list[dict[str, Any]] = []

    for change in relevant_changes:
        path = str(change.get("path", "")).lower()
        if not (
            "accessible_side" in path
            or "available_side" in path
            or "free_side" in path
        ):
            continue

        new_value = change.get("after", change.get("value"))
        if not isinstance(new_value, str) or not new_value.strip():
            continue

        for action_index, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            for field in ("Object contact side", "contact_side"):
                if field not in action:
                    continue
                if action[field] == new_value:
                    continue
                candidates.append(
                    {
                        "action_index": action_index,
                        "field": field,
                        "old_value": deepcopy(action[field]),
                        "new_value": new_value,
                        "justification": (
                            f"The post-execution scene explicitly reports "
                            f"'{new_value}' through {change.get('path')}."
                        ),
                        "source": "scene_transition",
                    }
                )

    # Deduplicate exact changes.
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in candidates:
        key = (
            item["action_index"],
            item["field"],
            json.dumps(item["new_value"], sort_keys=True),
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def analyze_scene_transition(
    *,
    scene_graph_before: dict[str, Any],
    scene_graph_after: dict[str, Any],
    failed_stage: dict[str, Any],
    actions: list[dict[str, Any]],
    before_goal_validation: dict[str, Any] | None = None,
    after_goal_validation: dict[str, Any] | None = None,
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

    before_presence = _presence_by_target(before_flat, targets)
    after_presence = _presence_by_target(after_flat, targets)
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
    }
