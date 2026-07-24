from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
import hashlib
import json


RECOVERY_DECISIONS = {"repeat", "modify", "replace", "replan", "abort"}
SCHEDULER_MODES = {"local_reschedule", "global_replan"}

# These are the only symbolic fields that this module is allowed to modify.
# A modification is applied only when the failure report explicitly proposes
# a concrete new value for one of these fields.
SUPPORTED_SYMBOLIC_FIELDS = {
    "Object contact side",
    "contact_side",
    "Manipulation mode",
    "manipulation_mode",
    "Contact mode",
    "contact_mode",
}


def _conditions_with_status(
    validation: dict[str, Any] | None,
    statuses: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(validation, dict):
        return []
    results = validation.get("results")
    if not isinstance(results, list):
        return []
    return [
        deepcopy(item)
        for item in results
        if isinstance(item, dict) and item.get("status") in statuses
    ]


def failure_signature(failure_report: dict[str, Any] | None) -> str | None:
    if not isinstance(failure_report, dict):
        return None

    conditions = failure_report.get("failed_conditions", [])
    uncertain = failure_report.get("uncertain_conditions", [])
    payload = {
        "stage_id": failure_report.get("stage_id"),
        "failure_phase": failure_report.get("failure_phase"),
        "failure_type": failure_report.get("failure_type"),
        "failed": sorted(
            str(item.get("condition", ""))
            for item in conditions
            if isinstance(item, dict)
        ),
        "uncertain": sorted(
            str(item.get("condition", ""))
            for item in uncertain
            if isinstance(item, dict)
        ),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def extract_relevant_history(
    *,
    attempts: list[dict[str, Any]],
    stage_id: int,
    current_failure_report: dict[str, Any] | None = None,
    latest_scene_graph: dict[str, Any] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Select recovery-relevant evidence from complete attempts."""
    same_stage = [
        deepcopy(item)
        for item in attempts
        if isinstance(item, dict)
        and item.get("stage_id") == stage_id
        and item.get("closed_at")
    ][-limit:]

    current_signature = failure_signature(current_failure_report)
    repeated_failures: list[dict[str, Any]] = []
    strategies: list[dict[str, Any]] = []
    unsatisfied: list[dict[str, Any]] = []

    for attempt in same_stage:
        report = attempt.get("failure_report")
        signature = failure_signature(report)
        if current_signature and signature == current_signature:
            repeated_failures.append(
                {
                    "attempt_id": attempt.get("attempt_id"),
                    "failure_type": (
                        report.get("failure_type")
                        if isinstance(report, dict)
                        else None
                    ),
                    "failure_phase": (
                        report.get("failure_phase")
                        if isinstance(report, dict)
                        else None
                    ),
                    "signature": signature,
                }
            )

        recovery = attempt.get("recovery")
        if isinstance(recovery, dict) and recovery.get("recovery_type"):
            strategies.append(
                {
                    "attempt_id": attempt.get("attempt_id"),
                    "recovery_type": recovery.get("recovery_type"),
                    "changes": deepcopy(recovery.get("changes", {})),
                }
            )

        for phase in ("pre", "post"):
            validation = attempt.get(phase, {}).get("validation")
            for item in _conditions_with_status(
                validation,
                {"violated", "uncertain"},
            ):
                unsatisfied.append(
                    {
                        "attempt_id": attempt.get("attempt_id"),
                        "phase": phase,
                        **item,
                    }
                )

    counts = {
        decision: sum(
            1
            for item in strategies
            if item.get("recovery_type") == decision
        )
        for decision in ("repeat", "modify", "replace", "replan")
    }

    return {
        "stage_id": stage_id,
        "same_stage_attempts": same_stage,
        "repeated_failures": repeated_failures,
        "strategies_already_tried": strategies,
        "conditions_remaining_unsatisfied": unsatisfied,
        "retry_count": max(0, len(same_stage) - 1),
        "strategy_counts": counts,
        "last_observed_state": deepcopy(latest_scene_graph or {}),
        "current_failure_signature": current_signature,
        "extracted_at": datetime.now().isoformat(),
    }


def _execution_completed(
    failure_report: dict[str, Any],
    relevant_history: dict[str, Any],
) -> bool:
    execution = failure_report.get("execution")
    if isinstance(execution, dict) and "completed" in execution:
        return bool(execution.get("completed"))

    same_stage_attempts = relevant_history.get("same_stage_attempts", [])
    if same_stage_attempts:
        latest = same_stage_attempts[-1]
        execution = latest.get("execution")
        if isinstance(execution, dict):
            return bool(execution.get("completed"))

    # A POST failure normally implies that execution reached post-validation.
    return failure_report.get("failure_phase") == "post"


def _extract_supported_modifications(
    *,
    failure_report: dict[str, Any],
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Read explicit symbolic changes from the failure report.

    Supported input formats:
      "supported_symbolic_modifications": [
        {"action_index": 0, "field": "Object contact side",
         "new_value": "left", "justification": "..."}
      ]

    or:
      "recovery_recommendation": {
        "modifications": [...]
      }
    """
    raw = failure_report.get("supported_symbolic_modifications")
    if not isinstance(raw, list):
        recommendation = failure_report.get("recovery_recommendation")
        raw = (
            recommendation.get("modifications")
            if isinstance(recommendation, dict)
            else []
        )

    result: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue

        field = item.get("field")
        new_value = item.get("new_value")
        action_index = item.get("action_index", 0)

        if field not in SUPPORTED_SYMBOLIC_FIELDS:
            continue
        if not isinstance(action_index, int):
            continue
        if action_index < 0 or action_index >= len(actions):
            continue
        if not isinstance(actions[action_index], dict):
            continue
        if field not in actions[action_index]:
            continue
        if new_value is None:
            continue

        old_value = actions[action_index].get(field)
        if old_value == new_value:
            continue

        result.append(
            {
                "action_index": action_index,
                "field": field,
                "old_value": deepcopy(old_value),
                "new_value": deepcopy(new_value),
                "justification": str(
                    item.get(
                        "justification",
                        "Explicitly proposed by the failure report.",
                    )
                ),
            }
        )

    return result


def _extract_replacement_spec(
    failure_report: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Accept a replacement only when the failure report explicitly provides it.

    Expected shape:
      "replacement_subplan": {
        "stages": [...],
        "actions": [...],
        "justification": "..."
      }
    """
    replacement = failure_report.get("replacement_subplan")
    if not isinstance(replacement, dict):
        recommendation = failure_report.get("recovery_recommendation")
        replacement = (
            recommendation.get("replacement_subplan")
            if isinstance(recommendation, dict)
            else None
        )

    if not isinstance(replacement, dict):
        return None

    stages = replacement.get("stages")
    actions = replacement.get("actions")
    if not isinstance(stages, list) or not stages:
        return None
    if not isinstance(actions, list) or not actions:
        return None
    if not all(isinstance(item, dict) for item in stages):
        return None
    if not all(isinstance(item, dict) for item in actions):
        return None

    return {
        "stages": deepcopy(stages),
        "actions": deepcopy(actions),
        "justification": str(
            replacement.get(
                "justification",
                "Explicit replacement supplied by the failure report.",
            )
        ),
    }


def interpret_failure(
    *,
    failure_report: dict[str, Any],
    relevant_history: dict[str, Any],
    failed_stage: dict[str, Any],
    actions: list[dict[str, Any]],
    scene_transition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Describe what is actually supported by the available observations.

    This function does not choose a recovery strategy.
    """
    failure_type = str(failure_report.get("failure_type") or "unknown")
    failure_phase = str(failure_report.get("failure_phase") or "unknown")
    repeated_count = len(relevant_history.get("repeated_failures", []))
    execution_completed = _execution_completed(
        failure_report,
        relevant_history,
    )

    evidence_status = (
        "insufficient"
        if failure_type == "insufficient_evidence"
        else "sufficient"
    )

    cause = failure_report.get("causal_diagnosis")
    if isinstance(cause, dict):
        cause_status = str(cause.get("status") or "unknown")
    else:
        cause_status = str(
            failure_report.get("cause_status") or "unknown"
        )

    transition = deepcopy(scene_transition or {})
    modifications = _extract_supported_modifications(
        failure_report=failure_report,
        actions=actions,
    )

    for candidate in transition.get(
        "supported_symbolic_modifications",
        [],
    ):
        if not isinstance(candidate, dict):
            continue
        key = (
            candidate.get("action_index"),
            candidate.get("field"),
            json.dumps(candidate.get("new_value"), sort_keys=True),
        )
        existing_keys = {
            (
                item.get("action_index"),
                item.get("field"),
                json.dumps(item.get("new_value"), sort_keys=True),
            )
            for item in modifications
        }
        if key not in existing_keys:
            modifications.append(deepcopy(candidate))

    replacement_spec = _extract_replacement_spec(failure_report)
    stage_still_applicable = transition.get(
        "stage_still_applicable",
        True,
    )

    # PRE failures and disappearance of required targets invalidate execution
    # of the original local stage.
    replan_required = (
        failure_phase == "pre"
        or not stage_still_applicable
    )

    return {
        "failure_confirmed": evidence_status == "sufficient",
        "evidence_status": evidence_status,
        "failure_type": failure_type,
        "failure_phase": failure_phase,
        "cause_status": cause_status,
        "execution_completed": execution_completed,
        "same_failure_repeated": repeated_count >= 2,
        "same_failure_count": repeated_count,
        "observable_scene_change": transition.get(
            "observable_change",
            False,
        ),
        "target_state_changed": transition.get(
            "target_state_changed",
            False,
        ),
        "goal_progress": transition.get(
            "goal_progress",
            "unknown",
        ),
        "stage_still_applicable": stage_still_applicable,
        "targets_missing": deepcopy(
            transition.get("targets_missing", [])
        ),
        "scene_transition": transition,
        "supported_symbolic_modifications": modifications,
        "replacement_spec": replacement_spec,
        "replacement_supported": replacement_spec is not None,
        "replan_required": replan_required,
        "failed_stage_id": failed_stage.get("Stage_id"),
        "interpreted_at": datetime.now().isoformat(),
    }


def _repeat_assessment(
    *,
    interpretation: dict[str, Any],
) -> tuple[bool, str]:
    if interpretation["evidence_status"] != "sufficient":
        return False, "The execution outcome is not sufficiently observed."

    if interpretation["failure_phase"] == "pre":
        return False, (
            "Repeating execution cannot repair violated preconditions."
        )

    if not interpretation.get("stage_still_applicable", True):
        return False, (
            "The current scene no longer supports the failed local stage."
        )

    if interpretation.get("goal_progress") in {"negative", "mixed"}:
        return False, (
            "The observed condition transitions do not support an identical "
            "retry toward the local goal."
        )

    if interpretation["same_failure_repeated"]:
        return False, (
            "The same observable failure has already persisted across "
            "multiple attempts."
        )

    if not interpretation["execution_completed"]:
        return True, (
            "The previous execution did not complete, so one identical "
            "retry is justified."
        )

    if interpretation.get("goal_progress") == "positive":
        return True, (
            "The PRE/POST scene evidence shows progress toward the local "
            "goal while the stage remains applicable."
        )

    if (
        interpretation.get("goal_progress") == "none"
        and not interpretation.get("target_state_changed", False)
    ):
        return False, (
            "The completed action produced no verified target-state change "
            "and no postcondition improvement."
        )

    return False, (
        "The scene transition does not establish that repeating the same "
        "completed action would advance the local goal."
    )


def _decision(
    *,
    decision: str,
    reason: str,
    interpretation: dict[str, Any],
    remaining_task_goal: str,
    proposed_changes: dict[str, Any] | None = None,
    admissibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if decision not in RECOVERY_DECISIONS:
        raise ValueError(f"Unsupported recovery decision: {decision}")

    scheduler_request: dict[str, Any] = {}
    if decision != "abort":
        scheduler_request = {
            "mode": (
                "global_replan"
                if decision == "replan"
                else "local_reschedule"
            ),
            "remaining_task_goal": remaining_task_goal,
        }

    return {
        "decision": decision,
        "reason": reason,
        "proposed_changes": deepcopy(proposed_changes or {}),
        "failure_interpretation": deepcopy(interpretation),
        "admissibility": deepcopy(admissibility or {}),
        "scheduler_request": scheduler_request,
        "policy": "evidence_based_v1",
        "created_at": datetime.now().isoformat(),
    }


def plan_recovery_evidence_based(
    *,
    failure_report: dict[str, Any],
    relevant_history: dict[str, Any],
    failure_interpretation: dict[str, Any],
    failed_stage: dict[str, Any],
    actions: list[dict[str, Any]],
    remaining_task_goal: str,
    limits: dict[str, int],
    counters: dict[str, int],
) -> dict[str, Any]:
    """
    Select only recovery strategies justified by current evidence.

    Budgets constrain admissible choices but never constitute the reason
    for selecting repeat, modify, or replace.
    """
    del failure_report, failed_stage, actions  # Already represented above.

    strategy_counts = relevant_history.get("strategy_counts", {})
    interpretation = failure_interpretation

    repeat_ok, repeat_reason = _repeat_assessment(
        interpretation=interpretation,
    )
    modifications = interpretation[
        "supported_symbolic_modifications"
    ]
    replacement_spec = interpretation["replacement_spec"]

    admissibility = {
        "repeat": {
            "admissible": (
                repeat_ok
                and strategy_counts.get("repeat", 0)
                < limits["max_repeats"]
            ),
            "reason": repeat_reason,
        },
        "modify": {
            "admissible": (
                bool(modifications)
                and strategy_counts.get("modify", 0)
                < limits["max_modifications"]
            ),
            "reason": (
                "At least one explicit, represented symbolic change is "
                "supported by the failure report."
                if modifications
                else "No concrete symbolic modification is supported."
            ),
        },
        "replace": {
            "admissible": (
                replacement_spec is not None
                and strategy_counts.get("replace", 0)
                < limits["max_replacements"]
            ),
            "reason": (
                "An explicit replacement subplan is available."
                if replacement_spec is not None
                else "No semantically grounded replacement subplan is available."
            ),
        },
        "replan": {
            "admissible": (
                counters.get("replans", 0) < limits["max_replans"]
            ),
            "reason": (
                "Global replanning remains available from the latest "
                "observed world state."
            ),
        },
    }

    if counters.get("total_actions", 0) >= limits["max_total_actions"]:
        return _decision(
            decision="abort",
            reason="Maximum total action budget reached.",
            interpretation=interpretation,
            remaining_task_goal=remaining_task_goal,
            admissibility=admissibility,
        )

    if relevant_history.get("retry_count", 0) >= (
        limits["max_attempts_per_stage"] - 1
    ):
        return _decision(
            decision="abort",
            reason="Maximum attempts for this stage reached.",
            interpretation=interpretation,
            remaining_task_goal=remaining_task_goal,
            admissibility=admissibility,
        )

    # Uncertainty should normally have been handled by evidence gathering.
    # If it remains unresolved, do not execute an arbitrary local change.
    if interpretation["evidence_status"] == "insufficient":
        if admissibility["replan"]["admissible"]:
            return _decision(
                decision="replan",
                reason=(
                    "Evidence gathering did not resolve the state; no local "
                    "execution change is justified."
                ),
                interpretation=interpretation,
                remaining_task_goal=remaining_task_goal,
                admissibility=admissibility,
            )
        return _decision(
            decision="abort",
            reason=(
                "Evidence remains insufficient and the replan budget is "
                "exhausted."
            ),
            interpretation=interpretation,
            remaining_task_goal=remaining_task_goal,
            admissibility=admissibility,
        )

    if (
        interpretation["replan_required"]
        and admissibility["replan"]["admissible"]
    ):
        return _decision(
            decision="replan",
            reason=(
                "The failed stage's preconditions are not satisfied in the "
                "current world state; repeating or locally modifying its "
                "execution cannot repair them."
            ),
            interpretation=interpretation,
            remaining_task_goal=remaining_task_goal,
            admissibility=admissibility,
        )

    if admissibility["repeat"]["admissible"]:
        return _decision(
            decision="repeat",
            reason=repeat_reason,
            interpretation=interpretation,
            remaining_task_goal=remaining_task_goal,
            proposed_changes={"reuse_same_action": True},
            admissibility=admissibility,
        )

    if admissibility["modify"]["admissible"]:
        return _decision(
            decision="modify",
            reason=(
                "The failure report supports concrete changes to symbolic "
                "fields already represented by the action."
            ),
            interpretation=interpretation,
            remaining_task_goal=remaining_task_goal,
            proposed_changes={
                "symbolic_modifications": modifications,
            },
            admissibility=admissibility,
        )

    if admissibility["replace"]["admissible"]:
        return _decision(
            decision="replace",
            reason=replacement_spec["justification"],
            interpretation=interpretation,
            remaining_task_goal=remaining_task_goal,
            proposed_changes={
                "replacement_subplan": replacement_spec,
            },
            admissibility=admissibility,
        )

    if admissibility["replan"]["admissible"]:
        return _decision(
            decision="replan",
            reason=(
                "No local recovery strategy is sufficiently supported by "
                "the available evidence; replan from the latest world state."
            ),
            interpretation=interpretation,
            remaining_task_goal=remaining_task_goal,
            admissibility=admissibility,
        )

    return _decision(
        decision="abort",
        reason=(
            "No evidence-supported recovery remains within the configured "
            "budgets."
        ),
        interpretation=interpretation,
        remaining_task_goal=remaining_task_goal,
        admissibility=admissibility,
    )


# Backward-compatible alias for callers not yet migrated.
def plan_recovery_deterministic(**kwargs: Any) -> dict[str, Any]:
    interpretation = kwargs.pop("failure_interpretation", None)
    if interpretation is None:
        interpretation = interpret_failure(
            failure_report=kwargs["failure_report"],
            relevant_history=kwargs["relevant_history"],
            failed_stage=kwargs["failed_stage"],
            actions=kwargs["actions"],
            scene_transition=kwargs.pop("scene_transition", None),
        )
    return plan_recovery_evidence_based(
        failure_interpretation=interpretation,
        **kwargs,
    )


def apply_symbolic_modifications(
    actions: list[dict[str, Any]],
    modifications: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = deepcopy(actions)
    applied: list[dict[str, Any]] = []

    for item in modifications:
        index = item["action_index"]
        field = item["field"]
        new_value = deepcopy(item["new_value"])

        if index < 0 or index >= len(result):
            raise IndexError(f"Invalid action index in modification: {index}")
        if field not in SUPPORTED_SYMBOLIC_FIELDS:
            raise ValueError(f"Unsupported symbolic field: {field}")
        if field not in result[index]:
            raise KeyError(
                f"Action {index} does not contain field '{field}'."
            )

        old_value = deepcopy(result[index][field])
        result[index][field] = new_value
        applied.append(
            {
                **deepcopy(item),
                "old_value": old_value,
                "new_value": new_value,
            }
        )

    return result, {"symbolic_modifications": applied}


def _materialize_replacement(
    *,
    replacement_spec: dict[str, Any],
    next_stage_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stages = deepcopy(replacement_spec["stages"])
    actions = deepcopy(replacement_spec["actions"])

    # Normalize IDs while preserving the supplied semantic decomposition.
    step_cursor = next_stage_id * 100 + 1
    step_map: dict[Any, int] = {}

    for offset, stage in enumerate(stages):
        stage["Stage_id"] = next_stage_id + offset
        original_steps = stage.get("Step_id", [])
        if not isinstance(original_steps, list):
            original_steps = [original_steps]

        normalized_steps: list[int] = []
        for original in original_steps:
            new_step = step_cursor
            step_cursor += 1
            step_map[original] = new_step
            normalized_steps.append(new_step)
        stage["Step_id"] = normalized_steps
        stage["_recovery_generated"] = True

    for action in actions:
        original = action.get("Step_id")
        if original in step_map:
            action["Step_id"] = step_map[original]
        elif stages:
            action["Step_id"] = stages[0]["Step_id"][0]

    return stages, actions


def schedule_recovery(
    *,
    recovery_plan: dict[str, Any],
    failed_stage: dict[str, Any],
    failed_actions: list[dict[str, Any]],
    remaining_stages: list[dict[str, Any]],
    next_stage_id: int,
    parent_attempt_id: str,
    next_attempt_number: int,
) -> dict[str, Any]:
    """
    Materialize a recovery decision without adding new reasoning.

    All semantic choices must already be present in recovery_plan.
    """
    decision = recovery_plan["decision"]

    if decision == "abort":
        return {
            "mode": None,
            "stages": [],
            "actions": [],
            "decision": decision,
        }

    if decision == "replan":
        return {
            "mode": "global_replan",
            "stages": [],
            "actions": [],
            "decision": decision,
            "parent_attempt_id": parent_attempt_id,
            "reason": recovery_plan.get("reason"),
        }

    if decision == "repeat":
        stage = deepcopy(failed_stage)
        actions = deepcopy(failed_actions)
        recovery = {
            "parent_attempt_id": parent_attempt_id,
            "recovery_type": "repeat",
            "attempt_number": next_attempt_number,
            "changes": {"reuse_same_action": True},
            "reason": recovery_plan.get("reason"),
        }
        stage["_recovery"] = recovery
        stage["_actions"] = actions
        return {
            "mode": "local_reschedule",
            "stages": [stage, *deepcopy(remaining_stages)],
            "actions": actions,
            "decision": decision,
            "recovery": recovery,
        }

    if decision == "modify":
        modifications = recovery_plan.get(
            "proposed_changes",
            {},
        ).get("symbolic_modifications", [])
        if not modifications:
            raise ValueError(
                "Modify was selected without explicit symbolic changes."
            )

        modified_actions, changes = apply_symbolic_modifications(
            failed_actions,
            modifications,
        )
        stage = deepcopy(failed_stage)
        recovery = {
            "parent_attempt_id": parent_attempt_id,
            "recovery_type": "modify",
            "attempt_number": next_attempt_number,
            "changes": changes,
            "reason": recovery_plan.get("reason"),
        }
        stage["_recovery"] = recovery
        stage["_actions"] = modified_actions
        return {
            "mode": "local_reschedule",
            "stages": [stage, *deepcopy(remaining_stages)],
            "actions": modified_actions,
            "decision": decision,
            "recovery": recovery,
        }

    if decision != "replace":
        raise ValueError(f"Unsupported recovery decision: {decision}")

    replacement_spec = recovery_plan.get(
        "proposed_changes",
        {},
    ).get("replacement_subplan")
    if not isinstance(replacement_spec, dict):
        raise ValueError(
            "Replace was selected without an explicit replacement subplan."
        )

    replacement_stages, replacement_actions = _materialize_replacement(
        replacement_spec=replacement_spec,
        next_stage_id=next_stage_id,
    )
    recovery = {
        "parent_attempt_id": parent_attempt_id,
        "recovery_type": "replace",
        "attempt_number": next_attempt_number,
        "changes": {
            "replacement_stage_count": len(replacement_stages),
            "justification": replacement_spec.get("justification"),
        },
        "reason": recovery_plan.get("reason"),
    }

    for stage in replacement_stages:
        stage["_recovery"] = deepcopy(recovery)

    # Associate actions to stages using normalized Step_id values.
    for stage in replacement_stages:
        step_ids = set(stage.get("Step_id", []))
        stage["_actions"] = [
            deepcopy(action)
            for action in replacement_actions
            if action.get("Step_id") in step_ids
        ]

    return {
        "mode": "local_reschedule",
        "stages": [*replacement_stages, *deepcopy(remaining_stages)],
        "actions": replacement_actions,
        "decision": decision,
        "recovery": recovery,
    }


def check_recovery_limits(
    *,
    limits: dict[str, int],
    counters: dict[str, int],
) -> None:
    for key, maximum in limits.items():
        if maximum < 0:
            raise ValueError(f"{key} must be >= 0")
    if counters.get("total_actions", 0) > limits["max_total_actions"]:
        raise RuntimeError("Maximum total action budget exceeded.")





# from __future__ import annotations

# from copy import deepcopy
# from datetime import datetime
# from typing import Any
# import hashlib
# import json


# RECOVERY_DECISIONS = {"repeat", "modify", "replace", "replan", "abort"}
# SCHEDULER_MODES = {"local_reschedule", "global_replan"}

# # These are the only symbolic fields that this module is allowed to modify.
# # A modification is applied only when the failure report explicitly proposes
# # a concrete new value for one of these fields.
# SUPPORTED_SYMBOLIC_FIELDS = {
#     "Object contact side",
#     "contact_side",
#     "Manipulation mode",
#     "manipulation_mode",
#     "Contact mode",
#     "contact_mode",
# }


# def _conditions_with_status(
#     validation: dict[str, Any] | None,
#     statuses: set[str],
# ) -> list[dict[str, Any]]:
#     if not isinstance(validation, dict):
#         return []
#     results = validation.get("results")
#     if not isinstance(results, list):
#         return []
#     return [
#         deepcopy(item)
#         for item in results
#         if isinstance(item, dict) and item.get("status") in statuses
#     ]


# def failure_signature(failure_report: dict[str, Any] | None) -> str | None:
#     if not isinstance(failure_report, dict):
#         return None

#     conditions = failure_report.get("failed_conditions", [])
#     uncertain = failure_report.get("uncertain_conditions", [])
#     payload = {
#         "stage_id": failure_report.get("stage_id"),
#         "failure_phase": failure_report.get("failure_phase"),
#         "failure_type": failure_report.get("failure_type"),
#         "failed": sorted(
#             str(item.get("condition", ""))
#             for item in conditions
#             if isinstance(item, dict)
#         ),
#         "uncertain": sorted(
#             str(item.get("condition", ""))
#             for item in uncertain
#             if isinstance(item, dict)
#         ),
#     }
#     raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
#     return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# def extract_relevant_history(
#     *,
#     attempts: list[dict[str, Any]],
#     stage_id: int,
#     current_failure_report: dict[str, Any] | None = None,
#     latest_scene_graph: dict[str, Any] | None = None,
#     limit: int = 10,
# ) -> dict[str, Any]:
#     """Select recovery-relevant evidence from complete attempts."""
#     same_stage = [
#         deepcopy(item)
#         for item in attempts
#         if isinstance(item, dict)
#         and item.get("stage_id") == stage_id
#         and item.get("closed_at")
#     ][-limit:]

#     current_signature = failure_signature(current_failure_report)
#     repeated_failures: list[dict[str, Any]] = []
#     strategies: list[dict[str, Any]] = []
#     unsatisfied: list[dict[str, Any]] = []

#     for attempt in same_stage:
#         report = attempt.get("failure_report")
#         signature = failure_signature(report)
#         if current_signature and signature == current_signature:
#             repeated_failures.append(
#                 {
#                     "attempt_id": attempt.get("attempt_id"),
#                     "failure_type": (
#                         report.get("failure_type")
#                         if isinstance(report, dict)
#                         else None
#                     ),
#                     "failure_phase": (
#                         report.get("failure_phase")
#                         if isinstance(report, dict)
#                         else None
#                     ),
#                     "signature": signature,
#                 }
#             )

#         recovery = attempt.get("recovery")
#         if isinstance(recovery, dict) and recovery.get("recovery_type"):
#             strategies.append(
#                 {
#                     "attempt_id": attempt.get("attempt_id"),
#                     "recovery_type": recovery.get("recovery_type"),
#                     "changes": deepcopy(recovery.get("changes", {})),
#                 }
#             )

#         for phase in ("pre", "post"):
#             validation = attempt.get(phase, {}).get("validation")
#             for item in _conditions_with_status(
#                 validation,
#                 {"violated", "uncertain"},
#             ):
#                 unsatisfied.append(
#                     {
#                         "attempt_id": attempt.get("attempt_id"),
#                         "phase": phase,
#                         **item,
#                     }
#                 )

#     counts = {
#         decision: sum(
#             1
#             for item in strategies
#             if item.get("recovery_type") == decision
#         )
#         for decision in ("repeat", "modify", "replace", "replan")
#     }

#     return {
#         "stage_id": stage_id,
#         "same_stage_attempts": same_stage,
#         "repeated_failures": repeated_failures,
#         "strategies_already_tried": strategies,
#         "conditions_remaining_unsatisfied": unsatisfied,
#         "retry_count": max(0, len(same_stage) - 1),
#         "strategy_counts": counts,
#         "last_observed_state": deepcopy(latest_scene_graph or {}),
#         "current_failure_signature": current_signature,
#         "extracted_at": datetime.now().isoformat(),
#     }


# def _execution_completed(
#     failure_report: dict[str, Any],
#     relevant_history: dict[str, Any],
# ) -> bool:
#     execution = failure_report.get("execution")
#     if isinstance(execution, dict) and "completed" in execution:
#         return bool(execution.get("completed"))

#     same_stage_attempts = relevant_history.get("same_stage_attempts", [])
#     if same_stage_attempts:
#         latest = same_stage_attempts[-1]
#         execution = latest.get("execution")
#         if isinstance(execution, dict):
#             return bool(execution.get("completed"))

#     # A POST failure normally implies that execution reached post-validation.
#     return failure_report.get("failure_phase") == "post"


# def _extract_supported_modifications(
#     *,
#     failure_report: dict[str, Any],
#     actions: list[dict[str, Any]],
# ) -> list[dict[str, Any]]:
#     """
#     Read explicit symbolic changes from the failure report.

#     Supported input formats:
#       "supported_symbolic_modifications": [
#         {"action_index": 0, "field": "Object contact side",
#          "new_value": "left", "justification": "..."}
#       ]

#     or:
#       "recovery_recommendation": {
#         "modifications": [...]
#       }
#     """
#     raw = failure_report.get("supported_symbolic_modifications")
#     if not isinstance(raw, list):
#         recommendation = failure_report.get("recovery_recommendation")
#         raw = (
#             recommendation.get("modifications")
#             if isinstance(recommendation, dict)
#             else []
#         )

#     result: list[dict[str, Any]] = []
#     for item in raw or []:
#         if not isinstance(item, dict):
#             continue

#         field = item.get("field")
#         new_value = item.get("new_value")
#         action_index = item.get("action_index", 0)

#         if field not in SUPPORTED_SYMBOLIC_FIELDS:
#             continue
#         if not isinstance(action_index, int):
#             continue
#         if action_index < 0 or action_index >= len(actions):
#             continue
#         if not isinstance(actions[action_index], dict):
#             continue
#         if field not in actions[action_index]:
#             continue
#         if new_value is None:
#             continue

#         old_value = actions[action_index].get(field)
#         if old_value == new_value:
#             continue

#         result.append(
#             {
#                 "action_index": action_index,
#                 "field": field,
#                 "old_value": deepcopy(old_value),
#                 "new_value": deepcopy(new_value),
#                 "justification": str(
#                     item.get(
#                         "justification",
#                         "Explicitly proposed by the failure report.",
#                     )
#                 ),
#             }
#         )

#     return result


# def _extract_replacement_spec(
#     failure_report: dict[str, Any],
# ) -> dict[str, Any] | None:
#     """
#     Accept a replacement only when the failure report explicitly provides it.

#     Expected shape:
#       "replacement_subplan": {
#         "stages": [...],
#         "actions": [...],
#         "justification": "..."
#       }
#     """
#     replacement = failure_report.get("replacement_subplan")
#     if not isinstance(replacement, dict):
#         recommendation = failure_report.get("recovery_recommendation")
#         replacement = (
#             recommendation.get("replacement_subplan")
#             if isinstance(recommendation, dict)
#             else None
#         )

#     if not isinstance(replacement, dict):
#         return None

#     stages = replacement.get("stages")
#     actions = replacement.get("actions")
#     if not isinstance(stages, list) or not stages:
#         return None
#     if not isinstance(actions, list) or not actions:
#         return None
#     if not all(isinstance(item, dict) for item in stages):
#         return None
#     if not all(isinstance(item, dict) for item in actions):
#         return None

#     return {
#         "stages": deepcopy(stages),
#         "actions": deepcopy(actions),
#         "justification": str(
#             replacement.get(
#                 "justification",
#                 "Explicit replacement supplied by the failure report.",
#             )
#         ),
#     }


# def interpret_failure(
#     *,
#     failure_report: dict[str, Any],
#     relevant_history: dict[str, Any],
#     failed_stage: dict[str, Any],
#     actions: list[dict[str, Any]],
# ) -> dict[str, Any]:
#     """
#     Describe what is actually supported by the available observations.

#     This function does not choose a recovery strategy.
#     """
#     failure_type = str(failure_report.get("failure_type") or "unknown")
#     failure_phase = str(failure_report.get("failure_phase") or "unknown")
#     repeated_count = len(relevant_history.get("repeated_failures", []))
#     execution_completed = _execution_completed(
#         failure_report,
#         relevant_history,
#     )

#     evidence_status = (
#         "insufficient"
#         if failure_type == "insufficient_evidence"
#         else "sufficient"
#     )

#     cause = failure_report.get("causal_diagnosis")
#     if isinstance(cause, dict):
#         cause_status = str(cause.get("status") or "unknown")
#     else:
#         cause_status = str(
#             failure_report.get("cause_status") or "unknown"
#         )

#     modifications = _extract_supported_modifications(
#         failure_report=failure_report,
#         actions=actions,
#     )
#     replacement_spec = _extract_replacement_spec(failure_report)

#     # PRE failures indicate that executing the same stage cannot repair the
#     # missing preconditions. A global replan is therefore immediately justified.
#     replan_required = failure_phase == "pre"

#     return {
#         "failure_confirmed": evidence_status == "sufficient",
#         "evidence_status": evidence_status,
#         "failure_type": failure_type,
#         "failure_phase": failure_phase,
#         "cause_status": cause_status,
#         "execution_completed": execution_completed,
#         "same_failure_repeated": repeated_count >= 2,
#         "same_failure_count": repeated_count,
#         "supported_symbolic_modifications": modifications,
#         "replacement_spec": replacement_spec,
#         "replacement_supported": replacement_spec is not None,
#         "replan_required": replan_required,
#         "failed_stage_id": failed_stage.get("Stage_id"),
#         "interpreted_at": datetime.now().isoformat(),
#     }


# def _repeat_assessment(
#     *,
#     interpretation: dict[str, Any],
# ) -> tuple[bool, str]:
#     if interpretation["evidence_status"] != "sufficient":
#         return False, "The execution outcome is not sufficiently observed."

#     if interpretation["failure_phase"] == "pre":
#         return False, (
#             "Repeating execution cannot repair violated preconditions."
#         )

#     if not interpretation["execution_completed"]:
#         return True, (
#             "The previous execution did not complete, so one identical "
#             "retry is justified."
#         )

#     if interpretation["same_failure_repeated"]:
#         return False, (
#             "The same observable failure has already persisted across "
#             "multiple attempts."
#         )

#     return True, (
#         "One completed execution failed, but no persistent repeated-failure "
#         "pattern has yet been established; one controlled retry is admissible."
#     )


# def _decision(
#     *,
#     decision: str,
#     reason: str,
#     interpretation: dict[str, Any],
#     remaining_task_goal: str,
#     proposed_changes: dict[str, Any] | None = None,
#     admissibility: dict[str, Any] | None = None,
# ) -> dict[str, Any]:
#     if decision not in RECOVERY_DECISIONS:
#         raise ValueError(f"Unsupported recovery decision: {decision}")

#     scheduler_request: dict[str, Any] = {}
#     if decision != "abort":
#         scheduler_request = {
#             "mode": (
#                 "global_replan"
#                 if decision == "replan"
#                 else "local_reschedule"
#             ),
#             "remaining_task_goal": remaining_task_goal,
#         }

#     return {
#         "decision": decision,
#         "reason": reason,
#         "proposed_changes": deepcopy(proposed_changes or {}),
#         "failure_interpretation": deepcopy(interpretation),
#         "admissibility": deepcopy(admissibility or {}),
#         "scheduler_request": scheduler_request,
#         "policy": "evidence_based_v1",
#         "created_at": datetime.now().isoformat(),
#     }


# def plan_recovery_evidence_based(
#     *,
#     failure_report: dict[str, Any],
#     relevant_history: dict[str, Any],
#     failure_interpretation: dict[str, Any],
#     failed_stage: dict[str, Any],
#     actions: list[dict[str, Any]],
#     remaining_task_goal: str,
#     limits: dict[str, int],
#     counters: dict[str, int],
# ) -> dict[str, Any]:
#     """
#     Select only recovery strategies justified by current evidence.

#     Budgets constrain admissible choices but never constitute the reason
#     for selecting repeat, modify, or replace.
#     """
#     del failure_report, failed_stage, actions  # Already represented above.

#     strategy_counts = relevant_history.get("strategy_counts", {})
#     interpretation = failure_interpretation

#     repeat_ok, repeat_reason = _repeat_assessment(
#         interpretation=interpretation,
#     )
#     modifications = interpretation[
#         "supported_symbolic_modifications"
#     ]
#     replacement_spec = interpretation["replacement_spec"]

#     admissibility = {
#         "repeat": {
#             "admissible": (
#                 repeat_ok
#                 and strategy_counts.get("repeat", 0)
#                 < limits["max_repeats"]
#             ),
#             "reason": repeat_reason,
#         },
#         "modify": {
#             "admissible": (
#                 bool(modifications)
#                 and strategy_counts.get("modify", 0)
#                 < limits["max_modifications"]
#             ),
#             "reason": (
#                 "At least one explicit, represented symbolic change is "
#                 "supported by the failure report."
#                 if modifications
#                 else "No concrete symbolic modification is supported."
#             ),
#         },
#         "replace": {
#             "admissible": (
#                 replacement_spec is not None
#                 and strategy_counts.get("replace", 0)
#                 < limits["max_replacements"]
#             ),
#             "reason": (
#                 "An explicit replacement subplan is available."
#                 if replacement_spec is not None
#                 else "No semantically grounded replacement subplan is available."
#             ),
#         },
#         "replan": {
#             "admissible": (
#                 counters.get("replans", 0) < limits["max_replans"]
#             ),
#             "reason": (
#                 "Global replanning remains available from the latest "
#                 "observed world state."
#             ),
#         },
#     }

#     if counters.get("total_actions", 0) >= limits["max_total_actions"]:
#         return _decision(
#             decision="abort",
#             reason="Maximum total action budget reached.",
#             interpretation=interpretation,
#             remaining_task_goal=remaining_task_goal,
#             admissibility=admissibility,
#         )

#     if relevant_history.get("retry_count", 0) >= (
#         limits["max_attempts_per_stage"] - 1
#     ):
#         return _decision(
#             decision="abort",
#             reason="Maximum attempts for this stage reached.",
#             interpretation=interpretation,
#             remaining_task_goal=remaining_task_goal,
#             admissibility=admissibility,
#         )

#     # Uncertainty should normally have been handled by evidence gathering.
#     # If it remains unresolved, do not execute an arbitrary local change.
#     if interpretation["evidence_status"] == "insufficient":
#         if admissibility["replan"]["admissible"]:
#             return _decision(
#                 decision="replan",
#                 reason=(
#                     "Evidence gathering did not resolve the state; no local "
#                     "execution change is justified."
#                 ),
#                 interpretation=interpretation,
#                 remaining_task_goal=remaining_task_goal,
#                 admissibility=admissibility,
#             )
#         return _decision(
#             decision="abort",
#             reason=(
#                 "Evidence remains insufficient and the replan budget is "
#                 "exhausted."
#             ),
#             interpretation=interpretation,
#             remaining_task_goal=remaining_task_goal,
#             admissibility=admissibility,
#         )

#     if (
#         interpretation["replan_required"]
#         and admissibility["replan"]["admissible"]
#     ):
#         return _decision(
#             decision="replan",
#             reason=(
#                 "The failed stage's preconditions are not satisfied in the "
#                 "current world state; repeating or locally modifying its "
#                 "execution cannot repair them."
#             ),
#             interpretation=interpretation,
#             remaining_task_goal=remaining_task_goal,
#             admissibility=admissibility,
#         )

#     if admissibility["repeat"]["admissible"]:
#         return _decision(
#             decision="repeat",
#             reason=repeat_reason,
#             interpretation=interpretation,
#             remaining_task_goal=remaining_task_goal,
#             proposed_changes={"reuse_same_action": True},
#             admissibility=admissibility,
#         )

#     if admissibility["modify"]["admissible"]:
#         return _decision(
#             decision="modify",
#             reason=(
#                 "The failure report supports concrete changes to symbolic "
#                 "fields already represented by the action."
#             ),
#             interpretation=interpretation,
#             remaining_task_goal=remaining_task_goal,
#             proposed_changes={
#                 "symbolic_modifications": modifications,
#             },
#             admissibility=admissibility,
#         )

#     if admissibility["replace"]["admissible"]:
#         return _decision(
#             decision="replace",
#             reason=replacement_spec["justification"],
#             interpretation=interpretation,
#             remaining_task_goal=remaining_task_goal,
#             proposed_changes={
#                 "replacement_subplan": replacement_spec,
#             },
#             admissibility=admissibility,
#         )

#     if admissibility["replan"]["admissible"]:
#         return _decision(
#             decision="replan",
#             reason=(
#                 "No local recovery strategy is sufficiently supported by "
#                 "the available evidence; replan from the latest world state."
#             ),
#             interpretation=interpretation,
#             remaining_task_goal=remaining_task_goal,
#             admissibility=admissibility,
#         )

#     return _decision(
#         decision="abort",
#         reason=(
#             "No evidence-supported recovery remains within the configured "
#             "budgets."
#         ),
#         interpretation=interpretation,
#         remaining_task_goal=remaining_task_goal,
#         admissibility=admissibility,
#     )


# # Backward-compatible alias for callers not yet migrated.
# def plan_recovery_deterministic(**kwargs: Any) -> dict[str, Any]:
#     interpretation = kwargs.pop("failure_interpretation", None)
#     if interpretation is None:
#         interpretation = interpret_failure(
#             failure_report=kwargs["failure_report"],
#             relevant_history=kwargs["relevant_history"],
#             failed_stage=kwargs["failed_stage"],
#             actions=kwargs["actions"],
#         )
#     return plan_recovery_evidence_based(
#         failure_interpretation=interpretation,
#         **kwargs,
#     )


# def apply_symbolic_modifications(
#     actions: list[dict[str, Any]],
#     modifications: list[dict[str, Any]],
# ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
#     result = deepcopy(actions)
#     applied: list[dict[str, Any]] = []

#     for item in modifications:
#         index = item["action_index"]
#         field = item["field"]
#         new_value = deepcopy(item["new_value"])

#         if index < 0 or index >= len(result):
#             raise IndexError(f"Invalid action index in modification: {index}")
#         if field not in SUPPORTED_SYMBOLIC_FIELDS:
#             raise ValueError(f"Unsupported symbolic field: {field}")
#         if field not in result[index]:
#             raise KeyError(
#                 f"Action {index} does not contain field '{field}'."
#             )

#         old_value = deepcopy(result[index][field])
#         result[index][field] = new_value
#         applied.append(
#             {
#                 **deepcopy(item),
#                 "old_value": old_value,
#                 "new_value": new_value,
#             }
#         )

#     return result, {"symbolic_modifications": applied}


# def _materialize_replacement(
#     *,
#     replacement_spec: dict[str, Any],
#     next_stage_id: int,
# ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
#     stages = deepcopy(replacement_spec["stages"])
#     actions = deepcopy(replacement_spec["actions"])

#     # Normalize IDs while preserving the supplied semantic decomposition.
#     step_cursor = next_stage_id * 100 + 1
#     step_map: dict[Any, int] = {}

#     for offset, stage in enumerate(stages):
#         stage["Stage_id"] = next_stage_id + offset
#         original_steps = stage.get("Step_id", [])
#         if not isinstance(original_steps, list):
#             original_steps = [original_steps]

#         normalized_steps: list[int] = []
#         for original in original_steps:
#             new_step = step_cursor
#             step_cursor += 1
#             step_map[original] = new_step
#             normalized_steps.append(new_step)
#         stage["Step_id"] = normalized_steps
#         stage["_recovery_generated"] = True

#     for action in actions:
#         original = action.get("Step_id")
#         if original in step_map:
#             action["Step_id"] = step_map[original]
#         elif stages:
#             action["Step_id"] = stages[0]["Step_id"][0]

#     return stages, actions


# def schedule_recovery(
#     *,
#     recovery_plan: dict[str, Any],
#     failed_stage: dict[str, Any],
#     failed_actions: list[dict[str, Any]],
#     remaining_stages: list[dict[str, Any]],
#     next_stage_id: int,
#     parent_attempt_id: str,
#     next_attempt_number: int,
# ) -> dict[str, Any]:
#     """
#     Materialize a recovery decision without adding new reasoning.

#     All semantic choices must already be present in recovery_plan.
#     """
#     decision = recovery_plan["decision"]

#     if decision == "abort":
#         return {
#             "mode": None,
#             "stages": [],
#             "actions": [],
#             "decision": decision,
#         }

#     if decision == "replan":
#         return {
#             "mode": "global_replan",
#             "stages": [],
#             "actions": [],
#             "decision": decision,
#             "parent_attempt_id": parent_attempt_id,
#             "reason": recovery_plan.get("reason"),
#         }

#     if decision == "repeat":
#         stage = deepcopy(failed_stage)
#         actions = deepcopy(failed_actions)
#         recovery = {
#             "parent_attempt_id": parent_attempt_id,
#             "recovery_type": "repeat",
#             "attempt_number": next_attempt_number,
#             "changes": {"reuse_same_action": True},
#             "reason": recovery_plan.get("reason"),
#         }
#         stage["_recovery"] = recovery
#         stage["_actions"] = actions
#         return {
#             "mode": "local_reschedule",
#             "stages": [stage, *deepcopy(remaining_stages)],
#             "actions": actions,
#             "decision": decision,
#             "recovery": recovery,
#         }

#     if decision == "modify":
#         modifications = recovery_plan.get(
#             "proposed_changes",
#             {},
#         ).get("symbolic_modifications", [])
#         if not modifications:
#             raise ValueError(
#                 "Modify was selected without explicit symbolic changes."
#             )

#         modified_actions, changes = apply_symbolic_modifications(
#             failed_actions,
#             modifications,
#         )
#         stage = deepcopy(failed_stage)
#         recovery = {
#             "parent_attempt_id": parent_attempt_id,
#             "recovery_type": "modify",
#             "attempt_number": next_attempt_number,
#             "changes": changes,
#             "reason": recovery_plan.get("reason"),
#         }
#         stage["_recovery"] = recovery
#         stage["_actions"] = modified_actions
#         return {
#             "mode": "local_reschedule",
#             "stages": [stage, *deepcopy(remaining_stages)],
#             "actions": modified_actions,
#             "decision": decision,
#             "recovery": recovery,
#         }

#     if decision != "replace":
#         raise ValueError(f"Unsupported recovery decision: {decision}")

#     replacement_spec = recovery_plan.get(
#         "proposed_changes",
#         {},
#     ).get("replacement_subplan")
#     if not isinstance(replacement_spec, dict):
#         raise ValueError(
#             "Replace was selected without an explicit replacement subplan."
#         )

#     replacement_stages, replacement_actions = _materialize_replacement(
#         replacement_spec=replacement_spec,
#         next_stage_id=next_stage_id,
#     )
#     recovery = {
#         "parent_attempt_id": parent_attempt_id,
#         "recovery_type": "replace",
#         "attempt_number": next_attempt_number,
#         "changes": {
#             "replacement_stage_count": len(replacement_stages),
#             "justification": replacement_spec.get("justification"),
#         },
#         "reason": recovery_plan.get("reason"),
#     }

#     for stage in replacement_stages:
#         stage["_recovery"] = deepcopy(recovery)

#     # Associate actions to stages using normalized Step_id values.
#     for stage in replacement_stages:
#         step_ids = set(stage.get("Step_id", []))
#         stage["_actions"] = [
#             deepcopy(action)
#             for action in replacement_actions
#             if action.get("Step_id") in step_ids
#         ]

#     return {
#         "mode": "local_reschedule",
#         "stages": [*replacement_stages, *deepcopy(remaining_stages)],
#         "actions": replacement_actions,
#         "decision": decision,
#         "recovery": recovery,
#     }


# def check_recovery_limits(
#     *,
#     limits: dict[str, int],
#     counters: dict[str, int],
# ) -> None:
#     for key, maximum in limits.items():
#         if maximum < 0:
#             raise ValueError(f"{key} must be >= 0")
#     if counters.get("total_actions", 0) > limits["max_total_actions"]:
#         raise RuntimeError("Maximum total action budget exceeded.")





# # from __future__ import annotations

# # from copy import deepcopy
# # from datetime import datetime
# # from typing import Any
# # import json
# # import hashlib


# # RECOVERY_DECISIONS = {"repeat", "modify", "replace", "replan", "abort"}
# # SCHEDULER_MODES = {"local_reschedule", "global_replan"}


# # def _conditions_with_status(
# #     validation: dict[str, Any] | None,
# #     statuses: set[str],
# # ) -> list[dict[str, Any]]:
# #     if not isinstance(validation, dict):
# #         return []
# #     results = validation.get("results")
# #     if not isinstance(results, list):
# #         return []
# #     return [
# #         deepcopy(item)
# #         for item in results
# #         if isinstance(item, dict) and item.get("status") in statuses
# #     ]


# # def failure_signature(failure_report: dict[str, Any] | None) -> str | None:
# #     if not isinstance(failure_report, dict):
# #         return None
# #     conditions = failure_report.get("failed_conditions", [])
# #     uncertain = failure_report.get("uncertain_conditions", [])
# #     payload = {
# #         "stage_id": failure_report.get("stage_id"),
# #         "failure_phase": failure_report.get("failure_phase"),
# #         "failure_type": failure_report.get("failure_type"),
# #         "failed": sorted(
# #             str(item.get("condition", ""))
# #             for item in conditions
# #             if isinstance(item, dict)
# #         ),
# #         "uncertain": sorted(
# #             str(item.get("condition", ""))
# #             for item in uncertain
# #             if isinstance(item, dict)
# #         ),
# #     }
# #     raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
# #     return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# # def extract_relevant_history(
# #     *,
# #     attempts: list[dict[str, Any]],
# #     stage_id: int,
# #     current_failure_report: dict[str, Any] | None = None,
# #     latest_scene_graph: dict[str, Any] | None = None,
# #     limit: int = 10,
# # ) -> dict[str, Any]:
# #     """Select only recovery-relevant information from complete attempts."""
# #     same_stage = [
# #         deepcopy(item)
# #         for item in attempts
# #         if isinstance(item, dict)
# #         and item.get("stage_id") == stage_id
# #         and item.get("closed_at")
# #     ][-limit:]

# #     current_signature = failure_signature(current_failure_report)
# #     repeated_failures: list[dict[str, Any]] = []
# #     strategies: list[dict[str, Any]] = []
# #     unsatisfied: list[dict[str, Any]] = []

# #     for attempt in same_stage:
# #         report = attempt.get("failure_report")
# #         signature = failure_signature(report)
# #         if current_signature and signature == current_signature:
# #             repeated_failures.append(
# #                 {
# #                     "attempt_id": attempt.get("attempt_id"),
# #                     "failure_type": report.get("failure_type") if isinstance(report, dict) else None,
# #                     "failure_phase": report.get("failure_phase") if isinstance(report, dict) else None,
# #                     "signature": signature,
# #                 }
# #             )

# #         recovery = attempt.get("recovery")
# #         if isinstance(recovery, dict) and recovery.get("recovery_type"):
# #             strategies.append(
# #                 {
# #                     "attempt_id": attempt.get("attempt_id"),
# #                     "recovery_type": recovery.get("recovery_type"),
# #                     "changes": deepcopy(recovery.get("changes", {})),
# #                 }
# #             )

# #         for phase in ("pre", "post"):
# #             validation = attempt.get(phase, {}).get("validation")
# #             for item in _conditions_with_status(validation, {"violated", "uncertain"}):
# #                 unsatisfied.append(
# #                     {
# #                         "attempt_id": attempt.get("attempt_id"),
# #                         "phase": phase,
# #                         **item,
# #                     }
# #                 )

# #     counts = {
# #         "repeat": sum(1 for x in strategies if x.get("recovery_type") == "repeat"),
# #         "modify": sum(1 for x in strategies if x.get("recovery_type") == "modify"),
# #         "replace": sum(1 for x in strategies if x.get("recovery_type") == "replace"),
# #         "replan": sum(1 for x in strategies if x.get("recovery_type") == "replan"),
# #     }

# #     return {
# #         "stage_id": stage_id,
# #         "same_stage_attempts": same_stage,
# #         "repeated_failures": repeated_failures,
# #         "strategies_already_tried": strategies,
# #         "conditions_remaining_unsatisfied": unsatisfied,
# #         "retry_count": max(0, len(same_stage) - 1),
# #         "strategy_counts": counts,
# #         "last_observed_state": deepcopy(latest_scene_graph or {}),
# #         "current_failure_signature": current_signature,
# #         "extracted_at": datetime.now().isoformat(),
# #     }


# # def _target_from_actions(actions: list[dict[str, Any]]) -> str:
# #     for action in actions:
# #         if isinstance(action, dict):
# #             target = action.get("Target", action.get("target"))
# #             if isinstance(target, str) and target.strip():
# #                 return target.strip()
# #     return "target object"


# # def build_modified_actions(
# #     actions: list[dict[str, Any]],
# #     modification_index: int,
# # ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
# #     modified = deepcopy(actions)
# #     changes: dict[str, Any] = {
# #         "contact_side": None,
# #         "manipulation_mode": None,
# #         "approach": "slower_reapproach",
# #         "constraints": [
# #             "reacquire target pose before contact",
# #             "reduce execution speed",
# #             "verify contact before completing the action",
# #         ],
# #     }

# #     side_cycle = {
# #         "right": "left",
# #         "left": "right",
# #         "right & left": "front",
# #         "front": "right & left",
# #         "Top": "front",
# #         "top": "front",
# #     }
# #     mode_cycle = {
# #         "Dual-arm symmetric": "Single-arm (right)",
# #         "Single-arm (right)": "Single-arm (left)",
# #         "Single-arm (left)": "Dual-arm symmetric",
# #     }

# #     for action in modified:
# #         if not isinstance(action, dict):
# #             continue
# #         side_key = "Object contact side" if "Object contact side" in action else "contact_side"
# #         old_side = action.get(side_key)
# #         new_side = side_cycle.get(str(old_side), "front")
# #         action[side_key] = new_side
# #         changes["contact_side"] = {"from": old_side, "to": new_side}

# #         mode_key = "Manipulation mode" if "Manipulation mode" in action else "manipulation_mode"
# #         old_mode = action.get(mode_key)
# #         new_mode = mode_cycle.get(str(old_mode), "Single-arm (right)")
# #         action[mode_key] = new_mode
# #         changes["manipulation_mode"] = {"from": old_mode, "to": new_mode}
# #         action["Recovery approach"] = changes["approach"]
# #         action["Recovery constraints"] = deepcopy(changes["constraints"])
# #         action["Modification index"] = modification_index

# #     return modified, changes


# # def build_replacement_subplan(
# #     *,
# #     failed_stage: dict[str, Any],
# #     actions: list[dict[str, Any]],
# #     next_stage_id: int,
# # ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
# #     target = _target_from_actions(actions)
# #     original_pre = deepcopy(failed_stage.get("Preconditions", []))
# #     original_post = deepcopy(failed_stage.get("Postconditions", []))

# #     prep_stage = {
# #         "Stage_id": next_stage_id,
# #         "Step_id": [next_stage_id * 100 + 1],
# #         "Local_goal": f"Restore access and alignment for {target} before retrying the failed goal.",
# #         "Preconditions": [
# #             "The robot can observe the target workspace.",
# #         ],
# #         "Postconditions": original_pre or [
# #             f"{target} is observable and reachable."
# #         ],
# #         "_recovery_generated": True,
# #     }
# #     execution_stage = {
# #         "Stage_id": next_stage_id + 1,
# #         "Step_id": [next_stage_id * 100 + 2],
# #         "Local_goal": str(failed_stage.get("Local_goal", "")),
# #         "Preconditions": original_pre,
# #         "Postconditions": original_post,
# #         "_recovery_generated": True,
# #     }

# #     prep_action = {
# #         "Step_id": prep_stage["Step_id"][0],
# #         "Target": target,
# #         "Action concept": "Reposition",
# #         "Manipulation mode": "Single-arm (right)",
# #         "Contact mode": "Without claw",
# #         "Object contact side": "front",
# #         "Recovery approach": "restore_access_and_alignment",
# #     }
# #     modified_actions, _ = build_modified_actions(actions, modification_index=1)
# #     for idx, action in enumerate(modified_actions):
# #         action["Step_id"] = execution_stage["Step_id"][0] + idx

# #     return [prep_stage, execution_stage], [prep_action, *modified_actions]


# # def plan_recovery_deterministic(
# #     *,
# #     failure_report: dict[str, Any],
# #     relevant_history: dict[str, Any],
# #     failed_stage: dict[str, Any],
# #     actions: list[dict[str, Any]],
# #     remaining_task_goal: str,
# #     limits: dict[str, int],
# #     counters: dict[str, int],
# # ) -> dict[str, Any]:
# #     """Deterministic initial recovery policy."""
# #     failure_type = failure_report.get("failure_type")
# #     repeated_count = len(relevant_history.get("repeated_failures", []))
# #     strategy_counts = relevant_history.get("strategy_counts", {})

# #     if counters.get("total_actions", 0) >= limits["max_total_actions"]:
# #         decision, reason = "abort", "Maximum total action budget reached."
# #     elif relevant_history.get("retry_count", 0) >= limits["max_attempts_per_stage"] - 1:
# #         decision, reason = "abort", "Maximum attempts for this stage reached."
# #     elif failure_type == "insufficient_evidence":
# #         decision, reason = "replan", "The state remains uncertain; repeating the same execution is not justified."
# #     elif repeated_count >= 2 and strategy_counts.get("modify", 0) >= limits["max_modifications"]:
# #         if strategy_counts.get("replace", 0) < limits["max_replacements"]:
# #             decision, reason = "replace", "The same failure repeated after prior strategies; replace the stage structurally."
# #         else:
# #             decision, reason = "replan", "Local recovery strategies were exhausted."
# #     elif strategy_counts.get("repeat", 0) < limits["max_repeats"]:
# #         decision, reason = "repeat", "First recovery attempt: retry the same stage from the latest observed state."
# #     elif strategy_counts.get("modify", 0) < limits["max_modifications"]:
# #         decision, reason = "modify", "A plain repeat was already tried; modify execution parameters."
# #     elif strategy_counts.get("replace", 0) < limits["max_replacements"]:
# #         decision, reason = "replace", "Repeat and modification budgets were exhausted; replace the stage."
# #     elif counters.get("replans", 0) < limits["max_replans"]:
# #         decision, reason = "replan", "All local strategies were exhausted; request a global replan."
# #     else:
# #         decision, reason = "abort", "All configured recovery budgets were exhausted."

# #     proposed_changes: dict[str, Any] = {}
# #     scheduler_request: dict[str, Any] = {
# #         "mode": "global_replan" if decision == "replan" else "local_reschedule",
# #         "remaining_task_goal": remaining_task_goal,
# #     }

# #     if decision == "modify":
# #         _, proposed_changes = build_modified_actions(
# #             actions,
# #             modification_index=strategy_counts.get("modify", 0) + 1,
# #         )
# #     elif decision == "repeat":
# #         proposed_changes = {"reuse_same_action": True}
# #     elif decision == "replace":
# #         proposed_changes = {"replace_failed_stage": True}
# #     elif decision == "abort":
# #         scheduler_request = {}

# #     return {
# #         "decision": decision,
# #         "reason": reason,
# #         "proposed_changes": proposed_changes,
# #         "scheduler_request": scheduler_request,
# #         "policy": "deterministic_v1",
# #         "created_at": datetime.now().isoformat(),
# #     }


# # def schedule_recovery(
# #     *,
# #     recovery_plan: dict[str, Any],
# #     failed_stage: dict[str, Any],
# #     failed_actions: list[dict[str, Any]],
# #     remaining_stages: list[dict[str, Any]],
# #     next_stage_id: int,
# #     parent_attempt_id: str,
# #     next_attempt_number: int,
# # ) -> dict[str, Any]:
# #     decision = recovery_plan["decision"]

# #     if decision == "abort":
# #         return {
# #             "mode": None,
# #             "stages": [],
# #             "actions": [],
# #             "decision": decision,
# #         }

# #     if decision == "replan":
# #         return {
# #             "mode": "global_replan",
# #             "stages": [],
# #             "actions": [],
# #             "decision": decision,
# #             "parent_attempt_id": parent_attempt_id,
# #         }

# #     if decision == "repeat":
# #         stage = deepcopy(failed_stage)
# #         actions = deepcopy(failed_actions)
# #         recovery = {
# #             "parent_attempt_id": parent_attempt_id,
# #             "recovery_type": "repeat",
# #             "attempt_number": next_attempt_number,
# #             "changes": {},
# #         }
# #         stage["_recovery"] = recovery
# #         stage["_actions"] = actions
# #         new_stages = [stage, *deepcopy(remaining_stages)]
# #         return {
# #             "mode": "local_reschedule",
# #             "stages": new_stages,
# #             "actions": actions,
# #             "decision": decision,
# #             "recovery": recovery,
# #         }

# #     if decision == "modify":
# #         modified_actions, changes = build_modified_actions(
# #             failed_actions,
# #             modification_index=next_attempt_number,
# #         )
# #         stage = deepcopy(failed_stage)
# #         recovery = {
# #             "parent_attempt_id": parent_attempt_id,
# #             "recovery_type": "modify",
# #             "attempt_number": next_attempt_number,
# #             "changes": changes,
# #         }
# #         stage["_recovery"] = recovery
# #         stage["_actions"] = modified_actions
# #         new_stages = [stage, *deepcopy(remaining_stages)]
# #         return {
# #             "mode": "local_reschedule",
# #             "stages": new_stages,
# #             "actions": modified_actions,
# #             "decision": decision,
# #             "recovery": recovery,
# #         }

# #     replacement_stages, replacement_actions = build_replacement_subplan(
# #         failed_stage=failed_stage,
# #         actions=failed_actions,
# #         next_stage_id=next_stage_id,
# #     )
# #     recovery = {
# #         "parent_attempt_id": parent_attempt_id,
# #         "recovery_type": "replace",
# #         "attempt_number": next_attempt_number,
# #         "changes": {"replacement_stage_count": len(replacement_stages)},
# #     }
# #     for stage in replacement_stages:
# #         stage["_recovery"] = deepcopy(recovery)
# #     if replacement_stages:
# #         replacement_stages[0]["_actions"] = [replacement_actions[0]]
# #     if len(replacement_stages) > 1:
# #         replacement_stages[1]["_actions"] = replacement_actions[1:]

# #     return {
# #         "mode": "local_reschedule",
# #         "stages": [*replacement_stages, *deepcopy(remaining_stages)],
# #         "actions": replacement_actions,
# #         "decision": decision,
# #         "recovery": recovery,
# #     }


# # def check_recovery_limits(
# #     *,
# #     limits: dict[str, int],
# #     counters: dict[str, int],
# # ) -> None:
# #     for key, maximum in limits.items():
# #         if maximum < 0:
# #             raise ValueError(f"{key} must be >= 0")
# #     if counters.get("total_actions", 0) > limits["max_total_actions"]:
# #         raise RuntimeError("Maximum total action budget exceeded.")
