from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

ATTEMPT_TERMINAL_STATUSES = {
    "closed_success",
    "closed_failure",
    "closed_not_executed",
}

ATTEMPT_STATUSES = {
    "open",
    "awaiting_pre_validation",
    "preconditions_satisfied",
    "awaiting_pre_evidence",
    "executing",
    "awaiting_post_validation",
    "postconditions_satisfied",
    "postconditions_violated",
    "awaiting_post_evidence",
    *ATTEMPT_TERMINAL_STATUSES,
}

ALLOWED_ATTEMPT_TRANSITIONS: dict[str, set[str]] = {
    "open": {"awaiting_pre_validation"},
    "awaiting_pre_validation": {
        "preconditions_satisfied",
        "awaiting_pre_evidence",
        "closed_not_executed",
    },
    "awaiting_pre_evidence": {
        "awaiting_pre_validation",
        "closed_not_executed",
    },
    "preconditions_satisfied": {"executing"},
    "executing": {"awaiting_post_validation", "closed_failure"},
    "awaiting_post_validation": {
        "postconditions_satisfied",
        "postconditions_violated",
        "awaiting_post_evidence",
        "closed_failure",
    },
    "awaiting_post_evidence": {
        "awaiting_post_validation",
        "closed_failure",
    },
    "postconditions_satisfied": {"closed_success"},
    "postconditions_violated": {"closed_failure"},
    "closed_success": set(),
    "closed_failure": set(),
    "closed_not_executed": set(),
}

CONDITION_STATUSES = {"satisfied", "violated", "uncertain"}
VALIDATION_PHASES = {"pre", "post"}
VALIDATION_SCHEMA_VERSION = "1.0"


def utcish_now() -> str:
    return datetime.now().isoformat()


def is_attempt_closed(attempt: dict[str, Any]) -> bool:
    return attempt.get("status") in ATTEMPT_TERMINAL_STATUSES


def assert_attempt_invariants(attempt: dict[str, Any]) -> None:
    status = attempt.get("status")
    if status not in ATTEMPT_STATUSES:
        raise ValueError(f"Invalid attempt status: {status!r}")

    execution = attempt.get("execution")
    pre = attempt.get("pre")
    post = attempt.get("post")
    if not isinstance(execution, dict) or not isinstance(pre, dict) or not isinstance(post, dict):
        raise ValueError("Attempt must contain pre, execution, and post objects.")

    if execution.get("completed") and not execution.get("started"):
        raise ValueError("Execution cannot be completed before it has started.")

    if post.get("image_path") is not None and not execution.get("completed"):
        raise ValueError("I_post cannot exist before execution is completed.")

    closed_at = attempt.get("closed_at")
    if status in ATTEMPT_TERMINAL_STATUSES:
        if not closed_at:
            raise ValueError("A terminal attempt must have closed_at.")
        if status == "closed_success" and attempt.get("failure_report") is not None:
            raise ValueError("closed_success cannot contain a failure_report.")
        if status in {"closed_failure", "closed_not_executed"} and not isinstance(
            attempt.get("failure_report"), dict
        ):
            raise ValueError(f"{status} requires a failure_report.")
    elif closed_at is not None:
        raise ValueError("A non-terminal attempt cannot have closed_at.")


def transition_attempt(attempt: dict[str, Any], new_status: str) -> None:
    old_status = attempt.get("status")
    if old_status not in ATTEMPT_STATUSES:
        raise ValueError(f"Invalid current attempt status: {old_status!r}")
    if new_status not in ATTEMPT_STATUSES:
        raise ValueError(f"Invalid target attempt status: {new_status!r}")
    if old_status in ATTEMPT_TERMINAL_STATUSES:
        raise ValueError(f"Attempt {attempt.get('attempt_id')} is already closed.")
    if new_status not in ALLOWED_ATTEMPT_TRANSITIONS[old_status]:
        raise ValueError(
            f"Invalid attempt transition for {attempt.get('attempt_id')}: "
            f"{old_status} -> {new_status}."
        )

    attempt["status"] = new_status
    attempt.setdefault("status_history", []).append(
        {"from": old_status, "to": new_status, "timestamp": utcish_now()}
    )
    assert_attempt_invariants(attempt)


def close_attempt_state(
    attempt: dict[str, Any],
    status: str,
    failure_report: dict[str, Any] | None = None,
) -> None:
    if status not in ATTEMPT_TERMINAL_STATUSES:
        raise ValueError(f"Invalid terminal attempt status: {status!r}")
    if status == "closed_success" and failure_report is not None:
        raise ValueError("closed_success cannot contain a failure_report.")
    if status != "closed_success" and not isinstance(failure_report, dict):
        raise ValueError(f"{status} requires a failure_report.")

    old_status = attempt.get("status")
    if old_status not in ATTEMPT_STATUSES:
        raise ValueError(f"Invalid current attempt status: {old_status!r}")
    if old_status in ATTEMPT_TERMINAL_STATUSES:
        raise ValueError(f"Attempt {attempt.get('attempt_id')} is already closed.")
    if status not in ALLOWED_ATTEMPT_TRANSITIONS[old_status]:
        raise ValueError(
            f"Invalid attempt transition for {attempt.get('attempt_id')}: "
            f"{old_status} -> {status}."
        )

    closed_at = utcish_now()
    attempt["status"] = status
    attempt.setdefault("status_history", []).append(
        {"from": old_status, "to": status, "timestamp": closed_at}
    )
    attempt["outcome"] = {
        "closed_success": "success",
        "closed_failure": "failure",
        "closed_not_executed": "not_executed",
    }[status]
    attempt["closed_at"] = closed_at
    attempt["failure_report"] = deepcopy(failure_report)
    assert_attempt_invariants(attempt)


def compute_overall_status(results: list[dict[str, Any]]) -> str:
    statuses = [item.get("status") for item in results]
    if "violated" in statuses:
        return "violated"
    if "uncertain" in statuses:
        return "uncertain"
    return "satisfied"


def normalize_validation_result(
    raw_response: Any,
    expected_conditions: list[str],
    phase: str,
    *,
    evidence_used: list[dict[str, Any]] | None = None,
    validator_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in VALIDATION_PHASES:
        raise ValueError(f"Validation phase must be one of {sorted(VALIDATION_PHASES)}.")
    if not isinstance(raw_response, dict):
        raise ValueError("Validator output must be a JSON object.")

    results = raw_response.get("results")
    if not isinstance(results, list):
        raise ValueError("Validator field 'results' must be a list.")
    if len(results) != len(expected_conditions):
        raise ValueError(
            "Validator returned a different number of results than expected conditions."
        )

    normalized_results: list[dict[str, Any]] = []
    for idx, (item, expected_condition) in enumerate(zip(results, expected_conditions)):
        if not isinstance(item, dict):
            raise ValueError(f"Validator result at index {idx} must be an object.")
        if item.get("condition") != expected_condition:
            raise ValueError(
                f"Validator result at index {idx} does not preserve condition text/order."
            )
        status = item.get("status")
        if status not in CONDITION_STATUSES:
            raise ValueError(f"Validator result at index {idx} has invalid status: {status!r}.")
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Validator result at index {idx} has invalid reason.")
        normalized_results.append(
            {
                "condition": expected_condition,
                "status": status,
                "reason": reason.strip(),
            }
        )

    computed = compute_overall_status(normalized_results)
    declared = raw_response.get("overall_status")
    if declared is not None and declared not in CONDITION_STATUSES:
        raise ValueError(f"Validator field 'overall_status' has invalid value: {declared!r}.")
    if declared is not None and declared != computed:
        raise ValueError(
            f"Inconsistent overall_status: expected {computed!r}, got {declared!r}."
        )

    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "validation_phase": phase,
        "overall_status": computed,
        "results": normalized_results,
        "evidence_used": deepcopy(evidence_used or []),
        "validator_metadata": deepcopy(validator_metadata or {}),
        "normalized_at": utcish_now(),
    }







# from __future__ import annotations

# from copy import deepcopy
# from datetime import datetime
# from typing import Any

# ATTEMPT_TERMINAL_STATUSES = {
#     "closed_success",
#     "closed_failure",
#     "closed_not_executed",
# }

# ATTEMPT_STATUSES = {
#     "open",
#     "awaiting_pre_validation",
#     "preconditions_satisfied",
#     "awaiting_pre_evidence",
#     "executing",
#     "awaiting_post_validation",
#     "postconditions_satisfied",
#     "postconditions_violated",
#     "awaiting_post_evidence",
#     *ATTEMPT_TERMINAL_STATUSES,
# }

# ALLOWED_ATTEMPT_TRANSITIONS: dict[str, set[str]] = {
#     "open": {"awaiting_pre_validation"},
#     "awaiting_pre_validation": {
#         "preconditions_satisfied",
#         "awaiting_pre_evidence",
#         "closed_not_executed",
#     },
#     "awaiting_pre_evidence": {
#         "awaiting_pre_validation",
#         "closed_not_executed",
#     },
#     "preconditions_satisfied": {"executing"},
#     "executing": {"awaiting_post_validation", "closed_failure"},
#     "awaiting_post_validation": {
#         "postconditions_satisfied",
#         "postconditions_violated",
#         "awaiting_post_evidence",
#         "closed_failure",
#     },
#     "awaiting_post_evidence": {
#         "awaiting_post_validation",
#         "closed_failure",
#     },
#     "postconditions_satisfied": {"closed_success"},
#     "postconditions_violated": {"closed_failure"},
#     "closed_success": set(),
#     "closed_failure": set(),
#     "closed_not_executed": set(),
# }

# CONDITION_STATUSES = {"satisfied", "violated", "uncertain"}
# VALIDATION_PHASES = {"pre", "post"}
# VALIDATION_SCHEMA_VERSION = "1.0"


# def utcish_now() -> str:
#     return datetime.now().isoformat()


# def is_attempt_closed(attempt: dict[str, Any]) -> bool:
#     return attempt.get("status") in ATTEMPT_TERMINAL_STATUSES


# def assert_attempt_invariants(attempt: dict[str, Any]) -> None:
#     status = attempt.get("status")
#     if status not in ATTEMPT_STATUSES:
#         raise ValueError(f"Invalid attempt status: {status!r}")

#     execution = attempt.get("execution")
#     pre = attempt.get("pre")
#     post = attempt.get("post")
#     if not isinstance(execution, dict) or not isinstance(pre, dict) or not isinstance(post, dict):
#         raise ValueError("Attempt must contain pre, execution, and post objects.")

#     if execution.get("completed") and not execution.get("started"):
#         raise ValueError("Execution cannot be completed before it has started.")

#     if post.get("image_path") is not None and not execution.get("completed"):
#         raise ValueError("I_post cannot exist before execution is completed.")

#     closed_at = attempt.get("closed_at")
#     if status in ATTEMPT_TERMINAL_STATUSES:
#         if not closed_at:
#             raise ValueError("A terminal attempt must have closed_at.")
#         if status == "closed_success" and attempt.get("failure_report") is not None:
#             raise ValueError("closed_success cannot contain a failure_report.")
#         if status in {"closed_failure", "closed_not_executed"} and not isinstance(
#             attempt.get("failure_report"), dict
#         ):
#             raise ValueError(f"{status} requires a failure_report.")
#     elif closed_at is not None:
#         raise ValueError("A non-terminal attempt cannot have closed_at.")


# def transition_attempt(attempt: dict[str, Any], new_status: str) -> None:
#     old_status = attempt.get("status")
#     if old_status not in ATTEMPT_STATUSES:
#         raise ValueError(f"Invalid current attempt status: {old_status!r}")
#     if new_status not in ATTEMPT_STATUSES:
#         raise ValueError(f"Invalid target attempt status: {new_status!r}")
#     if old_status in ATTEMPT_TERMINAL_STATUSES:
#         raise ValueError(f"Attempt {attempt.get('attempt_id')} is already closed.")
#     if new_status not in ALLOWED_ATTEMPT_TRANSITIONS[old_status]:
#         raise ValueError(
#             f"Invalid attempt transition for {attempt.get('attempt_id')}: "
#             f"{old_status} -> {new_status}."
#         )

#     attempt["status"] = new_status
#     attempt.setdefault("status_history", []).append(
#         {"from": old_status, "to": new_status, "timestamp": utcish_now()}
#     )
#     assert_attempt_invariants(attempt)


# def close_attempt_state(
#     attempt: dict[str, Any],
#     status: str,
#     failure_report: dict[str, Any] | None = None,
# ) -> None:
#     if status not in ATTEMPT_TERMINAL_STATUSES:
#         raise ValueError(f"Invalid terminal attempt status: {status!r}")
#     if status == "closed_success" and failure_report is not None:
#         raise ValueError("closed_success cannot contain a failure_report.")
#     if status != "closed_success" and not isinstance(failure_report, dict):
#         raise ValueError(f"{status} requires a failure_report.")

#     old_status = attempt.get("status")
#     if old_status not in ATTEMPT_STATUSES:
#         raise ValueError(f"Invalid current attempt status: {old_status!r}")
#     if old_status in ATTEMPT_TERMINAL_STATUSES:
#         raise ValueError(f"Attempt {attempt.get('attempt_id')} is already closed.")
#     if status not in ALLOWED_ATTEMPT_TRANSITIONS[old_status]:
#         raise ValueError(
#             f"Invalid attempt transition for {attempt.get('attempt_id')}: "
#             f"{old_status} -> {status}."
#         )

#     closed_at = utcish_now()
#     attempt["status"] = status
#     attempt.setdefault("status_history", []).append(
#         {"from": old_status, "to": status, "timestamp": closed_at}
#     )
#     attempt["outcome"] = {
#         "closed_success": "success",
#         "closed_failure": "failure",
#         "closed_not_executed": "not_executed",
#     }[status]
#     attempt["closed_at"] = closed_at
#     attempt["failure_report"] = deepcopy(failure_report)
#     assert_attempt_invariants(attempt)


# def compute_overall_status(results: list[dict[str, Any]]) -> str:
#     statuses = [item.get("status") for item in results]
#     if "violated" in statuses:
#         return "violated"
#     if "uncertain" in statuses:
#         return "uncertain"
#     return "satisfied"


# def normalize_validation_result(
#     raw_response: Any,
#     expected_conditions: list[str],
#     phase: str,
#     *,
#     evidence_used: list[dict[str, Any]] | None = None,
#     validator_metadata: dict[str, Any] | None = None,
# ) -> dict[str, Any]:
#     if phase not in VALIDATION_PHASES:
#         raise ValueError(f"Validation phase must be one of {sorted(VALIDATION_PHASES)}.")
#     if not isinstance(raw_response, dict):
#         raise ValueError("Validator output must be a JSON object.")

#     results = raw_response.get("results")
#     if not isinstance(results, list):
#         raise ValueError("Validator field 'results' must be a list.")
#     if len(results) != len(expected_conditions):
#         raise ValueError(
#             "Validator returned a different number of results than expected conditions."
#         )

#     normalized_results: list[dict[str, Any]] = []
#     for idx, (item, expected_condition) in enumerate(zip(results, expected_conditions)):
#         if not isinstance(item, dict):
#             raise ValueError(f"Validator result at index {idx} must be an object.")
#         if item.get("condition") != expected_condition:
#             raise ValueError(
#                 f"Validator result at index {idx} does not preserve condition text/order."
#             )
#         status = item.get("status")
#         if status not in CONDITION_STATUSES:
#             raise ValueError(f"Validator result at index {idx} has invalid status: {status!r}.")
#         reason = item.get("reason")
#         if not isinstance(reason, str) or not reason.strip():
#             raise ValueError(f"Validator result at index {idx} has invalid reason.")
#         normalized_results.append(
#             {
#                 "condition": expected_condition,
#                 "status": status,
#                 "reason": reason.strip(),
#             }
#         )

#     computed = compute_overall_status(normalized_results)
#     declared = raw_response.get("overall_status")
#     if declared is not None and declared not in CONDITION_STATUSES:
#         raise ValueError(f"Validator field 'overall_status' has invalid value: {declared!r}.")
#     if declared is not None and declared != computed:
#         raise ValueError(
#             f"Inconsistent overall_status: expected {computed!r}, got {declared!r}."
#         )

#     return {
#         "schema_version": VALIDATION_SCHEMA_VERSION,
#         "validation_phase": phase,
#         "overall_status": computed,
#         "results": normalized_results,
#         "evidence_used": deepcopy(evidence_used or []),
#         "validator_metadata": deepcopy(validator_metadata or {}),
#         "normalized_at": utcish_now(),
#     }







# # from __future__ import annotations

# # from copy import deepcopy
# # from datetime import datetime
# # from typing import Any

# # ATTEMPT_TERMINAL_STATUSES = {
# #     "closed_success",
# #     "closed_failure",
# #     "closed_not_executed",
# # }

# # ATTEMPT_STATUSES = {
# #     "open",
# #     "awaiting_pre_validation",
# #     "preconditions_satisfied",
# #     "awaiting_pre_evidence",
# #     "executing",
# #     "awaiting_post_validation",
# #     "postconditions_satisfied",
# #     "postconditions_violated",
# #     "awaiting_post_evidence",
# #     *ATTEMPT_TERMINAL_STATUSES,
# # }

# # ALLOWED_ATTEMPT_TRANSITIONS: dict[str, set[str]] = {
# #     "open": {"awaiting_pre_validation"},
# #     "awaiting_pre_validation": {
# #         "preconditions_satisfied",
# #         "awaiting_pre_evidence",
# #         "closed_not_executed",
# #     },
# #     "awaiting_pre_evidence": {
# #         "awaiting_pre_validation",
# #         "closed_not_executed",
# #     },
# #     "preconditions_satisfied": {"executing"},
# #     "executing": {"awaiting_post_validation", "closed_failure"},
# #     "awaiting_post_validation": {
# #         "postconditions_satisfied",
# #         "postconditions_violated",
# #         "awaiting_post_evidence",
# #         "closed_failure",
# #     },
# #     "awaiting_post_evidence": {
# #         "awaiting_post_validation",
# #         "closed_failure",
# #     },
# #     "postconditions_satisfied": {"closed_success"},
# #     "postconditions_violated": {"closed_failure"},
# #     "closed_success": set(),
# #     "closed_failure": set(),
# #     "closed_not_executed": set(),
# # }

# # CONDITION_STATUSES = {"satisfied", "violated", "uncertain"}
# # VALIDATION_PHASES = {"pre", "post"}
# # VALIDATION_SCHEMA_VERSION = "1.0"


# # def utcish_now() -> str:
# #     return datetime.now().isoformat()


# # def is_attempt_closed(attempt: dict[str, Any]) -> bool:
# #     return attempt.get("status") in ATTEMPT_TERMINAL_STATUSES


# # def assert_attempt_invariants(attempt: dict[str, Any]) -> None:
# #     status = attempt.get("status")
# #     if status not in ATTEMPT_STATUSES:
# #         raise ValueError(f"Invalid attempt status: {status!r}")

# #     execution = attempt.get("execution")
# #     pre = attempt.get("pre")
# #     post = attempt.get("post")
# #     if not isinstance(execution, dict) or not isinstance(pre, dict) or not isinstance(post, dict):
# #         raise ValueError("Attempt must contain pre, execution, and post objects.")

# #     if execution.get("completed") and not execution.get("started"):
# #         raise ValueError("Execution cannot be completed before it has started.")

# #     if post.get("image_path") is not None and not execution.get("completed"):
# #         raise ValueError("I_post cannot exist before execution is completed.")

# #     closed_at = attempt.get("closed_at")
# #     if status in ATTEMPT_TERMINAL_STATUSES:
# #         if not closed_at:
# #             raise ValueError("A terminal attempt must have closed_at.")
# #         if status == "closed_success" and attempt.get("failure_report") is not None:
# #             raise ValueError("closed_success cannot contain a failure_report.")
# #         if status in {"closed_failure", "closed_not_executed"} and not isinstance(
# #             attempt.get("failure_report"), dict
# #         ):
# #             raise ValueError(f"{status} requires a failure_report.")
# #     elif closed_at is not None:
# #         raise ValueError("A non-terminal attempt cannot have closed_at.")


# # def transition_attempt(attempt: dict[str, Any], new_status: str) -> None:
# #     old_status = attempt.get("status")
# #     if old_status not in ATTEMPT_STATUSES:
# #         raise ValueError(f"Invalid current attempt status: {old_status!r}")
# #     if new_status not in ATTEMPT_STATUSES:
# #         raise ValueError(f"Invalid target attempt status: {new_status!r}")
# #     if old_status in ATTEMPT_TERMINAL_STATUSES:
# #         raise ValueError(f"Attempt {attempt.get('attempt_id')} is already closed.")
# #     if new_status not in ALLOWED_ATTEMPT_TRANSITIONS[old_status]:
# #         raise ValueError(
# #             f"Invalid attempt transition for {attempt.get('attempt_id')}: "
# #             f"{old_status} -> {new_status}."
# #         )

# #     attempt["status"] = new_status
# #     attempt.setdefault("status_history", []).append(
# #         {"from": old_status, "to": new_status, "timestamp": utcish_now()}
# #     )
# #     assert_attempt_invariants(attempt)


# # def close_attempt_state(
# #     attempt: dict[str, Any],
# #     status: str,
# #     failure_report: dict[str, Any] | None = None,
# # ) -> None:
# #     if status not in ATTEMPT_TERMINAL_STATUSES:
# #         raise ValueError(f"Invalid terminal attempt status: {status!r}")
# #     if status == "closed_success" and failure_report is not None:
# #         raise ValueError("closed_success cannot contain a failure_report.")
# #     if status != "closed_success" and not isinstance(failure_report, dict):
# #         raise ValueError(f"{status} requires a failure_report.")

# #     old_status = attempt.get("status")
# #     if old_status not in ATTEMPT_STATUSES:
# #         raise ValueError(f"Invalid current attempt status: {old_status!r}")
# #     if old_status in ATTEMPT_TERMINAL_STATUSES:
# #         raise ValueError(f"Attempt {attempt.get('attempt_id')} is already closed.")
# #     if status not in ALLOWED_ATTEMPT_TRANSITIONS[old_status]:
# #         raise ValueError(
# #             f"Invalid attempt transition for {attempt.get('attempt_id')}: "
# #             f"{old_status} -> {status}."
# #         )

# #     closed_at = utcish_now()
# #     attempt["status"] = status
# #     attempt.setdefault("status_history", []).append(
# #         {"from": old_status, "to": status, "timestamp": closed_at}
# #     )
# #     attempt["outcome"] = {
# #         "closed_success": "success",
# #         "closed_failure": "failure",
# #         "closed_not_executed": "not_executed",
# #     }[status]
# #     attempt["closed_at"] = closed_at
# #     attempt["failure_report"] = deepcopy(failure_report)
# #     assert_attempt_invariants(attempt)


# # def compute_overall_status(results: list[dict[str, Any]]) -> str:
# #     statuses = [item.get("status") for item in results]
# #     if "violated" in statuses:
# #         return "violated"
# #     if "uncertain" in statuses:
# #         return "uncertain"
# #     return "satisfied"


# # def normalize_validation_result(
# #     raw_response: Any,
# #     expected_conditions: list[str],
# #     phase: str,
# #     *,
# #     evidence_used: list[dict[str, Any]] | None = None,
# #     validator_metadata: dict[str, Any] | None = None,
# # ) -> dict[str, Any]:
# #     if phase not in VALIDATION_PHASES:
# #         raise ValueError(f"Validation phase must be one of {sorted(VALIDATION_PHASES)}.")
# #     if not isinstance(raw_response, dict):
# #         raise ValueError("Validator output must be a JSON object.")

# #     results = raw_response.get("results")
# #     if not isinstance(results, list):
# #         raise ValueError("Validator field 'results' must be a list.")
# #     if len(results) != len(expected_conditions):
# #         raise ValueError(
# #             "Validator returned a different number of results than expected conditions."
# #         )

# #     normalized_results: list[dict[str, Any]] = []
# #     for idx, (item, expected_condition) in enumerate(zip(results, expected_conditions)):
# #         if not isinstance(item, dict):
# #             raise ValueError(f"Validator result at index {idx} must be an object.")
# #         if item.get("condition") != expected_condition:
# #             raise ValueError(
# #                 f"Validator result at index {idx} does not preserve condition text/order."
# #             )
# #         status = item.get("status")
# #         if status not in CONDITION_STATUSES:
# #             raise ValueError(f"Validator result at index {idx} has invalid status: {status!r}.")
# #         reason = item.get("reason")
# #         if not isinstance(reason, str) or not reason.strip():
# #             raise ValueError(f"Validator result at index {idx} has invalid reason.")
# #         normalized_results.append(
# #             {
# #                 "condition": expected_condition,
# #                 "status": status,
# #                 "reason": reason.strip(),
# #             }
# #         )

# #     computed = compute_overall_status(normalized_results)
# #     declared = raw_response.get("overall_status")
# #     if declared is not None and declared not in CONDITION_STATUSES:
# #         raise ValueError(f"Validator field 'overall_status' has invalid value: {declared!r}.")
# #     if declared is not None and declared != computed:
# #         raise ValueError(
# #             f"Inconsistent overall_status: expected {computed!r}, got {declared!r}."
# #         )

# #     return {
# #         "schema_version": VALIDATION_SCHEMA_VERSION,
# #         "validation_phase": phase,
# #         "overall_status": computed,
# #         "results": normalized_results,
# #         "evidence_used": deepcopy(evidence_used or []),
# #         "validator_metadata": deepcopy(validator_metadata or {}),
# #         "normalized_at": utcish_now(),
# #     }




# # # from __future__ import annotations

# # # from copy import deepcopy
# # # from datetime import datetime
# # # from typing import Any

# # # ATTEMPT_TERMINAL_STATUSES = {
# # #     "closed_success",
# # #     "closed_failure",
# # #     "closed_not_executed",
# # # }

# # # ATTEMPT_STATUSES = {
# # #     "open",
# # #     "awaiting_pre_validation",
# # #     "preconditions_satisfied",
# # #     "awaiting_pre_evidence",
# # #     "executing",
# # #     "awaiting_post_validation",
# # #     "postconditions_satisfied",
# # #     "postconditions_violated",
# # #     "awaiting_post_evidence",
# # #     *ATTEMPT_TERMINAL_STATUSES,
# # # }

# # # ALLOWED_ATTEMPT_TRANSITIONS: dict[str, set[str]] = {
# # #     "open": {"awaiting_pre_validation"},
# # #     "awaiting_pre_validation": {
# # #         "preconditions_satisfied",
# # #         "awaiting_pre_evidence",
# # #         "closed_not_executed",
# # #     },
# # #     "awaiting_pre_evidence": {
# # #         "awaiting_pre_validation",
# # #         "closed_not_executed",
# # #     },
# # #     "preconditions_satisfied": {"executing"},
# # #     "executing": {"awaiting_post_validation", "closed_failure"},
# # #     "awaiting_post_validation": {
# # #         "postconditions_satisfied",
# # #         "postconditions_violated",
# # #         "awaiting_post_evidence",
# # #         "closed_failure",
# # #     },
# # #     "awaiting_post_evidence": {
# # #         "awaiting_post_validation",
# # #         "closed_failure",
# # #     },
# # #     "postconditions_satisfied": {"closed_success"},
# # #     "postconditions_violated": {"closed_failure"},
# # #     "closed_success": set(),
# # #     "closed_failure": set(),
# # #     "closed_not_executed": set(),
# # # }

# # # CONDITION_STATUSES = {"satisfied", "violated", "uncertain"}
# # # VALIDATION_PHASES = {"pre", "post"}
# # # VALIDATION_SCHEMA_VERSION = "1.0"


# # # def utcish_now() -> str:
# # #     return datetime.now().isoformat()


# # # def is_attempt_closed(attempt: dict[str, Any]) -> bool:
# # #     return attempt.get("status") in ATTEMPT_TERMINAL_STATUSES


# # # def assert_attempt_invariants(attempt: dict[str, Any]) -> None:
# # #     status = attempt.get("status")
# # #     if status not in ATTEMPT_STATUSES:
# # #         raise ValueError(f"Invalid attempt status: {status!r}")

# # #     execution = attempt.get("execution")
# # #     pre = attempt.get("pre")
# # #     post = attempt.get("post")
# # #     if not isinstance(execution, dict) or not isinstance(pre, dict) or not isinstance(post, dict):
# # #         raise ValueError("Attempt must contain pre, execution, and post objects.")

# # #     if execution.get("completed") and not execution.get("started"):
# # #         raise ValueError("Execution cannot be completed before it has started.")

# # #     if post.get("image_path") is not None and not execution.get("completed"):
# # #         raise ValueError("I_post cannot exist before execution is completed.")

# # #     closed_at = attempt.get("closed_at")
# # #     if status in ATTEMPT_TERMINAL_STATUSES:
# # #         if not closed_at:
# # #             raise ValueError("A terminal attempt must have closed_at.")
# # #         if status == "closed_success" and attempt.get("failure_report") is not None:
# # #             raise ValueError("closed_success cannot contain a failure_report.")
# # #         if status in {"closed_failure", "closed_not_executed"} and not isinstance(
# # #             attempt.get("failure_report"), dict
# # #         ):
# # #             raise ValueError(f"{status} requires a failure_report.")
# # #     elif closed_at is not None:
# # #         raise ValueError("A non-terminal attempt cannot have closed_at.")


# # # def transition_attempt(attempt: dict[str, Any], new_status: str) -> None:
# # #     old_status = attempt.get("status")
# # #     if old_status not in ATTEMPT_STATUSES:
# # #         raise ValueError(f"Invalid current attempt status: {old_status!r}")
# # #     if new_status not in ATTEMPT_STATUSES:
# # #         raise ValueError(f"Invalid target attempt status: {new_status!r}")
# # #     if old_status in ATTEMPT_TERMINAL_STATUSES:
# # #         raise ValueError(f"Attempt {attempt.get('attempt_id')} is already closed.")
# # #     if new_status not in ALLOWED_ATTEMPT_TRANSITIONS[old_status]:
# # #         raise ValueError(
# # #             f"Invalid attempt transition for {attempt.get('attempt_id')}: "
# # #             f"{old_status} -> {new_status}."
# # #         )

# # #     attempt["status"] = new_status
# # #     attempt.setdefault("status_history", []).append(
# # #         {"from": old_status, "to": new_status, "timestamp": utcish_now()}
# # #     )
# # #     assert_attempt_invariants(attempt)


# # # def close_attempt_state(
# # #     attempt: dict[str, Any],
# # #     status: str,
# # #     failure_report: dict[str, Any] | None = None,
# # # ) -> None:
# # #     if status not in ATTEMPT_TERMINAL_STATUSES:
# # #         raise ValueError(f"Invalid terminal attempt status: {status!r}")
# # #     if status == "closed_success" and failure_report is not None:
# # #         raise ValueError("closed_success cannot contain a failure_report.")
# # #     if status != "closed_success" and not isinstance(failure_report, dict):
# # #         raise ValueError(f"{status} requires a failure_report.")

# # #     old_status = attempt.get("status")
# # #     if old_status not in ATTEMPT_STATUSES:
# # #         raise ValueError(f"Invalid current attempt status: {old_status!r}")
# # #     if old_status in ATTEMPT_TERMINAL_STATUSES:
# # #         raise ValueError(f"Attempt {attempt.get('attempt_id')} is already closed.")
# # #     if status not in ALLOWED_ATTEMPT_TRANSITIONS[old_status]:
# # #         raise ValueError(
# # #             f"Invalid attempt transition for {attempt.get('attempt_id')}: "
# # #             f"{old_status} -> {status}."
# # #         )

# # #     closed_at = utcish_now()
# # #     attempt["status"] = status
# # #     attempt.setdefault("status_history", []).append(
# # #         {"from": old_status, "to": status, "timestamp": closed_at}
# # #     )
# # #     attempt["outcome"] = {
# # #         "closed_success": "success",
# # #         "closed_failure": "failure",
# # #         "closed_not_executed": "not_executed",
# # #     }[status]
# # #     attempt["closed_at"] = closed_at
# # #     attempt["failure_report"] = deepcopy(failure_report)
# # #     assert_attempt_invariants(attempt)


# # # def compute_overall_status(results: list[dict[str, Any]]) -> str:
# # #     statuses = [item.get("status") for item in results]
# # #     if "violated" in statuses:
# # #         return "violated"
# # #     if "uncertain" in statuses:
# # #         return "uncertain"
# # #     return "satisfied"


# # # def normalize_validation_result(
# # #     raw_response: Any,
# # #     expected_conditions: list[str],
# # #     phase: str,
# # #     *,
# # #     evidence_used: list[dict[str, Any]] | None = None,
# # #     validator_metadata: dict[str, Any] | None = None,
# # # ) -> dict[str, Any]:
# # #     if phase not in VALIDATION_PHASES:
# # #         raise ValueError(f"Validation phase must be one of {sorted(VALIDATION_PHASES)}.")
# # #     if not isinstance(raw_response, dict):
# # #         raise ValueError("Validator output must be a JSON object.")

# # #     results = raw_response.get("results")
# # #     if not isinstance(results, list):
# # #         raise ValueError("Validator field 'results' must be a list.")
# # #     if len(results) != len(expected_conditions):
# # #         raise ValueError(
# # #             "Validator returned a different number of results than expected conditions."
# # #         )

# # #     normalized_results: list[dict[str, Any]] = []
# # #     for idx, (item, expected_condition) in enumerate(zip(results, expected_conditions)):
# # #         if not isinstance(item, dict):
# # #             raise ValueError(f"Validator result at index {idx} must be an object.")
# # #         if item.get("condition") != expected_condition:
# # #             raise ValueError(
# # #                 f"Validator result at index {idx} does not preserve condition text/order."
# # #             )
# # #         status = item.get("status")
# # #         if status not in CONDITION_STATUSES:
# # #             raise ValueError(f"Validator result at index {idx} has invalid status: {status!r}.")
# # #         reason = item.get("reason")
# # #         if not isinstance(reason, str) or not reason.strip():
# # #             raise ValueError(f"Validator result at index {idx} has invalid reason.")
# # #         normalized_results.append(
# # #             {
# # #                 "condition": expected_condition,
# # #                 "status": status,
# # #                 "reason": reason.strip(),
# # #             }
# # #         )

# # #     computed = compute_overall_status(normalized_results)
# # #     declared = raw_response.get("overall_status")
# # #     if declared is not None and declared not in CONDITION_STATUSES:
# # #         raise ValueError(f"Validator field 'overall_status' has invalid value: {declared!r}.")
# # #     if declared is not None and declared != computed:
# # #         raise ValueError(
# # #             f"Inconsistent overall_status: expected {computed!r}, got {declared!r}."
# # #         )

# # #     return {
# # #         "schema_version": VALIDATION_SCHEMA_VERSION,
# # #         "validation_phase": phase,
# # #         "overall_status": computed,
# # #         "results": normalized_results,
# # #         "evidence_used": deepcopy(evidence_used or []),
# # #         "validator_metadata": deepcopy(validator_metadata or {}),
# # #         "normalized_at": utcish_now(),
# # #     }
