from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

FAILURE_REPORT_SCHEMA_VERSION = "1.0"
FAILURE_PHASES = {"pre", "execution", "post"}
FAILURE_TYPES = {
    "precondition_failure",
    "execution_failure",
    "postcondition_failure",
    "insufficient_evidence",
    "technical_error",
}


def now_iso() -> str:
    return datetime.now().isoformat()


def _copy_dict(value: Any) -> dict[str, Any]:
    return deepcopy(value) if isinstance(value, dict) else {}


def _copy_list(value: Any) -> list[Any]:
    return deepcopy(value) if isinstance(value, list) else []


def _select_results(
    validation: dict[str, Any] | None,
    status: str,
) -> list[dict[str, Any]]:
    if not isinstance(validation, dict):
        return []
    results = validation.get("results")
    if not isinstance(results, list):
        return []
    return [deepcopy(item) for item in results if isinstance(item, dict) and item.get("status") == status]


def assert_failure_report(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise ValueError("Failure report must be a dictionary.")
    if report.get("schema_version") != FAILURE_REPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing failure-report schema_version.")
    if not isinstance(report.get("attempt_id"), str) or not report["attempt_id"].strip():
        raise ValueError("Failure report requires a non-empty attempt_id.")
    if not isinstance(report.get("stage_id"), int):
        raise ValueError("Failure report requires an integer stage_id.")
    if report.get("failure_phase") not in FAILURE_PHASES:
        raise ValueError(f"Invalid failure_phase: {report.get('failure_phase')!r}.")
    if report.get("failure_type") not in FAILURE_TYPES:
        raise ValueError(f"Invalid failure_type: {report.get('failure_type')!r}.")

    for key in (
        "failed_conditions",
        "uncertain_conditions",
        "relevant_history",
        "evidence_rounds",
    ):
        if not isinstance(report.get(key), list):
            raise ValueError(f"Failure report field {key!r} must be a list.")

    for key in ("action", "scene_graph_before", "scene_graph_after", "technical_error"):
        if not isinstance(report.get(key), dict):
            raise ValueError(f"Failure report field {key!r} must be an object.")

    for key in ("i_pre", "i_post"):
        value = report.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Failure report field {key!r} must be a string or null.")

    if not isinstance(report.get("local_goal"), str):
        raise ValueError("Failure report local_goal must be a string.")
    if not isinstance(report.get("created_at"), str) or not report["created_at"]:
        raise ValueError("Failure report requires created_at.")

    failure_type = report["failure_type"]
    phase = report["failure_phase"]
    if failure_type == "precondition_failure" and phase != "pre":
        raise ValueError("precondition_failure must use failure_phase='pre'.")
    if failure_type == "postcondition_failure" and phase != "post":
        raise ValueError("postcondition_failure must use failure_phase='post'.")
    if failure_type == "execution_failure" and phase != "execution":
        raise ValueError("execution_failure must use failure_phase='execution'.")
    if failure_type == "insufficient_evidence" and not report["uncertain_conditions"]:
        raise ValueError("insufficient_evidence requires uncertain_conditions.")
    if failure_type == "technical_error" and not report["technical_error"].get("message"):
        raise ValueError("technical_error requires technical_error.message.")


def build_failure_report(
    *,
    attempt: dict[str, Any],
    failure_phase: str,
    failure_type: str,
    validation: dict[str, Any] | None = None,
    action: dict[str, Any] | list[dict[str, Any]] | None = None,
    scene_graph_before: dict[str, Any] | None = None,
    scene_graph_after: dict[str, Any] | None = None,
    relevant_history: list[dict[str, Any]] | None = None,
    evidence_rounds: list[dict[str, Any]] | None = None,
    technical_error: BaseException | dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if failure_phase not in FAILURE_PHASES:
        raise ValueError(f"Invalid failure_phase: {failure_phase!r}.")
    if failure_type not in FAILURE_TYPES:
        raise ValueError(f"Invalid failure_type: {failure_type!r}.")

    failed_conditions = _select_results(validation, "violated")
    uncertain_conditions = _select_results(validation, "uncertain")

    if action is None:
        normalized_action: dict[str, Any] = {}
    elif isinstance(action, dict):
        normalized_action = deepcopy(action)
    elif isinstance(action, list):
        normalized_action = {"steps": deepcopy(action)}
    else:
        raise ValueError("action must be a dictionary, a list of step dictionaries, or None.")

    if isinstance(technical_error, BaseException):
        technical_error_payload = {
            "exception_type": type(technical_error).__name__,
            "message": str(technical_error),
        }
    elif isinstance(technical_error, dict):
        technical_error_payload = deepcopy(technical_error)
    else:
        technical_error_payload = {}

    report = {
        "schema_version": FAILURE_REPORT_SCHEMA_VERSION,
        "attempt_id": str(attempt.get("attempt_id", "")),
        "stage_id": attempt.get("stage_id"),
        "failure_phase": failure_phase,
        "failure_type": failure_type,
        "failed_conditions": failed_conditions,
        "uncertain_conditions": uncertain_conditions,
        "i_pre": attempt.get("pre", {}).get("image_path"),
        "i_post": attempt.get("post", {}).get("image_path"),
        "local_goal": str(attempt.get("local_goal", "")),
        "action": normalized_action,
        "scene_graph_before": _copy_dict(scene_graph_before),
        "scene_graph_after": _copy_dict(scene_graph_after),
        "relevant_history": _copy_list(relevant_history),
        "evidence_rounds": _copy_list(evidence_rounds),
        "technical_error": technical_error_payload,
        "notes": notes,
        "created_at": now_iso(),
    }
    assert_failure_report(report)
    return report


def build_uncertainty_exhausted_report(
    *,
    attempt: dict[str, Any],
    phase: str,
    validation: dict[str, Any],
    action: dict[str, Any] | list[dict[str, Any]] | None = None,
    scene_graph_before: dict[str, Any] | None = None,
    scene_graph_after: dict[str, Any] | None = None,
    relevant_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    evidence_rounds = attempt.get(phase, {}).get("evidence_rounds", [])
    return build_failure_report(
        attempt=attempt,
        failure_phase=phase,
        failure_type="insufficient_evidence",
        validation=validation,
        action=action,
        scene_graph_before=scene_graph_before,
        scene_graph_after=scene_graph_after,
        relevant_history=relevant_history,
        evidence_rounds=evidence_rounds,
        notes="Maximum evidence rounds exhausted while one or more conditions remained uncertain.",
    )




# from __future__ import annotations

# from copy import deepcopy
# from datetime import datetime
# from typing import Any

# FAILURE_REPORT_SCHEMA_VERSION = "1.0"
# FAILURE_PHASES = {"pre", "execution", "post"}
# FAILURE_TYPES = {
#     "precondition_failure",
#     "execution_failure",
#     "postcondition_failure",
#     "insufficient_evidence",
#     "technical_error",
# }


# def now_iso() -> str:
#     return datetime.now().isoformat()


# def _copy_dict(value: Any) -> dict[str, Any]:
#     return deepcopy(value) if isinstance(value, dict) else {}


# def _copy_list(value: Any) -> list[Any]:
#     return deepcopy(value) if isinstance(value, list) else []


# def _select_results(
#     validation: dict[str, Any] | None,
#     status: str,
# ) -> list[dict[str, Any]]:
#     if not isinstance(validation, dict):
#         return []
#     results = validation.get("results")
#     if not isinstance(results, list):
#         return []
#     return [deepcopy(item) for item in results if isinstance(item, dict) and item.get("status") == status]


# def assert_failure_report(report: dict[str, Any]) -> None:
#     if not isinstance(report, dict):
#         raise ValueError("Failure report must be a dictionary.")
#     if report.get("schema_version") != FAILURE_REPORT_SCHEMA_VERSION:
#         raise ValueError("Unsupported or missing failure-report schema_version.")
#     if not isinstance(report.get("attempt_id"), str) or not report["attempt_id"].strip():
#         raise ValueError("Failure report requires a non-empty attempt_id.")
#     if not isinstance(report.get("stage_id"), int):
#         raise ValueError("Failure report requires an integer stage_id.")
#     if report.get("failure_phase") not in FAILURE_PHASES:
#         raise ValueError(f"Invalid failure_phase: {report.get('failure_phase')!r}.")
#     if report.get("failure_type") not in FAILURE_TYPES:
#         raise ValueError(f"Invalid failure_type: {report.get('failure_type')!r}.")

#     for key in (
#         "failed_conditions",
#         "uncertain_conditions",
#         "relevant_history",
#         "evidence_rounds",
#     ):
#         if not isinstance(report.get(key), list):
#             raise ValueError(f"Failure report field {key!r} must be a list.")

#     for key in ("action", "scene_graph_before", "scene_graph_after", "technical_error"):
#         if not isinstance(report.get(key), dict):
#             raise ValueError(f"Failure report field {key!r} must be an object.")

#     for key in ("i_pre", "i_post"):
#         value = report.get(key)
#         if value is not None and not isinstance(value, str):
#             raise ValueError(f"Failure report field {key!r} must be a string or null.")

#     if not isinstance(report.get("local_goal"), str):
#         raise ValueError("Failure report local_goal must be a string.")
#     if not isinstance(report.get("created_at"), str) or not report["created_at"]:
#         raise ValueError("Failure report requires created_at.")

#     failure_type = report["failure_type"]
#     phase = report["failure_phase"]
#     if failure_type == "precondition_failure" and phase != "pre":
#         raise ValueError("precondition_failure must use failure_phase='pre'.")
#     if failure_type == "postcondition_failure" and phase != "post":
#         raise ValueError("postcondition_failure must use failure_phase='post'.")
#     if failure_type == "execution_failure" and phase != "execution":
#         raise ValueError("execution_failure must use failure_phase='execution'.")
#     if failure_type == "insufficient_evidence" and not report["uncertain_conditions"]:
#         raise ValueError("insufficient_evidence requires uncertain_conditions.")
#     if failure_type == "technical_error" and not report["technical_error"].get("message"):
#         raise ValueError("technical_error requires technical_error.message.")


# def build_failure_report(
#     *,
#     attempt: dict[str, Any],
#     failure_phase: str,
#     failure_type: str,
#     validation: dict[str, Any] | None = None,
#     action: dict[str, Any] | list[dict[str, Any]] | None = None,
#     scene_graph_before: dict[str, Any] | None = None,
#     scene_graph_after: dict[str, Any] | None = None,
#     relevant_history: list[dict[str, Any]] | None = None,
#     evidence_rounds: list[dict[str, Any]] | None = None,
#     technical_error: BaseException | dict[str, Any] | None = None,
#     notes: str | None = None,
# ) -> dict[str, Any]:
#     if failure_phase not in FAILURE_PHASES:
#         raise ValueError(f"Invalid failure_phase: {failure_phase!r}.")
#     if failure_type not in FAILURE_TYPES:
#         raise ValueError(f"Invalid failure_type: {failure_type!r}.")

#     failed_conditions = _select_results(validation, "violated")
#     uncertain_conditions = _select_results(validation, "uncertain")

#     if action is None:
#         normalized_action: dict[str, Any] = {}
#     elif isinstance(action, dict):
#         normalized_action = deepcopy(action)
#     elif isinstance(action, list):
#         normalized_action = {"steps": deepcopy(action)}
#     else:
#         raise ValueError("action must be a dictionary, a list of step dictionaries, or None.")

#     if isinstance(technical_error, BaseException):
#         technical_error_payload = {
#             "exception_type": type(technical_error).__name__,
#             "message": str(technical_error),
#         }
#     elif isinstance(technical_error, dict):
#         technical_error_payload = deepcopy(technical_error)
#     else:
#         technical_error_payload = {}

#     report = {
#         "schema_version": FAILURE_REPORT_SCHEMA_VERSION,
#         "attempt_id": str(attempt.get("attempt_id", "")),
#         "stage_id": attempt.get("stage_id"),
#         "failure_phase": failure_phase,
#         "failure_type": failure_type,
#         "failed_conditions": failed_conditions,
#         "uncertain_conditions": uncertain_conditions,
#         "i_pre": attempt.get("pre", {}).get("image_path"),
#         "i_post": attempt.get("post", {}).get("image_path"),
#         "local_goal": str(attempt.get("local_goal", "")),
#         "action": normalized_action,
#         "scene_graph_before": _copy_dict(scene_graph_before),
#         "scene_graph_after": _copy_dict(scene_graph_after),
#         "relevant_history": _copy_list(relevant_history),
#         "evidence_rounds": _copy_list(evidence_rounds),
#         "technical_error": technical_error_payload,
#         "notes": notes,
#         "created_at": now_iso(),
#     }
#     assert_failure_report(report)
#     return report


# def build_uncertainty_exhausted_report(
#     *,
#     attempt: dict[str, Any],
#     phase: str,
#     validation: dict[str, Any],
#     action: dict[str, Any] | list[dict[str, Any]] | None = None,
#     scene_graph_before: dict[str, Any] | None = None,
#     scene_graph_after: dict[str, Any] | None = None,
#     relevant_history: list[dict[str, Any]] | None = None,
# ) -> dict[str, Any]:
#     evidence_rounds = attempt.get(phase, {}).get("evidence_rounds", [])
#     return build_failure_report(
#         attempt=attempt,
#         failure_phase=phase,
#         failure_type="insufficient_evidence",
#         validation=validation,
#         action=action,
#         scene_graph_before=scene_graph_before,
#         scene_graph_after=scene_graph_after,
#         relevant_history=relevant_history,
#         evidence_rounds=evidence_rounds,
#         notes="Maximum evidence rounds exhausted while one or more conditions remained uncertain.",
#     )




# # from __future__ import annotations

# # from copy import deepcopy
# # from datetime import datetime
# # from typing import Any

# # FAILURE_REPORT_SCHEMA_VERSION = "1.0"
# # FAILURE_PHASES = {"pre", "execution", "post"}
# # FAILURE_TYPES = {
# #     "precondition_failure",
# #     "execution_failure",
# #     "postcondition_failure",
# #     "insufficient_evidence",
# #     "technical_error",
# # }


# # def now_iso() -> str:
# #     return datetime.now().isoformat()


# # def _copy_dict(value: Any) -> dict[str, Any]:
# #     return deepcopy(value) if isinstance(value, dict) else {}


# # def _copy_list(value: Any) -> list[Any]:
# #     return deepcopy(value) if isinstance(value, list) else []


# # def _select_results(
# #     validation: dict[str, Any] | None,
# #     status: str,
# # ) -> list[dict[str, Any]]:
# #     if not isinstance(validation, dict):
# #         return []
# #     results = validation.get("results")
# #     if not isinstance(results, list):
# #         return []
# #     return [deepcopy(item) for item in results if isinstance(item, dict) and item.get("status") == status]


# # def assert_failure_report(report: dict[str, Any]) -> None:
# #     if not isinstance(report, dict):
# #         raise ValueError("Failure report must be a dictionary.")
# #     if report.get("schema_version") != FAILURE_REPORT_SCHEMA_VERSION:
# #         raise ValueError("Unsupported or missing failure-report schema_version.")
# #     if not isinstance(report.get("attempt_id"), str) or not report["attempt_id"].strip():
# #         raise ValueError("Failure report requires a non-empty attempt_id.")
# #     if not isinstance(report.get("stage_id"), int):
# #         raise ValueError("Failure report requires an integer stage_id.")
# #     if report.get("failure_phase") not in FAILURE_PHASES:
# #         raise ValueError(f"Invalid failure_phase: {report.get('failure_phase')!r}.")
# #     if report.get("failure_type") not in FAILURE_TYPES:
# #         raise ValueError(f"Invalid failure_type: {report.get('failure_type')!r}.")

# #     for key in (
# #         "failed_conditions",
# #         "uncertain_conditions",
# #         "relevant_history",
# #         "evidence_rounds",
# #     ):
# #         if not isinstance(report.get(key), list):
# #             raise ValueError(f"Failure report field {key!r} must be a list.")

# #     for key in ("action", "scene_graph_before", "scene_graph_after", "technical_error"):
# #         if not isinstance(report.get(key), dict):
# #             raise ValueError(f"Failure report field {key!r} must be an object.")

# #     for key in ("i_pre", "i_post"):
# #         value = report.get(key)
# #         if value is not None and not isinstance(value, str):
# #             raise ValueError(f"Failure report field {key!r} must be a string or null.")

# #     if not isinstance(report.get("local_goal"), str):
# #         raise ValueError("Failure report local_goal must be a string.")
# #     if not isinstance(report.get("created_at"), str) or not report["created_at"]:
# #         raise ValueError("Failure report requires created_at.")

# #     failure_type = report["failure_type"]
# #     phase = report["failure_phase"]
# #     if failure_type == "precondition_failure" and phase != "pre":
# #         raise ValueError("precondition_failure must use failure_phase='pre'.")
# #     if failure_type == "postcondition_failure" and phase != "post":
# #         raise ValueError("postcondition_failure must use failure_phase='post'.")
# #     if failure_type == "execution_failure" and phase != "execution":
# #         raise ValueError("execution_failure must use failure_phase='execution'.")
# #     if failure_type == "insufficient_evidence" and not report["uncertain_conditions"]:
# #         raise ValueError("insufficient_evidence requires uncertain_conditions.")
# #     if failure_type == "technical_error" and not report["technical_error"].get("message"):
# #         raise ValueError("technical_error requires technical_error.message.")


# # def build_failure_report(
# #     *,
# #     attempt: dict[str, Any],
# #     failure_phase: str,
# #     failure_type: str,
# #     validation: dict[str, Any] | None = None,
# #     action: dict[str, Any] | list[dict[str, Any]] | None = None,
# #     scene_graph_before: dict[str, Any] | None = None,
# #     scene_graph_after: dict[str, Any] | None = None,
# #     relevant_history: list[dict[str, Any]] | None = None,
# #     evidence_rounds: list[dict[str, Any]] | None = None,
# #     technical_error: BaseException | dict[str, Any] | None = None,
# #     notes: str | None = None,
# # ) -> dict[str, Any]:
# #     if failure_phase not in FAILURE_PHASES:
# #         raise ValueError(f"Invalid failure_phase: {failure_phase!r}.")
# #     if failure_type not in FAILURE_TYPES:
# #         raise ValueError(f"Invalid failure_type: {failure_type!r}.")

# #     failed_conditions = _select_results(validation, "violated")
# #     uncertain_conditions = _select_results(validation, "uncertain")

# #     if action is None:
# #         normalized_action: dict[str, Any] = {}
# #     elif isinstance(action, dict):
# #         normalized_action = deepcopy(action)
# #     elif isinstance(action, list):
# #         normalized_action = {"steps": deepcopy(action)}
# #     else:
# #         raise ValueError("action must be a dictionary, a list of step dictionaries, or None.")

# #     if isinstance(technical_error, BaseException):
# #         technical_error_payload = {
# #             "exception_type": type(technical_error).__name__,
# #             "message": str(technical_error),
# #         }
# #     elif isinstance(technical_error, dict):
# #         technical_error_payload = deepcopy(technical_error)
# #     else:
# #         technical_error_payload = {}

# #     report = {
# #         "schema_version": FAILURE_REPORT_SCHEMA_VERSION,
# #         "attempt_id": str(attempt.get("attempt_id", "")),
# #         "stage_id": attempt.get("stage_id"),
# #         "failure_phase": failure_phase,
# #         "failure_type": failure_type,
# #         "failed_conditions": failed_conditions,
# #         "uncertain_conditions": uncertain_conditions,
# #         "i_pre": attempt.get("pre", {}).get("image_path"),
# #         "i_post": attempt.get("post", {}).get("image_path"),
# #         "local_goal": str(attempt.get("local_goal", "")),
# #         "action": normalized_action,
# #         "scene_graph_before": _copy_dict(scene_graph_before),
# #         "scene_graph_after": _copy_dict(scene_graph_after),
# #         "relevant_history": _copy_list(relevant_history),
# #         "evidence_rounds": _copy_list(evidence_rounds),
# #         "technical_error": technical_error_payload,
# #         "notes": notes,
# #         "created_at": now_iso(),
# #     }
# #     assert_failure_report(report)
# #     return report


# # def build_uncertainty_exhausted_report(
# #     *,
# #     attempt: dict[str, Any],
# #     phase: str,
# #     validation: dict[str, Any],
# #     action: dict[str, Any] | list[dict[str, Any]] | None = None,
# #     scene_graph_before: dict[str, Any] | None = None,
# #     scene_graph_after: dict[str, Any] | None = None,
# #     relevant_history: list[dict[str, Any]] | None = None,
# # ) -> dict[str, Any]:
# #     evidence_rounds = attempt.get(phase, {}).get("evidence_rounds", [])
# #     return build_failure_report(
# #         attempt=attempt,
# #         failure_phase=phase,
# #         failure_type="insufficient_evidence",
# #         validation=validation,
# #         action=action,
# #         scene_graph_before=scene_graph_before,
# #         scene_graph_after=scene_graph_after,
# #         relevant_history=relevant_history,
# #         evidence_rounds=evidence_rounds,
# #         notes="Maximum evidence rounds exhausted while one or more conditions remained uncertain.",
# #     )
