"""`run_validation_live3.py` runs the new attempt/recovery/evidence-based
validation loop (scheduler + PRE/goal-baseline/POST validator + evidence
gathering + failure reporting + recovery) against the REAL robot, instead of
against a pre-recorded `--frames-dir` sequence.

It is derived from `run_validation_loop_fixed_on_demand.py`. Everything that
is pure planning/scheduling/validation/recovery logic (attempt state machine,
failure reporting, recovery_and_history, scene_transition_analysis, the
validator step functions and their prompts) is reused unchanged. Only the
"what image do we get after an action" seam is replaced:

  fixed_on_demand.execute_stage_offline()   -> pops the next frame from
                                                --frames-dir

  live3.execute_stage_live()                -> prints an [ACTION_REQUEST]
                                                line (same schema/format as
                                                run_validation_live2.py) and
                                                blocks on
                                                wait_for_new_realsense_frame()
                                                until a real new RGB-D frame
                                                appears under
                                                shared_data/realsense/rgb/.

Two more changes were required to run this on the real deployment:

  1. scene_enrichment_simulation (Gazebo-catalog based) is replaced with the
     real scene_enrichment module (FoundationPose poses.json / mesh_metadata
     / mesh_assignments based), exactly like run_validation_live2.py uses.

  2. call_azure_chat_completion is replaced with call_llm_completion (the
     provider-dispatching helper already used by run_validation_live.py /
     run_validation_live2.py), so --scene-model / --plan-model / --sim-model
     / --validator-model can be Gemini models, not only o3 / gpt-5.2.
     The two-image validator calls (POST validation, evidence review) needed
     `image_paths` support added to call_llm_completion / call_gemini_completion
     (see llm_client.py / gemini_client.py) - call_azure_chat_completion
     already supported it.

Scenario image folders / scenario.json are NOT used here (this mirrors
run_validation_live2.py): the task must be passed explicitly with --task.
"""

from __future__ import annotations

import argparse
import sys
import traceback
import json
import re
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from src.settings import load_settings
from src.llm_client import call_llm_completion
from src.humanoids.attempt_state_and_validation import (
    ATTEMPT_STATUSES,
    assert_attempt_invariants,
    close_attempt_state,
    compute_overall_status,
    normalize_validation_result,
    transition_attempt,
)
from src.humanoids.failure_reporting import (
    assert_failure_report,
    build_failure_report,
    build_uncertainty_exhausted_report,
)
from src.humanoids.recovery_and_history import (
    check_recovery_limits,
    extract_relevant_history,
    interpret_failure,
    plan_recovery_evidence_based,
    schedule_recovery,
)
from src.humanoids.scene_transition_analysis import (
    analyze_scene_transition,
)
from src.build_scene_object_list import build_scene_object_list_from_cycle
from src.scene_enrichment import enrich_scene
from src.run_validation_live import (
    execute_scene_description_step,
    execute_scene_description_full_step,
    execute_vlm_planning_step,
    execute_simultaneous_actions_step,
)
from src.utils import (
    load_base_prompt,
    make_experiment_timestamp,
    make_cycle_name,
    make_stage_name,
    render_prompt,
    save_rendered_prompt_for_cycle,
    save_module_outputs_for_cycle,
    save_scene_description_full_artifact_for_cycle,
    get_validator_prompt_cycle_dir,
    get_validator_output_cycle_dir,
    get_validation_loop_output_dir,
    get_validation_loop_cycle_dir,
    try_parse_json,
    write_json,
    read_json,
)

SUPPORTED_MODELS = ["o3", "gpt-5.2", "gemini-robotics-er-1.6-preview"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ============================================================
# PARSER
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the attempt/recovery/evidence-based validation loop against "
            "the real robot: pipeline -> stage PRE/goal-baseline validation -> "
            "live action execution -> POST validation -> evidence gathering / "
            "recovery on failure."
        )
    )

    parser.add_argument("--scenario", type=str, required=True)

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help=(
            "Task for planning/scheduling/recovery. Unlike the offline "
            "validation loop, live3 does not read scenario.json."
        ),
    )

    parser.add_argument(
        "--shared-data-root",
        type=str,
        default="/workspace/shared_data",
        help="Root directory shared with robot_docker (realsense/rgb, outputs_by_object, ...).",
    )
    parser.add_argument(
        "--realsense-timestamp",
        type=str,
        default=None,
        help="Explicit initial Realsense timestamp. Defaults to the latest available.",
    )
    parser.add_argument("--poses-timestamp", type=str, default=None)
    parser.add_argument("--poses-path", type=str, default=None)

    parser.add_argument("--scene-v", type=str, required=True)
    parser.add_argument("--plan-v", type=str, required=True)
    parser.add_argument("--sim-v", type=str, required=True)
    parser.add_argument(
        "--validator-v",
        type=str,
        required=False,
        default=None,
        help=(
            "Legacy validator prompt version. For example v6/precondition. "
            "When explicit PRE/POST/baseline versions are omitted, sibling "
            "versions are derived automatically."
        ),
    )
    parser.add_argument(
        "--validator-pre-v",
        type=str,
        default=None,
        help="PRE-condition validator prompt version, e.g. v6/precondition.",
    )
    parser.add_argument(
        "--validator-post-v",
        type=str,
        default=None,
        help="POST-condition validator prompt version, e.g. v6/postcondition.",
    )
    parser.add_argument(
        "--validator-baseline-v",
        type=str,
        default=None,
        help="Goal-baseline validator prompt version, e.g. v6/goal_baseline.",
    )

    parser.add_argument("--scene-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--plan-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--sim-model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--validator-model", type=str, required=True, choices=SUPPORTED_MODELS)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--max-replans", type=int, default=10)
    parser.add_argument("--max-evidence-rounds", type=int, default=2)
    parser.add_argument("--max-attempts-per-stage", type=int, default=5)
    parser.add_argument("--max-repeats", type=int, default=1)
    parser.add_argument("--max-modifications", type=int, default=2)
    parser.add_argument("--max-replacements", type=int, default=1)
    parser.add_argument("--max-total-actions", type=int, default=20)

    parser.add_argument(
        "--new-frame-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait after a robot action for a newer Realsense RGB timestamp to appear.",
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
        help="Seconds to wait after a new timestamp is detected until rgb/<ts>/000000.png is stable.",
    )
    parser.add_argument(
        "--new-frame-file-poll-seconds",
        type=float,
        default=0.1,
        help="Polling interval while waiting for rgb/<ts>/000000.png to be ready.",
    )

    parser.add_argument(
        "--terminal-log-path",
        type=str,
        default=None,
        help=(
            "Optional path for the complete terminal log. When omitted, "
            "a timestamped .txt file is created under outputs/terminal_logs/<scenario>/."
        ),
    )
    parser.add_argument(
        "--no-terminal-log",
        action="store_true",
        help="Disable automatic capture of stdout and stderr to a .txt file.",
    )

    parser.add_argument("--grounding-safety-threshold", type=float, default=0.21)
    parser.add_argument("--grounding-debug-mapping", action="store_true")

    return parser


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def save_json_file(path: Path, data: Any) -> Path:
    ensure_dir(path.parent)
    write_json(path, data)
    return path


# ============================================================
# ATTEMPT STATE (unchanged from run_validation_loop_fixed_on_demand.py)
# ============================================================

def make_attempt_id(cycle_idx: int, stage_id: int, attempt_idx: int) -> str:
    return (
        f"cycle_{cycle_idx:03d}_"
        f"stage_{stage_id:03d}_"
        f"attempt_{attempt_idx:03d}"
    )


def open_attempt(
    cycle_idx: int,
    stage: dict[str, Any],
    attempt_idx: int,
    pre_image_path: str,
    pre_scene_description_full_path: str,
    parent_attempt_id: str | None = None,
    recovery_type: str | None = None,
    recovery_changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Open a stage attempt before pre-condition validation."""
    stage_id = stage["Stage_id"]

    return {
        "attempt_id": make_attempt_id(cycle_idx=cycle_idx, stage_id=stage_id, attempt_idx=attempt_idx),
        "attempt_index": attempt_idx,
        "cycle_index": cycle_idx,
        "stage_id": stage_id,
        "step_ids": list(stage["Step_id"]),
        "local_goal": stage["Local_goal"],
        "status": "open",
        "status_history": [
            {"from": None, "to": "open", "timestamp": datetime.now().isoformat()}
        ],
        "outcome": None,
        "opened_at": datetime.now().isoformat(),
        "closed_at": None,
        "pre": {
            "image_path": str(Path(pre_image_path).resolve()),
            "image_name": Path(pre_image_path).name,
            "scene_description_full_path": str(Path(pre_scene_description_full_path).resolve()),
            "conditions": list(stage["Preconditions"]),
            "validation": None,
            "goal_baseline_validation": None,
            "goal_baseline_paths": None,
            "evidence_rounds": [],
        },
        "execution": {
            "started": False,
            "completed": False,
            "started_at": None,
            "completed_at": None,
        },
        "post": {
            "image_path": None,
            "image_name": None,
            "scene_description_full_path": None,
            "conditions": list(stage["Postconditions"]),
            "validation": None,
            "evidence_rounds": [],
        },
        "failure_report": None,
        "parent_attempt_id": parent_attempt_id,
        "recovery": {
            "parent_attempt_id": parent_attempt_id,
            "recovery_type": recovery_type,
            "attempt_number": attempt_idx,
            "changes": deepcopy(recovery_changes or {}),
        } if recovery_type else {},
    }


def set_attempt_status(attempt: dict[str, Any], status: str) -> None:
    """Compatibility wrapper around the strict attempt state machine."""
    transition_attempt(attempt, status)


def close_attempt(
    attempt: dict[str, Any],
    status: str,
    failure_report: dict[str, Any] | None = None,
) -> None:
    """Close an attempt through the strict terminal-state transition."""
    close_attempt_state(attempt=attempt, status=status, failure_report=failure_report)


def get_validation_status(validation: dict[str, Any] | None) -> str | None:
    if not isinstance(validation, dict):
        return None
    status = validation.get("overall_status")
    return status if isinstance(status, str) else None


def build_attempt_history_event(attempt: dict[str, Any], event_index: int) -> dict[str, Any]:
    """Build an immutable history event from a closed attempt."""
    assert_attempt_invariants(attempt)

    if attempt.get("closed_at") is None:
        raise ValueError("Only closed attempts can be added to attempt history.")

    return {
        "event_id": f"attempt_event_{event_index:04d}",
        "event_type": "attempt_closed",
        "timestamp": attempt["closed_at"],
        "attempt_id": attempt["attempt_id"],
        "attempt_index": attempt["attempt_index"],
        "cycle_index": attempt["cycle_index"],
        "stage_id": attempt["stage_id"],
        "step_ids": deepcopy(attempt["step_ids"]),
        "local_goal": attempt["local_goal"],
        "terminal_status": attempt["status"],
        "outcome": attempt["outcome"],
        "pre_status": get_validation_status(attempt["pre"].get("validation")),
        "post_status": get_validation_status(attempt["post"].get("validation")),
        "execution_started": bool(attempt["execution"].get("started")),
        "execution_completed": bool(attempt["execution"].get("completed")),
        "execution_mode": attempt["execution"].get("mode"),
        "i_pre": attempt["pre"].get("image_path"),
        "i_post": attempt["post"].get("image_path"),
        "failure_type": (
            attempt["failure_report"].get("failure_type")
            if isinstance(attempt.get("failure_report"), dict)
            else None
        ),
    }


def append_attempt_history(
    full_summary: dict[str, Any],
    cycle_record: dict[str, Any],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    """Append one terminal event to both global and cycle-local history."""
    global_history = full_summary.setdefault("attempt_history", [])
    cycle_history = cycle_record.setdefault("attempt_history", [])

    if any(event.get("attempt_id") == attempt.get("attempt_id") for event in global_history):
        raise ValueError(
            f"Attempt {attempt.get('attempt_id')} is already present in global history."
        )

    event = build_attempt_history_event(attempt=attempt, event_index=len(global_history) + 1)
    global_history.append(event)
    cycle_history.append(deepcopy(event))
    return event


def get_relevant_attempt_history(
    full_summary: dict[str, Any],
    stage_id: int,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return the latest closed-attempt events for the same stage."""
    history = full_summary.get("attempt_history", [])
    if not isinstance(history, list):
        return []

    matching = [
        deepcopy(event)
        for event in history
        if isinstance(event, dict) and event.get("stage_id") == stage_id
    ]
    return matching[-limit:]


# ============================================================
# LIVE STAGE EXECUTION (replaces execute_stage_offline())
# ============================================================

def normalize_action_concept(action_concept: Any) -> str:
    if not isinstance(action_concept, str) or not action_concept.strip():
        raise ValueError("Planning step has invalid or missing 'Action concept'.")

    normalized = action_concept.strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "grasp": "grasp",
        "push": "push",
        "place": "place",
        "pick": "grasp",
        "pick_up": "grasp",
        "pick-up": "grasp",
    }

    return aliases.get(normalized, normalized)


def build_action_request_for_stage_actions(
    stage: dict[str, Any],
    stage_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the [ACTION_REQUEST] payload for one stage.

    The local host executor (see run_box_pipeline_utils_simple.py in the
    eurobin_robot repo) currently supports exactly one robot action per
    stage, same constraint as run_validation_live2.py's build_execution_plan().
    """
    if len(stage_actions) != 1:
        raise ValueError(
            f"Stage {stage['Stage_id']} schedules {len(stage_actions)} planner "
            "steps. The local host executor currently supports exactly one "
            "robot action per stage. Split simultaneous actions into "
            "separate stages or extend the local host executor."
        )

    step = stage_actions[0]
    target = step.get("Target")
    action_concept = step.get("Action concept")

    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"Stage {stage['Stage_id']} step has invalid or missing 'Target'.")
    if not isinstance(action_concept, str) or not action_concept.strip():
        raise ValueError(f"Stage {stage['Stage_id']} step has invalid or missing 'Action concept'.")

    return {
        "stage_id": stage["Stage_id"],
        "step_ids": list(stage["Step_id"]),
        "action": normalize_action_concept(action_concept),
        "object": target.strip(),
        "action_concept": action_concept.strip(),
        "target": target.strip(),
        "manipulation_mode": step.get("Manipulation mode"),
        "contact_mode": step.get("Contact mode"),
        "object_contact_side": step.get("Object contact side"),
    }


def print_action_request(action_request: dict[str, Any]) -> None:
    print(
        "[ACTION_REQUEST] " + json.dumps(action_request, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def execute_stage_live(
    attempt: dict[str, Any],
    stage: dict[str, Any],
    stage_actions: list[dict[str, Any]],
    args: argparse.Namespace,
    current_realsense_timestamp: str,
) -> tuple[str, str]:
    """
    Execute one stage on the real robot.

    Prints an [ACTION_REQUEST] line (same schema as
    run_validation_live2.py's print_action_request()) for the local host
    executor to pick up, then blocks until a new Realsense RGB-D frame
    (captured after the physical action) appears under
    <shared_data_root>/realsense/rgb/.

    Mirrors execute_stage_offline()'s attempt-state contract: must be called
    on an attempt whose status is "preconditions_satisfied", and leaves it in
    "awaiting_post_validation" with attempt["post"]["image_path"/"image_name"]
    populated.
    """
    if attempt.get("status") != "preconditions_satisfied":
        raise ValueError(
            "Live execution can start only from an attempt whose "
            "preconditions are satisfied."
        )

    action_request = build_action_request_for_stage_actions(stage, stage_actions)

    set_attempt_status(attempt, "executing")
    attempt["execution"]["started"] = True
    attempt["execution"]["started_at"] = datetime.now().isoformat()
    attempt["execution"]["mode"] = "live_realsense_action_request"

    print(f"\n[LOOP] Stage {stage['Stage_id']} ACTION")
    print("[LOOP] Requesting robot action from the local host executor.")
    print_action_request(action_request)
    print(
        "[LOOP] Waiting for the local host executor to perform the action "
        "and capture a new RGB-D frame."
    )

    post_image_path, post_realsense_timestamp = wait_for_new_realsense_frame(
        shared_data_root=args.shared_data_root,
        previous_timestamp=current_realsense_timestamp,
        timeout_seconds=args.new_frame_timeout,
        poll_seconds=args.new_frame_poll_seconds,
        file_timeout_seconds=args.new_frame_file_timeout,
        file_poll_seconds=args.new_frame_file_poll_seconds,
    )

    attempt["execution"]["completed"] = True
    attempt["execution"]["completed_at"] = datetime.now().isoformat()
    attempt["post"]["image_path"] = str(post_image_path)
    attempt["post"]["image_name"] = Path(post_image_path).name

    set_attempt_status(attempt, "awaiting_post_validation")

    print(f"[LOOP] Stored I_post:  {post_image_path}")
    print(f"[LOOP] NEW timestamp:  {post_realsense_timestamp}")

    return str(post_image_path), post_realsense_timestamp


def extract_stage_actions(
    sequential_plan: Any,
    step_ids: list[int],
) -> list[dict[str, Any]]:
    """Return the planner actions whose Step_id belongs to the current stage."""
    if not isinstance(sequential_plan, list):
        return []

    selected: list[dict[str, Any]] = []
    wanted = set(step_ids)
    for item in sequential_plan:
        if not isinstance(item, dict):
            continue
        raw_step_id = item.get("Step_id", item.get("step_id"))
        if isinstance(raw_step_id, int) and raw_step_id in wanted:
            selected.append(deepcopy(item))

    return selected


# ============================================================
# VALIDATOR PROMPT RENDERING (unchanged)
# ============================================================

def render_condition_prompt_from_file(
    *,
    base_prompt: str,
    planned_stage_context: dict[str, Any],
    conditions: list[str],
    scene_object_list: Any,
    actions: list[dict[str, Any]] | None = None,
    condition_label: str,
) -> str:
    """
    Render PRE, goal-baseline, or POST prompts loaded from prompt.txt.

    Supported placeholders:
      <PLANNED_STAGE_CONTEXT>
      <PRECONDITIONS>
      <POSTCONDITIONS>
      <EXPECTED_POSTCONDITIONS>
      <CONDITIONS>
      <EXECUTED_ACTIONS>
      <SCENE_OBJECT_LIST>
      <SCENE_DESCRIPTION_FULL>

    When a prompt file contains none of these placeholders, the structured
    payload is appended so that external prompt files remain usable.
    """
    replacements = {
        "<PLANNED_STAGE_CONTEXT>": json.dumps(planned_stage_context, indent=2, ensure_ascii=False),
        "<PRECONDITIONS>": json.dumps(conditions, indent=2, ensure_ascii=False),
        "<POSTCONDITIONS>": json.dumps(conditions, indent=2, ensure_ascii=False),
        "<EXPECTED_POSTCONDITIONS>": json.dumps(conditions, indent=2, ensure_ascii=False),
        "<CONDITIONS>": json.dumps(conditions, indent=2, ensure_ascii=False),
        "<EXECUTED_ACTIONS>": json.dumps(actions or [], indent=2, ensure_ascii=False),
        "<SCENE_OBJECT_LIST>": json.dumps(scene_object_list, indent=2, ensure_ascii=False),
        "<SCENE_DESCRIPTION_FULL>": json.dumps(scene_object_list, indent=2, ensure_ascii=False),
    }

    prompt = base_prompt
    used_placeholder = False
    for placeholder, value in replacements.items():
        if placeholder in prompt:
            used_placeholder = True
            prompt = prompt.replace(placeholder, value)

    unresolved = sorted(set(re.findall(r"<[A-Z][A-Z0-9_]*>", prompt)))
    if unresolved:
        raise ValueError("Unresolved validator prompt placeholders: " + ", ".join(unresolved))

    if not used_placeholder:
        prompt = (
            prompt.strip()
            + f"\n\nPLANNED STAGE CONTEXT\n"
            + replacements["<PLANNED_STAGE_CONTEXT>"]
            + f"\n\n{condition_label}\n"
            + replacements["<CONDITIONS>"]
            + "\n\nEXECUTED ACTIONS\n"
            + replacements["<EXECUTED_ACTIONS>"]
            + "\n\nSCENE OBJECT LIST\n"
            + replacements["<SCENE_OBJECT_LIST>"]
        )

    return prompt.strip()


def render_postcondition_validator_prompt(
    *,
    base_prompt: str,
    planned_stage_context: dict[str, Any],
    actions: list[dict[str, Any]],
    expected_postconditions: list[str],
    scene_object_list: Any,
) -> str:
    return render_condition_prompt_from_file(
        base_prompt=base_prompt,
        planned_stage_context=planned_stage_context,
        conditions=expected_postconditions,
        scene_object_list=scene_object_list,
        actions=actions,
        condition_label="EXPECTED POSTCONDITIONS",
    )


def render_goal_baseline_validator_prompt(
    *,
    base_prompt: str,
    planned_stage_context: dict[str, Any],
    expected_postconditions: list[str],
    scene_object_list: Any,
) -> str:
    return render_condition_prompt_from_file(
        base_prompt=base_prompt,
        planned_stage_context=planned_stage_context,
        conditions=expected_postconditions,
        scene_object_list=scene_object_list,
        actions=[],
        condition_label="POSTCONDITIONS TO EVALUATE ON I_PRE",
    )


def _replace_validator_leaf(version: str, leaf: str) -> str:
    """
    Replace the last validator-version component.

    Examples:
        v6/precondition -> v6/postcondition
        v6              -> v6/postcondition
    """
    cleaned = version.strip().strip("/")
    if not cleaned:
        raise ValueError("Validator prompt version cannot be empty.")

    parts = cleaned.split("/")
    known_leaves = {"precondition", "postcondition", "goal_baseline"}

    if parts[-1] in known_leaves:
        parts[-1] = leaf
        return "/".join(parts)

    return f"{cleaned}/{leaf}"


def resolve_validator_prompt_versions(args: argparse.Namespace) -> None:
    """
    Resolve three independent prompt versions.

    Backward compatibility:
        --validator-v v6/precondition

    automatically becomes:
        PRE      -> v6/precondition
        baseline -> v6/goal_baseline
        POST     -> v6/postcondition
    """
    legacy = args.validator_v

    pre_version = args.validator_pre_v or legacy
    if pre_version is None:
        raise ValueError("Provide --validator-pre-v or the legacy --validator-v.")

    args.validator_pre_v = pre_version
    args.validator_post_v = args.validator_post_v or _replace_validator_leaf(pre_version, "postcondition")
    args.validator_baseline_v = args.validator_baseline_v or _replace_validator_leaf(pre_version, "goal_baseline")

    if args.validator_v is None:
        args.validator_v = args.validator_pre_v


def validate_sampling_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.temperature <= 1.0:
        raise ValueError("--temperature must be between 0.0 and 1.0")
    if not 0.0 <= args.top_p <= 1.0:
        raise ValueError("--top-p must be between 0.0 and 1.0")
    if args.temperature != 0.0 and args.top_p != 1.0:
        raise ValueError(
            "Use either temperature or top_p for sampling control, not both at the same time."
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.max_replans < 0:
        raise ValueError("--max-replans must be >= 0")
    if args.max_evidence_rounds < 0:
        raise ValueError("--max-evidence-rounds must be >= 0")
    for name in (
        "max_attempts_per_stage",
        "max_repeats",
        "max_modifications",
        "max_replacements",
        "max_total_actions",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")

    if args.poses_path is not None:
        poses_path = Path(args.poses_path)
        if not poses_path.exists():
            raise FileNotFoundError(f"poses-path not found: {poses_path}")


def print_pose_dict_for_image(
    poses_by_image: dict[str, dict[str, Any]],
    image_path: str,
    label: str,
) -> None:
    image_name = Path(image_path).name

    if image_name not in poses_by_image:
        print(f"\n[DEBUG][{label}] No poses found for image: {image_name}")
        return

    pose_dict = poses_by_image[image_name]

    print(f"\n[DEBUG][{label}] Image path: {image_path}")
    print(f"[DEBUG][{label}] Image key:  {image_name}")
    print(f"[DEBUG][{label}] Pose entries:")
    for obj_name, pose in pose_dict.items():
        print(f"  - {obj_name}: {pose}")


def make_scenario_context(scenario_data: dict[str, Any], image_path: str) -> dict[str, Any]:
    ctx = deepcopy(scenario_data)
    ctx["image"] = Path(image_path).name
    ctx["image_path_abs"] = str(Path(image_path).resolve())
    return ctx


# ============================================================
# LIVE REALSENSE CAPTURE (ported from run_validation_live2.py)
# ============================================================

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
    """Wait until rgb/<timestamp>/000000.png exists, is non-empty, and is stable."""
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be greater than 0.")
    if poll_seconds <= 0.0:
        raise ValueError("poll_seconds must be greater than 0.")
    if stable_checks_required < 1:
        raise ValueError("stable_checks_required must be at least 1.")

    image_path = (
        Path(shared_data_root) / "realsense" / "rgb" / realsense_timestamp / image_name
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
        f"{image_path} (timeout={timeout_seconds:.1f}s, poll={poll_seconds:.3f}s)"
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
    previous_mtime = previous_dir.stat().st_mtime if previous_dir.exists() else 0.0

    deadline = time.monotonic() + timeout_seconds

    print(f"[LOOP] Waiting for a new Realsense frame newer than {previous_timestamp}...")

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
        f"Timed out waiting for a new Realsense RGB frame in {rgb_root} after "
        f"{timeout_seconds:.1f}s. Run the capture script after the robot action finishes."
    )


def resolve_realsense_timestamp(shared_data_root: str | Path, explicit_timestamp: str | None) -> str:
    if explicit_timestamp is not None and explicit_timestamp.strip():
        return explicit_timestamp.strip()
    return get_latest_timestamp(Path(shared_data_root) / "realsense" / "rgb")


def resolve_realsense_image_path(shared_data_root: str | Path, realsense_timestamp: str) -> Path:
    image_path = (
        Path(shared_data_root) / "realsense" / "rgb" / realsense_timestamp / "000000.png"
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
    raise ValueError("live3 needs a non-empty --task for vlm_planning, scheduling, and recovery.")


def clean_live_pose_entry(image_name: str, obj_name: str, pose: Any) -> Any:
    if isinstance(pose, list):
        if len(pose) not in {3, 4}:
            raise ValueError(
                f"poses_by_image['{image_name}']['{obj_name}'] must be "
                "[x, y, z], [x, y, z, yaw], or an object with position."
            )
        if not all(isinstance(v, (int, float)) for v in pose):
            raise ValueError(
                f"poses_by_image['{image_name}']['{obj_name}'] must contain only numeric values."
            )
        return [float(v) for v in pose]

    if isinstance(pose, dict):
        position = pose.get("position")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(
                f"poses_by_image['{image_name}']['{obj_name}'] must contain position=[x, y, z]."
            )
        if not all(isinstance(v, (int, float)) for v in position):
            raise ValueError(
                f"poses_by_image['{image_name}']['{obj_name}'].position must contain only numeric values."
            )

        cleaned: dict[str, Any] = {"position": [float(v) for v in position]}

        for key in ["yaw", "yaw_raw"]:
            value = pose.get(key)
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise ValueError(f"poses_by_image['{image_name}']['{obj_name}'].{key} must be numeric.")
            cleaned[key] = float(value)

        orientation_quadrant = pose.get("orientation_quadrant")
        if orientation_quadrant is not None:
            if isinstance(orientation_quadrant, bool) or not isinstance(orientation_quadrant, int):
                raise ValueError(
                    f"poses_by_image['{image_name}']['{obj_name}'].orientation_quadrant must be an integer."
                )
            cleaned["orientation_quadrant"] = orientation_quadrant % 4

        return cleaned

    raise ValueError(
        f"poses_by_image['{image_name}']['{obj_name}'] must be "
        "[x, y, z], [x, y, z, yaw], or an object with position."
    )


def load_poses_by_image_map(path: str | Path) -> dict[str, dict[str, Any]]:
    """
    Load poses_by_image (e.g. shared_data/realsense/outputs_by_object/<ts>/poses.json).

    Richer schema than the offline validation loop's: each entry may be
    [x, y, z], [x, y, z, yaw], or {"position": [...], "yaw": ..., "orientation_quadrant": ...} -
    exactly what run_validation_live2.py accepts, since this is the real
    FoundationPose output, not a hand-written scenario fixture.
    """
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"poses_by_image mapping must be a JSON object. Found: {type(data).__name__}")

    validated: dict[str, dict[str, Any]] = {}
    for image_name, pose_dict in data.items():
        if not isinstance(image_name, str):
            raise ValueError("Each poses_by_image key must be an image filename string.")
        if not isinstance(pose_dict, dict):
            raise ValueError(
                f"poses_by_image['{image_name}'] must be an object mapping object names to poses."
            )

        cleaned_pose_dict: dict[str, Any] = {}
        for obj_name, pose in pose_dict.items():
            if not isinstance(obj_name, str):
                raise ValueError(f"poses_by_image['{image_name}'] contains a non-string object name.")
            cleaned_pose_dict[obj_name] = clean_live_pose_entry(
                image_name=image_name, obj_name=obj_name, pose=pose
            )

        validated[image_name] = cleaned_pose_dict

    return validated


def get_pose_dict_for_image(
    poses_by_image: dict[str, dict[str, Any]],
    image_path: str,
) -> dict[str, Any]:
    image_name = Path(image_path).name

    if image_name not in poses_by_image:
        available = ", ".join(sorted(poses_by_image.keys())[:10])
        raise KeyError(
            f"No pose entry found for image '{image_name}' in poses_by_image mapping. "
            f"Available examples: {available}"
        )

    return poses_by_image[image_name]


def write_temp_pose_file(pose_dict: dict[str, Any]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as tmp:
        json.dump(pose_dict, tmp, indent=2, ensure_ascii=False)
        return tmp.name


def load_scene_object_list_from_cycle(
    settings,
    scenario_name: str,
    scene_version: str,
    loop_timestamp: str,
    scene_model: str,
    cycle_name: str,
) -> Any:
    """Load the static object-list artifact generated for the cycle."""
    path = (
        settings.project_root
        / "outputs"
        / "scene_description"
        / scenario_name
        / scene_version
        / loop_timestamp
        / scene_model
        / cycle_name
        / "scene_object_list.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"scene_object_list.json not found: {path}")
    data = read_json(path)
    if not isinstance(data, (dict, list)):
        raise ValueError("scene_object_list.json must contain a JSON object or array.")
    return data


def extract_stages(compact_parallel_plan: Any) -> list[dict[str, Any]]:
    if not isinstance(compact_parallel_plan, list):
        raise ValueError("simultaneous_actions output must be a JSON array of stages.")

    stages: list[dict[str, Any]] = []
    for idx, stage in enumerate(compact_parallel_plan):
        if not isinstance(stage, dict):
            raise ValueError(f"Stage at index {idx} is not a JSON object.")

        stage_id = stage.get("Stage_id")
        step_ids = stage.get("Step_id")
        local_goal = stage.get("Local_goal")
        preconditions = stage.get("Preconditions")
        postconditions = stage.get("Postconditions")

        if not isinstance(stage_id, int):
            raise ValueError(f"Stage at index {idx} has invalid or missing 'Stage_id'.")
        if not isinstance(step_ids, list) or not step_ids or not all(isinstance(v, int) for v in step_ids):
            raise ValueError(f"Stage {stage_id} has invalid or missing 'Step_id'.")
        if not isinstance(local_goal, str) or not local_goal.strip():
            raise ValueError(f"Stage {stage_id} has invalid or missing 'Local_goal'.")
        if not isinstance(preconditions, list) or not preconditions:
            raise ValueError(f"Stage {stage_id} has invalid or missing 'Preconditions'.")
        if not all(isinstance(v, str) and v.strip() for v in preconditions):
            raise ValueError(f"Stage {stage_id} contains an invalid precondition.")
        if not isinstance(postconditions, list):
            raise ValueError(f"Stage {stage_id} has invalid or missing 'Postconditions'.")
        if not all(isinstance(v, str) and v.strip() for v in postconditions):
            raise ValueError(f"Stage {stage_id} contains an invalid postcondition.")

        stages.append(
            {
                "Stage_id": stage_id,
                "Step_id": step_ids,
                "Local_goal": local_goal,
                "Preconditions": preconditions,
                "Postconditions": postconditions,
            }
        )

    return stages


def build_planned_stage_context(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "Stage_id": stage["Stage_id"],
        "Step_id": stage["Step_id"],
        "Local_goal": stage["Local_goal"],
    }


def render_validator_prompt(
    base_prompt: str,
    planned_stage_context: dict[str, Any],
    preconditions: list[str],
    scene_object_list: Any,
) -> str:
    """Render the PRE-validator prompt; fail on an unresolved placeholder."""
    planned_stage_json = json.dumps(planned_stage_context, indent=2, ensure_ascii=False)
    preconditions_json = json.dumps(preconditions, indent=2, ensure_ascii=False)
    scene_context_json = json.dumps(scene_object_list, indent=2, ensure_ascii=False)

    prompt = base_prompt
    prompt = prompt.replace("<PLANNED_STAGE_CONTEXT>", planned_stage_json)
    prompt = prompt.replace("<PRECONDITIONS>", preconditions_json)
    prompt = prompt.replace("<SCENE_OBJECT_LIST>", scene_context_json)
    prompt = prompt.replace("<SCENE_DESCRIPTION_FULL>", scene_context_json)

    unresolved_placeholders = sorted(set(re.findall(r"<[A-Z][A-Z0-9_]*>", prompt)))
    if unresolved_placeholders:
        raise ValueError(
            "Unresolved PRE-validator prompt placeholders: " + ", ".join(unresolved_placeholders)
        )

    return prompt.strip()


def validate_validator_response(
    parsed_response: Any,
    expected_conditions: list[str],
    phase: str = "pre",
) -> dict[str, Any]:
    """Validate and normalize PRE/POST output to one shared schema."""
    return normalize_validation_result(
        raw_response=parsed_response,
        expected_conditions=expected_conditions,
        phase=phase,
    )


def build_global_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sampling": {"temperature": args.temperature, "top_p": args.top_p},
        "scene_description": {"prompt_version": args.scene_v, "model": args.scene_model},
        "scene_description_full": {
            "stored_under_module": "scene_description",
            "artifact_filename": "scene_description_full.json",
            "prompt_version": args.scene_v,
            "model": args.scene_model,
            "mode": "deterministic_scene_enrichment_per_image",
            "grounding_safety_threshold": args.grounding_safety_threshold,
            "grounding_debug_mapping": args.grounding_debug_mapping,
        },
        "vlm_planning": {"prompt_version": args.plan_v, "model": args.plan_model},
        "simultaneous_actions": {"prompt_version": args.sim_v, "model": args.sim_model},
        "validator": {
            "pre_prompt_version": args.validator_pre_v,
            "goal_baseline_prompt_version": args.validator_baseline_v,
            "post_prompt_version": args.validator_post_v,
            "model": args.validator_model,
        },
        "live_frame_refresh": {
            "mode": "wait_for_new_realsense_rgb_timestamp_after_action_request",
            "timeout_seconds": args.new_frame_timeout,
            "poll_seconds": args.new_frame_poll_seconds,
            "file_timeout_seconds": args.new_frame_file_timeout,
            "file_poll_seconds": args.new_frame_file_poll_seconds,
        },
        "max_replans": args.max_replans,
        "max_evidence_rounds": args.max_evidence_rounds,
        "max_attempts_per_stage": args.max_attempts_per_stage,
        "max_repeats": args.max_repeats,
        "max_modifications": args.max_modifications,
        "max_replacements": args.max_replacements,
        "max_total_actions": args.max_total_actions,
    }


def build_cycle_config(
    args: argparse.Namespace,
    cycle_timestamp: str,
    cycle_name: str,
    cycle_idx: int,
    loop_timestamp: str,
) -> dict[str, Any]:
    return {
        "cycle_name": cycle_name,
        "cycle_index": cycle_idx,
        "cycle_timestamp": cycle_timestamp,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p},
        "scene_description": {
            "prompt_version": args.scene_v,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": args.scene_model,
        },
        "scene_description_full": {
            "stored_under_module": "scene_description",
            "artifact_filename": "scene_description_full.json",
            "prompt_version": args.scene_v,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": args.scene_model,
            "mode": "deterministic_scene_enrichment_per_image",
            "grounding_safety_threshold": args.grounding_safety_threshold,
            "grounding_debug_mapping": args.grounding_debug_mapping,
        },
        "vlm_planning": {
            "prompt_version": args.plan_v,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": args.plan_model,
        },
        "simultaneous_actions": {
            "prompt_version": args.sim_v,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": args.sim_model,
        },
        "validator": {
            "pre_prompt_version": args.validator_pre_v,
            "goal_baseline_prompt_version": args.validator_baseline_v,
            "post_prompt_version": args.validator_post_v,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": args.validator_model,
        },
        "max_evidence_rounds": args.max_evidence_rounds,
    }


# ============================================================
# MODULE EXECUTION HELPERS
# (execute_scene_description_step / execute_scene_description_full_step /
#  execute_vlm_planning_step / execute_simultaneous_actions_step are reused
#  unchanged from src.run_validation_live - already call_llm_completion based
#  and, for scene_description_full, already scene_enrichment (real) based)
# ============================================================

def execute_validator_step(
    settings,
    scenario_name: str,
    validator_version: str,
    validator_model: str,
    loop_timestamp: str,
    cycle_name: str,
    cycle_idx: int,
    cycle_timestamp: str,
    stage_id: int,
    planned_stage_context: dict[str, Any],
    preconditions: list[str],
    image_path: str,
    scene_version: str,
    scene_model: str,
    plan_version: str,
    plan_model: str,
    sim_version: str,
    sim_model: str,
    temperature: float,
    top_p: float,
    condition_kind: str = "pre",
    validation_phase: str = "pre",
    image_role: str = "I_pre",
    user_instruction: str = "Validate all supplied stage conditions and return valid JSON only.",
) -> dict[str, Any]:
    stage_name = make_stage_name(stage_id)

    scene_object_list = load_scene_object_list_from_cycle(
        settings=settings,
        scenario_name=scenario_name,
        scene_version=scene_version,
        loop_timestamp=loop_timestamp,
        scene_model=scene_model,
        cycle_name=cycle_name,
    )

    print(f"[validator:{condition_kind}_{stage_id}] Scene context source: scene_object_list.json")

    base_prompt = load_base_prompt(settings, "validator", validator_version)
    system_prompt = render_validator_prompt(
        base_prompt=base_prompt,
        planned_stage_context=planned_stage_context,
        preconditions=preconditions,
        scene_object_list=scene_object_list,
    )

    prompt_dir = get_validator_prompt_cycle_dir(
        settings=settings,
        scenario_name=scenario_name,
        version=validator_version,
        loop_timestamp=loop_timestamp,
        model_name=validator_model,
        cycle_name=cycle_name,
        stage_name=stage_name,
        condition_kind=condition_kind,
    )
    prompt_path = prompt_dir / "prompt.txt"
    write_text(prompt_path, system_prompt)

    result = call_llm_completion(
        settings=settings,
        model_name=validator_model,
        system_prompt=system_prompt,
        user_text=user_instruction,
        image_path=image_path,
        temperature=temperature,
        top_p=top_p,
    )

    parse_ok, parsed_response = try_parse_json(result["raw_response"])
    if not parse_ok:
        raise ValueError(
            f"[validator:{condition_kind}_{stage_id}] Model response could not be parsed as valid JSON.\n\n"
            f"Raw response:\n{result['raw_response']}"
        )

    parsed_response = normalize_validation_result(
        raw_response=parsed_response,
        expected_conditions=preconditions,
        phase=validation_phase,
        evidence_used=[
            {"type": "image", "role": image_role, "path": str(Path(image_path).resolve())}
        ],
        validator_metadata={
            "stage_id": stage_id,
            "condition_kind": condition_kind,
            "model": result["model_name"],
            "deployment_name": result["deployment_name"],
        },
    )

    dependencies = {
        "scene_object_list": {
            "stored_under_module": "scene_description",
            "artifact_filename": "scene_object_list.json",
            "prompt_version": scene_version,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": scene_model,
        },
        "vlm_planning": {
            "prompt_version": plan_version,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": plan_model,
        },
        "simultaneous_actions": {
            "prompt_version": sim_version,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": sim_model,
        },
    }

    output_dir = get_validator_output_cycle_dir(
        settings=settings,
        scenario_name=scenario_name,
        version=validator_version,
        loop_timestamp=loop_timestamp,
        model_name=result["model_name"],
        cycle_name=cycle_name,
        stage_name=stage_name,
        condition_kind=condition_kind,
    )
    ensure_dir(output_dir)

    parsed_path = save_json_file(output_dir / "response_parsed.json", parsed_response)
    run_info = {
        "module": "validator",
        "execution_mode": f"{condition_kind}_batch_validation",
        "scenario_name": scenario_name,
        "prompt_version": validator_version,
        "loop_timestamp": loop_timestamp,
        "cycle_name": cycle_name,
        "cycle_index": cycle_idx,
        "cycle_timestamp": cycle_timestamp,
        "stage_id": stage_id,
        "stage_name": stage_name,
        "condition_kind": condition_kind,
        "planned_stage_context": planned_stage_context,
        "preconditions": preconditions,
        "model": result["model_name"],
        "deployment_name": result["deployment_name"],
        "execution_time_seconds": result["execution_time_seconds"],
        "timestamp": datetime.now().isoformat(),
        "image_path": str(Path(image_path).resolve()),
        "dependencies": dependencies,
        "sampling_config": {"temperature": temperature, "top_p": top_p},
        "response_parsed": parsed_response,
    }
    run_info_path = save_json_file(output_dir / "run_info.json", run_info)

    print(f"[OK][validator:{condition_kind}_{stage_id}] Prompt saved to:        {prompt_path}")
    print(f"[OK][validator:{condition_kind}_{stage_id}] Parsed output saved to: {parsed_path}")
    print(f"[OK][validator:{condition_kind}_{stage_id}] Run info saved to:      {run_info_path}")
    print(f"[OK][validator:{condition_kind}_{stage_id}] Execution time:         {result['execution_time_seconds']:.3f}s")

    return {
        "output": parsed_response,
        "paths": {
            "prompt": str(prompt_path),
            "response_parsed": str(parsed_path),
            "run_info": str(run_info_path),
        },
        "model_name": result["model_name"],
        "execution_time_seconds": result["execution_time_seconds"],
    }


def execute_goal_baseline_validator_step(
    settings,
    scenario_name: str,
    validator_version: str,
    validator_model: str,
    loop_timestamp: str,
    cycle_name: str,
    cycle_idx: int,
    cycle_timestamp: str,
    stage_id: int,
    planned_stage_context: dict[str, Any],
    postconditions: list[str],
    image_path: str,
    scene_version: str,
    scene_model: str,
    plan_version: str,
    plan_model: str,
    sim_version: str,
    sim_model: str,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """
    Evaluate expected POST conditions on I_pre before execution.

    This is auxiliary evidence for scene-transition analysis. It does not
    change the attempt state and is not a PRE-condition validation.
    """
    if not postconditions:
        raise ValueError(f"Stage {stage_id} has no postconditions for goal baseline.")

    stage_name = make_stage_name(stage_id)
    condition_kind = "goal_baseline"

    scene_object_list = load_scene_object_list_from_cycle(
        settings=settings,
        scenario_name=scenario_name,
        scene_version=scene_version,
        loop_timestamp=loop_timestamp,
        scene_model=scene_model,
        cycle_name=cycle_name,
    )

    print(f"[validator:goal_baseline_{stage_id}] Scene context source: scene_object_list.json")

    base_prompt = load_base_prompt(settings, "validator", validator_version)
    system_prompt = render_goal_baseline_validator_prompt(
        base_prompt=base_prompt,
        planned_stage_context=planned_stage_context,
        expected_postconditions=postconditions,
        scene_object_list=scene_object_list,
    )

    prompt_dir = get_validator_prompt_cycle_dir(
        settings=settings,
        scenario_name=scenario_name,
        version=validator_version,
        loop_timestamp=loop_timestamp,
        model_name=validator_model,
        cycle_name=cycle_name,
        stage_name=stage_name,
        condition_kind=condition_kind,
    )
    prompt_path = prompt_dir / "prompt.txt"
    write_text(prompt_path, system_prompt)

    result = call_llm_completion(
        settings=settings,
        model_name=validator_model,
        system_prompt=system_prompt,
        user_text=(
            "Evaluate every expected postcondition on I_pre only. "
            "This is a pre-execution goal baseline, not an execution-success "
            "judgment. Return valid JSON only."
        ),
        image_path=image_path,
        temperature=temperature,
        top_p=top_p,
    )

    parse_ok, parsed_response = try_parse_json(result["raw_response"])
    if not parse_ok:
        raise ValueError(
            f"[validator:goal_baseline_{stage_id}] Model response could not be parsed as valid JSON.\n\n"
            f"Raw response:\n{result['raw_response']}"
        )

    parsed_response = normalize_validation_result(
        raw_response=parsed_response,
        expected_conditions=postconditions,
        phase="post",
        evidence_used=[
            {"type": "image", "role": "I_pre_goal_baseline", "path": str(Path(image_path).resolve())}
        ],
        validator_metadata={
            "stage_id": stage_id,
            "condition_kind": condition_kind,
            "model": result["model_name"],
            "deployment_name": result["deployment_name"],
        },
    )

    output_dir = get_validator_output_cycle_dir(
        settings=settings,
        scenario_name=scenario_name,
        version=validator_version,
        loop_timestamp=loop_timestamp,
        model_name=result["model_name"],
        cycle_name=cycle_name,
        stage_name=stage_name,
        condition_kind=condition_kind,
    )
    ensure_dir(output_dir)

    parsed_path = save_json_file(output_dir / "response_parsed.json", parsed_response)
    run_info = {
        "module": "validator",
        "execution_mode": "goal_baseline_single_image",
        "scenario_name": scenario_name,
        "prompt_version": validator_version,
        "loop_timestamp": loop_timestamp,
        "cycle_name": cycle_name,
        "cycle_index": cycle_idx,
        "cycle_timestamp": cycle_timestamp,
        "stage_id": stage_id,
        "stage_name": stage_name,
        "condition_kind": condition_kind,
        "planned_stage_context": planned_stage_context,
        "postconditions": postconditions,
        "model": result["model_name"],
        "deployment_name": result["deployment_name"],
        "execution_time_seconds": result["execution_time_seconds"],
        "timestamp": datetime.now().isoformat(),
        "image_path": str(Path(image_path).resolve()),
        "sampling_config": {"temperature": temperature, "top_p": top_p},
        "response_parsed": parsed_response,
    }
    run_info_path = save_json_file(output_dir / "run_info.json", run_info)

    print(f"[OK][validator:goal_baseline_{stage_id}] Prompt saved to:        {prompt_path}")
    print(f"[OK][validator:goal_baseline_{stage_id}] Parsed output saved to: {parsed_path}")
    print(f"[OK][validator:goal_baseline_{stage_id}] Run info saved to:      {run_info_path}")
    print(f"[OK][validator:goal_baseline_{stage_id}] Execution time:         {result['execution_time_seconds']:.3f}s")

    return {
        "output": parsed_response,
        "paths": {
            "prompt": str(prompt_path),
            "response_parsed": str(parsed_path),
            "run_info": str(run_info_path),
        },
        "model_name": result["model_name"],
        "execution_time_seconds": result["execution_time_seconds"],
    }


def execute_postcondition_validator_step(
    settings,
    scenario_name: str,
    validator_version: str,
    validator_model: str,
    loop_timestamp: str,
    cycle_name: str,
    cycle_idx: int,
    cycle_timestamp: str,
    stage_id: int,
    planned_stage_context: dict[str, Any],
    actions: list[dict[str, Any]],
    postconditions: list[str],
    pre_image_path: str,
    post_image_path: str,
    scene_version: str,
    scene_model: str,
    plan_version: str,
    plan_model: str,
    sim_version: str,
    sim_model: str,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """
    Validate postconditions using I_pre, I_post, and the static object list.
    The enriched scene graph is excluded from normal validator input.
    """
    if not postconditions:
        raise ValueError(f"Stage {stage_id} has no postconditions to validate.")

    stage_name = make_stage_name(stage_id)
    condition_kind = "post"

    scene_object_list = load_scene_object_list_from_cycle(
        settings=settings,
        scenario_name=scenario_name,
        scene_version=scene_version,
        loop_timestamp=loop_timestamp,
        scene_model=scene_model,
        cycle_name=cycle_name,
    )
    print(f"[validator:post_{stage_id}] Scene context source: scene_object_list.json")

    base_prompt = load_base_prompt(settings, "validator", validator_version)
    system_prompt = render_postcondition_validator_prompt(
        base_prompt=base_prompt,
        planned_stage_context=planned_stage_context,
        actions=actions,
        expected_postconditions=postconditions,
        scene_object_list=scene_object_list,
    )

    prompt_dir = get_validator_prompt_cycle_dir(
        settings=settings,
        scenario_name=scenario_name,
        version=validator_version,
        loop_timestamp=loop_timestamp,
        model_name=validator_model,
        cycle_name=cycle_name,
        stage_name=stage_name,
        condition_kind=condition_kind,
    )
    prompt_path = prompt_dir / "prompt.txt"
    write_text(prompt_path, system_prompt)

    result = call_llm_completion(
        settings=settings,
        model_name=validator_model,
        system_prompt=system_prompt,
        user_text=(
            "Validate all expected stage postconditions by comparing I_pre "
            "and I_post. Return valid JSON only."
        ),
        image_path=None,
        image_paths=[pre_image_path, post_image_path],
        temperature=temperature,
        top_p=top_p,
    )

    parse_ok, parsed_response = try_parse_json(result["raw_response"])
    if not parse_ok:
        raise ValueError(
            f"[validator:post_{stage_id}] Model response could not be parsed as valid JSON.\n\n"
            f"Raw response:\n{result['raw_response']}"
        )

    parsed_response = normalize_validation_result(
        raw_response=parsed_response,
        expected_conditions=postconditions,
        phase="post",
        evidence_used=[
            {"type": "image", "role": "I_pre", "path": str(Path(pre_image_path).resolve())},
            {"type": "image", "role": "I_post", "path": str(Path(post_image_path).resolve())},
        ],
        validator_metadata={
            "stage_id": stage_id,
            "condition_kind": condition_kind,
            "model": result["model_name"],
            "deployment_name": result["deployment_name"],
        },
    )

    dependencies = {
        "scene_object_list_context": {
            "prompt_version": scene_version,
            "loop_timestamp": loop_timestamp,
            "cycle_name": cycle_name,
            "model": scene_model,
            "artifact_filename": "scene_object_list.json",
            "refresh_policy": "cycle_initialization_only",
        },
        "vlm_planning": {"prompt_version": plan_version, "model": plan_model},
        "simultaneous_actions": {"prompt_version": sim_version, "model": sim_model},
    }

    output_dir = get_validator_output_cycle_dir(
        settings=settings,
        scenario_name=scenario_name,
        version=validator_version,
        loop_timestamp=loop_timestamp,
        model_name=result["model_name"],
        cycle_name=cycle_name,
        stage_name=stage_name,
        condition_kind=condition_kind,
    )
    ensure_dir(output_dir)

    parsed_path = save_json_file(output_dir / "response_parsed.json", parsed_response)
    run_info = {
        "module": "validator",
        "execution_mode": "postcondition_two_images",
        "scenario_name": scenario_name,
        "prompt_version": validator_version,
        "loop_timestamp": loop_timestamp,
        "cycle_name": cycle_name,
        "cycle_index": cycle_idx,
        "cycle_timestamp": cycle_timestamp,
        "stage_id": stage_id,
        "stage_name": stage_name,
        "condition_kind": condition_kind,
        "planned_stage_context": planned_stage_context,
        "actions": actions,
        "postconditions": postconditions,
        "model": result["model_name"],
        "deployment_name": result["deployment_name"],
        "execution_time_seconds": result["execution_time_seconds"],
        "timestamp": datetime.now().isoformat(),
        "pre_image_path": str(Path(pre_image_path).resolve()),
        "post_image_path": str(Path(post_image_path).resolve()),
        "image_order": ["I_pre", "I_post"],
        "dependencies": dependencies,
        "sampling_config": {"temperature": temperature, "top_p": top_p},
        "response_parsed": parsed_response,
    }
    run_info_path = save_json_file(output_dir / "run_info.json", run_info)

    print(f"[OK][validator:post_{stage_id}] Prompt saved to:        {prompt_path}")
    print(f"[OK][validator:post_{stage_id}] I_pre:                  {pre_image_path}")
    print(f"[OK][validator:post_{stage_id}] I_post:                 {post_image_path}")
    print(f"[OK][validator:post_{stage_id}] Parsed output saved to: {parsed_path}")
    print(f"[OK][validator:post_{stage_id}] Run info saved to:      {run_info_path}")
    print(f"[OK][validator:post_{stage_id}] Execution time:         {result['execution_time_seconds']:.3f}s")

    return {
        "output": parsed_response,
        "paths": {
            "prompt": str(prompt_path),
            "pre_image": str(Path(pre_image_path).resolve()),
            "post_image": str(Path(post_image_path).resolve()),
            "response_parsed": str(parsed_path),
            "run_info": str(run_info_path),
        },
        "model_name": result["model_name"],
        "execution_time_seconds": result["execution_time_seconds"],
    }


def get_evidence_round_dir(
    settings,
    scenario_name: str,
    loop_timestamp: str,
    cycle_name: str,
    stage_id: int,
    phase: str,
    round_index: int,
) -> Path:
    if phase not in {"pre", "post"}:
        raise ValueError(f"Unsupported evidence phase: {phase!r}")
    return (
        get_validation_loop_cycle_dir(settings, scenario_name, loop_timestamp, cycle_name)
        / "evidence"
        / make_stage_name(stage_id)
        / phase
        / f"round_{round_index:03d}"
    )


def build_evidence_request(validation: dict[str, Any], phase: str, round_index: int) -> dict[str, Any]:
    uncertain_conditions = [
        deepcopy(item)
        for item in validation.get("results", [])
        if isinstance(item, dict) and item.get("status") == "uncertain"
    ]
    return {
        "phase": phase,
        "round": round_index,
        "uncertain_conditions": uncertain_conditions,
        "requested_evidence": [
            "refreshed_scene_description",
            "updated_pose_enrichment",
            "independent_validator_pass",
        ],
        "instruction": (
            "Re-observe the current image, rebuild the structured scene graph "
            "using the pose entry associated with that image, and validate the "
            "same conditions again without assuming the previous answer."
        ),
        "created_at": datetime.now().isoformat(),
    }


def execute_scene_perception_for_state(
    *,
    settings,
    scenario_name: str,
    scenario_data: dict[str, Any],
    image_path: str,
    poses_by_image: dict[str, dict[str, Any]],
    scene_version: str,
    scene_model: str,
    temperature: float,
    top_p: float,
    safety_threshold: float,
    include_debug_mapping: bool,
    output_dir: Path,
    purpose: str,
    direct_name_matching: bool = True,
) -> dict[str, Any]:
    """
    Reconstruct a structured world state from one image.

    This helper is intentionally independent from the cycle-level perception
    artifacts, so POST perception and evidence rounds never overwrite the
    initial scene-description outputs.
    """
    ensure_dir(output_dir)

    base_prompt = load_base_prompt(settings, "scene_description", scene_version)
    result = call_llm_completion(
        settings=settings,
        model_name=scene_model,
        system_prompt=base_prompt,
        user_text=(
            "Analyze the current scene again and return the structured JSON "
            "output. Treat this as a fresh observation."
        ),
        image_path=image_path,
        temperature=temperature,
        top_p=top_p,
    )

    parse_ok, scene_description = try_parse_json(result["raw_response"])
    if not parse_ok:
        raise ValueError(
            f"[scene_perception:{purpose}] Model response could not be parsed as valid JSON.\n\n"
            f"Raw response:\n{result['raw_response']}"
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
            direct_name_matching=direct_name_matching,
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

    write_text(prompt_path, base_prompt)
    save_json_file(scene_description_path, scene_description)
    save_json_file(scene_graph_path, scene_graph)
    save_json_file(
        run_info_path,
        {
            "module": "scene_perception",
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
            "sampling_config": {"temperature": temperature, "top_p": top_p},
            "scenario_context": make_scenario_context(scenario_data=scenario_data, image_path=image_path),
            "created_at": datetime.now().isoformat(),
        },
    )

    print(f"[OK][scene_perception:{purpose}] Updated scene graph saved to: {scene_graph_path}")
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
        "execution_time_seconds": result["execution_time_seconds"] + enrichment_seconds,
    }


def execute_evidence_validator_step(
    *,
    settings,
    scenario_name: str,
    validator_model: str,
    stage_id: int,
    phase: str,
    round_index: int,
    planned_stage_context: dict[str, Any],
    conditions: list[str],
    actions: list[dict[str, Any]],
    pre_image_path: str,
    post_image_path: str | None,
    scene_graph: dict[str, Any],
    temperature: float,
    top_p: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Run an independent validator pass using refreshed perception evidence."""
    ensure_dir(output_dir)

    if phase == "pre":
        system_prompt = f"""
You are performing an independent evidence-review pass for PRE conditions.

Use the attached current image and the refreshed structured scene graph.
Do not copy the previous validator decision. Evaluate each condition again.

PLANNED STAGE CONTEXT
{json.dumps(planned_stage_context, indent=2, ensure_ascii=False)}

PRECONDITIONS
{json.dumps(conditions, indent=2, ensure_ascii=False)}

REFRESHED STRUCTURED SCENE GRAPH
{json.dumps(scene_graph, indent=2, ensure_ascii=False)}

Return exactly one JSON object:
{{
  "overall_status": "satisfied|violated|uncertain",
  "results": [
    {{
      "condition": "Exact input condition text.",
      "status": "satisfied|violated|uncertain",
      "reason": "Brief evidence-grounded explanation."
    }}
  ]
}}
Preserve condition text and order. Return JSON only.
""".strip()
        image_path = pre_image_path
        image_paths = None
        user_text = (
            "Independently revalidate every PRE condition using the refreshed "
            "scene evidence. Return valid JSON only."
        )
    elif phase == "post":
        if post_image_path is None:
            raise ValueError("POST evidence validation requires I_post.")

        system_prompt = f"""
You are performing an independent evidence-review pass for POST conditions.

Compare I_pre and I_post and use the refreshed structured scene graph,
which was reconstructed from I_post. Do not copy the previous validator
decision. Evaluate every expected postcondition again.

PLANNED STAGE CONTEXT
{json.dumps(planned_stage_context, indent=2, ensure_ascii=False)}

EXECUTED ACTIONS
{json.dumps(actions, indent=2, ensure_ascii=False)}

EXPECTED POSTCONDITIONS
{json.dumps(conditions, indent=2, ensure_ascii=False)}

REFRESHED I_POST STRUCTURED SCENE GRAPH
{json.dumps(scene_graph, indent=2, ensure_ascii=False)}

Return exactly one JSON object:
{{
  "overall_status": "satisfied|violated|uncertain",
  "results": [
    {{
      "condition": "Exact input condition text.",
      "status": "satisfied|violated|uncertain",
      "reason": "Brief evidence-grounded explanation."
    }}
  ]
}}
Preserve condition text and order. Return JSON only.
""".strip()
        image_path = None
        image_paths = [pre_image_path, post_image_path]
        user_text = (
            "Independently revalidate every POST condition using I_pre, I_post, "
            "and the refreshed I_post scene graph. Return valid JSON only."
        )
    else:
        raise ValueError(f"Unsupported evidence phase: {phase!r}")

    prompt_path = output_dir / "prompt.txt"
    write_text(prompt_path, system_prompt)

    result = call_llm_completion(
        settings=settings,
        model_name=validator_model,
        system_prompt=system_prompt,
        user_text=user_text,
        image_path=image_path,
        image_paths=image_paths,
        temperature=temperature,
        top_p=top_p,
    )
    parse_ok, parsed_response = try_parse_json(result["raw_response"])
    if not parse_ok:
        raise ValueError(
            f"[evidence_validator:{phase}:{round_index}] Response could not be "
            f"parsed as JSON.\n\nRaw response:\n{result['raw_response']}"
        )

    evidence_used = [
        {"type": "image", "role": "I_pre", "path": str(Path(pre_image_path).resolve())},
        {
            "type": "scene_graph",
            "role": "refreshed_world_state",
            "path": str(output_dir.parent / "scene_perception" / "scene_description_full.json"),
        },
    ]
    if phase == "post" and post_image_path is not None:
        evidence_used.insert(
            1, {"type": "image", "role": "I_post", "path": str(Path(post_image_path).resolve())}
        )

    normalized = normalize_validation_result(
        raw_response=parsed_response,
        expected_conditions=conditions,
        phase=phase,
        evidence_used=evidence_used,
        validator_metadata={
            "stage_id": stage_id,
            "condition_kind": phase,
            "evidence_round": round_index,
            "model": result["model_name"],
            "deployment_name": result["deployment_name"],
            "independent_pass": True,
        },
    )

    response_path = save_json_file(output_dir / "response_parsed.json", normalized)
    run_info_path = save_json_file(
        output_dir / "run_info.json",
        {
            "module": "validator",
            "execution_mode": "evidence_review",
            "scenario_name": scenario_name,
            "stage_id": stage_id,
            "phase": phase,
            "round": round_index,
            "model": result["model_name"],
            "deployment_name": result["deployment_name"],
            "execution_time_seconds": result["execution_time_seconds"],
            "pre_image_path": str(Path(pre_image_path).resolve()),
            "post_image_path": (
                str(Path(post_image_path).resolve()) if post_image_path is not None else None
            ),
            "response_parsed": normalized,
            "created_at": datetime.now().isoformat(),
        },
    )
    return {
        "output": normalized,
        "paths": {
            "prompt": str(prompt_path),
            "response_parsed": str(response_path),
            "run_info": str(run_info_path),
        },
        "model_name": result["model_name"],
        "execution_time_seconds": result["execution_time_seconds"],
    }


def gather_and_revalidate_evidence(
    *,
    settings,
    scenario_name: str,
    scenario_data: dict[str, Any],
    poses_by_image: dict[str, dict[str, Any]],
    attempt: dict[str, Any],
    phase: str,
    initial_validation: dict[str, Any],
    max_evidence_rounds: int,
    planned_stage_context: dict[str, Any],
    actions: list[dict[str, Any]],
    conditions: list[str],
    scene_version: str,
    scene_model: str,
    validator_model: str,
    loop_timestamp: str,
    cycle_name: str,
    temperature: float,
    top_p: float,
    safety_threshold: float,
    include_debug_mapping: bool,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """
    Resolve an uncertain validation by repeatedly acquiring fresh perception
    evidence and running an independent validator pass.

    Returns the latest validation and latest reconstructed scene graph.
    """
    if phase not in {"pre", "post"}:
        raise ValueError(f"Unsupported evidence phase: {phase!r}")

    latest_validation = deepcopy(initial_validation)
    current_graph: dict[str, Any] = {}
    current_graph_path: str | None = None
    image_path = attempt["pre"]["image_path"] if phase == "pre" else attempt["post"]["image_path"]
    if not isinstance(image_path, str):
        raise ValueError(f"{phase.upper()} evidence gathering requires an image.")

    for round_index in range(1, max_evidence_rounds + 1):
        request = build_evidence_request(validation=latest_validation, phase=phase, round_index=round_index)
        round_dir = get_evidence_round_dir(
            settings=settings,
            scenario_name=scenario_name,
            loop_timestamp=loop_timestamp,
            cycle_name=cycle_name,
            stage_id=attempt["stage_id"],
            phase=phase,
            round_index=round_index,
        )
        perception = execute_scene_perception_for_state(
            settings=settings,
            scenario_name=scenario_name,
            scenario_data=scenario_data,
            image_path=image_path,
            poses_by_image=poses_by_image,
            scene_version=scene_version,
            scene_model=scene_model,
            temperature=temperature,
            top_p=top_p,
            safety_threshold=safety_threshold,
            include_debug_mapping=include_debug_mapping,
            output_dir=round_dir / "scene_perception",
            purpose=f"{phase}_evidence_round_{round_index}",
        )
        current_graph = perception["scene_graph"]
        current_graph_path = perception["paths"]["scene_graph"]

        transition_attempt(
            attempt, "awaiting_pre_validation" if phase == "pre" else "awaiting_post_validation"
        )
        validation_artifact = execute_evidence_validator_step(
            settings=settings,
            scenario_name=scenario_name,
            validator_model=validator_model,
            stage_id=attempt["stage_id"],
            phase=phase,
            round_index=round_index,
            planned_stage_context=planned_stage_context,
            conditions=conditions,
            actions=actions,
            pre_image_path=attempt["pre"]["image_path"],
            post_image_path=attempt["post"]["image_path"],
            scene_graph=current_graph,
            temperature=temperature,
            top_p=top_p,
            output_dir=round_dir / "validator",
        )
        latest_validation = validation_artifact["output"]

        round_record = {
            "round": round_index,
            "request": request,
            "acquired_evidence": {
                "image_path": str(Path(image_path).resolve()),
                "pose_key": Path(image_path).name,
                "scene_perception": perception["paths"],
            },
            "validation": latest_validation,
            "validator_paths": validation_artifact["paths"],
            "uncertain_conditions": [
                deepcopy(item) for item in latest_validation["results"] if item["status"] == "uncertain"
            ],
            "timestamp": datetime.now().isoformat(),
        }
        attempt[phase]["evidence_rounds"].append(round_record)

        if latest_validation["overall_status"] != "uncertain":
            break

        if round_index < max_evidence_rounds:
            transition_attempt(attempt, "awaiting_pre_evidence" if phase == "pre" else "awaiting_post_evidence")

    return latest_validation, current_graph, current_graph_path


def collect_all_attempts(
    full_summary: dict[str, Any],
    current_cycle: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for cycle in full_summary.get("cycles", []):
        if isinstance(cycle, dict):
            attempts.extend(item for item in cycle.get("attempts", []) if isinstance(item, dict))
    if isinstance(current_cycle, dict):
        attempts.extend(item for item in current_cycle.get("attempts", []) if isinstance(item, dict))
    return attempts


def extract_remaining_task_goal(scenario_data: dict[str, Any]) -> str:
    for key in ("task_goal", "goal", "objective", "task", "instruction", "description"):
        value = scenario_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Complete the manipulation task defined by the scenario."


def execute_final_goal_validator(
    *,
    settings,
    model_name: str,
    scenario_name: str,
    task_goal: str,
    final_image_path: str,
    final_scene_graph: dict[str, Any],
    temperature: float,
    top_p: float,
    output_dir: Path,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    prompt = f"""
You are the final-goal validator for a robotic manipulation task.

TASK GOAL
{task_goal}

FINAL UPDATED SCENE GRAPH
{json.dumps(final_scene_graph, indent=2, ensure_ascii=False)}

Inspect the attached final image and the structured final scene graph.
Validate the task goal as a whole. Do not infer success merely because all
intermediate stages were reported successful.

Return exactly one JSON object:
{{
  "overall_status": "satisfied|violated|uncertain",
  "reason": "Brief evidence-grounded explanation.",
  "unsatisfied_requirements": ["..."],
  "evidence_used": ["final_image", "final_scene_graph"]
}}
Return JSON only.
""".strip()

    prompt_path = output_dir / "prompt.txt"
    write_text(prompt_path, prompt)
    result = call_llm_completion(
        settings=settings,
        model_name=model_name,
        system_prompt=prompt,
        user_text="Validate the final task goal and return valid JSON only.",
        image_path=final_image_path,
        temperature=temperature,
        top_p=top_p,
    )
    ok, parsed = try_parse_json(result["raw_response"])
    if not ok or not isinstance(parsed, dict):
        raise ValueError("[final_goal_validator] Response is not a valid JSON object.")
    status = parsed.get("overall_status")
    if status not in {"satisfied", "violated", "uncertain"}:
        raise ValueError(f"[final_goal_validator] Invalid overall_status: {status!r}")
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("[final_goal_validator] Missing reason.")
    unsatisfied = parsed.get("unsatisfied_requirements", [])
    if not isinstance(unsatisfied, list):
        raise ValueError("[final_goal_validator] unsatisfied_requirements must be a list.")

    normalized = {
        "schema_version": "1.0",
        "overall_status": status,
        "reason": reason.strip(),
        "unsatisfied_requirements": deepcopy(unsatisfied),
        "task_goal": task_goal,
        "final_image_path": str(Path(final_image_path).resolve()),
        "final_scene_graph": deepcopy(final_scene_graph),
        "model": result["model_name"],
        "deployment_name": result["deployment_name"],
        "execution_time_seconds": result["execution_time_seconds"],
        "validated_at": datetime.now().isoformat(),
    }
    response_path = save_json_file(output_dir / "response_parsed.json", normalized)
    run_info_path = save_json_file(
        output_dir / "run_info.json",
        {
            "module": "final_goal_validator",
            "scenario_name": scenario_name,
            "prompt": str(prompt_path),
            "response": str(response_path),
            **normalized,
        },
    )
    return {
        "output": normalized,
        "paths": {"prompt": str(prompt_path), "response_parsed": str(response_path), "run_info": str(run_info_path)},
    }


# ============================================================
# SUMMARY HELPERS (unchanged)
# ============================================================

def build_run_info(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "validation_loop",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": full_summary["timestamp"],
        "initial_image_path": full_summary["initial_image_path"],
        "shared_data_root": full_summary["shared_data_root"],
        "poses_by_image_path": full_summary["poses_by_image_path"],
        "config": full_summary["config"],
    }


def build_loop_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "validation_loop_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": full_summary["timestamp"],
        "config": full_summary["config"],
        "initial_image_path": full_summary["initial_image_path"],
        "final_image_path": full_summary.get("final_image_path"),
        "task_completed": full_summary["task_completed"],
        "replans_done": full_summary["replans_done"],
        "total_cycles": len(full_summary["cycles"]),
        "error": full_summary.get("error"),
        "attempt_history": full_summary.get("attempt_history", []),
        "recovery_history": full_summary.get("recovery_history", []),
        "final_goal_validation": full_summary.get("final_goal_validation"),
        "cycles": [
            {
                "cycle_name": cycle["cycle_name"],
                "cycle_index": cycle["cycle_index"],
                "cycle_timestamp": cycle["cycle_timestamp"],
                "start_image_path": cycle["start_image_path"],
                "start_image_name": cycle["start_image_name"],
                "outcome": cycle["outcome"],
            }
            for cycle in full_summary["cycles"]
        ],
    }


def build_scene_description_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "scene_description_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": datetime.now().isoformat(),
        "config": {
            "sampling": full_summary["config"]["sampling"],
            "scene_description": full_summary["config"]["scene_description"],
            "scene_description_full": full_summary["config"]["scene_description_full"],
        },
        "cycles": [
            {
                "cycle_name": cycle["cycle_name"],
                "cycle_index": cycle["cycle_index"],
                "cycle_timestamp": cycle["cycle_timestamp"],
                "image_path": cycle["start_image_path"],
                "image_name": cycle["start_image_name"],
                "scene_description_paths": {
                    "prompt": cycle["scene_description"]["paths"]["prompt"],
                    "response_parsed": cycle["scene_description"]["paths"]["response_parsed"],
                    "run_info": cycle["scene_description"]["paths"]["run_info"],
                    "scene_object_list": cycle["scene_description"]["paths"]["scene_object_list"],
                    "scene_description_full": cycle["scene_description_full"]["paths"]["artifact"],
                    "scene_description_full_run_info": cycle["scene_description_full"]["paths"]["run_info"],
                },
                "scene_description_output": cycle["scene_description"]["output"],
                "scene_description_full_output": cycle["scene_description_full"]["output"],
            }
            for cycle in full_summary["cycles"]
            if cycle.get("scene_description") is not None and cycle.get("scene_description_full") is not None
        ],
    }


def build_vlm_planning_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "vlm_planning_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": datetime.now().isoformat(),
        "config": {"sampling": full_summary["config"]["sampling"], "vlm_planning": full_summary["config"]["vlm_planning"]},
        "cycles": [
            {
                "cycle_name": cycle["cycle_name"],
                "cycle_index": cycle["cycle_index"],
                "cycle_timestamp": cycle["cycle_timestamp"],
                "input_image_path": cycle["start_image_path"],
                "input_image_name": cycle["start_image_name"],
                "dependencies": {
                    "scene_description_cycle": cycle["cycle_name"],
                    "scene_description_full_path": cycle["scene_description_full"]["paths"]["artifact"],
                },
                "vlm_planning_paths": cycle["vlm_planning"]["paths"],
                "vlm_planning_output": cycle["vlm_planning"]["output"],
            }
            for cycle in full_summary["cycles"]
            if cycle.get("vlm_planning") is not None
        ],
    }


def build_simultaneous_actions_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "simultaneous_actions_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": datetime.now().isoformat(),
        "config": {
            "sampling": full_summary["config"]["sampling"],
            "simultaneous_actions": full_summary["config"]["simultaneous_actions"],
        },
        "cycles": [
            {
                "cycle_name": cycle["cycle_name"],
                "cycle_index": cycle["cycle_index"],
                "cycle_timestamp": cycle["cycle_timestamp"],
                "input_image_path": cycle["start_image_path"],
                "input_image_name": cycle["start_image_name"],
                "dependencies": {
                    "scene_description_cycle": cycle["cycle_name"],
                    "scene_description_full_path": cycle["scene_description_full"]["paths"]["artifact"],
                    "vlm_planning_cycle": cycle["cycle_name"],
                    "vlm_planning_path": cycle["vlm_planning"]["paths"]["response_parsed"],
                },
                "simultaneous_actions_paths": cycle["simultaneous_actions"]["paths"],
                "simultaneous_actions_output": cycle["simultaneous_actions"]["output"],
            }
            for cycle in full_summary["cycles"]
            if cycle.get("simultaneous_actions") is not None
        ],
    }


def build_validator_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "validator_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "timestamp": datetime.now().isoformat(),
        "config": {
            "sampling": full_summary["config"]["sampling"],
            "validator": full_summary["config"]["validator"],
            "max_replans": full_summary["config"]["max_replans"],
        },
        "replans_done": full_summary["replans_done"],
        "task_completed": full_summary["task_completed"],
        "cycles": [
            {
                "cycle_name": cycle["cycle_name"],
                "cycle_index": cycle["cycle_index"],
                "cycle_timestamp": cycle["cycle_timestamp"],
                "start_image_path": cycle["start_image_path"],
                "start_image_name": cycle["start_image_name"],
                "outcome": cycle["outcome"],
                "stages": cycle["stages"],
                "attempts": cycle.get("attempts", []),
                "attempt_history": cycle.get("attempt_history", []),
            }
            for cycle in full_summary["cycles"]
        ],
    }


def build_full_pipeline_summary(full_summary: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(full_summary)


def build_cycle_summary(full_summary: dict[str, Any], cycle_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "cycle_summary",
        "scenario_name": full_summary["scenario_name"],
        "loop_timestamp": full_summary["loop_timestamp"],
        "cycle_name": cycle_record["cycle_name"],
        "cycle_index": cycle_record["cycle_index"],
        "cycle_timestamp": cycle_record["cycle_timestamp"],
        "start_image_path": cycle_record["start_image_path"],
        "start_image_name": cycle_record["start_image_name"],
        "scene_description": cycle_record.get("scene_description"),
        "scene_description_full": cycle_record.get("scene_description_full"),
        "vlm_planning": cycle_record.get("vlm_planning"),
        "simultaneous_actions": cycle_record.get("simultaneous_actions"),
        "stages": cycle_record["stages"],
        "attempts": cycle_record.get("attempts", []),
        "attempt_history": cycle_record.get("attempt_history", []),
        "recovery": cycle_record.get("recovery"),
        "recovery_schedule": cycle_record.get("recovery_schedule"),
        "final_goal_validation": cycle_record.get("final_goal_validation"),
        "current_world_state": cycle_record.get("current_world_state"),
        "outcome": cycle_record["outcome"],
        "end_image_path": cycle_record.get("end_image_path"),
        "end_image_name": cycle_record.get("end_image_name"),
    }


def save_validation_loop_artifacts(
    settings,
    scenario_name: str,
    loop_timestamp: str,
    run_info: dict[str, Any],
    loop_summary: dict[str, Any],
    scene_description_summary: dict[str, Any],
    vlm_planning_summary: dict[str, Any],
    simultaneous_actions_summary: dict[str, Any],
    validator_summary: dict[str, Any],
    full_pipeline_summary: dict[str, Any],
) -> dict[str, Path]:
    output_dir = get_validation_loop_output_dir(settings, scenario_name, loop_timestamp)
    ensure_dir(output_dir)

    paths = {
        "run_info": save_json_file(output_dir / "run_info.json", run_info),
        "loop_summary": save_json_file(output_dir / "loop_summary.json", loop_summary),
        "scene_description_summary": save_json_file(
            output_dir / "scene_description_summary.json", scene_description_summary
        ),
        "vlm_planning_summary": save_json_file(output_dir / "vlm_planning_summary.json", vlm_planning_summary),
        "simultaneous_actions_summary": save_json_file(
            output_dir / "simultaneous_actions_summary.json", simultaneous_actions_summary
        ),
        "validator_summary": save_json_file(output_dir / "validator_summary.json", validator_summary),
        "attempt_history": save_json_file(
            output_dir / "attempt_history.json",
            {
                "module": "attempt_history",
                "scenario_name": full_pipeline_summary["scenario_name"],
                "loop_timestamp": full_pipeline_summary["loop_timestamp"],
                "timestamp": datetime.now().isoformat(),
                "events": full_pipeline_summary.get("attempt_history", []),
            },
        ),
        "full_pipeline_summary": save_json_file(output_dir / "full_pipeline_summary.json", full_pipeline_summary),
    }
    return paths


def save_cycle_summary(
    settings,
    scenario_name: str,
    loop_timestamp: str,
    cycle_name: str,
    cycle_summary: dict[str, Any],
) -> Path:
    cycle_dir = get_validation_loop_cycle_dir(settings, scenario_name, loop_timestamp, cycle_name)
    ensure_dir(cycle_dir)
    return save_json_file(cycle_dir / "cycle_summary.json", cycle_summary)


class TeeTextStream:
    """Write the same text to the original terminal stream and a log file."""

    def __init__(self, terminal_stream, log_stream) -> None:
        self.terminal_stream = terminal_stream
        self.log_stream = log_stream

    def write(self, data: str) -> int:
        terminal_written = self.terminal_stream.write(data)
        self.log_stream.write(data)
        self.flush()
        return terminal_written if terminal_written is not None else len(data)

    def flush(self) -> None:
        self.terminal_stream.flush()
        self.log_stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal_stream, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self.terminal_stream, "encoding", "utf-8")

    def fileno(self) -> int:
        return self.terminal_stream.fileno()


def resolve_terminal_log_path(args: argparse.Namespace) -> Path:
    if args.terminal_log_path:
        return Path(args.terminal_log_path).expanduser().resolve()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (
        Path("outputs") / "terminal_logs" / str(args.scenario) / f"validation_live3_{timestamp}.txt"
    ).resolve()


def run_with_terminal_log() -> None:
    """
    Execute the complete application while duplicating stdout and stderr.

    Parsing is done once here only to resolve the logging configuration.
    main() performs the authoritative parsing and validation.
    """
    bootstrap_parser = build_parser()
    bootstrap_args, _ = bootstrap_parser.parse_known_args()

    if bootstrap_args.no_terminal_log:
        main()
        return

    log_path = resolve_terminal_log_path(bootstrap_args)
    ensure_dir(log_path.parent)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeTextStream(original_stdout, log_file)
        sys.stderr = TeeTextStream(original_stderr, log_file)

        exit_code = 0
        try:
            print(f"[LOG] Complete terminal output: {log_path}")
            print(f"[LOG] Started at: {datetime.now().isoformat()}")
            print("=" * 54)
            main()
        except KeyboardInterrupt:
            exit_code = 130
            print("\n[LOG] Execution interrupted by the user.", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        except BaseException:
            exit_code = 1
            print("\n[LOG] Unhandled exception:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            print("=" * 54)
            print(f"[LOG] Finished at: {datetime.now().isoformat()}")
            print(f"[LOG] Exit code: {exit_code}")
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    original_stdout.write(f"\n[LOG] Terminal log saved to: {log_path}\n")
    original_stdout.flush()

    if exit_code:
        raise SystemExit(exit_code)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    resolve_validator_prompt_versions(args)

    validate_sampling_args(args)
    validate_args(args)

    settings = load_settings()
    task = resolve_task(args.task)
    scenario_data: dict[str, Any] = {"task": task}

    poses_by_image_path, poses_timestamp = resolve_live_poses_path(
        shared_data_root=args.shared_data_root,
        explicit_timestamp=args.poses_timestamp,
        explicit_path=args.poses_path,
    )
    poses_by_image = load_poses_by_image_map(poses_by_image_path)

    realsense_timestamp = resolve_realsense_timestamp(
        shared_data_root=args.shared_data_root,
        explicit_timestamp=args.realsense_timestamp,
    )
    initial_image_path = str(
        resolve_realsense_image_path(args.shared_data_root, realsense_timestamp)
    )

    loop_timestamp = make_experiment_timestamp()

    current_image = initial_image_path
    current_realsense_timestamp = realsense_timestamp
    task_completed = False
    cycle_idx = 0

    full_summary: dict[str, Any] = {
        "module": "full_pipeline_summary",
        "scenario_name": args.scenario,
        "loop_timestamp": loop_timestamp,
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "initial_image_path": str(Path(initial_image_path).resolve()),
        "initial_realsense_timestamp": realsense_timestamp,
        "shared_data_root": str(Path(args.shared_data_root).resolve()),
        "poses_by_image_path": str(poses_by_image_path),
        "poses_timestamp": poses_timestamp,
        "config": build_global_config(args),
        "replans_done": 0,
        "task_completed": False,
        "precondition_validation_completed": False,
        "execution_abstraction_completed": False,
        "post_image_acquired": False,
        "current_stage_id": None,
        "current_stage_overall_status": None,
        "final_image_path": None,
        "attempt_history": [],
        "recovery_history": [],
        "recovery_counters": {"replans": 0, "total_actions": 0},
        "pending_recovery_schedule": None,
        "final_goal_validation": None,
        "cycles": [],
    }

    print("\n======================================================")
    print("LIVE3 | REAL-ROBOT VALIDATION LOOP CONFIG")
    print(f"Scenario:                  {args.scenario}")
    print(f"Task:                      {task}")
    print(f"Shared data root:          {args.shared_data_root}")
    print(f"Initial Realsense ts:      {realsense_timestamp}")
    print(f"Poses file:                {poses_by_image_path}")
    print(f"Temperature:               {args.temperature}")
    print(f"Top-p:                     {args.top_p}")
    print(f"Max replans:               {args.max_replans}")
    print(f"Max evidence rounds:       {args.max_evidence_rounds}")
    print(f"Max attempts/stage:        {args.max_attempts_per_stage}")
    print(f"Max repeats:               {args.max_repeats}")
    print(f"Max modifications:         {args.max_modifications}")
    print(f"Max replacements:          {args.max_replacements}")
    print(f"Max total actions:         {args.max_total_actions}")
    print("======================================================")

    while not task_completed:
        cycle_idx += 1
        cycle_name = make_cycle_name(cycle_idx)
        cycle_timestamp = make_experiment_timestamp()

        print("\n======================================================")
        print(f"LIVE3 CYCLE STARTED | cycle={cycle_idx} | {cycle_name}")
        print(f"Current image:   {current_image}")
        print(f"Realsense ts:    {current_realsense_timestamp}")
        print(f"Loop ts:         {loop_timestamp}")
        print(f"Cycle ts meta:   {cycle_timestamp}")
        print("======================================================")

        scenario_context = make_scenario_context(scenario_data=scenario_data, image_path=current_image)

        pipeline_config = build_cycle_config(
            args=args,
            cycle_timestamp=cycle_timestamp,
            cycle_name=cycle_name,
            cycle_idx=cycle_idx,
            loop_timestamp=loop_timestamp,
        )

        cycle_record: dict[str, Any] = {
            "cycle_name": cycle_name,
            "cycle_index": cycle_idx,
            "cycle_timestamp": cycle_timestamp,
            "start_image_path": str(Path(current_image).resolve()),
            "start_image_name": Path(current_image).name,
            "start_realsense_timestamp": current_realsense_timestamp,
            "scene_description": None,
            "scene_description_full": None,
            "vlm_planning": None,
            "simultaneous_actions": None,
            "stages": [],
            "attempts": [],
            "attempt_history": [],
            "outcome": None,
            "end_image_path": None,
            "end_image_name": None,
        }

        cycle_error = False

        try:
            scene_description_artifact = execute_scene_description_step(
                settings=settings,
                scenario_name=args.scenario,
                scenario_context=scenario_context,
                version=args.scene_v,
                model_name=args.scene_model,
                loop_timestamp=loop_timestamp,
                cycle_name=cycle_name,
                cycle_idx=cycle_idx,
                cycle_timestamp=cycle_timestamp,
                pipeline_config=pipeline_config,
                image_path=current_image,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            cycle_record["scene_description"] = scene_description_artifact

            print("\n[scene_description] Parsed JSON:")
            print(json.dumps(scene_description_artifact["output"], indent=2, ensure_ascii=False))

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
                image_path=current_image,
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
            if not stages:
                sequential_plan_output = sequential_plan_artifact.get("output")
                scheduler_output = simultaneous_actions_artifact.get("output")

                if sequential_plan_output == [] and scheduler_output == []:
                    cycle_record["outcome"] = "completed_no_actions_required"
                    cycle_record["end_image_path"] = str(Path(current_image).resolve())
                    cycle_record["end_image_name"] = Path(current_image).name
                    full_summary["final_image_path"] = str(Path(current_image).resolve())
                    full_summary["task_completed"] = True
                    task_completed = True
                    print(
                        "[LOOP] Planner and scheduler returned empty plans: "
                        "no further actions are required from the current state."
                    )
                else:
                    raise ValueError(
                        "Scheduler returned no stages for a non-empty or invalid sequential plan."
                    )

            pending_schedule = full_summary.get("pending_recovery_schedule")
            if isinstance(pending_schedule, dict):
                if pending_schedule.get("mode") == "local_reschedule":
                    scheduled = pending_schedule.get("stages", [])
                    if not isinstance(scheduled, list) or not scheduled:
                        raise ValueError("Local recovery schedule contains no stages.")
                    stages = deepcopy(scheduled)
                    cycle_record["recovery_schedule"] = deepcopy(pending_schedule)
                    full_summary["pending_recovery_schedule"] = None
                    print(f"[RECOVERY] Applying local schedule with {len(stages)} pending stages.")
                elif pending_schedule.get("mode") == "global_replan":
                    cycle_record["recovery_schedule"] = deepcopy(pending_schedule)
                    full_summary["pending_recovery_schedule"] = None
                    print("[RECOVERY] Applying global replan from latest world state.")

            # Execute every scheduled stage on the real robot:
            #   Stage 1: PRE=current_image, ACTION_REQUEST, POST=new Realsense frame
            #   Stage 2: PRE=that new frame, ...
            all_stages_succeeded = True

            for stage_position, stage in enumerate(stages, start=1):
                stage_id = stage["Stage_id"]
                stage_name = make_stage_name(stage_id)
                planned_stage_context = build_planned_stage_context(stage)
                preconditions = stage["Preconditions"]

                previous_stage_attempts = [
                    item
                    for item in collect_all_attempts(full_summary, cycle_record)
                    if item.get("stage_id") == stage_id
                ]
                attempt_idx = len(previous_stage_attempts) + 1
                recovery_metadata = stage.get("_recovery") if isinstance(stage.get("_recovery"), dict) else {}
                attempt_record = open_attempt(
                    cycle_idx=cycle_idx,
                    stage=stage,
                    attempt_idx=attempt_idx,
                    pre_image_path=current_image,
                    pre_scene_description_full_path=scene_description_full_artifact["paths"]["artifact"],
                    parent_attempt_id=recovery_metadata.get("parent_attempt_id"),
                    recovery_type=recovery_metadata.get("recovery_type"),
                    recovery_changes=recovery_metadata.get("changes"),
                )
                cycle_record["attempts"].append(attempt_record)

                print("\n[LOOP] Attempt opened")
                print(f"[LOOP] Attempt ID:     {attempt_record['attempt_id']}")
                print(f"[LOOP] Attempt status: {attempt_record['status']}")
                print(f"[LOOP] Stored I_pre:   {attempt_record['pre']['image_path']}")

                set_attempt_status(attempt_record, "awaiting_pre_validation")
                print(f"[LOOP] Attempt status: {attempt_record['status']}")

                stage_record: dict[str, Any] = {
                    "stage_id": stage_id,
                    "stage_position": stage_position,
                    "stage_name": stage_name,
                    "step_ids": stage["Step_id"],
                    "local_goal": stage["Local_goal"],
                    "preconditions": preconditions,
                    "postconditions": stage["Postconditions"],
                    "planned_stage_context": planned_stage_context,
                    "attempt_ids": [attempt_record["attempt_id"]],
                    "pre_image_path": str(Path(current_image).resolve()),
                    "pre_image_name": Path(current_image).name,
                    "pre_validation": None,
                    "goal_baseline_validation": None,
                    "validator_paths": {"pre": None, "goal_baseline": None},
                }

                print(f"\n[LOOP] Stage {stage_id} PRE batch validation")
                print(f"[LOOP] PRE image: {current_image}")
                print(json.dumps(preconditions, indent=2, ensure_ascii=False))

                print_pose_dict_for_image(
                    poses_by_image=poses_by_image,
                    image_path=current_image,
                    label=f"validator-pre-stage-{stage_id}",
                )

                pre_artifact = execute_validator_step(
                    settings=settings,
                    scenario_name=args.scenario,
                    validator_version=args.validator_pre_v,
                    validator_model=args.validator_model,
                    loop_timestamp=loop_timestamp,
                    cycle_name=cycle_name,
                    cycle_idx=cycle_idx,
                    cycle_timestamp=cycle_timestamp,
                    stage_id=stage_id,
                    planned_stage_context=planned_stage_context,
                    preconditions=preconditions,
                    image_path=current_image,
                    scene_version=args.scene_v,
                    scene_model=args.scene_model,
                    plan_version=args.plan_v,
                    plan_model=args.plan_model,
                    sim_version=args.sim_v,
                    sim_model=args.sim_model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    condition_kind="pre",
                    validation_phase="pre",
                    image_role="I_pre",
                    user_instruction="Validate all stage preconditions and return valid JSON only.",
                )

                pre_response = pre_artifact["output"]
                stage_record["pre_validation"] = pre_response
                stage_record["validator_paths"]["pre"] = pre_artifact["paths"]
                attempt_record["pre"]["validation"] = pre_response
                cycle_record["stages"].append(stage_record)

                print(f"\n[PRE validator:pre_{stage_id}] Parsed JSON:")
                print(json.dumps(pre_response, indent=2, ensure_ascii=False))
                print(f"[LOOP] PRE overall status: {pre_response['overall_status']}")

                pre_status = pre_response["overall_status"]
                full_summary["precondition_validation_completed"] = True
                full_summary["current_stage_id"] = stage_id
                full_summary["current_stage_overall_status"] = pre_status

                if isinstance(stage.get("_actions"), list):
                    stage_actions = deepcopy(stage["_actions"])
                else:
                    stage_actions = extract_stage_actions(
                        sequential_plan=sequential_plan_artifact["output"],
                        step_ids=stage["Step_id"],
                    )

                if pre_status == "uncertain":
                    set_attempt_status(attempt=attempt_record, status="awaiting_pre_evidence")
                    pre_response, refreshed_pre_graph, refreshed_pre_graph_path = gather_and_revalidate_evidence(
                        settings=settings,
                        scenario_name=args.scenario,
                        scenario_data=scenario_data,
                        poses_by_image=poses_by_image,
                        attempt=attempt_record,
                        phase="pre",
                        initial_validation=pre_response,
                        max_evidence_rounds=args.max_evidence_rounds,
                        planned_stage_context=planned_stage_context,
                        actions=stage_actions,
                        conditions=preconditions,
                        scene_version=args.scene_v,
                        scene_model=args.scene_model,
                        validator_model=args.validator_model,
                        loop_timestamp=loop_timestamp,
                        cycle_name=cycle_name,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        safety_threshold=args.grounding_safety_threshold,
                        include_debug_mapping=args.grounding_debug_mapping,
                    )
                    pre_status = pre_response["overall_status"]
                    attempt_record["pre"]["validation"] = pre_response
                    stage_record["pre_validation"] = pre_response
                    stage_record["pre_evidence_rounds"] = deepcopy(attempt_record["pre"]["evidence_rounds"])
                    if refreshed_pre_graph:
                        scene_description_full_artifact["output"] = refreshed_pre_graph
                        if refreshed_pre_graph_path is not None:
                            scene_description_full_artifact["paths"]["artifact"] = refreshed_pre_graph_path
                    print(f"[LOOP] PRE status after evidence gathering: {pre_status}")

                if pre_status == "satisfied":
                    set_attempt_status(attempt=attempt_record, status="preconditions_satisfied")
                    print(f"[LOOP] Attempt {attempt_record['attempt_id']} is ready for execution.")

                    print(f"\n[LOOP] Stage {stage_id} goal baseline validation on I_pre")
                    goal_baseline_artifact = execute_goal_baseline_validator_step(
                        settings=settings,
                        scenario_name=args.scenario,
                        validator_version=args.validator_baseline_v,
                        validator_model=args.validator_model,
                        loop_timestamp=loop_timestamp,
                        cycle_name=cycle_name,
                        cycle_idx=cycle_idx,
                        cycle_timestamp=cycle_timestamp,
                        stage_id=stage_id,
                        planned_stage_context=planned_stage_context,
                        postconditions=list(stage["Postconditions"]),
                        image_path=current_image,
                        scene_version=args.scene_v,
                        scene_model=args.scene_model,
                        plan_version=args.plan_v,
                        plan_model=args.plan_model,
                        sim_version=args.sim_v,
                        sim_model=args.sim_model,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    goal_baseline_response = goal_baseline_artifact["output"]
                    attempt_record["pre"]["goal_baseline_validation"] = goal_baseline_response
                    attempt_record["pre"]["goal_baseline_paths"] = goal_baseline_artifact["paths"]
                    stage_record["goal_baseline_validation"] = goal_baseline_response
                    stage_record["validator_paths"]["goal_baseline"] = goal_baseline_artifact["paths"]
                    print(f"[LOOP] Goal baseline overall status: {goal_baseline_response['overall_status']}")

                    print(f"\n[LOOP] Executing Stage {stage_id} on the real robot")
                    full_summary["recovery_counters"]["total_actions"] += max(1, len(stage_actions))
                    check_recovery_limits(
                        limits={
                            "max_attempts_per_stage": args.max_attempts_per_stage,
                            "max_repeats": args.max_repeats,
                            "max_modifications": args.max_modifications,
                            "max_replacements": args.max_replacements,
                            "max_replans": args.max_replans,
                            "max_total_actions": args.max_total_actions,
                        },
                        counters=full_summary["recovery_counters"],
                    )
                    try:
                        post_image, post_realsense_timestamp = execute_stage_live(
                            attempt=attempt_record,
                            stage=stage,
                            stage_actions=stage_actions,
                            args=args,
                            current_realsense_timestamp=current_realsense_timestamp,
                        )
                    except Exception as execution_exc:
                        failure_report = build_failure_report(
                            attempt=attempt_record,
                            failure_phase="execution",
                            failure_type="execution_failure",
                            action=stage_actions,
                            scene_graph_before=scene_description_full_artifact["output"],
                            scene_graph_after={},
                            relevant_history=get_relevant_attempt_history(full_summary, stage_id),
                            evidence_rounds=[],
                            technical_error=execution_exc,
                            notes="Live stage execution failed before I_post was acquired.",
                        )
                        assert_failure_report(failure_report)
                        close_attempt(attempt=attempt_record, status="closed_failure", failure_report=failure_report)
                        history_event = append_attempt_history(
                            full_summary=full_summary, cycle_record=cycle_record, attempt=attempt_record
                        )
                        stage_record["execution"] = deepcopy(attempt_record["execution"])
                        stage_record["attempt_outcome"] = "failure"
                        stage_record["attempt_history_event_id"] = history_event["event_id"]
                        cycle_record["outcome"] = f"execution_failure_stage_{stage_id}"
                        all_stages_succeeded = False
                        print(
                            f"[LOOP] Attempt {attempt_record['attempt_id']} closed after "
                            f"execution failure: {execution_exc}"
                        )
                        break

                    stage_record["execution"] = deepcopy(attempt_record["execution"])
                    stage_record["post_image_path"] = post_image
                    stage_record["post_image_name"] = Path(post_image).name
                    stage_record["post_validation"] = None
                    stage_record["validator_paths"]["post"] = None

                    print(f"[LOOP] Attempt status: {attempt_record['status']}")

                    post_perception: dict[str, Any] | None = None
                    post_scene_graph: dict[str, Any] = {}
                    recovery_pre_scene_graph = scene_description_full_artifact["output"]

                    postconditions = stage["Postconditions"]
                    print(f"\n[LOOP] Stage {stage_id} POST batch validation")
                    print(json.dumps(postconditions, indent=2, ensure_ascii=False))

                    post_artifact = execute_postcondition_validator_step(
                        settings=settings,
                        scenario_name=args.scenario,
                        validator_version=args.validator_post_v,
                        validator_model=args.validator_model,
                        loop_timestamp=loop_timestamp,
                        cycle_name=cycle_name,
                        cycle_idx=cycle_idx,
                        cycle_timestamp=cycle_timestamp,
                        stage_id=stage_id,
                        planned_stage_context=planned_stage_context,
                        actions=stage_actions,
                        postconditions=postconditions,
                        pre_image_path=attempt_record["pre"]["image_path"],
                        post_image_path=post_image,
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
                    attempt_record["post"]["validation"] = post_response
                    stage_record["post_validation"] = post_response
                    stage_record["validator_paths"]["post"] = post_artifact["paths"]

                    post_status = post_response["overall_status"]
                    print(f"\n[POST validator:post_{stage_id}] Parsed JSON:")
                    print(json.dumps(post_response, indent=2, ensure_ascii=False))
                    print(f"[LOOP] POST overall status: {post_status}")

                    if post_status == "uncertain":
                        set_attempt_status(attempt_record, "awaiting_post_evidence")
                        post_response, refreshed_post_graph, refreshed_post_graph_path = gather_and_revalidate_evidence(
                            settings=settings,
                            scenario_name=args.scenario,
                            scenario_data=scenario_data,
                            poses_by_image=poses_by_image,
                            attempt=attempt_record,
                            phase="post",
                            initial_validation=post_response,
                            max_evidence_rounds=args.max_evidence_rounds,
                            planned_stage_context=planned_stage_context,
                            actions=stage_actions,
                            conditions=postconditions,
                            scene_version=args.scene_v,
                            scene_model=args.scene_model,
                            validator_model=args.validator_model,
                            loop_timestamp=loop_timestamp,
                            cycle_name=cycle_name,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            safety_threshold=args.grounding_safety_threshold,
                            include_debug_mapping=args.grounding_debug_mapping,
                        )
                        post_status = post_response["overall_status"]
                        attempt_record["post"]["validation"] = post_response
                        stage_record["post_validation"] = post_response
                        stage_record["post_evidence_rounds"] = deepcopy(attempt_record["post"]["evidence_rounds"])
                        if refreshed_post_graph:
                            post_scene_graph = refreshed_post_graph
                            if refreshed_post_graph_path is not None:
                                attempt_record["post"]["scene_description_full_path"] = refreshed_post_graph_path
                        print(f"[LOOP] POST status after evidence gathering: {post_status}")

                    if post_status != "satisfied" and not post_scene_graph:
                        post_perception_dir = (
                            get_evidence_round_dir(
                                settings=settings,
                                scenario_name=args.scenario,
                                loop_timestamp=loop_timestamp,
                                cycle_name=cycle_name,
                                stage_id=stage_id,
                                phase="post",
                                round_index=0,
                            )
                            / "scene_perception"
                        )
                        post_perception = execute_scene_perception_for_state(
                            settings=settings,
                            scenario_name=args.scenario,
                            scenario_data=scenario_data,
                            image_path=post_image,
                            poses_by_image=poses_by_image,
                            scene_version=args.scene_v,
                            scene_model=args.scene_model,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            safety_threshold=args.grounding_safety_threshold,
                            include_debug_mapping=args.grounding_debug_mapping,
                            output_dir=post_perception_dir,
                            purpose=f"post_recovery_state_stage_{stage_id}",
                        )
                        post_scene_graph = post_perception["scene_graph"]
                        attempt_record["post"]["scene_description_full_path"] = post_perception["paths"]["scene_graph"]
                        stage_record["post_scene_perception"] = post_perception

                    if post_status != "satisfied":
                        pre_recovery_dir = (
                            get_evidence_round_dir(
                                settings=settings,
                                scenario_name=args.scenario,
                                loop_timestamp=loop_timestamp,
                                cycle_name=cycle_name,
                                stage_id=stage_id,
                                phase="pre",
                                round_index=0,
                            )
                            / "scene_perception_recovery"
                        )
                        pre_recovery_perception = execute_scene_perception_for_state(
                            settings=settings,
                            scenario_name=args.scenario,
                            scenario_data=scenario_data,
                            image_path=attempt_record["pre"]["image_path"],
                            poses_by_image=poses_by_image,
                            scene_version=args.scene_v,
                            scene_model=args.scene_model,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            safety_threshold=args.grounding_safety_threshold,
                            include_debug_mapping=args.grounding_debug_mapping,
                            output_dir=pre_recovery_dir,
                            purpose=f"pre_recovery_state_stage_{stage_id}",
                        )
                        recovery_pre_scene_graph = pre_recovery_perception["scene_graph"]
                        attempt_record["pre"]["scene_description_full_path"] = pre_recovery_perception["paths"]["scene_graph"]
                        stage_record["pre_recovery_scene_perception"] = pre_recovery_perception

                    if post_status == "satisfied":
                        set_attempt_status(attempt_record, "postconditions_satisfied")
                        close_attempt(attempt=attempt_record, status="closed_success")
                        history_event = append_attempt_history(
                            full_summary=full_summary, cycle_record=cycle_record, attempt=attempt_record
                        )
                        stage_record["attempt_outcome"] = "success"
                        stage_record["attempt_history_event_id"] = history_event["event_id"]
                        print(f"[LOOP] Attempt {attempt_record['attempt_id']} closed successfully.")

                    elif post_status == "violated":
                        set_attempt_status(attempt_record, "postconditions_violated")
                        failure_report = build_failure_report(
                            attempt=attempt_record,
                            failure_phase="post",
                            failure_type="postcondition_failure",
                            validation=post_response,
                            action=stage_actions,
                            scene_graph_before=recovery_pre_scene_graph,
                            scene_graph_after=post_scene_graph,
                            relevant_history=get_relevant_attempt_history(full_summary, stage_id),
                            evidence_rounds=attempt_record["post"]["evidence_rounds"],
                            notes="One or more expected postconditions were violated.",
                        )
                        assert_failure_report(failure_report)
                        close_attempt(attempt=attempt_record, status="closed_failure", failure_report=failure_report)
                        history_event = append_attempt_history(
                            full_summary=full_summary, cycle_record=cycle_record, attempt=attempt_record
                        )
                        stage_record["attempt_outcome"] = "failure"
                        stage_record["attempt_history_event_id"] = history_event["event_id"]
                        all_stages_succeeded = False
                        print(f"[LOOP] Attempt {attempt_record['attempt_id']} closed with postcondition failure.")

                    else:
                        failure_report = build_uncertainty_exhausted_report(
                            attempt=attempt_record,
                            phase="post",
                            validation=post_response,
                            action=stage_actions,
                            scene_graph_before=recovery_pre_scene_graph,
                            scene_graph_after=post_scene_graph,
                            relevant_history=get_relevant_attempt_history(full_summary, stage_id),
                        )
                        assert_failure_report(failure_report)
                        close_attempt(attempt=attempt_record, status="closed_failure", failure_report=failure_report)
                        history_event = append_attempt_history(
                            full_summary=full_summary, cycle_record=cycle_record, attempt=attempt_record
                        )
                        stage_record["attempt_outcome"] = "failure"
                        stage_record["attempt_history_event_id"] = history_event["event_id"]
                        all_stages_succeeded = False
                        print(
                            f"[LOOP] Attempt {attempt_record['attempt_id']} closed because "
                            "evidence remained insufficient."
                        )

                    current_image = post_image
                    current_realsense_timestamp = post_realsense_timestamp

                    if post_status != "satisfied":
                        if post_perception is not None:
                            post_run_info_path = post_perception["paths"]["run_info"]
                            post_perception_seconds = post_perception["execution_time_seconds"]
                        else:
                            post_run_info_path = None
                            post_perception_seconds = None

                        scene_description_full_artifact = {
                            "output": post_scene_graph,
                            "paths": {
                                "artifact": attempt_record["post"].get("scene_description_full_path"),
                                "run_info": post_run_info_path,
                            },
                            "execution_time_seconds": post_perception_seconds,
                        }
                        cycle_record["current_world_state"] = deepcopy(post_scene_graph)
                        full_summary["current_world_state"] = deepcopy(post_scene_graph)

                    cycle_record["outcome"] = f"postconditions_{post_status}_stage_{stage_id}"
                    full_summary["execution_abstraction_completed"] = True
                    full_summary["post_image_acquired"] = True
                    full_summary["postcondition_validation_completed"] = True
                    full_summary["current_stage_post_status"] = post_status
                    full_summary["final_image_path"] = str(Path(current_image).resolve())

                    print(f"[LOOP] Attempt status after POST validation: {attempt_record['status']}")

                    if post_status != "satisfied":
                        print(f"[LOOP] Stopping stage sequence at Stage {stage_id}: POST status is {post_status}.")
                        break

                elif pre_status == "violated":
                    failure_report = build_failure_report(
                        attempt=attempt_record,
                        failure_phase="pre",
                        failure_type="precondition_failure",
                        validation=pre_response,
                        action=stage_actions,
                        scene_graph_before=scene_description_full_artifact["output"],
                        scene_graph_after={},
                        relevant_history=get_relevant_attempt_history(full_summary, stage_id),
                        evidence_rounds=attempt_record["pre"]["evidence_rounds"],
                        notes="One or more preconditions were violated; execution was not started.",
                    )
                    assert_failure_report(failure_report)
                    close_attempt(attempt=attempt_record, status="closed_not_executed", failure_report=failure_report)
                    history_event = append_attempt_history(
                        full_summary=full_summary, cycle_record=cycle_record, attempt=attempt_record
                    )
                    stage_record["attempt_outcome"] = "not_executed"
                    stage_record["attempt_history_event_id"] = history_event["event_id"]
                    cycle_record["outcome"] = f"preconditions_violated_stage_{stage_id}"
                    all_stages_succeeded = False
                    print(f"[LOOP] Attempt {attempt_record['attempt_id']} closed without execution.")
                    print(f"[LOOP] Stopping stage sequence at Stage {stage_id}: PRE conditions were violated.")
                    break

                else:
                    failure_report = build_uncertainty_exhausted_report(
                        attempt=attempt_record,
                        phase="pre",
                        validation=pre_response,
                        action=stage_actions,
                        scene_graph_before=scene_description_full_artifact["output"],
                        scene_graph_after={},
                        relevant_history=get_relevant_attempt_history(full_summary, stage_id),
                    )
                    assert_failure_report(failure_report)
                    close_attempt(attempt=attempt_record, status="closed_not_executed", failure_report=failure_report)
                    history_event = append_attempt_history(
                        full_summary=full_summary, cycle_record=cycle_record, attempt=attempt_record
                    )
                    stage_record["attempt_outcome"] = "not_executed"
                    stage_record["attempt_history_event_id"] = history_event["event_id"]
                    cycle_record["outcome"] = f"preconditions_insufficient_evidence_stage_{stage_id}"
                    all_stages_succeeded = False
                    print(
                        f"[LOOP] Attempt {attempt_record['attempt_id']} closed because "
                        "PRE evidence remained insufficient."
                    )
                    break

            cycle_record["end_image_path"] = str(Path(current_image).resolve())
            cycle_record["end_image_name"] = Path(current_image).name
            full_summary["final_image_path"] = str(Path(current_image).resolve())

            completed_stage_count = sum(
                1 for stage_record in cycle_record["stages"] if stage_record.get("attempt_outcome") == "success"
            )
            full_summary["completed_stage_count"] = completed_stage_count
            full_summary["scheduled_stage_count"] = len(stages)

            if all_stages_succeeded and completed_stage_count == len(stages):
                if cycle_record.get("outcome") != "completed_no_actions_required":
                    cycle_record["outcome"] = "all_scheduled_stages_succeeded"
                cycle_record["final_goal_validation"] = {
                    "mode": "stage_postconditions_conjunction",
                    "overall_status": "satisfied",
                    "reason": (
                        "All scheduled stages completed and every stage POST "
                        "validator returned satisfied."
                    ),
                    "validated_stage_count": completed_stage_count,
                    "scheduled_stage_count": len(stages),
                    "final_image_path": str(Path(current_image).resolve()),
                    "created_at": datetime.now().isoformat(),
                }
                full_summary["final_goal_validation"] = deepcopy(cycle_record["final_goal_validation"])
                full_summary["task_completed"] = True
                task_completed = True
                print(f"\n[LOOP] Final goal validated after {len(stages)} successful scheduled stages.")
            else:
                full_summary["task_completed"] = False
                failed_attempt = next(
                    (
                        item
                        for item in reversed(cycle_record.get("attempts", []))
                        if item.get("status") in {"closed_failure", "closed_not_executed"}
                    ),
                    None,
                )
                if not isinstance(failed_attempt, dict):
                    task_completed = True
                    cycle_record["outcome"] = "recovery_unavailable_no_failed_attempt"
                else:
                    failure_report = failed_attempt.get("failure_report")
                    failed_stage_id = failed_attempt["stage_id"]
                    failed_stage = next(
                        (item for item in stages if item.get("Stage_id") == failed_stage_id), stage
                    )
                    failed_index = stages.index(failed_stage)
                    remaining_stages = stages[failed_index + 1:]
                    failed_actions = (
                        deepcopy(failed_stage.get("_actions"))
                        if isinstance(failed_stage.get("_actions"), list)
                        else extract_stage_actions(sequential_plan_artifact["output"], failed_stage["Step_id"])
                    )
                    relevant_history = extract_relevant_history(
                        attempts=collect_all_attempts(full_summary, cycle_record),
                        stage_id=failed_stage_id,
                        current_failure_report=failure_report,
                        latest_scene_graph=full_summary.get(
                            "current_world_state", scene_description_full_artifact["output"]
                        ),
                    )
                    limits = {
                        "max_attempts_per_stage": args.max_attempts_per_stage,
                        "max_repeats": args.max_repeats,
                        "max_modifications": args.max_modifications,
                        "max_replacements": args.max_replacements,
                        "max_replans": args.max_replans,
                        "max_total_actions": args.max_total_actions,
                    }
                    scene_transition = analyze_scene_transition(
                        scene_graph_before=failure_report.get("scene_graph_before", {}),
                        scene_graph_after=failure_report.get("scene_graph_after", {}),
                        failed_stage=failed_stage,
                        actions=failed_actions,
                        before_goal_validation=failed_attempt.get("pre", {}).get("goal_baseline_validation"),
                        after_goal_validation=failed_attempt.get("post", {}).get("validation"),
                    )

                    failure_interpretation = interpret_failure(
                        failure_report=failure_report,
                        relevant_history=relevant_history,
                        failed_stage=failed_stage,
                        actions=failed_actions,
                        scene_transition=scene_transition,
                    )

                    print(
                        "\n[RECOVERY][INTERPRETATION] "
                        f"evidence={failure_interpretation['evidence_status']} | "
                        f"phase={failure_interpretation['failure_phase']} | "
                        f"cause={failure_interpretation['cause_status']} | "
                        f"execution_completed={failure_interpretation['execution_completed']} | "
                        f"same_failure_count={failure_interpretation['same_failure_count']} | "
                        f"goal_progress={failure_interpretation['goal_progress']} | "
                        f"target_state_changed={failure_interpretation['target_state_changed']} | "
                        f"stage_still_applicable={failure_interpretation['stage_still_applicable']}"
                    )
                    print(
                        "[RECOVERY][INTERPRETATION] "
                        f"supported_modifications={len(failure_interpretation['supported_symbolic_modifications'])} | "
                        f"replacement_supported={failure_interpretation['replacement_supported']} | "
                        f"replan_required={failure_interpretation['replan_required']}"
                    )

                    recovery_plan = plan_recovery_evidence_based(
                        failure_report=failure_report,
                        relevant_history=relevant_history,
                        failure_interpretation=failure_interpretation,
                        failed_stage=failed_stage,
                        actions=failed_actions,
                        remaining_task_goal=extract_remaining_task_goal(scenario_data),
                        limits=limits,
                        counters=full_summary["recovery_counters"],
                    )

                    for candidate, assessment in recovery_plan.get("admissibility", {}).items():
                        print(
                            f"[RECOVERY][CANDIDATE] {candidate}: "
                            f"admissible={assessment.get('admissible')} | {assessment.get('reason')}"
                        )
                    recovery_schedule = schedule_recovery(
                        recovery_plan=recovery_plan,
                        failed_stage=failed_stage,
                        failed_actions=failed_actions,
                        remaining_stages=remaining_stages,
                        next_stage_id=max([item.get("Stage_id", 0) for item in stages] + [0]) + 1,
                        parent_attempt_id=failed_attempt["attempt_id"],
                        next_attempt_number=failed_attempt["attempt_index"] + 1,
                    )
                    recovery_record = {
                        "failed_attempt_id": failed_attempt["attempt_id"],
                        "failure_report": deepcopy(failure_report),
                        "relevant_history": relevant_history,
                        "scene_transition": scene_transition,
                        "failure_interpretation": failure_interpretation,
                        "recovery_plan": recovery_plan,
                        "recovery_schedule": recovery_schedule,
                        "created_at": datetime.now().isoformat(),
                    }
                    cycle_record["recovery"] = deepcopy(recovery_record)
                    full_summary["recovery_history"].append(recovery_record)

                    decision = recovery_plan["decision"]
                    if decision == "abort":
                        task_completed = True
                        cycle_record["outcome"] = "recovery_aborted"
                    else:
                        if decision == "replan":
                            full_summary["recovery_counters"]["replans"] += 1
                            full_summary["replans_done"] += 1
                        full_summary["pending_recovery_schedule"] = recovery_schedule
                        task_completed = False
                        cycle_record["outcome"] = f"recovery_{decision}_scheduled"
                        print(f"\n[RECOVERY] decision={decision} | {recovery_plan['reason']}")
                        print(f"[RECOVERY] Resume from current image: {current_image}")

        except Exception as exc:
            print(f"\n[ERROR][live3] {exc}")
            cycle_record["outcome"] = f"cycle_error: {exc}"
            cycle_record["end_image_path"] = str(Path(current_image).resolve())
            cycle_record["end_image_name"] = Path(current_image).name
            full_summary["task_completed"] = False
            full_summary["error"] = str(exc)
            cycle_error = True

        full_summary["cycles"].append(cycle_record)

        cycle_summary = build_cycle_summary(full_summary, cycle_record)
        cycle_summary_path = save_cycle_summary(
            settings=settings,
            scenario_name=args.scenario,
            loop_timestamp=loop_timestamp,
            cycle_name=cycle_name,
            cycle_summary=cycle_summary,
        )
        print(f"[OK][live3] Cycle summary saved to: {cycle_summary_path}")

        if cycle_error:
            break

    run_info = build_run_info(full_summary)
    loop_summary = build_loop_summary(full_summary)
    scene_description_summary = build_scene_description_summary(full_summary)
    vlm_planning_summary = build_vlm_planning_summary(full_summary)
    simultaneous_actions_summary = build_simultaneous_actions_summary(full_summary)
    validator_summary = build_validator_summary(full_summary)
    full_pipeline_summary = build_full_pipeline_summary(full_summary)

    summary_paths = save_validation_loop_artifacts(
        settings=settings,
        scenario_name=args.scenario,
        loop_timestamp=loop_timestamp,
        run_info=run_info,
        loop_summary=loop_summary,
        scene_description_summary=scene_description_summary,
        vlm_planning_summary=vlm_planning_summary,
        simultaneous_actions_summary=simultaneous_actions_summary,
        validator_summary=validator_summary,
        full_pipeline_summary=full_pipeline_summary,
    )

    print("\n======================================================")
    print("LIVE3 COMPLETED")
    print(f"Scenario:                  {args.scenario}")
    print(f"Loop timestamp:            {loop_timestamp}")
    print(f"Task completed:            {full_summary['task_completed']}")
    print(f"Replans done:              {full_summary['replans_done']}")
    print(f"Run info saved:            {summary_paths['run_info']}")
    print(f"Attempt history saved:     {summary_paths['attempt_history']}")
    print(f"Loop summary saved:        {summary_paths['loop_summary']}")
    print(f"Full summary saved:        {summary_paths['full_pipeline_summary']}")
    print("======================================================")


if __name__ == "__main__":
    run_with_terminal_log()
