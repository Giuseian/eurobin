from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_ROOT = PROJECT_ROOT / "evaluation_outputs"
RAW_PARSED_DIR = EVAL_ROOT / "raw_parsed"
METRICS_DIR = EVAL_ROOT / "metrics"
AGGREGATED_DIR = EVAL_ROOT / "aggregated_results"

INPUT_JSONL = RAW_PARSED_DIR / "per_run_raw.jsonl"
PER_RUN_CSV = METRICS_DIR / "per_run_metrics.csv"
AGG_CSV = AGGREGATED_DIR / "aggregated_metrics.csv"


ALLOWED_ACTION_PRIMITIVES = {
    "grasp",
    "place",
    "push",
    "stabilize",
    "open",
    "close",
}

ALLOWED_MANIPULATION_MODES = {
    "single_arm",
    "dual_arm_symmetric",
    "dual_arm_asymmetric",
}

ALLOWED_CONTACT_MODES = {
    "with_claw",
    "without_claw",
}


# ----------------------------
# Utility
# ----------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_nonempty_string(x: Any) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0


def is_list(x: Any) -> bool:
    return isinstance(x, list)


def is_dict(x: Any) -> bool:
    return isinstance(x, dict)


def is_arm_action_valid_shape(action: Any) -> bool:
    if action is None:
        return True
    if not isinstance(action, dict):
        return False
    required = {"target_object", "action_primitive", "contact_mode"}
    if not required.issubset(action.keys()):
        return False
    if not is_nonempty_string(action["target_object"]):
        return False
    if not is_nonempty_string(action["action_primitive"]):
        return False
    if not is_nonempty_string(action["contact_mode"]):
        return False
    return True


# ----------------------------
# Schema checks
# ----------------------------

def check_schema(parsed_json: Any) -> bool:
    if not is_dict(parsed_json):
        return False

    required_top = {"task_interpretation", "scene_analysis", "plan"}
    if not required_top.issubset(parsed_json.keys()):
        return False

    scene_analysis = parsed_json.get("scene_analysis")
    plan = parsed_json.get("plan")

    if not is_dict(scene_analysis) or not is_dict(plan):
        return False

    required_scene = {
        "target_object",
        "target_attributes",
        "blocking_objects",
        "relevant_relations",
        "requires_clearing",
    }
    if not required_scene.issubset(scene_analysis.keys()):
        return False

    if not is_nonempty_string(scene_analysis["target_object"]):
        return False
    if not is_list(scene_analysis["target_attributes"]):
        return False
    if not is_list(scene_analysis["blocking_objects"]):
        return False
    if not is_list(scene_analysis["relevant_relations"]):
        return False
    if not isinstance(scene_analysis["requires_clearing"], bool):
        return False

    if "steps" not in plan or not is_list(plan["steps"]):
        return False

    for step in plan["steps"]:
        if not is_dict(step):
            return False

        required_step = {
            "step_id",
            "subgoal",
            "manipulation_mode",
            "arm_actions",
            "preconditions",
            "postconditions",
        }
        if not required_step.issubset(step.keys()):
            return False

        if not isinstance(step["step_id"], int):
            return False
        if not is_nonempty_string(step["subgoal"]):
            return False
        if not is_nonempty_string(step["manipulation_mode"]):
            return False
        if not is_list(step["preconditions"]):
            return False
        if not is_list(step["postconditions"]):
            return False

        arm_actions = step["arm_actions"]
        if not is_dict(arm_actions):
            return False
        if "left_arm" not in arm_actions or "right_arm" not in arm_actions:
            return False

        if not is_arm_action_valid_shape(arm_actions["left_arm"]):
            return False
        if not is_arm_action_valid_shape(arm_actions["right_arm"]):
            return False

    return True


# ----------------------------
# Taxonomy checks
# ----------------------------

def check_taxonomy(parsed_json: Any) -> bool:
    if not check_schema(parsed_json):
        return False

    steps = parsed_json["plan"]["steps"]

    for step in steps:
        mode = step["manipulation_mode"]
        if mode not in ALLOWED_MANIPULATION_MODES:
            return False

        for arm_name in ("left_arm", "right_arm"):
            action = step["arm_actions"][arm_name]
            if action is None:
                continue

            if action["action_primitive"] not in ALLOWED_ACTION_PRIMITIVES:
                return False

            if action["contact_mode"] not in ALLOWED_CONTACT_MODES:
                return False

    return True


# ----------------------------
# Step-level metrics
# ----------------------------

def get_steps(parsed_json: Any) -> list[dict[str, Any]]:
    if not check_schema(parsed_json):
        return []
    return parsed_json["plan"]["steps"]


def compute_step_count(parsed_json: Any) -> float:
    steps = get_steps(parsed_json)
    if not steps:
        return float("nan")
    return float(len(steps))


def check_step_manipulation_consistency(step: dict[str, Any]) -> bool:
    mode = step["manipulation_mode"]
    left = step["arm_actions"]["left_arm"]
    right = step["arm_actions"]["right_arm"]

    left_active = left is not None
    right_active = right is not None

    if not left_active and not right_active:
        return False

    if mode == "single_arm":
        # esattamente uno attivo
        return (left_active ^ right_active)

    if mode == "dual_arm_symmetric":
        if not (left_active and right_active):
            return False
        return (
            left["target_object"] == right["target_object"]
            and left["action_primitive"] == right["action_primitive"]
        )

    if mode == "dual_arm_asymmetric":
        if not (left_active and right_active):
            return False
        return left["target_object"] != right["target_object"]

    return False


def compute_manipulation_consistency(parsed_json: Any) -> float:
    steps = get_steps(parsed_json)
    if not steps:
        return 0.0

    valid_steps = sum(check_step_manipulation_consistency(step) for step in steps)
    return valid_steps / len(steps)


def check_step_action_feasibility(step: dict[str, Any]) -> bool:
    """
    Prima versione semplice:
    - step ben formato
    - manipulation_mode valido
    - almeno un braccio attivo
    - action fields presenti e nella taxonomy
    - preconditions/postconditions sono liste
    """
    if step["manipulation_mode"] not in ALLOWED_MANIPULATION_MODES:
        return False

    if not is_list(step["preconditions"]):
        return False
    if not is_list(step["postconditions"]):
        return False

    left = step["arm_actions"]["left_arm"]
    right = step["arm_actions"]["right_arm"]

    if left is None and right is None:
        return False

    for action in (left, right):
        if action is None:
            continue

        if action["action_primitive"] not in ALLOWED_ACTION_PRIMITIVES:
            return False
        if action["contact_mode"] not in ALLOWED_CONTACT_MODES:
            return False
        if not is_nonempty_string(action["target_object"]):
            return False

    return True


def compute_action_feasibility(parsed_json: Any) -> float:
    steps = get_steps(parsed_json)
    if not steps:
        return 0.0

    valid_steps = sum(check_step_action_feasibility(step) for step in steps)
    return valid_steps / len(steps)


# ----------------------------
# Per-run metric computation
# ----------------------------

def compute_metrics_for_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed_json = row.get("parsed_json")
    json_valid = 1 if row.get("json_parse_ok") is True else 0

    schema_ok = 1 if check_schema(parsed_json) else 0
    taxonomy_ok = 1 if check_taxonomy(parsed_json) else 0
    step_count = compute_step_count(parsed_json)
    manipulation_consistency = compute_manipulation_consistency(parsed_json)
    action_feasibility = compute_action_feasibility(parsed_json)

    return {
        "source_file": row.get("source_file"),
        "scenario_id": row.get("scenario_id"),
        "task_text": row.get("task_text"),
        "model_name": row.get("model_name"),
        "prompt_filename": row.get("prompt_filename"),
        "repeat_idx": row.get("repeat_idx"),
        "image_path": row.get("image_path"),
        "input_mode": row.get("input_mode"),
        "json_valid": json_valid,
        "schema_ok": schema_ok,
        "taxonomy_ok": taxonomy_ok,
        "step_count": step_count,
        "manipulation_consistency": manipulation_consistency,
        "action_feasibility": action_feasibility,
        "inference_time_sec": row.get("inference_time_sec"),
        "is_error_file": row.get("is_error_file"),
    }


# ----------------------------
# Aggregation
# ----------------------------

def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_name", "prompt_filename", "input_mode"]

    agg_df = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("json_valid", "size"),
            json_valid_mean=("json_valid", "mean"),
            schema_ok_mean=("schema_ok", "mean"),
            taxonomy_ok_mean=("taxonomy_ok", "mean"),
            step_count_mean=("step_count", "mean"),
            step_count_std=("step_count", "std"),
            manipulation_consistency_mean=("manipulation_consistency", "mean"),
            manipulation_consistency_std=("manipulation_consistency", "std"),
            action_feasibility_mean=("action_feasibility", "mean"),
            action_feasibility_std=("action_feasibility", "std"),
            inference_time_mean=("inference_time_sec", "mean"),
            inference_time_std=("inference_time_sec", "std"),
        )
        .reset_index()
    )

    return agg_df


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_JSONL}")

    rows = load_jsonl(INPUT_JSONL)
    metrics_rows = [compute_metrics_for_row(row) for row in rows]

    df = pd.DataFrame(metrics_rows)
    agg_df = aggregate_metrics(df)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PER_RUN_CSV, index=False)
    agg_df.to_csv(AGG_CSV, index=False)

    print(f"Loaded runs: {len(rows)}")
    print(f"Saved per-run metrics to: {PER_RUN_CSV}")
    print(f"Saved aggregated metrics to: {AGG_CSV}")


if __name__ == "__main__":
    main()