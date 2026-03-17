import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
# ============================================================

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


# ============================================================
# 2. DATA STRUCTURES
# ============================================================

@dataclass
class EvaluationResult:
    scene_id: str
    task_id: str
    model: str
    prompt: str
    input_type: str
    run_id: int

    json_valid: int
    schema_ok: int
    taxonomy_ok: int

    step_count: int
    manipulation_consistency: float
    action_feasibility: float
    plan_correct: int

    hallucination_rate: float
    constraint_violation_rate: float

    parse_error: Optional[str] = None


# ============================================================
# 3. JSON / SCHEMA HELPERS
# ============================================================

def try_parse_json(raw_output: str) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    """
    Try to parse model output as JSON.
    Returns: (parsed_json, json_valid, parse_error)
    """
    try:
        parsed = json.loads(raw_output)
        if not isinstance(parsed, dict):
            return None, 0, "Parsed JSON is not an object"
        return parsed, 1, None
    except Exception as e:
        return None, 0, str(e)


def check_required_fields(parsed: Dict[str, Any]) -> int:
    """
    Minimal schema compliance check for the first prompt structure.
    Returns 1 if schema is minimally valid, else 0.
    """
    required_top = {"task_interpretation", "scene_analysis", "plan"}
    if not required_top.issubset(parsed.keys()):
        return 0

    scene_analysis = parsed.get("scene_analysis")
    if not isinstance(scene_analysis, dict):
        return 0

    required_scene = {
        "target_object",
        "target_attributes",
        "blocking_objects",
        "relevant_relations",
        "requires_clearing",
    }
    if not required_scene.issubset(scene_analysis.keys()):
        return 0

    plan = parsed.get("plan")
    if not isinstance(plan, dict):
        return 0

    if "steps" not in plan or not isinstance(plan["steps"], list):
        return 0

    for step in plan["steps"]:
        if not isinstance(step, dict):
            return 0

        required_step = {
            "step_id",
            "subgoal",
            "manipulation_mode",
            "arm_actions",
            "preconditions",
            "postconditions",
        }
        if not required_step.issubset(step.keys()):
            return 0

        if not isinstance(step["arm_actions"], dict):
            return 0

        if "left_arm" not in step["arm_actions"] or "right_arm" not in step["arm_actions"]:
            return 0

        if not isinstance(step["preconditions"], list):
            return 0
        if not isinstance(step["postconditions"], list):
            return 0

    return 1


def check_taxonomy(parsed: Dict[str, Any]) -> int:
    """
    Checks whether all taxonomy fields are within allowed sets.
    Returns 1 if all valid, else 0.
    """
    plan = parsed.get("plan", {})
    steps = plan.get("steps", [])
    if not isinstance(steps, list):
        return 0

    for step in steps:
        manipulation_mode = step.get("manipulation_mode")
        if manipulation_mode not in ALLOWED_MANIPULATION_MODES:
            return 0

        arm_actions = step.get("arm_actions", {})
        for arm_name in ["left_arm", "right_arm"]:
            arm_action = arm_actions.get(arm_name)
            if arm_action is None:
                continue
            if not isinstance(arm_action, dict):
                return 0

            action_primitive = arm_action.get("action_primitive")
            contact_mode = arm_action.get("contact_mode")

            if action_primitive not in ALLOWED_ACTION_PRIMITIVES:
                return 0
            if contact_mode not in ALLOWED_CONTACT_MODES:
                return 0

    return 1


# ============================================================
# 4. SCENE HELPERS
# ============================================================

def normalize_object_name(name: str) -> str:
    """
    Normalize names like 'cup_1' -> 'cup_1', trim spaces, lower-case.
    """
    return str(name).strip().lower()


def extract_scene_object_info(scene_objects: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a dictionary keyed by object name.
    """
    obj_map = {}
    for obj in scene_objects:
        name = normalize_object_name(obj.get("name", ""))
        if name:
            obj_map[name] = obj
    return obj_map


def get_all_scene_object_names(scene_objects: List[Dict[str, Any]]) -> Set[str]:
    return set(extract_scene_object_info(scene_objects).keys())


def infer_object_graspable_with_claw(obj: Dict[str, Any]) -> Optional[bool]:
    """
    Heuristic parser for 'graspable with claw'.
    """
    value = obj.get("graspable with claw")
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    s = str(value).strip().lower()
    if s in {"yes", "true"}:
        return True
    if s in {"no", "false"}:
        return False
    return None


def infer_object_size_large(obj: Dict[str, Any]) -> Optional[bool]:
    """
    Very simple heuristic:
    - if explicitly marked not graspable with claw, assume likely large/non-graspable
    - otherwise unknown
    """
    graspable = infer_object_graspable_with_claw(obj)
    if graspable is False:
        return True
    if graspable is True:
        return False
    return None


# ============================================================
# 5. METRIC HELPERS
# ============================================================

def get_steps(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    return parsed.get("plan", {}).get("steps", [])


def collect_objects_referenced_in_plan(parsed: Dict[str, Any]) -> Set[str]:
    """
    Collect object names referenced in:
    - scene_analysis fields
    - arm_actions target_object
    """
    referenced = set()

    scene_analysis = parsed.get("scene_analysis", {})
    if isinstance(scene_analysis, dict):
        target_object = scene_analysis.get("target_object")
        if isinstance(target_object, str):
            referenced.add(normalize_object_name(target_object))

        for field in ["blocking_objects"]:
            values = scene_analysis.get(field, [])
            if isinstance(values, list):
                for v in values:
                    if isinstance(v, str):
                        referenced.add(normalize_object_name(v))

    for step in get_steps(parsed):
        arm_actions = step.get("arm_actions", {})
        for arm_name in ["left_arm", "right_arm"]:
            arm_action = arm_actions.get(arm_name)
            if isinstance(arm_action, dict):
                target_object = arm_action.get("target_object")
                if isinstance(target_object, str):
                    referenced.add(normalize_object_name(target_object))

    return referenced


def compute_hallucination_rate(parsed: Dict[str, Any], scene_objects: List[Dict[str, Any]]) -> float:
    scene_names = get_all_scene_object_names(scene_objects)
    referenced = collect_objects_referenced_in_plan(parsed)

    # Exclude empty set case
    if len(referenced) == 0:
        return 0.0

    hallucinated = [obj for obj in referenced if obj not in scene_names]
    return len(hallucinated) / len(referenced)


def check_arm_mode_consistency(step: Dict[str, Any]) -> int:
    """
    Returns 1 if the step respects the arm-mode rules, else 0.
    """
    mode = step.get("manipulation_mode")
    arm_actions = step.get("arm_actions", {})
    left = arm_actions.get("left_arm")
    right = arm_actions.get("right_arm")

    left_is_null = left is None
    right_is_null = right is None

    if mode == "single_arm":
        # Exactly one null, one filled
        return int((left_is_null and not right_is_null) or (not left_is_null and right_is_null))

    if mode == "dual_arm_symmetric":
        if left_is_null or right_is_null:
            return 0
        if not isinstance(left, dict) or not isinstance(right, dict):
            return 0
        same_target = left.get("target_object") == right.get("target_object")
        same_action = left.get("action_primitive") == right.get("action_primitive")
        return int(same_target and same_action)

    if mode == "dual_arm_asymmetric":
        if left_is_null or right_is_null:
            return 0
        if not isinstance(left, dict) or not isinstance(right, dict):
            return 0
        different_targets = left.get("target_object") != right.get("target_object")
        return int(different_targets)

    return 0


def check_object_existence_in_step(step: Dict[str, Any], scene_objects: List[Dict[str, Any]]) -> int:
    scene_names = get_all_scene_object_names(scene_objects)
    arm_actions = step.get("arm_actions", {})
    referenced = []

    for arm_name in ["left_arm", "right_arm"]:
        arm_action = arm_actions.get(arm_name)
        if isinstance(arm_action, dict):
            obj_name = arm_action.get("target_object")
            if isinstance(obj_name, str):
                referenced.append(normalize_object_name(obj_name))

    if not referenced:
        return 0

    return int(all(obj in scene_names for obj in referenced))


def check_action_feasibility_step(step: Dict[str, Any], scene_objects: List[Dict[str, Any]]) -> int:
    """
    Very simple feasibility heuristics:
    - if object is marked not graspable with claw, then with_claw is invalid
    - if object appears large, with_claw is invalid
    """
    obj_map = extract_scene_object_info(scene_objects)
    arm_actions = step.get("arm_actions", {})

    checked_any = False

    for arm_name in ["left_arm", "right_arm"]:
        arm_action = arm_actions.get(arm_name)
        if not isinstance(arm_action, dict):
            continue

        checked_any = True

        obj_name = normalize_object_name(arm_action.get("target_object", ""))
        action_primitive = arm_action.get("action_primitive")
        contact_mode = arm_action.get("contact_mode")

        if obj_name not in obj_map:
            return 0

        obj = obj_map[obj_name]
        graspable = infer_object_graspable_with_claw(obj)
        large = infer_object_size_large(obj)

        if contact_mode == "with_claw":
            if graspable is False:
                return 0
            if large is True:
                return 0

        if action_primitive == "grasp" and graspable is False:
            return 0

    return int(checked_any)


def compute_manipulation_consistency(parsed: Dict[str, Any], scene_objects: List[Dict[str, Any]]) -> float:
    steps = get_steps(parsed)
    if len(steps) == 0:
        return 0.0

    valid_steps = 0
    for step in steps:
        arm_ok = check_arm_mode_consistency(step)
        obj_ok = check_object_existence_in_step(step, scene_objects)
        if arm_ok and obj_ok:
            valid_steps += 1

    return valid_steps / len(steps)


def compute_action_feasibility(parsed: Dict[str, Any], scene_objects: List[Dict[str, Any]]) -> float:
    steps = get_steps(parsed)
    if len(steps) == 0:
        return 0.0

    feasible_steps = 0
    for step in steps:
        feasible_steps += check_action_feasibility_step(step, scene_objects)

    return feasible_steps / len(steps)


def compute_constraint_violation_rate(parsed: Dict[str, Any], scene_objects: List[Dict[str, Any]]) -> float:
    """
    Counts violations over steps.
    Violations considered:
    - arm-mode inconsistency
    - object not in scene
    - infeasible action
    """
    steps = get_steps(parsed)
    if len(steps) == 0:
        return 1.0

    violations = 0
    for step in steps:
        arm_ok = check_arm_mode_consistency(step)
        obj_ok = check_object_existence_in_step(step, scene_objects)
        feasible_ok = check_action_feasibility_step(step, scene_objects)

        if not (arm_ok and obj_ok and feasible_ok):
            violations += 1

    return violations / len(steps)


def compute_step_count(parsed: Dict[str, Any]) -> int:
    return len(get_steps(parsed))


def compute_plan_correctness(
    parsed: Dict[str, Any],
    task_text: str,
    scene_objects: List[Dict[str, Any]],
) -> int:
    """
    Placeholder heuristic for plan correctness.
    Realistically, this can later be replaced by:
    - human annotation
    - validator output
    - task-specific rule set

    For now:
    - require at least one step
    - require low hallucination
    - require decent consistency and feasibility
    """
    step_count = compute_step_count(parsed)
    if step_count == 0:
        return 0

    hallucination = compute_hallucination_rate(parsed, scene_objects)
    consistency = compute_manipulation_consistency(parsed, scene_objects)
    feasibility = compute_action_feasibility(parsed, scene_objects)

    if hallucination > 0.0:
        return 0
    if consistency < 1.0:
        return 0
    if feasibility < 1.0:
        return 0

    # Basic placeholder
    return 1


# ============================================================
# 6. EVALUATION OF A SINGLE RUN
# ============================================================

def evaluate_single_run(
    scene_id: str,
    task_id: str,
    model: str,
    prompt: str,
    input_type: str,
    run_id: int,
    task_text: str,
    scene_objects: List[Dict[str, Any]],
    scene_relations: List[Dict[str, Any]],
    raw_output: str,
) -> EvaluationResult:
    parsed, json_valid, parse_error = try_parse_json(raw_output)

    if not json_valid or parsed is None:
        return EvaluationResult(
            scene_id=scene_id,
            task_id=task_id,
            model=model,
            prompt=prompt,
            input_type=input_type,
            run_id=run_id,
            json_valid=0,
            schema_ok=0,
            taxonomy_ok=0,
            step_count=0,
            manipulation_consistency=0.0,
            action_feasibility=0.0,
            plan_correct=0,
            hallucination_rate=1.0,
            constraint_violation_rate=1.0,
            parse_error=parse_error,
        )

    schema_ok = check_required_fields(parsed)
    taxonomy_ok = check_taxonomy(parsed) if schema_ok else 0

    if not schema_ok:
        return EvaluationResult(
            scene_id=scene_id,
            task_id=task_id,
            model=model,
            prompt=prompt,
            input_type=input_type,
            run_id=run_id,
            json_valid=1,
            schema_ok=0,
            taxonomy_ok=0,
            step_count=0,
            manipulation_consistency=0.0,
            action_feasibility=0.0,
            plan_correct=0,
            hallucination_rate=1.0,
            constraint_violation_rate=1.0,
            parse_error=None,
        )

    step_count = compute_step_count(parsed)
    manipulation_consistency = compute_manipulation_consistency(parsed, scene_objects)
    action_feasibility = compute_action_feasibility(parsed, scene_objects)
    hallucination_rate = compute_hallucination_rate(parsed, scene_objects)
    constraint_violation_rate = compute_constraint_violation_rate(parsed, scene_objects)
    plan_correct = compute_plan_correctness(parsed, task_text, scene_objects)

    return EvaluationResult(
        scene_id=scene_id,
        task_id=task_id,
        model=model,
        prompt=prompt,
        input_type=input_type,
        run_id=run_id,
        json_valid=json_valid,
        schema_ok=schema_ok,
        taxonomy_ok=taxonomy_ok,
        step_count=step_count,
        manipulation_consistency=manipulation_consistency,
        action_feasibility=action_feasibility,
        plan_correct=plan_correct,
        hallucination_rate=hallucination_rate,
        constraint_violation_rate=constraint_violation_rate,
        parse_error=None,
    )


# ============================================================
# 7. DATAFRAME UTILITIES
# ============================================================

def results_to_dataframe(results: List[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def add_setting_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["setting"] = (
        df["model"].astype(str)
        + " | "
        + df["prompt"].astype(str)
        + " | "
        + df["input_type"].astype(str)
    )
    return df


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "json_valid",
        "schema_ok",
        "taxonomy_ok",
        "step_count",
        "manipulation_consistency",
        "action_feasibility",
        "plan_correct",
        "hallucination_rate",
        "constraint_violation_rate",
    ]

    grouped = df.groupby(["model", "prompt", "input_type"], as_index=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns.to_flat_index()
    ]
    return grouped


# ============================================================
# 8. PLOTTING
# ============================================================

def plot_output_validity(df: pd.DataFrame) -> None:
    """
    Grouped bar plot for:
    - json_valid
    - schema_ok
    - taxonomy_ok
    """
    df = add_setting_column(df)

    summary = (
        df.groupby("setting")[["json_valid", "schema_ok", "taxonomy_ok"]]
        .mean()
        .reset_index()
    )

    x = np.arange(len(summary))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, summary["json_valid"], width=width, label="JSON validity")
    plt.bar(x, summary["schema_ok"], width=width, label="Schema compliance")
    plt.bar(x + width, summary["taxonomy_ok"], width=width, label="Taxonomy compliance")

    plt.xticks(x, summary["setting"], rotation=45, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Output validity metrics")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_step_count_boxplot(df: pd.DataFrame) -> None:
    df = add_setting_column(df)

    settings = df["setting"].unique().tolist()
    data = [df[df["setting"] == s]["step_count"].values for s in settings]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, tick_labels=settings)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of steps")
    plt.title("Step count distribution across settings")
    plt.tight_layout()
    plt.show()


def plot_plan_quality(df: pd.DataFrame) -> None:
    """
    Bar plot with error bars for:
    - manipulation_consistency
    - action_feasibility
    - plan_correct
    """
    df = add_setting_column(df)

    means = (
        df.groupby("setting")[["manipulation_consistency", "action_feasibility", "plan_correct"]]
        .mean()
        .reset_index()
    )

    stds = (
        df.groupby("setting")[["manipulation_consistency", "action_feasibility", "plan_correct"]]
        .std()
        .reset_index()
    )

    x = np.arange(len(means))
    width = 0.25

    plt.figure(figsize=(13, 6))
    plt.bar(
        x - width,
        means["manipulation_consistency"],
        yerr=stds["manipulation_consistency"],
        width=width,
        label="Manipulation consistency",
        capsize=4,
    )
    plt.bar(
        x,
        means["action_feasibility"],
        yerr=stds["action_feasibility"],
        width=width,
        label="Action feasibility",
        capsize=4,
    )
    plt.bar(
        x + width,
        means["plan_correct"],
        yerr=stds["plan_correct"],
        width=width,
        label="Plan correctness",
        capsize=4,
    )

    plt.xticks(x, means["setting"], rotation=45, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Plan quality metrics")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_grounding(df: pd.DataFrame) -> None:
    df = add_setting_column(df)

    summary = (
        df.groupby("setting")[["hallucination_rate"]]
        .mean()
        .reset_index()
    )

    x = np.arange(len(summary))

    plt.figure(figsize=(12, 6))
    plt.bar(x, summary["hallucination_rate"])
    plt.xticks(x, summary["setting"], rotation=45, ha="right")
    plt.ylabel("Hallucination rate")
    plt.ylim(0, 1.05)
    plt.title("Object hallucination rate")
    plt.tight_layout()
    plt.show()


def plot_constraint_violations(df: pd.DataFrame) -> None:
    df = add_setting_column(df)

    summary = (
        df.groupby("setting")[["constraint_violation_rate"]]
        .mean()
        .reset_index()
    )

    x = np.arange(len(summary))

    plt.figure(figsize=(12, 6))
    plt.bar(x, summary["constraint_violation_rate"])
    plt.xticks(x, summary["setting"], rotation=45, ha="right")
    plt.ylabel("Constraint violation rate")
    plt.ylim(0, 1.05)
    plt.title("Constraint violations across settings")
    plt.tight_layout()
    plt.show()


def plot_plan_stability(df: pd.DataFrame) -> None:
    """
    Plan stability = mean of json_valid * schema_ok * taxonomy_ok
    or, if preferred, fraction of runs with a valid structured plan.
    """
    df = add_setting_column(df)
    df = df.copy()
    df["plan_valid"] = df["json_valid"] * df["schema_ok"] * df["taxonomy_ok"]

    summary = (
        df.groupby("setting")[["plan_valid"]]
        .mean()
        .reset_index()
    )

    x = np.arange(len(summary))

    plt.figure(figsize=(12, 6))
    plt.bar(x, summary["plan_valid"])
    plt.xticks(x, summary["setting"], rotation=45, ha="right")
    plt.ylabel("Plan stability")
    plt.ylim(0, 1.05)
    plt.title("Plan stability across repeated runs")
    plt.tight_layout()
    plt.show()


def plot_metric_heatmap(df: pd.DataFrame) -> None:
    """
    Simple matplotlib heatmap for summary metrics.
    """
    df = add_setting_column(df)

    summary = (
        df.groupby("setting")[
            [
                "json_valid",
                "schema_ok",
                "taxonomy_ok",
                "manipulation_consistency",
                "action_feasibility",
                "plan_correct",
                "hallucination_rate",
                "constraint_violation_rate",
            ]
        ]
        .mean()
    )

    matrix = summary.values
    row_labels = summary.index.tolist()
    col_labels = summary.columns.tolist()

    plt.figure(figsize=(12, 6))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Score")

    plt.xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)

    plt.title("Summary heatmap")
    plt.tight_layout()
    plt.show()


# ============================================================
# 9. EXAMPLE USAGE
# ============================================================

def main():
    # Example dataset with two fake runs
    runs = [
        {
            "scene_id": "scene_001",
            "task_id": "task_001",
            "model": "model_A",
            "prompt": "prompt_1",
            "input_type": "structured",
            "run_id": 1,
            "task_text": "move the cup behind the box",
            "scene_objects": [
                {"name": "box_1", "color": "dark brown", "graspable with claw": "no"},
                {"name": "box_2", "color": "light brown", "graspable with claw": "no"},
                {"name": "cup_1", "color": "black", "graspable with claw": "yes"},
                {"name": "table", "color": "brown"},
            ],
            "scene_relations": [
                {"object_1": "cup_1", "relation": "in_front_of", "object_2": "box_2"},
            ],
            "raw_output": json.dumps({
                "task_interpretation": "move cup_1 behind box_2",
                "scene_analysis": {
                    "target_object": "cup_1",
                    "target_attributes": ["black"],
                    "blocking_objects": [],
                    "relevant_relations": ["cup_1 in_front_of box_2"],
                    "requires_clearing": False
                },
                "plan": {
                    "steps": [
                        {
                            "step_id": 1,
                            "subgoal": "move cup_1 behind box_2",
                            "manipulation_mode": "single_arm",
                            "arm_actions": {
                                "left_arm": None,
                                "right_arm": {
                                    "target_object": "cup_1",
                                    "action_primitive": "grasp",
                                    "contact_mode": "with_claw"
                                }
                            },
                            "preconditions": ["cup_1 is reachable"],
                            "postconditions": ["cup_1 is grasped"]
                        }
                    ]
                }
            }),
        },
        {
            "scene_id": "scene_001",
            "task_id": "task_001",
            "model": "model_A",
            "prompt": "prompt_1",
            "input_type": "structured",
            "run_id": 2,
            "task_text": "move the cup behind the box",
            "scene_objects": [
                {"name": "box_1", "color": "dark brown", "graspable with claw": "no"},
                {"name": "box_2", "color": "light brown", "graspable with claw": "no"},
                {"name": "cup_1", "color": "black", "graspable with claw": "yes"},
                {"name": "table", "color": "brown"},
            ],
            "scene_relations": [
                {"object_1": "cup_1", "relation": "in_front_of", "object_2": "box_2"},
            ],
            "raw_output": json.dumps({
                "task_interpretation": "move cup_1 behind box_2",
                "scene_analysis": {
                    "target_object": "cup_1",
                    "target_attributes": ["black"],
                    "blocking_objects": [],
                    "relevant_relations": ["cup_1 in_front_of box_2"],
                    "requires_clearing": False
                },
                "plan": {
                    "steps": [
                        {
                            "step_id": 1,
                            "subgoal": "push cup_1 behind box_2",
                            "manipulation_mode": "single_arm",
                            "arm_actions": {
                                "left_arm": {
                                    "target_object": "cup_1",
                                    "action_primitive": "push",
                                    "contact_mode": "without_claw"
                                },
                                "right_arm": None
                            },
                            "preconditions": ["cup_1 is reachable"],
                            "postconditions": ["cup_1 moved backward"]
                        }
                    ]
                }
            }),
        },
    ]

    results = []
    for run in runs:
        result = evaluate_single_run(
            scene_id=run["scene_id"],
            task_id=run["task_id"],
            model=run["model"],
            prompt=run["prompt"],
            input_type=run["input_type"],
            run_id=run["run_id"],
            task_text=run["task_text"],
            scene_objects=run["scene_objects"],
            scene_relations=run["scene_relations"],
            raw_output=run["raw_output"],
        )
        results.append(result)

    df = results_to_dataframe(results)
    print("\n=== Per-run results ===")
    print(df)

    agg = aggregate_metrics(df)
    print("\n=== Aggregated metrics ===")
    print(agg)

    plot_output_validity(df)
    plot_step_count_boxplot(df)
    plot_plan_quality(df)
    plot_grounding(df)
    plot_constraint_violations(df)
    plot_plan_stability(df)
    plot_metric_heatmap(df)


if __name__ == "__main__":
    main()