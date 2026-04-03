from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_planner_scenarios(scenarios_path: str) -> list[dict[str, Any]]:
    path = Path(scenarios_path)
    if not path.exists():
        raise FileNotFoundError(f"Scenarios file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if "scenarios" not in data or not isinstance(data["scenarios"], list):
        raise ValueError("scenarios.json must contain a top-level 'scenarios' list")

    return data["scenarios"]


def load_validator_cases(root_dir: str) -> list[dict[str, Any]]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Validator scenarios dir not found: {root}")

    all_cases: list[dict[str, Any]] = []

    for scenario_dir in root.iterdir():
        if not scenario_dir.is_dir():
            continue

        cases_file = scenario_dir / "cases.json"
        scene_description_file = scenario_dir / "scene_description.json"

        if not cases_file.exists():
            continue
        if not scene_description_file.exists():
            raise FileNotFoundError(
                f"Missing scene_description.json in {scenario_dir}"
            )

        cases_data = json.loads(cases_file.read_text(encoding="utf-8"))
        scene_description = json.loads(scene_description_file.read_text(encoding="utf-8"))

        scenario_id = cases_data.get("scenario_id", scenario_dir.name)
        image_name = cases_data.get("image")
        image_path = str((scenario_dir / image_name).resolve()) if image_name else None

        cases = cases_data.get("cases")
        if not isinstance(cases, list):
            raise ValueError(f"'cases' must be a list in {cases_file}")

        for case in cases:
            case_id = case.get("case_id")
            condition_to_check = case.get("condition_to_check")

            if not case_id:
                raise ValueError(f"Missing case_id in {cases_file}")
            if not condition_to_check:
                raise ValueError(f"Missing condition_to_check in {cases_file}")

            all_cases.append(
                {
                    "scenario_id": scenario_id,
                    "case_id": case_id,
                    "image_path": image_path,
                    "scene_description": scene_description,
                    "condition_to_check": condition_to_check,
                    "action_context": case.get("action_context"),
                }
            )

    return all_cases