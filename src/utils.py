from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from settings import Settings


VALID_MODULES = {
    "scene_description",
    "vlm_planning",
    "simultaneous_actions",
    "validator",
}


def validate_module_name(module_name: str) -> None:
    if module_name not in VALID_MODULES:
        allowed = ", ".join(sorted(VALID_MODULES))
        raise ValueError(f"Invalid module '{module_name}'. Allowed values: {allowed}")


def make_experiment_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def make_run_name(repeat_idx: int) -> str:
    return f"run_{repeat_idx:03d}"


def make_cycle_name(cycle_idx: int) -> str:
    return f"cycle_{cycle_idx:03d}"


def make_stage_name(stage_id: int) -> str:
    return f"stage_{stage_id:03d}"


def get_base_prompt_path(settings: Settings, module_name: str, version: str) -> Path:
    validate_module_name(module_name)
    return settings.project_root / "prompts" / module_name / version / "prompt.txt"


def get_prompt_scenario_dir(
    settings: Settings,
    module_name: str,
    scenario_name: str,
    version: str,
    experiment_timestamp: str,
    model_name: str,
    run_name: str,
) -> Path:
    validate_module_name(module_name)
    return (
        settings.project_root
        / "prompts_scenarios"
        / module_name
        / scenario_name
        / version
        / experiment_timestamp
        / model_name
        / run_name
    )


def get_output_dir(
    settings: Settings,
    module_name: str,
    scenario_name: str,
    version: str,
    experiment_timestamp: str,
    model_name: str,
    run_name: str,
) -> Path:
    validate_module_name(module_name)
    return (
        settings.project_root
        / "outputs"
        / module_name
        / scenario_name
        / version
        / experiment_timestamp
        / model_name
        / run_name
    )


# ============================================================
# NEW: CYCLE-BASED PATH HELPERS
# ============================================================

def get_prompt_scenario_cycle_dir(
    settings: Settings,
    module_name: str,
    scenario_name: str,
    version: str,
    loop_timestamp: str,
    model_name: str,
    cycle_name: str,
) -> Path:
    validate_module_name(module_name)
    if module_name == "validator":
        raise ValueError(
            "Use get_validator_prompt_cycle_dir(...) for validator cycle-based prompt paths."
        )

    return (
        settings.project_root
        / "prompts_scenarios"
        / module_name
        / scenario_name
        / version
        / loop_timestamp
        / model_name
        / cycle_name
    )


def get_output_cycle_dir(
    settings: Settings,
    module_name: str,
    scenario_name: str,
    version: str,
    loop_timestamp: str,
    model_name: str,
    cycle_name: str,
) -> Path:
    validate_module_name(module_name)
    if module_name == "validator":
        raise ValueError(
            "Use get_validator_output_cycle_dir(...) for validator cycle-based output paths."
        )

    return (
        settings.project_root
        / "outputs"
        / module_name
        / scenario_name
        / version
        / loop_timestamp
        / model_name
        / cycle_name
    )


def get_validator_prompt_cycle_dir(
    settings: Settings,
    scenario_name: str,
    version: str,
    loop_timestamp: str,
    model_name: str,
    cycle_name: str,
    stage_name: str,
    condition_kind: str,
) -> Path:
    if condition_kind not in {"pre", "post"}:
        raise ValueError("condition_kind must be either 'pre' or 'post'.")

    return (
        settings.project_root
        / "prompts_scenarios"
        / "validator"
        / scenario_name
        / version
        / loop_timestamp
        / model_name
        / cycle_name
        / stage_name
        / condition_kind
    )


def get_validator_output_cycle_dir(
    settings: Settings,
    scenario_name: str,
    version: str,
    loop_timestamp: str,
    model_name: str,
    cycle_name: str,
    stage_name: str,
    condition_kind: str,
) -> Path:
    if condition_kind not in {"pre", "post"}:
        raise ValueError("condition_kind must be either 'pre' or 'post'.")

    return (
        settings.project_root
        / "outputs"
        / "validator"
        / scenario_name
        / version
        / loop_timestamp
        / model_name
        / cycle_name
        / stage_name
        / condition_kind
    )


def get_validation_loop_output_dir(
    settings: Settings,
    scenario_name: str,
    loop_timestamp: str,
) -> Path:
    return (
        settings.project_root
        / "outputs"
        / "validation_loop"
        / scenario_name
        / loop_timestamp
    )


def get_validation_loop_cycle_dir(
    settings: Settings,
    scenario_name: str,
    loop_timestamp: str,
    cycle_name: str,
) -> Path:
    return (
        get_validation_loop_output_dir(
            settings=settings,
            scenario_name=scenario_name,
            loop_timestamp=loop_timestamp,
        )
        / "cycles"
        / cycle_name
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_base_prompt(settings: Settings, module_name: str, version: str) -> str:
    prompt_path = get_base_prompt_path(settings, module_name, version)

    if not prompt_path.exists():
        raise FileNotFoundError(f"Base prompt not found: {prompt_path}")

    return read_text(prompt_path)


def try_parse_json(text: str) -> tuple[bool, Any]:
    text = text.strip()

    try:
        return True, json.loads(text)
    except Exception:
        pass

    extracted = extract_first_json_block(text)
    if extracted is not None:
        try:
            return True, json.loads(extracted)
        except Exception:
            pass

    return False, None


def extract_first_json_block(text: str) -> str | None:
    text = text.strip()

    array_match = re.search(r"(\[\s*[\s\S]*\])", text)
    if array_match:
        return array_match.group(1)

    obj_match = re.search(r"(\{\s*[\s\S]*\})", text)
    if obj_match:
        return obj_match.group(1)

    return None


def render_prompt(
    module_name: str,
    base_prompt: str,
    scenario_data: dict[str, Any],
    scene_description: Any | None = None,
    sequential_plan: Any | None = None,
) -> str:
    if module_name == "scene_description":
        return base_prompt.strip()

    if module_name == "vlm_planning":
        if scene_description is None:
            raise ValueError("scene_description is required for vlm_planning")

        task = scenario_data.get("task")
        if not task:
            raise ValueError("Scenario is missing 'task' for vlm_planning")

        scene_description_str = json.dumps(scene_description, indent=2, ensure_ascii=False)

        return (
            base_prompt.strip()
            + "\n\nSCENE DESCRIPTION\n"
            + scene_description_str
            + "\n\nTASK\n"
            + task
        )

    if module_name == "simultaneous_actions":
        if scene_description is None:
            raise ValueError("scene_description is required for simultaneous_actions")
        if sequential_plan is None:
            raise ValueError("sequential_plan is required for simultaneous_actions")

        scene_description_str = json.dumps(scene_description, indent=2, ensure_ascii=False)
        sequential_plan_str = json.dumps(sequential_plan, indent=2, ensure_ascii=False)

        return (
            base_prompt.strip()
            + "\n\nSCENE DESCRIPTION\n"
            + scene_description_str
            + "\n\nSEQUENTIAL PLAN\n"
            + sequential_plan_str
        )

    raise ValueError(f"Unsupported module for rendering: {module_name}")


def load_previous_module_output(
    settings: Settings,
    module_name: str,
    scenario_name: str,
    version: str,
    experiment_timestamp: str,
    model_name: str,
    run_name: str,
) -> Any:
    output_dir = get_output_dir(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
    )
    parsed_path = output_dir / "response_parsed.json"

    if not parsed_path.exists():
        raise FileNotFoundError(f"Previous module output not found: {parsed_path}")

    return read_json(parsed_path)


def load_previous_module_output_from_cycle(
    settings: Settings,
    module_name: str,
    scenario_name: str,
    version: str,
    loop_timestamp: str,
    model_name: str,
    cycle_name: str,
) -> Any:
    output_dir = get_output_cycle_dir(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=model_name,
        cycle_name=cycle_name,
    )
    parsed_path = output_dir / "response_parsed.json"

    if not parsed_path.exists():
        raise FileNotFoundError(f"Previous module output not found: {parsed_path}")

    return read_json(parsed_path)


def save_rendered_prompt(
    settings: Settings,
    module_name: str,
    scenario_name: str,
    version: str,
    experiment_timestamp: str,
    model_name: str,
    run_name: str,
    prompt_text: str,
) -> Path:
    prompt_dir = get_prompt_scenario_dir(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
    )
    ensure_dir(prompt_dir)

    prompt_path = prompt_dir / "prompt.txt"
    write_text(prompt_path, prompt_text)
    return prompt_path


def save_rendered_prompt_for_cycle(
    settings: Settings,
    module_name: str,
    scenario_name: str,
    version: str,
    loop_timestamp: str,
    model_name: str,
    cycle_name: str,
    prompt_text: str,
) -> Path:
    prompt_dir = get_prompt_scenario_cycle_dir(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=model_name,
        cycle_name=cycle_name,
    )
    ensure_dir(prompt_dir)

    prompt_path = prompt_dir / "prompt.txt"
    write_text(prompt_path, prompt_text)
    return prompt_path


def save_module_outputs(
    settings: Settings,
    module_name: str,
    scenario_name: str,
    version: str,
    experiment_timestamp: str,
    model_name: str,
    run_name: str,
    deployment_name: str,
    execution_time_seconds: float,
    scenario_data: dict[str, Any],
    parsed_response: Any,
    execution_mode: str,
    dependencies: dict[str, Any] | None = None,
    pipeline_config: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    output_dir = get_output_dir(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
    )
    ensure_dir(output_dir)

    parsed_path = output_dir / "response_parsed.json"
    run_info_path = output_dir / "run_info.json"

    write_json(parsed_path, parsed_response)

    run_info: dict[str, Any] = {
        "module": module_name,
        "execution_mode": execution_mode,
        "scenario_name": scenario_name,
        "prompt_version": version,
        "experiment_timestamp": experiment_timestamp,
        "run_name": run_name,
        "model": model_name,
        "deployment_name": deployment_name,
        "execution_time_seconds": execution_time_seconds,
        "timestamp": datetime.now().isoformat(),
        "scenario": {
            "task": scenario_data.get("task"),
            "image": scenario_data.get("image"),
            "image_path_abs": scenario_data.get("image_path_abs"),
        },
        "response_parsed": parsed_response,
    }

    if dependencies is not None:
        run_info["dependencies"] = dependencies

    if pipeline_config is not None:
        run_info["pipeline_config"] = pipeline_config

    write_json(run_info_path, run_info)

    return parsed_path, run_info_path


def save_module_outputs_for_cycle(
    settings: Settings,
    module_name: str,
    scenario_name: str,
    version: str,
    loop_timestamp: str,
    model_name: str,
    cycle_name: str,
    cycle_index: int,
    cycle_timestamp: str,
    deployment_name: str,
    execution_time_seconds: float,
    scenario_data: dict[str, Any],
    parsed_response: Any,
    execution_mode: str,
    dependencies: dict[str, Any] | None = None,
    pipeline_config: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    output_dir = get_output_cycle_dir(
        settings=settings,
        module_name=module_name,
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=model_name,
        cycle_name=cycle_name,
    )
    ensure_dir(output_dir)

    parsed_path = output_dir / "response_parsed.json"
    run_info_path = output_dir / "run_info.json"

    write_json(parsed_path, parsed_response)

    run_info: dict[str, Any] = {
        "module": module_name,
        "execution_mode": execution_mode,
        "scenario_name": scenario_name,
        "prompt_version": version,
        "loop_timestamp": loop_timestamp,
        "cycle_name": cycle_name,
        "cycle_index": cycle_index,
        "cycle_timestamp": cycle_timestamp,
        "model": model_name,
        "deployment_name": deployment_name,
        "execution_time_seconds": execution_time_seconds,
        "timestamp": datetime.now().isoformat(),
        "scenario": {
            "task": scenario_data.get("task"),
            "image": scenario_data.get("image"),
            "image_path_abs": scenario_data.get("image_path_abs"),
        },
        "response_parsed": parsed_response,
    }

    if dependencies is not None:
        run_info["dependencies"] = dependencies

    if pipeline_config is not None:
        run_info["pipeline_config"] = pipeline_config

    write_json(run_info_path, run_info)

    return parsed_path, run_info_path


def get_scene_description_full_artifact_paths(
    settings: Settings,
    scenario_name: str,
    version: str,
    experiment_timestamp: str,
    model_name: str,
    run_name: str,
) -> tuple[Path, Path]:
    """
    scene_description_full is stored as a side artifact inside the
    scene_description output directory.
    """
    output_dir = get_output_dir(
        settings=settings,
        module_name="scene_description",
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
    )
    return (
        output_dir / "scene_description_full.json",
        output_dir / "scene_description_full_run_info.json",
    )


def get_scene_description_full_artifact_paths_for_cycle(
    settings: Settings,
    scenario_name: str,
    version: str,
    loop_timestamp: str,
    model_name: str,
    cycle_name: str,
) -> tuple[Path, Path]:
    """
    scene_description_full is stored as a side artifact inside the
    cycle-based scene_description output directory.
    """
    output_dir = get_output_cycle_dir(
        settings=settings,
        module_name="scene_description",
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=model_name,
        cycle_name=cycle_name,
    )
    return (
        output_dir / "scene_description_full.json",
        output_dir / "scene_description_full_run_info.json",
    )


def save_scene_description_full_artifact(
    settings: Settings,
    scenario_name: str,
    version: str,
    experiment_timestamp: str,
    model_name: str,
    run_name: str,
    parsed_response: Any,
    scenario_data: dict[str, Any],
    execution_time_seconds: float,
    dependencies: dict[str, Any] | None = None,
    pipeline_config: dict[str, Any] | None = None,
    pose_file: str | None = None,
    safety_threshold: float | None = None,
    include_debug_mapping: bool = False,
    execution_mode: str = "single_module_side_artifact",
) -> tuple[Path, Path]:
    parsed_path, run_info_path = get_scene_description_full_artifact_paths(
        settings=settings,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
    )

    ensure_dir(parsed_path.parent)
    write_json(parsed_path, parsed_response)

    run_info: dict[str, Any] = {
        "module": "scene_description_full",
        "stored_under_module": "scene_description",
        "artifact_filename": "scene_description_full.json",
        "execution_mode": execution_mode,
        "scenario_name": scenario_name,
        "prompt_version": version,
        "experiment_timestamp": experiment_timestamp,
        "run_name": run_name,
        "model": model_name,
        "deployment_name": "deterministic_scene_enrichment",
        "execution_time_seconds": execution_time_seconds,
        "timestamp": datetime.now().isoformat(),
        "scenario": {
            "task": scenario_data.get("task"),
            "image": scenario_data.get("image"),
            "image_path_abs": scenario_data.get("image_path_abs"),
        },
        "enrichment_config": {
            "pose_source": "static",
            "pose_file": pose_file,
            "safety_threshold": safety_threshold,
            "include_debug_mapping": include_debug_mapping,
        },
        "response_parsed": parsed_response,
    }

    if dependencies is not None:
        run_info["dependencies"] = dependencies

    if pipeline_config is not None:
        run_info["pipeline_config"] = pipeline_config

    write_json(run_info_path, run_info)
    return parsed_path, run_info_path


def save_scene_description_full_artifact_for_cycle(
    settings: Settings,
    scenario_name: str,
    version: str,
    loop_timestamp: str,
    model_name: str,
    cycle_name: str,
    cycle_index: int,
    cycle_timestamp: str,
    parsed_response: Any,
    scenario_data: dict[str, Any],
    execution_time_seconds: float,
    dependencies: dict[str, Any] | None = None,
    pipeline_config: dict[str, Any] | None = None,
    pose_file: str | None = None,
    safety_threshold: float | None = None,
    include_debug_mapping: bool = False,
    execution_mode: str = "single_module_side_artifact_cycle",
) -> tuple[Path, Path]:
    parsed_path, run_info_path = get_scene_description_full_artifact_paths_for_cycle(
        settings=settings,
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=model_name,
        cycle_name=cycle_name,
    )

    ensure_dir(parsed_path.parent)
    write_json(parsed_path, parsed_response)

    run_info: dict[str, Any] = {
        "module": "scene_description_full",
        "stored_under_module": "scene_description",
        "artifact_filename": "scene_description_full.json",
        "execution_mode": execution_mode,
        "scenario_name": scenario_name,
        "prompt_version": version,
        "loop_timestamp": loop_timestamp,
        "cycle_name": cycle_name,
        "cycle_index": cycle_index,
        "cycle_timestamp": cycle_timestamp,
        "model": model_name,
        "deployment_name": "deterministic_scene_enrichment",
        "execution_time_seconds": execution_time_seconds,
        "timestamp": datetime.now().isoformat(),
        "scenario": {
            "task": scenario_data.get("task"),
            "image": scenario_data.get("image"),
            "image_path_abs": scenario_data.get("image_path_abs"),
        },
        "enrichment_config": {
            "pose_source": "static",
            "pose_file": pose_file,
            "safety_threshold": safety_threshold,
            "include_debug_mapping": include_debug_mapping,
        },
        "response_parsed": parsed_response,
    }

    if dependencies is not None:
        run_info["dependencies"] = dependencies

    if pipeline_config is not None:
        run_info["pipeline_config"] = pipeline_config

    write_json(run_info_path, run_info)
    return parsed_path, run_info_path


def load_scene_description_full_artifact(
    settings: Settings,
    scenario_name: str,
    version: str,
    experiment_timestamp: str,
    model_name: str,
    run_name: str,
) -> Any:
    parsed_path, _ = get_scene_description_full_artifact_paths(
        settings=settings,
        scenario_name=scenario_name,
        version=version,
        experiment_timestamp=experiment_timestamp,
        model_name=model_name,
        run_name=run_name,
    )

    if not parsed_path.exists():
        raise FileNotFoundError(
            f"scene_description_full artifact not found: {parsed_path}"
        )

    return read_json(parsed_path)


def load_scene_description_full_artifact_for_cycle(
    settings: Settings,
    scenario_name: str,
    version: str,
    loop_timestamp: str,
    model_name: str,
    cycle_name: str,
) -> Any:
    parsed_path, _ = get_scene_description_full_artifact_paths_for_cycle(
        settings=settings,
        scenario_name=scenario_name,
        version=version,
        loop_timestamp=loop_timestamp,
        model_name=model_name,
        cycle_name=cycle_name,
    )

    if not parsed_path.exists():
        raise FileNotFoundError(
            f"scene_description_full cycle artifact not found: {parsed_path}"
        )

    return read_json(parsed_path)










# from __future__ import annotations

# import json
# import re
# from datetime import datetime
# from pathlib import Path
# from typing import Any

# from settings import Settings


# VALID_MODULES = {
#     "scene_description",
#     "vlm_planning",
#     "simultaneous_actions",
#     "validator",
# }


# def validate_module_name(module_name: str) -> None:
#     if module_name not in VALID_MODULES:
#         allowed = ", ".join(sorted(VALID_MODULES))
#         raise ValueError(f"Invalid module '{module_name}'. Allowed values: {allowed}")


# def make_experiment_timestamp() -> str:
#     return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# def make_run_name(repeat_idx: int) -> str:
#     return f"run_{repeat_idx:03d}"


# def get_base_prompt_path(settings: Settings, module_name: str, version: str) -> Path:
#     validate_module_name(module_name)
#     return settings.project_root / "prompts" / module_name / version / "prompt.txt"


# def get_prompt_scenario_dir(
#     settings: Settings,
#     module_name: str,
#     scenario_name: str,
#     version: str,
#     experiment_timestamp: str,
#     model_name: str,
#     run_name: str,
# ) -> Path:
#     validate_module_name(module_name)
#     return (
#         settings.project_root
#         / "prompts_scenarios"
#         / module_name
#         / scenario_name
#         / version
#         / experiment_timestamp
#         / model_name
#         / run_name
#     )


# def get_output_dir(
#     settings: Settings,
#     module_name: str,
#     scenario_name: str,
#     version: str,
#     experiment_timestamp: str,
#     model_name: str,
#     run_name: str,
# ) -> Path:
#     validate_module_name(module_name)
#     return (
#         settings.project_root
#         / "outputs"
#         / module_name
#         / scenario_name
#         / version
#         / experiment_timestamp
#         / model_name
#         / run_name
#     )


# def ensure_dir(path: Path) -> Path:
#     path.mkdir(parents=True, exist_ok=True)
#     return path


# def read_text(path: str | Path) -> str:
#     path = Path(path)
#     if not path.exists():
#         raise FileNotFoundError(f"File not found: {path}")
#     return path.read_text(encoding="utf-8")


# def write_text(path: str | Path, content: str) -> None:
#     path = Path(path)
#     ensure_dir(path.parent)
#     path.write_text(content, encoding="utf-8")


# def read_json(path: str | Path) -> Any:
#     path = Path(path)
#     if not path.exists():
#         raise FileNotFoundError(f"JSON file not found: {path}")
#     return json.loads(path.read_text(encoding="utf-8"))


# def write_json(path: str | Path, data: Any) -> None:
#     path = Path(path)
#     ensure_dir(path.parent)
#     path.write_text(
#         json.dumps(data, indent=2, ensure_ascii=False),
#         encoding="utf-8",
#     )


# def load_base_prompt(settings: Settings, module_name: str, version: str) -> str:
#     prompt_path = get_base_prompt_path(settings, module_name, version)

#     if not prompt_path.exists():
#         raise FileNotFoundError(f"Base prompt not found: {prompt_path}")

#     return read_text(prompt_path)


# def try_parse_json(text: str) -> tuple[bool, Any]:
#     text = text.strip()

#     try:
#         return True, json.loads(text)
#     except Exception:
#         pass

#     extracted = extract_first_json_block(text)
#     if extracted is not None:
#         try:
#             return True, json.loads(extracted)
#         except Exception:
#             pass

#     return False, None


# def extract_first_json_block(text: str) -> str | None:
#     text = text.strip()

#     array_match = re.search(r"(\[\s*[\s\S]*\])", text)
#     if array_match:
#         return array_match.group(1)

#     obj_match = re.search(r"(\{\s*[\s\S]*\})", text)
#     if obj_match:
#         return obj_match.group(1)

#     return None


# def render_prompt(
#     module_name: str,
#     base_prompt: str,
#     scenario_data: dict[str, Any],
#     scene_description: Any | None = None,
#     sequential_plan: Any | None = None,
# ) -> str:
#     if module_name == "scene_description":
#         return base_prompt.strip()

#     if module_name == "vlm_planning":
#         if scene_description is None:
#             raise ValueError("scene_description is required for vlm_planning")

#         task = scenario_data.get("task")
#         if not task:
#             raise ValueError("Scenario is missing 'task' for vlm_planning")

#         scene_description_str = json.dumps(scene_description, indent=2, ensure_ascii=False)

#         return (
#             base_prompt.strip()
#             + "\n\nSCENE DESCRIPTION\n"
#             + scene_description_str
#             + "\n\nTASK\n"
#             + task
#         )

#     if module_name == "simultaneous_actions":
#         if scene_description is None:
#             raise ValueError("scene_description is required for simultaneous_actions")
#         if sequential_plan is None:
#             raise ValueError("sequential_plan is required for simultaneous_actions")

#         scene_description_str = json.dumps(scene_description, indent=2, ensure_ascii=False)
#         sequential_plan_str = json.dumps(sequential_plan, indent=2, ensure_ascii=False)

#         return (
#             base_prompt.strip()
#             + "\n\nSCENE DESCRIPTION\n"
#             + scene_description_str
#             + "\n\nSEQUENTIAL PLAN\n"
#             + sequential_plan_str
#         )

#     raise ValueError(f"Unsupported module for rendering: {module_name}")


# def load_previous_module_output(
#     settings: Settings,
#     module_name: str,
#     scenario_name: str,
#     version: str,
#     experiment_timestamp: str,
#     model_name: str,
#     run_name: str,
# ) -> Any:
#     output_dir = get_output_dir(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=model_name,
#         run_name=run_name,
#     )
#     parsed_path = output_dir / "response_parsed.json"

#     if not parsed_path.exists():
#         raise FileNotFoundError(f"Previous module output not found: {parsed_path}")

#     return read_json(parsed_path)


# def save_rendered_prompt(
#     settings: Settings,
#     module_name: str,
#     scenario_name: str,
#     version: str,
#     experiment_timestamp: str,
#     model_name: str,
#     run_name: str,
#     prompt_text: str,
# ) -> Path:
#     prompt_dir = get_prompt_scenario_dir(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=model_name,
#         run_name=run_name,
#     )
#     ensure_dir(prompt_dir)

#     prompt_path = prompt_dir / "prompt.txt"
#     write_text(prompt_path, prompt_text)
#     return prompt_path


# def save_module_outputs(
#     settings: Settings,
#     module_name: str,
#     scenario_name: str,
#     version: str,
#     experiment_timestamp: str,
#     model_name: str,
#     run_name: str,
#     deployment_name: str,
#     execution_time_seconds: float,
#     scenario_data: dict[str, Any],
#     parsed_response: Any,
#     execution_mode: str,
#     dependencies: dict[str, Any] | None = None,
#     pipeline_config: dict[str, Any] | None = None,
# ) -> tuple[Path, Path]:
#     output_dir = get_output_dir(
#         settings=settings,
#         module_name=module_name,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=model_name,
#         run_name=run_name,
#     )
#     ensure_dir(output_dir)

#     parsed_path = output_dir / "response_parsed.json"
#     run_info_path = output_dir / "run_info.json"

#     write_json(parsed_path, parsed_response)

#     run_info: dict[str, Any] = {
#         "module": module_name,
#         "execution_mode": execution_mode,
#         "scenario_name": scenario_name,
#         "prompt_version": version,
#         "experiment_timestamp": experiment_timestamp,
#         "run_name": run_name,
#         "model": model_name,
#         "deployment_name": deployment_name,
#         "execution_time_seconds": execution_time_seconds,
#         "timestamp": datetime.now().isoformat(),
#         "scenario": {
#             "task": scenario_data.get("task"),
#             "image": scenario_data.get("image"),
#             "image_path_abs": scenario_data.get("image_path_abs"),
#         },
#         "response_parsed": parsed_response,
#     }

#     if dependencies is not None:
#         run_info["dependencies"] = dependencies

#     if pipeline_config is not None:
#         run_info["pipeline_config"] = pipeline_config

#     write_json(run_info_path, run_info)

#     return parsed_path, run_info_path


# def get_scene_description_full_artifact_paths(
#     settings: Settings,
#     scenario_name: str,
#     version: str,
#     experiment_timestamp: str,
#     model_name: str,
#     run_name: str,
# ) -> tuple[Path, Path]:
#     """
#     scene_description_full is stored as a side artifact inside the
#     scene_description output directory.
#     """
#     output_dir = get_output_dir(
#         settings=settings,
#         module_name="scene_description",
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=model_name,
#         run_name=run_name,
#     )
#     return (
#         output_dir / "scene_description_full.json",
#         output_dir / "scene_description_full_run_info.json",
#     )


# def save_scene_description_full_artifact(
#     settings: Settings,
#     scenario_name: str,
#     version: str,
#     experiment_timestamp: str,
#     model_name: str,
#     run_name: str,
#     parsed_response: Any,
#     scenario_data: dict[str, Any],
#     execution_time_seconds: float,
#     dependencies: dict[str, Any] | None = None,
#     pipeline_config: dict[str, Any] | None = None,
#     pose_file: str | None = None,
#     safety_threshold: float | None = None,
#     include_debug_mapping: bool = False,
#     execution_mode: str = "single_module_side_artifact",
# ) -> tuple[Path, Path]:
#     parsed_path, run_info_path = get_scene_description_full_artifact_paths(
#         settings=settings,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=model_name,
#         run_name=run_name,
#     )

#     ensure_dir(parsed_path.parent)
#     write_json(parsed_path, parsed_response)

#     run_info: dict[str, Any] = {
#         "module": "scene_description_full",
#         "stored_under_module": "scene_description",
#         "artifact_filename": "scene_description_full.json",
#         "execution_mode": execution_mode,
#         "scenario_name": scenario_name,
#         "prompt_version": version,
#         "experiment_timestamp": experiment_timestamp,
#         "run_name": run_name,
#         "model": model_name,
#         "deployment_name": "deterministic_scene_enrichment",
#         "execution_time_seconds": execution_time_seconds,
#         "timestamp": datetime.now().isoformat(),
#         "scenario": {
#             "task": scenario_data.get("task"),
#             "image": scenario_data.get("image"),
#             "image_path_abs": scenario_data.get("image_path_abs"),
#         },
#         "enrichment_config": {
#             "pose_source": "static",
#             "pose_file": pose_file,
#             "safety_threshold": safety_threshold,
#             "include_debug_mapping": include_debug_mapping,
#         },
#         "response_parsed": parsed_response,
#     }

#     if dependencies is not None:
#         run_info["dependencies"] = dependencies

#     if pipeline_config is not None:
#         run_info["pipeline_config"] = pipeline_config

#     write_json(run_info_path, run_info)
#     return parsed_path, run_info_path


# def load_scene_description_full_artifact(
#     settings: Settings,
#     scenario_name: str,
#     version: str,
#     experiment_timestamp: str,
#     model_name: str,
#     run_name: str,
# ) -> Any:
#     parsed_path, _ = get_scene_description_full_artifact_paths(
#         settings=settings,
#         scenario_name=scenario_name,
#         version=version,
#         experiment_timestamp=experiment_timestamp,
#         model_name=model_name,
#         run_name=run_name,
#     )

#     if not parsed_path.exists():
#         raise FileNotFoundError(
#             f"scene_description_full artifact not found: {parsed_path}"
#         )

#     return read_json(parsed_path)

