# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any

# from .config import Settings


# def ensure_outputs_dir(
#     settings: Settings,
#     scenario_id: str | None = None,
#     prompt_name: str | None = None,
#     run_id: str | None = None,
#     model_name: str | None = None,
# ) -> Path:
#     output_dir = settings.project_root / settings.outputs_subdir

#     if scenario_id:
#         output_dir = output_dir / scenario_id

#     if prompt_name:
#         output_dir = output_dir / prompt_name

#     if run_id:
#         output_dir = output_dir / run_id

#     if model_name:
#         output_dir = output_dir / model_name

#     output_dir.mkdir(parents=True, exist_ok=True)
#     return output_dir


# def save_result(
#     settings: Settings,
#     filename: str,
#     payload: dict[str, Any],
#     scenario_id: str | None = None,
#     prompt_name: str | None = None,
#     run_id: str | None = None,
#     model_name: str | None = None,
# ) -> Path:
#     output_dir = ensure_outputs_dir(
#         settings=settings,
#         scenario_id=scenario_id,
#         prompt_name=prompt_name,
#         run_id=run_id,
#         model_name=model_name,
#     )

#     out_path = output_dir / filename

#     out_path.write_text(
#         json.dumps(payload, indent=2, ensure_ascii=False),
#         encoding="utf-8",
#     )

#     return out_path


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings


def ensure_outputs_dir(
    settings: Settings,
    task_type: str,
    scenario_id: str | None = None,
    prompt_name: str | None = None,
    run_id: str | None = None,
    model_name: str | None = None,
) -> Path:
    output_dir = settings.project_root / settings.outputs_root / task_type

    if scenario_id:
        output_dir = output_dir / scenario_id

    if prompt_name:
        output_dir = output_dir / prompt_name

    if run_id:
        output_dir = output_dir / run_id

    if model_name:
        output_dir = output_dir / model_name

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_result(
    settings: Settings,
    task_type: str,
    filename: str,
    payload: dict[str, Any],
    scenario_id: str | None = None,
    prompt_name: str | None = None,
    run_id: str | None = None,
    model_name: str | None = None,
) -> Path:
    output_dir = ensure_outputs_dir(
        settings=settings,
        task_type=task_type,
        scenario_id=scenario_id,
        prompt_name=prompt_name,
        run_id=run_id,
        model_name=model_name,
    )

    out_path = output_dir / filename

    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return out_path