# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any

# from .config import Settings


# def ensure_outputs_dir(settings: Settings) -> Path:
#     output_dir = settings.project_root / "outputs" / settings.outputs_subdir
#     output_dir.mkdir(parents=True, exist_ok=True)
#     return output_dir


# def save_result(settings: Settings, filename: str, payload: dict[str, Any]) -> Path:
#     output_dir = ensure_outputs_dir(settings)
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


def ensure_outputs_dir(settings: Settings, scenario_id: str | None = None) -> Path:
    output_dir = settings.project_root / "outputs" / settings.outputs_subdir

    if scenario_id:
        output_dir = output_dir / scenario_id

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_result(
    settings: Settings,
    filename: str,
    payload: dict[str, Any],
    scenario_id: str | None = None,
) -> Path:

    output_dir = ensure_outputs_dir(settings, scenario_id)

    out_path = output_dir / filename

    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return out_path