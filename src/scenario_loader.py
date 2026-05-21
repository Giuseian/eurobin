""" 
`scenario_loader.py` is a small helper file used to load a scenario from disk.
It takes a scenario name, looks for ```text scenarios/<scenario_name>/scenario.json``` and reads that JSON file into a Python dictionary. It also checks that the scenario directory and `scenario.json` actually exist.
If the `scenario.json` contains an `image` field, it resolves that image into an absolute path, checks that the image exists, and adds it to the returned data as ```python image_path_abs```. 
It also adds: ```python scenario_name scenario_dir_abs ``` . So other scripts do not need to manually reconstruct where the scenario folder or image file are.
In short: `scenario_loader.py` is the common utility that loads a scenario, validates its image path, and returns a ready-to-use `scenario_data` dictionary for the pipeline scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from settings import Settings


def load_scenario(settings: Settings, scenario_name: str) -> dict[str, Any]:
    """
    Carica lo scenario da:
    scenarios/<scenario_name>/scenario.json

    Restituisce un dizionario con i dati dello scenario e,
    se presente, aggiunge il campo:
    - image_path_abs
    """
    scenario_dir = settings.project_root / "scenarios" / scenario_name
    scenario_file = scenario_dir / "scenario.json"

    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    if not scenario_file.exists():
        raise FileNotFoundError(f"scenario.json not found: {scenario_file}")

    scenario_data = json.loads(scenario_file.read_text(encoding="utf-8"))

    if not isinstance(scenario_data, dict):
        raise ValueError(f"Scenario file must contain a JSON object: {scenario_file}")

    scenario_data["scenario_name"] = scenario_data.get("scenario_name", scenario_name)

    image_rel = scenario_data.get("image")
    if image_rel:
        image_abs = (scenario_dir / image_rel).resolve()
        if not image_abs.exists():
            raise FileNotFoundError(
                f"Image file declared in scenario.json not found: {image_abs}"
            )
        scenario_data["image_path_abs"] = str(image_abs)
    else:
        scenario_data["image_path_abs"] = None

    scenario_data["scenario_dir_abs"] = str(scenario_dir.resolve())

    return scenario_data