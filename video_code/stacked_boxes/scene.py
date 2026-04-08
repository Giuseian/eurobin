#!/usr/bin/env python3

import json
import re
import sys
import subprocess
from typing import Dict, List, Any, Tuple, Set, Optional


SIDES = ["left", "right", "front", "back", "top", "bottom"]


# =========================================================
# CONFIGURAZIONE GAZEBO
# =========================================================
GZ_POSE_TOPIC = "/world/default/dynamic_pose/info"


# =========================================================
# CONFIGURAZIONE DIMENSIONI
# =========================================================
# Priorità:
# 1) object["dimension"] nel JSON
# 2) DIMENSIONS_BY_NAME[name]
# 3) DEFAULT_DIMENSIONS_BY_CATEGORY[category]
DIMENSIONS_BY_NAME = {
    "box_red_001": [0.28, 0.28, 0.28],
    "box_green_001": [0.28, 0.28, 0.28],
    "glass_001": [0.08, 0.08, 0.08],
    "glass_002": [0.08, 0.08, 0.08],
    "ball_001": [0.07, 0.07, 0.07],
}

DEFAULT_DIMENSIONS_BY_CATEGORY = {
    "box": [0.28, 0.28, 0.28],
    "glass": [0.08, 0.08, 0.08],
    "ball": [0.07, 0.07, 0.07],
}


# =========================================================
# GEOMETRIA
# =========================================================
def get_aabb(pose: List[float], dimension: List[float]) -> Dict[str, Tuple[float, float]]:
    x, y, z = pose
    dx, dy, dz = dimension

    return {
        "x": (x - dx / 2.0, x + dx / 2.0),
        "y": (y - dy / 2.0, y + dy / 2.0),
        "z": (z - dz / 2.0, z + dz / 2.0),
    }


def gap_between_intervals(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    a_min, a_max = a
    b_min, b_max = b

    if a_max < b_min:
        return b_min - a_max
    elif b_max < a_min:
        return a_min - b_max
    else:
        return 0.0


def add_blocker(blockers: Dict[str, Dict[str, Set[str]]], obj_name: str, side: str, blocker: str):
    blockers[obj_name][side].add(blocker)


def format_side_status(blocker_set: Set[str]) -> str:
    if not blocker_set:
        return "accessible"
    if len(blocker_set) == 1:
        blocker = next(iter(blocker_set))
        if blocker == "table":
            return "blocked by the table"
        return f"blocked by {blocker}"
    return "blocked by " + ", ".join(sorted(blocker_set))


def get_location(aabb: Dict[str, Tuple[float, float]]) -> str:
    """
    Regole:
    - zone C: faccia destra < -0.75
    - zone A: faccia sinistra > 0.65
    - zone B: altrimenti

    Convenzione:
    - left face = y_min
    - right face = y_max
    """
    y_min, y_max = aabb["y"]

    if y_max < -0.75:
        return "zone C"
    elif y_min > 0.65:
        return "zone A"
    else:
        return "zone B"


def is_back_not_reachable(aabb: Dict[str, Tuple[float, float]]) -> bool:
    """
    Se la faccia di dietro ha x > 3.75 -> not reachable

    Convenzione:
    - back face = x_max
    """
    _, x_max = aabb["x"]
    return x_max > 3.75


# =========================================================
# LETTURA POSE DA GAZEBO
# =========================================================
def read_gz_dynamic_pose_once(topic: str = GZ_POSE_TOPIC, timeout_sec: float = 3.0) -> str:
    cmd = [
        "gz", "topic",
        "-e",
        "-n", "1",
        "-t", topic
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=True
        )
    except FileNotFoundError:
        raise RuntimeError(
            'Comando "gz" non trovato. Assicurati che Gazebo sia installato e che l’ambiente sia caricato.'
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout mentre leggevo il topic Gazebo {topic}")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else str(e)
        raise RuntimeError(f'Errore eseguendo "gz topic": {stderr}')

    return result.stdout


def parse_all_entity_positions(gz_output: str) -> Dict[str, List[float]]:
    """
    Estrae tutte le entità e le loro posizioni da un dump del topic Gazebo.

    Restituisce:
    {
        "box_red_001": [x, y, z],
        "box_green_001": [x, y, z],
        ...
    }
    """
    positions = {}

    pattern = re.compile(
        r'name:\s*"([^"]+)"\s*'
        r'id:\s*\d+\s*'
        r'position\s*\{\s*'
        r'x:\s*([-\d.eE+]+)\s*'
        r'y:\s*([-\d.eE+]+)\s*'
        r'z:\s*([-\d.eE+]+)\s*'
        r'\}',
        re.MULTILINE | re.DOTALL
    )

    for match in pattern.finditer(gz_output):
        name = match.group(1)
        x = float(match.group(2))
        y = float(match.group(3))
        z = float(match.group(4))
        positions[name] = [x, y, z]

    return positions


def get_positions_from_gazebo(entity_names: List[str], topic: str = GZ_POSE_TOPIC) -> Dict[str, List[float]]:
    """
    Legge una sola volta il topic Gazebo e restituisce le pose
    delle entità richieste.
    """
    gz_output = read_gz_dynamic_pose_once(topic=topic)
    all_positions = parse_all_entity_positions(gz_output)

    missing = [name for name in entity_names if name not in all_positions]
    if missing:
        raise RuntimeError(
            f"Entità non trovate nel topic Gazebo {topic}: {', '.join(missing)}"
        )

    return {name: all_positions[name] for name in entity_names}


# =========================================================
# DIMENSIONI
# =========================================================
def resolve_dimension(obj: Dict[str, Any]) -> List[float]:
    """
    Risolve la dimensione di un oggetto con la seguente priorità:
    1) campo 'dimension' nell'oggetto
    2) DIMENSIONS_BY_NAME[name]
    3) DEFAULT_DIMENSIONS_BY_CATEGORY[category]
    """
    if "dimension" in obj:
        dim = obj["dimension"]
        if isinstance(dim, list) and len(dim) == 3:
            return dim
        raise ValueError(f'Dimensione non valida per {obj.get("name")}: {dim}')

    name = obj.get("name")
    if name in DIMENSIONS_BY_NAME:
        return DIMENSIONS_BY_NAME[name]

    category = obj.get("category")
    if category in DEFAULT_DIMENSIONS_BY_CATEGORY:
        return DEFAULT_DIMENSIONS_BY_CATEGORY[category]

    raise ValueError(
        f"Nessuna dimensione disponibile per oggetto '{name}'. "
        f"Aggiungi 'dimension' nel JSON oppure configura DIMENSIONS_BY_NAME / DEFAULT_DIMENSIONS_BY_CATEGORY."
    )


# =========================================================
# ACCESSIBILITÀ
# =========================================================
def compute_accessibility(
    objects_with_geometry: List[Dict[str, Any]],
    spatial_relationships: List[Dict[str, str]],
    safety_threshold: float = 0.21
) -> Dict[str, Dict[str, Any]]:
    """
    Restituisce:
    {
        "box_red_001": {
            "location": "zone B",
            "sides": {...}
        },
        ...
    }
    """
    object_map = {obj["name"]: obj for obj in objects_with_geometry}

    aabbs = {
        name: get_aabb(obj["pose"], obj["dimension"])
        for name, obj in object_map.items()
    }

    blockers = {
        name: {side: set() for side in SIDES}
        for name in object_map.keys()
    }

    grasped_objects = set()
    objects_on_top_of_something = set()

    # --- ON TOP OF + GRASPED ---
    for rel in spatial_relationships:
        subj = rel["subject"]
        relation = rel["relation"].lower()
        obj = rel["object"]

        if subj not in object_map:
            continue

        if relation == "on top of":
            if obj not in object_map:
                continue
            add_blocker(blockers, subj, "bottom", obj)
            add_blocker(blockers, obj, "top", subj)
            objects_on_top_of_something.add(subj)

        elif relation == "grasped by":
            grasped_objects.add(subj)

    # --- TABLE ---
    for name in object_map.keys():
        if name not in objects_on_top_of_something:
            add_blocker(blockers, name, "bottom", "table")

    # --- RELAZIONI ORIZZONTALI ---
    for rel in spatial_relationships:
        subj = rel["subject"]
        relation = rel["relation"].lower()
        obj = rel["object"]

        if subj not in object_map or obj not in object_map:
            continue

        if relation == "right of":
            gap_y = gap_between_intervals(aabbs[subj]["y"], aabbs[obj]["y"])
            if gap_y <= safety_threshold:
                add_blocker(blockers, subj, "left", obj)
                add_blocker(blockers, obj, "right", subj)

        elif relation == "in front of":
            gap_x = gap_between_intervals(aabbs[subj]["x"], aabbs[obj]["x"])
            if gap_x <= safety_threshold:
                add_blocker(blockers, subj, "back", obj)
                add_blocker(blockers, obj, "front", subj)

    # --- GRASP OVERRIDE ---
    for name in grasped_objects:
        if name in blockers:
            for side in SIDES:
                blockers[name][side].clear()

    result = {}
    for name in object_map.keys():
        aabb = aabbs[name]

        sides = {
            side: format_side_status(blockers[name][side])
            for side in SIDES
        }

        if is_back_not_reachable(aabb):
            sides["back"] = "not reachable"

        result[name] = {
            "location": get_location(aabb),
            "sides": sides
        }

    return result


# =========================================================
# PIPELINE COMPLETA
# =========================================================
def build_objects_with_geometry(scene_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Costruisce la lista oggetti con:
    - name
    - pose (da Gazebo)
    - dimension (da JSON/config)
    """
    object_names = [obj["name"] for obj in scene_objects]
    positions_by_name = get_positions_from_gazebo(object_names)

    objects_with_geometry = []
    for obj in scene_objects:
        name = obj["name"]
        dimension = resolve_dimension(obj)
        pose = positions_by_name[name]

        objects_with_geometry.append({
            "name": name,
            "pose": pose,
            "dimension": dimension
        })

    return objects_with_geometry


def enrich_scene(input_data: Dict[str, Any], safety_threshold: float = 0.21) -> Dict[str, Any]:
    if "scene_description" not in input_data:
        raise ValueError("Campo mancante: 'scene_description'")

    scene_description = input_data["scene_description"]
    scene_objects = scene_description.get("objects", [])
    spatial_relationships = scene_description.get("spatial_relationships", [])

    objects_with_geometry = build_objects_with_geometry(scene_objects)

    computed_info = compute_accessibility(
        objects_with_geometry=objects_with_geometry,
        spatial_relationships=spatial_relationships,
        safety_threshold=safety_threshold
    )

    enriched_objects = []
    for obj in scene_objects:
        name = obj["name"]

        enriched_obj = dict(obj)
        enriched_obj["location"] = computed_info[name]["location"]
        enriched_obj["sides"] = computed_info[name]["sides"]
        enriched_objects.append(enriched_obj)

    output = {
        "scene_description": {
            "objects": enriched_objects,
            "end_effectors": scene_description.get("end_effectors", []),
            "spatial_relationships": spatial_relationships
        }
    }

    return output


# =========================================================
# MAIN
# =========================================================
def main():
    if len(sys.argv) < 2:
        print("Uso: python scene.py scena.json", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        output_data = enrich_scene(input_data)

        output_path = "scena_complete.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"JSON salvato in {output_path}")

    except FileNotFoundError:
        print(f"Errore: file non trovato: {input_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Errore: JSON non valido: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()