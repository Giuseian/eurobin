#!/usr/bin/env python3
"""Enrich a VLM scene description with geometric accessibility.

Real-robot mode:
- poses are read from FoundationPose poses.json
- mesh assignments are read from mesh_assignments.json
- object dimensions are read from meshes/mesh_metadata.json

Important convention:
poses.json stores object position in the real robot camera/world convention used
by the manipulation pipeline:

    x = depth
    y = left/right
    z = up

Therefore, object dimensions for AABB must be ordered as:

    dimension = [depth, width, height]

where:
    width  = left/right size
    height = vertical size
    depth  = front/back size

Normal output intentionally keeps the original scene_description structure clean:
- objects keep their original fields
- sides are added
- location, pose, dimension, mesh_id are NOT added to the normal output

Use --include-debug-mapping if you want pose/dimension/mesh debug information.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SIDES = ["left", "right", "front", "back", "top", "bottom"]

GZ_POSE_TOPIC = "/world/default/dynamic_pose/info"

POSE_SOURCE_GAZEBO = "gazebo"
POSE_SOURCE_STATIC = "static"

DEFAULT_DATA_ROOT = "/workspace/shared_data/realsense"
DEFAULT_MESH_METADATA_FILE = f"{DEFAULT_DATA_ROOT}/meshes/mesh_metadata.json"

RELATION_ON_TOP_OF = "on top of"
RELATION_RIGHT_OF = "right of"
RELATION_IN_FRONT_OF = "in front of"
RELATION_GRASPED_BY = "grasped by"

# Used only for Gazebo mode fallback.
GAZEBO_DEFAULT_OBJECT_DIMENSION: List[float] = [0.28, 0.28, 0.28]

# Tolerance for matching temporary poses.json with real outputs_by_object/<timestamp>/poses.json.
POSE_MATCH_TOLERANCE = 1e-4

# Kept for compatibility with older helper functions.
CATEGORY_ALIASES: Dict[str, str] = {}
COLOR_ALIASES: Dict[str, str] = {}
NON_DISCRIMINATIVE_COLORS: Set[str] = set()
KNOWN_COLORS: Set[str] = set()
KNOWN_CATEGORIES: Set[str] = set()


# =========================================================
# Utility helpers
# =========================================================
def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower())


def split_tokens(value: str) -> List[str]:
    return [token for token in normalize_text(value).split("_") if token]


def canonicalize_token(token: str) -> str:
    token = normalize_text(token)
    token = CATEGORY_ALIASES.get(token, token)
    token = COLOR_ALIASES.get(token, token)
    return token


def split_canonical_tokens(value: str) -> List[str]:
    return [canonicalize_token(token) for token in split_tokens(value)]


def canonicalize_category(category: Optional[str]) -> Optional[str]:
    if category is None:
        return None
    normalized = normalize_text(category)
    return CATEGORY_ALIASES.get(normalized, normalized)


def canonicalize_color(color: Optional[str]) -> Optional[str]:
    if color is None:
        return None

    normalized = normalize_text(color)
    normalized = COLOR_ALIASES.get(normalized, normalized)

    if normalized in NON_DISCRIMINATIVE_COLORS:
        return None

    if KNOWN_COLORS and normalized not in KNOWN_COLORS:
        return None

    return normalized


def normalize_numeric_suffix(value: str) -> str:
    tokens = split_tokens(value)
    if not tokens:
        return ""

    normalized: List[str] = []
    for token in tokens:
        if token.isdigit():
            normalized.append(str(int(token)))
        else:
            normalized.append(token)

    return "_".join(normalized)


def extract_index_suffix(value: str) -> Optional[int]:
    tokens = split_tokens(value)
    if not tokens:
        return None

    last = tokens[-1]
    if last.isdigit():
        return int(last)

    return None


def infer_category_from_name(name: str) -> Optional[str]:
    tokens = set(split_canonical_tokens(name))
    for token in tokens:
        if token in KNOWN_CATEGORIES:
            return token
    return None


def infer_color_from_name(name: str) -> Optional[str]:
    tokens = set(split_canonical_tokens(name))
    for token in tokens:
        if token in KNOWN_COLORS:
            return token
    return None


def semantic_prefix(name: str) -> str:
    tokens = split_canonical_tokens(name)
    if tokens and tokens[-1].isdigit():
        tokens = tokens[:-1]
    return "_".join(tokens)


def intervals_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def clamp_lower(value: float, threshold: float) -> bool:
    return value <= threshold


def original_color_is_non_discriminative(color: Optional[str]) -> bool:
    if color is None:
        return False

    normalized = normalize_text(color)
    normalized = COLOR_ALIASES.get(normalized, normalized)

    return normalized in NON_DISCRIMINATIVE_COLORS


# =========================================================
# JSON helpers
# =========================================================
def read_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in file '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"JSON file must contain an object: {path}")

    return data


def require_existing_file(path: str, label: str) -> str:
    resolved = Path(path).resolve()

    if not resolved.exists():
        raise RuntimeError(f"{label} not found: {resolved}")

    if not resolved.is_file():
        raise RuntimeError(f"{label} path is not a file: {resolved}")

    return str(resolved)


# =========================================================
# Static pose IO
# =========================================================
def validate_positions_dict(raw_positions: Dict[str, Any], source_name: str) -> Dict[str, List[float]]:
    positions: Dict[str, List[float]] = {}

    if not isinstance(raw_positions, dict):
        raise RuntimeError(f"{source_name} must contain a JSON object mapping object names to [x, y, z].")

    for name, pose in raw_positions.items():
        if not isinstance(name, str):
            raise RuntimeError(f"{source_name}: each object name must be a string.")

        if not isinstance(pose, list) or len(pose) != 3:
            raise RuntimeError(
                f"{source_name}: pose for '{name}' must be a list of exactly 3 numeric values."
            )

        if not all(isinstance(v, (int, float)) for v in pose):
            raise RuntimeError(
                f"{source_name}: pose for '{name}' must contain only numeric values."
            )

        positions[name] = [float(v) for v in pose]

    return positions


def extract_positions_from_foundationpose_poses_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports both formats:

    1. Direct:
       {
         "brown_box": [x, y, z],
         "red_box": [x, y, z]
       }

    2. FoundationPose:
       {
         "000000.png": {
           "brown_box": [x, y, z],
           "red_box": [x, y, z]
         }
       }
    """
    if not isinstance(data, dict):
        raise RuntimeError("poses.json must contain a JSON object.")

    direct_like = True
    for value in data.values():
        if not (isinstance(value, list) and len(value) == 3):
            direct_like = False
            break

    if direct_like:
        return data

    frame_keys = sorted(
        key for key, value in data.items()
        if isinstance(key, str) and key.endswith(".png") and isinstance(value, dict)
    )

    if frame_keys:
        return data[frame_keys[0]]

    if len(data) == 1:
        only_value = next(iter(data.values()))
        if isinstance(only_value, dict):
            return only_value

    raise RuntimeError(
        "Unsupported poses.json format. Expected either object -> [x,y,z] "
        "or frame.png -> object -> [x,y,z]."
    )


def read_all_positions_from_static_file(pose_file: str) -> Dict[str, List[float]]:
    data = read_json_file(pose_file)
    positions_data = extract_positions_from_foundationpose_poses_json(data)
    return validate_positions_dict(positions_data, f"Static pose file '{pose_file}'")


# =========================================================
# Mesh metadata / assignments
# =========================================================
def infer_data_root(
    data_root: Optional[str],
    mesh_metadata_file: Optional[str],
) -> str:
    if data_root:
        return str(Path(data_root).resolve())

    if mesh_metadata_file:
        metadata_path = Path(mesh_metadata_file).resolve()
        if metadata_path.name == "mesh_metadata.json" and metadata_path.parent.name == "meshes":
            return str(metadata_path.parent.parent)

    return DEFAULT_DATA_ROOT


def load_mesh_metadata(mesh_metadata_file: Optional[str]) -> Dict[str, Any]:
    if not mesh_metadata_file:
        raise RuntimeError("mesh_metadata_file is required.")

    mesh_metadata_file = require_existing_file(mesh_metadata_file, "mesh_metadata.json")
    data = read_json_file(mesh_metadata_file)

    meshes = data.get("meshes", {})
    if not isinstance(meshes, dict):
        raise RuntimeError(
            f"mesh_metadata file must contain a 'meshes' object: {mesh_metadata_file}"
        )

    print(f"[DEBUG] Loaded mesh metadata: {mesh_metadata_file}")
    print(f"[DEBUG] Mesh metadata IDs: {', '.join(sorted(meshes.keys()))}")

    return data


def load_mesh_assignments(mesh_assignments_file: str) -> Dict[str, Any]:
    mesh_assignments_file = require_existing_file(mesh_assignments_file, "mesh_assignments.json")
    data = read_json_file(mesh_assignments_file)

    objects = data.get("objects", {})
    if not isinstance(objects, dict):
        raise RuntimeError(
            f"mesh_assignments file must contain an 'objects' object: {mesh_assignments_file}"
        )

    print(f"[DEBUG] Loaded mesh assignments: {mesh_assignments_file}")
    print(f"[DEBUG] Mesh assignment objects: {', '.join(sorted(objects.keys()))}")

    return data


def assignment_object_names(mesh_assignments: Dict[str, Any]) -> Set[str]:
    objects = mesh_assignments.get("objects", {})
    if not isinstance(objects, dict):
        return set()
    return set(objects.keys())


def positions_match(
    reference_positions: Dict[str, List[float]],
    candidate_positions: Dict[str, List[float]],
    tolerance: float = POSE_MATCH_TOLERANCE,
) -> bool:
    for name, reference_pose in reference_positions.items():
        candidate_pose = candidate_positions.get(name)
        if candidate_pose is None:
            return False

        if len(reference_pose) != 3 or len(candidate_pose) != 3:
            return False

        for ref_value, cand_value in zip(reference_pose, candidate_pose):
            if abs(float(ref_value) - float(cand_value)) > tolerance:
                return False

    return True


def try_load_candidate_positions(candidate_dir: Path) -> Optional[Dict[str, List[float]]]:
    poses_file = candidate_dir / "poses.json"
    if not poses_file.exists() or not poses_file.is_file():
        return None

    try:
        data = read_json_file(str(poses_file))
        positions_data = extract_positions_from_foundationpose_poses_json(data)
        return validate_positions_dict(positions_data, f"Candidate poses file '{poses_file}'")
    except Exception as exc:
        print(f"[WARN] Could not read candidate poses file {poses_file}: {exc}")
        return None


def find_mesh_assignments_candidates(data_root: str) -> List[Path]:
    root = Path(data_root).resolve()

    candidate_roots = [
        root / "outputs_by_object",
        root / "outputs_by_objects",
    ]

    candidates: List[Path] = []

    for candidate_root in candidate_roots:
        if not candidate_root.exists():
            continue

        for path in candidate_root.glob("*/mesh_assignments.json"):
            if path.is_file():
                candidates.append(path)

        for path in candidate_root.glob("*/mesh_assignment.json"):
            if path.is_file():
                candidates.append(path)

    candidates = sorted(
        candidates,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return candidates


def resolve_mesh_assignments_file(
    pose_file: Optional[str],
    mesh_assignments_file: Optional[str],
    pose_positions: Dict[str, List[float]],
    data_root: str,
) -> str:
    if mesh_assignments_file:
        return require_existing_file(mesh_assignments_file, "mesh_assignments.json")

    if pose_file:
        pose_path = Path(pose_file).resolve()
        pose_dir = pose_path.parent

        adjacent_candidates = [
            pose_dir / "mesh_assignments.json",
            pose_dir / "mesh_assignment.json",
        ]

        for candidate in adjacent_candidates:
            if candidate.exists() and candidate.is_file():
                print(f"[DEBUG] Found mesh assignments next to poses.json: {candidate}")
                return str(candidate)

        print()
        print("[DEBUG] mesh_assignments.json was not found next to pose_file.")
        print(f"[DEBUG] pose_file: {pose_path}")
        print(f"[DEBUG] pose_dir:  {pose_dir}")
        print("[DEBUG] This is expected if live2 passed a temporary poses file in /tmp.")
        print("[DEBUG] Searching shared_data outputs instead...")
        print()

    pose_object_names = set(pose_positions.keys())

    candidates = find_mesh_assignments_candidates(data_root)

    if not candidates:
        raise RuntimeError(
            "No mesh_assignments.json files found under shared_data outputs.\n"
            f"data_root: {data_root}\n"
            "Expected under:\n"
            f"  - {Path(data_root) / 'outputs_by_object'}\n"
            f"  - {Path(data_root) / 'outputs_by_objects'}"
        )

    exact_pose_matches: List[Path] = []
    object_name_matches: List[Path] = []

    for candidate in candidates:
        try:
            assignment_data = read_json_file(str(candidate))
        except Exception as exc:
            print(f"[WARN] Could not read candidate mesh assignments {candidate}: {exc}")
            continue

        candidate_object_names = assignment_object_names(assignment_data)

        if not pose_object_names.issubset(candidate_object_names):
            continue

        object_name_matches.append(candidate)

        candidate_positions = try_load_candidate_positions(candidate.parent)
        if candidate_positions is not None and positions_match(pose_positions, candidate_positions):
            exact_pose_matches.append(candidate)

    if exact_pose_matches:
        selected = exact_pose_matches[0]
        print(f"[DEBUG] Selected mesh assignments by exact pose match: {selected}")
        return str(selected)

    if object_name_matches:
        selected = object_name_matches[0]
        print()
        print("[WARN] No mesh assignment candidate had poses matching the temporary pose file exactly.")
        print("[WARN] Selecting the newest mesh_assignments.json with matching object names.")
        print(f"[WARN] Pose object names: {', '.join(sorted(pose_object_names))}")
        print(f"[WARN] Selected: {selected}")
        print()
        return str(selected)

    available_summaries: List[str] = []

    for candidate in candidates[:20]:
        try:
            assignment_data = read_json_file(str(candidate))
            names = sorted(assignment_object_names(assignment_data))
            available_summaries.append(f"  - {candidate}: {', '.join(names)}")
        except Exception:
            available_summaries.append(f"  - {candidate}: unreadable")

    available_text = "\n".join(available_summaries)

    raise RuntimeError(
        "Could not find a mesh_assignments.json compatible with the pose objects.\n"
        f"Pose object names: {', '.join(sorted(pose_object_names))}\n"
        f"data_root: {data_root}\n"
        "Recent candidates inspected:\n"
        f"{available_text}"
    )


def get_mesh_assignment_for_object(
    mesh_assignments: Dict[str, Any],
    object_name: str,
) -> Dict[str, Any]:
    objects = mesh_assignments.get("objects", {})
    if not isinstance(objects, dict):
        raise RuntimeError("mesh_assignments.json does not contain a valid 'objects' object.")

    item = objects.get(object_name)
    if item is None:
        available = ", ".join(sorted(objects.keys()))
        raise RuntimeError(
            f"No mesh assignment found for object '{object_name}'. "
            f"Available objects in mesh_assignments.json: {available}"
        )

    if not isinstance(item, dict):
        raise RuntimeError(
            f"Invalid mesh assignment for object '{object_name}': expected object/dict."
        )

    return item


def get_mesh_metadata_for_mesh_id(
    mesh_metadata: Dict[str, Any],
    mesh_id: str,
) -> Dict[str, Any]:
    meshes = mesh_metadata.get("meshes", {})
    if not isinstance(meshes, dict):
        raise RuntimeError("mesh_metadata.json does not contain a valid 'meshes' object.")

    item = meshes.get(mesh_id)
    if item is None:
        available = ", ".join(sorted(meshes.keys()))
        raise RuntimeError(
            f"No mesh metadata found for mesh_id={mesh_id!r}. "
            f"Available mesh IDs in mesh_metadata.json: {available}"
        )

    if not isinstance(item, dict):
        raise RuntimeError(
            f"Invalid mesh metadata for mesh_id={mesh_id!r}: expected object/dict."
        )

    return item


def dimension_from_mesh_metadata(
    object_name: str,
    mesh_assignment: Dict[str, Any],
    mesh_metadata: Dict[str, Any],
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Return dimension for AABB in pose frame order:

        [depth, width, height]

    mesh_metadata.json stores:

        dimensions_m.width   -> y axis size, left/right
        dimensions_m.height  -> z axis size, vertical
        dimensions_m.depth   -> x axis size, front/back
    """
    mesh_id = mesh_assignment.get("mesh_id")

    if not isinstance(mesh_id, str) or not mesh_id.strip():
        raise RuntimeError(
            f"No valid mesh_id found for object '{object_name}'. "
            "Check mesh_assignments.json."
        )

    mesh_id = mesh_id.strip()
    metadata_item = get_mesh_metadata_for_mesh_id(mesh_metadata, mesh_id)

    dimensions_m = metadata_item.get("dimensions_m", {})
    if not isinstance(dimensions_m, dict):
        raise RuntimeError(
            f"Invalid or missing dimensions_m for object '{object_name}', mesh_id={mesh_id!r}."
        )

    width = dimensions_m.get("width")
    height = dimensions_m.get("height")
    depth = dimensions_m.get("depth")

    missing = [
        key
        for key, value in [
            ("width", width),
            ("height", height),
            ("depth", depth),
        ]
        if value is None
    ]

    if missing:
        raise RuntimeError(
            f"Missing dimensions {missing} for object '{object_name}', mesh_id={mesh_id!r}. "
            "mesh_metadata.json must contain dimensions_m.width, dimensions_m.height, "
            "and dimensions_m.depth."
        )

    try:
        width_f = float(width)
        height_f = float(height)
        depth_f = float(depth)
    except Exception as exc:
        raise RuntimeError(
            f"Non-numeric dimensions for object '{object_name}', mesh_id={mesh_id!r}: "
            f"width={width!r}, height={height!r}, depth={depth!r}"
        ) from exc

    dimension = [depth_f, width_f, height_f]

    debug_info = {
        "object_name": object_name,
        "mesh_id": mesh_id,
        "mesh_metadata_found": True,
        "dimension_source": "mesh_metadata",
        "mesh_assignment": mesh_assignment,
        "dimensions_m": {
            "width": width_f,
            "height": height_f,
            "depth": depth_f,
        },
        "dimension_aabb_order": {
            "x_depth": depth_f,
            "y_width": width_f,
            "z_height": height_f,
        },
    }

    return dimension, debug_info


# =========================================================
# Geometry
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
    if b_max < a_min:
        return a_min - b_max
    return 0.0


def add_blocker(
    blockers: Dict[str, Dict[str, Set[str]]],
    obj_name: str,
    side: str,
    blocker: str,
) -> None:
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


def is_back_not_reachable(aabb: Dict[str, Tuple[float, float]]) -> bool:
    """
    Back is not reachable if x_max > 0.0.

    Convention:
    - back face = x_max
    """
    _, x_max = aabb["x"]
    return x_max > 0.0


# =========================================================
# Gazebo IO
# =========================================================
def read_gz_dynamic_pose_once(topic: str = GZ_POSE_TOPIC, timeout_sec: float = 3.0) -> str:
    cmd = ["gz", "topic", "-e", "-n", "1", "-t", topic]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            'Command "gz" not found. Make sure Gazebo is installed and the environment is sourced.'
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timeout while reading Gazebo topic {topic}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        raise RuntimeError(f'Error while executing "gz topic": {stderr}') from exc

    return result.stdout


def parse_all_entity_positions(gz_output: str) -> Dict[str, List[float]]:
    positions: Dict[str, List[float]] = {}

    pattern = re.compile(
        r'name:\s*"([^"]+)"\s*'
        r'id:\s*\d+\s*'
        r'position\s*\{\s*'
        r'x:\s*([-\d.eE+]+)\s*'
        r'y:\s*([-\d.eE+]+)\s*'
        r'z:\s*([-\d.eE+]+)\s*'
        r'\}',
        re.MULTILINE | re.DOTALL,
    )

    for match in pattern.finditer(gz_output):
        name = match.group(1)
        x = float(match.group(2))
        y = float(match.group(3))
        z = float(match.group(4))
        positions[name] = [x, y, z]

    return positions


def read_all_positions_from_gazebo(
    topic: str = GZ_POSE_TOPIC,
    timeout_sec: float = 3.0,
) -> Dict[str, List[float]]:
    gz_output = read_gz_dynamic_pose_once(topic=topic, timeout_sec=timeout_sec)
    return parse_all_entity_positions(gz_output)


def read_all_positions(
    pose_source: str = POSE_SOURCE_GAZEBO,
    pose_file: Optional[str] = None,
    topic: str = GZ_POSE_TOPIC,
    timeout_sec: float = 3.0,
) -> Dict[str, List[float]]:
    if pose_source == POSE_SOURCE_GAZEBO:
        return read_all_positions_from_gazebo(topic=topic, timeout_sec=timeout_sec)

    if pose_source == POSE_SOURCE_STATIC:
        if not pose_file:
            raise RuntimeError("pose_file is required when pose_source='static'")
        return read_all_positions_from_static_file(pose_file)

    raise RuntimeError(f"Unsupported pose source: {pose_source}")


# =========================================================
# Accessibility
# =========================================================
def compute_accessibility(
    objects_with_geometry: List[Dict[str, Any]],
    spatial_relationships: List[Dict[str, str]],
    safety_threshold: float = 0.21,
) -> Dict[str, Dict[str, Any]]:
    object_map = {obj["name"]: obj for obj in objects_with_geometry}

    aabbs = {
        name: get_aabb(obj["pose"], obj["dimension"])
        for name, obj in object_map.items()
    }

    blockers = {
        name: {side: set() for side in SIDES}
        for name in object_map.keys()
    }

    grasped_objects: Set[str] = set()
    objects_on_top_of_something: Set[str] = set()

    for rel in spatial_relationships:
        subj = rel["subject"]
        relation = rel["relation"].lower()
        obj = rel["object"]

        if subj not in object_map:
            continue

        if relation == RELATION_ON_TOP_OF:
            if obj not in object_map:
                continue

            add_blocker(blockers, subj, "bottom", obj)
            add_blocker(blockers, obj, "top", subj)
            objects_on_top_of_something.add(subj)

        elif relation == RELATION_GRASPED_BY:
            grasped_objects.add(subj)

    for name in object_map.keys():
        if name not in objects_on_top_of_something:
            add_blocker(blockers, name, "bottom", "table")

    for rel in spatial_relationships:
        subj = rel["subject"]
        relation = rel["relation"].lower()
        obj = rel["object"]

        if subj not in object_map or obj not in object_map:
            continue

        if relation == RELATION_RIGHT_OF:
            gap_y = gap_between_intervals(aabbs[subj]["y"], aabbs[obj]["y"])
            aligned_on_x = intervals_overlap(aabbs[subj]["x"], aabbs[obj]["x"])

            if clamp_lower(gap_y, safety_threshold) and aligned_on_x:
                add_blocker(blockers, subj, "left", obj)
                add_blocker(blockers, obj, "right", subj)

        elif relation == RELATION_IN_FRONT_OF:
            gap_x = gap_between_intervals(aabbs[subj]["x"], aabbs[obj]["x"])
            aligned_on_y = intervals_overlap(aabbs[subj]["y"], aabbs[obj]["y"])

            if clamp_lower(gap_x, safety_threshold) and aligned_on_y:
                add_blocker(blockers, subj, "back", obj)
                add_blocker(blockers, obj, "front", subj)

    for name in grasped_objects:
        if name in blockers:
            for side in SIDES:
                blockers[name][side].clear()

    result: Dict[str, Dict[str, Any]] = {}

    for name in object_map.keys():
        aabb = aabbs[name]

        sides = {
            side: format_side_status(blockers[name][side])
            for side in SIDES
        }

        if is_back_not_reachable(aabb) and not blockers[name]["back"]:
            sides["back"] = "not reachable"

        result[name] = {
            "sides": sides,
            "aabb": {
                axis: [bounds[0], bounds[1]]
                for axis, bounds in aabb.items()
            },
        }

    return result


# =========================================================
# Scene grounding pipeline
# =========================================================
def build_objects_with_geometry(
    scene_objects: List[Dict[str, Any]],
    spatial_relationships: List[Dict[str, str]],
    pose_source: str = POSE_SOURCE_GAZEBO,
    pose_file: Optional[str] = None,
    topic: str = GZ_POSE_TOPIC,
    timeout_sec: float = 3.0,
    direct_name_matching: bool = False,
    mesh_metadata_file: Optional[str] = None,
    mesh_assignments_file: Optional[str] = None,
    data_root: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str], Dict[str, Any]]:
    all_positions = read_all_positions(
        pose_source=pose_source,
        pose_file=pose_file,
        topic=topic,
        timeout_sec=timeout_sec,
    )

    resolved_data_root = infer_data_root(
        data_root=data_root,
        mesh_metadata_file=mesh_metadata_file,
    )

    resolved_mesh_assignments_file: Optional[str] = None
    mesh_metadata: Dict[str, Any] = {}
    mesh_assignments: Dict[str, Any] = {}

    if pose_source == POSE_SOURCE_STATIC:
        resolved_mesh_assignments_file = resolve_mesh_assignments_file(
            pose_file=pose_file,
            mesh_assignments_file=mesh_assignments_file,
            pose_positions=all_positions,
            data_root=resolved_data_root,
        )
        mesh_metadata = load_mesh_metadata(mesh_metadata_file)
        mesh_assignments = load_mesh_assignments(resolved_mesh_assignments_file)

    print(f"\n[DEBUG] Pose source: {pose_source}")
    if pose_source == POSE_SOURCE_STATIC and pose_file:
        print(f"[DEBUG] Static pose file: {pose_file}")
    print("[DEBUG] Direct pose lookup: scene perception names must match pose keys.")
    print(f"[DEBUG] data_root: {resolved_data_root}")
    print(f"[DEBUG] Mesh metadata file: {mesh_metadata_file}")
    print(f"[DEBUG] Mesh assignments file: {resolved_mesh_assignments_file}")

    vlm_to_gazebo: Dict[str, str] = {}
    objects_with_geometry: List[Dict[str, Any]] = []
    matching_warnings: List[str] = []

    for obj in scene_objects:
        vlm_name = obj["name"]

        if vlm_name not in all_positions:
            available = ", ".join(sorted(all_positions.keys()))
            raise RuntimeError(
                f"No pose found for scene object '{vlm_name}'. "
                f"Available pose keys: {available}"
            )

        pose = all_positions[vlm_name]
        vlm_to_gazebo[vlm_name] = vlm_name

        if pose_source == POSE_SOURCE_STATIC:
            mesh_assignment = get_mesh_assignment_for_object(
                mesh_assignments=mesh_assignments,
                object_name=vlm_name,
            )

            dimension, dimension_debug = dimension_from_mesh_metadata(
                object_name=vlm_name,
                mesh_assignment=mesh_assignment,
                mesh_metadata=mesh_metadata,
            )

            mesh_id = dimension_debug["mesh_id"]

        else:
            mesh_assignment = {}
            mesh_id = None
            dimension = list(GAZEBO_DEFAULT_OBJECT_DIMENSION)
            dimension_debug = {
                "object_name": vlm_name,
                "mesh_id": None,
                "dimension_source": "gazebo_default",
                "dimension_aabb_order": {
                    "x_depth": dimension[0],
                    "y_width": dimension[1],
                    "z_height": dimension[2],
                },
            }

        objects_with_geometry.append(
            {
                "name": vlm_name,
                "gazebo_name": vlm_name,
                "pose": pose,
                "dimension": dimension,
                "mesh_id": mesh_id,
                "mesh_assignment": mesh_assignment,
                "dimension_debug": dimension_debug,
            }
        )

        print(
            f"[DEBUG] Object {vlm_name}: "
            f"pose={pose}, "
            f"mesh_id={mesh_id}, "
            f"dimension_aabb=[depth,width,height]={dimension}"
        )

    debug_context = {
        "data_root": resolved_data_root,
        "mesh_metadata_file": mesh_metadata_file,
        "mesh_assignments_file": resolved_mesh_assignments_file,
    }

    return objects_with_geometry, vlm_to_gazebo, matching_warnings, debug_context


def clean_output_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep the scene-description object clean for the VLM.

    Removes any geometry/debug fields that may already exist in the input.
    """
    cleaned = dict(obj)

    for key in [
        "location",
        "pose",
        "dimension",
        "mesh_id",
        "mesh_assignment",
        "dimension_debug",
        "aabb",
    ]:
        cleaned.pop(key, None)

    return cleaned


def enrich_scene(
    input_data: Dict[str, Any],
    safety_threshold: float = 0.21,
    pose_source: str = POSE_SOURCE_GAZEBO,
    pose_file: Optional[str] = None,
    topic: str = GZ_POSE_TOPIC,
    timeout_sec: float = 3.0,
    include_debug_mapping: bool = False,
    direct_name_matching: bool = False,
    mesh_metadata_file: Optional[str] = DEFAULT_MESH_METADATA_FILE,
    mesh_assignments_file: Optional[str] = None,
    data_root: Optional[str] = None,
) -> Dict[str, Any]:
    if "scene_description" not in input_data:
        raise ValueError("Missing required field: 'scene_description'")

    scene_description = input_data["scene_description"]
    scene_objects = scene_description.get("objects", [])
    spatial_relationships = scene_description.get("spatial_relationships", [])

    (
        objects_with_geometry,
        vlm_to_gazebo,
        matching_warnings,
        debug_context,
    ) = build_objects_with_geometry(
        scene_objects=scene_objects,
        spatial_relationships=spatial_relationships,
        pose_source=pose_source,
        pose_file=pose_file,
        topic=topic,
        timeout_sec=timeout_sec,
        direct_name_matching=direct_name_matching,
        mesh_metadata_file=mesh_metadata_file,
        mesh_assignments_file=mesh_assignments_file,
        data_root=data_root,
    )

    computed_info = compute_accessibility(
        objects_with_geometry=objects_with_geometry,
        spatial_relationships=spatial_relationships,
        safety_threshold=safety_threshold,
    )

    enriched_objects: List[Dict[str, Any]] = []

    for obj in scene_objects:
        name = obj["name"]

        enriched_obj = clean_output_object(obj)
        enriched_obj["sides"] = computed_info[name]["sides"]

        enriched_objects.append(enriched_obj)

    output: Dict[str, Any] = {
        "scene_description": {
            "objects": enriched_objects,
            "end_effectors": scene_description.get("end_effectors", []),
            "spatial_relationships": spatial_relationships,
        }
    }

    if include_debug_mapping:
        output["_debug"] = {
            "pose_source": pose_source,
            "pose_file": pose_file,
            "direct_name_matching": direct_name_matching,
            "direct_name_mapping": vlm_to_gazebo,
            "matching_warnings": matching_warnings,
            "objects_with_geometry": objects_with_geometry,
            "computed_accessibility": computed_info,
            "data_root": debug_context.get("data_root"),
            "mesh_metadata_file": debug_context.get("mesh_metadata_file"),
            "mesh_assignments_file": debug_context.get("mesh_assignments_file"),
        }

    return output


# =========================================================
# CLI
# =========================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ground a VLM scene description into Gazebo or a static pose file and enrich it with side accessibility."
    )

    parser.add_argument("input", type=str, help="Path to the input JSON file.")

    parser.add_argument(
        "--output",
        type=str,
        default="scene_description_full.json",
        help="Path to the output JSON file.",
    )

    parser.add_argument(
        "--pose-source",
        type=str,
        choices=[POSE_SOURCE_GAZEBO, POSE_SOURCE_STATIC],
        default=POSE_SOURCE_GAZEBO,
        help="Source used to read object poses: 'gazebo' or 'static'.",
    )

    parser.add_argument(
        "--pose-file",
        type=str,
        default=None,
        help="Path to a JSON file containing static object poses. Required if --pose-source static.",
    )

    parser.add_argument(
        "--mesh-metadata-file",
        type=str,
        default=DEFAULT_MESH_METADATA_FILE,
        help="Path to meshes/mesh_metadata.json. Required in static real-robot mode.",
    )

    parser.add_argument(
        "--mesh-assignments-file",
        type=str,
        default=None,
        help=(
            "Path to mesh_assignments.json. "
            "If omitted, the script first looks next to poses.json; "
            "if live2 passed a temporary poses file, it searches shared_data outputs."
        ),
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Shared Realsense data root. Used to search outputs_by_object/*/mesh_assignments.json "
            "when poses.json is a temporary file."
        ),
    )

    parser.add_argument(
        "--topic",
        type=str,
        default=GZ_POSE_TOPIC,
        help="Gazebo topic used to read dynamic poses.",
    )

    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=3.0,
        help="Timeout in seconds for the Gazebo topic read.",
    )

    parser.add_argument(
        "--safety-threshold",
        type=float,
        default=0.21,
        help="Safety threshold used in accessibility computation.",
    )

    parser.add_argument(
        "--include-debug-mapping",
        action="store_true",
        help="Include debug geometry and direct scene-object-to-pose-key mapping in the output under '_debug'.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.pose_source == POSE_SOURCE_STATIC and not args.pose_file:
            raise ValueError("--pose-file is required when --pose-source static")

        with open(args.input, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        output_data = enrich_scene(
            input_data=input_data,
            safety_threshold=args.safety_threshold,
            pose_source=args.pose_source,
            pose_file=args.pose_file,
            topic=args.topic,
            timeout_sec=args.timeout_sec,
            include_debug_mapping=args.include_debug_mapping,
            mesh_metadata_file=args.mesh_metadata_file,
            mesh_assignments_file=args.mesh_assignments_file,
            data_root=args.data_root,
        )

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Enriched scene saved to: {args.output}")

    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()