
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

SIDES = ["left", "right", "front", "back", "top", "bottom"]

GZ_POSE_TOPIC = "/world/default/dynamic_pose/info"

POSE_SOURCE_GAZEBO = "gazebo"
POSE_SOURCE_STATIC = "static"

# Priority for dimensions:
# 1) object["dimension"] in the input JSON
# 2) DIMENSIONS_BY_NAME[gazebo_name]
# 3) DEFAULT_DIMENSIONS_BY_CATEGORY[canonical_category]
DIMENSIONS_BY_NAME: Dict[str, List[float]] = {
    "box_red_001": [0.28, 0.28, 0.28],
    "box_green_001": [0.28, 0.28, 0.28],
    "glass_001": [0.08, 0.08, 0.08],
    "glass_002": [0.08, 0.08, 0.08],
    "ball_001": [0.07, 0.07, 0.07],
}

DEFAULT_DIMENSIONS_BY_CATEGORY: Dict[str, List[float]] = {
    "box": [0.28, 0.28, 0.28],
    "glass": [0.08, 0.08, 0.08],
    "ball": [0.07, 0.07, 0.07],
}

# Optional explicit metadata overrides for Gazebo entities.
# If an entity is missing here, category/color are inferred from the name.
GAZEBO_OBJECT_METADATA: Dict[str, Dict[str, Any]] = {
    "box_red_001": {"category": "box", "color": "red"},
    "box_green_001": {"category": "box", "color": "green"},
    "glass_001": {"category": "glass"},
    "glass_002": {"category": "glass"},
    "ball_001": {"category": "ball"},
}

CATEGORY_ALIASES: Dict[str, str] = {
    "box": "box",
    "boxes": "box",
    "block": "box",
    "blocks": "box",
    "cube": "box",
    "cubes": "box",
    "container": "box",
    "containers": "box",
    "pot": "box",
    "pots": "box",
    "glass": "glass",
    "glasses": "glass",
    "cup": "glass",
    "cups": "glass",
    "cylinder": "glass",
    "cylinders": "glass",
    "ball": "ball",
    "balls": "ball",
}

COLOR_ALIASES: Dict[str, str] = {
    "grey": "gray",
    "clear": "transparent",
}

KNOWN_CATEGORIES = {"box", "glass", "ball"}
KNOWN_COLORS = {
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "white",
    "black",
    "gray",
    "brown",
    "pink",
    "transparent",
}

# Colors that should not be used as hard discriminative evidence.
NON_DISCRIMINATIVE_COLORS = {
    "multicolor",
    "multi_color",
    "mixed",
    "various",
    "varied",
    "unknown",
    "unspecified",
    "any",
}

RELATION_ON_TOP_OF = "on top of"
RELATION_RIGHT_OF = "right of"
RELATION_IN_FRONT_OF = "in front of"
RELATION_GRASPED_BY = "grasped by"


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
    """
    Return a canonical discriminative color or None if the input color is:
    - missing
    - generic / weakly informative
    - unknown to the system
    """
    if color is None:
        return None

    normalized = normalize_text(color)
    normalized = COLOR_ALIASES.get(normalized, normalized)

    if normalized in NON_DISCRIMINATIVE_COLORS:
        return None

    if normalized not in KNOWN_COLORS:
        return None

    return normalized


def normalize_numeric_suffix(value: str) -> str:
    """
    Normalize trailing numeric groups so that:
    box_red_001 -> box_red_1
    box_02      -> box_2
    """
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
    """
    Semantic prefix without numeric suffix.
    Example:
    - box_red_001 -> box_red
    - glass_2     -> glass
    """
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


def get_location(aabb: Dict[str, Tuple[float, float]]) -> str:
    """
    Zone logic:
    - zone C: right face < -0.75
    - zone A: left face > 0.65
    - zone B: otherwise

    Convention:
    - left face  = y_min
    - right face = y_max
    """
    y_min, y_max = aabb["y"]

    if y_max < -0.75:
        return "zone C"
    if y_min > 0.65:
        return "zone A"
    return "zone B"


def is_back_not_reachable(aabb: Dict[str, Tuple[float, float]]) -> bool:
    """
    Back is not reachable if x_max > 3.75

    Convention:
    - back face = x_max
    """
    _, x_max = aabb["x"]
    return x_max > 3.75


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


def read_all_positions_from_static_file(pose_file: str) -> Dict[str, List[float]]:
    try:
        with open(pose_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Static pose file not found: {pose_file}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in static pose file '{pose_file}': {exc}") from exc

    return validate_positions_dict(data, f"Static pose file '{pose_file}'")


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
# Catalog / metadata
# =========================================================
def build_gazebo_object_catalog(all_positions: Dict[str, List[float]]) -> List[Dict[str, Any]]:
    """
    Build a catalog of Gazebo objects that are relevant for scene grounding.
    The catalog is restricted to entities for which at least one of the following is true:
    - explicit metadata exists
    - a known category can be inferred from the name
    - a dimension override exists for the name
    """
    catalog: List[Dict[str, Any]] = []

    for gazebo_name, pose in all_positions.items():
        # Ignore Gazebo model aliases such as box_red_model
        if normalize_text(gazebo_name).endswith("_model"):
            continue

        explicit = GAZEBO_OBJECT_METADATA.get(gazebo_name, {})
        inferred_category = infer_category_from_name(gazebo_name)
        inferred_color = infer_color_from_name(gazebo_name)

        category = canonicalize_category(explicit.get("category", inferred_category))
        color = canonicalize_color(explicit.get("color", inferred_color))

        if category is None and gazebo_name not in DIMENSIONS_BY_NAME:
            continue

        catalog.append(
            {
                "gazebo_name": gazebo_name,
                "category": category,
                "color": color,
                "pose": pose,
            }
        )

    return catalog


# =========================================================
# Dimension resolution
# =========================================================
def resolve_dimension(obj: Dict[str, Any], gazebo_name: str) -> List[float]:
    """
    Resolve object dimensions with the following priority:
    1) obj["dimension"] in the VLM JSON
    2) DIMENSIONS_BY_NAME[gazebo_name]
    3) DEFAULT_DIMENSIONS_BY_CATEGORY[canonical_category]
    """
    if "dimension" in obj:
        dim = obj["dimension"]
        if isinstance(dim, list) and len(dim) == 3:
            return dim
        raise ValueError(f"Invalid dimension for object '{obj.get('name')}': {dim}")

    if gazebo_name in DIMENSIONS_BY_NAME:
        return DIMENSIONS_BY_NAME[gazebo_name]

    category = canonicalize_category(obj.get("category"))
    if category in DEFAULT_DIMENSIONS_BY_CATEGORY:
        return DEFAULT_DIMENSIONS_BY_CATEGORY[category]

    raise ValueError(
        f"No dimension available for object '{obj.get('name')}' "
        f"(mapped to Gazebo entity '{gazebo_name}'). "
        "Add 'dimension' to the JSON or configure DIMENSIONS_BY_NAME / "
        "DEFAULT_DIMENSIONS_BY_CATEGORY."
    )


# =========================================================
# Matching helpers
# =========================================================
def strong_name_score(vlm_name: str, gazebo_name: str) -> int:
    """
    Higher score means stronger name compatibility.
    """
    vlm_norm = normalize_numeric_suffix(vlm_name)
    gz_norm = normalize_numeric_suffix(gazebo_name)

    if vlm_norm == gz_norm:
        return 100

    vlm_prefix = semantic_prefix(vlm_name)
    gz_prefix = semantic_prefix(gazebo_name)

    vlm_index = extract_index_suffix(vlm_name)
    gz_index = extract_index_suffix(gazebo_name)

    if vlm_prefix and vlm_prefix == gz_prefix:
        if vlm_index is not None and gz_index is not None and vlm_index == gz_index:
            return 95
        return 80

    vlm_tokens = set(split_canonical_tokens(vlm_name))
    gz_tokens = set(split_canonical_tokens(gazebo_name))
    common = vlm_tokens & gz_tokens

    if len(common) >= 2:
        return 60
    if len(common) == 1:
        return 30

    return 0


def is_semantically_compatible(vlm_obj: Dict[str, Any], gz_obj: Dict[str, Any]) -> bool:
    """
    Hard semantic filter:
    - category mismatch is incompatible
    - color is intentionally NOT a hard constraint, because VLM color can be noisy
    """
    vlm_category = canonicalize_category(vlm_obj.get("category"))
    gz_category = canonicalize_category(gz_obj.get("category"))

    if vlm_category and gz_category and vlm_category != gz_category:
        return False

    return True


def build_candidate_lists(
    scene_objects: List[Dict[str, Any]],
    gazebo_catalog: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    candidates_by_vlm_name: Dict[str, List[Dict[str, Any]]] = {}

    for vlm_obj in scene_objects:
        compatible = [gz_obj for gz_obj in gazebo_catalog if is_semantically_compatible(vlm_obj, gz_obj)]
        if not compatible:
            raise ValueError(
                f"No Gazebo candidates found for VLM object '{vlm_obj.get('name')}' "
                f"with category='{vlm_obj.get('category')}' and color='{vlm_obj.get('color')}'."
            )
        candidates_by_vlm_name[vlm_obj["name"]] = compatible

    return candidates_by_vlm_name


def relation_pair_score(
    relation: str,
    subj_pose: List[float],
    obj_pose: List[float],
    subj_dim: Optional[List[float]],
    obj_dim: Optional[List[float]],
) -> int:
    relation = relation.lower()

    subj_x, subj_y, subj_z = subj_pose
    obj_x, obj_y, obj_z = obj_pose

    score = 0

    if relation == RELATION_ON_TOP_OF:
        if subj_z > obj_z:
            score += 6
        else:
            score -= 6

        if subj_dim and obj_dim:
            subj_aabb = get_aabb(subj_pose, subj_dim)
            obj_aabb = get_aabb(obj_pose, obj_dim)

            if intervals_overlap(subj_aabb["x"], obj_aabb["x"]):
                score += 1
            if intervals_overlap(subj_aabb["y"], obj_aabb["y"]):
                score += 1

    elif relation == RELATION_RIGHT_OF:
        if subj_y > obj_y:
            score += 4
        else:
            score -= 4

    elif relation == RELATION_IN_FRONT_OF:
        # Convention used by the existing accessibility logic:
        # larger x means farther back; therefore "in front of" means smaller x.
        if subj_x < obj_x:
            score += 4
        else:
            score -= 4

    elif relation == RELATION_GRASPED_BY:
        score += 0

    return score


def estimate_dimension_for_scoring(vlm_obj: Dict[str, Any], gz_obj: Dict[str, Any]) -> Optional[List[float]]:
    gazebo_name = gz_obj["gazebo_name"]

    if "dimension" in vlm_obj:
        dim = vlm_obj["dimension"]
        if isinstance(dim, list) and len(dim) == 3:
            return dim

    if gazebo_name in DIMENSIONS_BY_NAME:
        return DIMENSIONS_BY_NAME[gazebo_name]

    category = canonicalize_category(vlm_obj.get("category") or gz_obj.get("category"))
    if category in DEFAULT_DIMENSIONS_BY_CATEGORY:
        return DEFAULT_DIMENSIONS_BY_CATEGORY[category]

    return None


def color_score(vlm_obj: Dict[str, Any], gz_obj: Dict[str, Any]) -> int:
    """
    Soft color evidence:
    - if VLM color is missing / generic / unknown -> neutral
    - if both are present and equal -> reward
    - if both are present and different -> mild penalty
    """
    vlm_color = canonicalize_color(vlm_obj.get("color"))
    gz_color = canonicalize_color(gz_obj.get("color"))

    if vlm_color is None or gz_color is None:
        return 0

    if vlm_color == gz_color:
        return 12

    return -4


def score_assignment(
    assignment: Dict[str, Dict[str, Any]],
    scene_objects_by_name: Dict[str, Dict[str, Any]],
    spatial_relationships: List[Dict[str, str]],
) -> int:
    score = 0

    for vlm_name, gz_obj in assignment.items():
        score += strong_name_score(vlm_name, gz_obj["gazebo_name"])

        vlm_obj = scene_objects_by_name[vlm_name]
        vlm_category = canonicalize_category(vlm_obj.get("category"))
        gz_category = canonicalize_category(gz_obj.get("category"))

        if vlm_category and gz_category and vlm_category == gz_category:
            score += 10

        score += color_score(vlm_obj, gz_obj)

    for rel in spatial_relationships:
        subj = rel.get("subject")
        obj = rel.get("object")
        relation = rel.get("relation", "").lower()

        if subj not in assignment or obj not in assignment:
            continue

        subj_gz = assignment[subj]
        obj_gz = assignment[obj]

        subj_dim = estimate_dimension_for_scoring(scene_objects_by_name[subj], subj_gz)
        obj_dim = estimate_dimension_for_scoring(scene_objects_by_name[obj], obj_gz)

        score += relation_pair_score(
            relation=relation,
            subj_pose=subj_gz["pose"],
            obj_pose=obj_gz["pose"],
            subj_dim=subj_dim,
            obj_dim=obj_dim,
        )

    return score


def build_matching_warnings(
    scene_objects: List[Dict[str, Any]],
    gazebo_catalog: List[Dict[str, Any]],
    vlm_to_gazebo: Dict[str, str],
) -> List[str]:
    warnings: List[str] = []
    gazebo_by_name = {obj["gazebo_name"]: obj for obj in gazebo_catalog}

    for vlm_obj in scene_objects:
        vlm_name = vlm_obj["name"]
        gazebo_name = vlm_to_gazebo[vlm_name]
        gz_obj = gazebo_by_name[gazebo_name]

        raw_vlm_color = vlm_obj.get("color")
        raw_gz_color = gz_obj.get("color")

        vlm_color = canonicalize_color(raw_vlm_color)
        gz_color = canonicalize_color(raw_gz_color)

        if original_color_is_non_discriminative(raw_vlm_color):
            warnings.append(
                f"VLM object '{vlm_name}' has weak/non-discriminative color '{raw_vlm_color}', "
                f"so color was ignored as a hard constraint."
            )

        if raw_vlm_color is not None and vlm_color is None and not original_color_is_non_discriminative(raw_vlm_color):
            warnings.append(
                f"VLM object '{vlm_name}' has unsupported or unknown color '{raw_vlm_color}', "
                f"so color was ignored in semantic filtering."
            )

        if vlm_color is not None and gz_color is not None and vlm_color != gz_color:
            warnings.append(
                f"Color mismatch for '{vlm_name}': VLM color='{vlm_color}', "
                f"matched Gazebo entity '{gazebo_name}' with color='{gz_color}'. "
                "Assignment kept because color is treated as soft evidence."
            )

    return warnings


def resolve_vlm_to_gazebo_mapping(
    scene_objects: List[Dict[str, Any]],
    spatial_relationships: List[Dict[str, str]],
    gazebo_catalog: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], List[str]]:
    """
    Resolve a one-to-one mapping:
        VLM object name -> Gazebo entity name

    Strategy:
    1) semantic candidate filtering using hard constraints only (mainly category)
    2) global one-to-one search across remaining candidates
    3) score assignments using:
       - strong name compatibility
       - category agreement
       - color as soft evidence
       - relation consistency
    4) fail explicitly if best assignment is still ambiguous
    """
    if not scene_objects:
        return {}, []

    scene_objects_by_name = {obj["name"]: obj for obj in scene_objects}
    candidates_by_vlm_name = build_candidate_lists(scene_objects, gazebo_catalog)

    ordered_vlm_names = sorted(
        [obj["name"] for obj in scene_objects],
        key=lambda name: (
            len(candidates_by_vlm_name[name]),
            -max(strong_name_score(name, gz["gazebo_name"]) for gz in candidates_by_vlm_name[name]),
            name,
        ),
    )

    best_assignments: List[Dict[str, Dict[str, Any]]] = []
    best_score: Optional[int] = None

    def backtrack(
        index: int,
        partial_assignment: Dict[str, Dict[str, Any]],
        used_gazebo_names: Set[str],
    ) -> None:
        nonlocal best_score, best_assignments

        if index == len(ordered_vlm_names):
            total_score = score_assignment(
                assignment=partial_assignment,
                scene_objects_by_name=scene_objects_by_name,
                spatial_relationships=spatial_relationships,
            )

            if best_score is None or total_score > best_score:
                best_score = total_score
                best_assignments = [dict(partial_assignment)]
            elif total_score == best_score:
                best_assignments.append(dict(partial_assignment))
            return

        vlm_name = ordered_vlm_names[index]
        raw_candidates = candidates_by_vlm_name[vlm_name]

        sorted_candidates = sorted(
            raw_candidates,
            key=lambda gz: (
                -strong_name_score(vlm_name, gz["gazebo_name"]),
                gz["gazebo_name"],
            ),
        )

        for gz_obj in sorted_candidates:
            gazebo_name = gz_obj["gazebo_name"]
            if gazebo_name in used_gazebo_names:
                continue

            partial_assignment[vlm_name] = gz_obj
            used_gazebo_names.add(gazebo_name)

            backtrack(index + 1, partial_assignment, used_gazebo_names)

            used_gazebo_names.remove(gazebo_name)
            del partial_assignment[vlm_name]

    backtrack(0, {}, set())

    if not best_assignments:
        raise ValueError("Unable to resolve any valid one-to-one VLM-to-Gazebo assignment.")

    if len(best_assignments) > 1:
        rendered = []
        for assignment in best_assignments[:5]:
            rendered.append(
                {vlm_name: gz_obj["gazebo_name"] for vlm_name, gz_obj in sorted(assignment.items())}
            )
        raise ValueError(
            "Ambiguous VLM-to-Gazebo mapping. Multiple equally valid assignments were found. "
            f"Examples: {rendered}"
        )

    best_assignment = best_assignments[0]
    vlm_to_gazebo = {vlm_name: gz_obj["gazebo_name"] for vlm_name, gz_obj in best_assignment.items()}
    warnings = build_matching_warnings(scene_objects, gazebo_catalog, vlm_to_gazebo)
    return vlm_to_gazebo, warnings


# =========================================================
# Accessibility
# =========================================================
def compute_accessibility(
    objects_with_geometry: List[Dict[str, Any]],
    spatial_relationships: List[Dict[str, str]],
    safety_threshold: float = 0.21,
) -> Dict[str, Dict[str, Any]]:
    """
    Return:
    {
        "vlm_object_name": {
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
            "location": get_location(aabb),
            "sides": sides,
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
) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
    """
    Build objects with:
    - VLM name (kept unchanged)
    - Gazebo name (internal only)
    - pose (from Gazebo or static file)
    - dimension
    """
    all_positions = read_all_positions(
        pose_source=pose_source,
        pose_file=pose_file,
        topic=topic,
        timeout_sec=timeout_sec,
    )

    gazebo_catalog = build_gazebo_object_catalog(all_positions)

    print(f"\n[DEBUG] Pose source: {pose_source}")
    if pose_source == POSE_SOURCE_STATIC and pose_file:
        print(f"[DEBUG] Static pose file: {pose_file}")
    print("[DEBUG] Filtered object catalog:")
    for obj in gazebo_catalog:
        print(obj["gazebo_name"])
    print()

    if not gazebo_catalog:
        raise RuntimeError(
            "No eligible object entities were found in the selected pose source. "
            "Check the pose file/topic output and the configured metadata/categories."
        )

    vlm_to_gazebo, matching_warnings = resolve_vlm_to_gazebo_mapping(
        scene_objects=scene_objects,
        spatial_relationships=spatial_relationships,
        gazebo_catalog=gazebo_catalog,
    )

    objects_with_geometry: List[Dict[str, Any]] = []

    for obj in scene_objects:
        vlm_name = obj["name"]
        gazebo_name = vlm_to_gazebo[vlm_name]

        if gazebo_name not in all_positions:
            raise RuntimeError(
                f"Mapped entity '{gazebo_name}' was not found in the current pose source."
            )

        pose = all_positions[gazebo_name]
        dimension = resolve_dimension(obj=obj, gazebo_name=gazebo_name)

        objects_with_geometry.append(
            {
                "name": vlm_name,
                "gazebo_name": gazebo_name,
                "pose": pose,
                "dimension": dimension,
            }
        )

    return objects_with_geometry, vlm_to_gazebo, matching_warnings


def enrich_scene(
    input_data: Dict[str, Any],
    safety_threshold: float = 0.21,
    pose_source: str = POSE_SOURCE_GAZEBO,
    pose_file: Optional[str] = None,
    topic: str = GZ_POSE_TOPIC,
    timeout_sec: float = 3.0,
    include_debug_mapping: bool = False,
) -> Dict[str, Any]:
    if "scene_description" not in input_data:
        raise ValueError("Missing required field: 'scene_description'")

    scene_description = input_data["scene_description"]
    scene_objects = scene_description.get("objects", [])
    spatial_relationships = scene_description.get("spatial_relationships", [])

    objects_with_geometry, vlm_to_gazebo, matching_warnings = build_objects_with_geometry(
        scene_objects=scene_objects,
        spatial_relationships=spatial_relationships,
        pose_source=pose_source,
        pose_file=pose_file,
        topic=topic,
        timeout_sec=timeout_sec,
    )

    computed_info = compute_accessibility(
        objects_with_geometry=objects_with_geometry,
        spatial_relationships=spatial_relationships,
        safety_threshold=safety_threshold,
    )

    enriched_objects: List[Dict[str, Any]] = []

    for obj in scene_objects:
        name = obj["name"]
        enriched_obj = dict(obj)
        enriched_obj["location"] = computed_info[name]["location"]
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
            "vlm_to_gazebo_mapping": vlm_to_gazebo,
            "matching_warnings": matching_warnings,
        }

    return output


# =========================================================
# CLI
# =========================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ground a VLM scene description into Gazebo or a static pose file and enrich it with location/sides."
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
        help="Include the internal VLM-to-Gazebo mapping in the output under '_debug'.",
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















#### ORIGINAL 

#!/usr/bin/env python3

# from __future__ import annotations

# import argparse
# import json
# import re
# import subprocess
# import sys
# from typing import Any, Dict, List, Optional, Set, Tuple

# SIDES = ["left", "right", "front", "back", "top", "bottom"]

# GZ_POSE_TOPIC = "/world/default/dynamic_pose/info"

# POSE_SOURCE_GAZEBO = "gazebo"
# POSE_SOURCE_STATIC = "static"

# # Priority for dimensions:
# # 1) object["dimension"] in the input JSON
# # 2) DIMENSIONS_BY_NAME[gazebo_name]
# # 3) DEFAULT_DIMENSIONS_BY_CATEGORY[canonical_category]
# DIMENSIONS_BY_NAME: Dict[str, List[float]] = {
#     "box_red_001": [0.28, 0.28, 0.28],
#     "box_green_001": [0.28, 0.28, 0.28],
#     "glass_001": [0.08, 0.08, 0.08],
#     "glass_002": [0.08, 0.08, 0.08],
#     "ball_001": [0.07, 0.07, 0.07],
# }

# DEFAULT_DIMENSIONS_BY_CATEGORY: Dict[str, List[float]] = {
#     "box": [0.28, 0.28, 0.28],
#     "glass": [0.08, 0.08, 0.08],
#     "ball": [0.07, 0.07, 0.07],
# }

# # Optional explicit metadata overrides for Gazebo entities.
# # If an entity is missing here, category/color are inferred from the name.
# GAZEBO_OBJECT_METADATA: Dict[str, Dict[str, Any]] = {
#     "box_red_001": {"category": "box", "color": "red"},
#     "box_green_001": {"category": "box", "color": "green"},
#     "glass_001": {"category": "glass"},
#     "glass_002": {"category": "glass"},
#     "ball_001": {"category": "ball"},
# }

# CATEGORY_ALIASES: Dict[str, str] = {
#     "box": "box",
#     "boxes": "box",
#     "block": "box",
#     "blocks": "box",
#     "cube": "box",
#     "cubes": "box",
#     "container": "box",
#     "containers": "box",
#     "pot": "box",
#     "pots": "box",
#     "glass": "glass",
#     "glasses": "glass",
#     "cup": "glass",
#     "cups": "glass",
#     "cylinder": "glass",
#     "cylinders": "glass",
#     "ball": "ball",
#     "balls": "ball",
# }

# COLOR_ALIASES: Dict[str, str] = {
#     "grey": "gray",
#     "clear": "transparent",
# }

# KNOWN_CATEGORIES = {"box", "glass", "ball"}
# KNOWN_COLORS = {
#     "red",
#     "green",
#     "blue",
#     "yellow",
#     "orange",
#     "purple",
#     "white",
#     "black",
#     "gray",
#     "brown",
#     "pink",
#     "transparent",
# }

# # Colors that should not be used as hard discriminative evidence.
# NON_DISCRIMINATIVE_COLORS = {
#     "multicolor",
#     "multi_color",
#     "mixed",
#     "various",
#     "varied",
#     "unknown",
#     "unspecified",
#     "any",
# }

# RELATION_ON_TOP_OF = "on top of"
# RELATION_RIGHT_OF = "right of"
# RELATION_IN_FRONT_OF = "in front of"
# RELATION_GRASPED_BY = "grasped by"


# # =========================================================
# # Utility helpers
# # =========================================================
# def normalize_text(value: str) -> str:
#     return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower())


# def split_tokens(value: str) -> List[str]:
#     return [token for token in normalize_text(value).split("_") if token]


# def canonicalize_token(token: str) -> str:
#     token = normalize_text(token)
#     token = CATEGORY_ALIASES.get(token, token)
#     token = COLOR_ALIASES.get(token, token)
#     return token


# def split_canonical_tokens(value: str) -> List[str]:
#     return [canonicalize_token(token) for token in split_tokens(value)]


# def canonicalize_category(category: Optional[str]) -> Optional[str]:
#     if category is None:
#         return None
#     normalized = normalize_text(category)
#     return CATEGORY_ALIASES.get(normalized, normalized)


# def canonicalize_color(color: Optional[str]) -> Optional[str]:
#     """
#     Return a canonical discriminative color or None if the input color is:
#     - missing
#     - generic / weakly informative
#     - unknown to the system
#     """
#     if color is None:
#         return None

#     normalized = normalize_text(color)
#     normalized = COLOR_ALIASES.get(normalized, normalized)

#     if normalized in NON_DISCRIMINATIVE_COLORS:
#         return None

#     if normalized not in KNOWN_COLORS:
#         return None

#     return normalized


# def normalize_numeric_suffix(value: str) -> str:
#     """
#     Normalize trailing numeric groups so that:
#     box_red_001 -> box_red_1
#     box_02      -> box_2
#     """
#     tokens = split_tokens(value)
#     if not tokens:
#         return ""

#     normalized: List[str] = []
#     for token in tokens:
#         if token.isdigit():
#             normalized.append(str(int(token)))
#         else:
#             normalized.append(token)
#     return "_".join(normalized)


# def extract_index_suffix(value: str) -> Optional[int]:
#     tokens = split_tokens(value)
#     if not tokens:
#         return None
#     last = tokens[-1]
#     if last.isdigit():
#         return int(last)
#     return None


# def infer_category_from_name(name: str) -> Optional[str]:
#     tokens = set(split_canonical_tokens(name))
#     for token in tokens:
#         if token in KNOWN_CATEGORIES:
#             return token
#     return None


# def infer_color_from_name(name: str) -> Optional[str]:
#     tokens = set(split_canonical_tokens(name))
#     for token in tokens:
#         if token in KNOWN_COLORS:
#             return token
#     return None


# def semantic_prefix(name: str) -> str:
#     """
#     Semantic prefix without numeric suffix.
#     Example:
#     - box_red_001 -> box_red
#     - glass_2     -> glass
#     """
#     tokens = split_canonical_tokens(name)
#     if tokens and tokens[-1].isdigit():
#         tokens = tokens[:-1]
#     return "_".join(tokens)


# def intervals_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
#     return not (a[1] < b[0] or b[1] < a[0])


# def clamp_lower(value: float, threshold: float) -> bool:
#     return value <= threshold


# def original_color_is_non_discriminative(color: Optional[str]) -> bool:
#     if color is None:
#         return False
#     normalized = normalize_text(color)
#     normalized = COLOR_ALIASES.get(normalized, normalized)
#     return normalized in NON_DISCRIMINATIVE_COLORS


# # =========================================================
# # Geometry
# # =========================================================
# def get_aabb(pose: List[float], dimension: List[float]) -> Dict[str, Tuple[float, float]]:
#     x, y, z = pose
#     dx, dy, dz = dimension

#     return {
#         "x": (x - dx / 2.0, x + dx / 2.0),
#         "y": (y - dy / 2.0, y + dy / 2.0),
#         "z": (z - dz / 2.0, z + dz / 2.0),
#     }


# def gap_between_intervals(a: Tuple[float, float], b: Tuple[float, float]) -> float:
#     a_min, a_max = a
#     b_min, b_max = b

#     if a_max < b_min:
#         return b_min - a_max
#     if b_max < a_min:
#         return a_min - b_max
#     return 0.0


# def add_blocker(
#     blockers: Dict[str, Dict[str, Set[str]]],
#     obj_name: str,
#     side: str,
#     blocker: str,
# ) -> None:
#     blockers[obj_name][side].add(blocker)


# def format_side_status(blocker_set: Set[str]) -> str:
#     if not blocker_set:
#         return "accessible"

#     if len(blocker_set) == 1:
#         blocker = next(iter(blocker_set))
#         if blocker == "table":
#             return "blocked by the table"
#         return f"blocked by {blocker}"

#     return "blocked by " + ", ".join(sorted(blocker_set))


# def get_location(aabb: Dict[str, Tuple[float, float]]) -> str:
#     """
#     Zone logic:
#     - zone C: right face < -0.75
#     - zone A: left face > 0.65
#     - zone B: otherwise

#     Convention:
#     - left face  = y_min
#     - right face = y_max
#     """
#     y_min, y_max = aabb["y"]

#     if y_max < -0.75:
#         return "zone C"
#     if y_min > 0.65:
#         return "zone A"
#     return "zone B"


# def is_back_not_reachable(aabb: Dict[str, Tuple[float, float]]) -> bool:
#     """
#     Back is not reachable if x_max > 3.75

#     Convention:
#     - back face = x_max
#     """
#     _, x_max = aabb["x"]
#     return x_max > 3.75


# # =========================================================
# # Gazebo IO
# # =========================================================
# def read_gz_dynamic_pose_once(topic: str = GZ_POSE_TOPIC, timeout_sec: float = 3.0) -> str:
#     cmd = ["gz", "topic", "-e", "-n", "1", "-t", topic]

#     try:
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             timeout=timeout_sec,
#             check=True,
#         )
#     except FileNotFoundError as exc:
#         raise RuntimeError(
#             'Command "gz" not found. Make sure Gazebo is installed and the environment is sourced.'
#         ) from exc
#     except subprocess.TimeoutExpired as exc:
#         raise RuntimeError(f"Timeout while reading Gazebo topic {topic}") from exc
#     except subprocess.CalledProcessError as exc:
#         stderr = exc.stderr.strip() if exc.stderr else str(exc)
#         raise RuntimeError(f'Error while executing "gz topic": {stderr}') from exc

#     return result.stdout


# def parse_all_entity_positions(gz_output: str) -> Dict[str, List[float]]:
#     positions: Dict[str, List[float]] = {}

#     pattern = re.compile(
#         r'name:\s*"([^"]+)"\s*'
#         r'id:\s*\d+\s*'
#         r'position\s*\{\s*'
#         r'x:\s*([-\d.eE+]+)\s*'
#         r'y:\s*([-\d.eE+]+)\s*'
#         r'z:\s*([-\d.eE+]+)\s*'
#         r'\}',
#         re.MULTILINE | re.DOTALL,
#     )

#     for match in pattern.finditer(gz_output):
#         name = match.group(1)
#         x = float(match.group(2))
#         y = float(match.group(3))
#         z = float(match.group(4))
#         positions[name] = [x, y, z]

#     return positions


# def read_all_positions_from_gazebo(
#     topic: str = GZ_POSE_TOPIC,
#     timeout_sec: float = 3.0,
# ) -> Dict[str, List[float]]:
#     gz_output = read_gz_dynamic_pose_once(topic=topic, timeout_sec=timeout_sec)
#     return parse_all_entity_positions(gz_output)


# # =========================================================
# # Static pose IO
# # =========================================================
# def validate_positions_dict(raw_positions: Dict[str, Any], source_name: str) -> Dict[str, List[float]]:
#     positions: Dict[str, List[float]] = {}

#     if not isinstance(raw_positions, dict):
#         raise RuntimeError(f"{source_name} must contain a JSON object mapping object names to [x, y, z].")

#     for name, pose in raw_positions.items():
#         if not isinstance(name, str):
#             raise RuntimeError(f"{source_name}: each object name must be a string.")

#         if not isinstance(pose, list) or len(pose) != 3:
#             raise RuntimeError(
#                 f"{source_name}: pose for '{name}' must be a list of exactly 3 numeric values."
#             )

#         if not all(isinstance(v, (int, float)) for v in pose):
#             raise RuntimeError(
#                 f"{source_name}: pose for '{name}' must contain only numeric values."
#             )

#         positions[name] = [float(v) for v in pose]

#     return positions


# def read_all_positions_from_static_file(pose_file: str) -> Dict[str, List[float]]:
#     try:
#         with open(pose_file, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except FileNotFoundError as exc:
#         raise RuntimeError(f"Static pose file not found: {pose_file}") from exc
#     except json.JSONDecodeError as exc:
#         raise RuntimeError(f"Invalid JSON in static pose file '{pose_file}': {exc}") from exc

#     return validate_positions_dict(data, f"Static pose file '{pose_file}'")


# def read_all_positions(
#     pose_source: str = POSE_SOURCE_GAZEBO,
#     pose_file: Optional[str] = None,
#     topic: str = GZ_POSE_TOPIC,
#     timeout_sec: float = 3.0,
# ) -> Dict[str, List[float]]:
#     if pose_source == POSE_SOURCE_GAZEBO:
#         return read_all_positions_from_gazebo(topic=topic, timeout_sec=timeout_sec)

#     if pose_source == POSE_SOURCE_STATIC:
#         if not pose_file:
#             raise RuntimeError("pose_file is required when pose_source='static'")
#         return read_all_positions_from_static_file(pose_file)

#     raise RuntimeError(f"Unsupported pose source: {pose_source}")


# # =========================================================
# # Catalog / metadata
# # =========================================================
# def build_gazebo_object_catalog(all_positions: Dict[str, List[float]]) -> List[Dict[str, Any]]:
#     """
#     Build a catalog of Gazebo objects that are relevant for scene grounding.
#     The catalog is restricted to entities for which at least one of the following is true:
#     - explicit metadata exists
#     - a known category can be inferred from the name
#     - a dimension override exists for the name
#     """
#     catalog: List[Dict[str, Any]] = []

#     for gazebo_name, pose in all_positions.items():
#         # Ignore Gazebo model aliases such as box_red_model
#         if normalize_text(gazebo_name).endswith("_model"):
#             continue

#         explicit = GAZEBO_OBJECT_METADATA.get(gazebo_name, {})
#         inferred_category = infer_category_from_name(gazebo_name)
#         inferred_color = infer_color_from_name(gazebo_name)

#         category = canonicalize_category(explicit.get("category", inferred_category))
#         color = canonicalize_color(explicit.get("color", inferred_color))

#         if category is None and gazebo_name not in DIMENSIONS_BY_NAME:
#             continue

#         catalog.append(
#             {
#                 "gazebo_name": gazebo_name,
#                 "category": category,
#                 "color": color,
#                 "pose": pose,
#             }
#         )

#     return catalog


# # =========================================================
# # Dimension resolution
# # =========================================================
# def resolve_dimension(obj: Dict[str, Any], gazebo_name: str) -> List[float]:
#     """
#     Resolve object dimensions with the following priority:
#     1) obj["dimension"] in the VLM JSON
#     2) DIMENSIONS_BY_NAME[gazebo_name]
#     3) DEFAULT_DIMENSIONS_BY_CATEGORY[canonical_category]
#     """
#     if "dimension" in obj:
#         dim = obj["dimension"]
#         if isinstance(dim, list) and len(dim) == 3:
#             return dim
#         raise ValueError(f"Invalid dimension for object '{obj.get('name')}': {dim}")

#     if gazebo_name in DIMENSIONS_BY_NAME:
#         return DIMENSIONS_BY_NAME[gazebo_name]

#     category = canonicalize_category(obj.get("category"))
#     if category in DEFAULT_DIMENSIONS_BY_CATEGORY:
#         return DEFAULT_DIMENSIONS_BY_CATEGORY[category]

#     raise ValueError(
#         f"No dimension available for object '{obj.get('name')}' "
#         f"(mapped to Gazebo entity '{gazebo_name}'). "
#         "Add 'dimension' to the JSON or configure DIMENSIONS_BY_NAME / "
#         "DEFAULT_DIMENSIONS_BY_CATEGORY."
#     )


# # =========================================================
# # Matching helpers
# # =========================================================
# def strong_name_score(vlm_name: str, gazebo_name: str) -> int:
#     """
#     Higher score means stronger name compatibility.
#     """
#     vlm_norm = normalize_numeric_suffix(vlm_name)
#     gz_norm = normalize_numeric_suffix(gazebo_name)

#     if vlm_norm == gz_norm:
#         return 100

#     vlm_prefix = semantic_prefix(vlm_name)
#     gz_prefix = semantic_prefix(gazebo_name)

#     vlm_index = extract_index_suffix(vlm_name)
#     gz_index = extract_index_suffix(gazebo_name)

#     if vlm_prefix and vlm_prefix == gz_prefix:
#         if vlm_index is not None and gz_index is not None and vlm_index == gz_index:
#             return 95
#         return 80

#     vlm_tokens = set(split_canonical_tokens(vlm_name))
#     gz_tokens = set(split_canonical_tokens(gazebo_name))
#     common = vlm_tokens & gz_tokens

#     if len(common) >= 2:
#         return 60
#     if len(common) == 1:
#         return 30

#     return 0


# def is_semantically_compatible(vlm_obj: Dict[str, Any], gz_obj: Dict[str, Any]) -> bool:
#     """
#     Hard semantic filter:
#     - category mismatch is incompatible
#     - color is intentionally NOT a hard constraint, because VLM color can be noisy
#     """
#     vlm_category = canonicalize_category(vlm_obj.get("category"))
#     gz_category = canonicalize_category(gz_obj.get("category"))

#     if vlm_category and gz_category and vlm_category != gz_category:
#         return False

#     return True


# def build_candidate_lists(
#     scene_objects: List[Dict[str, Any]],
#     gazebo_catalog: List[Dict[str, Any]],
# ) -> Dict[str, List[Dict[str, Any]]]:
#     candidates_by_vlm_name: Dict[str, List[Dict[str, Any]]] = {}

#     for vlm_obj in scene_objects:
#         compatible = [gz_obj for gz_obj in gazebo_catalog if is_semantically_compatible(vlm_obj, gz_obj)]
#         if not compatible:
#             raise ValueError(
#                 f"No Gazebo candidates found for VLM object '{vlm_obj.get('name')}' "
#                 f"with category='{vlm_obj.get('category')}' and color='{vlm_obj.get('color')}'."
#             )
#         candidates_by_vlm_name[vlm_obj["name"]] = compatible

#     return candidates_by_vlm_name


# def relation_pair_score(
#     relation: str,
#     subj_pose: List[float],
#     obj_pose: List[float],
#     subj_dim: Optional[List[float]],
#     obj_dim: Optional[List[float]],
# ) -> int:
#     relation = relation.lower()

#     subj_x, subj_y, subj_z = subj_pose
#     obj_x, obj_y, obj_z = obj_pose

#     score = 0

#     if relation == RELATION_ON_TOP_OF:
#         if subj_z > obj_z:
#             score += 6
#         else:
#             score -= 6

#         if subj_dim and obj_dim:
#             subj_aabb = get_aabb(subj_pose, subj_dim)
#             obj_aabb = get_aabb(obj_pose, obj_dim)

#             if intervals_overlap(subj_aabb["x"], obj_aabb["x"]):
#                 score += 1
#             if intervals_overlap(subj_aabb["y"], obj_aabb["y"]):
#                 score += 1

#     elif relation == RELATION_RIGHT_OF:
#         if subj_y > obj_y:
#             score += 4
#         else:
#             score -= 4

#     elif relation == RELATION_IN_FRONT_OF:
#         # Convention used by the existing accessibility logic:
#         # larger x means farther back; therefore "in front of" means smaller x.
#         if subj_x < obj_x:
#             score += 4
#         else:
#             score -= 4

#     elif relation == RELATION_GRASPED_BY:
#         score += 0

#     return score


# def estimate_dimension_for_scoring(vlm_obj: Dict[str, Any], gz_obj: Dict[str, Any]) -> Optional[List[float]]:
#     gazebo_name = gz_obj["gazebo_name"]

#     if "dimension" in vlm_obj:
#         dim = vlm_obj["dimension"]
#         if isinstance(dim, list) and len(dim) == 3:
#             return dim

#     if gazebo_name in DIMENSIONS_BY_NAME:
#         return DIMENSIONS_BY_NAME[gazebo_name]

#     category = canonicalize_category(vlm_obj.get("category") or gz_obj.get("category"))
#     if category in DEFAULT_DIMENSIONS_BY_CATEGORY:
#         return DEFAULT_DIMENSIONS_BY_CATEGORY[category]

#     return None


# def color_score(vlm_obj: Dict[str, Any], gz_obj: Dict[str, Any]) -> int:
#     """
#     Soft color evidence:
#     - if VLM color is missing / generic / unknown -> neutral
#     - if both are present and equal -> reward
#     - if both are present and different -> mild penalty
#     """
#     vlm_color = canonicalize_color(vlm_obj.get("color"))
#     gz_color = canonicalize_color(gz_obj.get("color"))

#     if vlm_color is None or gz_color is None:
#         return 0

#     if vlm_color == gz_color:
#         return 12

#     return -4


# def score_assignment(
#     assignment: Dict[str, Dict[str, Any]],
#     scene_objects_by_name: Dict[str, Dict[str, Any]],
#     spatial_relationships: List[Dict[str, str]],
# ) -> int:
#     score = 0

#     for vlm_name, gz_obj in assignment.items():
#         score += strong_name_score(vlm_name, gz_obj["gazebo_name"])

#         vlm_obj = scene_objects_by_name[vlm_name]
#         vlm_category = canonicalize_category(vlm_obj.get("category"))
#         gz_category = canonicalize_category(gz_obj.get("category"))

#         if vlm_category and gz_category and vlm_category == gz_category:
#             score += 10

#         score += color_score(vlm_obj, gz_obj)

#     for rel in spatial_relationships:
#         subj = rel.get("subject")
#         obj = rel.get("object")
#         relation = rel.get("relation", "").lower()

#         if subj not in assignment or obj not in assignment:
#             continue

#         subj_gz = assignment[subj]
#         obj_gz = assignment[obj]

#         subj_dim = estimate_dimension_for_scoring(scene_objects_by_name[subj], subj_gz)
#         obj_dim = estimate_dimension_for_scoring(scene_objects_by_name[obj], obj_gz)

#         score += relation_pair_score(
#             relation=relation,
#             subj_pose=subj_gz["pose"],
#             obj_pose=obj_gz["pose"],
#             subj_dim=subj_dim,
#             obj_dim=obj_dim,
#         )

#     return score


# def build_matching_warnings(
#     scene_objects: List[Dict[str, Any]],
#     gazebo_catalog: List[Dict[str, Any]],
#     vlm_to_gazebo: Dict[str, str],
# ) -> List[str]:
#     warnings: List[str] = []
#     gazebo_by_name = {obj["gazebo_name"]: obj for obj in gazebo_catalog}

#     for vlm_obj in scene_objects:
#         vlm_name = vlm_obj["name"]
#         gazebo_name = vlm_to_gazebo[vlm_name]
#         gz_obj = gazebo_by_name[gazebo_name]

#         raw_vlm_color = vlm_obj.get("color")
#         raw_gz_color = gz_obj.get("color")

#         vlm_color = canonicalize_color(raw_vlm_color)
#         gz_color = canonicalize_color(raw_gz_color)

#         if original_color_is_non_discriminative(raw_vlm_color):
#             warnings.append(
#                 f"VLM object '{vlm_name}' has weak/non-discriminative color '{raw_vlm_color}', "
#                 f"so color was ignored as a hard constraint."
#             )

#         if raw_vlm_color is not None and vlm_color is None and not original_color_is_non_discriminative(raw_vlm_color):
#             warnings.append(
#                 f"VLM object '{vlm_name}' has unsupported or unknown color '{raw_vlm_color}', "
#                 f"so color was ignored in semantic filtering."
#             )

#         if vlm_color is not None and gz_color is not None and vlm_color != gz_color:
#             warnings.append(
#                 f"Color mismatch for '{vlm_name}': VLM color='{vlm_color}', "
#                 f"matched Gazebo entity '{gazebo_name}' with color='{gz_color}'. "
#                 "Assignment kept because color is treated as soft evidence."
#             )

#     return warnings


# def resolve_vlm_to_gazebo_mapping(
#     scene_objects: List[Dict[str, Any]],
#     spatial_relationships: List[Dict[str, str]],
#     gazebo_catalog: List[Dict[str, Any]],
# ) -> Tuple[Dict[str, str], List[str]]:
#     """
#     Resolve a one-to-one mapping:
#         VLM object name -> Gazebo entity name

#     Strategy:
#     1) semantic candidate filtering using hard constraints only (mainly category)
#     2) global one-to-one search across remaining candidates
#     3) score assignments using:
#        - strong name compatibility
#        - category agreement
#        - color as soft evidence
#        - relation consistency
#     4) fail explicitly if best assignment is still ambiguous
#     """
#     if not scene_objects:
#         return {}, []

#     scene_objects_by_name = {obj["name"]: obj for obj in scene_objects}
#     candidates_by_vlm_name = build_candidate_lists(scene_objects, gazebo_catalog)

#     ordered_vlm_names = sorted(
#         [obj["name"] for obj in scene_objects],
#         key=lambda name: (
#             len(candidates_by_vlm_name[name]),
#             -max(strong_name_score(name, gz["gazebo_name"]) for gz in candidates_by_vlm_name[name]),
#             name,
#         ),
#     )

#     best_assignments: List[Dict[str, Dict[str, Any]]] = []
#     best_score: Optional[int] = None

#     def backtrack(
#         index: int,
#         partial_assignment: Dict[str, Dict[str, Any]],
#         used_gazebo_names: Set[str],
#     ) -> None:
#         nonlocal best_score, best_assignments

#         if index == len(ordered_vlm_names):
#             total_score = score_assignment(
#                 assignment=partial_assignment,
#                 scene_objects_by_name=scene_objects_by_name,
#                 spatial_relationships=spatial_relationships,
#             )

#             if best_score is None or total_score > best_score:
#                 best_score = total_score
#                 best_assignments = [dict(partial_assignment)]
#             elif total_score == best_score:
#                 best_assignments.append(dict(partial_assignment))
#             return

#         vlm_name = ordered_vlm_names[index]
#         raw_candidates = candidates_by_vlm_name[vlm_name]

#         sorted_candidates = sorted(
#             raw_candidates,
#             key=lambda gz: (
#                 -strong_name_score(vlm_name, gz["gazebo_name"]),
#                 gz["gazebo_name"],
#             ),
#         )

#         for gz_obj in sorted_candidates:
#             gazebo_name = gz_obj["gazebo_name"]
#             if gazebo_name in used_gazebo_names:
#                 continue

#             partial_assignment[vlm_name] = gz_obj
#             used_gazebo_names.add(gazebo_name)

#             backtrack(index + 1, partial_assignment, used_gazebo_names)

#             used_gazebo_names.remove(gazebo_name)
#             del partial_assignment[vlm_name]

#     backtrack(0, {}, set())

#     if not best_assignments:
#         raise ValueError("Unable to resolve any valid one-to-one VLM-to-Gazebo assignment.")

#     if len(best_assignments) > 1:
#         rendered = []
#         for assignment in best_assignments[:5]:
#             rendered.append(
#                 {vlm_name: gz_obj["gazebo_name"] for vlm_name, gz_obj in sorted(assignment.items())}
#             )
#         raise ValueError(
#             "Ambiguous VLM-to-Gazebo mapping. Multiple equally valid assignments were found. "
#             f"Examples: {rendered}"
#         )

#     best_assignment = best_assignments[0]
#     vlm_to_gazebo = {vlm_name: gz_obj["gazebo_name"] for vlm_name, gz_obj in best_assignment.items()}
#     warnings = build_matching_warnings(scene_objects, gazebo_catalog, vlm_to_gazebo)
#     return vlm_to_gazebo, warnings


# # =========================================================
# # Accessibility
# # =========================================================
# def compute_accessibility(
#     objects_with_geometry: List[Dict[str, Any]],
#     spatial_relationships: List[Dict[str, str]],
#     safety_threshold: float = 0.21,
# ) -> Dict[str, Dict[str, Any]]:
#     """
#     Return:
#     {
#         "vlm_object_name": {
#             "location": "zone B",
#             "sides": {...}
#         },
#         ...
#     }
#     """
#     object_map = {obj["name"]: obj for obj in objects_with_geometry}

#     aabbs = {
#         name: get_aabb(obj["pose"], obj["dimension"])
#         for name, obj in object_map.items()
#     }

#     blockers = {
#         name: {side: set() for side in SIDES}
#         for name in object_map.keys()
#     }

#     grasped_objects: Set[str] = set()
#     objects_on_top_of_something: Set[str] = set()

#     for rel in spatial_relationships:
#         subj = rel["subject"]
#         relation = rel["relation"].lower()
#         obj = rel["object"]

#         if subj not in object_map:
#             continue

#         if relation == RELATION_ON_TOP_OF:
#             if obj not in object_map:
#                 continue
#             add_blocker(blockers, subj, "bottom", obj)
#             add_blocker(blockers, obj, "top", subj)
#             objects_on_top_of_something.add(subj)

#         elif relation == RELATION_GRASPED_BY:
#             grasped_objects.add(subj)

#     for name in object_map.keys():
#         if name not in objects_on_top_of_something:
#             add_blocker(blockers, name, "bottom", "table")

#     for rel in spatial_relationships:
#         subj = rel["subject"]
#         relation = rel["relation"].lower()
#         obj = rel["object"]

#         if subj not in object_map or obj not in object_map:
#             continue

#         if relation == RELATION_RIGHT_OF:
#             gap_y = gap_between_intervals(aabbs[subj]["y"], aabbs[obj]["y"])
#             if clamp_lower(gap_y, safety_threshold):
#                 add_blocker(blockers, subj, "left", obj)
#                 add_blocker(blockers, obj, "right", subj)

#         elif relation == RELATION_IN_FRONT_OF:
#             gap_x = gap_between_intervals(aabbs[subj]["x"], aabbs[obj]["x"])
#             if clamp_lower(gap_x, safety_threshold):
#                 add_blocker(blockers, subj, "back", obj)
#                 add_blocker(blockers, obj, "front", subj)

#     for name in grasped_objects:
#         if name in blockers:
#             for side in SIDES:
#                 blockers[name][side].clear()

#     result: Dict[str, Dict[str, Any]] = {}

#     for name in object_map.keys():
#         aabb = aabbs[name]

#         sides = {
#             side: format_side_status(blockers[name][side])
#             for side in SIDES
#         }

#         if is_back_not_reachable(aabb):
#             sides["back"] = "not reachable"

#         result[name] = {
#             "location": get_location(aabb),
#             "sides": sides,
#         }

#     return result


# # =========================================================
# # Scene grounding pipeline
# # =========================================================
# def build_objects_with_geometry(
#     scene_objects: List[Dict[str, Any]],
#     spatial_relationships: List[Dict[str, str]],
#     pose_source: str = POSE_SOURCE_GAZEBO,
#     pose_file: Optional[str] = None,
#     topic: str = GZ_POSE_TOPIC,
#     timeout_sec: float = 3.0,
# ) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
#     """
#     Build objects with:
#     - VLM name (kept unchanged)
#     - Gazebo name (internal only)
#     - pose (from Gazebo or static file)
#     - dimension
#     """
#     all_positions = read_all_positions(
#         pose_source=pose_source,
#         pose_file=pose_file,
#         topic=topic,
#         timeout_sec=timeout_sec,
#     )

#     gazebo_catalog = build_gazebo_object_catalog(all_positions)

#     print(f"\n[DEBUG] Pose source: {pose_source}")
#     if pose_source == POSE_SOURCE_STATIC and pose_file:
#         print(f"[DEBUG] Static pose file: {pose_file}")
#     print("[DEBUG] Filtered object catalog:")
#     for obj in gazebo_catalog:
#         print(obj["gazebo_name"])
#     print()

#     if not gazebo_catalog:
#         raise RuntimeError(
#             "No eligible object entities were found in the selected pose source. "
#             "Check the pose file/topic output and the configured metadata/categories."
#         )

#     vlm_to_gazebo, matching_warnings = resolve_vlm_to_gazebo_mapping(
#         scene_objects=scene_objects,
#         spatial_relationships=spatial_relationships,
#         gazebo_catalog=gazebo_catalog,
#     )

#     objects_with_geometry: List[Dict[str, Any]] = []

#     for obj in scene_objects:
#         vlm_name = obj["name"]
#         gazebo_name = vlm_to_gazebo[vlm_name]

#         if gazebo_name not in all_positions:
#             raise RuntimeError(
#                 f"Mapped entity '{gazebo_name}' was not found in the current pose source."
#             )

#         pose = all_positions[gazebo_name]
#         dimension = resolve_dimension(obj=obj, gazebo_name=gazebo_name)

#         objects_with_geometry.append(
#             {
#                 "name": vlm_name,
#                 "gazebo_name": gazebo_name,
#                 "pose": pose,
#                 "dimension": dimension,
#             }
#         )

#     return objects_with_geometry, vlm_to_gazebo, matching_warnings


# def enrich_scene(
#     input_data: Dict[str, Any],
#     safety_threshold: float = 0.21,
#     pose_source: str = POSE_SOURCE_GAZEBO,
#     pose_file: Optional[str] = None,
#     topic: str = GZ_POSE_TOPIC,
#     timeout_sec: float = 3.0,
#     include_debug_mapping: bool = False,
# ) -> Dict[str, Any]:
#     if "scene_description" not in input_data:
#         raise ValueError("Missing required field: 'scene_description'")

#     scene_description = input_data["scene_description"]
#     scene_objects = scene_description.get("objects", [])
#     spatial_relationships = scene_description.get("spatial_relationships", [])

#     objects_with_geometry, vlm_to_gazebo, matching_warnings = build_objects_with_geometry(
#         scene_objects=scene_objects,
#         spatial_relationships=spatial_relationships,
#         pose_source=pose_source,
#         pose_file=pose_file,
#         topic=topic,
#         timeout_sec=timeout_sec,
#     )

#     computed_info = compute_accessibility(
#         objects_with_geometry=objects_with_geometry,
#         spatial_relationships=spatial_relationships,
#         safety_threshold=safety_threshold,
#     )

#     enriched_objects: List[Dict[str, Any]] = []

#     for obj in scene_objects:
#         name = obj["name"]
#         enriched_obj = dict(obj)
#         enriched_obj["location"] = computed_info[name]["location"]
#         enriched_obj["sides"] = computed_info[name]["sides"]
#         enriched_objects.append(enriched_obj)

#     output: Dict[str, Any] = {
#         "scene_description": {
#             "objects": enriched_objects,
#             "end_effectors": scene_description.get("end_effectors", []),
#             "spatial_relationships": spatial_relationships,
#         }
#     }

#     if include_debug_mapping:
#         output["_debug"] = {
#             "pose_source": pose_source,
#             "pose_file": pose_file,
#             "vlm_to_gazebo_mapping": vlm_to_gazebo,
#             "matching_warnings": matching_warnings,
#         }

#     return output


# # =========================================================
# # CLI
# # =========================================================
# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description="Ground a VLM scene description into Gazebo or a static pose file and enrich it with location/sides."
#     )
#     parser.add_argument("input", type=str, help="Path to the input JSON file.")
#     parser.add_argument(
#         "--output",
#         type=str,
#         default="scene_description_full.json",
#         help="Path to the output JSON file.",
#     )
#     parser.add_argument(
#         "--pose-source",
#         type=str,
#         choices=[POSE_SOURCE_GAZEBO, POSE_SOURCE_STATIC],
#         default=POSE_SOURCE_GAZEBO,
#         help="Source used to read object poses: 'gazebo' or 'static'.",
#     )
#     parser.add_argument(
#         "--pose-file",
#         type=str,
#         default=None,
#         help="Path to a JSON file containing static object poses. Required if --pose-source static.",
#     )
#     parser.add_argument(
#         "--topic",
#         type=str,
#         default=GZ_POSE_TOPIC,
#         help="Gazebo topic used to read dynamic poses.",
#     )
#     parser.add_argument(
#         "--timeout-sec",
#         type=float,
#         default=3.0,
#         help="Timeout in seconds for the Gazebo topic read.",
#     )
#     parser.add_argument(
#         "--safety-threshold",
#         type=float,
#         default=0.21,
#         help="Safety threshold used in accessibility computation.",
#     )
#     parser.add_argument(
#         "--include-debug-mapping",
#         action="store_true",
#         help="Include the internal VLM-to-Gazebo mapping in the output under '_debug'.",
#     )
#     return parser


# def main() -> None:
#     parser = build_parser()
#     args = parser.parse_args()

#     try:
#         if args.pose_source == POSE_SOURCE_STATIC and not args.pose_file:
#             raise ValueError("--pose-file is required when --pose-source static")

#         with open(args.input, "r", encoding="utf-8") as f:
#             input_data = json.load(f)

#         output_data = enrich_scene(
#             input_data=input_data,
#             safety_threshold=args.safety_threshold,
#             pose_source=args.pose_source,
#             pose_file=args.pose_file,
#             topic=args.topic,
#             timeout_sec=args.timeout_sec,
#             include_debug_mapping=args.include_debug_mapping,
#         )

#         with open(args.output, "w", encoding="utf-8") as f:
#             json.dump(output_data, f, indent=2, ensure_ascii=False)

#         print(f"[OK] Enriched scene saved to: {args.output}")

#     except FileNotFoundError:
#         print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
#         sys.exit(1)
#     except json.JSONDecodeError as exc:
#         print(f"[ERROR] Invalid JSON: {exc}", file=sys.stderr)
#         sys.exit(1)
#     except Exception as exc:
#         print(f"[ERROR] {exc}", file=sys.stderr)
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

