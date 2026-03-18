from __future__ import annotations

import json
from typing import Any


def build_validator_user_block(
    condition_to_check: str,
    scene_description: dict[str, Any],
    action_context: dict[str, Any] | None = None,
) -> str:
    action_context_str = (
        json.dumps(action_context, ensure_ascii=False, indent=2)
        if action_context is not None
        else "null"
    )

    scene_description_str = json.dumps(
        scene_description,
        ensure_ascii=False,
        indent=2,
    )

    return f"""Validate whether the following condition is satisfied in the current scene.

Action context:
{action_context_str}

Condition to verify:
"{condition_to_check}"

Scene object list:
{scene_description_str}

Validation instructions:
- Use the object list as the main structured representation.
- Use the image to verify visual consistency.
- Return "matching" only if the condition is clearly satisfied.
- Return "non_matching" if the condition is contradicted, unsupported, incomplete, or uncertain.
- Return only valid JSON.

Output format:
{{
  "result": "matching" | "non_matching",
  "reason": "<short explanation>"
}}"""