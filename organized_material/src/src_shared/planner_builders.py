from __future__ import annotations


def build_planner_user_block(task_text: str) -> str:
    return f"""Task: {task_text}

# Robot instructions:
# - The robot has two arms mounted on its back.
# - It must output a high-level manipulation plan only.
# - It must reason about accessibility, obstacles, and arm coordination.
# - It must produce valid JSON only.

# Generate the output in the required JSON format."""