from __future__ import annotations

from pathlib import Path

from .config import Settings


def load_system_prompt(settings: Settings, prompt_filename: str) -> str:
    prompt_path = (
        settings.project_root
        / "prompts"
        / settings.prompts_subdir
        / prompt_filename
    )

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


def build_user_block(task_text: str) -> str:
    return f"""Task: {task_text}

Robot instructions:
- The robot has two arms mounted on its back.
- It must output a high-level manipulation plan only.
- It must reason about accessibility, obstacles, and arm coordination.
- It must produce valid JSON only.

Generate the output in the required JSON format."""