from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import load_settings
from .prompts import load_system_prompt
from .runner import run_single_experiment
from .save_results import save_result
from .validator_builders import build_validator_user_block


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single Azure OpenAI validator experiment.")
    parser.add_argument("--model", type=str, default=None, help="Logical model name: o3 or gpt-5.2")
    parser.add_argument("--prompt-file", type=str, required=True, help="Prompt filename inside prompts/validator/")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--scene-description", type=str, required=True, help="Path to scene_description.json")
    parser.add_argument("--condition", type=str, required=True, help="Condition to check")
    parser.add_argument("--action-context", type=str, default=None, help="Optional JSON string for action context")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--output-name", type=str, default=None, help="Optional output filename")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = load_settings()
    model_name = args.model or settings.default_model
    temperature = settings.default_temperature if args.temperature is None else args.temperature

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_stem = Path(args.prompt_file).stem

    scene_description = json.loads(Path(args.scene_description).read_text(encoding="utf-8"))
    action_context = json.loads(args.action_context) if args.action_context else None

    system_prompt = load_system_prompt(settings, prompt_group="validator", prompt_filename=args.prompt_file)
    user_text = build_validator_user_block(
        condition_to_check=args.condition,
        scene_description=scene_description,
        action_context=action_context,
    )

    result = run_single_experiment(
        settings=settings,
        model_name=model_name,
        system_prompt=system_prompt,
        user_text=user_text,
        image_path=args.image,
        temperature=temperature,
    )

    result["run_id"] = run_id
    result["prompt_name"] = prompt_stem
    result["condition_to_check"] = args.condition
    result["prompt_filename"] = args.prompt_file
    result["scene_description_path"] = args.scene_description
    result["action_context"] = action_context

    output_name = args.output_name
    if output_name is None:
        image_stem = Path(args.image).stem
        output_name = f"{model_name}_{prompt_stem}_{image_stem}.json"

    out_path = save_result(
        settings=settings,
        task_type="validator",
        filename=output_name,
        payload=result,
        prompt_name=prompt_stem,
        run_id=run_id,
        model_name=model_name,
    )

    print(f"Saved result to: {out_path}")
    print(f"Run ID: {run_id}")
    print("JSON parse ok:", result["json_parse_ok"])
    print("Raw response:")
    print(result["raw_response"])


if __name__ == "__main__":
    main()