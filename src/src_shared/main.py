from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_settings
from .runner import run_single_experiment
from .save_results import save_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single Azure OpenAI VLM experiment.")
    parser.add_argument("--model", type=str, default=None, help="Logical model name: o3 or gpt-5.1")
    parser.add_argument("--prompt-file", type=str, required=True, help="Prompt filename, e.g. prompt1_1.txt")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--task", type=str, required=True, help="Task text")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--output-name", type=str, default=None, help="Optional output filename")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = load_settings()

    model_name = args.model or settings.default_model
    temperature = settings.default_temperature if args.temperature is None else args.temperature

    result = run_single_experiment(
        settings=settings,
        model_name=model_name,
        prompt_filename=args.prompt_file,
        image_path=args.image,
        task_text=args.task,
        temperature=temperature,
    )

    output_name = args.output_name
    if output_name is None:
        image_stem = Path(args.image).stem
        prompt_stem = Path(args.prompt_file).stem
        output_name = f"{model_name}_{prompt_stem}_{image_stem}.json"

    out_path = save_result(settings, output_name, result)

    print(f"Saved result to: {out_path}")
    print("JSON parse ok:", result["json_parse_ok"])
    print("Raw response:")
    print(result["raw_response"])


if __name__ == "__main__":
    main()