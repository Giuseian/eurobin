from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import load_settings
from .prompts import load_system_prompt
from .runner import run_single_experiment
from .save_results import save_result
from .scenario_loaders import load_validator_cases
from .validator_builders import build_validator_user_block


def parse_csv_arg(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch VLM validator experiments.")
    parser.add_argument(
        "--scenarios-dir",
        type=str,
        default="scenarios/validator",
        help="Root directory containing validator scenario folders",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated logical model names, e.g. o3,gpt-5.2",
    )
    parser.add_argument(
        "--prompt-files",
        type=str,
        required=True,
        help="Comma-separated prompt filenames inside prompts/validator/",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature override",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs per configuration",
    )
    parser.add_argument(
        "--scenario-ids",
        type=str,
        default=None,
        help="Optional comma-separated subset of scenario ids to run",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = load_settings()

    models = parse_csv_arg(args.models)
    prompt_files = parse_csv_arg(args.prompt_files)
    selected_scenario_ids = set(parse_csv_arg(args.scenario_ids))

    if not models:
        raise ValueError("No models provided.")
    if not prompt_files:
        raise ValueError("No prompt files provided.")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    cases = load_validator_cases(args.scenarios_dir)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    total_runs = 0
    successful_runs = 0
    failed_runs = 0

    for case in cases:
        scenario_id = case["scenario_id"]
        case_id = case["case_id"]

        if selected_scenario_ids and scenario_id not in selected_scenario_ids:
            continue

        image_path = case["image_path"]
        image_stem = Path(image_path).stem if image_path else "no_image"

        user_text = build_validator_user_block(
            condition_to_check=case["condition_to_check"],
            scene_description=case["scene_description"],
            action_context=case.get("action_context"),
        )

        for model_name in models:
            for prompt_file in prompt_files:
                prompt_stem = Path(prompt_file).stem
                system_prompt = load_system_prompt(
                    settings=settings,
                    prompt_group="validator",
                    prompt_filename=prompt_file,
                )

                for repeat_idx in range(1, args.repeats + 1):
                    total_runs += 1

                    try:
                        temperature = (
                            args.temperature
                            if args.temperature is not None
                            else settings.default_temperature
                        )

                        result = run_single_experiment(
                            settings=settings,
                            model_name=model_name,
                            system_prompt=system_prompt,
                            user_text=user_text,
                            image_path=image_path,
                            temperature=temperature,
                        )

                        result["scenario_id"] = scenario_id
                        result["case_id"] = case_id
                        result["repeat_idx"] = repeat_idx
                        result["condition_to_check"] = case["condition_to_check"]
                        result["action_context"] = case.get("action_context")
                        result["prompt_filename"] = prompt_file
                        result["prompt_name"] = prompt_stem
                        result["run_id"] = run_id

                        output_name = (
                            f"{scenario_id}__{case_id}__{model_name}__{prompt_stem}"
                            f"__{image_stem}__r{repeat_idx}.json"
                        )

                        out_path = save_result(
                            settings=settings,
                            task_type="validator",
                            filename=output_name,
                            payload=result,
                            scenario_id=scenario_id,
                            prompt_name=prompt_stem,
                            run_id=run_id,
                            model_name=model_name,
                        )

                        successful_runs += 1

                        print(f"[OK] {out_path}")
                        print(f"   inference_time: {result['inference_time_sec']:.2f}s")

                        if result.get("json_parse_ok"):
                            print("   validation_output:")
                            print(json.dumps(result["parsed_json"], indent=2, ensure_ascii=False))
                        else:
                            print("   raw_output:")
                            print(result.get("raw_response", ""))

                        print("-" * 60)

                    except Exception as exc:
                        failed_runs += 1

                        error_payload = {
                            "scenario_id": scenario_id,
                            "case_id": case_id,
                            "image_path": image_path,
                            "model_name": model_name,
                            "prompt_filename": prompt_file,
                            "prompt_name": prompt_stem,
                            "repeat_idx": repeat_idx,
                            "run_id": run_id,
                            "error": str(exc),
                        }

                        output_name = (
                            f"{scenario_id}__{case_id}__{model_name}__{prompt_stem}"
                            f"__{image_stem}__r{repeat_idx}__ERROR.json"
                        )

                        out_path = save_result(
                            settings=settings,
                            task_type="validator",
                            filename=output_name,
                            payload=error_payload,
                            scenario_id=scenario_id,
                            prompt_name=prompt_stem,
                            run_id=run_id,
                            model_name=model_name,
                        )
                        print(f"[ERROR] {out_path} -> {exc}")

    print("\nBatch completed.")
    print(f"Run ID: {run_id}")
    print(f"Total runs: {total_runs}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")


if __name__ == "__main__":
    main()