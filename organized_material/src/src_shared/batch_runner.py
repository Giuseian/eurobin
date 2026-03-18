# from __future__ import annotations

# import argparse
# import json
# from datetime import datetime
# from pathlib import Path
# from typing import Any

# from .config import load_settings
# from .runner import run_single_experiment
# from .save_results import save_result


# def load_scenarios(scenarios_path: str) -> list[dict[str, Any]]:
#     path = Path(scenarios_path)
#     if not path.exists():
#         raise FileNotFoundError(f"Scenarios file not found: {path}")

#     data = json.loads(path.read_text(encoding="utf-8"))

#     if "scenarios" not in data or not isinstance(data["scenarios"], list):
#         raise ValueError("scenarios.json must contain a top-level 'scenarios' list")

#     return data["scenarios"]


# def parse_csv_arg(value: str | None) -> list[str]:
#     if value is None or not value.strip():
#         return []
#     return [item.strip() for item in value.split(",") if item.strip()]


# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description="Run batch VLM experiments.")
#     parser.add_argument(
#         "--scenarios-file",
#         type=str,
#         default="scenarios/scenarios.json",
#         help="Path to scenarios.json",
#     )
#     parser.add_argument(
#         "--models",
#         type=str,
#         required=True,
#         help="Comma-separated logical model names, e.g. o3,gpt-5.2",
#     )
#     parser.add_argument(
#         "--prompt-files",
#         type=str,
#         required=True,
#         help="Comma-separated prompt filenames, e.g. prompt1_1.txt,prompt1_2.txt",
#     )
#     parser.add_argument(
#         "--temperature",
#         type=float,
#         default=None,
#         help="Optional temperature override",
#     )
#     parser.add_argument(
#         "--repeats",
#         type=int,
#         default=1,
#         help="Number of repeated runs per configuration",
#     )
#     parser.add_argument(
#         "--scenario-ids",
#         type=str,
#         default=None,
#         help="Optional comma-separated subset of scenario ids to run",
#     )
#     return parser


# def main() -> None:
#     parser = build_parser()
#     args = parser.parse_args()

#     settings = load_settings()

#     models = parse_csv_arg(args.models)
#     prompt_files = parse_csv_arg(args.prompt_files)
#     selected_scenario_ids = set(parse_csv_arg(args.scenario_ids))

#     if not models:
#         raise ValueError("No models provided.")
#     if not prompt_files:
#         raise ValueError("No prompt files provided.")
#     if args.repeats < 1:
#         raise ValueError("--repeats must be >= 1")

#     scenarios = load_scenarios(args.scenarios_file)

#     # Identificatore unico del batch corrente
#     run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#     total_runs = 0
#     successful_runs = 0
#     failed_runs = 0

#     for scenario in scenarios:
#         scenario_id = scenario.get("scenario_id")
#         task_text = scenario.get("task")
#         images = scenario.get("images")

#         if not scenario_id:
#             raise ValueError("Each scenario must have a 'scenario_id'")
#         if not task_text:
#             raise ValueError(f"Scenario '{scenario_id}' is missing 'task'")

#         if images is None:
#             images = []
#         elif not isinstance(images, list):
#             raise ValueError(f"Scenario '{scenario_id}' field 'images' must be a list")

#         if selected_scenario_ids and scenario_id not in selected_scenario_ids:
#             continue

#         effective_images: list[str | None] = images if images else [None]

#         for image_path in effective_images:
#             image_stem = Path(image_path).stem if image_path else "no_image"

#             for model_name in models:
#                 for prompt_file in prompt_files:
#                     prompt_stem = Path(prompt_file).stem

#                     for repeat_idx in range(1, args.repeats + 1):
#                         total_runs += 1

#                         try:
#                             temperature = (
#                                 args.temperature
#                                 if args.temperature is not None
#                                 else settings.default_temperature
#                             )

#                             result = run_single_experiment(
#                                 settings=settings,
#                                 model_name=model_name,
#                                 prompt_filename=prompt_file,
#                                 image_path=image_path,
#                                 task_text=task_text,
#                                 temperature=temperature,
#                             )

#                             result["scenario_id"] = scenario_id
#                             result["repeat_idx"] = repeat_idx
#                             result["image_path"] = image_path
#                             result["run_id"] = run_id
#                             result["prompt_name"] = prompt_stem

#                             output_name = (
#                                 f"{scenario_id}__{model_name}__{prompt_stem}"
#                                 f"__{image_stem}__r{repeat_idx}.json"
#                             )

#                             out_path = save_result(
#                                 settings=settings,
#                                 filename=output_name,
#                                 payload=result,
#                                 scenario_id=scenario_id,
#                                 prompt_name=prompt_stem,
#                                 run_id=run_id,
#                                 model_name=model_name,
#                             )
#                             successful_runs += 1

#                             print(f"[OK] {out_path}")
#                             print(f"   inference_time: {result['inference_time_sec']:.2f}s")

#                             if result.get("json_parse_ok"):
#                                 print("   generated_plan:")
#                                 print(json.dumps(result["parsed_json"], indent=2))
#                             else:
#                                 print("   raw_output:")
#                                 print(result.get("raw_response", ""))

#                             print("-" * 60)

#                         except Exception as exc:
#                             failed_runs += 1

#                             error_payload = {
#                                 "scenario_id": scenario_id,
#                                 "image_path": image_path,
#                                 "task_text": task_text,
#                                 "model_name": model_name,
#                                 "prompt_filename": prompt_file,
#                                 "prompt_name": prompt_stem,
#                                 "repeat_idx": repeat_idx,
#                                 "run_id": run_id,
#                                 "error": str(exc),
#                             }

#                             output_name = (
#                                 f"{scenario_id}__{model_name}__{prompt_stem}"
#                                 f"__{image_stem}__r{repeat_idx}__ERROR.json"
#                             )

#                             out_path = save_result(
#                                 settings=settings,
#                                 filename=output_name,
#                                 payload=error_payload,
#                                 scenario_id=scenario_id,
#                                 prompt_name=prompt_stem,
#                                 run_id=run_id,
#                                 model_name=model_name,
#                             )
#                             print(f"[ERROR] {out_path} -> {exc}")

#     print("\nBatch completed.")
#     print(f"Run ID: {run_id}")
#     print(f"Total runs: {total_runs}")
#     print(f"Successful runs: {successful_runs}")
#     print(f"Failed runs: {failed_runs}")


# if __name__ == "__main__":
#     main()



from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import load_settings
from .planner_builders import build_planner_user_block
from .prompts import load_system_prompt
from .runner import run_single_experiment
from .save_results import save_result
from .scenario_loaders import load_planner_scenarios


def parse_csv_arg(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch VLM planner experiments.")
    parser.add_argument(
        "--scenarios-file",
        type=str,
        default="scenarios/planner/scenarios.json",
        help="Path to planner scenarios.json",
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
        help="Comma-separated prompt filenames inside prompts/planner/",
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

    scenarios = load_planner_scenarios(args.scenarios_file)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    total_runs = 0
    successful_runs = 0
    failed_runs = 0

    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id")
        task_text = scenario.get("task")
        images = scenario.get("images")

        if not scenario_id:
            raise ValueError("Each scenario must have a 'scenario_id'")
        if not task_text:
            raise ValueError(f"Scenario '{scenario_id}' is missing 'task'")

        if images is None:
            images = []
        elif not isinstance(images, list):
            raise ValueError(f"Scenario '{scenario_id}' field 'images' must be a list")

        if selected_scenario_ids and scenario_id not in selected_scenario_ids:
            continue

        effective_images: list[str | None] = images if images else [None]

        for image_path in effective_images:
            image_stem = Path(image_path).stem if image_path else "no_image"
            user_text = build_planner_user_block(task_text)

            for model_name in models:
                for prompt_file in prompt_files:
                    prompt_stem = Path(prompt_file).stem
                    system_prompt = load_system_prompt(
                        settings=settings,
                        prompt_group="planner",
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
                            result["repeat_idx"] = repeat_idx
                            result["task_text"] = task_text
                            result["prompt_filename"] = prompt_file
                            result["prompt_name"] = prompt_stem
                            result["run_id"] = run_id

                            output_name = (
                                f"{scenario_id}__{model_name}__{prompt_stem}"
                                f"__{image_stem}__r{repeat_idx}.json"
                            )

                            out_path = save_result(
                                settings=settings,
                                task_type="planner",
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
                                print("   generated_plan:")
                                print(json.dumps(result["parsed_json"], indent=2, ensure_ascii=False))
                            else:
                                print("   raw_output:")
                                print(result.get("raw_response", ""))

                            print("-" * 60)

                        except Exception as exc:
                            failed_runs += 1

                            error_payload = {
                                "scenario_id": scenario_id,
                                "image_path": image_path,
                                "task_text": task_text,
                                "model_name": model_name,
                                "prompt_filename": prompt_file,
                                "prompt_name": prompt_stem,
                                "repeat_idx": repeat_idx,
                                "run_id": run_id,
                                "error": str(exc),
                            }

                            output_name = (
                                f"{scenario_id}__{model_name}__{prompt_stem}"
                                f"__{image_stem}__r{repeat_idx}__ERROR.json"
                            )

                            out_path = save_result(
                                settings=settings,
                                task_type="planner",
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